"""Restartable GSM8K Cache-MoE sweep with real vLLM MTP verification."""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import math
import re
import shutil
import socket
import statistics
import time
from functools import partial
from pathlib import Path
from typing import Any

import vllm
from vllm import LLM, SamplingParams

from cache_moe_control import (
    collect_speculative_cache_misses,
    configure_speculative_cache_prior_worker,
)

INVALID_ANSWER = -9_999_999


def parse_csv(value: str, cast: type) -> list[Any]:
    """Parse one comma-separated CLI value."""
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, str]]:
    """Read the official GSM8K JSONL format."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def answer_value(text: str) -> int:
    """Match vLLM's GSM8K integer-answer extraction."""
    numbers = re.findall(r"\d+", text.replace(",", ""))
    if not numbers:
        return INVALID_ANSWER
    try:
        return int(ast.literal_eval(numbers[-1]))
    except (SyntaxError, ValueError):
        return INVALID_ANSWER


def build_gsm8k_prompts(
    train_path: Path,
    test_path: Path,
    *,
    question_start: int,
    num_questions: int,
    num_shots: int,
) -> tuple[list[str], list[int]]:
    """Build deterministic few-shot prompts and labels."""
    train = read_jsonl(train_path)
    test = read_jsonl(test_path)[question_start : question_start + num_questions]
    demonstrations = "".join(
        f"Question: {row['question']}\nAnswer: {row['answer']}\n\n"
        for row in train[:num_shots]
    )
    prompts = [demonstrations + f"Question: {row['question']}\nAnswer:" for row in test]
    labels = [answer_value(row["answer"]) for row in test]
    if any(label == INVALID_ANSWER for label in labels):
        raise RuntimeError("GSM8K contains an unparseable reference answer")
    return prompts, labels


def write_json_atomic(path: Path, value: object) -> None:
    """Write one durable result artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def response_summary(
    path: Path,
    prompts: list[str],
    labels: list[int],
    outputs: list[object],
) -> dict[str, int | float]:
    """Persist responses and calculate answer-level quality."""
    correct = 0
    invalid = 0
    output_tokens = 0
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as output_file:
        for index, (prompt, label, request_output) in enumerate(
            zip(prompts, labels, outputs, strict=True)
        ):
            completion = request_output.outputs[0]
            prediction = answer_value(completion.text)
            token_ids = [int(token_id) for token_id in completion.token_ids]
            correct += int(prediction == label)
            invalid += int(prediction == INVALID_ANSWER)
            output_tokens += len(token_ids)
            output_file.write(
                json.dumps(
                    {
                        "index": index,
                        "prompt": prompt,
                        "label": label,
                        "prediction": prediction,
                        "correct": prediction == label,
                        "output_token_ids": token_ids,
                        "response": completion.text,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    temporary.replace(path)
    questions = len(labels)
    return {
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / questions if questions else 0.0,
        "invalid_rate": invalid / questions if questions else 0.0,
        "output_tokens": output_tokens,
    }


def paired_quality_summary(
    baseline_path: Path,
    current_path: Path,
) -> dict[str, int | float]:
    """Compare accuracy and token sequences on exactly paired questions."""

    def load(path: Path) -> list[dict[str, object]]:
        with gzip.open(path, "rt", encoding="utf-8") as input_file:
            return [json.loads(line) for line in input_file if line.strip()]

    baseline = load(baseline_path)
    current = load(current_path)
    if len(baseline) != len(current):
        raise RuntimeError("paired GSM8K response counts differ")

    deltas: list[int] = []
    improved = 0
    regressed = 0
    identical_tokens = 0
    identical_predictions = 0
    for base, candidate in zip(baseline, current, strict=True):
        if base["index"] != candidate["index"] or base["label"] != candidate["label"]:
            raise RuntimeError("paired GSM8K response identities differ")
        base_correct = bool(base["correct"])
        candidate_correct = bool(candidate["correct"])
        delta = int(candidate_correct) - int(base_correct)
        deltas.append(delta)
        improved += int(delta == 1)
        regressed += int(delta == -1)
        identical_tokens += int(
            base["output_token_ids"] == candidate["output_token_ids"]
        )
        identical_predictions += int(base["prediction"] == candidate["prediction"])

    count = len(deltas)
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    standard_error = statistics.stdev(deltas) / math.sqrt(count) if count > 1 else 0.0
    return {
        "questions": count,
        "accuracy_delta": mean_delta,
        "accuracy_delta_ci95_low": mean_delta - 1.96 * standard_error,
        "accuracy_delta_ci95_high": mean_delta + 1.96 * standard_error,
        "improved": improved,
        "regressed": regressed,
        "discordant": improved + regressed,
        "exact_token_sequence_agreement": (identical_tokens / count if count else 0.0),
        "prediction_agreement": (identical_predictions / count if count else 0.0),
    }


def validate_worker_metrics(
    metrics: list[dict[str, object]],
) -> dict[str, object]:
    """Require model-parallel ranks to observe the same logical trace."""
    if not metrics:
        raise RuntimeError("no worker returned Cache-MoE metrics")
    reference = metrics[0]
    comparable = (
        "verification_steps",
        "cache_misses",
        "required_experts",
        "verification_tokens",
        "emitted_tokens",
        "accepted_draft_tokens",
        "verified_requests",
        "prefill_steps",
        "prefill_layer_blocks",
        "prefill_layer_overflows",
        "prefill_cache_misses",
        "total_cache_misses_including_prefill",
    )
    for other in metrics[1:]:
        for key in comparable:
            if other[key] != reference[key]:
                raise RuntimeError(
                    f"model-parallel workers disagree on {key}: "
                    f"{reference[key]} != {other[key]}"
                )
    return reference


def make_llm(args: argparse.Namespace) -> LLM:
    """Create one eager target plus its checkpoint MTP head."""
    kwargs: dict[str, object] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "trust_remote_code": True,
        "dtype": "auto",
        "load_format": args.load_format,
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": args.enable_prefix_caching,
        "async_scheduling": False,
        "enable_expert_parallel": args.enable_expert_parallel,
        "moe_backend": args.moe_backend,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "disable_log_stats": False,
        "seed": args.seed,
    }
    if args.speculative_method != "none":
        speculative_config: dict[str, object] = {
            "method": args.speculative_method,
            "num_speculative_tokens": args.num_speculative_tokens,
        }
        if args.speculative_method in ("ngram", "ngram_gpu"):
            speculative_config.update(
                {
                    "prompt_lookup_min": args.prompt_lookup_min,
                    "prompt_lookup_max": args.prompt_lookup_max,
                }
            )
        kwargs["speculative_config"] = speculative_config
    if args.language_model_only:
        kwargs["language_model_only"] = True
    if "Kimi-K3" in args.model:
        kwargs["limit_mm_per_prompt"] = {"image": 0}
        kwargs["distributed_executor_backend"] = "ray"
    if args.kv_cache_dtype:
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.mamba_cache_dtype:
        kwargs["mamba_cache_dtype"] = args.mamba_cache_dtype
    if args.mamba_ssm_cache_dtype:
        kwargs["mamba_ssm_cache_dtype"] = args.mamba_ssm_cache_dtype
    if args.mamba_cache_mode:
        kwargs["mamba_cache_mode"] = args.mamba_cache_mode
    if args.revision:
        kwargs["revision"] = args.revision
        kwargs["tokenizer_revision"] = args.revision
    return LLM(**kwargs)


def main() -> None:
    """Run every missing lambda point and retain outputs for paired analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comparison-responses", type=Path)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--lambdas", required=True)
    parser.add_argument("--top-j", type=int, default=1)
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--num-speculative-tokens", type=int, default=3)
    parser.add_argument("--prompt-lookup-min", type=int, default=1)
    parser.add_argument("--prompt-lookup-max", type=int, default=5)
    parser.add_argument(
        "--speculative-method",
        choices=("none", "mtp", "ngram", "ngram_gpu", "suffix"),
        default="mtp",
    )
    parser.add_argument("--tensor-parallel-size", type=int, required=True)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--native-routing", action="store_true")
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--load-format", default="instanttensor")
    parser.add_argument("--kv-cache-dtype")
    parser.add_argument("--mamba-cache-dtype")
    parser.add_argument("--mamba-ssm-cache-dtype")
    parser.add_argument("--mamba-cache-mode", choices=("none", "all", "align"))
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--stop-after-seconds", type=float)
    args = parser.parse_args()

    lambdas = parse_csv(args.lambdas, float)
    if not lambdas or 0.0 not in lambdas:
        raise ValueError("the sweep must include lambda=0")
    if args.native_routing and lambdas != [0.0]:
        raise ValueError("native-routing control requires exactly lambda=0")
    prompts, labels = build_gsm8k_prompts(
        args.train,
        args.test,
        question_start=args.question_start,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
    )
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    sampling_params = SamplingParams(
        temperature=0.0,
        seed=args.seed,
        max_tokens=args.max_tokens,
        stop=["Question:", "Assistant:", "<|separator|>"],
    )

    run_started = time.monotonic()
    load_started = time.monotonic()
    llm = make_llm(args)
    load_seconds = time.monotonic() - load_started

    for lambda_value in lambdas:
        point_name = f"lambda-{lambda_value:.8g}"
        point_root = args.output_root / point_name
        result_path = point_root / "result.json"
        responses_path = point_root / "responses.jsonl.gz"
        if result_path.exists() and responses_path.exists():
            print(json.dumps({"status": "skip", "point": point_name}), flush=True)
            continue
        if (
            args.stop_after_seconds is not None
            and time.monotonic() - run_started >= args.stop_after_seconds
        ):
            print(
                json.dumps(
                    {
                        "status": "time-budget-exhausted",
                        "next_lambda": lambda_value,
                    }
                ),
                flush=True,
            )
            return

        shutil.rmtree(point_root, ignore_errors=True)
        point_root.mkdir(parents=True)
        metrics_root = point_root / "router-metrics"
        if args.native_routing:
            configurations: list[dict[str, object]] = []
        else:
            configurations = llm.collective_rpc(
                partial(
                    configure_speculative_cache_prior_worker,
                    capacity=args.capacity,
                    lambda_value=lambda_value,
                    top_j=args.top_j,
                    metrics_dir_value=str(metrics_root),
                    num_speculative_tokens=args.num_speculative_tokens,
                )
            )
        if not args.native_routing and (
            not configurations
            or not all(
                bool(config["sample_hook_installed"]) for config in configurations
            )
        ):
            raise RuntimeError("not every worker installed the rejection hook")

        point_started = time.monotonic()
        outputs = llm.chat(
            conversations,
            sampling_params,
            chat_template_kwargs={"enable_thinking": False},
            use_tqdm=False,
        )
        elapsed = time.monotonic() - point_started
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM did not return every GSM8K response")
        quality = response_summary(responses_path, prompts, labels, outputs)
        baseline_responses_path = (
            args.comparison_responses
            if args.comparison_responses is not None
            else args.output_root / "lambda-0" / "responses.jsonl.gz"
        )
        if not baseline_responses_path.exists():
            raise FileNotFoundError(
                f"paired comparison responses do not exist: {baseline_responses_path}"
            )
        paired_quality = paired_quality_summary(
            baseline_responses_path,
            responses_path,
        )
        if args.native_routing:
            worker_metrics: list[dict[str, object]] = []
            cache = None
            emitted_tokens = 0
            proposed_drafts = 0
            accepted_drafts = 0
        else:
            worker_metrics = llm.apply_model(collect_speculative_cache_misses)
            cache = validate_worker_metrics(worker_metrics)
            emitted_tokens = int(cache["emitted_tokens"])
            verification_tokens = int(cache["verification_tokens"])
            verified_requests = int(cache["verified_requests"])
            proposed_drafts = verification_tokens - verified_requests
            accepted_drafts = int(cache["accepted_draft_tokens"])
            # Rejection-sampler output includes terminal sampled tokens that
            # vLLM may omit from the returned completion after applying stop
            # strings. Cache commits must therefore follow the sampler mask;
            # returned output length is not an upper bound at DL > 1.

        result = {
            "status": "complete",
            "host": socket.gethostname(),
            "model": args.model,
            "revision": args.revision,
            "vllm_version": vllm.__version__,
            "dataset": "openai/grade-school-math:test",
            "question_start": args.question_start,
            "questions": len(prompts),
            "num_shots": args.num_shots,
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "thinking": False,
            "capacity": args.capacity,
            "lambda": lambda_value,
            "top_j": args.top_j,
            "num_speculative_tokens": args.num_speculative_tokens,
            "speculative_method": args.speculative_method,
            "prompt_lookup_min": args.prompt_lookup_min,
            "prompt_lookup_max": args.prompt_lookup_max,
            "load_format": args.load_format,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_num_seqs": args.max_num_seqs,
            "enable_expert_parallel": args.enable_expert_parallel,
            "language_model_only": args.language_model_only,
            "native_routing": args.native_routing,
            "moe_backend": args.moe_backend,
            "mamba_cache_mode": args.mamba_cache_mode or "default",
            "mamba_cache_dtype": args.mamba_cache_dtype or "default",
            "enable_prefix_caching": args.enable_prefix_caching,
            "model_load_seconds": load_seconds,
            "point_seconds": elapsed,
            "output_tokens_per_second": (
                int(quality["output_tokens"]) / elapsed if elapsed else math.nan
            ),
            "acceptance_rate": (
                accepted_drafts / proposed_drafts
                if proposed_drafts
                else (None if args.native_routing else 0.0)
            ),
            "mean_emitted_tokens_per_verification": (
                None
                if cache is None
                else (
                    emitted_tokens / int(cache["verification_steps"])
                    if int(cache["verification_steps"])
                    else 0.0
                )
            ),
            "quality": quality,
            "paired_vs_lambda_0": paired_quality,
            "cache": cache,
            "decode_loads_per_generated_output_token": (
                None
                if cache is None
                else (
                    int(cache["decode_cache_misses"]) / int(quality["output_tokens"])
                    if int(quality["output_tokens"])
                    else 0.0
                )
            ),
            "total_loads_per_generated_output_token": (
                None
                if cache is None
                else (
                    int(cache["total_cache_misses_including_prefill"])
                    / int(quality["output_tokens"])
                    if int(quality["output_tokens"])
                    else 0.0
                )
            ),
            "worker_cache_metrics": worker_metrics,
            "worker_configurations": configurations,
            "responses_path": str(responses_path),
            "comparison_responses_path": str(baseline_responses_path),
        }
        write_json_atomic(result_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print(
        json.dumps(
            {
                "status": "sweep-complete",
                "lambdas": lambdas,
                "question_start": args.question_start,
                "questions": len(prompts),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
