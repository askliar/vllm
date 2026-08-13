"""Gate CacheMoE on exact concurrency-1 versus concurrency-64 parity."""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from pathlib import Path

from vllm import LLM, SamplingParams

from cache_moe_control import (
    collect_speculative_cache_misses,
    configure_speculative_cache_prior_worker,
    reset_speculative_request_caches_worker,
)
from speculative_gsm8k_sweep import validate_worker_metrics


def load_validated_prompts(path: Path, count: int) -> list[dict[str, object]]:
    """Load the exact seeded lm-eval prompts from the validated 84% run."""
    by_id: dict[int, dict[str, object]] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("filter") != "flexible-extract":
                continue
            doc_id = int(row["doc_id"])
            prompt = row["arguments"]["gen_args_0"]["arg_0"]
            by_id[doc_id] = {
                "doc_id": doc_id,
                "prompt": prompt,
                "target": row["target"],
            }
    rows = [by_id[index] for index in sorted(by_id)[:count]]
    if len(rows) != count:
        raise RuntimeError(f"expected {count} validated prompts, found {len(rows)}")
    return rows


def flexible_extract(text: str) -> str:
    """Apply the GSM8K flexible-extract regex used by lm-eval."""
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    if not matches:
        return "[invalid]"
    return next((value for value in matches[-1] if value), "[invalid]").strip()


def normalized_target(target: str) -> str:
    return re.sub(r"\.$", "", re.sub(r"(?s).*#### ", "", target).replace(",", "").replace("$", "")).lower()


def compact_outputs(rows: list[dict[str, object]], outputs: list[object]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for row, request_output in zip(rows, outputs, strict=True):
        completion = request_output.outputs[0]
        prediction = flexible_extract(completion.text)
        compact.append(
            {
                "doc_id": row["doc_id"],
                "token_ids": [int(token) for token in completion.token_ids],
                "text": completion.text,
                "prediction": prediction,
                "correct": prediction.replace(",", "").replace("$", "").lower()
                == normalized_target(str(row["target"])),
            }
        )
    return compact


def configure(llm: LLM, args: argparse.Namespace, metrics_dir: Path) -> object:
    return llm.collective_rpc(
        partial(
            configure_speculative_cache_prior_worker,
            capacity=args.capacity,
            lambda_value=args.lambda_value,
            top_j=1,
            metrics_dir_value=str(metrics_dir),
            num_speculative_tokens=args.num_speculative_tokens,
        )
    )


def run_pass(
    llm: LLM,
    rows: list[dict[str, object]],
    sampling: SamplingParams,
    *,
    concurrency: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    llm.collective_rpc(reset_speculative_request_caches_worker)
    outputs: list[object] = []
    prompts = [str(row["prompt"]) for row in rows]
    for start in range(0, len(prompts), concurrency):
        outputs.extend(
            llm.generate(
                prompts[start : start + concurrency],
                sampling,
                use_tqdm=False,
            )
        )
    compact = compact_outputs(rows, outputs)
    metrics = validate_worker_metrics(llm.apply_model(collect_speculative_cache_misses))
    return compact, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4")
    parser.add_argument("--capacity", type=int, default=96)
    parser.add_argument("--lambda-value", type=float, default=0.263)
    parser.add_argument("--num-speculative-tokens", type=int, default=5)
    parser.add_argument("--questions", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.questions != 64:
        raise ValueError("the prerequisite is deliberately fixed to exactly 64 samples")
    rows = load_validated_prompts(args.validated_samples, args.questions)
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["Question:", "</s>", "<|im_end|>"],
        seed=1234,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        trust_remote_code=True,
        dtype="auto",
        load_format="instanttensor",
        safetensors_load_strategy="prefetch",
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=False,
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
        seed=1234,
        speculative_config={
            "method": "mtp",
            "model": args.model,
            "num_speculative_tokens": args.num_speculative_tokens,
        },
    )
    configuration = configure(llm, args, args.output.parent / "metrics")
    serial_outputs, serial_metrics = run_pass(
        llm, rows, sampling, concurrency=1
    )
    concurrent_outputs, concurrent_metrics = run_pass(
        llm, rows, sampling, concurrency=64
    )
    output_equal = serial_outputs == concurrent_outputs
    metric_equal = serial_metrics == concurrent_metrics
    report = {
        "status": "pass" if output_equal and metric_equal else "fail",
        "model": args.model,
        "questions": 64,
        "capacity": args.capacity,
        "lambda": args.lambda_value,
        "top_j": 1,
        "draft_length": args.num_speculative_tokens,
        "bias_domain": "post_correction_selection_values",
        "concurrencies": [1, 64],
        "exact_outputs": output_equal,
        "exact_metrics": metric_equal,
        "serial_accuracy": sum(bool(row["correct"]) for row in serial_outputs) / 64,
        "concurrent_accuracy": sum(bool(row["correct"]) for row in concurrent_outputs) / 64,
        "serial_metrics": serial_metrics,
        "concurrent_metrics": concurrent_metrics,
        "serial_outputs": serial_outputs,
        "concurrent_outputs": concurrent_outputs,
        "configuration": configuration,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not key.endswith("outputs")}, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
