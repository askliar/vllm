# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-level parity: ReplaySSM decode vs the baseline recurrent kernels."""

import os
from importlib import import_module
from inspect import signature

import pytest

import vllm.envs as envs
from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from vllm.platforms import current_platform
from vllm.v1.metrics.reader import Counter

from ...models.utils import check_logprobs_close
from ...utils import large_gpu_mark, multi_gpu_test

try:
    from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
        CheckpointingSSURunner,
        allocate_checkpointing_ssu_scratch,
    )

    HAS_FLASHINFER_CHECKPOINTING_SSU = True
except ImportError:
    HAS_FLASHINFER_CHECKPOINTING_SSU = False

try:
    from flashinfer.mamba.replayssm_materialize import replayssm_materialize

    HAS_FLASHINFER_REPLAYSSM_MATERIALIZE = (
        "active_request_indices" in signature(replayssm_materialize).parameters
    )
except ImportError:
    HAS_FLASHINFER_REPLAYSSM_MATERIALIZE = False

# Mamba2 (Nemotron-3) hybrid.
MAMBA2_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
MAMBA2_MTP_MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
MAMBA2_PREFIX_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"
GDN_MODEL = "Qwen/Qwen3.5-0.8B"
# Same-node samples scored 0.3245 for standard GDN and 0.3245/0.3177 for
# ReplaySSM. Keep an absolute sanity floor plus a tighter matched-control bound.
GDN_GSM8K_MIN_ACCURACY = 0.30
GDN_GSM8K_MAX_ACCURACY_DROP = 0.015
# The BF16 FlashInfer gate casts can shift low-probability logits slightly;
# same-node n-gram coverage observed a 0.1054 chosen-token difference.
GDN_DECODE_LOGPROB_ATOL = 0.12
GDN_PROMPTS = [
    "The capital of France is",
    "The capital of Germany is",
]
MODELS = [
    pytest.param(MAMBA2_MODEL, marks=large_gpu_mark(min_gb=40)),
]

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
]

requires_flashinfer_replayssm_materialization = pytest.mark.skipif(
    not (HAS_FLASHINFER_CHECKPOINTING_SSU and HAS_FLASHINFER_REPLAYSSM_MATERIALIZE),
    reason="FlashInfer ReplaySSM materialization APIs not available",
)
requires_gdn_replayssm = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(90)),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
requires_gdn_gsm8k = pytest.mark.skipif(
    os.getenv("RUN_GDN_GSM8K") != "1",
    reason="set RUN_GDN_GSM8K=1 to run the local full GSM8K quality gate",
)


@pytest.fixture(autouse=True)
def _use_v1_model_runner_by_default(monkeypatch):
    # Triton ReplaySSM is V1-only. FlashInfer V2 tests override this locally.
    with monkeypatch.context() as patch:
        patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        envs.disable_envs_cache()
        yield
    envs.disable_envs_cache()


def _check_replayssm_parity(
    vllm_runner,
    model_name,
    *,
    tensor_parallel_size=1,
    mamba_backend: str = "triton",
    name_1: str = "replayssm",
    expected_v2: bool | None = None,
):
    # Compare logprobs, not greedy ids: ReplaySSM's fp arithmetic can flip a
    # near-tie. Baseline and ReplaySSM run at the same TP, so TP numerics are
    # common-mode and only ReplaySSM varies.
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        tensor_parallel_size=tensor_parallel_size,
        mamba_backend=mamba_backend,
    )
    with vllm_runner(model_name, **common) as llm:
        if expected_v2 is not None:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is expected_v2
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        if expected_v2 is not None:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is expected_v2
        replay = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline",
        name_1=name_1,
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_decode_matches_baseline(vllm_runner, model_name):
    _check_replayssm_parity(vllm_runner, model_name)


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model_name", [MAMBA2_MODEL])
def test_replayssm_decode_matches_baseline_tp2(vllm_runner, model_name):
    # Tensor-parallel correctness: ReplaySSM's caches and checkpoint state are
    # sharded per rank, so TP2 decode must still match the baseline at TP2.
    _check_replayssm_parity(vllm_runner, model_name, tensor_parallel_size=2)


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replayssm_flashinfer_decode_matches_baseline(
    vllm_runner, model_name, monkeypatch, use_v2_model_runner
):
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
            envs.disable_envs_cache()
            _check_replayssm_parity(
                vllm_runner,
                model_name,
                mamba_backend="flashinfer",
                name_1="replayssm_flashinfer",
                expected_v2=use_v2_model_runner,
            )
    finally:
        # The context restores the environment before the final cache reset.
        envs.disable_envs_cache()


@pytest.mark.parametrize("model_name", MODELS)
def test_replayssm_flashinfer_spec_decode_matches_baseline(vllm_runner, model_name):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 3,
            "prompt_lookup_max": 3,
        },
    )
    with vllm_runner(model_name, **common) as llm:
        baseline = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)
    with vllm_runner(
        model_name, use_replayssm=True, replayssm_buffer_len=16, **common
    ) as llm:
        replay = llm.generate_greedy_logprobs(PROMPTS, max_tokens=32, num_logprobs=5)

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline_spec",
        name_1="replayssm_flashinfer_spec",
    )


@multi_gpu_test(num_gpus=2)
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("use_v2_model_runner", [False, True], ids=["v1", "v2"])
def test_replayssm_flashinfer_mtp(vllm_runner, monkeypatch, use_v2_model_runner):
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        tensor_parallel_size=2,
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
            envs.disable_envs_cache()
            with vllm_runner(MAMBA2_MTP_MODEL, **common) as llm:
                assert (
                    llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    is use_v2_model_runner
                )
                baseline = llm.generate_greedy_logprobs(
                    PROMPTS, max_tokens=32, num_logprobs=5
                )
            with vllm_runner(
                MAMBA2_MTP_MODEL,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                assert (
                    llm.llm.llm_engine.vllm_config.use_v2_model_runner
                    is use_v2_model_runner
                )
                replay = llm.generate_greedy_logprobs(
                    PROMPTS, max_tokens=32, num_logprobs=5
                )
                draft_count = sum(
                    metric.value
                    for metric in llm.llm.get_metrics()
                    if isinstance(metric, Counter)
                    and metric.name == "vllm:spec_decode_num_drafts"
                )
    finally:
        envs.disable_envs_cache()

    assert any(len(output[0]) > 16 for output in replay)
    assert draft_count > 0
    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0=f"baseline_mtp_{'v2' if use_v2_model_runner else 'v1'}",
        name_1=f"replayssm_flashinfer_mtp_{'v2' if use_v2_model_runner else 'v1'}",
    )


def _require_gdn_replayssm_kernel(
    *, native_stp: bool = False, prefix_materialization: bool = False
) -> None:
    module = (
        "flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_stp"
        if native_stp
        else "flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_flush"
    )
    try:
        import_module(module)
        if prefix_materialization:
            import_module("flashinfer.gdn_kernels.gdn_prefix_materialize")
    except ImportError as exc:
        pytest.skip(
            f"FlashInfer GDN ReplaySSM API not available on SM90+: {exc}",
        )


def _gdn_speculative_config(method: str | None, num_speculative_tokens: int):
    if method is None:
        return None
    config: dict[str, int | str] = {
        "method": method,
        "num_speculative_tokens": num_speculative_tokens,
    }
    if method == "ngram":
        config["prompt_lookup_max"] = num_speculative_tokens
    return config


def _check_gdn_decode_trajectory(baseline, replay) -> None:
    """Compare every step after requiring both engines to follow one path."""
    assert len(baseline) == len(replay)
    for prompt_idx, (baseline_output, replay_output) in enumerate(
        zip(baseline, replay)
    ):
        baseline_ids, _, baseline_logprobs = baseline_output
        replay_ids, _, replay_logprobs = replay_output
        assert len(baseline_ids) > 16
        assert replay_ids == baseline_ids, (
            f"GDN decode trajectory diverged for prompt {prompt_idx}: "
            f"baseline={baseline_ids}, replay={replay_ids}"
        )
        assert baseline_logprobs is not None
        assert replay_logprobs is not None
        for step, token_id in enumerate(baseline_ids):
            baseline_value = baseline_logprobs[step][token_id].logprob
            replay_value = replay_logprobs[step][token_id].logprob
            assert replay_value == pytest.approx(
                baseline_value, abs=GDN_DECODE_LOGPROB_ATOL
            ), (
                f"GDN chosen-token logprob mismatch for prompt {prompt_idx}, "
                f"step {step}, token {token_id}: baseline={baseline_value}, "
                f"replay={replay_value}"
            )


def _check_gdn_replayssm_parity(
    vllm_runner,
    patch: pytest.MonkeyPatch,
    *,
    use_v2_model_runner: bool,
    speculative_method: str | None,
    num_speculative_tokens: int,
) -> None:
    patch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    _require_gdn_replayssm_kernel(native_stp=num_speculative_tokens == 0)
    patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))

    common = dict(
        max_model_len=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.55,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="bfloat16",
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    speculative_config = _gdn_speculative_config(
        speculative_method, num_speculative_tokens
    )
    if speculative_config is not None:
        common["speculative_config"] = speculative_config

    # Leave the control selector unset so vLLM may use the fused CUDA kernel
    # when it is built and otherwise take its normal Triton fallback.
    patch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    envs.disable_envs_cache()
    with vllm_runner(GDN_MODEL, **common) as llm:
        config = llm.llm.llm_engine.vllm_config
        assert config.use_v2_model_runner is use_v2_model_runner
        assert not config.is_gdn_replayssm_enabled()
        baseline = llm.generate_greedy_logprobs(
            GDN_PROMPTS, max_tokens=32, num_logprobs=5
        )

    patch.setenv("VLLM_GDN_DECODE_KERNEL", "flashinfer_replayssm")
    envs.disable_envs_cache()
    with vllm_runner(
        GDN_MODEL,
        use_replayssm=True,
        replayssm_buffer_len=16,
        **common,
    ) as llm:
        config = llm.llm.llm_engine.vllm_config
        assert config.use_v2_model_runner is use_v2_model_runner
        assert config.is_gdn_replayssm_enabled()
        replay = llm.generate_greedy_logprobs(
            GDN_PROMPTS, max_tokens=32, num_logprobs=5
        )
        draft_count = sum(
            metric.value
            for metric in llm.llm.get_metrics()
            if isinstance(metric, Counter)
            and metric.name == "vllm:spec_decode_num_drafts"
        )

    if speculative_method == "mtp":
        assert draft_count > 0
    _check_gdn_decode_trajectory(baseline, replay)


@requires_gdn_replayssm
@pytest.mark.parametrize(
    ("use_v2_model_runner", "speculative_method", "num_speculative_tokens"),
    [
        pytest.param(False, None, 0, id="v1-stp"),
        pytest.param(True, None, 0, id="v2-stp"),
        pytest.param(False, "ngram", 3, id="v1-ngram-t4"),
        pytest.param(False, "mtp", 3, id="v1-mtp-t4"),
        pytest.param(True, "mtp", 3, id="v2-mtp-t4"),
        pytest.param(False, "mtp", 5, id="v1-mtp-dl5-padded-t8"),
        pytest.param(True, "mtp", 5, id="v2-mtp-dl5-padded-t8"),
        pytest.param(False, "mtp", 7, id="v1-mtp-t8"),
        pytest.param(True, "mtp", 7, id="v2-mtp-t8"),
    ],
)
def test_gdn_replayssm_decode_matches_baseline(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    use_v2_model_runner: bool,
    speculative_method: str | None,
    num_speculative_tokens: int,
):
    try:
        with monkeypatch.context() as patch:
            _check_gdn_replayssm_parity(
                vllm_runner,
                patch,
                use_v2_model_runner=use_v2_model_runner,
                speculative_method=speculative_method,
                num_speculative_tokens=num_speculative_tokens,
            )
    finally:
        envs.disable_envs_cache()


@requires_gdn_replayssm
@requires_gdn_gsm8k
def test_gdn_replayssm_mtp_gsm8k_v2(vllm_runner, monkeypatch: pytest.MonkeyPatch):
    """Guard full GSM8K quality while the GDN ReplaySSM MTP path is active."""
    common = dict(
        max_model_len=4096,
        max_num_seqs=128,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        mamba_cache_mode="none",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="bfloat16",
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
            _require_gdn_replayssm_kernel()
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

            patch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
            envs.disable_envs_cache()
            with vllm_runner(GDN_MODEL, **common) as llm:
                config = llm.llm.llm_engine.vllm_config
                assert config.use_v2_model_runner
                assert not config.is_gdn_replayssm_enabled()
                baseline_results = evaluate_gsm8k_offline(llm.llm, num_questions=1319)

            patch.setenv("VLLM_GDN_DECODE_KERNEL", "flashinfer_replayssm")
            envs.disable_envs_cache()
            with vllm_runner(
                GDN_MODEL,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                config = llm.llm.llm_engine.vllm_config
                assert config.use_v2_model_runner
                assert config.is_gdn_replayssm_enabled()
                results = evaluate_gsm8k_offline(llm.llm, num_questions=1319)
                draft_count = sum(
                    metric.value
                    for metric in llm.llm.get_metrics()
                    if isinstance(metric, Counter)
                    and metric.name == "vllm:spec_decode_num_drafts"
                )
    finally:
        envs.disable_envs_cache()

    baseline_accuracy = float(baseline_results["accuracy"])
    accuracy = float(results["accuracy"])
    invalid_rate = float(results["invalid_rate"])
    assert baseline_results["num_questions"] == 1319
    assert results["num_questions"] == 1319
    assert draft_count > 0
    print(
        f"GDN V2 MTP3: baseline GSM8K accuracy={baseline_accuracy:.3f}, "
        f"ReplaySSM accuracy={accuracy:.3f}, "
        f"invalid_rate={invalid_rate:.3f}, drafts={draft_count:g}"
    )
    assert accuracy >= GDN_GSM8K_MIN_ACCURACY, (
        f"GDN ReplaySSM GSM8K accuracy {accuracy:.3f} is below the "
        f"Qwen3.5-0.8B floor {GDN_GSM8K_MIN_ACCURACY:.3f}; "
        f"invalid_rate={invalid_rate:.3f}, drafts={draft_count:g}"
    )
    assert accuracy >= baseline_accuracy - GDN_GSM8K_MAX_ACCURACY_DROP, (
        f"GDN ReplaySSM GSM8K accuracy {accuracy:.3f} trails matched baseline "
        f"{baseline_accuracy:.3f} by more than "
        f"{GDN_GSM8K_MAX_ACCURACY_DROP:.3f}; invalid_rate={invalid_rate:.3f}, "
        f"drafts={draft_count:g}"
    )


# Prefix spans several mamba blocks; prefix caching only reuses full blocks.
_PC_SENTENCE = (
    "In a detailed survey of state space models, the authors compared many "
    "architectures across a wide range of long-context language tasks and "
    "measured their throughput, memory use, and accuracy in careful detail. "
)
_PC_PREFIX = _PC_SENTENCE * 120
PREFIX_CACHING_PROMPTS = [
    _PC_PREFIX + "The most important conclusion was that",
    _PC_PREFIX + "Surprisingly, the experiments showed that",
    _PC_PREFIX + "The most important conclusion was that",
]
# All-mode MTP must cache at least two full state blocks: the drafter drops
# the volatile trailing block before resuming from the preceding boundary.
_PC_MTP_PREFIX = _PC_SENTENCE * 240
MTP_PREFIX_CACHING_PROMPTS = [
    _PC_MTP_PREFIX + prompt.removeprefix(_PC_PREFIX)
    for prompt in PREFIX_CACHING_PROMPTS
]


def _prefix_cache_hits(llm) -> int:
    return sum(
        m.value
        for m in llm.llm.get_metrics()
        if isinstance(m, Counter) and m.name == "vllm:prefix_cache_hits"
    )


def _check_gdn_replayssm_prefix_caching(
    vllm_runner,
    patch: pytest.MonkeyPatch,
    *,
    use_v2_model_runner: bool,
    num_speculative_tokens: int,
) -> None:
    patch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    patch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    _require_gdn_replayssm_kernel(
        native_stp=num_speculative_tokens == 0,
        prefix_materialization=True,
    )
    common = dict(
        max_model_len=8192,
        max_num_seqs=8,
        gpu_memory_utilization=0.55,
        trust_remote_code=True,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="align",
        mamba_backend="triton",
        mamba_cache_dtype="bfloat16",
        mamba_ssm_cache_dtype="bfloat16",
        disable_log_stats=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if num_speculative_tokens:
        common["speculative_config"] = {
            "method": "mtp",
            "num_speculative_tokens": num_speculative_tokens,
        }

    patch.delenv("VLLM_GDN_DECODE_KERNEL", raising=False)
    envs.disable_envs_cache()
    with vllm_runner(GDN_MODEL, **common) as llm:
        config = llm.llm.llm_engine.vllm_config
        assert config.use_v2_model_runner is use_v2_model_runner
        assert not config.is_gdn_replayssm_enabled()
        llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
        baseline = llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
        assert _prefix_cache_hits(llm) > 0

    patch.setenv("VLLM_GDN_DECODE_KERNEL", "flashinfer_replayssm")
    envs.disable_envs_cache()
    with vllm_runner(
        GDN_MODEL,
        use_replayssm=True,
        replayssm_buffer_len=16,
        **common,
    ) as llm:
        config = llm.llm.llm_engine.vllm_config
        assert config.use_v2_model_runner is use_v2_model_runner
        assert config.is_gdn_replayssm_enabled()
        first_pass = llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
        first_pass_hits = _prefix_cache_hits(llm)
        cached = llm.generate_greedy_logprobs(
            PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
        )
        cached_hits = _prefix_cache_hits(llm)
        draft_count = sum(
            metric.value
            for metric in llm.llm.get_metrics()
            if isinstance(metric, Counter)
            and metric.name == "vllm:spec_decode_num_drafts"
        )

    assert cached_hits > first_pass_hits
    if num_speculative_tokens:
        assert draft_count > 0
    _check_gdn_decode_trajectory(first_pass, cached)
    _check_gdn_decode_trajectory(baseline, cached)


@requires_gdn_replayssm
@pytest.mark.parametrize(
    ("use_v2_model_runner", "num_speculative_tokens"),
    [
        pytest.param(False, 0, id="v1-stp"),
        pytest.param(True, 0, id="v2-stp"),
        pytest.param(True, 3, id="v2-mtp-t4"),
        pytest.param(True, 5, id="v2-mtp-dl5-padded-t8"),
    ],
)
def test_gdn_replayssm_align_prefix_cache_matches_baseline(
    vllm_runner,
    monkeypatch: pytest.MonkeyPatch,
    use_v2_model_runner: bool,
    num_speculative_tokens: int,
):
    try:
        with monkeypatch.context() as patch:
            _check_gdn_replayssm_prefix_caching(
                vllm_runner,
                patch,
                use_v2_model_runner=use_v2_model_runner,
                num_speculative_tokens=num_speculative_tokens,
            )
    finally:
        envs.disable_envs_cache()


def _check_replayssm_prefix_caching(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mamba_cache_mode: str,
    moe_backend: str | None = None,
    use_ngram: bool,
    use_v2: bool,
    tensor_parallel_size: int,
    mamba_backend: str = "flashinfer",
):
    def run() -> None:
        # ReplaySSM materializes the exact SSM state at each cacheable block
        # boundary, so cached prefixes must match the always-materialized baseline.
        common = dict(
            max_model_len=8192,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            mamba_cache_mode=mamba_cache_mode,
            mamba_backend=mamba_backend,
            disable_log_stats=False,  # required for llm.get_metrics()
            tensor_parallel_size=tensor_parallel_size,
        )
        if moe_backend is not None:
            common["moe_backend"] = moe_backend
        if use_ngram:
            common["speculative_config"] = {
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_max": 3,
            }

        with vllm_runner(model_name, **common) as llm:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
            baseline_block_size = llm.llm.llm_engine.vllm_config.cache_config.block_size
            llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            baseline = llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            baseline_hits = _prefix_cache_hits(llm)

        with vllm_runner(
            model_name, use_replayssm=True, replayssm_buffer_len=16, **common
        ) as llm:
            assert llm.llm.llm_engine.vllm_config.use_v2_model_runner is use_v2
            replay_block_size = llm.llm.llm_engine.vllm_config.cache_config.block_size
            if mamba_backend == "flashinfer":
                # FlashInfer rings are auxiliary and cannot affect the shared page.
                assert replay_block_size == baseline_block_size
            else:
                # Triton retains the original packed five-state page. Its rings may
                # increase the attention block size needed to match that page.
                assert replay_block_size >= baseline_block_size
            llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            replay = llm.generate_greedy_logprobs(
                PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
            )
            replay_hits = _prefix_cache_hits(llm)

        assert baseline_hits > 0
        assert replay_hits > 0, (
            f"ReplaySSM {mamba_cache_mode}-mode run produced no prefix-cache hits; "
            "the shared prefix may be shorter than one mamba block, so prefix "
            "caching is inert"
        )
        check_logprobs_close(
            outputs_0_lst=baseline,
            outputs_1_lst=replay,
            name_0=f"{mamba_backend}_baseline_{mamba_cache_mode}_pc",
            name_1=f"{mamba_backend}_replayssm_{mamba_cache_mode}_pc",
        )

    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2 else "0")
            envs.disable_envs_cache()
            run()
    finally:
        envs.disable_envs_cache()


@requires_flashinfer_replayssm_materialization
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize(
    ("use_v2", "use_ngram"),
    [
        pytest.param(False, True, id="align-v1-ngram-t4"),
        pytest.param(True, False, id="align-v2-stp"),
    ],
)
def test_flashinfer_replayssm_prefix_cache_tp1(
    vllm_runner,
    model_name,
    monkeypatch: pytest.MonkeyPatch,
    use_v2: bool,
    use_ngram: bool,
):
    _check_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        mamba_cache_mode="align",
        use_ngram=use_ngram,
        use_v2=use_v2,
        tensor_parallel_size=1,
    )


@pytest.mark.parametrize("model_name", MODELS)
def test_triton_replayssm_align_prefix_cache_matches_baseline_v1(
    vllm_runner, model_name, monkeypatch: pytest.MonkeyPatch
):
    _check_replayssm_prefix_caching(
        vllm_runner,
        model_name,
        monkeypatch,
        mamba_cache_mode="align",
        use_ngram=False,
        use_v2=False,
        tensor_parallel_size=1,
        mamba_backend="triton",
    )


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
@pytest.mark.parametrize("use_v2", [False, True], ids=["v1", "v2"])
def test_flashinfer_replayssm_all_prefix_cache(vllm_runner, monkeypatch, use_v2: bool):
    _check_replayssm_prefix_caching(
        vllm_runner,
        MAMBA2_PREFIX_MODEL,
        monkeypatch,
        mamba_cache_mode="all",
        moe_backend="triton",
        use_ngram=False,
        use_v2=use_v2,
        tensor_parallel_size=1,
    )


@requires_flashinfer_replayssm_materialization
@large_gpu_mark(min_gb=40)
def test_flashinfer_replayssm_all_prefix_cache_mtp_v2(vllm_runner, monkeypatch):
    common = dict(
        max_model_len=12288,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        mamba_cache_mode="all",
        mamba_backend="flashinfer",
        disable_log_stats=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )
    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            envs.disable_envs_cache()
            with vllm_runner(
                MAMBA2_MTP_MODEL,
                use_replayssm=True,
                replayssm_buffer_len=16,
                **common,
            ) as llm:
                assert llm.llm.llm_engine.vllm_config.use_v2_model_runner
                first_pass = llm.generate_greedy_logprobs(
                    MTP_PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
                )
                first_pass_hits = _prefix_cache_hits(llm)
                cached = llm.generate_greedy_logprobs(
                    MTP_PREFIX_CACHING_PROMPTS, max_tokens=32, num_logprobs=5
                )
                cached_hits = _prefix_cache_hits(llm)
                draft_count = sum(
                    metric.value
                    for metric in llm.llm.get_metrics()
                    if isinstance(metric, Counter)
                    and metric.name == "vllm:spec_decode_num_drafts"
                )
    finally:
        envs.disable_envs_cache()

    assert cached_hits > first_pass_hits
    assert draft_count > 0
    check_logprobs_close(
        outputs_0_lst=first_pass,
        outputs_1_lst=cached,
        name_0="replayssm_all_mtp_v2_first_pass",
        name_1="replayssm_all_mtp_v2_cached",
    )
