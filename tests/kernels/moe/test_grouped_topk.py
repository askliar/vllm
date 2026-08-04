# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MoE grouped topk kernel

Run `pytest tests/kernels/moe/test_grouped_topk.py`.
"""

import json

import pytest
import torch

import vllm.envs as envs
from vllm.config import (
    CompilationConfig,
    VllmConfig,
    get_cached_compilation_config,
    set_current_vllm_config,
)
from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
from vllm.model_executor.layers.fused_moe.router.cache_prior_router import (
    CachePriorRouter,
    update_lru_state,
    update_lru_state_batched,
)
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopk,
    fused_grouped_topk,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


class _StaticRouter(BaseRouter):
    def __init__(self, weights: torch.Tensor, ids: torch.Tensor) -> None:
        super().__init__(top_k=ids.shape[-1], global_num_experts=4)
        self.weights = weights
        self.ids = ids

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return RoutingMethodType.Sigmoid

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output_ids = self.ids
        if indices_type is not None:
            output_ids = output_ids.to(indices_type)
        return self.weights, output_ids


def _cache_prior_router(
    base_router: BaseRouter,
    *,
    lambda_value: float,
    capacity: int = 2,
    top_j: int = 1,
    metrics_path: str = "",
    trace_dir: str = "",
    reset_path: str = "",
    correction_bias: torch.Tensor | None = None,
    scoring_func: str = "sigmoid",
) -> CachePriorRouter:
    return CachePriorRouter(
        base_router,
        capacity=capacity,
        lambda_value=lambda_value,
        top_j=top_j,
        scoring_func=scoring_func,
        renormalize=False,
        routed_scaling_factor=1.0,
        e_score_correction_bias=correction_bias,
        num_expert_group=1,
        topk_group=1,
        metrics_path=metrics_path,
        trace_dir=trace_dir,
        reset_path=reset_path,
    )


def test_cache_prior_vectorized_lru_matches_scalar_reference():
    generator = torch.Generator().manual_seed(17)
    priorities = torch.rand((31, 3), generator=generator)
    expert_ids = torch.stack(
        [torch.randperm(8, generator=generator)[:3] for _ in range(31)]
    )

    hits, last_use, clock = update_lru_state(
        expert_ids,
        priorities,
        capacity=5,
        num_experts=8,
    )

    order: list[int] = []
    expected_hits: list[list[bool]] = []
    for ids, weights in zip(expert_ids, priorities, strict=True):
        expected_hits.append([int(expert_id) in order for expert_id in ids])
        touches = sorted(
            zip(ids.tolist(), weights.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        )
        for expert_id, _ in touches:
            if expert_id in order:
                order.remove(expert_id)
            elif len(order) == 5:
                order.pop(0)
            order.append(expert_id)

    torch.testing.assert_close(hits, torch.tensor(expected_hits))
    assert clock == expert_ids.numel()
    assert last_use.topk(5).indices.flip(0).tolist() == order


def test_cache_prior_vectorized_lru_ignores_invalid_padding_ids():
    expert_ids = torch.tensor([[0, -1], [1, 0], [-1, 2]])
    priorities = torch.tensor([[0.9, 0.0], [0.8, 0.7], [0.0, 0.6]])

    hits, last_use, clock = update_lru_state(
        expert_ids,
        priorities,
        capacity=2,
        num_experts=3,
    )

    torch.testing.assert_close(
        hits,
        torch.tensor([[False, False], [False, True], [False, False]]),
    )
    assert last_use.topk(2).indices.flip(0).tolist() == [0, 2]
    assert clock == expert_ids.numel()


def test_cache_prior_batched_lru_matches_independent_sequences():
    generator = torch.Generator().manual_seed(29)
    priorities = torch.rand((4, 17, 3), generator=generator)
    expert_ids = torch.stack(
        [
            torch.stack([torch.randperm(8, generator=generator)[:3] for _ in range(17)])
            for _ in range(4)
        ]
    )

    hits, last_use, clock = update_lru_state_batched(
        expert_ids,
        priorities,
        capacity=5,
        num_experts=8,
    )
    expected = [
        update_lru_state(
            ids,
            weights,
            capacity=5,
            num_experts=8,
        )
        for ids, weights in zip(expert_ids, priorities, strict=True)
    ]

    torch.testing.assert_close(hits, torch.stack([result[0] for result in expected]))
    torch.testing.assert_close(
        last_use, torch.stack([result[1] for result in expected])
    )
    assert clock == expected[0][2]


def test_cache_prior_scheduler_chunk_matches_serial_windows():
    logits = torch.tensor(
        [
            [
                [2.0, 1.0, 0.0, -1.0],
                [0.0, -1.0, 2.0, 1.0],
                [1.0, 0.0, 2.0, -1.0],
            ],
            [
                [-1.0, 2.0, 1.0, 0.0],
                [2.0, 0.0, -1.0, 1.0],
                [0.0, 2.0, 1.0, -1.0],
            ],
        ]
    )
    flat_logits = logits.view(-1, logits.shape[-1])
    base = _StaticRouter(
        torch.empty((flat_logits.shape[0], 2)),
        torch.empty((flat_logits.shape[0], 2), dtype=torch.int32),
    )
    batched = _cache_prior_router(base, lambda_value=1.0)
    # The evaluator queued five requests, but this scheduler step contains two
    # complete requests. Cache-Prior must derive the current chunk size.
    batched.set_evaluation_batch_layout(5, 3)

    batched_weights, batched_ids = batched._compute_routing(
        torch.empty((6, 0)), flat_logits, torch.int32
    )

    serial = _cache_prior_router(base, lambda_value=1.0)
    serial_results = [
        serial._compute_routing(torch.empty((3, 0)), window, torch.int32)
        for window in logits
    ]
    serial_weights = torch.cat([result[0] for result in serial_results])
    serial_ids = torch.cat([result[1] for result in serial_results])

    torch.testing.assert_close(batched_weights, serial_weights)
    torch.testing.assert_close(batched_ids, serial_ids)
    assert batched.metrics == serial.metrics


def test_cache_prior_lambda_zero_preserves_base_router_outputs():
    weights = torch.tensor([[0.9, 0.1], [0.8, 0.7], [0.6, 0.5]])
    ids = torch.tensor([[0, 1], [0, 2], [2, 3]], dtype=torch.int32)
    router = _cache_prior_router(_StaticRouter(weights, ids), lambda_value=0.0)

    actual_weights, actual_ids = router._compute_routing(
        torch.empty((3, 0)),
        torch.randn((3, 4)),
        torch.int32,
    )

    torch.testing.assert_close(actual_weights, weights)
    torch.testing.assert_close(actual_ids, ids)
    assert router.metrics.accesses == 6
    assert router.metrics.hits == 2
    assert router.metrics.misses == 4


def test_cache_prior_resets_logical_cache_for_each_prefill():
    weights = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    router = _cache_prior_router(_StaticRouter(weights, ids), lambda_value=0.0)

    for _ in range(2):
        router._compute_routing(
            torch.empty((2, 0)),
            torch.randn((2, 4)),
            torch.int32,
        )

    assert router.metrics.accesses == 8
    assert router.metrics.hits == 4


def test_cache_prior_writes_prefill_metrics(tmp_path):
    weights = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
    metrics_path = tmp_path / "metrics.jsonl"
    router = _cache_prior_router(
        _StaticRouter(weights, ids),
        lambda_value=0.0,
        metrics_path=str(metrics_path),
    )

    router._compute_routing(
        torch.empty((2, 0)),
        torch.randn((2, 4)),
        torch.int32,
    )

    record = json.loads(metrics_path.read_text())
    assert record["accesses"] == 4
    assert record["hits"] == 2
    assert record["misses"] == 2


@pytest.mark.parametrize("lambda_value", [0.0, 1.0])
def test_cache_prior_writes_replayable_prefill_trace(tmp_path, lambda_value):
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, -1.0, 2.0, 1.0]])
    weights = logits.sigmoid().topk(2, dim=-1).values
    ids = logits.topk(2, dim=-1).indices.to(torch.int32)
    router = _cache_prior_router(
        _StaticRouter(weights, ids),
        lambda_value=lambda_value,
        trace_dir=str(tmp_path),
    )

    selected_weights, selected_ids = router._compute_routing(
        torch.empty((2, 0)), logits, torch.int32
    )

    metadata_path = next(tmp_path.glob("*.jsonl"))
    record = json.loads(metadata_path.read_text())
    assert record["tokens"] == 2
    assert record["top_k"] == 2
    assert record["range_count_start"] == 0
    assert record["range_count_end"] == 2

    stem = metadata_path.with_suffix("")
    original = torch.from_file(
        str(stem) + ".original_ids.i16", dtype=torch.int16, size=4
    ).view(2, 2)
    selected = torch.from_file(
        str(stem) + ".selected_ids.i16", dtype=torch.int16, size=4
    ).view(2, 2)
    saved_weights = torch.from_file(
        str(stem) + ".selected_weights.f32", dtype=torch.float32, size=4
    ).view(2, 2)
    logit_range = torch.from_file(
        str(stem) + ".logit_range.f32", dtype=torch.float32, size=2
    )
    range_mean = torch.from_file(
        str(stem) + ".range_mean.f32", dtype=torch.float32, size=2
    )

    torch.testing.assert_close(original, ids.to(torch.int16))
    torch.testing.assert_close(selected, selected_ids.to(torch.int16))
    torch.testing.assert_close(saved_weights, selected_weights)
    torch.testing.assert_close(logit_range, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(range_mean, torch.tensor([3.0, 3.0]))


def test_cache_prior_resets_profiled_range_state_when_signaled(tmp_path):
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, -1.0, 2.0, 1.0]])
    weights = logits.sigmoid().topk(2, dim=-1).values
    ids = logits.topk(2, dim=-1).indices.to(torch.int32)
    reset_path = tmp_path / "range-reset"
    router = _cache_prior_router(
        _StaticRouter(weights, ids),
        lambda_value=0.0,
        reset_path=str(reset_path),
    )
    router._compute_routing(torch.empty((2, 0)), logits, torch.int32)
    assert router._range_count == 2

    reset_path.touch()
    router._compute_routing(torch.empty((2, 0)), logits, torch.int32)

    assert router._range_count == 2
    assert router._range_reset_applied


def test_cache_prior_promotes_cached_expert_and_keeps_original_weight():
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 2.0, 1.0],
        ]
    )
    base = _StaticRouter(
        torch.empty((2, 2)),
        torch.empty((2, 2), dtype=torch.int32),
    )
    router = _cache_prior_router(base, lambda_value=1.0)

    selected_weights, selected_ids = router._compute_routing(
        torch.empty((2, 0)), logits, torch.int32
    )

    assert selected_ids[1].tolist() == [2, 0]
    expected_scores = logits.sigmoid()[1].gather(0, selected_ids[1].long())
    torch.testing.assert_close(selected_weights[1], expected_scores)
    assert router.metrics.hits == 1
    assert router.metrics.changed_tokens == 1
    assert router.metrics.top_j_violations == 0


def test_cache_prior_bias_is_applied_to_raw_logits_before_softmax():
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 10.0, 9.0],
        ]
    )
    base = _StaticRouter(
        torch.empty((2, 2)),
        torch.empty((2, 2), dtype=torch.int32),
    )
    router = _cache_prior_router(
        base,
        lambda_value=0.5,
        scoring_func="softmax",
    )

    _, selected_ids = router._compute_routing(torch.empty((2, 0)), logits, torch.int32)

    # Adding 0.5 * the raw-logit range does not yet promote cached expert 0.
    # Adding the same normalized-probability range would promote it here.
    assert selected_ids[1].tolist() == [2, 3]


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_cache_prior_restores_loader_migrated_state_to_cpu():
    logits = torch.tensor(
        [
            [2.0, 1.0, 0.0, -1.0],
            [0.0, -1.0, 2.0, 1.0],
        ],
        device="cuda",
    )
    base = _StaticRouter(
        torch.empty((2, 2), device="cuda"),
        torch.empty((2, 2), dtype=torch.int32, device="cuda"),
    )
    router = _cache_prior_router(base, lambda_value=1.0)
    router._last_use = router._last_use.cuda()

    selected_weights, selected_ids = router._compute_routing(
        torch.empty((2, 0), device="cuda"), logits, torch.int32
    )

    assert router._last_use.device.type == "cpu"
    assert selected_weights.is_cuda
    assert selected_ids.is_cuda
    assert selected_ids[1].tolist() == [2, 0]


def test_cache_prior_correction_bias_only_affects_expert_selection():
    logits = torch.zeros((1, 4))
    base = _StaticRouter(
        torch.empty((1, 2)),
        torch.empty((1, 2), dtype=torch.int32),
    )
    router = _cache_prior_router(
        base,
        lambda_value=1.0,
        correction_bias=torch.tensor([2.0, 1.0, 0.0, -1.0]),
    )

    selected_weights, selected_ids = router._compute_routing(
        torch.empty((1, 0)), logits, torch.int32
    )

    assert selected_ids.tolist() == [[0, 1]]
    torch.testing.assert_close(selected_weights, torch.full((1, 2), 0.5))


def _run_single_group_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return fused_grouped_topk(
        hidden_states=torch.empty(
            (logits.shape[0], 0), dtype=logits.dtype, device=logits.device
        ),
        gating_output=logits,
        topk=topk,
        renormalize=renormalize,
        e_score_correction_bias=bias,
        num_expert_group=1,
        topk_group=1,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
    )


def _single_group_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    *,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scoring_func == "sigmoid":
        scores = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5
    else:
        scores = torch.softmax(logits, dim=-1).float()
    indices = torch.argsort(
        scores + bias.float(), dim=-1, descending=True, stable=True
    )[:, :topk]
    values = scores.gather(1, indices)
    if renormalize:
        values /= values.sum(dim=-1, keepdim=True) + 1e-20
    values *= routed_scaling_factor
    return values, indices.to(torch.int32)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("n_token", [1, 33, 64])
@pytest.mark.parametrize("n_hidden", [1024, 2048])
@pytest.mark.parametrize(
    "n_expert,topk,num_expert_group,topk_group",
    [
        (16, 2, 8, 2),
        (128, 2, 8, 2),
        (256, 8, 8, 4),
        (384, 8, 1, 1),
        (512, 22, 1, 1),
    ],
)
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize("scoring_func", ["softmax", "sigmoid"])
@pytest.mark.parametrize("routed_scaling_factor", [1.0, 2.5])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias_dtype", [torch.float32])
def test_grouped_topk(
    monkeypatch: pytest.MonkeyPatch,
    n_token: int,
    n_hidden: int,
    n_expert: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    renormalize: bool,
    scoring_func: str,
    routed_scaling_factor: float,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
):
    vllm_config = VllmConfig(
        compilation_config=CompilationConfig(custom_ops=["all", "+grouped_topk"])
    )
    get_cached_compilation_config.cache_clear()

    set_random_seed(0)
    hidden_states = torch.randn((n_token, n_hidden), dtype=input_dtype, device="cuda")
    gating_output = torch.randn((n_token, n_expert), dtype=input_dtype, device="cuda")
    e_score_correction_bias = torch.randn((n_expert,), dtype=bias_dtype, device="cuda")

    with set_current_vllm_config(vllm_config), monkeypatch.context() as m:
        m.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
        m.setattr(envs, "VLLM_BATCH_INVARIANT", True)
        grouped_topk = GroupedTopk(
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )
        assert grouped_topk._forward_method.__name__ == "forward_cuda"
        baseline_topk_weights, baseline_topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            e_score_correction_bias=e_score_correction_bias,
        )

        test_topk_weights, test_topk_ids = fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
        )

        torch.testing.assert_close(
            baseline_topk_weights, test_topk_weights, atol=2e-2, rtol=0
        )
        torch.testing.assert_close(baseline_topk_ids, test_topk_ids, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_grouped_topk_single_group_large_batch():
    set_random_seed(0)
    logits = torch.randn((1536, 896), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((896,), dtype=torch.float32, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits, bias, 16, scoring_func="sigmoid", renormalize=True
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,input_dtype,bias_dtype",
    [
        (512, 9, torch.bfloat16, torch.float32),
        (512, 16, torch.float16, torch.float16),
        (513, 9, torch.float32, torch.bfloat16),
        (513, 16, torch.bfloat16, torch.float32),
        (895, 9, torch.float16, torch.bfloat16),
        (896, 16, torch.float32, torch.float16),
        (897, 9, torch.bfloat16, torch.bfloat16),
        (897, 16, torch.float16, torch.float32),
        (1024, 9, torch.float32, torch.bfloat16),
        (1024, 16, torch.bfloat16, torch.float16),
    ],
)
@pytest.mark.parametrize(
    "scoring_func,renormalize,routed_scaling_factor",
    [
        ("sigmoid", True, 1.0),
        ("sigmoid", False, 2.5),
        ("softmax", True, 2.5),
        ("softmax", False, 1.0),
    ],
)
def test_grouped_topk_single_group_tiers(
    num_experts: int,
    topk: int,
    input_dtype: torch.dtype,
    bias_dtype: torch.dtype,
    scoring_func: str,
    renormalize: bool,
    routed_scaling_factor: float,
):
    set_random_seed(7)
    logits = torch.randn((17, num_experts), dtype=input_dtype, device="cuda")
    bias = torch.randn((num_experts,), dtype=bias_dtype, device="cuda")

    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize(
    "num_experts,topk,scoring_func",
    [
        (128, 8, "sigmoid"),
        (129, 8, "sigmoid"),
        (257, 8, "sigmoid"),
        (385, 8, "sigmoid"),
        (512, 9, "sigmoid"),
        (513, 9, "sigmoid"),
        (769, 9, "sigmoid"),
        (897, 16, "sigmoid"),
        (1024, 16, "sigmoid"),
        (128, 4, "softmax"),
        (128, 5, "softmax"),
        (129, 8, "softmax"),
        (161, 8, "softmax"),
        (256, 9, "softmax"),
        (257, 8, "softmax"),
        (512, 9, "softmax"),
        (512, 17, "softmax"),
        (512, 23, "softmax"),
        (513, 8, "softmax"),
        (577, 9, "softmax"),
        (769, 9, "softmax"),
        (897, 9, "softmax"),
        (1024, 16, "softmax"),
    ],
)
def test_grouped_topk_single_group_capacity_tiers(
    num_experts: int,
    topk: int,
    scoring_func: str,
):
    set_random_seed(11)
    logits = torch.randn((3, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn((num_experts,), dtype=torch.float32, device="cuda")
    expected_values, expected_ids = _single_group_reference(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )
    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        topk,
        scoring_func=scoring_func,
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
def test_grouped_topk_single_group_stable_ties(num_experts: int):
    logits = torch.zeros((1, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=True,
        routed_scaling_factor=2.5,
    )

    expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
    expected_values = torch.full((1, 16), 2.5 / 16, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
@pytest.mark.parametrize("num_experts", [512, 896, 1024])
@pytest.mark.parametrize("num_finite", [0, 15])
@pytest.mark.parametrize("renormalize", [False, True])
def test_grouped_topk_single_group_nonfinite_scores(
    num_experts: int, num_finite: int, renormalize: bool
):
    logits = torch.full(
        (1, num_experts), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    if num_finite:
        logits[0, :num_finite] = torch.arange(
            num_finite, dtype=torch.bfloat16, device="cuda"
        )
    logits[0, num_finite] = torch.inf
    logits[0, num_finite + 1] = -torch.inf
    bias = torch.zeros((num_experts,), dtype=torch.float32, device="cuda")

    actual_values, actual_ids = _run_single_group_topk(
        logits,
        bias,
        16,
        scoring_func="sigmoid",
        renormalize=renormalize,
        routed_scaling_factor=2.5,
    )

    if num_finite == 0:
        expected_ids = torch.arange(16, dtype=torch.int32, device="cuda")[None]
        if renormalize:
            expected_values = torch.full(
                (1, 16), 1 / 16, dtype=torch.float32, device="cuda"
            )
        else:
            expected_values = torch.zeros((1, 16), dtype=torch.float32, device="cuda")
    else:
        expected_ids = torch.cat(
            (
                torch.arange(num_finite - 1, -1, -1, dtype=torch.int32, device="cuda"),
                torch.tensor([num_finite], dtype=torch.int32, device="cuda"),
            )
        )[None]
        finite_values = logits[0, :num_finite].float().sigmoid().flip(0)
        if renormalize:
            finite_values /= finite_values.sum()
        finite_values *= 2.5
        expected_values = torch.cat(
            (finite_values, torch.zeros(1, dtype=torch.float32, device="cuda"))
        )[None]

    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_values, expected_values, atol=2e-5, rtol=0)
