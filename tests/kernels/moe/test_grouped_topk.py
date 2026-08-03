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
    correction_bias: torch.Tensor | None = None,
) -> CachePriorRouter:
    return CachePriorRouter(
        base_router,
        capacity=capacity,
        lambda_value=lambda_value,
        top_j=top_j,
        scoring_func="sigmoid",
        renormalize=False,
        routed_scaling_factor=1.0,
        e_score_correction_bias=correction_bias,
        num_expert_group=1,
        topk_group=1,
        metrics_path=metrics_path,
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
