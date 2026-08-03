# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.model_executor.layers.fused_moe.config import RoutingMethodType
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


@dataclass(frozen=True)
class CachePriorMetrics:
    accesses: int
    hits: int
    changed_tokens: int
    top_j_violations: int

    @property
    def misses(self) -> int:
        return self.accesses - self.hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.accesses if self.accesses else 0.0


def update_lru_state(
    expert_ids: torch.Tensor,
    priorities: torch.Tensor,
    *,
    capacity: int,
    num_experts: int,
    last_use: torch.Tensor | None = None,
    clock: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute fixed-trace LRU hits and final state without a token loop."""
    if expert_ids.ndim != 2:
        raise ValueError("expert_ids must have shape [tokens, top_k]")
    if priorities.shape != expert_ids.shape:
        raise ValueError("priorities must have the same shape as expert_ids")
    if capacity <= 0 or capacity > num_experts:
        raise ValueError("capacity must be in [1, num_experts]")

    tokens, top_k = expert_ids.shape
    if last_use is None:
        last_use = torch.full(
            (num_experts,),
            -1,
            dtype=torch.int64,
            device=expert_ids.device,
        )
    elif last_use.shape != (num_experts,):
        raise ValueError("last_use must have shape [num_experts]")

    if tokens == 0:
        return torch.zeros_like(expert_ids, dtype=torch.bool), last_use, clock

    id_order = torch.argsort(expert_ids, dim=-1, stable=True)
    ids_by_id = expert_ids.gather(-1, id_order)
    priorities_by_id = priorities.gather(-1, id_order)
    priority_order = torch.argsort(
        priorities_by_id,
        dim=-1,
        descending=True,
        stable=True,
    )
    ordered_ids = ids_by_id.gather(-1, priority_order).to(torch.long)
    timestamps = torch.arange(
        clock,
        clock + tokens * top_k,
        dtype=torch.int64,
        device=expert_ids.device,
    ).view(tokens, top_k)

    events = torch.full(
        (tokens, num_experts),
        -1,
        dtype=torch.int64,
        device=expert_ids.device,
    )
    events.scatter_(1, ordered_ids, timestamps)
    last_use_inclusive = torch.maximum(
        torch.cummax(events, dim=0).values,
        last_use.unsqueeze(0),
    )
    last_use_before = torch.cat(
        (last_use.unsqueeze(0), last_use_inclusive[:-1]),
        dim=0,
    )

    resident_values, resident_ids = torch.topk(
        last_use_before,
        k=capacity,
        dim=-1,
    )
    membership = torch.zeros_like(last_use_before, dtype=torch.bool)
    membership.scatter_(1, resident_ids, resident_values >= 0)
    hits = membership.gather(1, expert_ids.to(torch.long))
    return hits, last_use_inclusive[-1], clock + tokens * top_k


class CachePriorRouter(BaseRouter):
    """Proof-oriented Cache-Prior wrapper around an existing MoE router.

    A multi-token call is treated as one batch-one, unchunked prefill and resets
    the logical cache. Single-token decode calls continue from that state.
    """

    def __init__(
        self,
        base_router: BaseRouter,
        *,
        capacity: int,
        lambda_value: float,
        top_j: int,
        scoring_func: str,
        renormalize: bool,
        routed_scaling_factor: float,
        e_score_correction_bias: torch.Tensor | None,
        num_expert_group: int | None,
        topk_group: int | None,
        layer_name: str = "",
        metrics_path: str = "",
    ) -> None:
        super().__init__(
            top_k=base_router.top_k,
            global_num_experts=base_router.global_num_experts,
            eplb_state=base_router.eplb_state,
        )
        if scoring_func not in ("softmax", "sigmoid"):
            raise ValueError("Cache-Prior supports softmax and sigmoid routing")
        if capacity < self.top_k or capacity > self.global_num_experts:
            raise ValueError(
                "Cache-Prior capacity must be between top_k and num_experts"
            )
        if lambda_value < 0:
            raise ValueError("Cache-Prior lambda must be non-negative")
        if top_j < 0 or top_j >= self.top_k:
            raise ValueError("Cache-Prior top_j must be in [0, top_k)")

        self.base_router = base_router
        self.capacity = capacity
        self.lambda_value = lambda_value
        self.top_j = top_j
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_expert_group = num_expert_group or 1
        self.topk_group = topk_group or 1
        self.layer_name = layer_name
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.global_num_experts % self.num_expert_group != 0:
            raise ValueError("num_experts must be divisible by num_expert_group")

        self._last_use = torch.full((self.global_num_experts,), -1, dtype=torch.int64)
        self._clock = 0
        self._range_total = 0.0
        self._range_count = 0
        self._accesses = 0
        self._hits = 0
        self._changed_tokens = 0
        self._top_j_violations = 0

    @property
    def routing_method_type(self) -> RoutingMethodType:
        return self.base_router.routing_method_type

    @property
    def metrics(self) -> CachePriorMetrics:
        return CachePriorMetrics(
            accesses=self._accesses,
            hits=self._hits,
            changed_tokens=self._changed_tokens,
            top_j_violations=self._top_j_violations,
        )

    def reset_cache(self) -> None:
        self._last_use.fill_(-1)
        self._clock = 0

    def reset_metrics(self) -> None:
        self._accesses = 0
        self._hits = 0
        self._changed_tokens = 0
        self._top_j_violations = 0

    def reset_range_estimator(self) -> None:
        self._range_total = 0.0
        self._range_count = 0

    def _scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        if self.scoring_func == "softmax":
            return torch.softmax(router_logits, dim=-1)
        return torch.sigmoid(router_logits)

    def _selection_values(self, scores: torch.Tensor) -> torch.Tensor:
        if self.e_score_correction_bias is None:
            return scores
        bias = self.e_score_correction_bias.detach().to(
            device=scores.device,
            dtype=scores.dtype,
        )
        return scores + bias.unsqueeze(0)

    def _select_ids(self, selection_values: torch.Tensor) -> torch.Tensor:
        num_tokens = selection_values.shape[0]
        if self.num_expert_group <= 1 and self.topk_group <= 1:
            return torch.topk(
                selection_values, k=self.top_k, dim=-1, sorted=True
            ).indices

        group_size = self.global_num_experts // self.num_expert_group
        grouped_values = selection_values.view(
            num_tokens, self.num_expert_group, group_size
        )
        if self.e_score_correction_bias is not None:
            group_scores = grouped_values.topk(min(2, group_size), dim=-1).values.sum(
                dim=-1
            )
        else:
            group_scores = grouped_values.max(dim=-1).values
        group_ids = torch.topk(
            group_scores,
            k=self.topk_group,
            dim=-1,
            sorted=True,
        ).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, group_ids, True)
        expert_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.num_expert_group, group_size)
            .reshape(num_tokens, self.global_num_experts)
        )
        masked_values = selection_values.masked_fill(~expert_mask, float("-inf"))
        return torch.topk(masked_values, k=self.top_k, dim=-1, sorted=True).indices

    def _running_means(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(torch.float64)
        prefix = values.cumsum(0) + self._range_total
        divisor = torch.arange(
            self._range_count + 1,
            self._range_count + values.numel() + 1,
            dtype=torch.float64,
            device=values.device,
        )
        means = prefix / divisor
        self._range_total += float(values.sum().item())
        self._range_count += values.numel()
        return means.to(torch.float32)

    def _membership(self) -> torch.Tensor:
        resident_values, resident_ids = torch.topk(self._last_use, k=self.capacity)
        membership = torch.zeros(
            self.global_num_experts,
            dtype=torch.bool,
        )
        membership.scatter_(0, resident_ids, resident_values >= 0)
        return membership

    def _touch(self, expert_ids: torch.Tensor, priorities: torch.Tensor) -> None:
        id_order = torch.argsort(expert_ids, stable=True)
        ids_by_id = expert_ids.gather(0, id_order)
        priorities_by_id = priorities.gather(0, id_order)
        priority_order = torch.argsort(
            priorities_by_id,
            descending=True,
            stable=True,
        )
        ordered_ids = ids_by_id.gather(0, priority_order).to(torch.long)
        timestamps = torch.arange(
            self._clock,
            self._clock + self.top_k,
            dtype=torch.int64,
        )
        self._last_use.scatter_(0, ordered_ids, timestamps)
        self._clock += self.top_k

    def _observe_fixed_trace(
        self,
        expert_ids: torch.Tensor,
        priorities: torch.Tensor,
    ) -> None:
        hits, self._last_use, self._clock = update_lru_state(
            expert_ids,
            priorities,
            capacity=self.capacity,
            num_experts=self.global_num_experts,
            last_use=self._last_use,
            clock=self._clock,
        )
        self._accesses += expert_ids.numel()
        self._hits += int(hits.sum().item())

    def _write_prefill_metrics(
        self,
        before: CachePriorMetrics,
        num_tokens: int,
    ) -> None:
        if self.metrics_path is None or num_tokens <= 1:
            return
        after = self.metrics
        accesses = after.accesses - before.accesses
        hits = after.hits - before.hits
        record = {
            "pid": os.getpid(),
            "layer": self.layer_name,
            "tokens": num_tokens,
            "capacity": self.capacity,
            "lambda": self.lambda_value,
            "top_j": self.top_j,
            "accesses": accesses,
            "hits": hits,
            "misses": accesses - hits,
            "hit_rate": hits / accesses if accesses else 0.0,
            "miss_rate": (accesses - hits) / accesses if accesses else 0.0,
            "changed_tokens": after.changed_tokens - before.changed_tokens,
            "top_j_violations": (after.top_j_violations - before.top_j_violations),
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = router_logits.shape[0]
        metrics_before = self.metrics
        if num_tokens > 1:
            self.reset_cache()

        logits_cpu = router_logits.detach().to(device="cpu", dtype=torch.float32)
        scores_cpu = self._scores(logits_cpu)
        selection_values = self._selection_values(scores_cpu)
        ranges = selection_values.amax(dim=-1) - selection_values.amin(dim=-1)
        range_means = self._running_means(ranges)

        if self.lambda_value == 0:
            topk_weights, topk_ids = self.base_router._compute_routing(
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
            self._observe_fixed_trace(
                topk_ids.detach().to(device="cpu", dtype=torch.long),
                topk_weights.detach().to(device="cpu", dtype=torch.float32),
            )
            self._write_prefill_metrics(metrics_before, num_tokens)
            return topk_weights, topk_ids

        selected_ids_rows: list[torch.Tensor] = []
        selected_weights_rows: list[torch.Tensor] = []
        hit_count = 0
        changed_tokens = 0
        top_j_violations = 0

        original_ids = self._select_ids(selection_values)
        for row in range(num_tokens):
            membership = self._membership()
            protected = membership.clone()
            if self.top_j:
                protected[original_ids[row, : self.top_j]] = True
            reranked_values = selection_values[row] + (
                self.lambda_value * range_means[row] * protected
            )
            selected_ids = self._select_ids(reranked_values.unsqueeze(0))[0]
            selected_weights = scores_cpu[row].gather(0, selected_ids)
            if self.renormalize:
                selected_weights = selected_weights / selected_weights.sum().clamp_min(
                    torch.finfo(selected_weights.dtype).tiny
                )
            selected_weights = selected_weights * self.routed_scaling_factor

            hit_count += int(membership.gather(0, selected_ids).sum().item())
            changed_tokens += int(
                not torch.equal(
                    selected_ids.sort().values,
                    original_ids[row].sort().values,
                )
            )
            if self.top_j:
                retained = torch.isin(
                    original_ids[row, : self.top_j], selected_ids
                ).all()
                top_j_violations += int(not bool(retained.item()))
            self._touch(selected_ids, selected_weights)
            selected_ids_rows.append(selected_ids)
            selected_weights_rows.append(selected_weights)

        self._accesses += num_tokens * self.top_k
        self._hits += hit_count
        self._changed_tokens += changed_tokens
        self._top_j_violations += top_j_violations
        self._write_prefill_metrics(metrics_before, num_tokens)

        output_ids_dtype = torch.int32 if indices_type is None else indices_type
        return (
            torch.stack(selected_weights_rows).to(
                device=router_logits.device,
                dtype=torch.float32,
            ),
            torch.stack(selected_ids_rows).to(
                device=router_logits.device,
                dtype=output_ids_dtype,
            ),
        )
