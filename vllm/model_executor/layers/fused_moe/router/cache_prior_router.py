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
    valid_ordered_ids = (ordered_ids >= 0) & (ordered_ids < num_experts)
    scatter_ids = torch.where(valid_ordered_ids, ordered_ids, num_experts)
    timestamps = torch.arange(
        clock,
        clock + tokens * top_k,
        dtype=torch.int64,
        device=expert_ids.device,
    ).view(tokens, top_k)

    events = torch.full(
        (tokens, num_experts + 1),
        -1,
        dtype=torch.int64,
        device=expert_ids.device,
    )
    events.scatter_(1, scatter_ids, timestamps)
    events = events[:, :num_experts]
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
    valid_expert_ids = (expert_ids >= 0) & (expert_ids < num_experts)
    gather_ids = expert_ids.to(torch.long).clamp(0, num_experts - 1)
    hits = membership.gather(1, gather_ids) & valid_expert_ids
    return hits, last_use_inclusive[-1], clock + tokens * top_k


def update_lru_state_batched(
    expert_ids: torch.Tensor,
    priorities: torch.Tensor,
    *,
    capacity: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Compute independent fixed-trace LRU states for a prompt batch."""
    if expert_ids.ndim != 3:
        raise ValueError("expert_ids must have shape [batch, tokens, top_k]")
    if priorities.shape != expert_ids.shape:
        raise ValueError("priorities must have the same shape as expert_ids")
    if capacity <= 0 or capacity > num_experts:
        raise ValueError("capacity must be in [1, num_experts]")

    batch_size, tokens, top_k = expert_ids.shape
    if tokens == 0:
        return (
            torch.zeros_like(expert_ids, dtype=torch.bool),
            torch.full(
                (batch_size, num_experts),
                -1,
                dtype=torch.int64,
                device=expert_ids.device,
            ),
            0,
        )

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
    valid_ordered_ids = (ordered_ids >= 0) & (ordered_ids < num_experts)
    scatter_ids = torch.where(valid_ordered_ids, ordered_ids, num_experts)
    timestamps = torch.arange(
        tokens * top_k,
        dtype=torch.int64,
        device=expert_ids.device,
    ).view(1, tokens, top_k)

    events = torch.full(
        (batch_size, tokens, num_experts + 1),
        -1,
        dtype=torch.int64,
        device=expert_ids.device,
    )
    events.scatter_(2, scatter_ids, timestamps.expand(batch_size, -1, -1))
    events = events[:, :, :num_experts]
    last_use_inclusive = torch.cummax(events, dim=1).values
    initial = torch.full(
        (batch_size, 1, num_experts),
        -1,
        dtype=torch.int64,
        device=expert_ids.device,
    )
    last_use_before = torch.cat((initial, last_use_inclusive[:, :-1]), dim=1)

    resident_values, resident_ids = torch.topk(
        last_use_before,
        k=capacity,
        dim=-1,
    )
    membership = torch.zeros_like(last_use_before, dtype=torch.bool)
    membership.scatter_(2, resident_ids, resident_values >= 0)
    valid_expert_ids = (expert_ids >= 0) & (expert_ids < num_experts)
    gather_ids = expert_ids.to(torch.long).clamp(0, num_experts - 1)
    hits = membership.gather(2, gather_ids) & valid_expert_ids
    return hits, last_use_inclusive[:, -1], tokens * top_k


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
        trace_dir: str = "",
        reset_path: str = "",
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
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.reset_path = Path(reset_path) if reset_path else None
        self._range_reset_applied = False
        layer_slug = "".join(
            character if character.isalnum() else "_" for character in layer_name
        )
        self._trace_stem = f"{layer_slug or 'layer'}.pid{os.getpid()}"
        self._trace_index = 0
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
        self._evaluation_batch_size = 0
        self._evaluation_sequence_length = 0

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

    def set_evaluation_batch_layout(
        self,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if batch_size <= 0 or sequence_length <= 1:
            raise ValueError("evaluation batch dimensions must be positive")
        self._evaluation_batch_size = batch_size
        self._evaluation_sequence_length = sequence_length

    def _maybe_reset_range_estimator(self) -> None:
        if (
            self._range_reset_applied
            or self.reset_path is None
            or not self.reset_path.exists()
        ):
            return
        self.reset_range_estimator()
        self._range_reset_applied = True

    def _ensure_cpu_state(self) -> None:
        # vLLM's model loader moves tensor-valued module attributes to CUDA,
        # including this proof-only logical state. Cache-Prior reranking is
        # intentionally evaluated on CPU, so keep its state colocated with the
        # detached routing scores.
        if self._last_use.device.type != "cpu":
            self._last_use = self._last_use.cpu()

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

    def _batched_membership(self, last_use: torch.Tensor) -> torch.Tensor:
        resident_values, resident_ids = torch.topk(
            last_use,
            k=self.capacity,
            dim=-1,
        )
        membership = torch.zeros_like(last_use, dtype=torch.bool)
        membership.scatter_(1, resident_ids, resident_values >= 0)
        return membership

    def _batched_touch(
        self,
        last_use: torch.Tensor,
        expert_ids: torch.Tensor,
        priorities: torch.Tensor,
        clock: int,
    ) -> int:
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
            clock + self.top_k,
            dtype=torch.int64,
        ).expand(expert_ids.shape[0], -1)
        last_use.scatter_(1, ordered_ids, timestamps)
        return clock + self.top_k

    def _observe_fixed_trace(
        self,
        expert_ids: torch.Tensor,
        priorities: torch.Tensor,
    ) -> torch.Tensor:
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
        return hits

    def _write_prefill_trace(
        self,
        *,
        original_ids: torch.Tensor,
        selected_ids: torch.Tensor,
        selected_weights: torch.Tensor,
        hit_mask: torch.Tensor,
        logit_range: torch.Tensor,
        range_mean: torch.Tensor,
        range_count_start: int,
        sequences: int = 1,
        tokens_per_sequence: int | None = None,
    ) -> None:
        if self.trace_dir is None or original_ids.shape[0] <= 1:
            return
        fields = {
            "original_ids.i16": original_ids.to(torch.int16),
            "selected_ids.i16": selected_ids.to(torch.int16),
            "selected_weights.f32": selected_weights.to(torch.float32),
            "hit_mask.u8": hit_mask.to(torch.uint8),
            "logit_range.f32": logit_range.to(torch.float32),
            "range_mean.f32": range_mean.to(torch.float32),
        }
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        for suffix, tensor in fields.items():
            path = self.trace_dir / f"{self._trace_stem}.{suffix}"
            array = tensor.detach().cpu().contiguous().numpy()
            with path.open("ab") as output:
                output.write(array.tobytes())

        record = {
            "pid": os.getpid(),
            "layer": self.layer_name,
            "record": self._trace_index,
            "tokens": original_ids.shape[0],
            "top_k": original_ids.shape[1],
            "num_experts": self.global_num_experts,
            "range_count_start": range_count_start,
            "range_count_end": self._range_count,
            "sequences": sequences,
            "tokens_per_sequence": tokens_per_sequence or original_ids.shape[0],
        }
        metadata_path = self.trace_dir / f"{self._trace_stem}.jsonl"
        with metadata_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
        self._trace_index += 1

    def _write_prefill_metrics(
        self,
        before: CachePriorMetrics,
        num_tokens: int,
        sequences: int = 1,
        tokens_per_sequence: int | None = None,
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
            "sequences": sequences,
            "tokens_per_sequence": tokens_per_sequence or num_tokens,
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_evaluation_call_shape(self, num_tokens: int) -> None:
        if self.metrics_path is None:
            return
        path = self.metrics_path.with_name(
            f"{self.metrics_path.stem}-calls{self.metrics_path.suffix}"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "pid": os.getpid(),
            "layer": self.layer_name,
            "num_tokens": num_tokens,
            "configured_batch_size": self._evaluation_batch_size,
            "configured_sequence_length": self._evaluation_sequence_length,
        }
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    def _compute_batched_prefill(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = self._evaluation_sequence_length
        num_tokens = router_logits.shape[0]
        batch_size, remainder = divmod(num_tokens, sequence_length)
        if not batch_size or remainder:
            raise RuntimeError("Cache-Prior received an unexpected packed batch shape")

        metrics_before = self.metrics
        logits_cpu = router_logits.detach().to(device="cpu", dtype=torch.float32)
        scores_cpu = self._scores(logits_cpu)
        selection_values = self._selection_values(scores_cpu)
        ranges = logits_cpu.amax(dim=-1) - logits_cpu.amin(dim=-1)
        range_count_start = self._range_count
        range_means = self._running_means(ranges)

        batch_shape = (batch_size, sequence_length, self.top_k)
        if self.lambda_value == 0:
            topk_weights, topk_ids = self.base_router._compute_routing(
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
            original_ids = topk_ids.detach().to(device="cpu", dtype=torch.long)
            selected_weights = topk_weights.detach().to(
                device="cpu", dtype=torch.float32
            )
            hit_mask, _, _ = update_lru_state_batched(
                original_ids.view(batch_shape),
                selected_weights.view(batch_shape),
                capacity=self.capacity,
                num_experts=self.global_num_experts,
            )
            self._accesses += original_ids.numel()
            self._hits += int(hit_mask.sum().item())
            self._write_prefill_trace(
                original_ids=original_ids,
                selected_ids=original_ids,
                selected_weights=selected_weights,
                hit_mask=hit_mask.view(num_tokens, self.top_k),
                logit_range=ranges,
                range_mean=range_means,
                range_count_start=range_count_start,
                sequences=batch_size,
                tokens_per_sequence=sequence_length,
            )
            self._write_prefill_metrics(
                metrics_before,
                num_tokens,
                sequences=batch_size,
                tokens_per_sequence=sequence_length,
            )
            return topk_weights, topk_ids

        logits_batch = logits_cpu.view(
            batch_size, sequence_length, self.global_num_experts
        )
        scores_batch = scores_cpu.view(
            batch_size, sequence_length, self.global_num_experts
        )
        range_means_batch = range_means.view(batch_size, sequence_length)
        original_ids = self._select_ids(selection_values).view(batch_shape)
        last_use = torch.full(
            (batch_size, self.global_num_experts),
            -1,
            dtype=torch.int64,
        )
        clock = 0
        selected_ids_steps: list[torch.Tensor] = []
        selected_weights_steps: list[torch.Tensor] = []
        hit_mask_steps: list[torch.Tensor] = []
        hit_count = 0
        changed_tokens = 0
        top_j_violations = 0

        for position in range(sequence_length):
            membership = self._batched_membership(last_use)
            protected = membership.clone()
            if self.top_j:
                protected.scatter_(
                    1,
                    original_ids[:, position, : self.top_j],
                    True,
                )
            reranked_logits = logits_batch[:, position] + (
                self.lambda_value * range_means_batch[:, position, None] * protected
            )
            reranked_scores = self._scores(reranked_logits)
            reranked_values = self._selection_values(reranked_scores)
            selected_ids = self._select_ids(reranked_values)
            selected_weights = scores_batch[:, position].gather(1, selected_ids)
            if self.renormalize:
                selected_weights = selected_weights / selected_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(selected_weights.dtype).tiny)
            selected_weights = selected_weights * self.routed_scaling_factor

            hit_mask = membership.gather(1, selected_ids)
            hit_count += int(hit_mask.sum().item())
            changed_tokens += int(
                (
                    ~(
                        selected_ids.sort(dim=-1).values
                        == original_ids[:, position].sort(dim=-1).values
                    ).all(dim=-1)
                )
                .sum()
                .item()
            )
            if self.top_j:
                retained = (
                    (
                        original_ids[:, position, : self.top_j, None]
                        == selected_ids[:, None, :]
                    )
                    .any(dim=-1)
                    .all(dim=-1)
                )
                top_j_violations += int((~retained).sum().item())
            clock = self._batched_touch(
                last_use,
                selected_ids,
                selected_weights,
                clock,
            )
            selected_ids_steps.append(selected_ids)
            selected_weights_steps.append(selected_weights)
            hit_mask_steps.append(hit_mask)

        selected_ids_tensor = torch.stack(selected_ids_steps, dim=1)
        selected_weights_tensor = torch.stack(selected_weights_steps, dim=1)
        hit_mask_tensor = torch.stack(hit_mask_steps, dim=1)
        self._accesses += num_tokens * self.top_k
        self._hits += hit_count
        self._changed_tokens += changed_tokens
        self._top_j_violations += top_j_violations
        self._write_prefill_trace(
            original_ids=original_ids.view(num_tokens, self.top_k),
            selected_ids=selected_ids_tensor.view(num_tokens, self.top_k),
            selected_weights=selected_weights_tensor.view(num_tokens, self.top_k),
            hit_mask=hit_mask_tensor.view(num_tokens, self.top_k),
            logit_range=ranges,
            range_mean=range_means,
            range_count_start=range_count_start,
            sequences=batch_size,
            tokens_per_sequence=sequence_length,
        )
        self._write_prefill_metrics(
            metrics_before,
            num_tokens,
            sequences=batch_size,
            tokens_per_sequence=sequence_length,
        )

        output_ids_dtype = torch.int32 if indices_type is None else indices_type
        return (
            selected_weights_tensor.view(num_tokens, self.top_k).to(
                device=router_logits.device,
                dtype=torch.float32,
            ),
            selected_ids_tensor.view(num_tokens, self.top_k).to(
                device=router_logits.device,
                dtype=output_ids_dtype,
            ),
        )

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = router_logits.shape[0]
        if self._evaluation_batch_size:
            self._write_evaluation_call_shape(num_tokens)
            # vLLM may split a queued prompt set into several scheduler steps
            # according to available KV-cache concurrency. Every evaluation
            # request has the same full sequence length, so a positive exact
            # multiple is a packed prefill chunk. Each row gets an independent
            # logical expert cache; the shorter generation call is excluded.
            batch_size, remainder = divmod(num_tokens, self._evaluation_sequence_length)
            if batch_size and remainder == 0:
                return self._compute_batched_prefill(
                    hidden_states,
                    router_logits,
                    indices_type,
                    input_ids=input_ids,
                )
            # The one-token generation requested only to obtain prompt logprobs
            # is outside the evaluated prefill. Keep it out of cache metrics.
            return self.base_router._compute_routing(
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
        metrics_before = self.metrics
        self._ensure_cpu_state()
        if num_tokens > 1:
            self.reset_cache()
        self._maybe_reset_range_estimator()

        logits_cpu = router_logits.detach().to(device="cpu", dtype=torch.float32)
        scores_cpu = self._scores(logits_cpu)
        selection_values = self._selection_values(scores_cpu)
        ranges = logits_cpu.amax(dim=-1) - logits_cpu.amin(dim=-1)
        range_count_start = self._range_count
        range_means = self._running_means(ranges)

        if self.lambda_value == 0:
            topk_weights, topk_ids = self.base_router._compute_routing(
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
            original_ids = topk_ids.detach().to(device="cpu", dtype=torch.long)
            selected_weights = topk_weights.detach().to(
                device="cpu", dtype=torch.float32
            )
            hit_mask = self._observe_fixed_trace(original_ids, selected_weights)
            self._write_prefill_trace(
                original_ids=original_ids,
                selected_ids=original_ids,
                selected_weights=selected_weights,
                hit_mask=hit_mask,
                logit_range=ranges,
                range_mean=range_means,
                range_count_start=range_count_start,
            )
            self._write_prefill_metrics(metrics_before, num_tokens)
            return topk_weights, topk_ids

        selected_ids_rows: list[torch.Tensor] = []
        selected_weights_rows: list[torch.Tensor] = []
        hit_mask_rows: list[torch.Tensor] = []
        hit_count = 0
        changed_tokens = 0
        top_j_violations = 0

        original_ids = self._select_ids(selection_values)
        for row in range(num_tokens):
            membership = self._membership()
            protected = membership.clone()
            if self.top_j:
                protected[original_ids[row, : self.top_j]] = True
            reranked_logits = logits_cpu[row] + (
                self.lambda_value * range_means[row] * protected
            )
            reranked_scores = self._scores(reranked_logits.unsqueeze(0))
            reranked_values = self._selection_values(reranked_scores)
            selected_ids = self._select_ids(reranked_values)[0]
            selected_weights = scores_cpu[row].gather(0, selected_ids)
            if self.renormalize:
                selected_weights = selected_weights / selected_weights.sum().clamp_min(
                    torch.finfo(selected_weights.dtype).tiny
                )
            selected_weights = selected_weights * self.routed_scaling_factor

            hit_mask = membership.gather(0, selected_ids)
            hit_count += int(hit_mask.sum().item())
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
            hit_mask_rows.append(hit_mask)

        self._accesses += num_tokens * self.top_k
        self._hits += hit_count
        self._changed_tokens += changed_tokens
        self._top_j_violations += top_j_violations
        selected_ids_tensor = torch.stack(selected_ids_rows)
        selected_weights_tensor = torch.stack(selected_weights_rows)
        self._write_prefill_trace(
            original_ids=original_ids,
            selected_ids=selected_ids_tensor,
            selected_weights=selected_weights_tensor,
            hit_mask=torch.stack(hit_mask_rows),
            logit_range=ranges,
            range_mean=range_means,
            range_count_start=range_count_start,
        )
        self._write_prefill_metrics(metrics_before, num_tokens)

        output_ids_dtype = torch.int32 if indices_type is None else indices_type
        return (
            selected_weights_tensor.to(
                device=router_logits.device,
                dtype=torch.float32,
            ),
            selected_ids_tensor.to(
                device=router_logits.device,
                dtype=output_ids_dtype,
            ),
        )
