# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from vllm.forward_context import get_forward_context, is_forward_context_available
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


CACHE_PRIOR_BATCH_METADATA_KEY = "cache_prior_batch_metadata"


@dataclass(frozen=True)
class CachePriorBatchMetadata:
    """Compatibility metadata emitted by the earlier DFW model runner."""

    request_ids: tuple[str, ...]
    num_scheduled_tokens: tuple[int, ...]
    num_computed_tokens: tuple[int, ...]
    num_prompt_tokens: tuple[int, ...]
    num_draft_tokens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        request_count = len(self.request_ids)
        sized_values = (
            self.num_scheduled_tokens,
            self.num_computed_tokens,
            self.num_prompt_tokens,
        )
        if self.num_draft_tokens:
            sized_values += (self.num_draft_tokens,)
        if not all(len(values) == request_count for values in sized_values):
            raise ValueError("Cache-Prior batch metadata lengths must match")
        if any(value < 0 for value in self.num_scheduled_tokens):
            raise ValueError("scheduled token counts must be non-negative")


class SpeculativeCacheOverflow(RuntimeError):
    """The experts required by one verification block do not fit in cache."""

    def __init__(
        self,
        required_experts: int,
        capacity: int,
        cache_misses: int,
    ) -> None:
        super().__init__(
            "speculative block requires "
            f"{required_experts} distinct experts, but cache capacity is {capacity}"
        )
        self.required_experts = required_experts
        self.capacity = capacity
        self.cache_misses = cache_misses


@dataclass(frozen=True)
class SpeculativeCacheMetrics:
    blocks: int
    cache_misses: int
    required_experts: int
    tokens: int
    committed_tokens: int
    requests: int
    fallback_blocks: int
    fallback_cache_misses: int
    fallback_required_experts: int
    fallback_tokens: int
    prefill_blocks: int
    prefill_overflows: int
    prefill_required_experts: int
    prefill_cache_misses: int
    residency_observations: int
    residency_token_steps_sum: int
    residency_block_steps_sum: int
    residency_token_histogram: tuple[tuple[int, int], ...]
    residency_block_histogram: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SpeculativeCacheTransaction:
    """Logical cache state spanning speculative routing and token acceptance."""

    selected_ids: torch.Tensor
    priorities: torch.Tensor
    pre_last_use: torch.Tensor
    pre_clock: int
    resident_before: torch.Tensor
    required_experts: torch.Tensor
    newly_loaded: torch.Tensor
    admitted_last_use: torch.Tensor
    admitted_clock: int
    resident_after_admission: torch.Tensor

    @property
    def num_tokens(self) -> int:
        return self.selected_ids.shape[0]

    @property
    def required_count(self) -> int:
        return int(self.required_experts.sum().item())

    @property
    def newly_loaded_count(self) -> int:
        return int(self.newly_loaded.sum().item())

    @property
    def cache_misses(self) -> int:
        """Number of distinct expert loads required for this block."""
        return self.newly_loaded_count


@dataclass
class SpeculativeRequestState:
    """Logical expert-cache state owned by one vLLM request."""

    last_use: torch.Tensor
    clock: int
    residency_loaded_at_tokens: torch.Tensor
    residency_loaded_at_blocks: torch.Tensor
    residency_token_clock: int
    residency_block_clock: int
    transaction: SpeculativeCacheTransaction | None = None
    range_total: float = 0.0
    range_count: int = 0


def speculative_commit_mask(
    num_draft_tokens: list[int] | tuple[int, ...],
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Map rejection-sampler output to committed target-input tokens.

    Each target verification block contains one already committed input token
    followed by its draft tokens. Rejection-sampler output contains every
    accepted draft followed by one bonus or replacement token, and pads the
    remaining positions with ``-1``.
    """
    if sampled_token_ids.ndim != 2:
        raise ValueError("sampled_token_ids must have shape [requests, tokens]")
    if sampled_token_ids.shape[0] != len(num_draft_tokens):
        raise ValueError("one draft-token count is required for every request")

    valid_counts = (sampled_token_ids != -1).sum(dim=1).to(device="cpu")
    masks: list[torch.Tensor] = []
    for draft_count, valid_count_tensor in zip(num_draft_tokens, valid_counts):
        draft_count = int(draft_count)
        valid_count = int(valid_count_tensor.item())
        if draft_count < 0:
            raise ValueError("draft-token counts must be non-negative")
        if valid_count < 1 or valid_count > draft_count + 1:
            raise ValueError(
                "valid sampled-token count must be in [1, num_draft_tokens + 1]"
            )
        accepted_drafts = valid_count - 1
        mask = torch.zeros(draft_count + 1, dtype=torch.bool)
        mask[: accepted_drafts + 1] = True
        masks.append(mask)
    return torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool)


def _resident_mask(last_use: torch.Tensor, capacity: int) -> torch.Tensor:
    resident_values, resident_ids = torch.topk(last_use, k=capacity)
    membership = torch.zeros_like(last_use, dtype=torch.bool)
    membership.scatter_(0, resident_ids, resident_values >= 0)
    return membership


def begin_speculative_lru_block(
    expert_ids: torch.Tensor,
    priorities: torch.Tensor,
    *,
    capacity: int,
    num_experts: int,
    last_use: torch.Tensor | None = None,
    clock: int = 0,
) -> SpeculativeCacheTransaction:
    """Admit the complete expert union needed by one verification block.

    Routing for the block must already have been computed against one frozen
    pre-block cache snapshot. The returned admitted state contains every expert
    needed to execute the block. If that union does not fit, no cache state is
    returned and the caller must use a different execution strategy.
    """
    if expert_ids.ndim != 2 or expert_ids.shape[0] == 0:
        raise ValueError("expert_ids must have non-empty shape [tokens, top_k]")
    if priorities.shape != expert_ids.shape:
        raise ValueError("priorities must have the same shape as expert_ids")
    if capacity <= 0 or capacity > num_experts:
        raise ValueError("capacity must be in [1, num_experts]")

    selected_ids = expert_ids.detach().to(dtype=torch.long)
    selected_priorities = priorities.detach().to(device=selected_ids.device)
    if not bool(((selected_ids >= 0) & (selected_ids < num_experts)).all().item()):
        raise ValueError("speculative expert_ids must all be valid expert indices")

    if last_use is None:
        pre_last_use = torch.full(
            (num_experts,),
            -1,
            dtype=torch.int64,
            device=selected_ids.device,
        )
    else:
        if last_use.shape != (num_experts,):
            raise ValueError("last_use must have shape [num_experts]")
        pre_last_use = (
            last_use.detach()
            .to(
                device=selected_ids.device,
                dtype=torch.int64,
            )
            .clone()
        )
    if clock < 0:
        raise ValueError("clock must be non-negative")
    if bool((pre_last_use >= clock).any().item()):
        raise ValueError("clock must be greater than every last-use timestamp")

    resident_before = _resident_mask(pre_last_use, capacity)
    required_experts = torch.zeros(
        num_experts,
        dtype=torch.bool,
        device=selected_ids.device,
    )
    required_experts.scatter_(0, selected_ids.reshape(-1), True)
    required_count = int(required_experts.sum().item())
    newly_loaded = required_experts & ~resident_before
    cache_misses = int(newly_loaded.sum().item())
    if required_count > capacity:
        raise SpeculativeCacheOverflow(required_count, capacity, cache_misses)

    _, admitted_last_use, admitted_clock = update_lru_state(
        selected_ids,
        selected_priorities,
        capacity=capacity,
        num_experts=num_experts,
        last_use=pre_last_use,
        clock=clock,
    )
    resident_after_admission = _resident_mask(admitted_last_use, capacity)
    if bool((required_experts & ~resident_after_admission).any().item()):
        raise RuntimeError("speculative admission failed to retain a required expert")

    return SpeculativeCacheTransaction(
        selected_ids=selected_ids.clone(),
        priorities=selected_priorities.clone(),
        pre_last_use=pre_last_use,
        pre_clock=clock,
        resident_before=resident_before,
        required_experts=required_experts,
        newly_loaded=newly_loaded,
        admitted_last_use=admitted_last_use,
        admitted_clock=admitted_clock,
        resident_after_admission=resident_after_admission,
    )


def commit_speculative_lru_block(
    transaction: SpeculativeCacheTransaction,
    committed_tokens: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Finalize LRU order after the target accepts a prefix or token subset.

    Committed accesses become most-recent in their original order. Existing
    residents touched only by rejected tokens regain their pre-block order.
    Experts loaded only for rejected tokens stay resident but move to the LRU
    end, older than every retained pre-block resident.
    """
    if committed_tokens.ndim != 1:
        raise ValueError("committed_tokens must have shape [tokens]")
    if committed_tokens.shape[0] != transaction.num_tokens:
        raise ValueError("committed_tokens must match the speculative block length")

    device = transaction.selected_ids.device
    commit_mask = committed_tokens.detach().to(device=device, dtype=torch.bool)
    num_experts = transaction.pre_last_use.numel()
    committed_last_use = torch.full(
        (num_experts,),
        -1,
        dtype=torch.int64,
        device=device,
    )
    committed_accesses = 0
    if bool(commit_mask.any().item()):
        committed_ids = transaction.selected_ids[commit_mask]
        committed_priorities = transaction.priorities[commit_mask]
        committed_accesses = committed_ids.numel()
        _, committed_last_use, _ = update_lru_state(
            committed_ids,
            committed_priorities,
            capacity=num_experts,
            num_experts=num_experts,
        )

    committed_experts = committed_last_use >= 0
    admitted = transaction.resident_after_admission
    if bool((committed_experts & ~admitted).any().item()):
        raise RuntimeError("a committed expert is missing from the admitted cache")

    rejected_loads = admitted & transaction.newly_loaded & ~committed_experts
    carried_residents = admitted & transaction.resident_before & ~committed_experts

    def ordered_ids(mask: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        ids = torch.nonzero(mask, as_tuple=False).flatten()
        if ids.numel() == 0:
            return ids
        return ids[torch.argsort(order[ids], stable=True)]

    # The concatenation is the final queue from least to most recently used.
    final_order = torch.cat(
        (
            ordered_ids(rejected_loads, transaction.admitted_last_use),
            ordered_ids(carried_residents, transaction.pre_last_use),
            ordered_ids(committed_experts, committed_last_use),
        )
    )
    resident_count = int(admitted.sum().item())
    if final_order.numel() != resident_count:
        raise RuntimeError("speculative commit did not rank every admitted expert")

    new_clock = max(
        transaction.pre_clock + committed_accesses,
        resident_count,
    )
    final_last_use = torch.full_like(transaction.pre_last_use, -1)
    final_last_use.scatter_(
        0,
        final_order,
        torch.arange(
            new_clock - resident_count,
            new_clock,
            dtype=torch.int64,
            device=device,
        ),
    )
    return final_last_use, new_clock


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
        cache_bias_mode: str = "logit",
        top_j: int,
        scoring_func: str,
        renormalize: bool,
        routed_scaling_factor: float,
        e_score_correction_bias: torch.Tensor | None,
        num_expert_group: int | None,
        topk_group: int | None,
        decode_only: bool = False,
        layer_name: str = "",
        metrics_path: str = "",
        trace_dir: str = "",
        reset_path: str = "",
        score_stats_path: str = "",
        speculative_only: bool = False,
        speculative_max_tokens: int = 0,
        write_speculative_events: bool = False,
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
        if cache_bias_mode not in ("logit", "selection"):
            raise ValueError("Cache-Prior bias mode must be logit or selection")
        if top_j < 0 or top_j >= self.top_k:
            raise ValueError("Cache-Prior top_j must be in [0, top_k)")

        self.base_router = base_router
        self.capacity = capacity
        self.lambda_value = lambda_value
        self.cache_bias_mode = cache_bias_mode
        self.top_j = top_j
        self.scoring_func = scoring_func
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.e_score_correction_bias = e_score_correction_bias
        self.num_expert_group = num_expert_group or 1
        self.topk_group = topk_group or 1
        self.decode_only = decode_only
        self.layer_name = layer_name
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.reset_path = Path(reset_path) if reset_path else None
        self.score_stats_path = Path(score_stats_path) if score_stats_path else None
        self.speculative_only = speculative_only
        self.speculative_max_tokens = speculative_max_tokens
        self.write_speculative_events = write_speculative_events
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
        self._residency_loaded_at_tokens = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int64
        )
        self._residency_loaded_at_blocks = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int64
        )
        self._residency_token_clock = 0
        self._residency_block_clock = 0
        self._residency_observations = 0
        self._residency_token_steps_sum = 0
        self._residency_block_steps_sum = 0
        self._residency_token_histogram: Counter[int] = Counter()
        self._residency_block_histogram: Counter[int] = Counter()
        self._speculative_transaction: SpeculativeCacheTransaction | None = None
        self._speculative_blocks = 0
        self._speculative_cache_misses = 0
        self._speculative_required_experts = 0
        self._speculative_tokens = 0
        self._speculative_committed_tokens = 0
        self._speculative_requests = 0
        self._speculative_fallback_blocks = 0
        self._speculative_fallback_cache_misses = 0
        self._speculative_fallback_required_experts = 0
        self._speculative_fallback_tokens = 0
        self._speculative_prefill_blocks = 0
        self._speculative_prefill_overflows = 0
        self._speculative_prefill_required_experts = 0
        self._speculative_prefill_cache_misses = 0
        self._last_speculative_cache_misses = 0
        self._range_total = 0.0
        self._range_count = 0
        self._accesses = 0
        self._hits = 0
        self._changed_tokens = 0
        self._top_j_violations = 0
        self._evaluation_batch_size = 0
        self._evaluation_sequence_length = 0
        self._speculative_request_states: dict[str, SpeculativeRequestState] = {}
        self._active_speculative_request_id: str | None = None

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

    @property
    def speculative_metrics(self) -> SpeculativeCacheMetrics:
        (
            residency_observations,
            residency_token_steps_sum,
            residency_block_steps_sum,
            residency_token_histogram,
            residency_block_histogram,
        ) = self._residency_snapshot()
        return SpeculativeCacheMetrics(
            blocks=self._speculative_blocks,
            cache_misses=self._speculative_cache_misses,
            required_experts=self._speculative_required_experts,
            tokens=self._speculative_tokens,
            committed_tokens=self._speculative_committed_tokens,
            requests=self._speculative_requests,
            fallback_blocks=self._speculative_fallback_blocks,
            fallback_cache_misses=self._speculative_fallback_cache_misses,
            fallback_required_experts=self._speculative_fallback_required_experts,
            fallback_tokens=self._speculative_fallback_tokens,
            prefill_blocks=self._speculative_prefill_blocks,
            prefill_overflows=self._speculative_prefill_overflows,
            prefill_required_experts=self._speculative_prefill_required_experts,
            prefill_cache_misses=self._speculative_prefill_cache_misses,
            residency_observations=residency_observations,
            residency_token_steps_sum=residency_token_steps_sum,
            residency_block_steps_sum=residency_block_steps_sum,
            residency_token_histogram=tuple(sorted(residency_token_histogram.items())),
            residency_block_histogram=tuple(sorted(residency_block_histogram.items())),
        )

    @property
    def last_speculative_cache_misses(self) -> int:
        return self._last_speculative_cache_misses

    @property
    def has_pending_speculative_transactions(self) -> bool:
        if self._speculative_transaction is not None:
            return True
        return any(
            state.transaction is not None
            for state in self._speculative_request_states.values()
        )

    def _empty_request_state(self) -> SpeculativeRequestState:
        return SpeculativeRequestState(
            last_use=torch.full(
                (self.global_num_experts,), -1, dtype=torch.int64
            ),
            clock=0,
            residency_loaded_at_tokens=torch.full(
                (self.global_num_experts,), -1, dtype=torch.int64
            ),
            residency_loaded_at_blocks=torch.full(
                (self.global_num_experts,), -1, dtype=torch.int64
            ),
            residency_token_clock=0,
            residency_block_clock=0,
        )

    def _activate_request_state(self, request_id: str, *, reset: bool = False) -> None:
        if self._active_speculative_request_id is not None:
            raise RuntimeError("another Cache-Prior request state is already active")
        state = self._speculative_request_states.pop(request_id, None)
        if state is None:
            state = self._empty_request_state()
        self._last_use = state.last_use
        self._clock = state.clock
        self._residency_loaded_at_tokens = state.residency_loaded_at_tokens
        self._residency_loaded_at_blocks = state.residency_loaded_at_blocks
        self._residency_token_clock = state.residency_token_clock
        self._residency_block_clock = state.residency_block_clock
        self._speculative_transaction = state.transaction
        self._range_total = state.range_total
        self._range_count = state.range_count
        self._active_speculative_request_id = request_id
        if reset:
            self._reset_active_request_state()

    def _store_request_state(self) -> None:
        request_id = self._active_speculative_request_id
        if request_id is None:
            raise RuntimeError("there is no active Cache-Prior request state")
        self._speculative_request_states[request_id] = SpeculativeRequestState(
            last_use=self._last_use,
            clock=self._clock,
            residency_loaded_at_tokens=self._residency_loaded_at_tokens,
            residency_loaded_at_blocks=self._residency_loaded_at_blocks,
            residency_token_clock=self._residency_token_clock,
            residency_block_clock=self._residency_block_clock,
            transaction=self._speculative_transaction,
            range_total=self._range_total,
            range_count=self._range_count,
        )
        self._active_speculative_request_id = None
        # Pending transactions live in their request state between target
        # routing and rejection sampling.  Do not expose the last request's
        # transaction through the legacy scalar slot.
        self._speculative_transaction = None

    def _reset_active_request_state(self) -> None:
        self._close_residencies(
            self._residency_loaded_at_tokens >= 0,
            token_clock=self._residency_token_clock,
            block_clock=self._residency_block_clock,
        )
        empty = self._empty_request_state()
        self._last_use = empty.last_use
        self._clock = empty.clock
        self._residency_loaded_at_tokens = empty.residency_loaded_at_tokens
        self._residency_loaded_at_blocks = empty.residency_loaded_at_blocks
        self._residency_token_clock = 0
        self._residency_block_clock = 0
        self._speculative_transaction = None
        self._range_total = 0.0
        self._range_count = 0

    def reset_cache(self, *, discard_residencies: bool = False) -> None:
        if self._active_speculative_request_id is not None:
            self._store_request_state()
        had_request_states = bool(self._speculative_request_states)
        if not discard_residencies:
            for request_id in tuple(self._speculative_request_states):
                self._activate_request_state(request_id)
                self._close_residencies(
                    self._residency_loaded_at_tokens >= 0,
                    token_clock=self._residency_token_clock,
                    block_clock=self._residency_block_clock,
                )
                self._active_speculative_request_id = None
        self._speculative_request_states.clear()
        if not discard_residencies and not had_request_states:
            self._close_residencies(
                self._residency_loaded_at_tokens >= 0,
                token_clock=self._residency_token_clock,
                block_clock=self._residency_block_clock,
            )
        # Router state may have been captured by PyTorch inference mode during
        # a prior generation. Replacing it avoids an illegal in-place write
        # outside that mode. Lambda reconfiguration discards prior-point
        # lifetimes because reset_metrics follows immediately; request resets
        # first right-censor the previous request's active expert lifetimes.
        self._last_use = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int64
        )
        self._clock = 0
        self._residency_loaded_at_tokens = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int64
        )
        self._residency_loaded_at_blocks = torch.full(
            (self.global_num_experts,), -1, dtype=torch.int64
        )
        self._residency_token_clock = 0
        self._residency_block_clock = 0
        self._speculative_transaction = None
        self._active_speculative_request_id = None

    def reset_metrics(self) -> None:
        self._accesses = 0
        self._hits = 0
        self._changed_tokens = 0
        self._top_j_violations = 0
        self._speculative_blocks = 0
        self._speculative_cache_misses = 0
        self._speculative_required_experts = 0
        self._speculative_tokens = 0
        self._speculative_committed_tokens = 0
        self._speculative_requests = 0
        self._speculative_fallback_blocks = 0
        self._speculative_fallback_cache_misses = 0
        self._speculative_fallback_required_experts = 0
        self._speculative_fallback_tokens = 0
        self._speculative_prefill_blocks = 0
        self._speculative_prefill_overflows = 0
        self._speculative_prefill_required_experts = 0
        self._speculative_prefill_cache_misses = 0
        self._last_speculative_cache_misses = 0
        self._residency_observations = 0
        self._residency_token_steps_sum = 0
        self._residency_block_steps_sum = 0
        self._residency_token_histogram.clear()
        self._residency_block_histogram.clear()
        active = self._residency_loaded_at_tokens >= 0
        self._residency_loaded_at_tokens[active] = self._residency_token_clock
        self._residency_loaded_at_blocks[active] = self._residency_block_clock

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

    def clear_evaluation_batch_layout(self) -> None:
        """Disable the fixed-window teacher-forcing layout."""
        self._evaluation_batch_size = 0
        self._evaluation_sequence_length = 0

    def _observe_speculative_prefill(
        self,
        expert_ids: torch.Tensor,
        priorities: torch.Tensor,
    ) -> None:
        """Warm logical LRU state while recording whole-prefill feasibility."""
        required_ids = torch.unique(expert_ids)
        required_experts = int(required_ids.numel())
        cache_misses = int((~self._membership()[required_ids]).sum().item())
        self._speculative_prefill_blocks += 1
        self._speculative_prefill_required_experts += required_experts
        self._speculative_prefill_cache_misses += cache_misses
        self._speculative_prefill_overflows += int(required_experts > self.capacity)
        self._observe_fixed_trace(expert_ids, priorities)

    def _observe_speculative_fallback(
        self,
        expert_ids: torch.Tensor,
        priorities: torch.Tensor,
    ) -> None:
        """Count unbiased target work when no speculative draft is available."""
        hit_mask = self._observe_fixed_trace(
            expert_ids,
            priorities,
            track_decode_residency=True,
        )
        self._speculative_fallback_blocks += 1
        self._speculative_fallback_cache_misses += int((~hit_mask).sum().item())
        self._speculative_fallback_required_experts += expert_ids.numel()
        self._speculative_fallback_tokens += expert_ids.shape[0]

    def _compute_speculative_with_batch_metadata(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        metadata: CachePriorBatchMetadata,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use DFW scheduler metadata to exclude padded routing rows."""
        topk_weights, topk_ids = self.base_router._compute_routing(
            hidden_states,
            router_logits,
            indices_type,
            input_ids=input_ids,
        )
        real_tokens = sum(metadata.num_scheduled_tokens)
        if real_tokens > router_logits.shape[0]:
            raise RuntimeError(
                "Cache-Prior request metadata contains more tokens than router logits"
            )
        if real_tokens == 0:
            return topk_weights, topk_ids

        prefill_counts = tuple(
            min(scheduled, max(prompt_tokens - computed, 0))
            for scheduled, computed, prompt_tokens in zip(
                metadata.num_scheduled_tokens,
                metadata.num_computed_tokens,
                metadata.num_prompt_tokens,
                strict=True,
            )
        )
        original_ids = topk_ids[:real_tokens].detach().to(
            device="cpu", dtype=torch.long
        )
        original_weights = topk_weights[:real_tokens].detach().to(
            device="cpu", dtype=torch.float32
        )
        output_weights = topk_weights.clone()
        output_ids = topk_ids.clone()
        draft_counts = (
            metadata.num_draft_tokens
            if metadata.num_draft_tokens
            else tuple(max(scheduled - 1, 0) for scheduled in metadata.num_scheduled_tokens)
        )
        row = 0
        for request_id, scheduled, computed, prefill_count, draft_count in zip(
            metadata.request_ids,
            metadata.num_scheduled_tokens,
            metadata.num_computed_tokens,
            prefill_counts,
            draft_counts,
            strict=True,
        ):
            request_slice = slice(row, row + scheduled)
            self._activate_request_state(
                request_id,
                reset=bool(scheduled and computed == 0),
            )
            try:
                if prefill_count:
                    # Prefill is fixed and outside CacheMoE. It neither
                    # reroutes experts nor warms the logical decode cache.
                    decode_count = scheduled - prefill_count
                    if decode_count:
                        decode_slice = slice(row + prefill_count, row + scheduled)
                        self._observe_speculative_fallback(
                            original_ids[decode_slice],
                            original_weights[decode_slice],
                        )
                elif draft_count > 0:
                    selected_weights, selected_ids = self.begin_speculative_routing(
                        hidden_states[request_slice],
                        router_logits[request_slice],
                        indices_type,
                        input_ids=(
                            input_ids[request_slice] if input_ids is not None else None
                        ),
                        precomputed_weights=topk_weights[request_slice],
                        precomputed_ids=topk_ids[request_slice],
                    )
                    output_weights[request_slice] = selected_weights
                    output_ids[request_slice] = selected_ids
                elif scheduled:
                    self._observe_speculative_fallback(
                        original_ids[request_slice],
                        original_weights[request_slice],
                    )
            finally:
                self._store_request_state()
            row += scheduled
        return output_weights, output_ids

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
        if self._residency_loaded_at_tokens.device.type != "cpu":
            self._residency_loaded_at_tokens = self._residency_loaded_at_tokens.cpu()
        if self._residency_loaded_at_blocks.device.type != "cpu":
            self._residency_loaded_at_blocks = self._residency_loaded_at_blocks.cpu()

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

    def _reranked_values(
        self,
        logits: torch.Tensor,
        selection_values: torch.Tensor,
        protected: torch.Tensor,
        range_mean: torch.Tensor,
    ) -> torch.Tensor:
        if self.cache_bias_mode == "selection":
            return selection_values + self.lambda_value * protected
        reranked_logits = logits + self.lambda_value * range_mean * protected
        return self._selection_values(self._scores(reranked_logits))

    def _select_ids(
        self,
        selection_values: torch.Tensor,
        protected_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        if protected_ids is not None:
            if protected_ids.ndim != 2 or protected_ids.shape[0] != num_tokens:
                raise ValueError("protected_ids must have shape [tokens, protected]")
            if bool(
                ((protected_ids < 0) | (protected_ids >= self.global_num_experts))
                .any()
                .item()
            ):
                raise ValueError("protected_ids contains an invalid expert id")
            protected_group_ids = torch.div(
                protected_ids,
                group_size,
                rounding_mode="floor",
            ).to(torch.long)
            protected_group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            protected_group_mask.scatter_(1, protected_group_ids, True)
            if bool((protected_group_mask.sum(dim=-1) > self.topk_group).any().item()):
                raise ValueError("protected experts span more than topk_group groups")
            group_scores = group_scores.masked_fill(
                protected_group_mask,
                float("inf"),
            )
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
        if protected_ids is not None:
            masked_values.scatter_(1, protected_ids, float("inf"))
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

    def _close_residencies(
        self,
        expert_mask: torch.Tensor,
        *,
        token_clock: int,
        block_clock: int,
    ) -> None:
        """Close decode-origin expert lifetimes selected by ``expert_mask``."""
        active = expert_mask.to(device="cpu", dtype=torch.bool) & (
            self._residency_loaded_at_tokens >= 0
        )
        if not bool(active.any().item()):
            return
        token_durations = token_clock - self._residency_loaded_at_tokens[active]
        block_durations = block_clock - self._residency_loaded_at_blocks[active]
        if bool((token_durations < 0).any().item()) or bool(
            (block_durations < 0).any().item()
        ):
            raise RuntimeError("cache residency clock moved backwards")
        token_values, token_counts = torch.unique(token_durations, return_counts=True)
        block_values, block_counts = torch.unique(block_durations, return_counts=True)
        self._residency_token_histogram.update(
            {
                int(value.item()): int(count.item())
                for value, count in zip(token_values, token_counts, strict=True)
            }
        )
        self._residency_block_histogram.update(
            {
                int(value.item()): int(count.item())
                for value, count in zip(block_values, block_counts, strict=True)
            }
        )
        observations = int(active.sum().item())
        self._residency_observations += observations
        self._residency_token_steps_sum += int(token_durations.sum().item())
        self._residency_block_steps_sum += int(block_durations.sum().item())
        self._residency_loaded_at_tokens[active] = -1
        self._residency_loaded_at_blocks[active] = -1

    def _observe_decode_membership_transition(
        self,
        resident_before: torch.Tensor,
        resident_after: torch.Tensor,
    ) -> None:
        """Record decode loads and evictions without changing LRU state."""
        resident_before = resident_before.to(device="cpu", dtype=torch.bool)
        resident_after = resident_after.to(device="cpu", dtype=torch.bool)
        if resident_before.shape != (self.global_num_experts,) or (
            resident_after.shape != resident_before.shape
        ):
            raise ValueError("resident masks must have shape [num_experts]")
        evicted = resident_before & ~resident_after
        loaded = resident_after & ~resident_before
        self._close_residencies(
            evicted,
            token_clock=self._residency_token_clock,
            block_clock=self._residency_block_clock,
        )
        if bool((loaded & (self._residency_loaded_at_tokens >= 0)).any().item()):
            raise RuntimeError("an already tracked expert was loaded again")
        self._residency_loaded_at_tokens[loaded] = self._residency_token_clock
        self._residency_loaded_at_blocks[loaded] = self._residency_block_clock

    def _residency_snapshot(
        self,
    ) -> tuple[int, int, int, Counter[int], Counter[int]]:
        """Return closed plus right-censored-at-run-end residency statistics."""
        observations = self._residency_observations
        token_steps_sum = self._residency_token_steps_sum
        block_steps_sum = self._residency_block_steps_sum
        token_histogram = self._residency_token_histogram.copy()
        block_histogram = self._residency_block_histogram.copy()
        states = list(self._speculative_request_states.values())
        if self._active_speculative_request_id is not None:
            states.append(
                SpeculativeRequestState(
                    self._last_use,
                    self._clock,
                    self._residency_loaded_at_tokens,
                    self._residency_loaded_at_blocks,
                    self._residency_token_clock,
                    self._residency_block_clock,
                    self._speculative_transaction,
                    self._range_total,
                    self._range_count,
                )
            )
        elif not states:
            # Preserve the legacy scalar-state path used by direct router
            # tests and non-metadata callers.
            states.append(
                SpeculativeRequestState(
                    self._last_use,
                    self._clock,
                    self._residency_loaded_at_tokens,
                    self._residency_loaded_at_blocks,
                    self._residency_token_clock,
                    self._residency_block_clock,
                    self._speculative_transaction,
                    self._range_total,
                    self._range_count,
                )
            )
        for state in states:
            active = state.residency_loaded_at_tokens >= 0
            if not bool(active.any().item()):
                continue
            token_durations = (
                state.residency_token_clock
                - state.residency_loaded_at_tokens[active]
            )
            block_durations = (
                state.residency_block_clock
                - state.residency_loaded_at_blocks[active]
            )
            token_values, token_counts = torch.unique(
                token_durations, return_counts=True
            )
            block_values, block_counts = torch.unique(
                block_durations, return_counts=True
            )
            token_histogram.update(
                {
                    int(value.item()): int(count.item())
                    for value, count in zip(token_values, token_counts, strict=True)
                }
            )
            block_histogram.update(
                {
                    int(value.item()): int(count.item())
                    for value, count in zip(block_values, block_counts, strict=True)
                }
            )
            observations += int(active.sum().item())
            token_steps_sum += int(token_durations.sum().item())
            block_steps_sum += int(block_durations.sum().item())
        return (
            observations,
            token_steps_sum,
            block_steps_sum,
            token_histogram,
            block_histogram,
        )

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
        *,
        track_decode_residency: bool = False,
    ) -> torch.Tensor:
        resident_before = self._membership() if track_decode_residency else None
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
        if track_decode_residency:
            if expert_ids.shape[0] != 1:
                raise RuntimeError(
                    "exact decode residency tracking requires max_num_seqs=1"
                )
            assert resident_before is not None
            self._observe_decode_membership_transition(
                resident_before,
                self._membership(),
            )
            self._residency_token_clock += expert_ids.shape[0]
            self._residency_block_clock += 1
        return hits

    def begin_speculative_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
        precomputed_weights: torch.Tensor | None = None,
        precomputed_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route and admit one verification block against frozen membership."""
        if self._speculative_transaction is not None:
            raise RuntimeError("the previous speculative block is still pending")
        if router_logits.ndim != 2 or router_logits.shape[0] == 0:
            raise ValueError(
                "router_logits must have non-empty shape [tokens, experts]"
            )

        self._ensure_cpu_state()
        self._maybe_reset_range_estimator()
        range_total_before = self._range_total
        range_count_before = self._range_count

        try:
            if self.lambda_value == 0 and precomputed_weights is not None:
                if precomputed_ids is None:
                    raise ValueError("precomputed IDs are required with weights")
                selected_weights_device = precomputed_weights
                selected_ids_device = precomputed_ids
            elif self.lambda_value == 0:
                selected_weights_device, selected_ids_device = (
                    self.base_router._compute_routing(
                        hidden_states,
                        router_logits,
                        indices_type,
                        input_ids=input_ids,
                    )
                )
            else:
                routing_device = (
                    router_logits.device
                    if self.cache_bias_mode == "selection"
                    else torch.device("cpu")
                )
                logits = router_logits.detach().to(
                    device=routing_device,
                    dtype=torch.float32,
                )
                scores = self._scores(logits)
                selection_values = self._selection_values(scores)
                original_ids = self._select_ids(selection_values)
                frozen_membership = self._membership().to(routing_device)
                protected = (
                    frozen_membership.unsqueeze(0)
                    .expand(
                        router_logits.shape[0],
                        -1,
                    )
                    .clone()
                )
                required_experts = None
                if self.top_j:
                    required_experts = torch.zeros_like(
                        selection_values,
                        dtype=torch.bool,
                    )
                    required_experts.scatter_(
                        1,
                        original_ids[:, : self.top_j],
                        True,
                    )
                    protected |= required_experts
                if self.cache_bias_mode == "selection":
                    reranked_values = selection_values + self.lambda_value * protected
                else:
                    ranges = logits.amax(dim=-1) - logits.amin(dim=-1)
                    range_means = self._running_means(ranges)
                    reranked_values = self._reranked_values(
                        logits,
                        selection_values,
                        protected,
                        range_means.unsqueeze(1),
                    )
                selected_ids_device = self._select_ids(
                    reranked_values,
                    protected_ids=original_ids[:, : self.top_j]
                    if self.top_j
                    else None,
                )
                selected_weights_device = scores.gather(1, selected_ids_device)
                if self.renormalize:
                    selected_weights_device = (
                        selected_weights_device
                        / selected_weights_device.sum(
                            dim=-1,
                            keepdim=True,
                        ).clamp_min(torch.finfo(selected_weights_device.dtype).tiny)
                    )
                selected_weights_device = (
                    selected_weights_device * self.routed_scaling_factor
                )

            selected_ids = selected_ids_device.detach().to(
                device="cpu",
                dtype=torch.long,
            )
            selected_weights = selected_weights_device.detach().to(
                device="cpu",
                dtype=torch.float32,
            )

            transaction = begin_speculative_lru_block(
                selected_ids,
                selected_weights,
                capacity=self.capacity,
                num_experts=self.global_num_experts,
                last_use=self._last_use,
                clock=self._clock,
            )
        except Exception:
            self._range_total = range_total_before
            self._range_count = range_count_before
            raise

        self._speculative_transaction = transaction
        self._last_use = transaction.admitted_last_use
        self._clock = transaction.admitted_clock
        self._observe_decode_membership_transition(
            transaction.resident_before,
            transaction.resident_after_admission,
        )
        self._speculative_blocks += 1
        self._speculative_cache_misses += transaction.cache_misses
        self._speculative_required_experts += transaction.required_count
        self._speculative_tokens += transaction.num_tokens
        self._last_speculative_cache_misses = transaction.cache_misses
        self._write_speculative_cache_metrics(transaction)
        output_ids_dtype = torch.int32 if indices_type is None else indices_type
        return (
            selected_weights_device.to(
                device=router_logits.device,
                dtype=torch.float32,
            ),
            selected_ids_device.to(
                device=router_logits.device,
                dtype=output_ids_dtype,
            ),
        )

    def commit_speculative_routing(
        self,
        committed_tokens: torch.Tensor,
        *,
        num_requests: int = 1,
    ) -> None:
        """Apply the target acceptance mask to the pending cache transaction."""
        transaction = self._speculative_transaction
        if transaction is None:
            raise RuntimeError("there is no pending speculative block")
        if num_requests <= 0 or num_requests > transaction.num_tokens:
            raise ValueError("num_requests must fit within the speculative block")
        last_use, clock = commit_speculative_lru_block(
            transaction,
            committed_tokens,
        )
        resident_before_commit = self._membership()
        self._last_use = last_use
        self._clock = clock
        self._observe_decode_membership_transition(
            resident_before_commit,
            self._membership(),
        )
        committed_count = int(committed_tokens.sum().item())
        self._residency_token_clock += committed_count
        self._residency_block_clock += 1
        self._speculative_committed_tokens += committed_count
        self._speculative_requests += num_requests
        self._speculative_transaction = None

    def commit_speculative_batch(
        self,
        committed_tokens: torch.Tensor,
        *,
        request_ids: tuple[str, ...],
        num_draft_tokens: tuple[int, ...],
    ) -> None:
        """Commit independent pending transactions from one fused sampler batch."""
        if len(request_ids) != len(num_draft_tokens):
            raise ValueError("request IDs and draft counts must match")
        if not self._speculative_request_states:
            # Backward-compatible scalar state for direct one-request callers
            # that bypass CachePriorBatchMetadata.
            if len(request_ids) != 1:
                raise RuntimeError(
                    "multiple speculative requests require request-indexed state"
                )
            self.commit_speculative_routing(
                committed_tokens,
                num_requests=1,
            )
            return
        offset = 0
        for request_id, draft_count in zip(
            request_ids, num_draft_tokens, strict=True
        ):
            block_tokens = int(draft_count) + 1
            block_mask = committed_tokens[offset : offset + block_tokens]
            if block_mask.numel() != block_tokens:
                raise ValueError("committed token mask is shorter than the batch")
            if draft_count > 0:
                self._activate_request_state(request_id)
                try:
                    self.commit_speculative_routing(block_mask, num_requests=1)
                finally:
                    self._store_request_state()
            offset += block_tokens
        if offset != committed_tokens.numel():
            raise ValueError("committed token mask is longer than the batch")

    def reject_speculative_routing(self) -> None:
        """Reject every token while retaining verification-only expert loads."""
        transaction = self._speculative_transaction
        if transaction is None:
            raise RuntimeError("there is no pending speculative block")
        self.commit_speculative_routing(
            torch.zeros(transaction.num_tokens, dtype=torch.bool)
        )

    def _write_speculative_cache_metrics(
        self,
        transaction: SpeculativeCacheTransaction,
    ) -> None:
        if self.metrics_path is None or not self.write_speculative_events:
            return
        record = {
            "pid": os.getpid(),
            "layer": self.layer_name,
            "event": "speculative_cache_block",
            "tokens": transaction.num_tokens,
            "capacity": self.capacity,
            "required_experts": transaction.required_count,
            "cache_misses": transaction.cache_misses,
            "miss_unit": "distinct_expert_load",
        }
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")

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
            "cache_bias_mode": self.cache_bias_mode,
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
            reranked_values = self._reranked_values(
                logits_batch[:, position],
                selection_values.view(
                    batch_size, sequence_length, self.global_num_experts
                )[:, position],
                protected,
                range_means_batch[:, position, None],
            )
            selected_ids = self._select_ids(
                reranked_values,
                protected_ids=original_ids[:, position, : self.top_j]
                if self.top_j
                else None,
            )
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
        speculative_tokens = None
        batch_metadata = None
        if is_forward_context_available():
            additional_kwargs = get_forward_context().additional_kwargs
            speculative_tokens = additional_kwargs.get("cache_prior_speculative_tokens")
            batch_metadata = additional_kwargs.get(CACHE_PRIOR_BATCH_METADATA_KEY)
        if self.speculative_only and batch_metadata is not None:
            if not isinstance(batch_metadata, CachePriorBatchMetadata):
                raise TypeError("invalid Cache-Prior batch metadata")
            return self._compute_speculative_with_batch_metadata(
                hidden_states,
                router_logits,
                indices_type,
                batch_metadata,
                input_ids=input_ids,
            )
        heuristic_speculative_tokens = (
            self.speculative_only
            and self.speculative_max_tokens > 1
            and 1 < router_logits.shape[0] <= self.speculative_max_tokens
        )
        if speculative_tokens is not None or heuristic_speculative_tokens:
            if (
                speculative_tokens is not None
                and int(speculative_tokens) != router_logits.shape[0]
            ):
                raise RuntimeError(
                    "Cache-Prior speculative routing requires an unpadded "
                    "target verification batch"
                )
            return self.begin_speculative_routing(
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
        if self._speculative_transaction is not None:
            raise RuntimeError(
                "commit or reject the pending speculative block before routing again"
            )
        num_tokens = router_logits.shape[0]
        if self.speculative_only:
            self._ensure_cpu_state()
            if num_tokens > 1:
                self.reset_cache()
            topk_weights, topk_ids = self.base_router._compute_routing(
                hidden_states,
                router_logits,
                indices_type,
                input_ids=input_ids,
            )
            original_ids = topk_ids.detach().to(device="cpu", dtype=torch.long)
            original_weights = topk_weights.detach().to(
                device="cpu", dtype=torch.float32
            )
            if num_tokens > 1:
                self._observe_speculative_prefill(original_ids, original_weights)
            else:
                self._observe_speculative_fallback(original_ids, original_weights)
            return topk_weights, topk_ids
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
            reranked_values = self._reranked_values(
                logits_cpu[row : row + 1],
                selection_values[row : row + 1],
                protected.unsqueeze(0),
                range_means[row],
            )
            selected_ids = self._select_ids(
                reranked_values,
                protected_ids=original_ids[row : row + 1, : self.top_j]
                if self.top_j
                else None,
            )[0]
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
