# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
the Triton, FlashInfer, or CPU backend based on the configured
`MambaBackendEnum`. On CPU-only platforms (PowerPC, x86 without CUDA)
the backend defaults to 'cpu'.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


@triton.jit(do_not_specialize=["num_reqs"])
def _preprocess_replayssm_kernel(
    idx_mapping,
    query_metadata,
    num_computed_tokens,
    is_prefilling,
    src_cols,
    dst_cols,
    block_table,
    tracker_start,
    tracker_committed,
    src_slots,
    dst_slots,
    plan_ring_start,
    plan_flush_count,
    active_request_indices,
    block_table_stride_req: tl.int64,
    slot_table_stride_layer: tl.int64,
    num_reqs,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    QUERY_METADATA_IS_CUMULATIVE: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr,
    MATERIALIZE_PREFIXES: tl.constexpr,
    MAX_NUM_REQS: tl.constexpr,
) -> None:
    """Reset prefill state and prepare an optional writable-slot move."""
    batch_idx = tl.program_id(0)
    active = batch_idx < num_reqs
    req_idx = batch_idx
    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping + batch_idx, mask=active, other=-1)
    valid_req = active & (req_idx >= 0)

    src_col = tl.load(src_cols + req_idx, mask=valid_req, other=-1)
    dst_col = tl.load(dst_cols + req_idx, mask=valid_req, other=-1)
    changed = valid_req & (src_col >= 0) & (src_col != dst_col)
    src_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + src_col,
        mask=changed,
        other=PAD_SLOT_ID,
    )
    dst_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + dst_col,
        mask=changed,
        other=PAD_SLOT_ID,
    )
    for layer_idx in tl.static_range(0, NUM_LAYERS):
        slot_offset = layer_idx * slot_table_stride_layer + batch_idx
        tl.store(src_slots + slot_offset, src_slot)
        tl.store(dst_slots + slot_offset, dst_slot)

    # Always clear the fixed-capacity plan row before deciding whether this
    # request has work. FlashInfer treats flush_count < 0 as a no-op.
    tl.store(plan_ring_start + batch_idx, 0)
    tl.store(plan_flush_count + batch_idx, -1)
    if changed:
        # BlockManager guarantees that distinct live columns map to allocated,
        # distinct physical slots. Snapshot the old owner before clearing the
        # destination tracker.
        tl.store(plan_ring_start + batch_idx, tl.load(tracker_start + src_slot))
        tl.store(
            plan_flush_count + batch_idx,
            tl.load(tracker_committed + src_slot),
        )
        tl.store(tracker_start + dst_slot, 0)
        tl.store(tracker_committed + dst_slot, 0)

    prefilling = tl.load(is_prefilling + batch_idx, mask=active, other=0)
    if valid_req & prefilling:
        if QUERY_METADATA_IS_CUMULATIVE:
            query_len = tl.load(query_metadata + batch_idx + 1) - tl.load(
                query_metadata + batch_idx
            )
        else:
            query_len = tl.load(query_metadata + batch_idx)
        computed = tl.load(num_computed_tokens + req_idx)
        first_col = tl.maximum(computed // MAMBA_BLOCK_SIZE, 0)
        last_col = tl.maximum(
            (computed + query_len + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1,
            0,
        )
        for col in tl.range(first_col, last_col + 1):
            slot = tl.load(block_table + batch_idx * block_table_stride_req + col)
            tl.store(tracker_start + slot, 0)
            tl.store(tracker_committed + slot, 0)

    if MATERIALIZE_PREFIXES & (batch_idx == 0):
        active_count = 0
        for candidate_idx in tl.range(0, num_reqs):
            candidate_req_idx = candidate_idx
            if HAS_IDX_MAPPING:
                candidate_req_idx = tl.load(idx_mapping + candidate_idx)
            valid_candidate = candidate_req_idx >= 0
            src_col = tl.load(
                src_cols + candidate_req_idx,
                mask=valid_candidate,
                other=-1,
            )
            dst_col = tl.load(
                dst_cols + candidate_req_idx,
                mask=valid_candidate,
                other=-1,
            )
            if valid_candidate & (src_col >= 0) & (src_col != dst_col):
                tl.store(active_request_indices + active_count, candidate_idx)
                active_count += 1
        if active_count < MAX_NUM_REQS:
            tl.store(active_request_indices + active_count, -1)


@triton.jit(do_not_specialize=["num_reqs"])
def _postprocess_replayssm_kernel(
    idx_mapping,
    query_metadata,
    num_accepted_tokens,
    is_prefilling,
    live_cols,
    materialize_src_cols,
    materialize_dst_cols,
    materialize_token_counts,
    block_table,
    tracker_start,
    tracker_committed,
    src_slots,
    dst_slots,
    plan_ring_start,
    plan_flush_count,
    active_request_indices,
    block_table_stride_req: tl.int64,
    slot_table_stride_layer: tl.int64,
    num_reqs,
    LOGICAL_WINDOW: tl.constexpr,
    RING_BUFFER_LEN: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
    QUERY_METADATA_IS_CUMULATIVE: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr,
    MATERIALIZE_PREFIXES: tl.constexpr,
    MAX_NUM_REQS: tl.constexpr,
) -> None:
    """Commit a completed step and prepare an optional prefix snapshot."""
    batch_idx = tl.program_id(0)
    active = batch_idx < num_reqs
    req_idx = batch_idx
    if HAS_IDX_MAPPING:
        req_idx = tl.load(idx_mapping + batch_idx, mask=active, other=-1)
    valid_req = active & (req_idx >= 0)

    src_col = tl.load(materialize_src_cols + batch_idx, mask=valid_req, other=-1)
    materialize = valid_req & (src_col >= 0)
    dst_col = tl.load(materialize_dst_cols + batch_idx, mask=materialize, other=-1)
    src_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + src_col,
        mask=materialize,
        other=PAD_SLOT_ID,
    )
    dst_slot = tl.load(
        block_table + batch_idx * block_table_stride_req + dst_col,
        mask=materialize,
        other=PAD_SLOT_ID,
    )
    for layer_idx in tl.static_range(0, NUM_LAYERS):
        slot_offset = layer_idx * slot_table_stride_layer + batch_idx
        tl.store(src_slots + slot_offset, src_slot)
        tl.store(dst_slots + slot_offset, dst_slot)

    tl.store(plan_ring_start + batch_idx, 0)
    tl.store(plan_flush_count + batch_idx, -1)

    if valid_req:
        # The live column and any materialization destination are allocated by
        # BlockManager. A bad mapping is an upstream lifecycle bug, not a
        # recoverable per-request condition for this kernel to hide.
        live_col = tl.load(live_cols + req_idx)
        live_slot = tl.load(block_table + batch_idx * block_table_stride_req + live_col)
        prefilling = tl.load(is_prefilling + batch_idx)
        if prefilling:
            if materialize:
                # Prefill produced canonical state, so publish an exact copy.
                tl.store(plan_flush_count + batch_idx, 0)
        else:
            if QUERY_METADATA_IS_CUMULATIVE:
                query_len = tl.load(query_metadata + batch_idx + 1) - tl.load(
                    query_metadata + batch_idx
                )
            else:
                query_len = tl.load(query_metadata + batch_idx)
            accepted = tl.maximum(tl.load(num_accepted_tokens + req_idx), 1)
            old_start = tl.load(tracker_start + live_slot)
            old_committed = tl.load(tracker_committed + live_slot)
            checkpointed = old_committed + query_len > LOGICAL_WINDOW
            next_start = tl.where(
                checkpointed,
                (old_start + old_committed) % RING_BUFFER_LEN,
                old_start,
            )
            next_committed = tl.where(checkpointed, accepted, old_committed + accepted)
            tl.store(tracker_start + live_slot, next_start)
            tl.store(tracker_committed + live_slot, next_committed)

            if materialize:
                boundary_count = tl.load(materialize_token_counts + batch_idx)
                tl.store(plan_ring_start + batch_idx, next_start)
                tl.store(
                    plan_flush_count + batch_idx,
                    next_committed - (accepted - boundary_count),
                )

        if materialize:
            # src_slot == dst_slot is a valid in-place checkpoint.
            tl.store(tracker_start + dst_slot, 0)
            tl.store(tracker_committed + dst_slot, 0)

    if MATERIALIZE_PREFIXES & (batch_idx == 0):
        active_count = 0
        for candidate_idx in tl.range(0, num_reqs):
            candidate_req_idx = candidate_idx
            if HAS_IDX_MAPPING:
                candidate_req_idx = tl.load(idx_mapping + candidate_idx)
            src_col = tl.load(
                materialize_src_cols + candidate_idx,
                mask=candidate_req_idx >= 0,
                other=-1,
            )
            if (candidate_req_idx >= 0) & (src_col >= 0):
                tl.store(active_request_indices + active_count, candidate_idx)
                active_count += 1
        if active_count < MAX_NUM_REQS:
            tl.store(active_request_indices + active_count, -1)


def _replayssm_specialization_key(mixer: Any) -> tuple[Any, ...]:
    ssm = mixer.kv_cache[1]
    x_cache = mixer.kv_cache[2]
    b_cache = mixer.kv_cache[4]
    return (
        ssm.dtype,
        x_cache.dtype,
        mixer.A.dtype,
        ssm.size(1),
        ssm.size(2),
        ssm.size(3),
        ssm.size(1) // b_cache.size(1),
        int(mixer.replayssm_buffer_len),
        x_cache.size(2),
        bool(mixer.mamba_config.enable_stochastic_rounding),
        int(mixer.mamba_config.stochastic_rounding_philox_rounds or 0),
    )


@dataclass
class _ReplaySSMGroupContext:
    """ReplaySSM state sharing one physical cache-slot namespace."""

    mixers: list[Any]
    block_table: torch.Tensor
    ring_start: torch.Tensor
    num_committed: torch.Tensor
    materialize_tables: tuple[torch.Tensor, ...]
    src_slots: torch.Tensor
    dst_slots: torch.Tensor
    plan_ring_start: torch.Tensor
    plan_flush_count: torch.Tensor
    active_request_indices: torch.Tensor
    max_num_reqs: int
    logical_window: int
    ring_buffer_len: int
    materialize_prefixes: bool

    @classmethod
    def create(
        cls,
        mixers: list[Any],
        block_table: torch.Tensor,
        cache_mode: str,
        max_num_reqs: int,
    ) -> "_ReplaySSMGroupContext":
        if (
            block_table.ndim != 2
            or block_table.dtype != torch.int32
            or not block_table.is_cuda
            or block_table.numel() == 0
        ):
            raise ValueError("ReplaySSM requires a non-empty 2D CUDA int32 block table")
        _validate_replayssm_cache(mixers)
        first = mixers[0]
        first_ssm = first.kv_cache[1]
        first_x = first.kv_cache[2]
        compatibility = _replayssm_specialization_key(first)
        for mixer in mixers[1:]:
            current = _replayssm_specialization_key(mixer)
            if current != compatibility:
                raise ValueError(
                    "Layers in one ReplaySSM cache group require identical "
                    "materialization specialization; got "
                    f"{compatibility} and {current}"
                )
            if (
                mixer._replayssm_ring_start.data_ptr()
                != first._replayssm_ring_start.data_ptr()
                or mixer._replayssm_prev_num_accepted.data_ptr()
                != first._replayssm_prev_num_accepted.data_ptr()
            ):
                raise ValueError(
                    "Layers in one ReplaySSM cache group must share ring trackers"
                )

        device = first_ssm.device
        zero_table = torch.zeros(len(mixers), dtype=torch.int64, device=device)
        return cls(
            mixers=mixers,
            block_table=block_table,
            ring_start=first._replayssm_ring_start,
            num_committed=first._replayssm_prev_num_accepted,
            materialize_tables=(
                _cuda_i64_ptrs([m.kv_cache[1] for m in mixers]),
                _cuda_i64_slot_strides([m.kv_cache[1] for m in mixers]),
                _cuda_i64_ptrs([m.kv_cache[2] for m in mixers]),
                _cuda_i64_slot_strides([m.kv_cache[2] for m in mixers]),
                _cuda_i64_ptrs([m.kv_cache[4] for m in mixers]),
                _cuda_i64_slot_strides([m.kv_cache[4] for m in mixers]),
                _cuda_i64_ptrs([m.kv_cache[3] for m in mixers]),
                _cuda_i64_slot_strides([m.kv_cache[3] for m in mixers]),
                _cuda_i64_ptrs([m.A for m in mixers]),
                zero_table,
                zero_table.clone(),
            ),
            src_slots=torch.full(
                (len(mixers), max_num_reqs),
                NULL_BLOCK_ID,
                dtype=torch.int32,
                device=device,
            ),
            dst_slots=torch.full(
                (len(mixers), max_num_reqs),
                NULL_BLOCK_ID,
                dtype=torch.int32,
                device=device,
            ),
            plan_ring_start=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
            plan_flush_count=torch.full(
                (max_num_reqs,), -1, dtype=torch.int32, device=device
            ),
            active_request_indices=torch.full(
                (max_num_reqs,), -1, dtype=torch.int32, device=device
            ),
            max_num_reqs=max_num_reqs,
            logical_window=int(first.replayssm_buffer_len),
            ring_buffer_len=first_x.size(2),
            materialize_prefixes=cache_mode in ("align", "all"),
        )

    def preprocess(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        query_metadata: torch.Tensor,
        query_metadata_is_cumulative: bool,
        num_computed_tokens: torch.Tensor,
        is_prefilling: torch.Tensor,
        src_cols: torch.Tensor,
        dst_cols: torch.Tensor,
        mamba_block_size: int,
        num_reqs: int,
    ) -> None:
        """Reset prefill state and prepare an optional writable-slot move."""
        if num_reqs == 0:
            return
        _preprocess_replayssm_kernel[(self.max_num_reqs,)](
            idx_mapping,
            query_metadata,
            num_computed_tokens,
            is_prefilling,
            src_cols,
            dst_cols,
            self.block_table,
            self.ring_start,
            self.num_committed,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.active_request_indices,
            self.block_table.stride(0),
            self.src_slots.stride(0),
            num_reqs,
            MAMBA_BLOCK_SIZE=mamba_block_size,
            NUM_LAYERS=len(self.mixers),
            PAD_SLOT_ID=NULL_BLOCK_ID,
            QUERY_METADATA_IS_CUMULATIVE=query_metadata_is_cumulative,
            HAS_IDX_MAPPING=idx_mapping is not None,
            MATERIALIZE_PREFIXES=self.materialize_prefixes,
            MAX_NUM_REQS=self.max_num_reqs,
        )

    def postprocess(
        self,
        *,
        idx_mapping: torch.Tensor | None,
        query_metadata: torch.Tensor,
        query_metadata_is_cumulative: bool,
        num_accepted_tokens: torch.Tensor,
        is_prefilling: torch.Tensor,
        live_cols: torch.Tensor,
        materialize_src_cols: torch.Tensor,
        materialize_dst_cols: torch.Tensor,
        materialize_token_counts: torch.Tensor,
        num_reqs: int,
    ) -> None:
        """Commit a completed step and prepare an optional prefix snapshot."""
        if num_reqs == 0:
            return
        _postprocess_replayssm_kernel[(self.max_num_reqs,)](
            idx_mapping,
            query_metadata,
            num_accepted_tokens,
            is_prefilling,
            live_cols,
            materialize_src_cols,
            materialize_dst_cols,
            materialize_token_counts,
            self.block_table,
            self.ring_start,
            self.num_committed,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.active_request_indices,
            self.block_table.stride(0),
            self.src_slots.stride(0),
            num_reqs,
            LOGICAL_WINDOW=self.logical_window,
            RING_BUFFER_LEN=self.ring_buffer_len,
            NUM_LAYERS=len(self.mixers),
            PAD_SLOT_ID=NULL_BLOCK_ID,
            QUERY_METADATA_IS_CUMULATIVE=query_metadata_is_cumulative,
            HAS_IDX_MAPPING=idx_mapping is not None,
            MATERIALIZE_PREFIXES=self.materialize_prefixes,
            MAX_NUM_REQS=self.max_num_reqs,
        )

    def materialize(self) -> None:
        """Execute the plan prepared by ``preprocess`` or ``postprocess``."""
        if not self.materialize_prefixes:
            raise RuntimeError("ReplaySSM materialization requires align or all mode")
        first = self.mixers[0]
        mamba_config = first.mamba_config
        rand_seed = None
        philox_rounds = 0
        if mamba_config.enable_stochastic_rounding:
            rand_seed = torch.randint(
                0, 2**32, (1,), device=self.src_slots.device, dtype=torch.int64
            )
            philox_rounds = mamba_config.stochastic_rounding_philox_rounds or 10
        _load_replayssm_materialize()(
            *self.materialize_tables,
            self.src_slots,
            self.dst_slots,
            self.plan_ring_start,
            self.plan_flush_count,
            self.active_request_indices,
            state_dtype=first.kv_cache[1].dtype,
            input_dtype=first.kv_cache[2].dtype,
            matrixA_dtype=first.A.dtype,
            dim=first.kv_cache[1].size(2),
            dstate=first.kv_cache[1].size(3),
            num_heads=first.kv_cache[1].size(1),
            heads_per_group=(first.kv_cache[1].size(1) // first.kv_cache[4].size(1)),
            max_window=self.logical_window,
            ring_buffer_len=self.ring_buffer_len,
            rand_seed=rand_seed,
            philox_rounds=philox_rounds,
        )


@dataclass
class ReplaySSMModelContext:
    """ReplaySSM lifecycle split by physical cache-slot namespace."""

    groups: list[_ReplaySSMGroupContext]
    materialize_prefixes: bool

    @classmethod
    def create(
        cls,
        kv_cache_config: KVCacheConfig,
        mamba_group_ids: Sequence[int],
        forward_context: Mapping[str, Any],
        block_tables: Sequence[torch.Tensor],
        max_num_reqs: int,
    ) -> "ReplaySSMModelContext | None":
        grouped = _flashinfer_replayssm_mixers_by_group(
            kv_cache_config, mamba_group_ids, forward_context
        )
        if not grouped:
            return None
        if len(block_tables) != len(mamba_group_ids):
            raise ValueError(
                f"expected {len(mamba_group_ids)} Mamba block tables, "
                f"got {len(block_tables)}"
            )

        block_table_by_gid = dict(zip(mamba_group_ids, block_tables))
        modes = set()
        group_args = []
        for gid, mixers in grouped:
            spec = kv_cache_config.kv_cache_groups[gid].kv_cache_spec
            if not isinstance(spec, MambaSpec):
                raise TypeError(
                    "FlashInfer ReplaySSM layers require a Mamba cache spec; "
                    f"got {type(spec).__name__}"
                )
            modes.add(spec.mamba_cache_mode)
            group_args.append((mixers, block_table_by_gid[gid], spec.mamba_cache_mode))
        if len(modes) != 1:
            raise ValueError(
                "model-wide ReplaySSM requires one Mamba cache mode; "
                f"got {sorted(modes)}"
            )

        groups = [
            _ReplaySSMGroupContext.create(*args, max_num_reqs) for args in group_args
        ]
        return cls(
            groups=groups,
            materialize_prefixes=next(iter(modes)) in ("align", "all"),
        )

    def preprocess(self, **kwargs: Any) -> None:
        for group in self.groups:
            group.preprocess(**kwargs)

    def postprocess(self, **kwargs: Any) -> None:
        for group in self.groups:
            group.postprocess(**kwargs)

    def materialize(self) -> None:
        if not self.materialize_prefixes:
            raise RuntimeError("ReplaySSM materialization requires align or all mode")
        for group in self.groups:
            group.materialize()


class MambaSSUBackend(ABC):
    """Abstract base class for Mamba SSU backends."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None: ...


class TritonSSUBackend(MambaSSUBackend):
    """Triton-based SSU backend (vLLM's default)."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )

        self._kernel = _triton_selective_state_update

    @property
    def name(self) -> str:
        return "triton"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            null_block_id=null_block_id,
            out=out,
            num_accepted_tokens=num_accepted_tokens,
            cu_seqlens=cu_seqlens,
            is_blackwell=is_blackwell,
            enable_stochastic_rounding=self._mamba_config.enable_stochastic_rounding,
            cache_philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds,
        )


class FlashInferSSUBackend(MambaSSUBackend):
    """FlashInfer-based SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        try:
            from flashinfer.mamba import selective_state_update as _fi_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba SSU backend. "
                "Please install flashinfer (>= 0.6.4): "
                "pip install flashinfer-python"
            ) from e
        logger.info_once("Using FlashInfer Mamba SSU algorithm: %s", self._algorithm)
        self._kernel = _fi_ssu

    @property
    def _algorithm(self) -> MambaSSUAlgorithm:
        return self._mamba_config.ssu_algorithm or "auto"

    @property
    def name(self) -> str:
        return "flashinfer"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        rand_seed = (
            torch.randint(0, 2**32, (1,), device=state.device)
            if self._mamba_config.enable_stochastic_rounding
            else None
        )
        self._kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            cu_seqlens=cu_seqlens,
            num_accepted_tokens=num_accepted_tokens,
            cache_steps=state_batch_indices.size(-1)
            if cu_seqlens is not None and state_batch_indices is not None
            else 0,
            pad_slot_id=null_block_id,
            out=out,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
            algorithm=self._algorithm,
        )


class CPUSSUBackend(MambaSSUBackend):
    """CPU SSU backend using the compiled C++ VSX/scalar kernel.

    On CPU-only platforms (PowerPC, x86 without CUDA) this dispatches to
    the vectorized C++ kernel registered as ``torch.ops._C.selective_state_update_cpu``.
    That kernel uses vec_op SIMD intrinsics (VSX on ppc64le, AVX2 on x86,
    scalar fallback elsewhere) and is parallelised with OpenMP across heads.

    Falls back to the pure-PyTorch implementation only if the C++ op is
    unavailable (e.g. a CPU-less build).
    """

    def __init__(self, mamba_config: MambaConfig):
        super().__init__(mamba_config)
        from vllm import _custom_ops as ops

        self._cpp_kernel = ops.selective_state_update_cpu
        logger.info("CPUSSUBackend: using compiled C++ selective_state_update kernel.")

    @property
    def name(self) -> str:
        return "cpu"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        dt_bias: torch.Tensor,
        z: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        dst_state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        out: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        is_blackwell: bool = False,
    ) -> None:
        # C++ kernel: state shape expected as (nstates, nheads, dim, dstate)
        # The kernel writes in-place into `out` and updates `state`.
        self._cpp_kernel(
            state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z,
            dt_bias,
            dt_softplus,
            state_batch_indices,
            dst_state_batch_indices,
            null_block_id,
            out,
            num_accepted_tokens,
            cu_seqlens,
        )


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
    MambaBackendEnum.CPU: CPUSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None


_flashinfer_replayssm_kernel: Callable[..., torch.Tensor] | None = None


@cache
def flashinfer_replayssm_autotune_supported() -> bool:
    """Return True when FlashInfer exposes ReplaySSM autotuning."""
    try:
        from flashinfer.mamba.checkpointing_ssu import (  # noqa: F401
            CheckpointingSSURunner,
        )
    except ImportError:
        return False
    return True


def selective_state_update_replayssm_flashinfer(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    x_cache: torch.Tensor,
    B_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    ring_start: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    scratch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    enable_stochastic_rounding: bool = False,
    stochastic_rounding_philox_rounds: int = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Run FlashInfer checkpointing SSU with model-owned tracker metadata."""
    if _flashinfer_replayssm_kernel is None:
        raise RuntimeError(
            "FlashInfer ReplaySSM has not been initialized. "
            "Call initialize_mamba_ssu_backend() with use_replayssm=True."
        )

    if x.dim() == 3:
        dim = 0 if cu_seqlens is not None else 1
        x = x.unsqueeze(dim)
        dt = dt.unsqueeze(dim)
        B = B.unsqueeze(dim)
        C = C.unsqueeze(dim)
        out = out.unsqueeze(dim)

    indices = state_batch_indices
    if indices is not None and indices.dim() > 1:
        indices = indices[:, 0]

    cb_scaled = cumAdt_vec = cb_old = None
    if scratch is not None:
        cb_scaled, cumAdt_vec, cb_old = scratch

    rand_seed = (
        torch.randint(0, 2**32, (1,), device=state.device, dtype=torch.int64)
        if enable_stochastic_rounding
        else None
    )
    return _flashinfer_replayssm_kernel(
        state,
        x_cache,
        B_cache,
        dt_cache,
        ring_start,
        prev_num_accepted_tokens,
        x,
        dt,
        A,
        B,
        C,
        out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=indices,
        pad_slot_id=null_block_id,
        rand_seed=rand_seed,
        philox_rounds=stochastic_rounding_philox_rounds or 10,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        enable_pdl=enable_pdl,
        cb_scaled=cb_scaled,
        cumAdt_vec=cumAdt_vec,
        cb_old=cb_old,
    )


def _reinterpret_u64_as_i64(value: int) -> int:
    """Preserve a uint64 pointer bit pattern in a torch.int64 tensor."""
    return value if value < (1 << 63) else value - (1 << 64)


def _cuda_i64_ptrs(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [_reinterpret_u64_as_i64(t.data_ptr()) for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


def _cuda_i64_slot_strides(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(
        [t.stride(0) for t in tensors],
        dtype=torch.int64,
        device=tensors[0].device,
    )


def _flashinfer_replayssm_mixers_by_group(
    kv_cache_config: KVCacheConfig,
    mamba_group_ids: Sequence[int],
    forward_context: Mapping[str, Any],
) -> list[tuple[int, list[Any]]]:
    grouped: list[tuple[int, list[Any]]] = []
    for gid in mamba_group_ids:
        mixers: list[Any] = []
        for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names:
            layer = forward_context.get(layer_name)
            if layer is None:
                continue
            kv_cache = getattr(layer, "kv_cache", ())
            mamba_config = getattr(layer, "mamba_config", None)
            backend = getattr(mamba_config, "backend", None)
            if (
                getattr(layer, "use_replayssm", False)
                and backend == MambaBackendEnum.FLASHINFER
                and len(kv_cache) >= 5
            ):
                mixers.append(layer)
        if mixers:
            grouped.append((gid, mixers))
    return grouped


@cache
def _load_replayssm_materialize() -> Callable[..., None]:
    try:
        from flashinfer.mamba.replayssm_materialize import (
            replayssm_materialize,
        )
    except ImportError as e:
        raise ImportError(
            "FlashInfer ReplaySSM prefix caching requires "
            "flashinfer.mamba.replayssm_materialize"
        ) from e
    return replayssm_materialize


def _validate_replayssm_cache(mixers: list[Any]) -> None:
    """Validate the cache tensors required by model-wide tracker ownership."""
    for layer_idx, mixer in enumerate(mixers):
        cache_tensors = mixer.kv_cache[1:5]
        if any(tensor.numel() == 0 for tensor in cache_tensors):
            raise RuntimeError(
                "FlashInfer ReplaySSM requires allocated SSM and replay-ring "
                f"cache tensors for every layer; layer {layer_idx} is empty"
            )
        if any(not tensor.is_cuda for tensor in cache_tensors):
            devices = [str(tensor.device) for tensor in cache_tensors]
            raise RuntimeError(
                "FlashInfer ReplaySSM requires CUDA cache tensors for every "
                f"layer; layer {layer_idx} uses {devices}"
            )

        ring_start = mixer._replayssm_ring_start
        num_committed = mixer._replayssm_prev_num_accepted
        if ring_start.numel() == 0 or num_committed.numel() == 0:
            raise RuntimeError(
                "FlashInfer ReplaySSM requires allocated ring trackers for "
                f"every layer; layer {layer_idx} is empty"
            )
        if not ring_start.is_cuda or not num_committed.is_cuda:
            raise RuntimeError(
                "FlashInfer ReplaySSM requires CUDA ring trackers for every layer"
            )
        if (
            ring_start.ndim != 1
            or num_committed.ndim != 1
            or ring_start.dtype != torch.int32
            or num_committed.dtype != torch.int32
        ):
            raise ValueError("ReplaySSM ring trackers must be 1D int32 tensors")
        if ring_start.numel() != num_committed.numel():
            raise ValueError(
                "ReplaySSM ring trackers must have equal capacities; got "
                f"{ring_start.numel()} and {num_committed.numel()}"
            )


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
    *,
    use_replayssm: bool = False,
) -> None:
    """Initialize the Mamba SSU backend and optional FlashInfer ReplaySSM."""
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _flashinfer_replayssm_kernel, _mamba_ssu_backend
    backend = mamba_config.backend

    if backend == MambaBackendEnum.TRITON:
        from vllm.platforms import current_platform

        if current_platform.is_cpu():
            logger.info(
                "CPU platform detected: overriding Mamba SSU backend "
                "from 'triton' to 'cpu'."
            )
            backend = MambaBackendEnum.CPU

    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )
    if use_replayssm and backend not in (
        MambaBackendEnum.TRITON,
        MambaBackendEnum.FLASHINFER,
    ):
        raise ValueError(f"ReplaySSM does not support mamba backend {backend.value!r}")

    backend_cls = _BACKEND_REGISTRY[backend]
    if not isinstance(_mamba_ssu_backend, backend_cls):
        _mamba_ssu_backend = backend_cls(mamba_config)
        logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)

    _flashinfer_replayssm_kernel = None
    if use_replayssm and backend == MambaBackendEnum.FLASHINFER:
        try:
            from flashinfer.mamba.checkpointing_ssu import checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer ReplaySSM requires a compatible flashinfer-python package"
            ) from e
        _flashinfer_replayssm_kernel = checkpointing_ssu
    if use_replayssm:
        logger.info("Using %s ReplaySSM backend.", backend.value)


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def selective_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    dst_state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    out: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    is_blackwell: bool = False,
) -> None:
    """Unified dispatch for Mamba selective state update.

    Delegates to the initialized backend (Triton or FlashInfer).
    """
    get_mamba_ssu_backend()(
        state,
        x,
        dt,
        A,
        B,
        C,
        D,
        dt_bias,
        z=z,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        dst_state_batch_indices=dst_state_batch_indices,
        null_block_id=null_block_id,
        out=out,
        num_accepted_tokens=num_accepted_tokens,
        cu_seqlens=cu_seqlens,
        is_blackwell=is_blackwell,
    )
