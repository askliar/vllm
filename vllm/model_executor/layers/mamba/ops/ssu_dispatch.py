# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Dispatch module for Mamba selective state update (SSU) backends.

Provides a unified `selective_state_update` function that dispatches to
either the Triton or FlashInfer backend based on the configured
`MambaBackendEnum`. Follows SGLang's dispatch pattern adapted for vLLM.
"""

from abc import ABC, abstractmethod

import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

logger = init_logger(__name__)


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

        self._kernel = _fi_ssu

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
        )


class FlashInferCheckpointingSSUBackend:
    """FlashInfer-based checkpointing SSU backend."""

    def __init__(self, mamba_config: MambaConfig):
        self._mamba_config = mamba_config
        try:
            from flashinfer.mamba import checkpointing_ssu as _fi_checkpointing_ssu
        except ImportError as e:
            raise ImportError(
                "FlashInfer is required for the flashinfer Mamba "
                "checkpointing SSU backend. Please install a FlashInfer "
                "version that provides flashinfer.mamba.checkpointing_ssu."
            ) from e

        self._kernel = _fi_checkpointing_ssu

    @property
    def name(self) -> str:
        return "flashinfer_checkpointing"

    def __call__(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        out: torch.Tensor,
        old_x: torch.Tensor,
        old_B: torch.Tensor,
        old_dt_proc: torch.Tensor,
        old_cumAdt: torch.Tensor,
        cache_buf_idx: torch.Tensor,
        prev_num_accepted_tokens: torch.Tensor,
        D: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        dt_bias: torch.Tensor | None = None,
        dt_softplus: bool = False,
        state_batch_indices: torch.Tensor | None = None,
        null_block_id: int = NULL_BLOCK_ID,
        state_scale: torch.Tensor | None = None,
        rand_seed: torch.Tensor | None = None,
        d_split: int | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> None:
        _validate_checkpointing_ssu_inputs(
            state=state,
            out=out,
            old_x=old_x,
            old_B=old_B,
            old_dt_proc=old_dt_proc,
            old_cumAdt=old_cumAdt,
            cache_buf_idx=cache_buf_idx,
            prev_num_accepted_tokens=prev_num_accepted_tokens,
            state_batch_indices=state_batch_indices,
            state_scale=state_scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        if rand_seed is None and self._mamba_config.enable_stochastic_rounding:
            rand_seed = torch.randint(0, 2**32, (1,), device=state.device)

        self._kernel(
            state,
            old_x,
            old_B,
            old_dt_proc,
            old_cumAdt,
            cache_buf_idx,
            prev_num_accepted_tokens,
            x,
            dt,
            A,
            B,
            C,
            out,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=dt_softplus,
            state_batch_indices=state_batch_indices,
            pad_slot_id=null_block_id,
            state_scale=state_scale,
            rand_seed=rand_seed,
            philox_rounds=self._mamba_config.stochastic_rounding_philox_rounds or 10,
            d_split=d_split,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )


_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
}

_mamba_ssu_backend: MambaSSUBackend | None = None
_mamba_ssu_checkpointing_backend: FlashInferCheckpointingSSUBackend | None = None


# TODO: Remove this function once the checkpointing SSU backend is fully tested.
def _validate_checkpointing_ssu_inputs(
    *,
    state: torch.Tensor,
    out: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt_proc: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    state_batch_indices: torch.Tensor | None,
    state_scale: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> None:
    if out.device != state.device:
        raise ValueError(
            "out must be on the same device as state. "
            f"Got {out.device=} and {state.device=}."
        )
    if state.dim() != 4:
        raise ValueError(
            "checkpointing_state_update expects state with shape "
            "(cache, nheads, dim, dstate)."
        )
    cache_size, nheads, dim, dstate = state.shape

    cache_tensors = {
        "old_x": old_x,
        "old_B": old_B,
        "old_dt_proc": old_dt_proc,
        "old_cumAdt": old_cumAdt,
    }
    for name, tensor in cache_tensors.items():
        if tensor.device != state.device:
            raise ValueError(
                f"{name} must be on the same device as state. "
                f"Got {tensor.device=} and {state.device=}."
            )
        if tensor.shape[0] != cache_size:
            raise ValueError(
                f"{name} must have cache dimension {cache_size}; "
                f"got shape {tuple(tensor.shape)}."
            )

    if old_x.dim() != 4:
        raise ValueError("old_x must have shape (cache, max_window, nheads, dim).")
    max_window = old_x.shape[1]
    if max_window < 1 or max_window > 16:
        raise ValueError(
            "checkpointing_state_update expects max_window in [1, 16]; "
            f"got {max_window}."
        )
    if old_x.shape[2:] != (nheads, dim):
        raise ValueError(
            "old_x shape mismatch: expected trailing dims "
            f"{(nheads, dim)}, got {tuple(old_x.shape[2:])}."
        )

    if old_B.dim() != 5 or old_B.shape[1] != 2 or old_B.shape[2] != max_window:
        raise ValueError(
            "old_B must have shape (cache, 2, max_window, ngroups, dstate)."
        )
    if old_B.shape[4] != dstate:
        raise ValueError(
            "old_B dstate dimension must match state. "
            f"Expected {dstate}, got {old_B.shape[4]}."
        )

    expected_dt_shape = (cache_size, 2, nheads, max_window)
    if old_dt_proc.shape != expected_dt_shape:
        raise ValueError(
            "old_dt_proc must have shape "
            f"{expected_dt_shape}; got {tuple(old_dt_proc.shape)}."
        )
    if old_cumAdt.shape != expected_dt_shape:
        raise ValueError(
            "old_cumAdt must have shape "
            f"{expected_dt_shape}; got {tuple(old_cumAdt.shape)}."
        )
    if old_dt_proc.dtype != torch.float32 or old_cumAdt.dtype != torch.float32:
        raise ValueError("old_dt_proc and old_cumAdt must be torch.float32.")

    expected_counter_shape = (cache_size,)
    for name, tensor in (
        ("cache_buf_idx", cache_buf_idx),
        ("prev_num_accepted_tokens", prev_num_accepted_tokens),
    ):
        if tensor.shape != expected_counter_shape:
            raise ValueError(
                f"{name} must have shape {expected_counter_shape}; "
                f"got {tuple(tensor.shape)}."
            )
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be torch.int32, got {tensor.dtype}.")
        if tensor.device != state.device:
            raise ValueError(
                f"{name} must be on the same device as state. "
                f"Got {tensor.device=} and {state.device=}."
            )

    if state_batch_indices is None:
        raise ValueError(
            "checkpointing_state_update requires a 1D state_batch_indices "
            "tensor mapping each decode row to a persistent Mamba cache slot."
        )
    if state_batch_indices.dim() != 1:
        raise ValueError(
            "checkpointing_state_update requires 1D state_batch_indices. "
            "Do not pass the speculative 2D state_indices_tensor_d directly."
        )
    if state_batch_indices.device != state.device:
        raise ValueError(
            "state_batch_indices must be on the same device as state. "
            f"Got {state_batch_indices.device=} and {state.device=}."
        )
    if state_batch_indices.dtype != torch.int32:
        raise ValueError(
            f"state_batch_indices must be torch.int32, got {state_batch_indices.dtype}."
        )

    if cu_seqlens is None:
        raise ValueError("checkpointing_state_update requires cu_seqlens.")
    if cu_seqlens.dim() != 1:
        raise ValueError("checkpointing_state_update requires 1D cu_seqlens.")
    if cu_seqlens.device != state.device:
        raise ValueError(
            "cu_seqlens must be on the same device as state. "
            f"Got {cu_seqlens.device=} and {state.device=}."
        )
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}.")
    if max_seqlen is None:
        raise ValueError("checkpointing_state_update requires max_seqlen.")
    if max_seqlen < 1:
        raise ValueError(
            "checkpointing_state_update requires positive max_seqlen; "
            f"got {max_seqlen}."
        )

    if state_scale is not None:
        expected_scale_shape = (cache_size, nheads, dim)
        if state_scale.device != state.device:
            raise ValueError(
                "state_scale must be on the same device as state. "
                f"Got {state_scale.device=} and {state.device=}."
            )
        if state_scale.shape != expected_scale_shape:
            raise ValueError(
                f"state_scale must have shape {expected_scale_shape}; "
                f"got {tuple(state_scale.shape)}."
            )
        if state_scale.dtype != torch.float32:
            raise ValueError("state_scale must be torch.float32.")


def initialize_mamba_ssu_backend(
    mamba_config: MambaConfig,
    kv_cache_config: KVCacheConfig,
) -> None:
    """Initialize the global Mamba SSU backend.

    No-op if `kv_cache_config` contains no specs that call
    selective_state_update.
    """
    if not any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.mamba_type
        in (MambaAttentionBackendEnum.MAMBA1, MambaAttentionBackendEnum.MAMBA2)
        for g in kv_cache_config.kv_cache_groups
    ):
        return

    global _mamba_ssu_backend, _mamba_ssu_checkpointing_backend

    backend = mamba_config.backend
    if backend not in _BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown Mamba SSU backend: {backend}. "
            f"Valid options: {list(_BACKEND_REGISTRY.keys())}"
        )

    backend_cls = _BACKEND_REGISTRY[backend]
    if not isinstance(_mamba_ssu_backend, backend_cls):
        _mamba_ssu_backend = backend_cls(mamba_config)
        logger.info("Using %s Mamba SSU backend.", _mamba_ssu_backend.name)
    else:
        _mamba_ssu_backend._mamba_config = mamba_config

    has_checkpointing_mamba = any(
        isinstance(g.kv_cache_spec, MambaSpec)
        and g.kv_cache_spec.ssm_checkpoint_interval > 1
        for g in kv_cache_config.kv_cache_groups
    )

    if has_checkpointing_mamba:
        if backend != MambaBackendEnum.FLASHINFER:
            raise ValueError("Mamba SSU checkpointing requires the flashinfer backend.")
        if not isinstance(
            _mamba_ssu_checkpointing_backend,
            FlashInferCheckpointingSSUBackend,
        ):
            _mamba_ssu_checkpointing_backend = FlashInferCheckpointingSSUBackend(
                mamba_config
            )
            logger.info(
                "Using %s Mamba SSU backend.",
                _mamba_ssu_checkpointing_backend.name,
            )
        else:
            _mamba_ssu_checkpointing_backend._mamba_config = mamba_config
    else:
        _mamba_ssu_checkpointing_backend = None


def get_mamba_ssu_backend() -> MambaSSUBackend:
    """Get the current Mamba SSU backend. Raises if not initialized."""
    if _mamba_ssu_backend is None:
        raise RuntimeError(
            "Mamba SSU backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_backend


def get_mamba_ssu_checkpointing_backend() -> FlashInferCheckpointingSSUBackend:
    """Get the current Mamba SSU checkpointing backend."""
    if _mamba_ssu_checkpointing_backend is None:
        raise RuntimeError(
            "Mamba SSU checkpointing backend has not been initialized. "
            "Call initialize_mamba_ssu_backend() first."
        )
    return _mamba_ssu_checkpointing_backend


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


def checkpointing_state_update(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    out: torch.Tensor,
    old_x: torch.Tensor,
    old_B: torch.Tensor,
    old_dt_proc: torch.Tensor,
    old_cumAdt: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    dt_softplus: bool = False,
    state_batch_indices: torch.Tensor | None = None,
    null_block_id: int = NULL_BLOCK_ID,
    state_scale: torch.Tensor | None = None,
    rand_seed: torch.Tensor | None = None,
    d_split: int | None = None,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> None:
    """Dispatch FlashInfer Mamba checkpointing SSU.

    This path is intentionally separate from `selective_state_update` because
    FlashInfer's checkpointing kernel has a different ABI and requires replay
    cache tensors plus per-cache-slot counters.
    """
    get_mamba_ssu_checkpointing_backend()(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        out=out,
        old_x=old_x,
        old_B=old_B,
        old_dt_proc=old_dt_proc,
        old_cumAdt=old_cumAdt,
        cache_buf_idx=cache_buf_idx,
        prev_num_accepted_tokens=prev_num_accepted_tokens,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        state_batch_indices=state_batch_indices,
        null_block_id=null_block_id,
        state_scale=state_scale,
        rand_seed=rand_seed,
        d_split=d_split,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
