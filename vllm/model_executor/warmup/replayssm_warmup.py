# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.mamba import MambaBackendEnum
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    _load_gdn_replayssm_materialize,
    flashinfer_replayssm_autotune_supported,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _replayssm_autotune_kwargs(
    runner: "GPUModelRunner",
    max_token_prefill_kwargs: dict[str, Any],
) -> tuple[int, dict[str, Any]] | None:
    config = runner.vllm_config
    if not (
        config.cache_config.use_replayssm
        and config.mamba_config.backend == MambaBackendEnum.FLASHINFER
    ):
        return None
    if not flashinfer_replayssm_autotune_supported():
        logger.info_once(
            "Skipping FlashInfer ReplaySSM autotuning because "
            "flashinfer.mamba.checkpointing_ssu.CheckpointingSSURunner "
            "is unavailable."
        )
        return None
    v2_runner: Any = runner
    query_len = (
        v2_runner.decode_query_len
        if config.use_v2_model_runner
        else runner.uniform_decode_query_len
    )
    max_num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        runner.max_num_tokens // query_len,
        runner.kv_cache_config.num_blocks - 1,
    )
    if max_num_reqs <= 0:
        logger.warning_once(
            "Skipping FlashInfer ReplaySSM autotuning because no non-padding "
            "state slot is available."
        )
        return None

    decode_kwargs = {
        **max_token_prefill_kwargs,
        "num_tokens": max_num_reqs * query_len,
        "uniform_decode": True,
    }
    if config.use_v2_model_runner:
        decode_kwargs["valid_dummy_state_slots"] = True
    else:
        decode_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=query_len + 1,
        )
    return max_num_reqs, decode_kwargs


@contextmanager
def _temporary_replayssm_autotune_state(
    runner: "GPUModelRunner", max_num_reqs: int
) -> Iterator[None]:
    reset_tensors: dict[int, torch.Tensor] = {}
    for module in runner.get_model().modules():
        if not getattr(module, "use_flashinfer_replayssm", False):
            continue
        assert module.replayssm_buffer_len is not None
        ring_start = module._replayssm_ring_start
        prev_num_accepted = module._replayssm_prev_num_accepted
        tensors = (
            *module.kv_cache,
            *module.replayssm_cache,
            ring_start,
            prev_num_accepted,
        )
        for tensor in tensors:
            if tensor.numel():
                reset_tensors.setdefault(tensor.data_ptr(), tensor)

    v2_runner: Any = runner
    block_tables = saved_block_ids = None
    if not runner.vllm_config.use_v2_model_runner:
        block_tables = runner.input_batch.block_table.block_tables
        saved_block_ids = tuple(
            block_table.block_table.np[:max_num_reqs, 0].copy()
            for block_table in block_tables
        )
        dummy_block_ids = range(1, max_num_reqs + 1)
        for block_table in block_tables:
            block_table.block_table.np[:max_num_reqs, 0] = dummy_block_ids
        runner.input_batch.block_table.commit_block_table(max_num_reqs)

    try:
        yield
    finally:
        if runner.vllm_config.use_v2_model_runner:
            v2_runner.block_tables.get_dummy_block_tables(max_num_reqs)
        else:
            assert block_tables is not None and saved_block_ids is not None
            for block_table, block_ids in zip(block_tables, saved_block_ids):
                block_table.block_table.np[:max_num_reqs, 0] = block_ids
            runner.input_batch.block_table.commit_block_table(max_num_reqs)
        for tensor in reset_tensors.values():
            tensor[1 : max_num_reqs + 1].zero_()


def replayssm_autotune_warmup(runner: "GPUModelRunner") -> None:
    max_token_prefill_kwargs = {
        "num_tokens": runner.scheduler_config.max_num_batched_tokens,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
    }
    autotune = _replayssm_autotune_kwargs(runner, max_token_prefill_kwargs)
    if autotune is None:
        return
    max_num_reqs, decode_kwargs = autotune
    with _temporary_replayssm_autotune_state(runner, max_num_reqs):
        runner._dummy_run(**decode_kwargs)


def _warm_gdn_replayssm_materializer(
    runner: "GPUModelRunner", max_num_reqs: int
) -> None:
    """Compile every distinct GDN prefix-materializer specialization."""
    if runner.vllm_config.cache_config.mamba_cache_mode != "align":
        return

    materialize = _load_gdn_replayssm_materialize()
    warmed: set[tuple[Any, ...]] = set()
    for module in runner.get_model().modules():
        if not getattr(module, "use_flashinfer_replayssm", False) or getattr(
            module, "replayssm_executed_query_width", None
        ) not in (1, 4, 8):
            continue
        state = module.kv_cache[1]
        u_cache, k_cache, g_cache = module.replayssm_cache
        signature = (
            state.device,
            state.dtype,
            tuple(state.shape),
            tuple(state.stride()),
            tuple(k_cache.shape),
            tuple(k_cache.stride()),
            tuple(u_cache.shape),
            tuple(u_cache.stride()),
            tuple(g_cache.shape),
            tuple(g_cache.stride()),
        )
        if signature in warmed:
            continue
        warmed.add(signature)
        inactive = torch.full(
            (max_num_reqs,), -1, dtype=torch.int32, device=state.device
        )
        zeros = torch.zeros_like(inactive)
        materialize(
            state=state,
            src_slots=inactive,
            dst_slots=inactive,
            k_cache=k_cache,
            u_cache=u_cache,
            g_cache=g_cache,
            cache_base=zeros,
            count=inactive,
            active_request_indices=inactive,
            num_active=torch.zeros(1, dtype=torch.int32, device=state.device),
        )


def gdn_replayssm_warmup(runner: "GPUModelRunner") -> None:
    """Compile GDN replay for every intended uniform-decode graph bucket."""
    config = runner.vllm_config
    if not config.is_gdn_replayssm_enabled():
        return
    query_len = 1 + config.num_speculative_tokens
    max_num_reqs = min(
        runner.scheduler_config.max_num_seqs,
        runner.max_num_tokens // query_len,
        runner.kv_cache_config.num_blocks - 1,
    )
    if max_num_reqs <= 0:
        raise ValueError(
            "FlashInfer GDN ReplaySSM warmup requires a non-padding state slot"
        )
    if config.use_v2_model_runner:
        v2_runner: Any = runner
        manager = v2_runner.cudagraph_manager
        capture_reqs = {
            desc.num_reqs
            for descs in manager._capture_descs.values()
            for desc in descs
            if desc.uniform_token_count == query_len
            and desc.num_reqs is not None
            and 0 < desc.num_reqs <= max_num_reqs
        }
    else:
        capture_reqs = {
            size // query_len
            for size in runner.cudagraph_batch_sizes
            if size % query_len == 0 and 0 < size // query_len <= max_num_reqs
        }
    if not capture_reqs:
        capture_reqs.add(max_num_reqs)

    decode_kwargs: dict[str, Any] = {
        "uniform_decode": True,
        "skip_eplb": True,
        "is_profile": True,
        "randomize_inputs": True,
    }
    if config.use_v2_model_runner:
        decode_kwargs["valid_dummy_state_slots"] = True
    else:
        decode_kwargs.update(
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=query_len + 1,
        )
    with _temporary_replayssm_autotune_state(runner, max_num_reqs):
        for num_reqs in sorted(capture_reqs):
            runner._dummy_run(num_tokens=num_reqs * query_len, **decode_kwargs)
        _warm_gdn_replayssm_materializer(runner, max_num_reqs)
