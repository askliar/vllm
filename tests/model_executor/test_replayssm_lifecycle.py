# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    share_replayssm_ring_trackers,
)
from vllm.model_executor.layers.mamba.ops import ssu_dispatch
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import ReplaySSMModelContext
from vllm.model_executor.warmup.replayssm_warmup import gdn_replayssm_warmup
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec


def _gdn_layer(
    num_blocks: int = 3, executed_query_width: int = 4, ring_slots: int = 32
):
    return SimpleNamespace(
        use_flashinfer_replayssm=True,
        replayssm_buffer_len=16,
        replayssm_executed_query_width=executed_query_width,
        kv_cache=(
            torch.zeros(num_blocks, 3, 384, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 1, 128, 128, dtype=torch.bfloat16),
        ),
        replayssm_cache=(
            torch.zeros(num_blocks, 1, ring_slots, 128, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 1, ring_slots, 128, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 1, ring_slots, dtype=torch.float32),
        ),
        _replayssm_ring_start=torch.empty(0, dtype=torch.int32),
        _replayssm_prev_num_accepted=torch.empty(0, dtype=torch.int32),
    )


def test_gdn_replayssm_reuses_group_lifecycle_without_materializer():
    layers = {"layer.0": _gdn_layer(), "layer.1": _gdn_layer()}
    spec = MambaSpec(
        block_size=16,
        shapes=((3, 384), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        replayssm_shapes=((1, 32, 128), (1, 32, 128), (1, 32)),
        replayssm_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="none",
    )
    group = KVCacheGroupSpec(layer_names=list(layers), kv_cache_spec=spec)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[group],
    )

    share_replayssm_ring_trackers(list(layers), layers, [group])
    assert layers["layer.0"]._replayssm_ring_start is (
        layers["layer.1"]._replayssm_ring_start
    )
    assert layers["layer.0"]._replayssm_prev_num_accepted is (
        layers["layer.1"]._replayssm_prev_num_accepted
    )

    context = ReplaySSMModelContext.create(
        config,
        mamba_group_ids=[0],
        forward_context=layers,
        block_tables=[torch.zeros(2, 1, dtype=torch.int32)],
        max_num_reqs=2,
    )
    assert context is not None
    assert not context.materialize_prefixes
    assert context.groups[0].materialize_tables is None
    assert context.groups[0].ring_buffer_len == 32
    assert context.groups[0].executed_query_width == 4


def test_gdn_replayssm_align_dispatches_single_layer_materializer(monkeypatch):
    layers = {
        "layer.0": _gdn_layer(executed_query_width=1, ring_slots=16),
        "layer.1": _gdn_layer(executed_query_width=1, ring_slots=16),
    }
    spec = MambaSpec(
        block_size=16,
        shapes=((3, 384), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        replayssm_shapes=((1, 16, 128), (1, 16, 128), (1, 16)),
        replayssm_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
    )
    group_spec = KVCacheGroupSpec(layer_names=list(layers), kv_cache_spec=spec)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[group_spec],
    )
    context = ReplaySSMModelContext.create(
        config,
        mamba_group_ids=[0],
        forward_context=layers,
        block_tables=[torch.zeros(2, 1, dtype=torch.int32)],
        max_num_reqs=2,
    )
    assert context is not None
    group = context.groups[0]
    group.src_slots.copy_(torch.tensor([[1, -1], [1, -1]], dtype=torch.int32))
    group.dst_slots.copy_(torch.tensor([[2, -1], [2, -1]], dtype=torch.int32))
    group.plan_ring_start.copy_(torch.tensor([0, 0], dtype=torch.int32))
    group.plan_flush_count.copy_(torch.tensor([5, -1], dtype=torch.int32))
    group.active_request_indices.copy_(torch.tensor([0, -1], dtype=torch.int32))
    group.num_active.fill_(1)

    materialize = Mock()
    monkeypatch.setattr(
        ssu_dispatch, "_load_gdn_replayssm_materialize", lambda: materialize
    )
    context.materialize()

    assert materialize.call_count == 2
    for layer_idx, materialize_call in enumerate(materialize.call_args_list):
        kwargs = materialize_call.kwargs
        layer = layers[f"layer.{layer_idx}"]
        assert kwargs["state"] is layer.kv_cache[1]
        assert kwargs["u_cache"] is layer.replayssm_cache[0]
        assert kwargs["k_cache"] is layer.replayssm_cache[1]
        assert kwargs["g_cache"] is layer.replayssm_cache[2]
        assert kwargs["src_slots"].data_ptr() == group.src_slots[layer_idx].data_ptr()
        assert kwargs["dst_slots"].data_ptr() == group.dst_slots[layer_idx].data_ptr()
        assert kwargs["cache_base"] is group.plan_ring_start
        assert kwargs["count"] is group.plan_flush_count
        assert kwargs["active_request_indices"] is group.active_request_indices
        assert kwargs["num_active"] is group.num_active


@pytest.mark.parametrize(
    ("executed_query_width", "ring_slots"),
    [(1, 32), (4, 16), (8, 16)],
)
def test_gdn_replayssm_rejects_mismatched_ring_geometry(
    executed_query_width, ring_slots
):
    layer = _gdn_layer(executed_query_width=executed_query_width, ring_slots=ring_slots)
    spec = MambaSpec(
        block_size=16,
        shapes=((3, 384), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        replayssm_shapes=(
            (1, ring_slots, 128),
            (1, ring_slots, 128),
            (1, ring_slots),
        ),
        replayssm_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="none",
    )
    group_spec = KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=spec)
    config = KVCacheConfig(
        num_blocks=3,
        kv_cache_tensors=[],
        kv_cache_groups=[group_spec],
    )

    with pytest.raises(ValueError, match="ring depth does not match"):
        ReplaySSMModelContext.create(
            config,
            mamba_group_ids=[0],
            forward_context={"layer.0": layer},
            block_tables=[torch.zeros(2, 1, dtype=torch.int32)],
            max_num_reqs=2,
        )


def test_gdn_replayssm_warmup_runs_fixed_width_decode():
    layer = _gdn_layer(num_blocks=5)
    block_ids = np.arange(5, dtype=np.int32).reshape(5, 1)
    multi_group_block_table = SimpleNamespace(
        block_tables=[SimpleNamespace(block_table=SimpleNamespace(np=block_ids))],
        commit_block_table=Mock(),
    )
    dummy_run = Mock()
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            is_gdn_replayssm_enabled=lambda: True,
            num_speculative_tokens=3,
            use_v2_model_runner=False,
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        max_num_tokens=16,
        cudagraph_batch_sizes=[4, 8, 16],
        kv_cache_config=SimpleNamespace(num_blocks=10),
        input_batch=SimpleNamespace(block_table=multi_group_block_table),
        get_model=lambda: SimpleNamespace(modules=lambda: (layer,)),
        _dummy_run=dummy_run,
    )

    gdn_replayssm_warmup(runner)

    assert dummy_run.call_args_list == [
        call(
            num_tokens=num_tokens,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            allow_microbatching=False,
            force_attention=True,
            profile_seq_lens=5,
        )
        for num_tokens in (4, 8, 16)
    ]
    assert multi_group_block_table.commit_block_table.call_count == 2
    assert block_ids[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_gdn_replayssm_warmup_uses_v2_capture_descriptors():
    layer = _gdn_layer(num_blocks=5)
    dummy_run = Mock()
    block_tables = SimpleNamespace(get_dummy_block_tables=Mock())
    descriptors = [
        SimpleNamespace(uniform_token_count=4, num_reqs=num_reqs)
        for num_reqs in (1, 2, 4)
    ]
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            is_gdn_replayssm_enabled=lambda: True,
            num_speculative_tokens=3,
            use_v2_model_runner=True,
            cache_config=SimpleNamespace(mamba_cache_mode="none"),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        max_num_tokens=16,
        cudagraph_manager=SimpleNamespace(_capture_descs={"full": descriptors}),
        kv_cache_config=SimpleNamespace(num_blocks=10),
        block_tables=block_tables,
        get_model=lambda: SimpleNamespace(modules=lambda: (layer,)),
        _dummy_run=dummy_run,
    )

    gdn_replayssm_warmup(runner)

    assert dummy_run.call_args_list == [
        call(
            num_tokens=num_tokens,
            uniform_decode=True,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            valid_dummy_state_slots=True,
        )
        for num_tokens in (4, 8, 16)
    ]
    block_tables.get_dummy_block_tables.assert_called_once_with(4)


def test_gdn_replayssm_warmup_compiles_prefix_materializer_once_per_shape(
    monkeypatch,
):
    layers = (
        _gdn_layer(num_blocks=5, executed_query_width=1, ring_slots=16),
        _gdn_layer(num_blocks=5, executed_query_width=1, ring_slots=16),
    )
    block_ids = np.arange(5, dtype=np.int32).reshape(5, 1)
    block_table = SimpleNamespace(
        block_tables=[SimpleNamespace(block_table=SimpleNamespace(np=block_ids))],
        commit_block_table=Mock(),
    )
    materialize = Mock()
    monkeypatch.setattr(
        "vllm.model_executor.warmup.replayssm_warmup._load_gdn_replayssm_materialize",
        lambda: materialize,
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            is_gdn_replayssm_enabled=lambda: True,
            num_speculative_tokens=0,
            use_v2_model_runner=False,
            cache_config=SimpleNamespace(mamba_cache_mode="align"),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        max_num_tokens=2,
        cudagraph_batch_sizes=[1, 2],
        kv_cache_config=SimpleNamespace(num_blocks=5),
        input_batch=SimpleNamespace(block_table=block_table),
        get_model=lambda: SimpleNamespace(modules=lambda: layers),
        _dummy_run=Mock(),
    )

    gdn_replayssm_warmup(runner)

    materialize.assert_called_once()
    kwargs = materialize.call_args.kwargs
    assert kwargs["state"] is layers[0].kv_cache[1]
    assert kwargs["u_cache"] is layers[0].replayssm_cache[0]
    assert kwargs["k_cache"] is layers[0].replayssm_cache[1]
    assert kwargs["g_cache"] is layers[0].replayssm_cache[2]
    assert kwargs["src_slots"].tolist() == [-1, -1]
    assert kwargs["dst_slots"].tolist() == [-1, -1]
    assert kwargs["count"].tolist() == [-1, -1]
    assert kwargs["active_request_indices"].tolist() == [-1, -1]
    assert kwargs["num_active"].tolist() == [0]
