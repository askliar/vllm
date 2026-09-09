# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import torch

from vllm.model_executor.layers.mamba.mamba_mixer2 import (
    share_replayssm_ring_trackers,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import ReplaySSMModelContext
from vllm.model_executor.warmup.replayssm_warmup import gdn_replayssm_warmup
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec


def _gdn_layer(num_blocks: int = 3):
    return SimpleNamespace(
        use_flashinfer_replayssm=True,
        replayssm_buffer_len=16,
        replayssm_executed_query_width=4,
        kv_cache=(
            torch.zeros(num_blocks, 3, 384, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 1, 128, 128, dtype=torch.bfloat16),
        ),
        replayssm_cache=(
            torch.zeros(num_blocks, 1, 32, 128, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 1, 32, 128, dtype=torch.bfloat16),
            torch.zeros(num_blocks, 1, 32, dtype=torch.float32),
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
