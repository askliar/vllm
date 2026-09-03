# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.worker.utils import (
    bind_kv_cache,
    copy_kv_cache_blocks_inplace,
    get_replayssm_block_copy_tensors,
)


class _TestReplaySSMMixer(MambaMixer2):
    _state_shapes = ((2,), (3,))
    _state_dtypes = (torch.float32, torch.float32)

    def __init__(self, backend: MambaBackendEnum = MambaBackendEnum.FLASHINFER) -> None:
        torch.nn.Module.__init__(self)
        self.use_replayssm = True
        self.mamba_config = MambaConfig(backend=backend)
        self._replayssm_ring_start = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32)

    def get_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return self._state_shapes

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return self._state_dtypes

    def get_replayssm_state_shape(self) -> tuple[tuple[int, ...], ...]:
        return ((4,), (5,), (6,))

    def get_replayssm_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (torch.float32,) * 3


def _packed_replayssm_cache(num_blocks: int, fill_value: int = 0) -> torch.Tensor:
    return torch.full((num_blocks, 1, 1, 20), fill_value, dtype=torch.int8)


def test_bind_kv_cache_shares_replayssm_trackers_by_cache_group():
    mixers = [_TestReplaySSMMixer() for _ in range(3)]
    layer_names = [f"layers.{i}.mixer" for i in range(3)]
    ctx = dict(zip(layer_names, mixers))
    # Reverse insertion order: updater must follow layer index, not dict order.
    kv_cache = {
        layer_names[2]: _packed_replayssm_cache(4),
        layer_names[1]: _packed_replayssm_cache(4),
        layer_names[0]: _packed_replayssm_cache(4),
    }
    kv_cache_groups = [
        SimpleNamespace(layer_names=[layer_names[0], layer_names[2]]),
        SimpleNamespace(layer_names=[layer_names[1]]),
    ]
    replayssm_caches = {
        name: [
            torch.zeros((4, *shape), dtype=torch.float32)
            for shape in mixer.get_replayssm_state_shape()
        ]
        for name, mixer in ctx.items()
    }

    bind_kv_cache(
        kv_cache,
        ctx,
        [],
        kv_cache_groups=kv_cache_groups,
        replayssm_caches={
            name: tuple(cache) for name, cache in replayssm_caches.items()
        },
    )

    assert all(len(mixer.kv_cache) == 5 for mixer in mixers)

    tracker_names = (
        "_replayssm_ring_start",
        "_replayssm_prev_num_accepted",
    )
    for tracker_name in tracker_names:
        group_tracker = getattr(mixers[0], tracker_name)
        assert group_tracker.data_ptr() == getattr(mixers[2], tracker_name).data_ptr()
        assert group_tracker.data_ptr() != getattr(mixers[1], tracker_name).data_ptr()
        assert group_tracker.shape == (4,)
        assert group_tracker.dtype == torch.int32
        assert torch.count_nonzero(group_tracker) == 0


def test_replayssm_block_copy_includes_rings_and_group_trackers(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.worker.utils.async_tensor_h2d",
        lambda array, *, device, **_: torch.from_numpy(array).to(device),
    )
    mixers = [_TestReplaySSMMixer() for _ in range(3)]
    layer_names = [f"layers.{i}.mixer" for i in range(3)]
    ctx = dict(zip(layer_names, mixers))
    kv_cache = {name: _packed_replayssm_cache(4) for name in layer_names}
    kv_cache_groups = [
        SimpleNamespace(layer_names=[layer_names[0], layer_names[2]]),
        SimpleNamespace(layer_names=[layer_names[1]]),
    ]
    replayssm_caches = {
        name: tuple(
            torch.zeros((4, *shape), dtype=torch.float32)
            for shape in mixer.get_replayssm_state_shape()
        )
        for name, mixer in ctx.items()
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(
        kv_cache,
        ctx,
        runner_kv_caches,
        kv_cache_groups=kv_cache_groups,
        replayssm_caches=replayssm_caches,
    )

    src, dst = 1, 2
    for layer_idx, mixer in enumerate(mixers):
        for state_idx, state in enumerate(mixer.kv_cache):
            state[src].fill_(10 * layer_idx + state_idx + 1)
            state[dst].fill_(-1)
    for group_idx, mixer in enumerate(mixers[:2]):
        mixer._replayssm_ring_start[src] = 20 + group_idx
        mixer._replayssm_prev_num_accepted[src] = 30 + group_idx

    copy_kv_cache_blocks_inplace(
        [*runner_kv_caches, *get_replayssm_block_copy_tensors(ctx)],
        4,
        [KVCacheBlockCopy(src, dst)],
    )

    for mixer in mixers:
        for state in mixer.kv_cache:
            torch.testing.assert_close(state[dst], state[src])
    assert mixers[0]._replayssm_ring_start[dst].item() == 20
    assert mixers[0]._replayssm_prev_num_accepted[dst].item() == 30
    assert mixers[1]._replayssm_ring_start[dst].item() == 21
    assert mixers[1]._replayssm_prev_num_accepted[dst].item() == 31


def test_replayssm_block_copy_validates_exact_cache_roles():
    mixer = _TestReplaySSMMixer()
    mixer.kv_cache = tuple(torch.zeros(4, 1) for _ in range(4))

    with pytest.raises(ValueError, match="exactly 5 cache roles"):
        get_replayssm_block_copy_tensors({"layers.0.mixer": mixer})


def test_replayssm_block_copy_includes_triton_rings_without_trackers():
    mixer = _TestReplaySSMMixer(MambaBackendEnum.TRITON)
    mixer.kv_cache = tuple(torch.zeros(4, 1) for _ in range(5))

    tensors = get_replayssm_block_copy_tensors({"layers.0.mixer": mixer})

    assert len(tensors) == 3
    assert all(
        actual is expected
        for actual, expected in zip(tensors, mixer.kv_cache[2:5], strict=True)
    )


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]
