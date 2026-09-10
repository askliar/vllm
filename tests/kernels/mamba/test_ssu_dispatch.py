# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.config.mamba import MambaBackendEnum, MambaConfig, MambaSSUAlgorithm
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    FlashInferSSUBackend,
    TritonSSUBackend,
    _postprocess_replayssm_kernel,
    get_mamba_ssu_backend,
    initialize_mamba_ssu_backend,
    selective_state_update,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

try:
    import flashinfer.mamba  # noqa: F401

    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False


@pytest.fixture(autouse=True)
def restore_backend_state():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old_backend = mod._mamba_ssu_backend
    old_replayssm_kernel = mod._flashinfer_replayssm_kernel
    yield
    mod._mamba_ssu_backend = old_backend
    mod._flashinfer_replayssm_kernel = old_replayssm_kernel


def _kv_cache_config_with_ssu(
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.MAMBA2,
) -> KVCacheConfig:
    spec = MambaSpec(
        block_size=16,
        shapes=((16, 64),),
        dtypes=(torch.float16,),
        mamba_type=mamba_type,
    )
    return KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
    )


def test_default_backend_is_triton():
    initialize_mamba_ssu_backend(MambaConfig(), _kv_cache_config_with_ssu())
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)
    assert backend.name == "triton"


def test_explicit_triton_backend():
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), _kv_cache_config_with_ssu()
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, TritonSSUBackend)


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
def test_flashinfer_backend_init():
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.FLASHINFER), _kv_cache_config_with_ssu()
    )
    backend = get_mamba_ssu_backend()
    assert isinstance(backend, FlashInferSSUBackend)
    assert backend.name == "flashinfer"


@pytest.mark.skipif(not HAS_FLASHINFER, reason="flashinfer not installed")
@pytest.mark.parametrize(
    ("algorithm", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("simple", "simple"),
        ("vertical", "vertical"),
        ("horizontal", "horizontal"),
    ],
)
def test_flashinfer_forwards_ssu_algorithm(
    algorithm: MambaSSUAlgorithm | None,
    expected: MambaSSUAlgorithm,
    monkeypatch,
):
    import flashinfer.mamba

    kernel = Mock()
    monkeypatch.setattr(flashinfer.mamba, "selective_state_update", kernel)
    backend = FlashInferSSUBackend(
        MambaConfig(
            backend=MambaBackendEnum.FLASHINFER,
            ssu_algorithm=algorithm,
        )
    )

    tensor = torch.empty(1)
    backend(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
    )

    assert kernel.call_args.kwargs["algorithm"] == expected


def test_uninitialized_backend_raises():
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    # restore_backend_state (autouse) puts the global back afterwards.
    mod._mamba_ssu_backend = None
    with pytest.raises(RuntimeError, match="not been initialized"):
        get_mamba_ssu_backend()


@pytest.mark.parametrize(
    "mamba_type",
    [
        MambaAttentionBackendEnum.LINEAR,
        MambaAttentionBackendEnum.GDN_ATTN,
        MambaAttentionBackendEnum.SHORT_CONV,
    ],
)
def test_init_is_noop_for_non_ssu_mamba_type(mamba_type):
    import vllm.model_executor.layers.mamba.ops.ssu_dispatch as mod

    old = mod._mamba_ssu_backend
    mod._mamba_ssu_backend = None
    try:
        initialize_mamba_ssu_backend(
            MambaConfig(), _kv_cache_config_with_ssu(mamba_type)
        )
        assert mod._mamba_ssu_backend is None
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_mamba_ssu_backend()
    finally:
        mod._mamba_ssu_backend = old


@pytest.mark.skipif(HAS_FLASHINFER, reason="flashinfer is installed")
def test_flashinfer_import_error():
    with pytest.raises(ImportError, match="FlashInfer is required"):
        FlashInferSSUBackend(MambaConfig())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_triton_basic_call():
    set_random_seed(0)
    initialize_mamba_ssu_backend(
        MambaConfig(backend=MambaBackendEnum.TRITON), _kv_cache_config_with_ssu()
    )
    device = "cuda"
    batch_size = 2
    dim = 64
    dstate = 16

    state = torch.randn(batch_size, dim, dstate, device=device)
    x = torch.randn(batch_size, dim, device=device)
    out = torch.empty_like(x)
    dt = torch.randn(batch_size, dim, device=device)
    dt_bias = torch.rand(dim, device=device) - 4.0
    A = -torch.rand(dim, dstate, device=device)
    B = torch.randn(batch_size, dstate, device=device)
    C = torch.randn(batch_size, dstate, device=device)
    D = torch.randn(dim, device=device)

    selective_state_update(
        state,
        x,
        dt,
        A,
        B,
        C,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
        out=out,
    )
    assert not torch.isnan(out).any()


@pytest.mark.parametrize(
    ("backend", "num_speculative_tokens", "expected_ring_len"),
    [
        (MambaBackendEnum.TRITON, 0, 16),
        (MambaBackendEnum.FLASHINFER, 0, 17),
        (MambaBackendEnum.FLASHINFER, 3, 20),
    ],
)
def test_replayssm_physical_ring_shape(
    backend, num_speculative_tokens, expected_ring_len
):
    shapes = MambaStateShapeCalculator.replayssm_ring_shapes(
        num_heads=16,
        head_dim=4,
        state_size=16,
        n_groups=4,
        tp_world_size=2,
        logical_window=16,
        backend=backend,
        num_speculative_tokens=num_speculative_tokens,
    )

    assert shapes == (
        (8, expected_ring_len, 4),
        (8, expected_ring_len),
        (2, expected_ring_len, 16),
    )


@pytest.mark.parametrize(
    ("ring_slots", "expected"),
    [
        (16, ((4, 16, 128), (2, 16, 128), (4, 16))),
        (32, ((4, 32, 128), (2, 32, 128), (4, 32))),
    ],
)
def test_gdn_replayssm_physical_ring_layout(ring_slots, expected):
    shapes = MambaStateShapeCalculator.gated_delta_net_replayssm_ring_shapes(
        tp_world_size=2,
        num_k_heads=4,
        num_v_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        ring_slots=ring_slots,
    )
    dtypes = MambaStateDtypeCalculator.gated_delta_net_replayssm_ring_dtypes(
        torch.bfloat16
    )

    assert shapes == expected
    assert dtypes == (torch.bfloat16, torch.bfloat16, torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("post_step", [False, True])
@pytest.mark.parametrize(
    ("computed_before", "query_len", "prefilling", "accepted", "expected"),
    [
        (256, 4, False, 1, 3),  # Padded cached prompt tail commits its real token.
        (1, 4, False, 1, 3),  # Rejected placeholders exceed the cached prefix.
        (256, 1, False, 1, 3),
        (0, 4, True, 1, 0),  # Initial prefill clears stale trackers.
        (256, 4, True, 1, 0),  # Multi-token prefill also clears trackers.
        (256, 4, False, 3, 5),  # Ordinary speculative decode commits acceptance.
    ],
)
def test_replayssm_postprocess_commits_staged_transition(
    post_step, computed_before, query_len, prefilling, accepted, expected
):
    """The staged kernel path must preserve accepted history on prompt tails."""

    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    computed = computed_before
    if post_step:
        computed += query_len if prefilling else accepted
    ring_start = tensor([3])
    committed = tensor([2])
    plan_start, plan_flush = tensor([0]), tensor([-1])
    slots = tensor([[0]])
    _postprocess_replayssm_kernel[(1,)](
        tensor([0]),
        tensor([0, query_len]) if post_step else tensor([query_len]),
        tensor([computed]),
        tensor([accepted]),
        torch.tensor([prefilling], device="cuda"),
        None,
        tensor([[0, 0]]),
        ring_start,
        committed,
        slots,
        slots,
        plan_start,
        plan_flush,
        2,
        1,
        MAMBA_BLOCK_SIZE=256,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=20,
        NUM_LAYERS=1,
        PAD_SLOT_ID=-1,
        QUERY_METADATA_IS_CUMULATIVE=post_step,
        NUM_COMPUTED_IS_POST_STEP=post_step,
        HAS_IDX_MAPPING=post_step,
        MATERIALIZE_PREFIXES=False,
        LIVE_COL_IS_ZERO=True,
        EXECUTED_QUERY_WIDTH=0,
        GDN_STP=False,
    )
    assert committed.item() == expected
    assert ring_start.item() == (0 if prefilling else 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("executed_width", "old_committed", "expected_start"),
    [(4, 13, 16), (8, 9, 12)],
)
def test_gdn_replayssm_postprocess_uses_executed_width(
    executed_width, old_committed, expected_start
):
    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    ring_start = tensor([3])
    committed = tensor([old_committed])
    slots = tensor([[0]])
    _postprocess_replayssm_kernel[(1,)](
        tensor([0]),
        tensor([1]),
        tensor([256]),
        tensor([1]),
        torch.tensor([False], device="cuda"),
        None,
        slots,
        ring_start,
        committed,
        slots,
        slots,
        tensor([0]),
        tensor([-1]),
        1,
        1,
        MAMBA_BLOCK_SIZE=256,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=32,
        NUM_LAYERS=1,
        PAD_SLOT_ID=-1,
        QUERY_METADATA_IS_CUMULATIVE=False,
        NUM_COMPUTED_IS_POST_STEP=False,
        HAS_IDX_MAPPING=False,
        MATERIALIZE_PREFIXES=False,
        LIVE_COL_IS_ZERO=True,
        EXECUTED_QUERY_WIDTH=executed_width,
        GDN_STP=False,
    )
    assert ring_start.item() == expected_start
    assert committed.item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("old_committed", "expected_committed"),
    [(14, 15), (15, 0)],
)
def test_gdn_stp_postprocess_matches_fold_absorb_commit(
    old_committed, expected_committed
):
    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    ring_start = tensor([0, 0])
    committed = tensor([0, old_committed])
    src_slots = tensor([[0]])
    dst_slots = tensor([[0]])
    plan_start = tensor([-1])
    plan_count = tensor([-1])
    _postprocess_replayssm_kernel[(1,)](
        tensor([0]),
        tensor([1]),
        tensor([15]),
        tensor([1]),
        torch.tensor([False], device="cuda"),
        tensor([0]),
        tensor([[1, 1]]),
        ring_start,
        committed,
        src_slots,
        dst_slots,
        plan_start,
        plan_count,
        2,
        1,
        MAMBA_BLOCK_SIZE=16,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=16,
        NUM_LAYERS=1,
        PAD_SLOT_ID=-1,
        QUERY_METADATA_IS_CUMULATIVE=False,
        NUM_COMPUTED_IS_POST_STEP=False,
        HAS_IDX_MAPPING=False,
        MATERIALIZE_PREFIXES=False,
        LIVE_COL_IS_ZERO=False,
        EXECUTED_QUERY_WIDTH=1,
        GDN_STP=True,
    )
    assert ring_start[1].item() == 0
    assert committed[1].item() == expected_committed
    assert plan_start.item() == 0
    assert plan_count.item() == -1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("old_committed", "expected_count"),
    [(14, 15), (15, 0)],
)
def test_gdn_stp_prefix_plan_resets_canonical_destination(
    old_committed, expected_count
):
    def tensor(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    ring_start = tensor([0, 0])
    committed = tensor([0, old_committed])
    src_slots = tensor([[0]])
    dst_slots = tensor([[0]])
    plan_start = tensor([-1])
    plan_count = tensor([-1])
    _postprocess_replayssm_kernel[(1,)](
        tensor([0]),
        tensor([1]),
        tensor([15]),
        tensor([1]),
        torch.tensor([False], device="cuda"),
        tensor([0]),
        tensor([[1, 1]]),
        ring_start,
        committed,
        src_slots,
        dst_slots,
        plan_start,
        plan_count,
        2,
        1,
        MAMBA_BLOCK_SIZE=16,
        LOGICAL_WINDOW=16,
        RING_BUFFER_LEN=16,
        NUM_LAYERS=1,
        PAD_SLOT_ID=-1,
        QUERY_METADATA_IS_CUMULATIVE=False,
        NUM_COMPUTED_IS_POST_STEP=False,
        HAS_IDX_MAPPING=False,
        MATERIALIZE_PREFIXES=True,
        LIVE_COL_IS_ZERO=False,
        EXECUTED_QUERY_WIDTH=1,
        GDN_STP=True,
    )
    assert src_slots.item() == 1
    assert dst_slots.item() == 1
    assert ring_start[1].item() == 0
    assert committed[1].item() == 0
    assert plan_start.item() == 0
    assert plan_count.item() == expected_count
