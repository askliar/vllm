# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import vllm.utils.gpu_sync_debug as gsd
from tests.v1.attention.test_gdn_metadata_builder import (
    BLOCK_SIZE,
    DEVICE,
    _create_gdn_builder,
)
from tests.v1.attention.utils import BatchSpec, create_common_attn_metadata
from tests.v1.worker.test_gpu_batch_ordering import _make_runner
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import (
    _postprocess_replayssm_kernel,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import (
    MambaHybridModelState,
)
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState


@pytest.mark.parametrize(
    ("use_flashinfer_replayssm", "expected_state_idx"), [(False, 2), (True, 2)]
)
def test_add_request_seeds_state_with_scoped_block_size(
    use_flashinfer_replayssm: bool, expected_state_idx: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.rope_state = None
    state.prompt_embeds_state = None
    state.cache_config = SimpleNamespace(
        block_size=16,
        mamba_block_size=8,
        mamba_cache_mode="align",
    )
    state._align_mode = True
    state._use_flashinfer_replayssm = use_flashinfer_replayssm
    state.num_accepted_tokens_gpu = torch.full((2,), 9, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.full((2,), -1, dtype=torch.int32)

    state.add_request(1, Mock(num_computed_tokens=17))

    assert state.num_accepted_tokens_gpu.tolist() == [9, 1]
    assert state._mamba_state_idx_gpu.tolist() == [-1, expected_state_idx]


@pytest.mark.parametrize(
    (
        "num_spec",
        "computed",
        "scheduled",
        "drafts",
        "prefilling",
        "expected_prefilling",
        "use_gdn_replayssm",
        "prompt_len",
    ),
    [
        (3, 256, 4, 3, True, False, False, 257),
        (3, 1, 4, 3, True, False, False, 2),
        (3, 256, 1, 0, True, False, False, 257),
        (3, 0, 4, 3, True, True, False, 1),
        (3, 256, 4, 0, True, True, False, 260),
        (3, 256, 4, 3, False, False, False, 256),
        (3, 256, 4, 3, True, False, True, 257),
        (3, 1, 4, 3, True, False, True, 2),
        (3, 256, 1, 0, True, False, True, 257),
        (3, 0, 4, 3, True, True, True, 1),
        (3, 256, 4, 0, True, True, True, 260),
        (3, 256, 4, 3, False, False, True, 256),
        (5, 256, 6, 5, False, False, True, 256),
        (3, 256, 8, 0, False, True, True, 256),
        (0, 256, 1, 0, True, False, True, 257),
        (0, 256, 2, 0, False, True, True, 256),
        # Mamba2 supports widths beyond GDN's fixed T=4/8 kernels.
        (4, 256, 5, 4, False, False, False, 256),
        (12, 256, 13, 12, False, False, False, 256),
        # Consecutive intermediate chunks follow each model's forward routing.
        (0, 64, 1, 0, True, False, False, 67),
        (0, 65, 1, 0, True, False, False, 67),
        (0, 66, 1, 0, True, False, False, 67),
        (0, 64, 1, 0, True, True, True, 67),
        (0, 65, 1, 0, True, True, True, 67),
        (0, 66, 1, 0, True, False, True, 67),
        (0, 64, 1, 0, True, True, True, 68),
        (0, 65, 3, 0, True, True, True, 68),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="Requires CUDA"
            ),
        ),
    ],
)
def test_prepare_attn_forwards_positions_and_stages_replayssm_prefill(
    monkeypatch: pytest.MonkeyPatch,
    num_spec,
    computed,
    scheduled,
    drafts,
    prefilling,
    expected_prefilling,
    use_gdn_replayssm,
    prompt_len,
    device,
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=num_spec)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state._use_flashinfer_replayssm = True
    state._use_gdn_replayssm = use_gdn_replayssm
    state._mamba_ctx = None
    if device == "cpu":
        monkeypatch.setattr("vllm.utils.torch_utils.PIN_MEMORY", False)
    state._is_prefilling_gpu = torch.zeros(1, dtype=torch.bool, device=device)
    state.num_accepted_tokens_gpu = torch.ones(1, dtype=torch.int32)
    state._get_mamba_group_info = Mock(return_value=([], None))
    state.recoverssm = None

    runner = _make_runner(
        {"req": (computed, prompt_len)}, num_spec + 1, use_gdn_replayssm
    )
    batch_state, _ = runner.gather_batch_req_state(
        SimpleNamespace(
            num_scheduled_tokens={"req": scheduled},
            total_num_scheduled_tokens=scheduled,
            scheduled_spec_decode_tokens={"req": [-1] * drafts} if drafts else {},
        ),
        False,
    )
    positions = torch.arange(computed, computed + scheduled, dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=scheduled,
        num_reqs_after_padding=1,
        num_tokens_after_padding=scheduled,
        max_query_len=None,
        prefill_runs_as_decode_np=batch_state.prefill_runs_as_decode_np,
        query_start_loc_np=torch.tensor([0, scheduled], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, scheduled], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([scheduled], dtype=torch.int32).numpy(),
        num_draft_tokens_per_req=torch.tensor([drafts], dtype=torch.int32).numpy(),
        idx_mapping=torch.tensor([0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([computed + scheduled]),
        seq_lens=torch.tensor([computed + scheduled]),
        is_prefilling_np=torch.tensor([prefilling]).numpy(),
        num_computed_prefill_tokens_np=torch.tensor([computed]).numpy(),
        prefill_len_np=torch.tensor([prompt_len]).numpy(),
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    prepare_attn = state.prepare_attn
    if device == "cuda":
        monkeypatch.setattr(gsd, "_SYNC_CHECK_MODE", "error")
        monkeypatch.setattr(gsd, "_sync_check_enabled", True)
        gsd.enable_gpu_sync_check()
        prepare_attn = gsd.with_gpu_sync_check(prepare_attn)

    metadata = prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions
    assert state._is_prefilling_gpu.item() == expected_prefilling

    if device == "cuda" and not use_gdn_replayssm:
        # Exercise the actual tracker publisher over partial acceptances. A
        # false prefill classification would erase the accepted history here.
        def ints(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        starts = ints([0, 0])
        committed = ints([0, 0])
        scratch = ints([0])
        accepted_total = 0
        for accepted in (1, min(2, scheduled), 1):
            _postprocess_replayssm_kernel[(1,)](
                None,
                ints([scheduled]),
                ints([computed + accepted_total]),
                ints([accepted]),
                state._is_prefilling_gpu,
                None,
                ints([[1]]),
                starts,
                committed,
                scratch,
                scratch,
                ints([0]),
                ints([-1]),
                1,
                1,
                MAMBA_BLOCK_SIZE=64,
                LOGICAL_WINDOW=16,
                RING_BUFFER_LEN=32,
                NUM_LAYERS=1,
                PAD_SLOT_ID=0,
                QUERY_METADATA_IS_CUMULATIVE=False,
                NUM_COMPUTED_IS_POST_STEP=False,
                HAS_IDX_MAPPING=False,
                MATERIALIZE_PREFIXES=False,
                LIVE_COL_IS_ZERO=True,
                EXECUTED_QUERY_WIDTH=0,
                GDN_STP=False,
            )
            accepted_total += accepted
            assert starts[1].item() == 0
            assert committed[1].item() == (0 if expected_prefilling else accepted_total)


def test_add_request_seeds_state_idx_in_mamba_blocks() -> None:
    """A drafter group can lower cache_config.block_size below the mamba
    block size; the seeded column must still be in mamba blocks."""
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(block_size=16, mamba_block_size=880)
    state._align_mode = True
    state.rope_state = None
    state.prompt_embeds_state = None
    state.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)

    state.add_request(1, SimpleNamespace(num_computed_tokens=107_360))

    assert state._mamba_state_idx_gpu[1] == 121


def test_padded_prompt_tail_builds_as_spec_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-token prompt tail over prior state, padded with K placeholder
    drafts (e.g. a P/D decode-node arrival), must reach the GDN builder as a
    spec-decode row. Built as a prefill, the placeholder tokens are folded into
    the recurrent state and can't be rolled back.
    """
    k = 3
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=k)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state._use_flashinfer_replayssm = False
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32)

    # A verify decode, a padded prompt tail (128 of 129 prompt tokens already
    # computed), and a fresh prompt chunk of the same length.
    query_lens = [k + 1] * 3
    seq_lens = [50, 128 + k + 1, k + 1]
    is_prefilling = [False, True, True]
    query_start_loc = np.array([0, 4, 8, 12], dtype=np.int32)
    input_batch = SimpleNamespace(
        num_reqs=3,
        num_tokens=12,
        num_reqs_after_padding=3,
        num_tokens_after_padding=12,
        idx_mapping=torch.arange(3),
        query_start_loc_np=query_start_loc,
        query_start_loc=torch.from_numpy(query_start_loc),
        num_scheduled_tokens=np.array(query_lens, dtype=np.int32),
        num_draft_tokens_per_req=np.array([k, k, 0], dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        is_prefilling_np=np.array(is_prefilling),
        num_computed_prefill_tokens_np=np.array([46, 128, 0]),
        prefill_len_np=np.array([40, 129, 8]),
        prefill_runs_as_decode_np=np.array([False, True, False]),
        dcp_local_seq_lens=None,
        positions=torch.zeros(12, dtype=torch.int64),
        prompt_lens=None,
    )
    build_attn_metadata = Mock(return_value={})
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    mamba_metadata = build_attn_metadata.call_args.kwargs[
        "model_specific_attn_metadata"
    ]
    assert mamba_metadata.is_prefilling.tolist() == [False, False, True]

    builder = _create_gdn_builder(num_speculative_tokens=k)
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens), BLOCK_SIZE, DEVICE
    ).replace(**mamba_metadata.get_extra_common_attn_kwargs(0, 3))
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common,
        num_accepted_tokens=mamba_metadata.num_accepted_tokens,
        num_decode_draft_tokens_cpu=mamba_metadata.num_decode_draft_tokens_cpu,
    )

    # Only the fresh prompt chunk needs the prefill kernels.
    assert meta.num_spec_decodes == 2
    assert meta.num_prefills == 1
    assert meta.num_prefill_tokens == k + 1


def _mamba_hybrid_state(num_speculative_tokens: int) -> MambaHybridModelState:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=num_speculative_tokens)
    state.model_config = SimpleNamespace(max_model_len=8192)
    state._align_mode = False
    state._use_flashinfer_replayssm = False
    state.recoverssm = None
    # A request that accepted 3 drafts (plus the bonus token) on the previous
    # step; the others have never had a draft accepted.
    state.num_accepted_tokens_gpu = torch.tensor([4, 1, 1], dtype=torch.int32)
    return state


def _input_batch(
    num_scheduled_tokens: list[int],
    num_draft_tokens_per_req: list[int] | None,
    is_prefilling: list[bool],
    num_reqs_after_padding: int | None = None,
) -> SimpleNamespace:
    """`num_draft_tokens_per_req=None` is a batch where nobody proposed a
    draft this step, e.g. because ngram found no match for anyone."""
    num_reqs = len(num_scheduled_tokens)
    query_start_loc = np.zeros(num_reqs + 1, dtype=np.int32)
    query_start_loc[1:] = np.cumsum(num_scheduled_tokens)
    num_tokens = int(query_start_loc[-1])
    seq_lens = torch.full((num_reqs,), 10, dtype=torch.int32)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs_after_padding or num_reqs,
        num_tokens=num_tokens,
        num_tokens_after_padding=num_tokens,
        query_start_loc_np=query_start_loc,
        num_scheduled_tokens=np.array(num_scheduled_tokens, dtype=np.int32),
        max_query_len=None,
        seq_lens_cpu_upper_bound=seq_lens,
        is_prefilling_np=np.array(is_prefilling),
        num_computed_prefill_tokens_np=np.where(is_prefilling, 0, 10),
        prefill_len_np=np.where(is_prefilling, 16, 10),
        prefill_runs_as_decode_np=None,
        num_draft_tokens_per_req=None
        if num_draft_tokens_per_req is None
        else np.array(num_draft_tokens_per_req, dtype=np.int32),
        idx_mapping=torch.arange(num_reqs),
        query_start_loc=torch.from_numpy(query_start_loc),
        seq_lens=seq_lens,
        dcp_local_seq_lens=None,
        positions=torch.zeros(num_tokens, dtype=torch.int64),
        prompt_lens=None,
    )


def _prepare_attn_metadata(
    state: MambaHybridModelState,
    monkeypatch,
    input_batch: SimpleNamespace | None = None,
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
) -> SimpleNamespace:
    captured = {}

    def fake_build_attn_metadata(**kwargs):
        captured.update(kwargs)
        return Mock()

    monkeypatch.setattr(
        "vllm.v1.worker.gpu.model_states.mamba_hybrid.build_attn_metadata",
        fake_build_attn_metadata,
    )

    if input_batch is None:
        input_batch = _input_batch([1, 1], None, [False, False])
    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=cudagraph_mode,
        block_tables=(),
        slot_mappings={},
        attn_groups=[],
        kv_cache_config=Mock(),
    )
    return captured["model_specific_attn_metadata"]


def test_prepare_attn_keeps_the_accepted_offset_when_the_batch_drafted_nothing(
    monkeypatch,
):
    """`num_accepted_tokens` must reach the attention builders whenever the
    speculative config is on, even when *this* batch scheduled no draft
    tokens at all. Before the fix, `use_spec_decode` was gated on this step's
    scheduled drafts (`len(scheduled_spec_decode_tokens) > 0` in the V1
    runner) rather than on the speculative config, so a request that accepted
    drafts on a previous step would silently lose that offset the moment
    ngram found no match for anyone in the batch.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is not None
    assert metadata.num_accepted_tokens.tolist() == [4, 1]
    assert metadata.num_decode_draft_tokens_cpu is not None


def test_prepare_attn_omits_the_offset_without_a_speculative_config(monkeypatch):
    """Mirrors the check above: with speculative decoding off entirely, there
    is no accepted-token offset to carry, and the metadata must say so
    explicitly (`None`) rather than a stale or default value."""
    state = _mamba_hybrid_state(num_speculative_tokens=0)

    metadata = _prepare_attn_metadata(state, monkeypatch)

    assert metadata.num_accepted_tokens is None
    assert metadata.num_decode_draft_tokens_cpu is None


def test_zero_draft_decode_row_is_marked_as_a_speculative_row(monkeypatch):
    """A decode row that got no drafts still has to reach the speculative path.

    The recurrent backends only apply a row's accepted-token offset on that
    path, and a row that accepted drafts on the previous step keeps that offset
    even when the current step proposes nothing (the scheduler drops drafts
    whenever the token budget truncates the step). Marking such a row with the
    -1 sentinel sends it to the plain decode path, which reads the state at
    column 0 and ignores the offset.
    """
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([3, 1, 1], [2, 0, 0], [False, False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [
        2,  # decode row with drafts
        0,  # decode row with no drafts: speculative path, offset applied
        -1,  # one-token prefill tail: no offset to apply, stays a plain row
    ]


def test_batch_without_any_drafts_still_marks_its_decode_rows(monkeypatch):
    """`num_draft_tokens_per_req` is None whenever no row in the batch drafted
    anything. Those rows are ordinary decode rows carrying an offset from the
    previous step, so the absence of drafts in this step must not be read as
    speculative decoding being off."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], None, [False, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, -1]


def test_padded_rows_keep_the_sentinel(monkeypatch):
    """Full CUDA-graph capture pads the batch; padded rows describe no request
    and must not be picked up as speculative rows."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([1, 1], [0, 0], [False, False], num_reqs_after_padding=4)

    metadata = _prepare_attn_metadata(
        state, monkeypatch, input_batch, cudagraph_mode=CUDAGraphMode.FULL
    )

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [0, 0, -1, -1]


def test_rows_wider_than_the_speculative_state_window_keep_the_sentinel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _mamba_hybrid_state(num_speculative_tokens=7)
    input_batch = _input_batch([12, 8, 1], None, [False, False, False])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, 0, 0]


def test_chunked_prefill_tail_of_two_or_three_tokens_keeps_the_sentinel(
    monkeypatch,
):
    """A prefill continuation is excluded by `is_prefilling`, not by its token
    count, so tails wider than one token are covered by the same rule."""
    state = _mamba_hybrid_state(num_speculative_tokens=3)
    input_batch = _input_batch([2, 3], [0, 0], [True, True])

    metadata = _prepare_attn_metadata(state, monkeypatch, input_batch)

    assert metadata.num_decode_draft_tokens_cpu.tolist() == [-1, -1]


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state._use_flashinfer_replayssm = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_flashinfer_replayssm_prefix_uses_original_accepted_counts() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._use_flashinfer_replayssm = True
    state.recoverssm = None
    state.num_accepted_tokens_gpu = torch.ones(4, dtype=torch.int32, device="cuda")
    state._mamba_state_idx_gpu = torch.zeros(4, dtype=torch.int32, device="cuda")
    state._is_prefilling_gpu = torch.zeros(4, dtype=torch.bool, device="cuda")
    state._replayssm_query_start_loc = torch.tensor(
        [0, 4], dtype=torch.int32, device="cuda"
    )
    replayssm = Mock()
    accepted_snapshot = torch.zeros(4, dtype=torch.int32, device="cuda")
    ctx = Mock(
        is_initialized=True,
        replayssm=replayssm,
        num_accepted_tokens_snapshot=accepted_snapshot,
    )

    def normalize_live(*_args) -> None:
        accepted_snapshot.copy_(state.num_accepted_tokens_gpu)
        state.num_accepted_tokens_gpu[2] = 1

    ctx.run_fused_postprocess_align.side_effect = normalize_live
    state._mamba_ctx = ctx

    state.postprocess_state(
        torch.tensor([2], dtype=torch.int32, device="cuda"),
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([0, 0, 8, 0], device="cuda"),
    )

    kwargs = replayssm.postprocess.call_args.kwargs
    assert kwargs["num_accepted_tokens"] is accepted_snapshot
    assert accepted_snapshot[2].item() == 3
    assert state.num_accepted_tokens_gpu[2].item() == 1
    assert state._replayssm_query_start_loc is None


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("computed", "sampled", "expected_column"),
    [(4, 3, 0), (4, 4, 0), (4, 5, 1), (8, 8, 1), (8, 9, 2)],
)
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias(
    computed: int, sampled: int, expected_column: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._use_flashinfer_replayssm = False
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor(
            [6, computed], dtype=torch.int32, device="cuda"
        ),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, sampled], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, expected_column, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
