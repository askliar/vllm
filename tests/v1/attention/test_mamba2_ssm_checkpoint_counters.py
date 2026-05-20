# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mamba2 SSM checkpointing per-cache-block counters.

These tests exercise the Phase-C bookkeeping that lives on
``Mamba2AttentionMetadataBuilder``:

* ``apply_post_step`` — mirrors FlashInfer's per-step ``must_checkpoint``
  decision and advances ``cache_buf_idx``/``prev_num_accepted_tokens``.
* ``zero_blocks`` — wipes counter slots for blocks the scheduler just
  handed out fresh (i.e. ``new_block_ids_to_zero``).

The tests construct a ``Mamba2AttentionMetadataBuilder`` shell via
``__new__`` and manually wire only the attributes the two methods touch.
No model forward, no GPU required.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder

DEVICE = torch.device("cpu")


# -----------------------------------------------------------------------------
# Builder construction (via __new__, no real vllm_config / model_config needed)
# -----------------------------------------------------------------------------


def _make_builder(
    *,
    L: int,
    num_blocks: int,
) -> Mamba2AttentionMetadataBuilder:
    """Build a minimal builder shell.

    Skips ``__init__`` so we don't need a real ``vllm_config`` with a Mamba
    model_config. The two methods under test only touch a handful of
    attributes; we set those explicitly.
    """
    builder = Mamba2AttentionMetadataBuilder.__new__(Mamba2AttentionMetadataBuilder)
    builder.device = DEVICE

    # Only ``mamba_ssm_checkpoint_interval`` is read by ``apply_post_step``.
    cache_config = MagicMock()
    cache_config.mamba_ssm_checkpoint_interval = L
    cache_config.num_gpu_blocks = num_blocks
    builder.vllm_config = MagicMock()
    builder.vllm_config.cache_config = cache_config

    # Counter buffers.
    if L > 1:
        builder.cache_buf_idx_d = torch.zeros(
            (num_blocks,), dtype=torch.int32, device=DEVICE
        )
        builder.prev_num_accepted_tokens_d = torch.zeros(
            (num_blocks,), dtype=torch.int32, device=DEVICE
        )
    else:
        builder.cache_buf_idx_d = None
        builder.prev_num_accepted_tokens_d = None

    # `_last_*` fields are set by build() before apply_post_step is called;
    # tests stamp them per call via _stage(...) below.
    builder._last_state_indices_tensor_d = None
    builder._last_query_start_loc_d = None
    builder._last_num_decodes = 0
    return builder


def _stage(
    builder: Mamba2AttentionMetadataBuilder,
    *,
    slots_col0: list[int],
    seq_lens: list[int],
    num_spec_tokens: int = 0,
    num_decodes: int | None = None,
    padded_bs: int | None = None,
    null_block_id: int = -1,
) -> None:
    """Stage the per-step inputs apply_post_step reads off the builder.

    ``slots_col0`` is the column-0 block ids for the live decode rows.
    ``seq_lens`` is the kernel input count per live decode row.
    ``padded_bs`` extends the persistent tensors with NULL-padded rows
    (simulating CUDA-graph padding).
    """
    assert len(slots_col0) == len(seq_lens)
    if num_decodes is None:
        num_decodes = len(slots_col0)
    bs = padded_bs if padded_bs is not None else num_decodes

    # state_indices_tensor_d shape: (bs, 1 + num_spec_tokens), col0 is live slot.
    state_indices = torch.full(
        (bs, 1 + num_spec_tokens),
        null_block_id,
        dtype=torch.int32,
        device=DEVICE,
    )
    for i, b in enumerate(slots_col0):
        state_indices[i, 0] = b
    # Spec scratch columns get arbitrary non-overflowing values for the live
    # rows; the SSU path never reads them but we keep them realistic.
    if num_spec_tokens > 0:
        for i in range(num_decodes):
            for j in range(1, 1 + num_spec_tokens):
                state_indices[i, j] = 0

    # query_start_loc_d is sized (bs + 1,) and padded past num_decodes with
    # the same final value (mirrors what mamba_attn.py does for cuda graphs).
    qsl = torch.zeros((bs + 1,), dtype=torch.int32, device=DEVICE)
    cum = 0
    for i, s in enumerate(seq_lens):
        cum += s
        qsl[i + 1] = cum
    for i in range(num_decodes + 1, bs + 1):
        qsl[i] = cum  # constant tail past live decodes -> seq_len = 0

    builder._last_state_indices_tensor_d = state_indices
    builder._last_query_start_loc_d = qsl
    builder._last_num_decodes = num_decodes


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------


@dataclass
class StepInputs:
    slots_col0: list[int]
    seq_lens: list[int]


def _run_steps(
    builder: Mamba2AttentionMetadataBuilder, steps: list[StepInputs]
) -> None:
    for s in steps:
        _stage(builder, slots_col0=s.slots_col0, seq_lens=s.seq_lens)
        builder.apply_post_step()


def test_normal_cycle_single_token_decode():
    """One request, single-token decode, runs past the overflow boundary.

    With L=4 and seq_len=1 per step, the counter should walk
    (0,0) -> (0,1) -> (0,2) -> (0,3) -> (0,4) -> (1,1) -> (1,2) ...
    """
    L = 4
    num_blocks = 8
    slot = 5
    builder = _make_builder(L=L, num_blocks=num_blocks)

    expected_after_step = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 1),  # overflow: 4 + 1 > 4 -> flip 0->1, prev=1
        (1, 2),
        (1, 3),
        (1, 4),
        (0, 1),  # overflow again: flip back to 0, prev=1
    ]

    for step, (exp_idx, exp_prev) in enumerate(expected_after_step):
        _stage(builder, slots_col0=[slot], seq_lens=[1])
        builder.apply_post_step()
        assert int(builder.cache_buf_idx_d[slot]) == exp_idx, (
            f"step {step}: idx mismatch"
        )
        assert int(builder.prev_num_accepted_tokens_d[slot]) == exp_prev, (
            f"step {step}: prev mismatch"
        )

    # Other blocks must be untouched.
    untouched = [b for b in range(num_blocks) if b != slot]
    assert torch.all(builder.cache_buf_idx_d[untouched] == 0)
    assert torch.all(builder.prev_num_accepted_tokens_d[untouched] == 0)


def test_admission_zero_via_zero_blocks():
    """zero_blocks() wipes the named counter slots; others untouched."""
    L = 8
    num_blocks = 8
    builder = _make_builder(L=L, num_blocks=num_blocks)

    # Pre-seed every slot to a non-zero value.
    builder.cache_buf_idx_d.fill_(1)
    builder.prev_num_accepted_tokens_d.fill_(5)

    builder.zero_blocks([2, 4])

    assert int(builder.cache_buf_idx_d[2]) == 0
    assert int(builder.cache_buf_idx_d[4]) == 0
    assert int(builder.prev_num_accepted_tokens_d[2]) == 0
    assert int(builder.prev_num_accepted_tokens_d[4]) == 0

    untouched = [0, 1, 3, 5, 6, 7]
    assert torch.all(builder.cache_buf_idx_d[untouched] == 1)
    assert torch.all(builder.prev_num_accepted_tokens_d[untouched] == 5)


def test_free_and_reuse_starts_from_zero():
    """A slot recycled through zero_blocks starts a fresh recurrence."""
    L = 4
    num_blocks = 4
    slot = 1
    builder = _make_builder(L=L, num_blocks=num_blocks)

    # Request A drives the slot past one overflow.
    _run_steps(
        builder,
        [
            StepInputs([slot], [1]),
            StepInputs([slot], [1]),
            StepInputs([slot], [1]),
            StepInputs([slot], [1]),
            StepInputs([slot], [1]),  # overflow -> (1, 1)
        ],
    )
    assert int(builder.cache_buf_idx_d[slot]) == 1
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 1

    # Block is freed and re-handed-out via new_block_ids_to_zero.
    builder.zero_blocks([slot])
    assert int(builder.cache_buf_idx_d[slot]) == 0
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 0

    # Request B on the same slot starts from (0, 0).
    _stage(builder, slots_col0=[slot], seq_lens=[1])
    builder.apply_post_step()
    assert int(builder.cache_buf_idx_d[slot]) == 0
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 1


def test_mtp_seq_len_greater_than_one_advances_by_kernel_input_count():
    """Under MTP, counter advances by seq_len (kernel input count)."""
    L = 8
    num_blocks = 4
    slot = 2
    builder = _make_builder(L=L, num_blocks=num_blocks)

    # seq_len=4 each step.
    _stage(builder, slots_col0=[slot], seq_lens=[4], num_spec_tokens=3)
    builder.apply_post_step()
    assert int(builder.cache_buf_idx_d[slot]) == 0
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 4

    # 4 + 4 = 8, equal to L, no overflow.
    _stage(builder, slots_col0=[slot], seq_lens=[4], num_spec_tokens=3)
    builder.apply_post_step()
    assert int(builder.cache_buf_idx_d[slot]) == 0
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 8

    # 8 + 3 = 11 > 8, overflow on a partial spec accept.
    _stage(builder, slots_col0=[slot], seq_lens=[3], num_spec_tokens=3)
    builder.apply_post_step()
    assert int(builder.cache_buf_idx_d[slot]) == 1  # flipped
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 3  # fresh buffer


def test_overflow_boundary_exact_equality_does_not_flip():
    """``prev_k + seq_len == L`` is NOT overflow (strict >)."""
    L = 4
    num_blocks = 2
    slot = 0
    builder = _make_builder(L=L, num_blocks=num_blocks)

    _stage(builder, slots_col0=[slot], seq_lens=[L])  # 0 + 4 = 4, not > 4
    builder.apply_post_step()
    assert int(builder.cache_buf_idx_d[slot]) == 0
    assert int(builder.prev_num_accepted_tokens_d[slot]) == L


def test_padded_rows_do_not_corrupt_other_blocks():
    """CUDA-graph padded rows (NULL_BLOCK_ID = -1 past num_decodes) must be
    excluded from the update. PyTorch's negative indexing would otherwise
    wrap -1 to block num_blocks-1 inside scatter_ and silently corrupt it.
    """
    L = 8
    num_blocks = 8
    padded_bs = 32  # persistent tensor is much larger than live num_decodes
    live_slots = [1, 3]
    builder = _make_builder(L=L, num_blocks=num_blocks)
    # Pre-seed the would-be-corrupted slot to something detectable.
    builder.cache_buf_idx_d[num_blocks - 1] = 1
    builder.prev_num_accepted_tokens_d[num_blocks - 1] = 7

    _stage(
        builder,
        slots_col0=live_slots,
        seq_lens=[1, 1],
        padded_bs=padded_bs,
        null_block_id=-1,
    )
    builder.apply_post_step()

    # Live slots advanced.
    assert int(builder.prev_num_accepted_tokens_d[live_slots[0]]) == 1
    assert int(builder.prev_num_accepted_tokens_d[live_slots[1]]) == 1
    # The slot that would have been wrapped-to by -1 must be untouched.
    assert int(builder.cache_buf_idx_d[num_blocks - 1]) == 1
    assert int(builder.prev_num_accepted_tokens_d[num_blocks - 1]) == 7


def test_mixed_prefill_decode_only_decode_rows_mutate():
    """Counters change only for cache blocks in state_indices_cpu[:num_decodes];
    blocks not in that slice stay put even if other batch rows ran in the
    forward (prefill side)."""
    L = 8
    num_blocks = 6
    builder = _make_builder(L=L, num_blocks=num_blocks)

    # 2 decode rows; persistent tensor has trailing padded entries
    # standing in for prefill / cuda-graph padding.
    _stage(
        builder,
        slots_col0=[2, 4],
        seq_lens=[1, 1],
        padded_bs=8,
        null_block_id=-1,
    )
    builder.apply_post_step()

    decoded = {2, 4}
    for b in range(num_blocks):
        if b in decoded:
            assert int(builder.prev_num_accepted_tokens_d[b]) == 1
        else:
            assert int(builder.cache_buf_idx_d[b]) == 0
            assert int(builder.prev_num_accepted_tokens_d[b]) == 0


def test_apply_post_step_noop_when_num_decodes_zero():
    """Prefill-only step (num_decodes=0) is a no-op."""
    L = 4
    num_blocks = 4
    builder = _make_builder(L=L, num_blocks=num_blocks)
    builder.prev_num_accepted_tokens_d.fill_(2)
    builder.cache_buf_idx_d.fill_(1)

    _stage(builder, slots_col0=[], seq_lens=[], num_decodes=0, padded_bs=4)
    builder.apply_post_step()

    assert torch.all(builder.cache_buf_idx_d == 1)
    assert torch.all(builder.prev_num_accepted_tokens_d == 2)


def test_apply_post_step_noop_when_checkpointing_disabled():
    """L=1 disables checkpointing; counter tensors are None and the method
    must short-circuit without errors."""
    builder = _make_builder(L=1, num_blocks=4)
    # Stage some bogus inputs to be sure they aren't read.
    builder._last_state_indices_tensor_d = torch.zeros(
        (1, 1), dtype=torch.int32, device=DEVICE
    )
    builder._last_query_start_loc_d = torch.zeros(
        (2,), dtype=torch.int32, device=DEVICE
    )
    builder._last_num_decodes = 1

    builder.apply_post_step()  # should not raise

    assert builder.cache_buf_idx_d is None
    assert builder.prev_num_accepted_tokens_d is None


def test_multiple_slots_one_step_independent_updates():
    """Multiple decode rows touching distinct slots update independently."""
    L = 4
    num_blocks = 8
    builder = _make_builder(L=L, num_blocks=num_blocks)

    # Pre-seed three slots at different occupancies so each takes a
    # different branch in the same step.
    builder.prev_num_accepted_tokens_d[1] = 0
    builder.prev_num_accepted_tokens_d[3] = 3
    builder.prev_num_accepted_tokens_d[5] = 4  # 4 + 1 > 4 overflows
    builder.cache_buf_idx_d[5] = 0

    _stage(builder, slots_col0=[1, 3, 5], seq_lens=[1, 1, 1])
    builder.apply_post_step()

    assert int(builder.prev_num_accepted_tokens_d[1]) == 1
    assert int(builder.cache_buf_idx_d[1]) == 0
    assert int(builder.prev_num_accepted_tokens_d[3]) == 4
    assert int(builder.cache_buf_idx_d[3]) == 0
    assert int(builder.prev_num_accepted_tokens_d[5]) == 1  # fresh buffer
    assert int(builder.cache_buf_idx_d[5]) == 1  # flipped


@pytest.mark.parametrize("L", [4, 8, 16])
def test_overflow_resets_prev_to_seq_len_not_zero(L: int):
    """On overflow the new active buffer holds exactly ``seq_len`` tokens
    (the ones the kernel just wrote), not zero."""
    num_blocks = 2
    slot = 0
    builder = _make_builder(L=L, num_blocks=num_blocks)
    builder.prev_num_accepted_tokens_d[slot] = L  # buffer is full

    _stage(builder, slots_col0=[slot], seq_lens=[3])  # L + 3 > L
    builder.apply_post_step()

    assert int(builder.cache_buf_idx_d[slot]) == 1
    assert int(builder.prev_num_accepted_tokens_d[slot]) == 3
