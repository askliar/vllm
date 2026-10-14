# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    fi_chunk_gated_delta_rule,
)
from vllm.model_executor.layers.mamba.gdn.replayssm import run_gdn_replayssm


def test_gdn_replayssm_packs_fixed_width_and_gathers_real_outputs():
    query_start_loc = torch.tensor([0, 1, 3], dtype=torch.int32)
    mixed_qkv = (
        torch.arange(3 * 384, dtype=torch.float32).view(3, 384).to(torch.bfloat16)
    )
    a = torch.arange(3, dtype=torch.float32).view(3, 1).to(torch.bfloat16)
    b = (a + 3).to(torch.bfloat16)
    checkpoint_state = torch.zeros(3, 1, 128, 128, dtype=torch.bfloat16)
    rings = (
        torch.zeros(3, 1, 32, 128, dtype=torch.bfloat16),
        torch.zeros(3, 1, 32, 128, dtype=torch.bfloat16),
        torch.zeros(3, 1, 32, dtype=torch.float32),
    )
    ring_start = torch.tensor([0, 1, 5], dtype=torch.int32)
    num_committed = torch.tensor([0, 2, 9], dtype=torch.int32)
    state_indices = torch.tensor([2, 0], dtype=torch.int32)
    output_indices = torch.tensor([0, 4, 5], dtype=torch.int32)

    def fake_kernel(**kwargs):
        assert kwargs["q"].shape == (2, 4, 1, 128)
        assert kwargs["k"].shape == (2, 4, 1, 128)
        assert kwargs["v"].shape == (2, 4, 1, 128)
        assert kwargs["a"].shape == (2, 4, 1)
        assert kwargs["b"].shape == (2, 4, 1)
        assert kwargs["a"].is_contiguous()
        assert kwargs["b"].is_contiguous()
        assert kwargs["a"].data_ptr() % 16 == 0
        assert kwargs["b"].data_ptr() % 16 == 0
        assert not kwargs["A_log"].requires_grad
        assert not kwargs["dt_bias"].requires_grad
        assert not torch.count_nonzero(kwargs["q"][0, 1:])
        assert not torch.count_nonzero(kwargs["q"][1, 2:])
        assert torch.isneginf(kwargs["a"][0, 1:]).all()
        assert torch.isneginf(kwargs["a"][1, 2:]).all()
        assert not torch.count_nonzero(kwargs["b"][1, 2:])
        assert kwargs["initial_state_indices"].tolist() == [2, -1]
        assert kwargs["hist_len"].tolist() == [9, 0]
        assert kwargs["cache_base"].tolist() == [5, 0]
        assert kwargs["flush_min"] == 13
        assert kwargs["restart_hist_on_flush"] is False
        dense_markers = torch.arange(8, dtype=torch.bfloat16).view(2, 4, 1, 1)
        kwargs["output"].copy_(dense_markers.expand_as(kwargs["output"]))

    result = run_gdn_replayssm(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=torch.zeros(1, requires_grad=True),
        dt_bias=torch.zeros(1, requires_grad=True),
        checkpoint_state=checkpoint_state,
        replayssm_cache=rings,
        ring_start=ring_start,
        num_committed=num_committed,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        output_indices=output_indices,
        executed_query_width=4,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        kernel=fake_kernel,
    )

    assert result.shape == (3, 1, 128)
    assert result[:, 0, 0].tolist() == [0, 4, 5]


def test_gdn_replayssm_stp_uses_native_width_and_private_cursor_commit():
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    mixed_qkv = torch.zeros(2, 384, dtype=torch.bfloat16)
    a = torch.zeros(2, 1, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    checkpoint_state = torch.zeros(3, 1, 128, 128, dtype=torch.bfloat16)
    rings = (
        torch.zeros(3, 1, 16, 128, dtype=torch.bfloat16),
        torch.zeros(3, 1, 16, 128, dtype=torch.bfloat16),
        torch.zeros(3, 1, 16, dtype=torch.float32),
    )
    ring_start = torch.zeros(3, dtype=torch.int32)
    num_committed = torch.tensor([0, 7, 15], dtype=torch.int32)

    def fake_kernel(**kwargs):
        assert kwargs["q"].shape == (2, 4, 1, 128)
        assert kwargs["k"].shape == (2, 4, 1, 128)
        assert kwargs["v"].shape == (2, 4, 1, 128)
        assert kwargs["q"].stride(1) == kwargs["k"].stride(1)
        assert kwargs["q"].stride(1) == kwargs["v"].stride(1)
        assert not torch.count_nonzero(kwargs["q"][:, 1:])
        assert not torch.count_nonzero(kwargs["k"][:, 1:])
        assert not torch.count_nonzero(kwargs["v"][:, 1:])
        assert not torch.count_nonzero(kwargs["a"][:, 1:])
        assert not torch.count_nonzero(kwargs["b"][:, 1:])
        assert kwargs["hist_len"].tolist() == [15, 7]
        assert kwargs["cache_base"].tolist() == [0, 0]
        assert kwargs["flush_min"] == 15
        assert kwargs["prepadded"] is True
        assert "restart_hist_on_flush" not in kwargs
        kwargs["hist_len"].fill_(99)
        kwargs["cache_base"].fill_(8)
        kwargs["output"][:, 0].copy_(
            torch.arange(2, dtype=torch.bfloat16).view(2, 1, 1).expand(2, 1, 128)
        )

    result = run_gdn_replayssm(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        checkpoint_state=checkpoint_state,
        replayssm_cache=rings,
        ring_start=ring_start,
        num_committed=num_committed,
        query_start_loc=query_start_loc,
        state_indices=torch.tensor([2, 1], dtype=torch.int32),
        output_indices=torch.tensor([0, 1], dtype=torch.int32),
        executed_query_width=1,
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=128,
        head_v_dim=128,
        kernel=fake_kernel,
    )

    assert result[:, 0, 0].tolist() == [0, 1]
    assert num_committed.tolist() == [0, 7, 15]
    assert ring_start.tolist() == [0, 0, 0]


def _reference_gdn_step(
    state,
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
):
    """FP32 recurrence using FlashInfer's effective BF16 gate constants."""
    state = state.float().clone()
    A_log = A_log.to(torch.bfloat16).float()
    dt_bias = dt_bias.to(torch.bfloat16).float()
    group_size = v.shape[1] // k.shape[1]
    outputs = []
    for token in range(q.shape[0]):
        k_hv = F.normalize(k[token].float(), dim=-1).repeat_interleave(
            group_size, dim=0
        )
        q_hv = F.normalize(q[token].float(), dim=-1).repeat_interleave(
            group_size, dim=0
        )
        q_hv *= 1.0 / math.sqrt(q.shape[-1])
        decay = -torch.exp(A_log) * F.softplus(a[token].float() + dt_bias)
        beta = torch.sigmoid(b[token].float())
        state *= torch.exp(decay)[:, None, None]
        prediction = torch.einsum("hvk,hk->hv", state, k_hv)
        update = (v[token].float() - prediction) * beta[:, None]
        state += update[:, :, None] * k_hv[:, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q_hv))
    return torch.stack(outputs), state


def _reconstruct_replayssm_state(
    checkpoint,
    k_cache,
    u_cache,
    g_cache,
    committed,
    ring_start,
):
    state = checkpoint.float().clone()
    if committed == 0:
        return state
    rows = torch.tensor(
        [(ring_start + offset) % k_cache.size(1) for offset in range(committed)],
        dtype=torch.long,
        device=checkpoint.device,
    )
    k_logical = k_cache.index_select(1, rows).float().repeat_interleave(2, dim=0)
    u_logical = u_cache.index_select(1, rows).float()
    g_logical = g_cache.index_select(1, rows).float()
    final_g = g_logical[:, -1]
    weights = torch.exp(final_g[:, None] - g_logical)
    return torch.exp(final_g)[:, None, None] * state + torch.einsum(
        "hpv,hpk->hvk", weights[:, :, None] * u_logical, k_logical
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN prefill requires an SM90+ CUDA device",
)
def test_gdn_canonical_prefill_64_1_4_matches_unchunked():
    """An intermediate one-token prompt chunk preserves output and state."""
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer-python is not installed")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260909)
    dtype = torch.bfloat16
    num_tokens, num_k_heads, num_v_heads, head_dim = 69, 1, 2, 128

    q = torch.randn(
        1,
        num_tokens,
        num_k_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        1,
        num_tokens,
        num_k_heads,
        head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    # Match _forward_core: fused_post_conv_prep normalizes q/k before the
    # FlashInfer canonical-prefill wrapper, which then receives this flag false.
    q = F.normalize(q.float(), dim=-1).to(dtype)
    k = F.normalize(k.float(), dim=-1).to(dtype)
    v = (
        torch.randn(
            1,
            num_tokens,
            num_v_heads,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.1
    ).to(dtype)
    g = -(
        torch.rand(
            1,
            num_tokens,
            num_v_heads,
            generator=generator,
            device=device,
        )
        * 0.03
        + 0.003
    )
    beta = torch.sigmoid(
        torch.randn(
            1,
            num_tokens,
            num_v_heads,
            generator=generator,
            device=device,
        )
    )
    initial_state = (
        torch.randn(
            1,
            num_v_heads,
            head_dim,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.03
    ).to(dtype)

    full_output, full_state = fi_chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, num_tokens], dtype=torch.int32, device=device),
        use_qk_l2norm_in_kernel=False,
    )
    assert full_state is not None

    chunk_outputs = []
    chunk_state = initial_state
    start = 0
    for chunk_len in (64, 1, 4):
        stop = start + chunk_len
        chunk_output, next_state = fi_chunk_gated_delta_rule(
            # Each scheduled step owns fresh activation buffers. Cloning the
            # reference slices matches that boundary and its alignment.
            q=q[:, start:stop].clone(),
            k=k[:, start:stop].clone(),
            v=v[:, start:stop].clone(),
            g=g[:, start:stop].clone(),
            beta=beta[:, start:stop].clone(),
            initial_state=chunk_state,
            output_final_state=True,
            cu_seqlens=torch.tensor([0, chunk_len], dtype=torch.int32, device=device),
            use_qk_l2norm_in_kernel=False,
        )
        assert next_state is not None
        chunk_outputs.append(chunk_output)
        # Match the canonical vLLM cache boundary between scheduled chunks.
        chunk_state = next_state.to(dtype)
        start = stop

    torch.testing.assert_close(
        torch.cat(chunk_outputs, dim=1).float(),
        full_output.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    torch.testing.assert_close(
        chunk_state.float(),
        full_state.to(dtype).float(),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
@pytest.mark.parametrize(
    (
        "width",
        "initial_committed",
        "initial_start",
        "query_lens",
        "acceptance",
    ),
    [
        (4, 12, 28, [4, 4, 4], [2, 3, 1]),
        (8, 8, 30, [8, 8, 8], [5, 2, 7]),
        (4, 12, 28, [1, 3, 2], [1, 2, 1]),
        (8, 8, 30, [1, 5, 2], [1, 3, 2]),
        (8, 8, 30, [6, 3, 5], [4, 2, 3]),
    ],
)
def test_gdn_replayssm_multistep_matches_accepted_state_reference(
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    initial_committed: int,
    initial_start: int,
    query_lens: list[int],
    acceptance: list[int],
):
    """Full and ragged proposals preserve accepted state across wrap and flush."""
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer-python is not installed")
    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260909 + width)
    dtype = torch.bfloat16
    num_k_heads, num_v_heads, head_dim = 1, 2, 128

    checkpoint = torch.zeros(
        2, num_v_heads, head_dim, head_dim, dtype=dtype, device=device
    )
    checkpoint[1] = (
        torch.randn(
            num_v_heads,
            head_dim,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.03
    ).to(dtype)
    k_cache = torch.zeros(2, num_k_heads, 32, head_dim, dtype=dtype, device=device)
    u_cache = torch.zeros(2, num_v_heads, 32, head_dim, dtype=dtype, device=device)
    g_cache = torch.zeros(2, num_v_heads, 32, dtype=torch.float32, device=device)
    initial_rows = torch.tensor(
        [(initial_start + offset) % 32 for offset in range(initial_committed)],
        dtype=torch.long,
        device=device,
    )
    initial_k = F.normalize(
        torch.randn(
            num_k_heads,
            initial_committed,
            head_dim,
            generator=generator,
            device=device,
        ),
        dim=-1,
    )
    k_cache[1, :, initial_rows] = initial_k.to(dtype)
    u_cache[1, :, initial_rows] = (
        torch.randn(
            num_v_heads,
            initial_committed,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.03
    ).to(dtype)
    initial_g = -(
        torch.rand(
            num_v_heads,
            initial_committed,
            generator=generator,
            device=device,
        )
        * 0.03
        + 0.003
    )
    g_cache[1, :, initial_rows] = torch.cumsum(initial_g, dim=-1)

    ring_start = torch.tensor([0, initial_start], dtype=torch.int32, device=device)
    committed = torch.tensor([0, initial_committed], dtype=torch.int32, device=device)
    A_log = (
        torch.full((num_v_heads,), -3.0, device=device)
        + torch.rand(num_v_heads, generator=generator, device=device) * 0.3
    )
    dt_bias = torch.randn(num_v_heads, generator=generator, device=device) * 0.2
    accepted_state = _reconstruct_replayssm_state(
        checkpoint[1],
        k_cache[1],
        u_cache[1],
        g_cache[1],
        initial_committed,
        initial_start,
    )

    for query_len, accepted in zip(query_lens, acceptance):
        q = torch.randn(
            query_len, num_k_heads, head_dim, generator=generator, device=device
        ).to(dtype)
        k = torch.randn(
            query_len, num_k_heads, head_dim, generator=generator, device=device
        ).to(dtype)
        v = (
            torch.randn(
                query_len,
                num_v_heads,
                head_dim,
                generator=generator,
                device=device,
            )
            * 0.1
        ).to(dtype)
        a = (
            torch.randn(query_len, num_v_heads, generator=generator, device=device)
            * 0.2
        ).to(dtype)
        b = torch.randn(query_len, num_v_heads, generator=generator, device=device).to(
            dtype
        )
        mixed_qkv = torch.cat(
            [
                q.flatten(1),
                k.flatten(1),
                v.flatten(1),
            ],
            dim=-1,
        )
        output = run_gdn_replayssm(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint,
            replayssm_cache=(u_cache, k_cache, g_cache),
            ring_start=ring_start,
            num_committed=committed,
            query_start_loc=torch.tensor(
                [0, query_len], dtype=torch.int32, device=device
            ),
            state_indices=torch.tensor([1], dtype=torch.int32, device=device),
            output_indices=torch.arange(query_len, dtype=torch.int32, device=device),
            executed_query_width=width,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_dim,
            head_v_dim=head_dim,
        )
        output_reference, _ = _reference_gdn_step(
            accepted_state, q, k, v, a, b, A_log, dt_bias
        )
        assert (output.float() - output_reference).abs().max().item() < 8e-3

        _, accepted_state = _reference_gdn_step(
            accepted_state,
            q[:accepted],
            k[:accepted],
            v[:accepted],
            a[:accepted],
            b[:accepted],
            A_log,
            dt_bias,
        )
        old_start = int(ring_start[1].item())
        old_committed = int(committed[1].item())
        if old_committed + width > 16:
            ring_start[1] = (old_start + old_committed) % 32
            committed[1] = accepted
        else:
            committed[1] = old_committed + accepted
        reconstructed = _reconstruct_replayssm_state(
            checkpoint[1],
            k_cache[1],
            u_cache[1],
            g_cache[1],
            int(committed[1].item()),
            int(ring_start[1].item()),
        )
        assert (reconstructed - accepted_state).abs().max().item() < 2e-2


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
def test_gdn_replayssm_stp_multistep_matches_state_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    if (
        importlib.util.find_spec("flashinfer.gdn_kernels.gdn_decode_bf16_wy_ucache_stp")
        is None
    ):
        pytest.skip("FlashInfer GDN STP ReplaySSM is not installed")
    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260910)
    dtype = torch.bfloat16
    num_k_heads, num_v_heads, head_dim = 1, 2, 128
    checkpoint = torch.zeros(
        2, num_v_heads, head_dim, head_dim, dtype=dtype, device=device
    )
    u_cache = torch.zeros(2, num_v_heads, 16, head_dim, dtype=dtype, device=device)
    k_cache = torch.zeros(2, num_k_heads, 16, head_dim, dtype=dtype, device=device)
    g_cache = torch.zeros(2, num_v_heads, 16, dtype=torch.float32, device=device)
    ring_start = torch.zeros(2, dtype=torch.int32, device=device)
    committed = torch.zeros(2, dtype=torch.int32, device=device)
    A_log = torch.full((num_v_heads,), -3.0, device=device)
    dt_bias = torch.randn(num_v_heads, generator=generator, device=device) * 0.2
    accepted_state = checkpoint[1].float().clone()

    for _ in range(32):
        q = torch.randn(
            1, num_k_heads, head_dim, generator=generator, device=device
        ).to(dtype)
        k = torch.randn(
            1, num_k_heads, head_dim, generator=generator, device=device
        ).to(dtype)
        v = (
            torch.randn(1, num_v_heads, head_dim, generator=generator, device=device)
            * 0.1
        ).to(dtype)
        a = (torch.randn(1, num_v_heads, generator=generator, device=device) * 0.2).to(
            dtype
        )
        b = torch.randn(1, num_v_heads, generator=generator, device=device).to(dtype)
        output = run_gdn_replayssm(
            mixed_qkv=torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=-1),
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            checkpoint_state=checkpoint,
            replayssm_cache=(u_cache, k_cache, g_cache),
            ring_start=ring_start,
            num_committed=committed,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            state_indices=torch.tensor([1], dtype=torch.int32, device=device),
            output_indices=torch.tensor([0], dtype=torch.int32, device=device),
            executed_query_width=1,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_dim,
            head_v_dim=head_dim,
        )
        output_reference, accepted_state = _reference_gdn_step(
            accepted_state, q, k, v, a, b, A_log, dt_bias
        )
        assert (output.float() - output_reference).abs().max().item() < 8e-3

        old_committed = int(committed[1].item())
        committed[1] = 0 if old_committed >= 15 else old_committed + 1
        reconstructed = _reconstruct_replayssm_state(
            checkpoint[1],
            k_cache[1],
            u_cache[1],
            g_cache[1],
            int(committed[1].item()),
            0,
        )
        assert ring_start[1].item() == 0
        assert (reconstructed - accepted_state).abs().max().item() < 2e-2


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN prefix materialization requires an SM90+ CUDA device",
)
@pytest.mark.parametrize("ring_slots", [16, 32])
def test_gdn_prefix_materialize_matches_reference(ring_slots: int):
    materialize_mod = pytest.importorskip(
        "flashinfer.gdn_kernels.gdn_prefix_materialize"
    )
    materialize = materialize_mod.gdn_prefix_materialize
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260910 + ring_slots)
    dtype = torch.bfloat16
    pool, num_k_heads, num_v_heads, head_dim = 5, 1, 2, 128
    state = (
        torch.randn(
            pool,
            num_v_heads,
            head_dim,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.03
    ).to(dtype)
    k_cache = (
        torch.randn(
            pool,
            num_k_heads,
            ring_slots,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.03
    ).to(dtype)
    u_cache = (
        torch.randn(
            pool,
            num_v_heads,
            ring_slots,
            head_dim,
            generator=generator,
            device=device,
        )
        * 0.03
    ).to(dtype)
    g_cache = torch.zeros(
        pool, num_v_heads, ring_slots, dtype=torch.float32, device=device
    )

    # Production KV-cache views have dense page contents but a padded stride
    # between blocks. Exercise that exact contract instead of accepting a
    # materializer that only works with compact standalone tensors.
    def block_strided(tensor: torch.Tensor) -> torch.Tensor:
        block_size = tensor[0].numel()
        storage = torch.empty(
            pool * (block_size + 64), dtype=tensor.dtype, device=device
        )
        view = torch.as_strided(
            storage,
            tensor.shape,
            (block_size + 64, *tensor.stride()[1:]),
        )
        view.copy_(tensor)
        return view

    state = block_strided(state)
    k_cache = block_strided(k_cache)
    u_cache = block_strided(u_cache)
    g_cache = block_strided(g_cache)
    src_slots = torch.tensor([1, 2, 0, -1], dtype=torch.int32, device=device)
    dst_slots = torch.tensor([3, 4, 0, -1], dtype=torch.int32, device=device)
    cache_base = torch.tensor(
        [0, ring_slots - 3, 2, 0], dtype=torch.int32, device=device
    )
    count = torch.tensor([0, 5, 5, -1], dtype=torch.int32, device=device)
    active = torch.tensor([2, 1, 0, -1], dtype=torch.int32, device=device)
    num_active = torch.tensor([3], dtype=torch.int32, device=device)
    rows = torch.tensor(
        [(ring_slots - 3 + i) % ring_slots for i in range(5)],
        dtype=torch.long,
        device=device,
    )
    increments = -(
        torch.rand(num_v_heads, 5, generator=generator, device=device) * 0.03 + 0.003
    )
    g_cache[2, :, rows] = torch.cumsum(increments, dim=-1)
    alias_rows = torch.tensor(
        [(2 + i) % ring_slots for i in range(5)],
        dtype=torch.long,
        device=device,
    )
    alias_increments = -(
        torch.rand(num_v_heads, 5, generator=generator, device=device) * 0.03 + 0.003
    )
    g_cache[0, :, alias_rows] = torch.cumsum(alias_increments, dim=-1)
    source_before = state[[0, 1, 2]].clone()
    expected_copy = state[1].clone()
    expected_fold = _reconstruct_replayssm_state(
        state[2], k_cache[2], u_cache[2], g_cache[2], 5, ring_slots - 3
    )
    expected_alias = _reconstruct_replayssm_state(
        state[0], k_cache[0], u_cache[0], g_cache[0], 5, 2
    )

    materialize(
        state=state,
        src_slots=src_slots,
        dst_slots=dst_slots,
        k_cache=k_cache,
        u_cache=u_cache,
        g_cache=g_cache,
        cache_base=cache_base,
        count=count,
        active_request_indices=active,
        num_active=num_active,
    )

    torch.testing.assert_close(state[3], expected_copy, rtol=0, atol=0)
    torch.testing.assert_close(state[4].float(), expected_fold, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state[0].float(), expected_alias, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state[[1, 2]], source_before[1:], rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
@pytest.mark.parametrize(
    ("executed_query_width", "query_len"),
    [(1, 1)]
    + [(width, query_len) for width in (4, 8) for query_len in range(1, width + 1)],
)
def test_gdn_replayssm_flashinfer_smoke(
    monkeypatch: pytest.MonkeyPatch,
    executed_query_width: int,
    query_len: int,
):
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer-python is not installed")
    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_k_heads, num_v_heads = 1, 2
    qkv_dim = 2 * num_k_heads * 128 + num_v_heads * 128
    # The second row is a CUDA-graph-style null request with no input tokens.
    mixed_qkv = torch.zeros(query_len, qkv_dim, dtype=dtype, device=device)
    checkpoint_storage = torch.zeros(
        2, num_v_heads * 128 * 128 + 64, dtype=dtype, device=device
    )
    checkpoint_state = torch.as_strided(
        checkpoint_storage,
        (2, num_v_heads, 128, 128),
        (checkpoint_storage.stride(0), 128 * 128, 128, 1),
    )
    ring_slots = 16 if executed_query_width == 1 else 32
    output = run_gdn_replayssm(
        mixed_qkv=mixed_qkv,
        a=torch.zeros(query_len, num_v_heads, dtype=dtype, device=device),
        b=torch.zeros(query_len, num_v_heads, dtype=dtype, device=device),
        A_log=torch.zeros(num_v_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32, device=device),
        checkpoint_state=checkpoint_state,
        replayssm_cache=(
            torch.zeros(2, num_v_heads, ring_slots, 128, dtype=dtype, device=device),
            torch.zeros(2, 1, ring_slots, 128, dtype=dtype, device=device),
            torch.zeros(2, num_v_heads, ring_slots, dtype=torch.float32, device=device),
        ),
        ring_start=torch.zeros(2, dtype=torch.int32, device=device),
        num_committed=torch.zeros(2, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor(
            [0, query_len, query_len], dtype=torch.int32, device=device
        ),
        state_indices=torch.tensor([1, 0], dtype=torch.int32, device=device),
        output_indices=torch.arange(query_len, dtype=torch.int32, device=device),
        executed_query_width=executed_query_width,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=128,
        head_v_dim=128,
    )
    assert output.shape == (query_len, num_v_heads, 128)
    assert torch.isfinite(output).all()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
def test_gdn_replayssm_cuda_graph_replays_with_a_null_row(
    monkeypatch: pytest.MonkeyPatch,
):
    if importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer-python is not installed")
    monkeypatch.setenv("SGLANG_GDN_WY_STRIDED_QKV", "1")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    width, num_k_heads, num_v_heads = 4, 1, 2
    qkv_dim = 2 * num_k_heads * 128 + num_v_heads * 128
    mixed_qkv = torch.randn(2 * width, qkv_dim, dtype=dtype, device=device) * 0.01
    a = torch.zeros(2 * width, num_v_heads, dtype=dtype, device=device)
    b = torch.zeros_like(a)
    checkpoint_storage = torch.zeros(
        3, num_v_heads * 128 * 128 + 64, dtype=dtype, device=device
    )
    checkpoint_state = torch.as_strided(
        checkpoint_storage,
        (3, num_v_heads, 128, 128),
        (checkpoint_storage.stride(0), 128 * 128, 128, 1),
    )
    rings = (
        torch.zeros(3, num_v_heads, 32, 128, dtype=dtype, device=device),
        torch.zeros(3, num_k_heads, 32, 128, dtype=dtype, device=device),
        torch.zeros(3, num_v_heads, 32, dtype=torch.float32, device=device),
    )
    ring_start = torch.tensor([0, 3, 17], dtype=torch.int32, device=device)
    num_committed = torch.tensor([0, 0, 13], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor(
        [0, width, 2 * width], dtype=torch.int32, device=device
    )
    state_indices = torch.tensor([1, 2], dtype=torch.int32, device=device)
    output_indices = torch.arange(2 * width, dtype=torch.int32, device=device)

    kwargs = dict(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=torch.zeros(num_v_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32, device=device),
        checkpoint_state=checkpoint_state,
        replayssm_cache=rings,
        ring_start=ring_start,
        num_committed=num_committed,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        output_indices=output_indices,
        executed_query_width=width,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        head_k_dim=128,
        head_v_dim=128,
    )

    run_gdn_replayssm(**kwargs)
    torch.accelerator.synchronize()
    checkpoint_state.zero_()
    for ring in rings:
        ring.zero_()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run_gdn_replayssm(**kwargs)

    # Reuse the captured shapes and addresses with one fewer active row and a
    # different live slot/history. The null row must leave its output zero.
    query_start_loc.copy_(torch.tensor([0, width, width], device=device))
    state_indices.copy_(torch.tensor([2, 0], dtype=torch.int32, device=device))
    ring_start[2] = 29
    num_committed[2] = 9
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.isfinite(graph_output).all()
    assert not torch.count_nonzero(graph_output[width:])
