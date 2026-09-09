# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch

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
        assert not torch.count_nonzero(kwargs["a"][0, 1:])
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="FlashInfer GDN ReplaySSM requires an SM90+ CUDA device",
)
@pytest.mark.parametrize(
    ("executed_query_width", "query_len"),
    [(width, query_len) for width in (4, 8) for query_len in range(1, width + 1)],
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
    output = run_gdn_replayssm(
        mixed_qkv=mixed_qkv,
        a=torch.zeros(query_len, num_v_heads, dtype=dtype, device=device),
        b=torch.zeros(query_len, num_v_heads, dtype=dtype, device=device),
        A_log=torch.zeros(num_v_heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(num_v_heads, dtype=torch.float32, device=device),
        checkpoint_state=checkpoint_state,
        replayssm_cache=(
            torch.zeros(2, num_v_heads, 32, 128, dtype=dtype, device=device),
            torch.zeros(2, 1, 32, 128, dtype=dtype, device=device),
            torch.zeros(2, num_v_heads, 32, dtype=torch.float32, device=device),
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
