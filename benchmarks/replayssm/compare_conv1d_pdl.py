# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure the paired vLLM causal-conv1d -> FlashInfer ReplaySSM chain.

Unlike an isolated ReplaySSM benchmark, this keeps the Triton conv1d producer
immediately before the FlashInfer consumer so PDL can overlap their independent
work. The modes distinguish a fully serialized chain from producer-only and
fully paired PDL launches.

Examples::

    python compare_conv1d_pdl.py --pdl-mode off
    python compare_conv1d_pdl.py --pdl-mode paired
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

PDL_MODES = {
    "off": (False, False),
    "producer-only": (True, False),
    "consumer-only": (False, True),
    "paired": (True, True),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdl-mode", choices=PDL_MODES, required=True)
    parser.add_argument("--batch-sizes", default="1,8,16,32,64,128")
    parser.add_argument("--spec-lengths", default="2,4,8")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--nheads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dstate", type=int, default=128)
    parser.add_argument("--ngroups", type=int, default=8)
    parser.add_argument("--conv-width", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--tune-flashinfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the native FlashInfer tactic tuner before paired timing.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


class _PlatformPDLOverride:
    def __init__(self, platform, enabled: bool):
        self._platform = platform
        self._enabled = enabled

    def __getattr__(self, name):
        return getattr(self._platform, name)

    def is_arch_support_pdl(self):
        return self._enabled


def _set_conv_pdl(enabled: bool) -> None:
    from vllm.model_executor.layers.mamba.ops import causal_conv1d
    from vllm.platforms import current_platform

    if enabled and not current_platform.is_arch_support_pdl():
        raise RuntimeError("PDL mode requires a PDL-capable GPU")
    causal_conv1d.current_platform = _PlatformPDLOverride(current_platform, enabled)


def _make_inputs(args: argparse.Namespace, batch: int, spec_len: int) -> dict:
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    slots = batch + 1
    ring_len = args.window + spec_len
    x_size = args.nheads * args.head_dim
    group_size = args.ngroups * args.dstate
    conv_dim = x_size + 2 * group_size
    conv_state_len = args.conv_width - 1 + spec_len - 1

    A_base = -torch.rand(args.nheads, device=device) - 0.5
    A = A_base[:, None, None].expand(args.nheads, args.head_dim, args.dstate)
    D = torch.randn(args.nheads, device=device)[:, None].expand(
        args.nheads, args.head_dim
    )
    dt_bias = torch.randn(args.nheads, device=device)[:, None].expand(
        args.nheads, args.head_dim
    )
    dt_base = torch.randn(batch, spec_len, args.nheads, device=device, dtype=dtype)
    dt = dt_base[..., None].expand(batch, spec_len, args.nheads, args.head_dim)
    state_initial = torch.randn(
        slots,
        args.nheads,
        args.head_dim,
        args.dstate,
        device=device,
        dtype=torch.float32,
    )
    conv_state_initial = torch.randn(
        slots,
        conv_dim,
        conv_state_len,
        device=device,
        dtype=dtype,
    )
    x_cache_initial = torch.randn(
        slots,
        args.nheads,
        ring_len,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    B_cache_initial = torch.randn(
        slots,
        args.ngroups,
        ring_len,
        args.dstate,
        device=device,
        dtype=dtype,
    )
    dt_cache_initial = torch.randn(
        slots,
        args.nheads,
        ring_len,
        device=device,
        dtype=torch.float32,
    ).abs()
    return {
        "state_initial": state_initial,
        "state": state_initial.clone(),
        "conv_state_initial": conv_state_initial,
        "conv_state": conv_state_initial.clone(),
        "x_cache_initial": x_cache_initial,
        "x_cache": x_cache_initial.clone(),
        "B_cache_initial": B_cache_initial,
        "B_cache": B_cache_initial.clone(),
        "dt_cache_initial": dt_cache_initial,
        "dt_cache": dt_cache_initial.clone(),
        "conv_input": torch.randn(
            batch * spec_len, conv_dim, device=device, dtype=dtype
        ),
        "conv_weight": torch.randn(
            conv_dim, args.conv_width, device=device, dtype=dtype
        ),
        "conv_bias": torch.randn(conv_dim, device=device, dtype=dtype),
        "conv_out": torch.empty(batch * spec_len, conv_dim, device=device, dtype=dtype),
        "ssm_out": torch.empty(
            batch,
            spec_len,
            args.nheads,
            args.head_dim,
            device=device,
            dtype=dtype,
        ),
        "dt": dt,
        "A": A,
        "D": D,
        "dt_bias": dt_bias,
        "indices": torch.arange(1, slots, device=device, dtype=torch.int32),
        "accepted": torch.full((batch,), spec_len, device=device, dtype=torch.int32),
        "query_start_loc": torch.arange(
            0, (batch + 1) * spec_len, spec_len, device=device, dtype=torch.int32
        ),
        "x_size": x_size,
        "group_size": group_size,
    }


def _make_case(
    args: argparse.Namespace,
    batch: int,
    spec_len: int,
    flush: bool,
    pdl_mode: str,
):
    from flashinfer.mamba.checkpointing_ssu import (
        allocate_checkpointing_ssu_scratch,
        checkpointing_ssu,
    )

    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    tensors = _make_inputs(args, batch, spec_len)
    conv_pdl, replayssm_pdl = PDL_MODES[pdl_mode]
    slots = batch + 1
    pnat = args.window if flush else 0
    ring_start = torch.zeros(slots, device="cuda", dtype=torch.int32)
    prev = torch.full((slots,), pnat, device="cuda", dtype=torch.int32)
    tensors["ring_start"] = ring_start
    tensors["prev"] = prev
    scratch = allocate_checkpointing_ssu_scratch(
        batch,
        args.nheads,
        spec_len,
        args.window,
        torch.bfloat16,
        "cuda",
    )

    def run() -> None:
        _set_conv_pdl(conv_pdl)
        mixed = causal_conv1d_update(
            tensors["conv_input"],
            tensors["conv_state"],
            tensors["conv_weight"],
            tensors["conv_bias"],
            activation="silu",
            conv_state_indices=tensors["indices"],
            num_accepted_tokens=tensors["accepted"],
            query_start_loc=tensors["query_start_loc"],
            max_query_len=spec_len,
            out=tensors["conv_out"],
        )
        x_flat, B_flat, C_flat = torch.split(
            mixed,
            [tensors["x_size"], tensors["group_size"], tensors["group_size"]],
            dim=-1,
        )
        x = x_flat.view(batch, spec_len, args.nheads, args.head_dim)
        B = B_flat.view(batch, spec_len, args.ngroups, args.dstate)
        C = C_flat.view(batch, spec_len, args.ngroups, args.dstate)
        checkpointing_ssu(
            tensors["state"],
            tensors["x_cache"],
            tensors["B_cache"],
            tensors["dt_cache"],
            ring_start,
            prev,
            x,
            tensors["dt"],
            tensors["A"],
            B,
            C,
            tensors["ssm_out"],
            D=tensors["D"],
            dt_bias=tensors["dt_bias"],
            dt_softplus=True,
            state_batch_indices=tensors["indices"],
            enable_pdl=replayssm_pdl,
            cb_scaled=scratch[0],
            cumAdt_vec=scratch[1],
            cb_old=scratch[2],
            algorithm="auto",
        )

    def reset() -> None:
        tensors["state"].copy_(tensors["state_initial"])
        tensors["conv_state"].copy_(tensors["conv_state_initial"])
        tensors["x_cache"].copy_(tensors["x_cache_initial"])
        tensors["B_cache"].copy_(tensors["B_cache_initial"])
        tensors["dt_cache"].copy_(tensors["dt_cache_initial"])
        ring_start.zero_()
        prev.fill_(pnat)

    if args.tune_flashinfer:
        from flashinfer.autotuner import autotune

        with autotune(True):
            run()
        reset()

    return run, reset, tensors


def _verify_case(
    args: argparse.Namespace, batch: int, spec_len: int, flush: bool
) -> None:
    """Check the selected launch policy against the fully serialized chain."""
    baseline_run, baseline_reset, baseline = _make_case(
        args, batch, spec_len, flush, "off"
    )
    candidate_run, candidate_reset, candidate = _make_case(
        args, batch, spec_len, flush, args.pdl_mode
    )
    baseline_reset()
    baseline_run()
    candidate_reset()
    candidate_run()
    torch.accelerator.synchronize()

    for name in (
        "conv_out",
        "ssm_out",
        "conv_state",
        "state",
        "x_cache",
        "B_cache",
        "dt_cache",
        "ring_start",
        "prev",
    ):
        torch.testing.assert_close(
            candidate[name], baseline[name], rtol=0, atol=0, equal_nan=True
        )


def _time_case(
    args: argparse.Namespace, batch: int, spec_len: int, flush: bool
) -> dict:
    _verify_case(args, batch, spec_len, flush)
    run, reset, _ = _make_case(args, batch, spec_len, flush, args.pdl_mode)
    for _ in range(args.warmup):
        reset()
        run()
    torch.accelerator.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
    label = (
        f"conv1d+replayssm/{args.pdl_mode}/b{batch}/t{spec_len}/"
        f"{'flush' if flush else 'verify'}"
    )
    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    for start, end in zip(starts, ends):
        reset()
        start.record()
        torch.cuda.nvtx.range_push(label)
        run()
        torch.cuda.nvtx.range_pop()
        end.record()
    torch.accelerator.synchronize()
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    times_us = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    result = {
        "pdl_mode": args.pdl_mode,
        "batch": batch,
        "spec_len": spec_len,
        "path": "flush" if flush else "verify",
        "median_us": statistics.median(times_us),
        "p95_us": _percentile(times_us, 0.95),
        "p99_us": _percentile(times_us, 0.99),
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    args = parse_args()
    spec_lengths = _csv_ints(args.spec_lengths)
    if max(spec_lengths) > args.window:
        raise ValueError("spec length must not exceed the logical replay window")
    results = [
        _time_case(args, batch, spec_len, flush)
        for batch in _csv_ints(args.batch_sizes)
        for spec_len in spec_lengths
        for flush in (False, True)
    ]
    payload = {"args": vars(args), "results": results}
    print("RESULT_JSON " + json.dumps(payload), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
