"""Importable vLLM worker callbacks for Cache-Prior experiments."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch.distributed as dist


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _cache_prior_routers(model: object) -> list[object]:
    """Return Cache-Prior strategies owned by registered MoE runners."""
    from vllm.model_executor.layers.fused_moe.router.cache_prior_router import (
        CachePriorRouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    return [
        module.router
        for module in model.modules()
        if isinstance(module, MoERunner) and isinstance(module.router, CachePriorRouter)
    ]


def configure_cache_prior(
    model: object,
    capacity: int,
    lambda_value: float,
    top_j: int,
    metrics_dir_value: str,
    batch_size: int,
    sequence_length: int,
    speculative_only: bool = False,
    speculative_max_tokens: int = 0,
    write_speculative_events: bool = False,
) -> dict[str, object]:
    """Install or reconfigure CachePriorRouter on one vLLM worker."""
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter
    from vllm.model_executor.layers.fused_moe.router.cache_prior_router import (
        CachePriorRouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    rank = _distributed_rank()
    metrics_dir = Path(metrics_dir_value)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"rank-{rank:04d}.jsonl"
    metrics_path.unlink(missing_ok=True)

    routers = _cache_prior_routers(model)
    if not routers:
        for module in model.modules():
            if not isinstance(module, MoERunner):
                continue
            base_router = module.router
            if not isinstance(base_router, BaseRouter):
                raise RuntimeError(
                    f"unsupported router {type(base_router).__name__} "
                    f"in {module.layer_name}"
                )
            experts = module.routed_experts
            if experts.custom_routing_function is not None:
                raise RuntimeError("custom MoE routing is not supported")
            module.router = CachePriorRouter(
                base_router,
                capacity=capacity,
                lambda_value=lambda_value,
                cache_bias_mode="selection",
                top_j=top_j,
                scoring_func=experts.scoring_func,
                renormalize=experts.renormalize,
                routed_scaling_factor=experts.routed_scaling_factor,
                e_score_correction_bias=experts.e_score_correction_bias,
                num_expert_group=(
                    experts.num_expert_group if experts.use_grouped_topk else None
                ),
                topk_group=(experts.topk_group if experts.use_grouped_topk else None),
                layer_name=module.layer_name,
                metrics_path=str(metrics_path),
                speculative_only=speculative_only,
                speculative_max_tokens=speculative_max_tokens,
                write_speculative_events=write_speculative_events,
            )
            routers.append(module.router)
    if not routers:
        raise RuntimeError("model contains no standard FusedMoE routers")

    for router in routers:
        router.capacity = capacity
        router.lambda_value = lambda_value
        router.cache_bias_mode = "selection"
        router.top_j = top_j
        router.speculative_only = speculative_only
        router.speculative_max_tokens = speculative_max_tokens
        router.write_speculative_events = write_speculative_events
        router.metrics_path = metrics_path
        # Retain one compact native-routing trace for baseline validation.
        router.trace_dir = (
            metrics_dir / f"traces-rank-{rank}" if lambda_value == 0 else None
        )
        router.reset_path = None
        router._range_reset_applied = True
        router.reset_cache(discard_residencies=True)
        router.reset_metrics()
        router.reset_range_estimator()
        router.clear_evaluation_batch_layout()
        if batch_size > 0 or sequence_length > 0:
            router.set_evaluation_batch_layout(batch_size, sequence_length)

    first = routers[0]
    return {
        "rank": rank,
        "routers": len(routers),
        "top_k": first.top_k,
        "num_experts": first.global_num_experts,
        "capacity": capacity,
        "lambda": lambda_value,
        "top_j": top_j,
        "cache_bias_mode": first.cache_bias_mode,
        "cache_bias_domain": "post_correction_selection_values",
        "speculative_only": speculative_only,
        "speculative_max_tokens": speculative_max_tokens,
        "write_speculative_events": write_speculative_events,
    }


def configure_speculative_cache_prior_worker(
    worker: object,
    capacity: int,
    lambda_value: float,
    top_j: int,
    metrics_dir_value: str,
    num_speculative_tokens: int,
) -> dict[str, object]:
    """Configure routers and commit them from vLLM rejection-sampler output."""
    if num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive")
    runner = getattr(worker, "model_runner")
    model = getattr(runner, "model")
    configuration = configure_cache_prior(
        model,
        capacity=capacity,
        lambda_value=lambda_value,
        top_j=top_j,
        metrics_dir_value=metrics_dir_value,
        batch_size=0,
        sequence_length=0,
        speculative_only=True,
        speculative_max_tokens=num_speculative_tokens + 1,
    )

    _install_speculative_sample_hook(runner, model)

    configuration["worker_rank"] = _distributed_rank()
    configuration["sample_hook_installed"] = True
    return configuration


def reset_speculative_request_caches_worker(worker: object) -> dict[str, int]:
    """Start a matched pass with empty request-local expert-cache state."""
    model = getattr(getattr(worker, "model_runner"), "model")
    routers = _cache_prior_routers(model)
    for router in routers:
        router.reset_cache(discard_residencies=True)
        router.reset_metrics()
        router.reset_range_estimator()
    return {"rank": _distributed_rank(), "routers": len(routers)}


def _install_speculative_sample_hook(runner: object, model: object) -> None:
    """Commit pending router transactions immediately after target sampling."""

    if not hasattr(runner, "_cache_prior_original_sample"):
        original_sample = runner._sample
        runner._cache_prior_original_sample = original_sample

        def sample_with_cache_commit(
            logits: object,
            spec_decode_metadata: object | None,
        ) -> Any:
            sampler_output = original_sample(logits, spec_decode_metadata)
            if spec_decode_metadata is None:
                return sampler_output

            from vllm.model_executor.layers.fused_moe.router.cache_prior_router import (
                speculative_commit_mask,
            )

            all_routers = _cache_prior_routers(model)
            pending_routers = [
                router
                for router in all_routers
                if router.has_pending_speculative_transactions
            ]
            draft_counts = tuple(
                int(count) for count in spec_decode_metadata.num_draft_tokens
            )
            if sum(draft_counts) and not pending_routers:
                diagnostic = [
                    {
                        "layer": router.layer_name,
                        "blocks": router.speculative_metrics.blocks,
                        "fallback_blocks": router.speculative_metrics.fallback_blocks,
                        "prefill_blocks": router.speculative_metrics.prefill_blocks,
                    }
                    for router in all_routers[:2]
                ]
                raise RuntimeError(
                    "speculative target sampling found no pending Cache-Prior block; "
                    f"routers={len(all_routers)}, draft_counts={draft_counts}, "
                    f"first_metrics={diagnostic}"
                )
            if pending_routers:
                committed_tokens = speculative_commit_mask(
                    draft_counts,
                    sampler_output.sampled_token_ids,
                )
                request_ids = tuple(
                    runner.input_batch.req_ids[: len(draft_counts)]
                )
                if len(request_ids) != len(draft_counts):
                    raise RuntimeError(
                        "speculative request IDs do not match draft counts"
                    )
                for router in pending_routers:
                    router.commit_speculative_batch(
                        committed_tokens,
                        request_ids=request_ids,
                        num_draft_tokens=draft_counts,
                    )
            return sampler_output

        runner._sample = sample_with_cache_commit


def collect_speculative_cache_misses(model: object) -> dict[str, object]:
    """Collect distinct expert-load misses from every Cache-Prior layer."""
    routers = _cache_prior_routers(model)
    if not routers:
        raise RuntimeError("model contains no Cache-Prior routers")

    metrics_by_layer = [router.speculative_metrics for router in routers]

    def histogram_dict(
        histogram: tuple[tuple[int, int], ...],
    ) -> dict[str, int]:
        return {str(duration): count for duration, count in histogram}

    def merge_histograms(
        histograms: list[tuple[tuple[int, int], ...]],
    ) -> Counter[int]:
        merged: Counter[int] = Counter()
        for histogram in histograms:
            merged.update(dict(histogram))
        return merged

    def percentile(histogram: Counter[int], quantile: float) -> float:
        count = sum(histogram.values())
        if not count:
            return 0.0
        threshold = max(1, int(quantile * count + 0.999999999))
        cumulative = 0
        for duration, frequency in sorted(histogram.items()):
            cumulative += frequency
            if cumulative >= threshold:
                return float(duration)
        raise RuntimeError("residency histogram count changed during traversal")

    layers = [
        {
            "layer": router.layer_name,
            "blocks": metrics.blocks,
            "cache_misses": metrics.cache_misses,
            "cache_hits": metrics.required_experts - metrics.cache_misses,
            "required_experts": metrics.required_experts,
            "tokens": metrics.tokens,
            "committed_tokens": metrics.committed_tokens,
            "requests": metrics.requests,
            "fallback_blocks": metrics.fallback_blocks,
            "fallback_cache_misses": metrics.fallback_cache_misses,
            "fallback_cache_hits": (
                metrics.fallback_required_experts - metrics.fallback_cache_misses
            ),
            "fallback_required_experts": metrics.fallback_required_experts,
            "fallback_tokens": metrics.fallback_tokens,
            "prefill_blocks": metrics.prefill_blocks,
            "prefill_overflows": metrics.prefill_overflows,
            "prefill_required_experts": metrics.prefill_required_experts,
            "prefill_cache_misses": metrics.prefill_cache_misses,
            "residency_observations": metrics.residency_observations,
            "residency_token_steps_sum": metrics.residency_token_steps_sum,
            "residency_block_steps_sum": metrics.residency_block_steps_sum,
            "mean_residency_tokens": (
                metrics.residency_token_steps_sum / metrics.residency_observations
                if metrics.residency_observations
                else 0.0
            ),
            "mean_residency_blocks": (
                metrics.residency_block_steps_sum / metrics.residency_observations
                if metrics.residency_observations
                else 0.0
            ),
            "residency_token_histogram": histogram_dict(
                metrics.residency_token_histogram
            ),
            "residency_block_histogram": histogram_dict(
                metrics.residency_block_histogram
            ),
        }
        for router, metrics in zip(routers, metrics_by_layer, strict=True)
    ]
    first = router_metrics = metrics_by_layer[0]
    for metrics in metrics_by_layer[1:]:
        if (
            metrics.blocks != first.blocks
            or metrics.tokens != first.tokens
            or metrics.committed_tokens != first.committed_tokens
            or metrics.requests != first.requests
            or metrics.fallback_blocks != first.fallback_blocks
            or metrics.fallback_tokens != first.fallback_tokens
            or metrics.prefill_blocks != first.prefill_blocks
        ):
            raise RuntimeError("Cache-Prior layer transaction counts disagree")

    total_layer_blocks = sum(metrics.blocks for metrics in metrics_by_layer)
    total_cache_misses = sum(metrics.cache_misses for metrics in metrics_by_layer)
    total_fallback_cache_misses = sum(
        metrics.fallback_cache_misses for metrics in metrics_by_layer
    )
    total_required_experts = sum(
        metrics.required_experts for metrics in metrics_by_layer
    )
    total_fallback_required_experts = sum(
        metrics.fallback_required_experts for metrics in metrics_by_layer
    )
    total_prefill_blocks = sum(metrics.prefill_blocks for metrics in metrics_by_layer)
    total_prefill_overflows = sum(
        metrics.prefill_overflows for metrics in metrics_by_layer
    )
    total_prefill_cache_misses = sum(
        metrics.prefill_cache_misses for metrics in metrics_by_layer
    )
    total_residency_observations = sum(
        metrics.residency_observations for metrics in metrics_by_layer
    )
    total_residency_token_steps = sum(
        metrics.residency_token_steps_sum for metrics in metrics_by_layer
    )
    total_residency_block_steps = sum(
        metrics.residency_block_steps_sum for metrics in metrics_by_layer
    )
    residency_token_histogram = merge_histograms(
        [metrics.residency_token_histogram for metrics in metrics_by_layer]
    )
    residency_block_histogram = merge_histograms(
        [metrics.residency_block_histogram for metrics in metrics_by_layer]
    )
    total_decode_cache_misses = total_cache_misses + total_fallback_cache_misses
    total_decode_required_experts = (
        total_required_experts + total_fallback_required_experts
    )
    if total_residency_observations != total_decode_cache_misses:
        raise RuntimeError(
            "decode residency observations must equal distinct expert loads: "
            f"{total_residency_observations} != {total_decode_cache_misses}"
        )
    return {
        "rank": _distributed_rank(),
        "miss_unit": "distinct_expert_load",
        "verification_steps": router_metrics.blocks,
        "layer_blocks": total_layer_blocks,
        "cache_misses": total_decode_cache_misses,
        "cache_hits": total_decode_required_experts - total_decode_cache_misses,
        "decode_cache_misses": total_decode_cache_misses,
        "decode_cache_hits": total_decode_required_experts - total_decode_cache_misses,
        "verification_cache_misses": total_cache_misses,
        "verification_cache_hits": total_required_experts - total_cache_misses,
        "fallback_decode_cache_misses": total_fallback_cache_misses,
        "fallback_decode_cache_hits": (
            total_fallback_required_experts - total_fallback_cache_misses
        ),
        "required_experts": total_decode_required_experts,
        "verification_required_experts": total_required_experts,
        "fallback_decode_required_experts": total_fallback_required_experts,
        "miss_fraction": (
            total_decode_cache_misses / total_decode_required_experts
            if total_decode_required_experts
            else 0.0
        ),
        "hit_fraction": (
            (total_decode_required_experts - total_decode_cache_misses)
            / total_decode_required_experts
            if total_decode_required_experts
            else 0.0
        ),
        "verification_tokens": router_metrics.tokens,
        "fallback_decode_steps": router_metrics.fallback_blocks,
        "fallback_decode_tokens": router_metrics.fallback_tokens,
        "committed_target_tokens": (
            router_metrics.committed_tokens + router_metrics.fallback_tokens
        ),
        "emitted_tokens": (
            router_metrics.committed_tokens + router_metrics.fallback_tokens
        ),
        "accepted_draft_tokens": (
            router_metrics.committed_tokens - router_metrics.requests
        ),
        "verified_requests": router_metrics.requests,
        "loads_per_emitted_token": (
            total_decode_cache_misses
            / (router_metrics.committed_tokens + router_metrics.fallback_tokens)
            if router_metrics.committed_tokens + router_metrics.fallback_tokens
            else 0.0
        ),
        "decode_loads_per_committed_target_token": (
            total_decode_cache_misses
            / (router_metrics.committed_tokens + router_metrics.fallback_tokens)
            if router_metrics.committed_tokens + router_metrics.fallback_tokens
            else 0.0
        ),
        "speculative_target_token_fraction": (
            router_metrics.committed_tokens
            / (router_metrics.committed_tokens + router_metrics.fallback_tokens)
            if router_metrics.committed_tokens + router_metrics.fallback_tokens
            else 0.0
        ),
        "residency_unit": "committed_target_token",
        "residency_observations": total_residency_observations,
        "residency_token_steps_sum": total_residency_token_steps,
        "mean_residency_tokens": (
            total_residency_token_steps / total_residency_observations
            if total_residency_observations
            else 0.0
        ),
        "residency_token_p50": percentile(residency_token_histogram, 0.50),
        "residency_token_p90": percentile(residency_token_histogram, 0.90),
        "residency_token_p95": percentile(residency_token_histogram, 0.95),
        "residency_token_histogram": {
            str(duration): count
            for duration, count in sorted(residency_token_histogram.items())
        },
        "residency_block_steps_sum": total_residency_block_steps,
        "mean_residency_blocks": (
            total_residency_block_steps / total_residency_observations
            if total_residency_observations
            else 0.0
        ),
        "residency_block_histogram": {
            str(duration): count
            for duration, count in sorted(residency_block_histogram.items())
        },
        "prefill_steps": router_metrics.prefill_blocks,
        "prefill_layer_blocks": total_prefill_blocks,
        "prefill_layer_overflows": total_prefill_overflows,
        "prefill_cache_misses": total_prefill_cache_misses,
        "total_cache_misses_including_prefill": (
            total_cache_misses
            + total_fallback_cache_misses
            + total_prefill_cache_misses
        ),
        "total_loads_per_committed_target_token": (
            (
                total_cache_misses
                + total_fallback_cache_misses
                + total_prefill_cache_misses
            )
            / (router_metrics.committed_tokens + router_metrics.fallback_tokens)
            if router_metrics.committed_tokens + router_metrics.fallback_tokens
            else 0.0
        ),
        "prefill_overflow_rate": (
            total_prefill_overflows / total_prefill_blocks
            if total_prefill_blocks
            else 0.0
        ),
        "layers": layers,
    }
