"""CPU-only invariants for speculative expert-cache transactions."""

from __future__ import annotations

import enum
import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import torch


class RoutingMethodType(enum.Enum):
    Sigmoid = "sigmoid"


class BaseRouter(torch.nn.Module):
    def __init__(self, *, top_k: int, global_num_experts: int, eplb_state=None) -> None:
        super().__init__()
        self.top_k = top_k
        self.global_num_experts = global_num_experts
        self.eplb_state = eplb_state


for module_name in (
    "vllm",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.fused_moe",
    "vllm.model_executor.layers.fused_moe.router",
):
    sys.modules[module_name] = types.ModuleType(module_name)
forward_context = types.ModuleType("vllm.forward_context")
forward_context.get_forward_context = lambda: None
forward_context.is_forward_context_available = lambda: False
sys.modules[forward_context.__name__] = forward_context
config = types.ModuleType("vllm.model_executor.layers.fused_moe.config")
config.RoutingMethodType = RoutingMethodType
sys.modules[config.__name__] = config
base_router_module = types.ModuleType(
    "vllm.model_executor.layers.fused_moe.router.base_router"
)
base_router_module.BaseRouter = BaseRouter
sys.modules[base_router_module.__name__] = base_router_module

source_root = Path(os.environ.get("CACHE_MOE_SOURCE_ROOT", "."))
spec = importlib.util.spec_from_file_location(
    "cache_prior_router_under_test",
    source_root / "vllm/model_executor/layers/fused_moe/router/cache_prior_router.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def queue(last_use: torch.Tensor) -> list[int]:
    resident = torch.nonzero(last_use >= 0, as_tuple=False).flatten()
    return resident[torch.argsort(last_use[resident], stable=True)].tolist()


try:
    module.begin_speculative_lru_block(
        torch.tensor([[0, 1], [2, 3]]),
        torch.ones((2, 2)),
        capacity=3,
        num_experts=6,
    )
except module.SpeculativeCacheOverflow as error:
    assert (error.required_experts, error.capacity, error.cache_misses) == (4, 3, 4)
else:
    raise AssertionError("overflow was not rejected")

mask = module.speculative_commit_mask(
    (4, 2),
    torch.tensor([[10, 11, 12, -1, -1], [20, -1, -1, -1, -1]]),
)
assert mask.tolist() == [True, True, True, False, False, True, False, False]

generator = torch.Generator().manual_seed(73)
random.seed(73)
for _ in range(500):
    num_experts = random.randint(4, 14)
    capacity = random.randint(2, num_experts)
    top_k = random.randint(1, min(3, capacity))
    tokens = random.randint(1, 8)
    resident_count = random.randint(0, capacity)
    pre_queue = torch.randperm(num_experts, generator=generator)[:resident_count].tolist()
    initial = torch.full((num_experts,), -1, dtype=torch.int64)
    if pre_queue:
        initial[pre_queue] = torch.arange(resident_count)
    ids = torch.stack(
        [torch.randperm(num_experts, generator=generator)[:top_k] for _ in range(tokens)]
    )
    priorities = torch.rand((tokens, top_k), generator=generator)
    if ids.unique().numel() > capacity:
        try:
            module.begin_speculative_lru_block(
                ids,
                priorities,
                capacity=capacity,
                num_experts=num_experts,
                last_use=initial,
                clock=resident_count,
            )
        except module.SpeculativeCacheOverflow:
            continue
        raise AssertionError("random overflow was not rejected")

    transaction = module.begin_speculative_lru_block(
        ids,
        priorities,
        capacity=capacity,
        num_experts=num_experts,
        last_use=initial,
        clock=resident_count,
    )
    admitted = pre_queue.copy()
    for row_ids, row_priorities in zip(ids, priorities, strict=True):
        for expert_id, _ in sorted(
            zip(row_ids.tolist(), row_priorities.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        ):
            if expert_id in admitted:
                admitted.remove(expert_id)
            elif len(admitted) == capacity:
                admitted.pop(0)
            admitted.append(expert_id)
    assert queue(transaction.admitted_last_use)[-capacity:] == admitted

    commit_mask = torch.rand(tokens, generator=generator) > 0.5
    actual, _ = module.commit_speculative_lru_block(transaction, commit_mask)
    committed = set(ids[commit_mask].flatten().tolist())
    newly_loaded = set(transaction.newly_loaded.nonzero().flatten().tolist())
    admitted_set = set(admitted)
    expected = [item for item in admitted if item in newly_loaded and item not in committed]
    expected.extend(
        item for item in pre_queue if item in admitted_set and item not in committed
    )
    for row_ids, row_priorities in zip(
        ids[commit_mask], priorities[commit_mask], strict=True
    ):
        for expert_id, _ in sorted(
            zip(row_ids.tolist(), row_priorities.tolist(), strict=True),
            key=lambda item: (-item[1], item[0]),
        ):
            if expert_id in expected:
                expected.remove(expert_id)
            expected.append(expert_id)
    assert queue(actual) == expected

print("speculative cache smoke passed")
