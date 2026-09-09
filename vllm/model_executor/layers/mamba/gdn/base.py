# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PretrainedConfig

from vllm.config import (
    VllmConfig,
)
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum


class GatedDeltaNetAttention(PluggableLayer, MambaBase):
    """Base class for GatedDeltaNet attention layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.quant_config = vllm_config.quant_config
        self.speculative_config = vllm_config.speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )
        self.use_replayssm = vllm_config.is_gdn_replayssm_enabled()
        self.use_flashinfer_replayssm = self.use_replayssm
        self.replayssm_buffer_len = (
            self.cache_config.replayssm_buffer_len if self.use_replayssm else None
        )
        self.replayssm_executed_query_width = (
            8 if self.num_spec == 7 else 4 if self.use_replayssm else None
        )
        if self.replayssm_executed_query_width is not None:
            self.register_buffer(
                "_replayssm_offsets",
                torch.arange(self.replayssm_executed_query_width, dtype=torch.int32),
                persistent=False,
            )
        self.replayssm_cache = (
            tuple(torch.tensor([]) for _ in range(3)) if self.use_replayssm else ()
        )
        self._replayssm_ring_start = torch.empty(0, dtype=torch.int32)
        self._replayssm_prev_num_accepted = torch.empty(0, dtype=torch.int32)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

    def get_replayssm_state_dtype(self) -> tuple[torch.dtype, ...]:
        if not self.use_replayssm:
            return ()
        return MambaStateDtypeCalculator.gated_delta_net_replayssm_ring_dtypes(
            self.model_config.dtype
        )
