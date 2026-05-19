# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from dataclasses import dataclass, replace
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec


def compute_varlen_chunk_metadata(
    query_start_loc: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build chunk-aligned, variable-length metadata used by Mamba2 SSD kernels.

    Given per-sequence cumulative token starts `query_start_loc` of shape [B+1]
    and a physical `chunk_size`, returns three tensors on the same device:
      - cu_chunk_seqlens:  (nchunks+1,) int32   exclusive prefix-sum of
        logical-chunk lengths (each logical chunk never crosses a sequence or
        physical-chunk boundary).
      - last_chunk_indices: (B,)       int32   index of the last logical chunk
        for each sequence (=-1 for empty sequences).
      - seq_idx_chunks:     (nchunks,) int32   sequence index for each logical
        chunk in order.

    This is intentionally lightweight and CPU-side; it mirrors the metadata
    produced by the V1 Mamba2 meta-data builder and is exported so tests
    (and other callers) can avoid duplicating the logic.
    """
    assert query_start_loc.ndim == 1, "query_start_loc must be 1-D [B+1]"
    assert int(query_start_loc[0].item()) == 0, "query_start_loc[0] must be 0"
    device = query_start_loc.device

    qsl64 = query_start_loc.to(torch.int64)
    starts = qsl64[:-1].tolist()
    ends = qsl64[1:].tolist()
    total = int(qsl64[-1].item())

    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for b, (s, e) in enumerate(zip(starts, ends)):
        if e <= s:
            # empty sequence
            continue
        pos = s
        while pos < e:
            # split at both sequence boundaries and physical chunk boundaries
            room = chunk_size - (pos % chunk_size)
            take = min(room, e - pos)
            chunk_lens.append(int(take))
            seq_idx_chunks.append(b)
            last_chunk_indices[b] = len(chunk_lens) - 1
            pos += take

    # Exclusive prefix sum over logical-chunk lengths
    if chunk_lens:
        cu_chunk_seqlens = torch.tensor(
            [0] + list(itertools.accumulate(chunk_lens)),
            device=device,
            dtype=torch.int32,
        )
        # Final boundary must equal total tokens
        assert int(cu_chunk_seqlens[-1].item()) == total
    else:
        cu_chunk_seqlens = torch.tensor([0], device=device, dtype=torch.int32)

    last_chunk_indices_t = (
        torch.tensor(last_chunk_indices, device=device, dtype=torch.int32)
        if len(starts) > 0
        else torch.empty((0,), device=device, dtype=torch.int32)
    )
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, device=device, dtype=torch.int32)
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


class Mamba2AttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MAMBA2_ATTN"

    @staticmethod
    def get_builder_cls() -> type["Mamba2AttentionMetadataBuilder"]:
        return Mamba2AttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


@dataclass
class Mamba2AttentionMetadata(BaseMambaAttentionMetadata):
    prep_initial_states: bool = False
    chunk_size: int = 0

    # Chunk-related metadata (only for prefill)
    seq_idx_p: torch.Tensor | None = None

    # SSM checkpointing metadata
    cache_buf_idx_d: torch.Tensor | None = None
    prev_num_accepted_tokens_d: torch.Tensor | None = None


class Mamba2AttentionMetadataBuilder(
    BaseMambaAttentionMetadataBuilder[Mamba2AttentionMetadata]
):
    metadata_cls = Mamba2AttentionMetadata

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        chunk_size = vllm_config.model_config.get_mamba_chunk_size()
        assert chunk_size is not None, (
            "chunk_size needs to be set in the model config for Mamba2 models"
        )
        self.chunk_size: int = chunk_size

        self.cache_buf_idx_d: torch.Tensor | None = None
        self.prev_num_accepted_tokens_d: torch.Tensor | None = None

        self._last_state_indices_tensor_d: torch.Tensor | None = None
        self._last_query_start_loc_d: torch.Tensor | None = None
        self._last_num_decodes: int = 0

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs: Any,
    ) -> Mamba2AttentionMetadata:
        # Finalize the previous step's SSM checkpoint counters before
        # constructing this step's metadata. No-op on the first call and
        # on any step that had no decode rows last time.
        self.apply_post_step()

        common = self._compute_common_metadata(
            common_attn_metadata, num_accepted_tokens=kwargs.get("num_accepted_tokens")
        )

        seq_idx_p = None
        cu_chunk_seqlen_p = None
        last_chunk_indices_p = None
        prep_initial_states = False

        # Compute seq_idx for prefill only
        if common.num_prefills > 0:
            prep_initial_states = (
                torch.any(common.has_initial_states_p).item()
                if common.has_initial_states_p is not None
                else False
            )

            cu_chunk_seqlen_p, seq_idx_p, last_chunk_indices_p = (
                self._build_chunk_metadata_tensors(
                    self.chunk_size,
                    common,
                    common_attn_metadata,
                )
            )

        if (
            self.cache_buf_idx_d is None
            and self.vllm_config.cache_config.mamba_ssm_checkpoint_interval > 1
        ):
            num_blocks = self.vllm_config.cache_config.num_gpu_blocks
            assert num_blocks is not None, (
                "cache_config.num_gpu_blocks must be set before "
                "Mamba2AttentionMetadataBuilder is constructed (profiling "
                "should have run already)."
            )
            self.cache_buf_idx_d = torch.zeros(
                (num_blocks,),
                dtype=torch.int32,
                device=self.device,
            )
            self.prev_num_accepted_tokens_d = torch.zeros(
                (num_blocks,),
                dtype=torch.int32,
                device=self.device,
            )

        self._last_state_indices_tensor_d = common.state_indices_tensor_d
        self._last_query_start_loc_d = common.query_start_loc_d
        self._last_num_decodes = common.num_decodes

        return replace(
            common,
            prep_initial_states=prep_initial_states,
            chunk_size=self.chunk_size,
            seq_idx_p=seq_idx_p,
            cu_chunk_seqlen_p=cu_chunk_seqlen_p,
            last_chunk_indices_p=last_chunk_indices_p,
            cache_buf_idx_d=self.cache_buf_idx_d,
            prev_num_accepted_tokens_d=self.prev_num_accepted_tokens_d,
        )

    def zero_blocks(self, block_ids: list[int]) -> None:
        valid = [b for b in block_ids]
        idx = torch.tensor(valid, dtype=torch.int64, device=self.cache_buf_idx_d.device)
        self.cache_buf_idx_d.index_fill_(0, idx, 0)
        self.prev_num_accepted_tokens_d.index_fill_(0, idx, 0)

    def apply_post_step(
        self,
    ) -> None:
        if self.cache_buf_idx_d is None or self._last_state_indices_tensor_d is None or self._last_num_decodes == 0:
            return

        num_decodes = self._last_num_decodes
        state_indices_tensor_d = self._last_state_indices_tensor_d
        query_start_loc_d = self._last_query_start_loc_d

        L = self.vllm_config.cache_config.mamba_ssm_checkpoint_interval

        query_start_loc_d = self._last_query_start_loc_d[: num_decodes + 1]

        slots = (
            state_indices_tensor_d[:num_decodes, 0].contiguous().to(torch.int64)
        )
        seq_lens = (
            query_start_loc_d[1 : num_decodes + 1]
            - query_start_loc_d[:num_decodes]
        )

        total_seq_lens = seq_lens + self.prev_num_accepted_tokens_d[slots]

        mask = total_seq_lens > L
        self.cache_buf_idx_d[slots[mask]] ^= 1

        new_prev = torch.where(mask, seq_lens, total_seq_lens)
        self.prev_num_accepted_tokens_d.scatter_(0, slots, new_prev)
