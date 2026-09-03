# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V2 MTP diagnostic that keeps two identical ReplaySSM rows live."""

import os

import vllm.envs as envs

from ...models.utils import check_logprobs_close
from .test_replayssm_decode import PROMPTS


def test_replayssm_flashinfer_mtp_v2_without_batch_compaction(vllm_runner, monkeypatch):
    model = os.environ["REPLAYSSM_MODEL"]
    prompts = [PROMPTS[1], PROMPTS[1]]
    common = dict(
        max_model_len=1024,
        trust_remote_code=True,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_backend="flashinfer",
        speculative_config={"method": "mtp", "num_speculative_tokens": 3},
    )

    try:
        with monkeypatch.context() as patch:
            patch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
            envs.disable_envs_cache()
            with vllm_runner(
                model, use_replayssm=True, replayssm_buffer_len=16, **common
            ) as llm:
                outputs = llm.generate_greedy_logprobs(
                    prompts, max_tokens=32, num_logprobs=5
                )
    finally:
        envs.disable_envs_cache()

    assert len(outputs) == 2
    row_0_ids = outputs[0][0]
    row_1_ids = outputs[1][0]
    matching_prefix = 0
    for row_0_id, row_1_id in zip(row_0_ids, row_1_ids):
        if row_0_id != row_1_id:
            break
        matching_prefix += 1
    print(
        "REPLAYSSM_COMPACTION_DIAGNOSTIC",
        {
            "matching_prefix": matching_prefix,
            "row_0": row_0_ids,
            "row_1": row_1_ids,
        },
    )

    check_logprobs_close(
        outputs_0_lst=outputs[:1],
        outputs_1_lst=outputs[1:],
        name_0="replayssm_mtp_v2_row_0",
        name_1="replayssm_mtp_v2_row_1",
    )
