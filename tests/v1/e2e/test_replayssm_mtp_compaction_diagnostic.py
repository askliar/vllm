# SPDX-License-Identifier: Apache-2.0
"""Temporary V2 MTP diagnostic that keeps both request rows live."""

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
            with vllm_runner(model, **common) as llm:
                baseline = llm.generate_greedy_logprobs(
                    prompts, max_tokens=32, num_logprobs=5
                )
            with vllm_runner(
                model, use_replayssm=True, replayssm_buffer_len=16, **common
            ) as llm:
                replay = llm.generate_greedy_logprobs(
                    prompts, max_tokens=32, num_logprobs=5
                )
    finally:
        envs.disable_envs_cache()

    for baseline_output, replay_output in zip(baseline, replay):
        baseline_ids = baseline_output[0]
        replay_ids = replay_output[0]
        matching_prefix = 0
        for baseline_id, replay_id in zip(baseline_ids, replay_ids):
            if baseline_id != replay_id:
                break
            matching_prefix += 1
        print(
            "REPLAYSSM_COMPACTION_DIAGNOSTIC",
            {
                "matching_prefix": matching_prefix,
                "baseline": baseline_ids,
                "replay": replay_ids,
            },
        )

    check_logprobs_close(
        outputs_0_lst=baseline,
        outputs_1_lst=replay,
        name_0="baseline_mtp_v2_no_compaction",
        name_1="replayssm_mtp_v2_no_compaction",
    )
