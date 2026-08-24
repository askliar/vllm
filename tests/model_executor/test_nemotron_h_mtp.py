# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.models.nemotron_h_mtp import get_mtp_layer_config
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig


def make_config(**kwargs) -> NemotronHConfig:
    return NemotronHConfig(
        num_hidden_layers=1,
        hybrid_override_pattern="*",
        **kwargs,
    )


def test_mtp_attention_defaults_when_optional_fields_are_absent():
    config = make_config(mtp_hybrid_override_pattern="*E")
    del config.mtp_window_size
    del config.mtp_softmax_type

    attention_config = get_mtp_layer_config(config, "*")

    assert attention_config.sliding_window is None
    assert attention_config.mtp_attention_softmax_type == "vanilla"
    assert config.sliding_window is None


def test_mtp_attention_uses_configured_sliding_window():
    config = make_config(
        mtp_hybrid_override_pattern="*E",
        mtp_window_size=[1024, 0],
    )
    del config.mtp_softmax_type

    attention_config = get_mtp_layer_config(config, "*")
    expert_config = get_mtp_layer_config(config, "E")

    assert attention_config.sliding_window == 1024
    assert attention_config.mtp_attention_softmax_type == "vanilla"
    assert expert_config.sliding_window is None
    assert expert_config.mtp_attention_softmax_type == "vanilla"
    assert config.sliding_window is None


def test_mtp_window_size_requires_left_and_right_values():
    with pytest.raises(AssertionError, match="mtp_window_size"):
        make_config(mtp_hybrid_override_pattern="*E", mtp_window_size=[1024])


def test_mtp_pattern_requires_exactly_one_attention_layer():
    with pytest.raises(AssertionError, match="exactly one attention layer"):
        make_config(mtp_hybrid_override_pattern="**E")
