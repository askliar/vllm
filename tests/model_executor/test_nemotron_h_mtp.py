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


def test_mtp_sliding_attention_uses_its_window_without_affecting_global_attention():
    config = make_config(
        mtp_hybrid_override_pattern="W*E",
        mtp_window_size=[1024, 0],
    )

    sliding_config = get_mtp_layer_config(config, "W")
    global_config = get_mtp_layer_config(config, "*")

    assert sliding_config.sliding_window == 1024
    assert global_config.sliding_window is None
    assert config.sliding_window is None


def test_mtp_sliding_attention_requires_window_size():
    with pytest.raises(AssertionError, match="mtp_window_size"):
        make_config(mtp_hybrid_override_pattern="W")
