# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.nemotron_h_mtp import (
    NemotronHMTP,
    get_mtp_layer_config,
)
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig


def make_config(**kwargs) -> NemotronHConfig:
    return NemotronHConfig(
        num_hidden_layers=1,
        hybrid_override_pattern="*",
        **kwargs,
    )


def test_mtp_attention_layers_receive_sink_softmax_type():
    config = make_config(
        mtp_hybrid_override_pattern="W*E",
        mtp_window_size=[1024, 0],
        mtp_softmax_type="learnable",
    )

    sliding_config = get_mtp_layer_config(config, "W")
    global_config = get_mtp_layer_config(config, "*")
    moe_config = get_mtp_layer_config(config, "E")

    assert sliding_config.mtp_attention_softmax_type == "learnable"
    assert global_config.mtp_attention_softmax_type == "learnable"
    assert moe_config.mtp_attention_softmax_type == "vanilla"


def test_mtp_loader_requires_learnable_sink_weights():
    model = NemotronHMTP.__new__(NemotronHMTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        model_type="nemotron_h",
        n_routed_experts=None,
        mtp_softmax_type="learnable",
    )
    model.model = SimpleNamespace(pattern_str="W*")

    with pytest.raises(ValueError, match="sink tensors were not loaded"):
        model.load_weights([])
