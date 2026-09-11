# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

from vllm.model_executor.layers.mamba.gdn import replayssm
from vllm.model_executor.layers.mamba.mamba_utils import gdn_replayssm_geometry
from vllm.model_executor.layers.mamba.ops import ssu_dispatch


@pytest.mark.parametrize("drafts,expected", [(0, (1, 16)), (3, (4, 32)), (7, (8, 32))])
def test_gdn_replayssm_geometry(drafts, expected):
    assert gdn_replayssm_geometry(drafts) == expected


@pytest.mark.parametrize("drafts", [-1, 1, 2, 4, 6, 8, 12])
def test_gdn_replayssm_geometry_rejects_unsupported(drafts):
    with pytest.raises(ValueError, match="supports 0, 3, or 7"):
        gdn_replayssm_geometry(drafts)


@pytest.fixture
def loaders(monkeypatch):
    stp, mtp, materialize = Mock(), Mock(), Mock()
    monkeypatch.setattr(replayssm, "_load_gdn_replayssm_stp_kernel", stp)
    monkeypatch.setattr(replayssm, "_load_gdn_replayssm_mtp_kernel", mtp)
    monkeypatch.setattr(ssu_dispatch, "_load_gdn_replayssm_materialize", materialize)
    return stp, mtp, materialize


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("needs_materializer", [False, True])
def test_gdn_replayssm_dependency_selection(loaders, width, needs_materializer):
    stp, mtp, materialize = loaders
    replayssm.check_gdn_replayssm_dependencies(width, needs_materializer)
    assert stp.call_count == int(width == 1)
    assert mtp.call_count == int(width != 1)
    assert materialize.call_count == int(needs_materializer)


@pytest.mark.parametrize(
    "width,needs_materializer,loader_index,api",
    [
        (1, False, 0, "gated_delta_rule_stp_ucache_flush"),
        (4, False, 1, "gated_delta_rule_mtp_ucache_flush"),
        (8, True, 1, "gated_delta_rule_mtp_ucache_flush"),
        (1, True, 2, "gdn_prefix_materialize"),
    ],
)
def test_gdn_replayssm_dependency_error(
    loaders, width, needs_materializer, loader_index, api
):
    error = ImportError("required export missing")
    loaders[loader_index].side_effect = error
    mode = "align" if needs_materializer else "none"
    with pytest.raises(
        ImportError, match=rf"width={width}, cache mode={mode}.*{api}"
    ) as exc:
        replayssm.check_gdn_replayssm_dependencies(width, needs_materializer)
    assert exc.value.__cause__ is error
