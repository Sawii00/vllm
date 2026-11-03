# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A pipeline stage must start on a layer that computes its own top-k."""

from types import SimpleNamespace

import pytest

from vllm.distributed.utils import get_pp_indices
from vllm.models.deepseek_v32.nvidia.model import _check_pp_stage_start


def _glm_layers(num_layers: int) -> list[SimpleNamespace]:
    # GLM-5.3 (index_topk_freq 4, index_skip_topk_offset 3): layers 0, 1, 2,
    # 6, 10, ... carry an indexer; the others reuse the previous top-k.
    producers = {0, 1, 2} | set(range(6, num_layers, 4))
    return [
        SimpleNamespace(
            self_attn=SimpleNamespace(indexer=object() if i in producers else None)
        )
        for i in range(num_layers)
    ]


def _check_partition(monkeypatch, partition: list[int]) -> None:
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", ",".join(map(str, partition)))
    num_layers = sum(partition)
    layers = _glm_layers(num_layers)
    for rank in range(len(partition)):
        start, _ = get_pp_indices(num_layers, rank, len(partition))
        _check_pp_stage_start(start, layers)


@pytest.mark.parametrize(
    "partition", [[14, 12, 13], [26, 28, 24], [18, 12, 12, 12, 12, 12]]
)
def test_stages_starting_on_indexer_layers_pass(monkeypatch, partition):
    _check_partition(monkeypatch, partition)


@pytest.mark.parametrize(
    "partition,bad_layer", [([13, 13, 13], 13), ([26, 26, 26], 52)]
)
def test_stage_starting_on_reusing_layer_fails(monkeypatch, partition, bad_layer):
    with pytest.raises(ValueError, match=f"starts at layer {bad_layer},"):
        _check_partition(monkeypatch, partition)
