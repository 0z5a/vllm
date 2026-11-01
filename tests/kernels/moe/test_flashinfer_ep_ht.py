# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HT prepare must not index routing metadata outside the received prefix."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize.flashinfer_ep_ht import (
    FlashInferEPHTPrepareAndFinalize,
)


@pytest.mark.parametrize("actual", [0, 1, 127, 128, 129, 256, -1, 257])
def test_prepare_masks_unreceived_metadata(monkeypatch, actual):
    capacity = 256
    recv_idx = torch.full((capacity, 2), 2**60, dtype=torch.int64)
    recv_weights = torch.full((capacity, 2), float("nan"))
    if 0 <= actual <= capacity:
        recv_idx[:actual] = torch.tensor([0, 1])
        recv_weights[:actual] = torch.tensor([0.25, 0.75])
        # Received non-local picks must retain their existing skip behavior.
        recv_idx[:actual:2, 1] = -1

    dispatched = SimpleNamespace(
        expert_tensors=torch.zeros(1, capacity, 2),
        recv_topk_idx=recv_idx,
        recv_topk_weights=recv_weights,
    )
    transport = ModuleType("flashinfer.moe_ep")
    transport.__dict__.update(
        DispatchInputParams=SimpleNamespace,
        HandleAlgoKnobNumReceivedTokens=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "flashinfer.moe_ep", transport)

    adapter = object.__new__(FlashInferEPHTPrepareAndFinalize)
    adapter.hidden_size = 2
    adapter._num_local_experts = 2
    adapter._l2g_cache = None

    def make_handle(topk_ids, weights, extra_knobs):
        extra_knobs[0].target.fill_(actual)
        return SimpleNamespace(dispatch=lambda params: dispatched)

    monkeypatch.setattr(adapter, "_make_handle", make_handle)
    a1 = torch.zeros(1, 2)
    weights = torch.ones(1, 2)
    ids = torch.zeros(1, 2, dtype=torch.int64)
    expert_map = torch.tensor([-1, 0, -1, 1], dtype=torch.int32)
    args = (a1, weights, ids, 4, expert_map, False, None)
    if not 0 <= actual <= capacity:
        with pytest.raises(RuntimeError, match="received-token count"):
            adapter.prepare(*args)
        return

    expert_x, _, _, mapped_ids, mapped_weights = adapter.prepare(*args)
    expected_rows = min(capacity, max(128, ((actual + 127) // 128) * 128))
    assert expert_x.shape == (expected_rows, 2)
    local_to_global = torch.tensor([1, 3])
    expected_ids = torch.where(
        recv_idx[:actual] >= 0,
        local_to_global[recv_idx[:actual].clamp_min(0)],
        0,
    )
    torch.testing.assert_close(mapped_ids[:actual], expected_ids)
    torch.testing.assert_close(mapped_weights[:actual], recv_weights[:actual])
    assert torch.all(mapped_ids[actual:] == 0)
    assert torch.all(mapped_weights[actual:] == 0)
    # Masking must not mutate the transport's address-stable receive buffers.
    assert torch.all(recv_idx[actual:] == 2**60)
    assert torch.isnan(recv_weights[actual:]).all()
