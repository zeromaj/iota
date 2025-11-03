import pytest
import torch

from subnet.utils.vector_utils import (
    extract_optimizer_state_section,
    flatten_optimizer_state,
)


def _make_optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    for param in model.parameters():
        param.grad = torch.ones_like(param)
    optimizer.step()

    return optimizer


def test_extract_optimizer_state_section_matches_flatten():
    optimizer = _make_optimizer()

    flat_state, _, _ = flatten_optimizer_state(optimizer, device="cpu", dtype=torch.bfloat16)
    extracted = extract_optimizer_state_section(optimizer, 0, flat_state.numel())

    assert extracted.dtype is torch.bfloat16
    assert extracted.device.type == "cpu"
    torch.testing.assert_close(extracted.float(), flat_state.float())


def test_extract_optimizer_state_section_partial_slice():
    optimizer = _make_optimizer()
    flat_state, _, _ = flatten_optimizer_state(optimizer, device="cpu", dtype=torch.bfloat16)

    start_idx = 3
    end_idx = 8
    extracted = extract_optimizer_state_section(optimizer, start_idx, end_idx)

    expected = flat_state[start_idx:end_idx]
    torch.testing.assert_close(extracted.float(), expected.float())


def test_extract_optimizer_state_section_invalid_bounds():
    optimizer = _make_optimizer()
    flat_state, _, _ = flatten_optimizer_state(optimizer, device="cpu", dtype=torch.bfloat16)
    total = flat_state.numel()

    with pytest.raises(ValueError):
        extract_optimizer_state_section(optimizer, -1, 1)

    with pytest.raises(ValueError):
        extract_optimizer_state_section(optimizer, 5, 4)

    with pytest.raises(ValueError):
        extract_optimizer_state_section(optimizer, 0, total + 1)
