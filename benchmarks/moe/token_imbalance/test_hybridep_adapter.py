"""CPU checks for the adapter boundary; GPU qualification is a separate finite run."""

import pytest
import torch
from hybridep_adapter import Geometry, canonical_routes, numerical_error


@pytest.mark.parametrize(
    "kwargs", [{"ep": 1}, {"ep": 65}, {"experts": 7}, {"topk": 9}, {"tokens": 0}, {"hidden": 127}]
)
def test_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        Geometry(**kwargs).validate()


def test_full_and_compact_routes_preserve_gate_gradients():
    scores = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], requires_grad=True)
    ids = torch.tensor([[3, 1], [2, 0]])
    dense = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, ids, True)
    full_ids, full = canonical_routes(scores, dense, 2)
    compact_ids, compact = canonical_routes(scores.gather(1, ids), ids, 2)
    assert torch.equal(full_ids, compact_ids)
    torch.testing.assert_close(full, compact, rtol=0, atol=0)
    full_grad = torch.autograd.grad(full.sum(), scores, retain_graph=True)[0]
    compact_grad = torch.autograd.grad(compact.sum(), scores)[0]
    torch.testing.assert_close(full_grad, compact_grad, rtol=0, atol=0)


def test_invalid_dense_topk_rejected():
    with pytest.raises(AssertionError):
        canonical_routes(torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool), 2)


def test_numerical_gate_checks_zeros_nonfinite_and_relative_error():
    assert numerical_error(torch.zeros(2), torch.zeros(2))["relative_l2"] == 0
    assert numerical_error(torch.ones(2), torch.zeros(2))["relative_l2"] == float("inf")
    assert numerical_error(torch.tensor([float("nan")]), torch.ones(1))["relative_l2"] == float(
        "inf"
    )
    assert numerical_error(torch.full((2,), 1.01), torch.ones(2))["relative_l2"] == pytest.approx(
        0.01, abs=1e-6
    )
