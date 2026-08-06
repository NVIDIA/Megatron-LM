# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""2-hop routing state machine for coordinator-native disaggregation (CPU)."""

import pytest

from megatron.core.inference.disaggregation.coordinator_routing import (
    DisaggRouter,
    DisaggRouting,
    make_disagg_router,
    register_disagg_router,
)


def test_router_registry_and_abc():
    # Default resolves to the round-robin DisaggRouting (a DisaggRouter).
    r = make_disagg_router("round_robin")
    assert isinstance(r, DisaggRouting) and isinstance(r, DisaggRouter)
    with pytest.raises(KeyError):
        make_disagg_router("does-not-exist")

    # A custom router registers by name and resolves (the Dynamo-router seam).
    class _Custom(DisaggRouter):
        def register(self, identity, role):
            pass

        def remove(self, identity):
            pass

        def route_submit(self, request_id):
            return "p"

        def route_prefill_done(self, request_id):
            return ("p", "d")

        def forget(self, request_id):
            pass

    register_disagg_router("custom_test", _Custom)
    assert isinstance(make_disagg_router("custom_test"), _Custom)

    # The ABC enforces the contract -- a partial implementation can't instantiate.
    class _Incomplete(DisaggRouter):
        def register(self, identity, role):
            pass

    with pytest.raises(TypeError):
        _Incomplete()


def _routing(n_prefill=1, n_decode=2):
    r = DisaggRouting()
    for i in range(n_prefill):
        r.register(f"p{i}", "prefill")
    for i in range(n_decode):
        r.register(f"d{i}", "decode")
    return r


def test_registration_partitions_by_role():
    r = DisaggRouting()
    r.register("p0", "prefill")
    r.register("d0", "decode")
    r.register("p0", "prefill")  # idempotent
    assert r.prefill_engines == ["p0"] and r.decode_engines == ["d0"]
    with pytest.raises(ValueError):
        r.register("x", "both")


def test_submit_goes_to_prefill_then_prefill_done_picks_decode_round_robin():
    r = _routing(n_prefill=1, n_decode=2)
    # every submit routes to the single prefill engine
    assert [r.route_submit(i) for i in range(3)] == ["p0", "p0", "p0"]
    # prefill-done fans across the decode pool, and remembers the pairing
    assert r.route_prefill_done(0) == ("p0", "d0")
    assert r.route_prefill_done(1) == ("p0", "d1")
    assert r.route_prefill_done(2) == ("p0", "d0")


def test_submit_round_robins_across_multiple_prefill():
    r = _routing(n_prefill=2, n_decode=2)
    # submits fan across the prefill pool...
    assert [r.route_submit(i) for i in range(4)] == ["p0", "p1", "p0", "p1"]
    # ...and each request's KV is sourced from the prefill that actually ran it.
    assert r.route_prefill_done(0)[0] == "p0"
    assert r.route_prefill_done(1)[0] == "p1"
    assert r.route_prefill_done(2)[0] == "p0"


def test_forget_clears_request_state():
    r = _routing()
    r.route_submit(7)
    assert r.route_prefill_done(7)[0] == "p0"  # the prefill that ran it
    r.forget(7)
    # forget dropped the pairing -- the prefill is no longer remembered
    assert r.route_prefill_done(7)[0] is None


def test_remove_engine_drops_from_pool():
    r = _routing(n_prefill=1, n_decode=2)
    r.remove("d0")
    assert r.decode_engines == ["d1"]
    assert all(r.route_prefill_done(i)[1] == "d1" for i in range(3))  # only d1 left


def test_route_without_engines_raises():
    r = DisaggRouting()
    with pytest.raises(RuntimeError):
        r.route_submit(0)
    r.register("p0", "prefill")
    r.route_submit(0)
    with pytest.raises(RuntimeError):
        r.route_prefill_done(0)  # no decode engine
