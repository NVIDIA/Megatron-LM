# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-draft-token state snapshots from the GDP decode kernel.

Speculative decoding runs `S` tokens per request in one decode step and then
learns how many of them were accepted. Rolling the recurrence back to the
accepted prefix needs the state *after each* of those tokens, which is what
`intermediate_states` asks the kernel to dump.

The reference here is the kernel itself, run one token at a time: a snapshot is
correct exactly when it equals the state cache after a decode step that stopped
at that token. That comparison is bitwise rather than approximate -- the
arithmetic and its order are identical on both sides, and the only difference
is where the state lands -- so a drifting snapshot cannot hide behind a
tolerance.
"""

import pytest
import torch

from megatron.core.ssm.ops.gdp.fused_recurrent import fused_recurrent_gated_delta_rule_update

# Householder copies per draft token. `M == 1` is the plain Gated Delta Net
# recurrence; anything above folds M steps into the sequence dimension per
# token, and the snapshot must land on the last of each group.
_HOUSEHOLDER = [1, 2, 3]

_HV = 4  # value heads
_K = 16  # key/query head dim
_V = 32  # value head dim, > BV so the state is split across programs


def _requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _make_inputs(batch, seq_len, num_householder, device="cuda", dtype=torch.float32):
    """Random `(q, k, v, g, beta)` for a `seq_len`-token, `M`-copy decode step."""
    torch.manual_seed(1234)
    t = seq_len * num_householder
    q = torch.randn(batch, t, _HV, _K, device=device, dtype=dtype)
    k = torch.randn(batch, t, _HV, _K, device=device, dtype=dtype)
    v = torch.randn(batch, t, _HV, _V, device=device, dtype=dtype)
    # `g` is a log decay: keep it negative so the state stays bounded.
    g = -torch.rand(batch, t, _HV, device=device, dtype=torch.float32)
    beta = torch.rand(batch, t, _HV, device=device, dtype=dtype)
    return q, k, v, g, beta


def _token_slice(tensor, token, num_householder):
    """The `num_householder` rows of the folded sequence belonging to `token`."""
    return tensor[:, token * num_householder : (token + 1) * num_householder]


@pytest.mark.internal
class TestGDPDecodeIntermediateStates:
    """Snapshot correctness for the speculative decode path."""

    @pytest.mark.parametrize("num_householder", _HOUSEHOLDER)
    @pytest.mark.parametrize("seq_len", [1, 2, 5])
    def test_snapshots_match_token_by_token_decode(self, num_householder, seq_len):
        """Snapshot `i` equals the state cache after decoding tokens `0..i`."""
        _requires_cuda()
        batch, num_slots = 3, 6
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)

        # Slots deliberately out of order, so a snapshot written at the batch
        # position instead of the cache slot fails here.
        state_indices = torch.tensor([4, 0, 3], dtype=torch.int32, device="cuda")
        initial_state = torch.randn(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)

        state = initial_state.clone()
        intermediate = torch.full(
            (num_slots, seq_len, _HV, _K, _V), float("nan"), device="cuda", dtype=torch.float32
        )
        out, _ = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            intermediate_states=intermediate,
            steps_per_token=num_householder,
        )

        # Replay the same step one token at a time. Each call leaves the state
        # cache holding exactly what the snapshot for that token should be.
        replay_state = initial_state.clone()
        for token in range(seq_len):
            replay_out, _ = fused_recurrent_gated_delta_rule_update(
                _token_slice(q, token, num_householder),
                _token_slice(k, token, num_householder),
                _token_slice(v, token, num_householder),
                g=_token_slice(g, token, num_householder),
                beta=_token_slice(beta, token, num_householder),
                state=replay_state,
                state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
            )
            assert torch.equal(
                intermediate[state_indices.long(), token], replay_state[state_indices.long()]
            ), f"snapshot for token {token} does not match a decode step that stopped there"
            # The outputs must agree too: snapshotting must not perturb the scan.
            assert torch.equal(_token_slice(out, token, num_householder), replay_out)

        assert torch.equal(state, replay_state)

    @pytest.mark.parametrize("num_householder", _HOUSEHOLDER)
    def test_last_snapshot_is_the_final_state(self, num_householder):
        """The final snapshot is the state the cache ends the step with, which is
        what a fully-accepted draft rolls back to (i.e. does not roll back)."""
        _requires_cuda()
        seq_len, batch, num_slots = 4, 2, 2
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)
        state_indices = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        state = torch.randn(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)
        intermediate = torch.empty(
            (num_slots, seq_len, _HV, _K, _V), device="cuda", dtype=torch.float32
        )

        fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            intermediate_states=intermediate,
            steps_per_token=num_householder,
        )

        assert torch.equal(intermediate[:, -1], state)

    @pytest.mark.parametrize("num_householder", _HOUSEHOLDER)
    def test_snapshots_do_not_change_the_scan(self, num_householder):
        """Passing the snapshot buffer must not perturb outputs or final state.

        A decode step under CUDA graph capture runs the same kernel whether or
        not speculation is on; the snapshot is a pure side effect.
        """
        _requires_cuda()
        seq_len, batch, num_slots = 3, 3, 4
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)
        state_indices = torch.tensor([2, 0, 3], dtype=torch.int32, device="cuda")
        initial_state = torch.randn(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)

        plain_state = initial_state.clone()
        plain_out, _ = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=plain_state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )

        snap_state = initial_state.clone()
        intermediate = torch.empty(
            (num_slots, seq_len, _HV, _K, _V), device="cuda", dtype=torch.float32
        )
        snap_out, _ = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=snap_state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            intermediate_states=intermediate,
            steps_per_token=num_householder,
        )

        assert torch.equal(plain_out, snap_out)
        assert torch.equal(plain_state, snap_state)

    @pytest.mark.parametrize("num_householder", _HOUSEHOLDER)
    def test_inputs_not_mutated_without_intermediates(self, num_householder):
        """Without a snapshot buffer, the kernel must not touch its inputs.

        Regression guard for the class of bug fixed in 6e5a8c1 for Mamba2's
        `selective_state_update`: its launcher substituted `x` as a dummy
        pointer when `intermediate_ssm_states` was None, so the
        `is not None` heuristic still enabled the dump, which collapsed its
        whole store grid onto `x` with all-zero strides -- corrupting the input
        and racing other programs' loads of it on every non-speculative decode
        step. This launcher passes the real `None` and zeroes only the strides;
        a future dummy pointer here would reproduce that bug, and fail here.
        """
        _requires_cuda()
        seq_len, batch, num_slots = 2, 4, 4
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)
        state_indices = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device="cuda")
        state = torch.randn(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)

        inputs = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        before = {name: tensor.clone() for name, tensor in inputs.items()}

        fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
        )
        torch.cuda.synchronize()

        for name, tensor in inputs.items():
            assert torch.equal(tensor, before[name]), (
                f"the decode kernel mutated input '{name}' when called without "
                "intermediate_states (dummy-pointer regression)"
            )

    def test_padding_requests_write_no_snapshot(self):
        """A `-1` slot leaves both the state cache and the snapshot buffer alone.

        Decode batches are padded up to the captured graph's shape, so padding
        rows run the recurrence over whatever the input buffer holds. They must
        not land anywhere a real request can read.
        """
        _requires_cuda()
        seq_len, num_householder, batch, num_slots = 3, 2, 4, 5
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)
        state_indices = torch.tensor([2, -1, 0, -1], dtype=torch.int32, device="cuda")

        initial_state = torch.randn(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)
        state = initial_state.clone()
        sentinel = torch.randn(
            (num_slots, seq_len, _HV, _K, _V), device="cuda", dtype=torch.float32
        )
        intermediate = sentinel.clone()

        out, _ = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            intermediate_states=intermediate,
            steps_per_token=num_householder,
        )

        # Padding rows produce zero output, as they do without snapshots.
        assert torch.count_nonzero(out[1]) == 0
        assert torch.count_nonzero(out[3]) == 0

        # Slots 1, 3 and 4 are named by no request: untouched in both buffers.
        untouched = [1, 3, 4]
        assert torch.equal(intermediate[untouched], sentinel[untouched])
        assert torch.equal(state[untouched], initial_state[untouched])
        # ... and the slots that were written are no longer the sentinel.
        assert not torch.equal(intermediate[[0, 2]], sentinel[[0, 2]])

    @pytest.mark.parametrize("num_householder", _HOUSEHOLDER)
    def test_cuda_graph_capture_and_replay(self, num_householder):
        """The snapshot stores are capturable and replay bit-identically.

        Every decode step in the engine is a graph replay, so the extra stores
        must add no host synchronization and no data-dependent shape. The
        comparison is against the same kernel launched eagerly, so any
        difference is the capture, not the arithmetic.
        """
        _requires_cuda()
        seq_len, batch, num_slots = 3, 3, 5
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)
        # One padding row, as a replayed graph captured at a larger batch has.
        state_indices = torch.tensor([4, -1, 1], dtype=torch.int32, device="cuda")
        initial_state = torch.randn(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)

        eager_state = initial_state.clone()
        eager_intermediate = torch.zeros(
            (num_slots, seq_len, _HV, _K, _V), device="cuda", dtype=torch.float32
        )
        eager_out, _ = fused_recurrent_gated_delta_rule_update(
            q,
            k,
            v,
            g=g,
            beta=beta,
            state=eager_state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            intermediate_states=eager_intermediate,
            steps_per_token=num_householder,
        )

        # Capture records these addresses, so replay reads its inputs from and
        # writes its outputs to exactly these buffers.
        graph_state = initial_state.clone()
        graph_intermediate = torch.zeros_like(eager_intermediate)

        def step():
            return fused_recurrent_gated_delta_rule_update(
                q,
                k,
                v,
                g=g,
                beta=beta,
                state=graph_state,
                state_indices=state_indices,
                use_qk_l2norm_in_kernel=True,
                intermediate_states=graph_intermediate,
                steps_per_token=num_householder,
            )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                graph_state.copy_(initial_state)
                step()
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        graph_state.copy_(initial_state)
        with torch.cuda.graph(graph):
            graph_out, _ = step()

        # Reset, so the values compared below come from the replay rather than
        # from the launch that happened during capture.
        graph_state.copy_(initial_state)
        graph_intermediate.zero_()
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(graph_out, eager_out)
        assert torch.equal(graph_state, eager_state)
        assert torch.equal(graph_intermediate, eager_intermediate)

    def test_rejects_mismatched_snapshot_buffer(self):
        """Shape and contract violations fail loudly rather than corrupting state."""
        _requires_cuda()
        seq_len, num_householder, batch, num_slots = 2, 2, 2, 3
        q, k, v, g, beta = _make_inputs(batch, seq_len, num_householder)
        state_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
        state = torch.zeros(num_slots, _HV, _K, _V, device="cuda", dtype=torch.float32)

        def run(intermediate, steps_per_token=num_householder, use_state=True):
            return fused_recurrent_gated_delta_rule_update(
                q,
                k,
                v,
                g=g,
                beta=beta,
                state=state if use_state else None,
                state_indices=state_indices if use_state else None,
                use_qk_l2norm_in_kernel=True,
                intermediate_states=intermediate,
                steps_per_token=steps_per_token,
            )

        good_shape = (num_slots, seq_len, _HV, _K, _V)

        # Too few token slots for the folded sequence length.
        with pytest.raises(AssertionError):
            run(torch.empty((num_slots, seq_len - 1, _HV, _K, _V), device="cuda"))

        # Fewer rows than the state cache has slots.
        with pytest.raises(AssertionError):
            run(torch.empty((num_slots - 1, seq_len, _HV, _K, _V), device="cuda"))

        # A step count the folded sequence does not divide evenly by.
        with pytest.raises(AssertionError):
            run(torch.empty(good_shape, device="cuda"), steps_per_token=3)

        # A non-contiguous trailing (K, V) block, which the kernel indexes flat.
        with pytest.raises(AssertionError):
            run(torch.empty((num_slots, seq_len, _HV, _V, _K), device="cuda").transpose(-1, -2))

        # Snapshots without the slot-indexed cache they shadow.
        with pytest.raises(AssertionError):
            run(torch.empty(good_shape, device="cuda"), use_state=False)
