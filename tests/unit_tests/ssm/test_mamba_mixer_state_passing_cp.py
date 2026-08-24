# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end tests for ``MambaMixer`` with state-passing context parallelism.

Each state-passing load-balancing mode is compared against a single-rank
full-sequence ``MambaMixer`` and against the existing all-to-all CP path, for
the output, the input gradient, and every parameter gradient.
"""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.ssm.mamba_state_passing_cp_utils import (
    STATE_PASSING_CP_MODES,
    MambaModelShape,
    assert_all_close_rms,
    broadcast_module_parameters,
    build_mamba_mixer,
    collect_parameter_grads,
    select_balanced_cp_shard,
    set_state_passing_cp_mode,
)
from tests.unit_tests.test_utilities import Utils

try:
    import causal_conv1d  # noqa: F401
    import mamba_ssm  # noqa: F401

    from megatron.core.ssm.mamba_mixer import HAVE_MAMBA_STATE_PASSING_CP
except ImportError:
    HAVE_MAMBA_STATE_PASSING_CP = False

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not HAVE_MAMBA_STATE_PASSING_CP, reason="mamba_ssm and causal_conv1d are required"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
]


def _run_mixer(mixer, mode, hidden_states, grad_output, parameter_reduce_group=None):
    """Run one forward/backward of ``mixer`` in ``mode`` and collect all gradients."""
    mixer.zero_grad(set_to_none=True)
    local_input = hidden_states.detach().clone().requires_grad_(True)
    set_state_passing_cp_mode(mixer, mode)
    output, output_bias = mixer(local_input)
    assert output_bias is None, "this test expects a bias-free output projection"
    torch.autograd.backward(output, grad_output)
    return (
        output.detach(),
        local_input.grad.detach(),
        collect_parameter_grads(mixer, parameter_reduce_group),
    )


@pytest.mark.parametrize("cp_size", (2, 4))
@pytest.mark.parametrize("batch_size,sequence_length", ((1, 2048), (3, 4096)))
def test_mixer_state_passing_cp_matches_full_sequence_and_a2a(cp_size, batch_size, sequence_length):
    """All state-passing modes must match the full-sequence and A2A CP references."""
    if Utils.world_size % cp_size != 0:
        pytest.skip(f"world size {Utils.world_size} is not a multiple of cp_size {cp_size}")
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        torch.manual_seed(1234)
        model_parallel_cuda_manual_seed(1234, force_reset_rng=True)
        cp_group = parallel_state.get_context_parallel_group()
        tp_group = parallel_state.get_tensor_model_parallel_group()
        assert tp_group.size() == 1, "the full-sequence reference needs a size-1 CP group"
        rank = cp_group.rank()
        device = torch.device(torch.cuda.current_device())

        shape = MambaModelShape()
        # ``virtual`` mode halves the segment length, and every causal segment
        # must align with the SSD chunk size.
        assert sequence_length % (4 * cp_size * shape.chunk_size) == 0

        mixer = build_mamba_mixer(shape, cp_group, tp_group, load_balancing="permute_p2p")
        broadcast_module_parameters(mixer, cp_group)
        # A size-1 CP group makes this mixer process the whole sequence.
        reference_mixer = build_mamba_mixer(
            shape, tp_group, tp_group, load_balancing="permute_p2p", use_state_passing_cp=False
        )
        reference_mixer.load_state_dict(mixer.state_dict())

        generator = torch.Generator(device=device).manual_seed(5678)
        global_input = torch.randn(
            sequence_length,
            batch_size,
            shape.hidden_size,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        global_grad_output = torch.randn(
            global_input.shape, device=device, dtype=torch.bfloat16, generator=generator
        )
        local_input = select_balanced_cp_shard(global_input, rank, cp_size)
        local_grad_output = select_balanced_cp_shard(global_grad_output, rank, cp_size)

        reference_output, reference_input_grad, reference_parameter_grads = _run_mixer(
            reference_mixer, None, global_input, global_grad_output
        )
        expected_output = select_balanced_cp_shard(reference_output, rank, cp_size)
        expected_input_grad = select_balanced_cp_shard(reference_input_grad, rank, cp_size)

        # The existing all-to-all CP path is the second reference: it is the
        # numerical behaviour production already ships.
        a2a_output, a2a_input_grad, a2a_parameter_grads = _run_mixer(
            mixer, None, local_input, local_grad_output, cp_group
        )
        checks = {
            "a2a output vs full sequence": (a2a_output, expected_output),
            "a2a input grad vs full sequence": (a2a_input_grad, expected_input_grad),
        }
        for mode in STATE_PASSING_CP_MODES:
            output, input_grad, parameter_grads = _run_mixer(
                mixer, mode, local_input, local_grad_output, cp_group
            )
            checks[f"{mode} output vs full sequence"] = (output, expected_output)
            checks[f"{mode} input grad vs full sequence"] = (input_grad, expected_input_grad)
            checks[f"{mode} output vs a2a"] = (output, a2a_output)
            checks[f"{mode} input grad vs a2a"] = (input_grad, a2a_input_grad)
            assert parameter_grads.keys() == reference_parameter_grads.keys()
            for name, gradient in parameter_grads.items():
                checks[f"{mode} {name} vs full sequence"] = (
                    gradient,
                    reference_parameter_grads[name],
                )
                checks[f"{mode} {name} vs a2a"] = (gradient, a2a_parameter_grads[name])

        # Asserted in one place so that a mismatch fails every rank identically
        # instead of leaving some ranks inside a collective.
        assert_all_close_rms(checks, cp_group)
    finally:
        Utils.destroy_model_parallel()
