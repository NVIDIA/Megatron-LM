# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix Mamba backends against dense per-row MambaLayer execution.

Each case runs one MambaLayer twice: as the dense rows ``[prompt, completion_g]`` through the
ordinary Mamba forward, and as one shared-prefix star or forest through a shared backend. Outputs,
input gradients (prompt copies summed) and all parameter gradients must agree.

Tolerances (whole-tensor relative L2 unless noted):

* FP32. The layer uses local (torch) projections with TF32 disabled, so every GEMM is IEEE FP32;
  shared and dense execution then differ only in summation order. Measured on GB200 over this
  whole matrix: outputs <= 7e-8, input gradients <= 2.5e-6, parameter gradients <= 4.7e-6.
  ``FP32_TOLERANCE = 5e-5`` sits 10x above that floor and 10x below the smallest defect it must
  catch: the channel-first ``causal_conv1d`` backward (5e-4..2e-3 here); TF32 GEMMs alone give
  1.6e-4..3.5e-4.
* BF16. There is no fixed floor. The dense BF16 run and the shared BF16 run are both compared
  with the FP32 dense run of the same, BF16-representable, weights, and shared execution must be
  within ``BF16_RATIO = 1.5`` of the dense BF16 error. Measured ratios are 0.91..1.08. The
  per-token maximum input-gradient error is also compared, because a defect confined to a few
  positions disappears in a whole-tensor norm: the BF16 ``causal_conv1d`` length class gives
  ratios of 2.5..6.0 there and at most 1.08 in the whole-tensor metrics.
"""

import pytest
import torch

from tests.unit_tests.models.hybrid.shared_prefix_test_utils import (
    MAMBA_BACKENDS,
    LayerProblemData,
    SharedPrefixProblem,
    build_local_mamba_layer,
    canonical_dense_run,
    canonical_star_run,
    compare_canonical,
    low_precision_copy,
    round_params_to,
    run_layer,
    shared_mamba_layer_forward,
)
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("mamba_ssm")
pytest.importorskip("causal_conv1d")

FP32_TOLERANCE = 5e-5
BF16_RATIO = 1.5
# Additive slack for BF16 metrics whose dense error is itself near zero.
BF16_SLACK = 1e-6
CHUNK = 128  # MambaMixer's default scan chunk size.
HIDDEN = 256


def _star(prefix_len, completion_lens, extra_padding=0):
    return SharedPrefixProblem(((prefix_len, tuple(completion_lens)),), extra_padding=extra_padding)


def _forest(roots, extra_padding=0):
    return SharedPrefixProblem(
        tuple((prefix_len, tuple(lens)) for prefix_len, lens in roots), extra_padding=extra_padding
    )


def _shape_cases():
    """Port of the review's ragged_check matrix: prefix alignment x completion lengths x G."""
    completions = [1, 7, CHUNK - 1, CHUNK, CHUNK + 1]
    cases = []
    for prefix_len, prefix_tag in (
        (2 * CHUNK, "aligned"),
        (2 * CHUNK + 1, "aligned+1"),
        (50, "short"),
    ):
        for group_size in (1, 2, 16):
            lens = [
                completions[(index + group_size) % len(completions)] for index in range(group_size)
            ]
            cases.append((f"star-{prefix_tag}-G{group_size}", _star(prefix_len, lens), False))
    cases.append(("star-aligned+1-all-pad3", _star(2 * CHUNK + 1, completions, 3), False))
    cases.append(
        ("forest2", _forest([(2 * CHUNK, [7, CHUNK + 1]), (50, [1, CHUNK - 1, CHUNK])]), True)
    )
    cases.append(
        (
            "forest3-pad5",
            _forest([(2 * CHUNK + 1, [7, CHUNK]), (3 * CHUNK, [CHUNK - 1]), (5, [1, 2, 3])], 5),
            True,
        )
    )
    cases.append(("star-long-G16", _star(1000, [(37 * i) % 300 + 1 for i in range(16)]), False))
    # The review's x-grad-gap problem: fork at 256 with a 44-token tail, six completions.
    completions = (180, 64, 257, 33, 129, 200)
    cases.append(("star-P300-G6", _star(300, completions), False))
    cases.append(
        ("forest-P300-halves", _forest([(300, completions[:3]), (300, completions[3:])]), True)
    )
    return cases


def _conv_length_problem(backend, conv_len):
    """A single star whose ``causal_conv1d`` input for ``backend`` has length ``conv_len``.

    With prefix 300 and chunk 128 the state forks at 256 and replays a 44-token tail.
    Fork-based paths convolve ``(d_conv - 1) + tail + longest completion`` per branch, replay
    convolves ``prefix + longest completion``, and the packed/ragged paths convolve the whole
    channel-last sequence.
    """
    prefix_len, short = 300, 33
    if backend in ("star_cp1", "state_fork"):
        longest = conv_len - 3 - (prefix_len - 2 * CHUNK)
    elif backend == "replay_prefix":
        longest = conv_len - prefix_len
    else:
        longest = conv_len - prefix_len - short
    return _star(prefix_len, (longest, short))


# MLM-H3: causal_conv1d 1.6's channel-first backward is wrong when the sequence length is
# 1..7 mod 1024 (BF16/FP16) or 1..3 mod 512 (FP32); channel-last storage is always correct.
CONV_LENGTH_CASES = [512 + r for r in (1, 2, 3)] + [1024 + r for r in range(1, 8)]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestSharedPrefixMambaNumerics:
    """Every shared Mamba backend must reproduce dense per-row Mamba."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(123)
        self._tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def teardown_method(self, method):
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self._tf32
        Utils.destroy_model_parallel()

    def _check(self, problem, backend, forest, seed=0):
        torch.manual_seed(seed)
        layer32 = build_local_mamba_layer(torch.float32, hidden_size=HIDDEN)
        round_params_to(layer32, torch.bfloat16)
        layer16 = low_precision_copy(layer32, torch.bfloat16, hidden_size=HIDDEN)
        data = LayerProblemData(problem, HIDDEN, seed)
        layout = problem.layout(forest)

        def runs(layer, dtype):
            dense = run_layer(
                layer,
                lambda x: layer(hidden_states=x, attention_mask=None),
                data.dense_input.cuda().to(dtype),
                data.dense_cotangent.cuda(),
            )
            shared = run_layer(
                layer,
                lambda x: shared_mamba_layer_forward(layer, x, layout, backend),
                data.star_input.cuda().to(dtype),
                data.star_cotangent.cuda(),
            )
            return canonical_dense_run(data, dense), canonical_star_run(data, shared)

        reference, shared32 = runs(layer32, torch.float32)
        dense16, shared16 = runs(layer16, torch.bfloat16)
        fp32 = compare_canonical(shared32, reference)
        bf16_dense = compare_canonical(dense16, reference)
        bf16_shared = compare_canonical(shared16, reference)
        print(
            f"\n[{backend}] T={problem.physical_len} fp32={fp32}\n"
            f"  bf16 dense={bf16_dense}\n  bf16 shared={bf16_shared}"
        )
        for metric in ("output", "input_grad", "params"):
            assert fp32[metric] <= FP32_TOLERANCE, (
                f"{backend}: FP32 shared-vs-dense {metric} error {fp32[metric]:.3e} exceeds "
                f"{FP32_TOLERANCE:.0e} ({fp32})"
            )
        for metric in ("output", "input_grad", "input_grad_max_row", "params"):
            assert bf16_shared[metric] <= BF16_RATIO * bf16_dense[metric] + BF16_SLACK, (
                f"{backend}: BF16 shared {metric} error {bf16_shared[metric]:.3e} exceeds "
                f"{BF16_RATIO}x the dense BF16 error {bf16_dense[metric]:.3e}"
            )

    @pytest.mark.parametrize("backend", MAMBA_BACKENDS)
    @pytest.mark.parametrize(
        "case", _shape_cases(), ids=lambda case: case[0] if isinstance(case, tuple) else None
    )
    def test_backend_matches_dense_rows(self, case, backend):
        _, problem, forest = case
        self._check(problem, backend, forest)

    @pytest.mark.parametrize("backend", MAMBA_BACKENDS)
    @pytest.mark.parametrize("conv_len", CONV_LENGTH_CASES)
    def test_conv_length_classes_match_dense_rows(self, conv_len, backend):
        """MLM-H3 regression: branch convolution lengths in the defective length classes."""
        self._check(_conv_length_problem(backend, conv_len), backend, forest=False)

    @pytest.mark.parametrize("backend", MAMBA_BACKENDS)
    def test_eval_forward_equals_training_forward(self, backend):
        """The logprob pass (eval, no grad) must reproduce the training forward bit for bit."""
        problem, forest = _forest([(2 * CHUNK + 1, [7, CHUNK + 1]), (50, [1, CHUNK])], 3), True
        if backend == "star_cp1":
            problem, forest = _star(2 * CHUNK + 1, [7, CHUNK + 1, 1, CHUNK], 3), False
        torch.manual_seed(0)
        layer = build_local_mamba_layer(torch.bfloat16, hidden_size=HIDDEN)
        data = LayerProblemData(problem, HIDDEN, seed=0)
        hidden = data.star_input.cuda().to(torch.bfloat16)
        layout = problem.layout(forest)
        train = shared_mamba_layer_forward(
            layer, hidden.clone().requires_grad_(True), layout, backend
        )
        layer.eval()
        with torch.no_grad():
            evaluation = shared_mamba_layer_forward(layer, hidden, layout, backend)
        assert torch.equal(evaluation, train.detach())
