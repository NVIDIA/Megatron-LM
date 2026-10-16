# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in mutation controls for the composite mHC comparison criteria.

Each case mutates one side of a composite test and must be detected by a
named criterion check: the comparison raises AssertionError whose message
carries one of the case's accepted fragments. An unmutated control must pass.
Run with GLM53_COMPOSITE_DIAG=1; this is a diagnostic of the criteria, not a
CI test.
"""

import os

import pytest
import torch

from tests.unit_tests.models.test_glm53_hybrid_native_parity import (  # noqa: F401
    _model_parallel,
    _run_dsa_moe,
    _run_kda_dense,
)

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        os.environ.get('GLM53_COMPOSITE_DIAG') != '1', reason='set GLM53_COMPOSITE_DIAG=1'
    ),
]


@torch.no_grad()
def zero_ffn_pre_bias(native):
    native.ffn_hc.base[0:4].zero_()


def swap_attention_post_streams(native):
    mappings = native.attn_hc.mappings

    def swapped(hidden):
        aggregated, post, comb = mappings(hidden)
        order = torch.tensor([1, 0, 2, 3], device=post.device)
        return aggregated, post.index_select(-1, order), comb

    native.attn_hc.mappings = swapped


@torch.no_grad()
def scale_target_ffn_pre_gradient(real):
    real.layers[1].hyper_connection.alpha_pre.grad.mul_(1.5)


@torch.no_grad()
def double_native_ffn_pre_addends(native):
    # Consistent on the reference side: its scale gradient stays the sum of its
    # contributions, so only the contribution tensor comparison can see it.
    native.ffn_hc.last_mix.grad[..., 0:4].mul_(2)
    native.ffn_hc.scale.grad[0:1].mul_(2)


def lower_target_clamp(config):
    config.activation_func_clamp_value = 9.5


def kda(monkeypatch, mutate):
    _run_kda_dense(False, False, monkeypatch, compare_all_parameters=True, mutate=mutate)


def dsa(monkeypatch, mutate):
    _run_dsa_moe('none', monkeypatch, mutate=mutate)


PARITY = ('cosine=', 'tensor_similarity=')
RECONSTRUCTION = ('is not the sum of its contributions',)
NEAR_TIE = ('decision is not a near tie',)

# case: (runner, hook point -> mutation, accepted message fragments)
CASES = {
    'M1_ffn_pre_bias_zeroed': (kda, {'native': zero_ffn_pre_bias}, PARITY + NEAR_TIE),
    'M2_attention_post_streams_swapped': (
        kda,
        {'native': swap_attention_post_streams},
        PARITY + NEAR_TIE,
    ),
    'Ms_target_ffn_pre_scale_gradient_x1p5': (
        kda,
        {'actual_gradients': scale_target_ffn_pre_gradient},
        RECONSTRUCTION,
    ),
    'Mg_reference_ffn_pre_addends_x2': (
        kda,
        {'native_gradients': double_native_ffn_pre_addends},
        ('layers.1.hyper_connection.alpha_pre contributions',),
    ),
    # The target clamps (9.5, 10] while decisions are replayed at 10, so only
    # the parity bars can see it; a near-tie trip would not count.
    'Mc_target_clamp_threshold_9p5': (dsa, {'config': lower_target_clamp}, PARITY),
}


def test_glm53_composite_criteria_unmutated_control(monkeypatch):
    """The KDA runner passes without a mutation (the DSA control is the parity test itself)."""
    kda(monkeypatch, None)


@pytest.mark.parametrize('case', sorted(CASES))
def test_glm53_composite_criteria_detect_mutation(case, monkeypatch):
    runner, mutate, fragments = CASES[case]
    with pytest.raises(AssertionError) as error:
        runner(monkeypatch, mutate)
    message = str(error.value)
    print(f'GLM53_COMPOSITE_DIAG {case} detected: {message.splitlines()[0][:300]}')
    assert any(fragment in message for fragment in fragments), message[:300]
