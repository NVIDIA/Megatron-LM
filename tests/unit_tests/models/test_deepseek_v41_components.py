# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical and boundary checks for released V4.1 conditional components."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.models.deepseek_v41.dspark import DSparkOutput, select_verification_length
from megatron.core.models.deepseek_v41.engram import EngramHasher
from megatron.core.models.deepseek_v41.moe import ModalityBalance, ModalityRouter
from megatron.core.models.deepseek_v41.vision import Aligner, get_vision_cos_sin
from megatron.core.transformer.module import convert_module_to_dtype_except_fp32_marked
from tests.unit_tests.models.test_deepseek_v41 import groups, tiny_config


def test_hashes_match_integer_oracle_and_reset_at_image_boundary():
    options = tiny_config(all_components=True).engram_config
    hasher = EngramHasher(options, torch.arange(128))
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10]])
    valid = torch.tensor([[True, True, False, True, True, True]])
    hashes = hasher(tokens, valid)
    expected = torch.empty_like(hashes)
    for position in range(tokens.shape[1]):
        history, blocked = [], False
        for shift in range(options.max_ngram_size):
            source = position - shift
            blocked = blocked or source < 0 or not valid[0, max(source, 0)]
            history.append(hasher.pad_id if blocked else int(tokens[0, source]))
        for layer in range(len(options.layer_ids)):
            rolling = history[0] * int(hasher.multipliers[layer, 0])
            head = 0
            for order in range(1, options.max_ngram_size):
                rolling ^= history[order] * int(hasher.multipliers[layer, order])
                for prime in hasher.primes[layer, order - 1]:
                    expected[0, position, layer, head] = (
                        rolling % int(prime) + hasher.offsets[layer, head]
                    )
                    head += 1
    torch.testing.assert_close(hashes, expected)
    changed = tokens.clone()
    changed[:, :2] += 30
    torch.testing.assert_close(hasher(changed, valid)[:, 3:], hashes[:, 3:])


def test_aligner_padding_and_channel_order_match_explicit_pixel_unshuffle():
    args = SimpleNamespace(vision_dim=4, dim=8, vision_downsample_ratio=3)
    aligner = Aligner(args)
    x = torch.randn(4, 5, 4, requires_grad=True)
    actual = aligner(x.flatten(0, 1), 4, 5)
    padded = F.pad(x.permute(2, 0, 1), (0, 1, 0, 2))
    rows = []
    for h in (0, 3):
        for w in (0, 3):
            rows.append(
                torch.stack(
                    [
                        padded[c, h + dh, w + dw]
                        for c in range(4)
                        for dh in range(3)
                        for dw in range(3)
                    ]
                )
            )
    oracle = aligner.w2(F.gelu(aligner.w1(torch.stack(rows))))
    torch.testing.assert_close(actual, oracle)
    actual.sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    cos, sin = get_vision_cos_sin(4, 5, 4, 10000, x.device)
    assert cos.shape == sin.shape == (20, 1, 4)
    torch.testing.assert_close(cos.square() + sin.square(), torch.ones_like(cos))


def test_modality_biases_select_experts_without_changing_weights(groups):
    config = tiny_config()
    router = ModalityRouter(config, groups).cuda()
    convert_module_to_dtype_except_fp32_marked(router, torch.bfloat16)
    assert router.text_balance.expert_bias.dtype == torch.float32
    with torch.no_grad():
        router.weight.zero_()
        router.text_balance.expert_bias.copy_(torch.tensor([4, 3, 2, 1], device="cuda"))
        router.image_balance.expert_bias.copy_(torch.tensor([1, 2, 3, 4], device="cuda"))
    probs, route = router(
        torch.ones(2, 1, 32, device="cuda", dtype=torch.bfloat16),
        image_mask=torch.tensor([[False], [True]], device="cuda"),
    )
    assert route[0].tolist() == [True, True, False, False]
    assert route[1].tolist() == [False, False, True, True]
    torch.testing.assert_close(probs.sum(-1).float(), torch.full((2,), 1.5, device="cuda"))
    assert router.text_balance.local_tokens_per_expert.tolist() == [1, 1, 0, 0]
    assert router.image_balance.local_tokens_per_expert.tolist() == [0, 0, 1, 1]


def test_mixed_backbone_and_draft_expert_counts_update_independently(groups):
    modules = torch.nn.ModuleList([ModalityBalance(4), ModalityBalance(2)]).cuda()
    modules[0].local_tokens_per_expert.copy_(torch.tensor([4, 0, 0, 0], device="cuda"))
    modules[1].local_tokens_per_expert.copy_(torch.tensor([0, 3], device="cuda"))
    config = tiny_config()
    _update_router_expert_bias([modules], config, groups.tp_dp_cp)
    rate = config.moe_router_bias_update_rate
    torch.testing.assert_close(
        modules[0].expert_bias, torch.tensor([-rate, rate, rate, rate], device="cuda")
    )
    torch.testing.assert_close(modules[1].expert_bias, torch.tensor([rate, -rate], device="cuda"))


def test_dspark_confidence_targets_use_distribution_overlap():
    logits = torch.tensor([[[[0.4, -0.3], [0.2, 0.8]]]], requires_grad=True)
    confidence = torch.zeros(1, 1, 2, requires_grad=True)
    teacher = torch.tensor([[[[0.3, -0.2], [0.1, 1.0]]]], requires_grad=True)
    result = DSparkOutput(logits, confidence, torch.tensor([[[0, 1]]]))
    loss = result.loss(teacher, ce_weight=0, l1_weight=0)
    overlap = torch.minimum(logits.softmax(-1), teacher.softmax(-1)).sum(-1).detach()
    expected = F.binary_cross_entropy_with_logits(confidence, overlap)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert teacher.grad is None
    torch.testing.assert_close(confidence.grad, (0.5 - overlap) / 2)
    torch.testing.assert_close(
        select_verification_length(torch.tensor([[0.9, 0.1]]), torch.tensor([1.0, 2.0])),
        torch.tensor([1]),
    )
