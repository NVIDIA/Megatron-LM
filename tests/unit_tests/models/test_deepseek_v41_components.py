# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from megatron.core.models.deepseek_v41.engram import EngramHasher
from megatron.core.models.deepseek_v41.moe import ModalityRouter
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
        history, blocked = ([], False)
        for shift in range(options.max_ngram_size):
            source = position - shift
            blocked = blocked or source < 0 or (not valid[0, max(source, 0)])
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
