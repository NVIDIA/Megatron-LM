# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.models.deepseek_v41.engram import EngramHasher
from tests.unit_tests.models.test_deepseek_v41 import tiny_config


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
