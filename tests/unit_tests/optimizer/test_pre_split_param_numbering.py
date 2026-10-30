# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The numbering a pre-split checkpoint was written with.

Splitting a parameter group renumbers parameters, and torch matches saved state to live
parameters by flattened position. Reconstructing the old numbering wrongly is therefore
silent: the count still checks out and training continues on some other parameter's
momentum, with nothing in the logs to say so.
"""

import torch


# The ordering the groups were laid out in before the split, copied from _get_param_groups so
# the assertions below describe the contract rather than re-using the implementation.
def _key_order(key):
    return (key[0] is not None, key[0], key[1], key[2] if len(key) > 2 else False)


def test_tagged_indices_match_the_numbering_a_pre_split_run_wrote():
    """REGRESSION: the tags must reproduce the old numbering when there are several groups.

    With one pre-split group the ordering is trivial and any implementation looks right. With
    two, the numbering depends on which group came first and on indexer and backbone
    parameters being interleaved within each -- so this is the case that can actually be
    wrong, and silently: a bad numbering still satisfies torch's count check and hands
    parameters each other's momentum.
    """
    from megatron.core.optimizer import _tag_pre_split_param_index

    override_a, override_b = None, (("wd_mult", 0.0),)
    # Model order: the two overrides interleave, and within each the indexer sits between
    # backbone parameters, as it does inside a real attention block.
    spec = [
        (override_a, False),
        (override_a, True),
        (override_b, False),
        (override_a, False),
        (override_b, True),
        (override_b, False),
    ]
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in spec]
    model_order = {id(p): i for i, p in enumerate(params)}

    params_map = {}
    for param, (override, is_indexer) in zip(params, spec):
        params_map.setdefault((override, False, is_indexer), []).append(param)

    _tag_pre_split_param_index(params_map, list(params_map), model_order, _key_order)

    # What a pre-split run would have produced: the same parameters bucketed without the
    # indexer flag, those buckets in _key_order, each filled in model order.
    pre_split = {}
    for param, (override, _) in zip(params, spec):
        pre_split.setdefault((override, False), []).append(param)
    expected, nxt = {}, 0
    for key in sorted(pre_split, key=_key_order):
        for param in sorted(pre_split[key], key=lambda p: model_order[id(p)]):
            expected[id(param)] = nxt
            nxt += 1

    actual = {id(p): p.pre_split_param_index for p in params}
    assert actual == expected, (
        "tagged indices do not match the numbering a pre-split run wrote:\n"
        f"  got      {[actual[id(p)] for p in params]}\n"
        f"  expected {[expected[id(p)] for p in params]}"
    )
    # And it must be a contiguous numbering, or the load-side check rejects it.
    assert sorted(actual.values()) == list(range(len(params)))
