# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.extensions.nonuniform_tp_transformer_engine import (
    normalize_tp_domains,
    transformer_engine_userbuffer_tp_domains,
)


class FakeDistributed:
    def __init__(self, rank):
        self.rank = rank
        self.created_groups = []

    def get_rank(self):
        return self.rank

    def new_group(self, ranks, **kwargs):
        group = {"ranks": tuple(ranks), "kwargs": kwargs}
        self.created_groups.append(group)
        return group

    def new_subgroups_by_enumeration(self, ranks_per_subgroup_list, *args, **kwargs):
        return "original", ranks_per_subgroup_list, args, kwargs


def test_normalize_tp_domains_sorts_domains_for_collective_creation_order():
    domains = normalize_tp_domains([[4, 5, 6, 7], [0, 1], [2, 3]])

    assert domains == ((0, 1), (2, 3), (4, 5, 6, 7))


def test_normalize_tp_domains_rejects_overlapping_domains():
    with pytest.raises(ValueError, match="overlap"):
        normalize_tp_domains([[0, 1], [1, 2]])


def test_userbuffer_tp_domains_overrides_and_restores_subgroup_enumeration():
    fake_dist = FakeDistributed(rank=5)
    original_new_subgroups = fake_dist.new_subgroups_by_enumeration

    with transformer_engine_userbuffer_tp_domains(
        [[4, 5, 6, 7], [0, 1]], distributed=fake_dist
    ) as domains:
        current_group, groups = fake_dist.new_subgroups_by_enumeration(
            [[0, 1], [2, 3]], backend="nccl", group_desc="UB"
        )

    assert domains == ((0, 1), (4, 5, 6, 7))
    assert current_group is groups[1]
    assert [group["ranks"] for group in groups] == [(0, 1), (4, 5, 6, 7)]
    assert groups[0]["kwargs"] == {"backend": "nccl", "group_desc": "UB_ntp_0"}
    assert groups[1]["kwargs"] == {"backend": "nccl", "group_desc": "UB_ntp_1"}
    assert fake_dist.new_subgroups_by_enumeration is original_new_subgroups


def test_userbuffer_tp_domains_redirects_default_group_to_current_tp_group():
    fake_dist = FakeDistributed(rank=5)
    tp_group = {"ranks": (4, 5, 6, 7)}
    original_new_group = fake_dist.new_group

    with transformer_engine_userbuffer_tp_domains(
        [[4, 5, 6, 7], [0, 1]], distributed=fake_dist, tp_group=tp_group
    ):
        assert fake_dist.new_group(backend="nccl") is tp_group
        explicit_group = fake_dist.new_group(ranks=[0, 1], backend="nccl")

    assert explicit_group == {"ranks": (0, 1), "kwargs": {"backend": "nccl"}}
    assert fake_dist.new_group is original_new_group


def test_userbuffer_tp_domains_requires_current_rank_to_be_in_a_domain():
    fake_dist = FakeDistributed(rank=8)

    with pytest.raises(RuntimeError, match="not present"):
        with transformer_engine_userbuffer_tp_domains([[0, 1], [4, 5]], distributed=fake_dist):
            pass
