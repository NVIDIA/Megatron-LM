# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Frozen benchmark snapshot: preserve the original prediction code body.
# isort: skip_file
# fmt: off
# ruff: noqa
# pylint: skip-file

"""Physical NVLink domains and provisional transfer of measured P2P exchanges."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import statistics

from ..validation import positive_int

CALIBRATION_DOMAIN_SIZES = {"h100": 8, "gb200": 72}


def _partition(domain_ids):
    names = {}
    return tuple(names.setdefault(value, len(names)) for value in domain_ids)


@dataclass(frozen=True)
class RingPlacement:
    ranks: tuple[int, ...]
    domain_size: int
    calibration_domain_size: int
    assumed: bool = False

    @classmethod
    def resolve(cls, ranks, cp_size, hardware, domain_size=None):
        positive_int(cp_size, "cp_size")
        reference = CALIBRATION_DOMAIN_SIZES[hardware]
        size = reference if domain_size is None else positive_int(domain_size, "domain_size")
        assumed = ranks is None
        ranks = tuple(range(cp_size)) if assumed else tuple(ranks)
        if len(ranks) != cp_size or any(type(r) is not int or r < 0 for r in ranks):
            raise ValueError("ranks must contain CP-size nonnegative integer global ranks in ring order")
        if len(set(ranks)) != cp_size:
            raise ValueError("ranks must not contain duplicates")
        return cls(ranks, size, reference, assumed)

    @property
    def domains(self):
        return tuple(r // self.domain_size for r in self.ranks)

    @property
    def partition(self):
        return _partition(self.domains)

    @property
    def reference_partition(self):
        return _partition(r // self.calibration_domain_size for r in range(len(self.ranks)))

    @property
    def changed(self):
        return self.partition != self.reference_partition

    @property
    def link_classes(self):
        domains, cp = self.domains, len(self.ranks)
        # The benchmark exchanges to i+1 and receives from i-1, including wraparound.
        return tuple((domains[i] != domains[(i+1) % cp], domains[i] != domains[(i-1) % cp])
                     for i in range(cp))

    def to_dict(self):
        cp = len(self.ranks)
        return dict(ranks=list(self.ranks), domain_size=self.domain_size,
            domain_ids=list(self.domains), domain_count=len(set(self.domains)),
            cross_domain=len(set(self.domains)) > 1, placement_assumed=self.assumed,
            calibration_domain_size=self.calibration_domain_size,
            calibration_partition=list(self.reference_partition),
            matches_calibration_partition=not self.changed,
            ring_edges=[dict(send_rank=self.ranks[i], recv_rank=self.ranks[(i+1) % cp],
                             cross_domain=self.link_classes[i][0]) for i in range(cp)])


def _map_rank_measurements(values, source_classes, target_classes):
    mapped, provenance = [], []
    for target in target_classes:
        indices = [i for i, value in enumerate(source_classes) if value == target]
        fallback = not indices
        if fallback:
            # A singleton domain has both neighbors outside; the old contiguous
            # anchors lack that class. Use the largest recorded boundary endpoint,
            # explicitly as a proxy, never as a measured upper bound.
            indices = [i for i, value in enumerate(source_classes) if any(value) == any(target)]
        if not indices:
            raise ValueError("no measured communication endpoint for this domain placement")
        samples = [float(values[i]) for i in indices]
        value = max(samples) if fallback else statistics.median(samples)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("invalid measured exchange latency")
        mapped.append(value)
        provenance.append(dict(source_local_ranks=indices,
            aggregation="max_boundary_proxy" if fallback else "median_matching_endpoints"))
    return tuple(mapped), provenance


def predict_forward_at_placement(model, lengths, cp_size, *, hardware, placement,
                                 sequence_alignment):
    """Keep CP compute geometry and transfer communication from measured anchors.

    This is a topology extrapolation, not a newly calibrated attention fit.
    Whole-call residuals stay attached to the original CP/pack, never to the
    donor CP. Shared frozen model instances and coefficients are not mutated.
    """
    if not placement.changed or cp_size <= 1:
        raise ValueError("placement transfer requires a changed multi-CP domain partition")
    domain_count = len(set(placement.domains))
    candidates = []
    for cp in model.communication_rank_ms:
        if cp & (cp-1):
            continue
        reference = RingPlacement.resolve(None, cp, hardware)
        count = len(set(reference.domains))
        if (count > 1) == (domain_count > 1):
            candidates.append((abs(count-domain_count), abs(cp-cp_size), cp, reference))
    if not candidates:
        raise ValueError(f"no measured {'cross' if domain_count > 1 else 'intra'}-domain "
                         f"communication anchor for {hardware}; cannot substitute another GPU")
    _, _, anchor_cp, reference = min(candidates, key=lambda value: value[:3])
    isolated, endpoints = _map_rank_measurements(model.communication_rank_ms[anchor_cp],
                                                reference.link_classes, placement.link_classes)
    steady_values = model.sustained_ring_communication_rank_ms.get(anchor_cp)
    steady, _ = _map_rank_measurements(steady_values or model.communication_rank_ms[anchor_cp],
                                      reference.link_classes, placement.link_classes)
    moved = copy.copy(model)
    for name, value in (("communication_rank_ms", isolated),
                        ("communication_ms", max(isolated)),
                        ("sustained_ring_communication_rank_ms", steady),
                        ("sustained_ring_communication_ms", max(steady)),
                        ("communication_host_ms", model.communication_host_ms[anchor_cp])):
        setattr(moved, name, {**getattr(model, name), cp_size: value})
    if model.host_cp_scaling is not None:
        moved.host_cp_scaling = copy.deepcopy(model.host_cp_scaling)
        moved.host_cp_scaling.setdefault("outside_single_nvlink_domain_by_cp", {})[str(cp_size)] = domain_count > 1

    kwargs = dict(max_cp_size=cp_size, sequence_alignment=sequence_alignment)
    baseline = model.predict_with_details(lengths, cp_size, **kwargs)
    # Whole-call floors/scales were measured at the original placement. Transfer
    # only the component-timeline delta, preserving the original residual once.
    moved.single_sequence_wall_floor_by_cp = {}
    moved.padded_path_wall_floor_by_cp = {}
    moved.wall_time_calibration = None
    changed = moved.predict_with_details(lengths, cp_size, **kwargs)
    host_exchange_delta = 0.0
    if model.host_cp_scaling is not None and model.host_communication_slowdown is None:
        # The CP-scaled host fit embeds the old enqueue cost. Replace that cost
        # for CP-1 exchanges, rather than charging another full CP16 host call.
        host_exchange_delta = (cp_size-1) * (model.communication_host_ms[anchor_cp]
                                            - model.communication_host_ms[cp_size])
        changed["host_time_ms"] = max(0.0, changed["host_time_ms"] + host_exchange_delta)
        if changed["host"] is not None:
            changed["host"]["placement_exchange_delta_ms"] = host_exchange_delta
            changed["host"]["time_ms"] = changed["host_time_ms"]
    original_components = max(baseline["gpu_time_ms"], baseline["host_time_ms"])
    new_components = max(changed["gpu_time_ms"], changed["host_time_ms"])
    residual = baseline["time_ms"] - original_components
    changed["time_ms"] = new_components + residual
    changed["limiting_timeline"] = "host" if changed["host_time_ms"] > changed["gpu_time_ms"] else "gpu"
    if not math.isfinite(changed["time_ms"]) or changed["time_ms"] <= 0:
        raise ValueError("placement transfer produced an invalid attention time")
    changed["provisional"] = True
    changed["placement_transfer"] = dict(method="measured_rank_exchange_component_delta",
        hardware=hardware, calibration_transport="te_nccl_p2p", anchor_cp_size=anchor_cp,
        anchor_ranks=list(reference.ranks), anchor_domain_ids=list(reference.domains),
        actual_cp_size=cp_size, logical_lengths=list(lengths),
        original_time_ms=baseline["time_ms"], original_components_ms=original_components,
        moved_components_ms=new_components, component_delta_ms=new_components-original_components,
        host_exchange_delta_ms=host_exchange_delta,
        retained_whole_call_residual_ms=residual,
        isolated_exchange_ms_per_rank=list(isolated),
        pipeline_exchange_ms_per_rank=list(steady), endpoint_sources=endpoints,
        pipeline_source="sustained_ring" if steady_values else "isolated_exchange_proxy",
        limitation="unvalidated placement transfer; not a native-transport calibration")
    return changed
