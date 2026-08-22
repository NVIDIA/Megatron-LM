# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Offline diagnostics for dynamic DSA inference.

The diagnostic path is intentionally detached from model outputs and optimizer state. It samples
only explicitly requested query positions, computes dense attention statistics for those rows, and
stores compact request-local records for later analysis.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Iterable, List

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from megatron.core import parallel_state


_INTEGER_RANGE_RE = re.compile(r"^(-?\d+)\.\.\.(-?\d+)(?::(\d+))?$")
_MASS_THRESHOLDS = (0.5, 0.9, 0.95, 0.99)


def expand_integer_ranges(values: Iterable[str]) -> List[int]:
    """Expand integer tokens and inclusive ranges such as ``0...32`` or ``0...32:4``."""

    expanded: List[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            match = _INTEGER_RANGE_RE.fullmatch(token)
            if match is None:
                try:
                    expanded.append(int(token))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid integer/range '{token}'; expected an integer or inclusive "
                        "range such as 0...32 or 0...32:4."
                    ) from exc
                continue

            start = int(match.group(1))
            end = int(match.group(2))
            step = int(match.group(3) or 1)
            if step <= 0:
                raise ValueError(f"Range step must be positive in '{token}'.")
            signed_step = step if end >= start else -step
            stop = end + (1 if signed_step > 0 else -1)
            expanded.extend(range(start, stop, signed_step))

    # Preserve user ordering while avoiding duplicate diagnostic work.
    return list(dict.fromkeys(expanded))


def _distributed_rank_info() -> Tuple[int, int, int, int]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0, 0, 0, 0
    if not parallel_state.model_parallel_is_initialized():
        return torch.distributed.get_rank(), 0, 0, 0
    return (
        torch.distributed.get_rank(),
        parallel_state.get_data_parallel_rank(with_context_parallel=False),
        parallel_state.get_pipeline_model_parallel_rank(),
        parallel_state.get_tensor_model_parallel_rank(),
    )


class DSADiagnosticsCollector:
    """Collect and write rank-local dynamic-inference DSA diagnostic records."""

    schema_version = 1

    def __init__(self, model_config):
        self.enabled = bool(getattr(model_config, "dsa_diagnostics", False))
        self.layers = set(getattr(model_config, "dsa_diagnostics_layers", None) or [])
        self.topk_values = tuple(
            sorted(set(getattr(model_config, "dsa_diagnostics_topk_values", None) or []))
        )
        self.prefill_tail_offsets = set(
            getattr(model_config, "dsa_diagnostics_prefill_tail_offsets", None) or []
        )
        self.decode_offsets = set(
            getattr(model_config, "dsa_diagnostics_decode_offsets", None) or []
        )
        self.output_dir = getattr(model_config, "dsa_diagnostics_output_dir", None)
        self.dump_support_indices = bool(
            getattr(model_config, "dsa_diagnostics_dump_support_indices", False)
        )
        self._prompt_lengths: Dict[int, int] = {}
        self._seen: Dict[int, set[Tuple[int, int]]] = {}
        self._pending: List[dict] = []

        self.global_rank, self.dp_rank, self.pp_rank, self.tp_rank = _distributed_rank_info()
        self._writes_records = self.enabled and self.tp_rank == 0
        self._output_path: Optional[Path] = None
        if self.enabled:
            if not self.output_dir:
                raise ValueError(
                    "dsa_diagnostics_output_dir must be set when diagnostics are enabled."
                )
            if not self.topk_values:
                raise ValueError("dsa_diagnostics_topk_values must contain at least one value.")
            if not self.prefill_tail_offsets and not self.decode_offsets:
                raise ValueError(
                    "Enable at least one prefill-tail or decode offset for DSA diagnostics."
                )
            if self._writes_records:
                output_dir = Path(self.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                self._output_path = output_dir / (
                    f"dsa_diag.dp{self.dp_rank:03d}.pp{self.pp_rank:03d}."
                    f"rank{self.global_rank:05d}.jsonl"
                )

    def register_request(self, request_id: int, prompt_length: int) -> None:
        if self.enabled:
            # Suspend/recompute can fold generated tokens into a checkpointed request's prompt.
            # Keep offsets anchored to the original prompt for the request's full lifetime.
            self._prompt_lengths.setdefault(int(request_id), int(prompt_length))

    def unregister_request(self, request_id: int) -> None:
        request_id = int(request_id)
        self._prompt_lengths.pop(request_id, None)
        self._seen.pop(request_id, None)

    def reset_runtime(self) -> None:
        self.flush()
        self._prompt_lengths.clear()
        self._seen.clear()

    def layer_enabled(self, layer_number: Optional[int]) -> bool:
        return self.enabled and layer_number is not None and (
            not self.layers or int(layer_number) in self.layers
        )

    def selected_queries(
        self,
        request_id: int,
        layer_number: int,
        query_start_position: int,
        query_length: int,
    ) -> List[dict]:
        """Return selected local query rows with prefill/decode-relative metadata."""

        if not self.layer_enabled(layer_number):
            return []
        prompt_length = self._prompt_lengths.get(int(request_id))
        if prompt_length is None:
            return []

        selected = []
        for local_idx in range(query_length):
            position = query_start_position + local_idx
            if position < prompt_length:
                offset = prompt_length - 1 - position
                if offset not in self.prefill_tail_offsets:
                    continue
                phase = "prefill_tail"
            else:
                offset = position - prompt_length
                if offset not in self.decode_offsets:
                    continue
                phase = "decode"

            request_seen = self._seen.setdefault(int(request_id), set())
            seen_key = (int(layer_number), int(position))
            if seen_key in request_seen:
                continue
            request_seen.add(seen_key)
            selected.append(
                {
                    "local_index": local_idx,
                    "position": position,
                    "phase": phase,
                    "offset": offset,
                    "prompt_length": prompt_length,
                }
            )
        return selected

    def record(self, record: dict) -> None:
        if not self._writes_records:
            return
        record = {
            "schema_version": self.schema_version,
            "global_rank": self.global_rank,
            "dp_rank": self.dp_rank,
            "pp_rank": self.pp_rank,
            **record,
        }
        self._pending.append(record)

    def flush(self) -> None:
        if not self._pending or self._output_path is None:
            return
        with self._output_path.open("a", encoding="utf-8") as output_file:
            for record in self._pending:
                output_file.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
        self._pending.clear()


def _tp_all_reduce(tensor: torch.Tensor, op: torch.distributed.ReduceOp, tp_group) -> None:
    if tp_group is None:
        return
    if torch.distributed.get_world_size(group=tp_group) > 1:
        torch.distributed.all_reduce(tensor, op=op, group=tp_group)


def _gather_head_vector(vector: torch.Tensor, tp_group) -> torch.Tensor:
    if tp_group is None or torch.distributed.get_world_size(group=tp_group) == 1:
        return vector
    gathered = [torch.empty_like(vector) for _ in range(torch.distributed.get_world_size(tp_group))]
    torch.distributed.all_gather(gathered, vector.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=0)


def _gather_head_indices(indices: torch.Tensor, tp_group) -> torch.Tensor:
    if tp_group is None or torch.distributed.get_world_size(group=tp_group) == 1:
        return indices
    gathered = [
        torch.empty_like(indices) for _ in range(torch.distributed.get_world_size(tp_group))
    ]
    torch.distributed.all_gather(gathered, indices.contiguous(), group=tp_group)
    return torch.cat(gathered, dim=0)


def assert_tp_support_consistent(indices: torch.Tensor, tp_group, label: str) -> None:
    """Fail diagnostics early when a supposedly replicated routing support differs across TP."""

    if tp_group is None or torch.distributed.get_world_size(group=tp_group) == 1:
        return
    canonical = torch.sort(indices, dim=-1).values
    minimum = canonical.clone()
    maximum = canonical.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=tp_group)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    if not torch.equal(minimum, maximum):
        mismatch = torch.nonzero(minimum != maximum, as_tuple=False)[0].cpu().tolist()
        raise RuntimeError(
            f"DSA diagnostic {label} routing support differs across TP ranks at {mismatch}."
        )


def _head_values_for_support(
    value: torch.Tensor, num_query_heads: int, support: torch.Tensor
) -> torch.Tensor:
    """Gather values as [query_head, support, value_dim] without expanding the KV sequence."""

    num_query_groups = value.size(2)
    repeat_factor = num_query_heads // num_query_groups
    gathered = []
    for group_idx in range(num_query_groups):
        selected = value[support, 0, group_idx, :].float()
        gathered.extend([selected] * repeat_factor)
    return torch.stack(gathered, dim=0)


def _shared_support_position_metrics(support: torch.Tensor, query_position: int) -> dict:
    distances = int(query_position) - support.long()
    count = max(int(support.numel()), 1)
    metrics = {
        "support_fraction_first_16": (support < 16).sum().item() / count,
    }
    for window in (128, 1024, 4096, 16384):
        metrics[f"support_fraction_last_{window}"] = (distances < window).sum().item() / count
    return metrics


def _shared_support_metrics(
    probs: torch.Tensor,
    scores: torch.Tensor,
    value: torch.Tensor,
    dense_output: torch.Tensor,
    support: torch.Tensor,
    tp_group,
    query_position: int,
) -> dict:
    support = torch.unique(support.long(), sorted=False)
    selected_probs = probs.index_select(1, support)
    captured_mass = selected_probs.sum(dim=-1)
    selected_logits = scores.index_select(1, support)
    sparse_probs = torch.softmax(selected_logits, dim=-1, dtype=torch.float32)
    selected_value = _head_values_for_support(value, probs.size(0), support)
    sparse_output = torch.einsum("hk,hkd->hd", sparse_probs, selected_value)
    dense_norm = torch.linalg.vector_norm(dense_output, dim=-1).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(sparse_output - dense_output, dim=-1) / dense_norm
    cosine = torch.nn.functional.cosine_similarity(sparse_output, dense_output, dim=-1)

    captured_mass = _gather_head_vector(captured_mass, tp_group)
    relative_l2 = _gather_head_vector(relative_l2, tp_group)
    cosine = _gather_head_vector(cosine, tp_group)
    return {
        **_shared_support_position_metrics(support, query_position),
        "captured_mass_per_head": captured_mass.cpu().tolist(),
        "captured_mass_mean": captured_mass.mean().item(),
        "captured_mass_min": captured_mass.min().item(),
        "output_relative_l2_per_head": relative_l2.cpu().tolist(),
        "output_relative_l2_mean": relative_l2.mean().item(),
        "output_relative_l2_max": relative_l2.max().item(),
        "output_cosine_per_head": cosine.cpu().tolist(),
        "output_cosine_mean": cosine.mean().item(),
    }


def _per_head_support_metrics(
    probs: torch.Tensor,
    scores: torch.Tensor,
    value: torch.Tensor,
    dense_output: torch.Tensor,
    support: torch.Tensor,
    tp_group,
    query_position: int,
) -> dict:
    selected_probs = torch.gather(probs, 1, support)
    captured_mass = selected_probs.sum(dim=-1)
    selected_logits = torch.gather(scores, 1, support)
    sparse_probs = torch.softmax(selected_logits, dim=-1, dtype=torch.float32)

    outputs = []
    num_query_groups = value.size(2)
    repeat_factor = probs.size(0) // num_query_groups
    for head_idx in range(probs.size(0)):
        group_idx = head_idx // repeat_factor
        selected_value = value[support[head_idx], 0, group_idx, :].float()
        outputs.append(sparse_probs[head_idx] @ selected_value)
    sparse_output = torch.stack(outputs, dim=0)
    dense_norm = torch.linalg.vector_norm(dense_output, dim=-1).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(sparse_output - dense_output, dim=-1) / dense_norm
    cosine = torch.nn.functional.cosine_similarity(sparse_output, dense_output, dim=-1)

    global_support = _gather_head_indices(support, tp_group)
    union_size = torch.unique(global_support).numel()
    captured_mass = _gather_head_vector(captured_mass, tp_group)
    relative_l2 = _gather_head_vector(relative_l2, tp_group)
    cosine = _gather_head_vector(cosine, tp_group)
    return {
        **_shared_support_position_metrics(global_support.reshape(-1), query_position),
        "captured_mass_per_head": captured_mass.cpu().tolist(),
        "captured_mass_mean": captured_mass.mean().item(),
        "captured_mass_min": captured_mass.min().item(),
        "output_relative_l2_per_head": relative_l2.cpu().tolist(),
        "output_relative_l2_mean": relative_l2.mean().item(),
        "output_relative_l2_max": relative_l2.max().item(),
        "output_cosine_per_head": cosine.cpu().tolist(),
        "output_cosine_mean": cosine.mean().item(),
        "head_support_union_size": int(union_size),
    }


def _distribution_width_1d(probabilities: torch.Tensor, max_topk: int) -> dict:
    probabilities = probabilities / probabilities.sum().clamp_min(1e-30)
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    top_probs = torch.topk(
        probabilities, k=min(max_topk, probabilities.numel()), sorted=True
    ).values
    cumulative_mass = top_probs.cumsum(dim=0)
    record = {
        "max_measured_topk": int(min(max_topk, probabilities.numel())),
        "entropy_effective_support": torch.exp(entropy).item(),
        "participation_support": torch.reciprocal(
            probabilities.square().sum().clamp_min(1e-30)
        ).item(),
    }
    for threshold in _MASS_THRESHOLDS:
        reached = cumulative_mass >= threshold
        record[f"k{int(threshold * 100)}"] = (
            int(reached.to(torch.int64).argmax().item()) + 1
            if bool(reached.any().item())
            else -1
        )
    return record


@torch.no_grad()
def compute_dsa_attention_diagnostics(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    indexer_support: torch.Tensor,
    model_support: torch.Tensor,
    softmax_scale: float,
    topk_values: Sequence[int],
    query_position: int,
    tp_group=None,
    model_output: Optional[torch.Tensor] = None,
    dump_support_indices: bool = False,
) -> dict:
    """Compute exact sampled-row DSA support diagnostics without changing model outputs.

    Args use the dynamic DSA per-request layouts: query ``[1,1,H,D]``, key/value
    ``[K,1,G,D]``, and supports ``[Kmax]``. Dense scores and reductions are FP32.
    """

    assert query.shape[:2] == (1, 1)
    assert key.size(1) == 1 and value.size(1) == 1
    num_query_heads = query.size(2)
    num_query_groups = key.size(2)
    assert num_query_heads % num_query_groups == 0
    valid_length = min(int(query_position) + 1, key.size(0))
    if valid_length <= 0:
        raise ValueError("DSA diagnostics require at least one causally valid key.")

    query_heads = query[0, 0].float()
    repeat_factor = num_query_heads // num_query_groups
    score_groups = []
    dense_output_groups = []
    for group_idx in range(num_query_groups):
        head_start = group_idx * repeat_factor
        head_end = head_start + repeat_factor
        group_scores = (
            query_heads[head_start:head_end]
            @ key[:valid_length, 0, group_idx, :].float().transpose(0, 1)
        ) * softmax_scale
        group_probs = torch.softmax(group_scores, dim=-1, dtype=torch.float32)
        score_groups.append(group_scores)
        dense_output_groups.append(group_probs @ value[:valid_length, 0, group_idx, :].float())
    scores = torch.cat(score_groups, dim=0)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    dense_output = torch.cat(dense_output_groups, dim=0)

    entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1)
    entropy_effective_support = torch.exp(entropy)
    participation_support = torch.reciprocal(probs.square().sum(dim=-1).clamp_min(1e-30))

    max_requested_topk = min(max(topk_values), valid_length)
    per_head_probs, per_head_indices = torch.topk(
        probs, k=max_requested_topk, dim=-1, largest=True, sorted=True
    )
    cumulative_mass = per_head_probs.cumsum(dim=-1)
    mass_widths = {}
    for threshold in _MASS_THRESHOLDS:
        reached = cumulative_mass >= threshold
        first = reached.to(torch.int64).argmax(dim=-1) + 1
        first = torch.where(reached.any(dim=-1), first, torch.full_like(first, -1))
        mass_widths[f"k{int(threshold * 100)}"] = _gather_head_vector(first, tp_group)

    entropy_effective_support = _gather_head_vector(entropy_effective_support, tp_group)
    participation_support = _gather_head_vector(participation_support, tp_group)

    teacher_sum = probs.sum(dim=0)
    teacher_max = probs.max(dim=0).values
    _tp_all_reduce(teacher_sum, torch.distributed.ReduceOp.SUM, tp_group)
    _tp_all_reduce(teacher_max, torch.distributed.ReduceOp.MAX, tp_group)
    aggregate_distribution_width = {
        "sum_head_teacher": _distribution_width_1d(teacher_sum, max_requested_topk),
        "max_head_teacher": _distribution_width_1d(teacher_max, max_requested_topk),
    }

    supports = {}
    for requested_topk in topk_values:
        support_size = min(int(requested_topk), valid_length)
        if support_size <= 0:
            continue
        indexer_indices = indexer_support[:support_size]
        indexer_indices = indexer_indices[indexer_indices < valid_length]
        sum_oracle = torch.topk(teacher_sum, k=support_size, sorted=False).indices
        max_oracle = torch.topk(teacher_max, k=support_size, sorted=False).indices
        per_head_oracle = per_head_indices[:, :support_size]

        indexer_set = torch.unique(indexer_indices)
        sum_overlap = torch.isin(indexer_set, sum_oracle).sum().item() / support_size
        max_overlap = torch.isin(indexer_set, max_oracle).sum().item() / support_size
        supports[str(requested_topk)] = {
            "indexer": {
                **_shared_support_metrics(
                    probs,
                    scores,
                    value[:valid_length],
                    dense_output,
                    indexer_indices,
                    tp_group,
                    query_position,
                ),
                "sum_oracle_recall": sum_overlap,
                "max_oracle_recall": max_overlap,
            },
            "sum_head_oracle": _shared_support_metrics(
                probs,
                scores,
                value[:valid_length],
                dense_output,
                sum_oracle,
                tp_group,
                query_position,
            ),
            "max_head_oracle": _shared_support_metrics(
                probs,
                scores,
                value[:valid_length],
                dense_output,
                max_oracle,
                tp_group,
                query_position,
            ),
            "per_head_oracle": _per_head_support_metrics(
                probs,
                scores,
                value[:valid_length],
                dense_output,
                per_head_oracle,
                tp_group,
                query_position,
            ),
        }
        if dump_support_indices:
            supports[str(requested_topk)]["support_indices"] = {
                "indexer": indexer_indices.cpu().tolist(),
                "sum_head_oracle": sum_oracle.cpu().tolist(),
                "max_head_oracle": max_oracle.cpu().tolist(),
                "per_head_oracle": _gather_head_indices(per_head_oracle, tp_group).cpu().tolist(),
            }

    model_support = model_support[model_support < valid_length]
    model_metrics = _shared_support_metrics(
        probs,
        scores,
        value[:valid_length],
        dense_output,
        model_support,
        tp_group,
        query_position,
    )
    if model_output is not None:
        actual_output = model_output.reshape(num_query_heads, value.size(-1)).float()
        dense_norm = torch.linalg.vector_norm(dense_output, dim=-1).clamp_min(1e-12)
        actual_relative_l2 = (
            torch.linalg.vector_norm(actual_output - dense_output, dim=-1) / dense_norm
        )
        actual_cosine = torch.nn.functional.cosine_similarity(
            actual_output, dense_output, dim=-1
        )
        actual_relative_l2 = _gather_head_vector(actual_relative_l2, tp_group)
        actual_cosine = _gather_head_vector(actual_cosine, tp_group)
        model_metrics.update(
            {
                "actual_output_relative_l2_per_head": actual_relative_l2.cpu().tolist(),
                "actual_output_relative_l2_mean": actual_relative_l2.mean().item(),
                "actual_output_relative_l2_max": actual_relative_l2.max().item(),
                "actual_output_cosine_per_head": actual_cosine.cpu().tolist(),
                "actual_output_cosine_mean": actual_cosine.mean().item(),
            }
        )
    width_record = {
        "max_measured_topk": max_requested_topk,
        "entropy_effective_support_per_head": entropy_effective_support.cpu().tolist(),
        "participation_support_per_head": participation_support.cpu().tolist(),
    }
    for name, widths in mass_widths.items():
        width_record[f"{name}_per_head"] = widths.cpu().tolist()

    return {
        "valid_key_count": valid_length,
        "distribution_width": width_record,
        "aggregate_distribution_width": aggregate_distribution_width,
        "model_support": model_metrics,
        "supports": supports,
    }



def _summarize_distribution_width(rows: Iterable[dict]) -> List[dict]:
    grouped = defaultdict(lambda: {"values": [], "unresolved_caps": [], "total_count": 0})
    metadata = {
        "dp_rank",
        "request_id",
        "layer",
        "phase",
        "offset",
        "query_position",
        "context_length",
        "max_measured_topk",
        "head",
    }
    for row in rows:
        for metric, value in row.items():
            if metric in metadata or value is None:
                continue
            group = grouped[(row["layer"], row["phase"], metric)]
            group["total_count"] += 1
            if metric.startswith("k") and value < 0:
                measurement_cap = row.get("max_measured_topk")
                if measurement_cap is not None:
                    group["unresolved_caps"].append(float(measurement_cap))
                continue
            group["values"].append(float(value))

    summary = []
    for (layer, phase, metric), group in sorted(grouped.items()):
        values = group["values"]
        unresolved_caps = group["unresolved_caps"]
        total_count = group["total_count"]
        resolved_count = len(values)
        unresolved_count = total_count - resolved_count
        summary.append(
            {
                "layer": layer,
                "phase": phase,
                "metric": metric,
                "count": total_count,
                "resolved_count": resolved_count,
                "unresolved_count": unresolved_count,
                "unresolved_fraction": unresolved_count / total_count,
                "mean": mean(values) if values else None,
                "p50": _percentile(values, 0.50) if values else None,
                "p90": _percentile(values, 0.90) if values else None,
                "p99": _percentile(values, 0.99) if values else None,
                "max": max(values) if values else None,
                "unresolved_measurement_cap_min": (
                    min(unresolved_caps) if unresolved_caps else None
                ),
                "unresolved_measurement_cap_mean": (
                    mean(unresolved_caps) if unresolved_caps else None
                ),
                "unresolved_measurement_cap_max": (
                    max(unresolved_caps) if unresolved_caps else None
                ),
            }
        )
    return summary
