# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""AST guards for async decode transaction ownership.

These checks are intentionally import-free.  They enforce the transactional
ownership contract in production inference code without importing torch or
initializing distributed state.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOT = REPO_ROOT / "megatron" / "core" / "inference"


ASYNC_TRANSACTION = "megatron/core/inference/async_transaction.py"
ASYNC_COORDINATOR = (
    "megatron/core/inference/text_generation_controllers/async_decode_coordinator.py"
)
TEXT_CONTROLLER = (
    "megatron/core/inference/text_generation_controllers/text_generation_controller.py"
)
DYNAMIC_CONTEXT = "megatron/core/inference/contexts/dynamic_context.py"


EXPECTED_VIOLATIONS = {
    ("async-dataclass-fields", ASYNC_TRANSACTION, "AsyncDecodePlan"),
    ("async-dataclass-fields", ASYNC_TRANSACTION, "AsyncDecodeTransaction"),
    ("async-dataclass-fields", ASYNC_TRANSACTION, "AsyncEligibilityDecision"),
    ("async-dataclass-fields", ASYNC_TRANSACTION, "AsyncResourceLedger"),
    ("async-dataclass-missing", ASYNC_TRANSACTION, "AsyncCoordinatorStepState"),
    ("async-dataclass-missing", ASYNC_TRANSACTION, "AsyncPreSamplingContextState"),
    ("async-dataclass-missing", ASYNC_TRANSACTION, "AsyncPreparedDecodeState"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncDecodeTransaction"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncEligibilityDecision"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncGraphShape"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncKVReservation"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncResourceLedger"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncSampleReadback"),
    ("async-dataclass-slots", ASYNC_TRANSACTION, "AsyncSampleTicket"),
    ("async-dataclass-unexpected", ASYNC_TRANSACTION, "AsyncEPParticipant"),
    ("async-dataclass-unexpected", ASYNC_TRANSACTION, "AsyncLogprobMTPParticipant"),
    ("async-dataclass-unexpected", ASYNC_TRANSACTION, "AsyncMambaLease"),
    ("async-dataclass-unexpected", ASYNC_TRANSACTION, "AsyncMambaStateParticipant"),
    ("async-dataclass-unexpected", ASYNC_TRANSACTION, "AsyncResourceParticipant"),
    ("async-dataclass-unexpected", ASYNC_TRANSACTION, "AsyncSampleReadbackParticipant"),
    ("async-removed-symbol", ASYNC_TRANSACTION, "AsyncMambaLease"),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_pre_sampling_prepared_state"),
    (
        "context-forbidden-lifecycle-field",
        DYNAMIC_CONTEXT,
        "_async_prepared_decode_input_is_identity",
    ),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_decode_plan"),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_paused_dest_rows"),
    (
        "context-forbidden-lifecycle-field",
        DYNAMIC_CONTEXT,
        "_async_prepared_paused_dest_rows_cuda",
    ),
    (
        "context-forbidden-lifecycle-field",
        DYNAMIC_CONTEXT,
        "_async_prepared_paused_source_count",
    ),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_paused_source_rows"),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_request_count"),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_request_ids"),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_sample_dest_rows"),
    (
        "context-forbidden-lifecycle-field",
        DYNAMIC_CONTEXT,
        "_async_prepared_sample_dest_rows_cuda",
    ),
    (
        "context-forbidden-lifecycle-field",
        DYNAMIC_CONTEXT,
        "_async_prepared_sample_source_count",
    ),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_prepared_sample_source_rows"),
    (
        "context-forbidden-lifecycle-field",
        DYNAMIC_CONTEXT,
        "_async_prepared_sample_source_rows_cuda",
    ),
    ("context-forbidden-lifecycle-field", DYNAMIC_CONTEXT, "_async_resource_ledger"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncDecodeTransaction"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncEPParticipant"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncLogprobMTPParticipant"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncMambaStateParticipant"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncResourceLedger"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncResourceParticipant"),
    ("controller-forbidden-lifecycle-import", TEXT_CONTROLLER, "AsyncTxnState"),
    ("participant-lifecycle-fields", ASYNC_TRANSACTION, "AsyncEPParticipant"),
    ("participant-lifecycle-fields", ASYNC_TRANSACTION, "AsyncLogprobMTPParticipant"),
    ("participant-lifecycle-fields", ASYNC_TRANSACTION, "AsyncMambaStateParticipant"),
    ("participant-lifecycle-fields", ASYNC_TRANSACTION, "AsyncResourceParticipant"),
    ("participant-lifecycle-fields", ASYNC_TRANSACTION, "AsyncSampleReadbackParticipant"),
}


EXACT_DATACLASS_FIELDS = {
    "AsyncGraphShape": (
        "active_request_count",
        "active_token_count",
        "padded_active_request_count",
        "tokens_per_request",
    ),
    "AsyncDecodeLayout": (
        "request_ids",
        "source_request_idxs",
        "graph_shape",
        "request_query_lengths",
        "request_kv_length_offsets",
        "request_to_kv_block_ids",
        "token_to_pos_ids",
        "token_to_request_idx",
        "token_to_position_in_request",
        "token_to_local_position_within_kv_block",
        "token_to_block_idx",
        "mamba_read_indices",
        "mamba_write_indices",
    ),
    "AsyncDecodePlan": (
        "layout",
        "finished_request_ids",
        "requires_mamba_state",
        "requires_mtp",
        "requires_logprobs",
    ),
    "AsyncPendingForwardDecision": (
        "reusable",
        "row_map",
        "row_mapped",
        "row_map_policy",
        "graph_compatible",
        "layout_compatible",
        "reason",
    ),
    "AsyncPendingForwardUse": (
        "reused",
        "row_indices",
        "row_mapped",
        "graph_request_count",
    ),
    "AsyncKVReservation": ("request_id", "block_column", "block_id"),
    "AsyncResourceLedger": (
        "kv_reservations",
        "deferred_kv_blocks",
        "deferred_mamba_slots",
        "in_flight",
        "consumed_reservation_count",
    ),
    "AsyncPreSamplingContextState": (
        "active_attn_metadata",
        "active_logit_idxs",
        "active_request_metadata",
        "active_token_count",
        "batch_dimensions",
        "cpu_bookkeeping_buf",
        "mha_metadata",
        "mha_state_data",
        "mha_max_seqlen_q",
        "mha_max_seqlen_k",
        "padded_active_request_count",
        "padded_active_token_count",
        "padded_batch_dimensions",
        "padding_slice",
        "pending_mamba_transfer",
        "using_cuda_graph_this_step",
    ),
    "AsyncPreparedDecodeState": (
        "plan",
        "resource_ledger",
        "sample_source_rows",
        "sample_dest_rows",
        "paused_source_rows",
        "paused_dest_rows",
        "sample_source_rows_cuda",
        "sample_dest_rows_cuda",
        "paused_dest_rows_cuda",
        "decode_input_is_identity",
        "pre_sampling_state",
    ),
    "AsyncDecodeTransaction": (
        "step_id",
        "state",
        "plan",
        "resolution",
        "resource_ledger",
        "participants",
        "sample_ticket",
        "h2d_done_event",
        "forward_done_event",
        "discard_reason",
    ),
    "AsyncCoordinatorStepState": (
        "ep_step_begin_decision",
        "ep_handoff_decision",
        "handoff_decided",
    ),
    "AsyncParticipantLifecycle": ("prepared", "committed", "rolled_back", "retired"),
    "AsyncSampleTicket": (
        "slot",
        "active_request_count",
        "sampled_tokens_cuda",
        "sample_values_cuda",
        "sampled_tokens_cpu",
        "sampled_mtp_tokens_cuda",
        "sampled_mtp_tokens_cpu",
        "source_ready_event",
        "copy_done_event",
        "copy_stream",
    ),
    "AsyncSampleReadback": (
        "sample_slot_count",
        "current_sample_slot",
        "sampled_tokens_cuda_slots",
        "sample_values_cuda_slots",
        "sampled_tokens_cpu_slots",
        "source_ready_events",
        "copy_done_events",
        "copy_stream",
        "sampled_mtp_tokens_cuda_slots",
        "sampled_mtp_tokens_cpu_slots",
    ),
    "AsyncEligibilityDecision": ("can_prepare", "reason"),
}

FROZEN_DATACLASSES = {
    "AsyncGraphShape",
    "AsyncDecodeLayout",
    "AsyncDecodePlan",
    "AsyncPendingForwardDecision",
    "AsyncPendingForwardUse",
    "AsyncKVReservation",
    "AsyncSampleTicket",
    "AsyncEligibilityDecision",
}

REQUIRED_DATACLASSES = {
    "AsyncDecodeLayout",
    "AsyncDecodePlan",
    "AsyncPendingForwardDecision",
    "AsyncPendingForwardUse",
    "AsyncKVReservation",
    "AsyncResourceLedger",
    "AsyncPreSamplingContextState",
    "AsyncPreparedDecodeState",
    "AsyncDecodeTransaction",
    "AsyncCoordinatorStepState",
}

REMOVED_ASYNC_SYMBOLS = {
    "AsyncLayoutSnapshot",
    "AsyncMambaLease",
}

PROTOCOL_METHODS = {
    "AsyncDecodeContextOps": (
        "build_prepared_state",
        "publish_prepared_state",
        "copy_prepared_input_ids",
        "current_layout",
        "queue_h2d_transfer",
        "current_input_and_position_ids",
        "row_mapped_reuse_allowed",
        "register_active_ledger",
        "clear_active_ledger",
    ),
    "AsyncDecodeAllocatorOps": (
        "release_kv_blocks",
        "release_mamba_slots",
        "record_deferred_kv_release",
        "record_deferred_mamba_release",
    ),
    "AsyncDecodeEPOps": (
        "begin_step",
        "async_handoff",
        "ensure_handoff_decided",
        "diagnostics",
    ),
    "AsyncDecodeDiagnosticsOps": (
        "increment_counter",
        "record_disable_reason",
        "record_eligibility",
    ),
    "AsyncDecodeModelCallbacks": (
        "launch_prepared_forward",
        "speculative_token_count",
        "row_map_policy",
        "requires_logprobs",
        "requires_mtp",
        "classify_eligibility",
    ),
}

COORDINATOR_OWNED_FIELDS = {
    "_prepared_state",
    "_pending_transaction",
    "_next_step_id",
    "_step_state",
}

FORBIDDEN_CONTROLLER_FIELDS = {
    "_async_step_transaction",
    "_async_transaction_next_step_id",
    "_async_ep_participant_this_step",
    "_ep_async_protocol",
    "_ep_async_handoff_decided_this_step",
    "_ep_async_handoff_decision_this_step",
}

FORBIDDEN_CONTROLLER_IMPORTS = {
    "AsyncDecodeTransaction",
    "AsyncEPParticipant",
    "AsyncLayoutSnapshot",
    "AsyncLogprobMTPParticipant",
    "AsyncMambaStateParticipant",
    "AsyncPendingForwardDecision",
    "AsyncResourceLedger",
    "AsyncResourceParticipant",
    "AsyncTxnState",
}

FORBIDDEN_CONTEXT_FIELDS = {
    "_async_resource_ledger",
    "_async_prepared_request_count",
    "_async_prepared_decode_plan",
    "_async_prepared_request_ids",
    "_async_prepared_sample_source_count",
    "_async_prepared_paused_source_count",
    "_async_prepared_decode_input_is_identity",
    "_async_prepared_sample_source_rows",
    "_async_prepared_sample_dest_rows",
    "_async_prepared_paused_source_rows",
    "_async_prepared_paused_dest_rows",
    "_async_prepared_sample_source_rows_cuda",
    "_async_prepared_sample_dest_rows_cuda",
    "_async_prepared_paused_dest_rows_cuda",
    "_async_pre_sampling_prepared_state",
}

PARTICIPANT_LIFECYCLE_FIELDS = {
    "prepared",
    "committed",
    "rolled_back",
    "retired",
}


@dataclass(frozen=True, order=True)
class Violation:
    rule_id: str
    file: str
    symbol: str

    def as_tuple(self) -> tuple[str, str, str]:
        return (self.rule_id, self.file, self.symbol)


def test_async_transaction_architecture_contract() -> None:
    actual = {violation.as_tuple() for violation in scan_production()}
    assert actual == EXPECTED_VIOLATIONS


def test_architecture_guard_positive_fixture() -> None:
    source = """
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True, slots=True)
class AsyncDecodeLayout:
    request_ids: object
    source_request_idxs: object
    graph_shape: object
    request_query_lengths: object
    request_kv_length_offsets: object
    request_to_kv_block_ids: object
    token_to_pos_ids: object
    token_to_request_idx: object
    token_to_position_in_request: object
    token_to_local_position_within_kv_block: object
    token_to_block_idx: object
    mamba_read_indices: object
    mamba_write_indices: object

@dataclass(slots=True)
class AsyncResourceLedger:
    kv_reservations: object
    deferred_kv_blocks: object
    deferred_mamba_slots: object
    in_flight: bool
    consumed_reservation_count: int

class AsyncDecodeContextOps(Protocol):
    def build_prepared_state(self): ...
    def publish_prepared_state(self): ...
    def copy_prepared_input_ids(self): ...
    def current_layout(self): ...
    def queue_h2d_transfer(self): ...
    def current_input_and_position_ids(self): ...
    def row_mapped_reuse_allowed(self): ...
    def register_active_ledger(self): ...
    def clear_active_ledger(self): ...

class AsyncDecodeCoordinator:
    def __init__(self):
        self._prepared_state = None
        self._pending_transaction = None
        self._next_step_id = 0
        self._step_state = None

    def _finalize_transaction(self): ...
    def _discard_prepared_state(self): ...
    def _build_transaction_participants(self): ...
"""
    violations = scan_tree(ast.parse(source), "fixture.py")
    assert not {
        violation.as_tuple()
        for violation in violations
        if violation.symbol in {"AsyncDecodeLayout", "AsyncResourceLedger", "AsyncDecodeCoordinator"}
    }


def test_architecture_guard_negative_fixture() -> None:
    source = """
from dataclasses import dataclass

@dataclass
class AsyncDecodePlan:
    request_ids: object
    row_map: object

class TextGenerationController:
    def __init__(self):
        self._async_step_transaction = None
"""
    violations = {violation.as_tuple() for violation in scan_tree(ast.parse(source), "fixture.py")}
    assert ("async-dataclass-slots", "fixture.py", "AsyncDecodePlan") in violations
    assert ("async-dataclass-fields", "fixture.py", "AsyncDecodePlan") in violations
    assert (
        "controller-forbidden-lifecycle-field",
        "fixture.py",
        "_async_step_transaction",
    ) in violations


def scan_production() -> list[Violation]:
    violations: list[Violation] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        violations.extend(scan_tree(tree, relative))
    return sorted(violations)


def scan_tree(tree: ast.Module, relative: str) -> list[Violation]:
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    violations: list[Violation] = []

    violations.extend(_scan_async_dataclasses(classes, relative))
    violations.extend(_scan_removed_symbols(classes, relative))
    violations.extend(_scan_protocols(classes, relative))
    violations.extend(_scan_coordinator(classes, relative))
    violations.extend(_scan_controller(classes, tree, relative))
    violations.extend(_scan_context(classes, relative))
    violations.extend(_scan_participants(classes, relative))
    return violations


def _scan_async_dataclasses(classes: dict[str, ast.ClassDef], relative: str) -> list[Violation]:
    violations: list[Violation] = []
    if relative == ASYNC_TRANSACTION:
        for required_name in REQUIRED_DATACLASSES:
            if required_name not in classes:
                violations.append(Violation("async-dataclass-missing", relative, required_name))

    for name, node in classes.items():
        if not name.startswith("Async"):
            continue
        if not _is_dataclass(node):
            continue
        if name not in EXACT_DATACLASS_FIELDS:
            violations.append(Violation("async-dataclass-unexpected", relative, name))
            continue
        if not _dataclass_kwarg_enabled(node, "slots"):
            violations.append(Violation("async-dataclass-slots", relative, name))
        frozen_enabled = _dataclass_kwarg_enabled(node, "frozen")
        if name in FROZEN_DATACLASSES and not frozen_enabled:
            violations.append(Violation("async-dataclass-frozen", relative, name))
        fields = _class_field_names(node)
        if fields != EXACT_DATACLASS_FIELDS[name]:
            violations.append(Violation("async-dataclass-fields", relative, name))
    return violations


def _scan_removed_symbols(classes: dict[str, ast.ClassDef], relative: str) -> list[Violation]:
    return [
        Violation("async-removed-symbol", relative, name)
        for name in sorted(REMOVED_ASYNC_SYMBOLS & set(classes))
    ]


def _scan_protocols(classes: dict[str, ast.ClassDef], relative: str) -> list[Violation]:
    violations: list[Violation] = []
    if relative != ASYNC_COORDINATOR:
        return violations
    for protocol_name, expected_methods in PROTOCOL_METHODS.items():
        node = classes.get(protocol_name)
        if node is None:
            violations.append(Violation("async-protocol-missing", relative, protocol_name))
            continue
        methods = tuple(
            item.name
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        if methods != expected_methods:
            violations.append(Violation("async-protocol-methods", relative, protocol_name))
    return violations


def _scan_coordinator(classes: dict[str, ast.ClassDef], relative: str) -> list[Violation]:
    violations: list[Violation] = []
    node = classes.get("AsyncDecodeCoordinator")
    if node is None:
        return violations
    coordinator_fields = set(_assigned_self_attrs(node))
    for field_name in sorted(COORDINATOR_OWNED_FIELDS - coordinator_fields):
        violations.append(Violation("coordinator-missing-owned-field", relative, field_name))
    if "controller" in coordinator_fields:
        violations.append(Violation("coordinator-raw-controller", relative, "controller"))
    method_names = [
        item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if method_names.count("_finalize_transaction") != 1:
        violations.append(Violation("coordinator-finalize-path", relative, "_finalize_transaction"))
    if method_names.count("_discard_prepared_state") != 1:
        violations.append(
            Violation("coordinator-discard-prepared-path", relative, "_discard_prepared_state")
        )
    if method_names.count("_build_transaction_participants") != 1:
        violations.append(
            Violation("coordinator-participant-helper", relative, "_build_transaction_participants")
        )
    return violations


def _scan_controller(
    classes: dict[str, ast.ClassDef], tree: ast.Module, relative: str
) -> list[Violation]:
    violations: list[Violation] = []
    node = classes.get("TextGenerationController")
    if node is not None:
        for attr in sorted(set(_self_attrs(node)) & FORBIDDEN_CONTROLLER_FIELDS):
            violations.append(Violation("controller-forbidden-lifecycle-field", relative, attr))
    if relative == TEXT_CONTROLLER:
        imported = set(_imported_names(tree))
        for name in sorted(imported & FORBIDDEN_CONTROLLER_IMPORTS):
            violations.append(Violation("controller-forbidden-lifecycle-import", relative, name))
    return violations


def _scan_context(classes: dict[str, ast.ClassDef], relative: str) -> list[Violation]:
    violations: list[Violation] = []
    node = classes.get("DynamicInferenceContext")
    if node is None:
        return violations
    attrs = set(_self_attrs(node))
    assigned_attrs = set(_assigned_self_attrs(node))
    for attr in sorted(attrs & FORBIDDEN_CONTEXT_FIELDS):
        violations.append(Violation("context-forbidden-lifecycle-field", relative, attr))
    for attr in sorted(assigned_attrs):
        if attr.startswith("_async_decode_") and not attr.endswith("_workspace"):
            violations.append(Violation("context-invalid-async-workspace-name", relative, attr))
    return violations


def _scan_participants(classes: dict[str, ast.ClassDef], relative: str) -> list[Violation]:
    violations: list[Violation] = []
    for name, node in classes.items():
        if not name.endswith("Participant"):
            continue
        if name in {"AsyncTransactionParticipant", "AsyncParticipantLifecycle"}:
            continue
        declared_fields = set(_class_field_names(node))
        if declared_fields & PARTICIPANT_LIFECYCLE_FIELDS:
            violations.append(Violation("participant-lifecycle-fields", relative, name))
    return violations


def _is_dataclass(node: ast.ClassDef) -> bool:
    return any(_decorator_name(decorator) == "dataclass" for decorator in node.decorator_list)


def _decorator_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dataclass_kwarg_enabled(node: ast.ClassDef, keyword_name: str) -> bool:
    for decorator in node.decorator_list:
        if _decorator_name(decorator) != "dataclass" or not isinstance(decorator, ast.Call):
            continue
        for keyword in decorator.keywords:
            if keyword.arg == keyword_name and isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value)
    return False


def _class_field_names(node: ast.ClassDef) -> tuple[str, ...]:
    fields = []
    for item in node.body:
        target = item.target if isinstance(item, ast.AnnAssign) else None
        if isinstance(target, ast.Name):
            fields.append(target.id)
    return tuple(fields)


def _assigned_self_attrs(node: ast.ClassDef) -> list[str]:
    attrs = []
    for child in ast.walk(node):
        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            for target in targets:
                if isinstance(target, ast.Attribute) and _is_self(target.value):
                    attrs.append(target.attr)
    return attrs


def _self_attrs(node: ast.ClassDef) -> list[str]:
    attrs = []
    for child in ast.walk(node):
        if isinstance(child, ast.Attribute) and _is_self(child.value):
            attrs.append(child.attr)
    return attrs


def _is_self(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "self"


def _imported_names(tree: ast.Module) -> list[str]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.extend(alias.asname or alias.name.partition(".")[0] for alias in node.names)
    return names
