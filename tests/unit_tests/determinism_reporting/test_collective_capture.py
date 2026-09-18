# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""CPU contract checks for opt-in collective snapshots, hooks and replay inputs."""

import copy
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.utils.deterministic

from tools.determinism import collective_capture
from tools.determinism.capture_recipe import Inventory, install_bindings, runtime_signature
from tools.determinism.collective_capture import (
    MAPPINGS,
    CollectiveCapture,
    load_captures,
    load_tensor,
    prepare_replay,
    process_group_options,
    restore_group_options,
)
from tools.determinism.recipe_coverage import DETERMINISTIC, UNVERIFIED, build_report


@pytest.fixture
def mappings(monkeypatch):
    module = ModuleType("megatron.core.tensor_parallel.mappings")

    def copy_mapping(input_, group=None):
        return input_ * 2

    def reduce_mapping(input_, group=None):
        with torch.no_grad():
            input_.add_(100)
        return input_ * 2

    def gather_mapping(
        input_,
        tensor_parallel_output_grad=True,
        group=None,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        return torch.cat([input_, input_], dim=0)

    for name in MAPPINGS.values():
        setattr(module, name, lambda input_, group=None: input_ * 3)
    module.copy_to_tensor_model_parallel_region = copy_mapping
    module.reduce_from_tensor_model_parallel_region = reduce_mapping
    module.gather_from_sequence_parallel_region = gather_mapping
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "gloo")
    return module


def binding(case="copy"):
    return {
        "target": "megatron.core.tensor_parallel.mappings:" + MAPPINGS[case],
        "op_id": "tensor_parallel_mappings",
        "implementation": "mcore:" + MAPPINGS[case],
        "adapter": "tensor_parallel_collective",
    }


def group(rank=0):
    return SimpleNamespace(size=lambda: 2, rank=lambda: rank)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "broadcast", "sliced"])
def test_snapshot_restores_exact_layout_and_bytes(tmp_path, dtype, layout):
    value = torch.arange(24, dtype=dtype).reshape(4, 6)
    value[0, 0], value[0, 1] = -0.0, float("nan")
    if layout == "transpose":
        value = value.t()
    elif layout == "offset":
        value = value[1:, 1:]
    elif layout == "broadcast":
        value = value[:1].expand(4, 6)
    elif layout == "sliced":
        value = value[:, ::2]
    value.requires_grad_()
    store = CollectiveCapture(tmp_path / "capture", max_bytes=4096, max_events=10)
    descriptor = store.snapshot(value)
    restored = load_tensor(store.root, descriptor, max_bytes=4096)
    original = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    assert descriptor["sha256"] == hashlib.sha256(original).hexdigest()
    assert restored.detach().contiguous().view(torch.uint8).numpy().tobytes() == original
    assert restored.shape == value.shape and restored.stride() == value.stride()
    assert restored.storage_offset() == value.storage_offset()
    assert restored.requires_grad == value.requires_grad
    assert restored.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("shape", [(1,), (1, 1), (2, 1), (2, 3, 1)])
def test_snapshot_preserves_singleton_broadcast_views(tmp_path, shape):
    value = torch.tensor([-0.0]).as_strided(shape, (0,) * len(shape))
    store = CollectiveCapture(tmp_path / "capture", max_bytes=4096, max_events=10)
    descriptor = store.snapshot(value)
    restored = load_tensor(store.root, descriptor, max_bytes=4096)
    assert restored.stride() == value.stride()
    assert torch.signbit(restored).all()


def test_event_reservation_bounds_overlapping_snapshots(tmp_path):
    store = CollectiveCapture(tmp_path / 'capture', max_bytes=4096, max_events=1)

    def snapshot(index):
        try:
            store.snapshot(torch.full((2, 3), float(index)))
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(snapshot, range(2)))
    assert sum(results) == 1
    assert len(list(store.root.glob('*.bin'))) == 1


@pytest.mark.parametrize(
    "fault", ["bytes", "size", "digest", "limit", "overlap", "broadcast_bytes"]
)
def test_loading_rejects_corrupt_or_unrestorable_snapshots(tmp_path, fault):
    store = CollectiveCapture(tmp_path / "capture", max_bytes=4096, max_events=10)
    descriptor = store.snapshot(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    path = store.root / (descriptor["sha256"] + ".bin")
    if fault == "bytes":
        path.write_bytes(b"\0" * path.stat().st_size)
    elif fault == "size":
        path.write_bytes(b"\0")
    elif fault == "digest":
        descriptor["sha256"] = "../bad"
    elif fault == "overlap":
        descriptor["stride"] = [1, 1]
    elif fault == "broadcast_bytes":
        descriptor["stride"] = [0, 1]
    with pytest.raises(ValueError):
        load_tensor(store.root, descriptor, max_bytes=1 if fault == "limit" else 4096)


def test_capture_precedes_inplace_collective_and_preserves_gradients(tmp_path, mappings):
    store = CollectiveCapture(tmp_path / "capture", max_bytes=4096, max_events=10)
    inventory = Inventory(torch, 10, collectives=store)
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    original = value.detach().clone()
    selected = binding("reduce")
    function = mappings.reduce_from_tensor_model_parallel_region
    with install_bindings(inventory, [selected]):
        output = mappings.reduce_from_tensor_model_parallel_region(value, group=group())
        output.sum().backward()
    assert mappings.reduce_from_tensor_model_parallel_region is function
    assert torch.equal(output, (original + 100) * 2)
    assert torch.equal(value.grad, torch.full_like(value, 2))
    assert len(store.events) == 2 and len(inventory.operations) == 2
    forward, backward = [event["signature"] for event in store.events]
    captured = load_tensor(
        store.root, forward["configuration"]["collective"]["input"], max_bytes=4096
    )
    assert torch.equal(captured, original)
    gradient = backward["configuration"]["collective"]["gradient"]
    assert gradient["stride"] == [0, 0]
    assert torch.equal(load_tensor(store.root, gradient, max_bytes=4096), torch.ones_like(value))
    assert not store.issues


def test_capture_records_each_backward_and_runtime_change(tmp_path, mappings):
    store = CollectiveCapture(tmp_path / "capture", max_bytes=4096, max_events=10)
    inventory = Inventory(torch, 10, collectives=store)
    previous = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.utils.deterministic.fill_uninitialized_memory = False
        with install_bindings(inventory, [binding()]):
            value = torch.ones(2, 3, requires_grad=True)
            output = mappings.copy_to_tensor_model_parallel_region(value, group=group())
            output.backward(torch.ones_like(value), retain_graph=True)
            torch.utils.deterministic.fill_uninitialized_memory = True
            output.backward(torch.full_like(value, 3))
        assert torch.equal(value.grad, torch.full_like(value, 8))
        assert len(store.events) == 3
        assert [e["call_id"] for e in store.events] == [0, 0, 0]
        signature = store.events[-1]["signature"]
        assert signature["runtime"]["fill_uninitialized_memory"] is False
        assert signature["backward_runtime"]["fill_uninitialized_memory"] is True
        assert len(inventory.operations) == 3
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous


@pytest.mark.parametrize(
    "fault",
    [
        "implicit_group",
        "single_rank",
        "byte_limit",
        "event_limit",
        "split",
        "buffer",
        "output_grad",
    ],
)
def test_unsupported_capture_keeps_training_behavior_but_marks_a_gap(tmp_path, mappings, fault):
    store = CollectiveCapture(
        tmp_path / "capture", max_bytes=1 if fault == "byte_limit" else 4096, max_events=1
    )
    inventory = Inventory(torch, 10, collectives=store)
    kwargs = {"group": group()}
    if fault == "implicit_group":
        kwargs.clear()
    elif fault == "single_rank":
        kwargs["group"] = SimpleNamespace(size=lambda: 1)
    elif fault == "split":
        kwargs["output_split_sizes"] = [1, 1]
    elif fault == "buffer":
        kwargs["use_global_buffer"] = True
    elif fault == "output_grad":
        kwargs["tensor_parallel_output_grad"] = False
    with install_bindings(inventory, [binding("gather_first")]):
        value = torch.ones(2, 3, requires_grad=True)
        result = mappings.gather_from_sequence_parallel_region(value, **kwargs)
        result.sum().backward()
    assert torch.equal(result, torch.ones(4, 3))
    assert torch.equal(value.grad, torch.full_like(value, 2))
    assert store.issues
    assert len(store.events) <= 1


def test_collective_binding_requires_explicit_optin_and_correct_identity(tmp_path, mappings):
    with pytest.raises(ValueError, match="--collective-capture"):
        with install_bindings(Inventory(torch, 10), [binding()]):
            pass
    store = CollectiveCapture(tmp_path / "capture", max_bytes=4096, max_events=10)
    selected = binding()
    selected["implementation"] = "mcore:another_mapping"
    with pytest.raises(ValueError, match="does not match"):
        with install_bindings(Inventory(torch, 10, collectives=store), [selected]):
            pass


def make_capture(tmp_path, monkeypatch, mappings):
    context = {"world_size": 2, "dirty": False, "revision": "a" * 40}
    reports = []
    for rank in range(2):
        monkeypatch.setattr(torch.distributed, "get_rank", lambda rank=rank: rank)
        store = CollectiveCapture(tmp_path / f"rank-{rank}", max_bytes=4096, max_events=10)
        inventory = Inventory(torch, 10, collectives=store)
        with install_bindings(inventory, [binding()]):
            value = torch.full((2, 3), float(rank + 1), requires_grad=True)
            output = mappings.copy_to_tensor_model_parallel_region(value, group=group(rank))
            output.sum().backward()
        report = {
            "schema_version": 1,
            "kind": "determinism_inventory",
            "recipe_id": "cpu-contract",
            "rank": rank,
            "context": context,
            "context_after": context,
            "complete": True,
            "truncated": False,
            "operations": list(inventory.operations.values()),
        }
        store.write(report)
        reports.append(report)
    return reports


def test_capture_rank_contract_and_exact_recipe_join(tmp_path, monkeypatch, mappings):
    inventories = make_capture(tmp_path, monkeypatch, mappings)
    captures = load_captures(tmp_path, max_bytes=4096)
    assert len(captures) == 2 and len(captures[0]["events"]) == 2
    cases = []
    for index in range(2):
        cases.append(
            {
                "case_id": f"cpu-replay-fixture-{index}",
                "status": DETERMINISTIC,
                "observations": [
                    {
                        "rank": rank,
                        "signature": report["events"][index]["signature"],
                        "protocol": {"replays": 3},
                        "status": DETERMINISTIC,
                    }
                    for rank, report in enumerate(captures)
                ],
            }
        )
    evidence = [
        {
            "schema_version": 1,
            "kind": "determinism_coverage",
            "context": inventories[0]["context"],
            "run_id": "cpu-fixture-only",
            "ranks_present": [0, 1],
            "cases": cases,
        }
    ]
    report = build_report(inventories, evidence)
    assert report["counts"][DETERMINISTIC] == 4
    assert report["recipe_status"] == "replay_required"
    inventories[0]["capture_issues"] = ["Collective capture limit reached"]
    report = build_report(inventories, evidence)
    assert report["counts"][UNVERIFIED] == 4
    assert report["capture_issues"]


@pytest.mark.parametrize(
    "fault", ["runtime", "uuid", "members", "nccl", "version", "warn_only", "group_options"]
)
def test_preparation_rejects_runtime_or_physical_rank_changes_before_loading_tensors(
    tmp_path, monkeypatch, mappings, fault
):
    make_capture(tmp_path, monkeypatch, mappings)
    captures = load_captures(tmp_path, max_bytes=4096)
    event = captures[0]["events"][0]
    collective = event["signature"]["configuration"]["collective"]
    collective.update(backend="nccl", device_uuid="GPU-original", nccl_version=[2, 20, 0])
    collective['group_options'] = {'is_high_priority_stream': False, 'config': {}}
    monkeypatch.setattr(
        collective_capture,
        'process_group_options',
        lambda group, device: {'is_high_priority_stream': False, 'config': {}},
    )
    monkeypatch.setattr(torch.distributed, "get_backend", lambda group: "nccl")
    monkeypatch.setattr(
        torch.cuda, "nccl", SimpleNamespace(version=lambda: (2, 20, 0)), raising=False
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-original")
    )

    def unexpected_load(*args, **kwargs):
        pytest.fail("A changed replay contract must be rejected before tensor materialization")

    monkeypatch.setattr(collective_capture, "load_tensor", unexpected_load)
    if fault == "runtime":
        event["signature"]["runtime"][
            "fill_uninitialized_memory"
        ] = not torch.utils.deterministic.fill_uninitialized_memory
    elif fault == "uuid":
        collective["device_uuid"] = "GPU-other"
    elif fault == "members":
        collective["group_ranks"] = [0, 2]
    elif fault == "nccl":
        collective["nccl_environment"]["NCCL_ALGO"] = "different"
    elif fault == "version":
        collective["nccl_version"] = [2, 21, 0]
    elif fault == "group_options":
        collective['group_options']['is_high_priority_stream'] = True
    else:
        monkeypatch.setattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: True)
    with pytest.raises(ValueError, match="differs"):
        prepare_replay(event, tmp_path / "rank-0", group(), max_bytes=4096)


@pytest.fixture
def nccl_options(monkeypatch):
    class Options:
        def __init__(self, is_high_priority_stream=False):
            self.is_high_priority_stream = is_high_priority_stream
            self.split_from = None
            self.enable_reconfigure = False
            self.use_pg_for_symm_mem_rendezvous = False
            self.config = SimpleNamespace(
                blocking=-1, min_ctas=-1, max_ctas=-1, cga_cluster_size=-1, net_name=None
            )

    monkeypatch.setattr(
        torch.distributed, 'ProcessGroupNCCL', SimpleNamespace(Options=Options), raising=False
    )
    return Options


@pytest.mark.parametrize('priority', [False, True])
def test_actual_group_options_round_trip_including_unknown_future_scalar_fields(
    nccl_options, priority
):
    options = nccl_options(is_high_priority_stream=priority)
    options.config.min_ctas = 4
    options.config.max_ctas = 16
    options.config.net_name = 'IB'
    options.use_pg_for_symm_mem_rendezvous = True
    group = SimpleNamespace(_get_backend=lambda device: SimpleNamespace(options=options))
    signature = process_group_options(group, torch.device('cuda'))
    assert signature == {
        'is_high_priority_stream': priority,
        'config': {
            'blocking': -1,
            'min_ctas': 4,
            'max_ctas': 16,
            'cga_cluster_size': -1,
            'net_name': 'IB',
        },
        'flags': {'enable_reconfigure': False, 'use_pg_for_symm_mem_rendezvous': True},
    }
    restored = restore_group_options(signature)
    assert restored is not options
    assert restored.is_high_priority_stream is priority
    assert restored.use_pg_for_symm_mem_rendezvous is True
    assert vars(restored.config) == vars(options.config)
    options.config.future_field = 7
    assert process_group_options(group, torch.device('cuda'))['config']['future_field'] == 7
    with pytest.raises(ValueError, match='fields differ'):
        restore_group_options(process_group_options(group, torch.device('cuda')))


@pytest.mark.parametrize(
    'fault', ['split_from', 'enable_reconfigure', 'use_pg_for_symm_mem_rendezvous', 'opaque_config']
)
def test_unsupported_group_options_cannot_be_erased(nccl_options, fault):
    options = nccl_options()
    if fault == 'opaque_config':
        options.config.opaque = object()
    else:
        setattr(options, fault, object())
    group = SimpleNamespace(_get_backend=lambda device: SimpleNamespace(options=options))
    with pytest.raises(ValueError):
        process_group_options(group, torch.device('cuda'))


def test_new_nccl_environment_overrides_are_recorded(monkeypatch):
    monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
    monkeypatch.setenv('TORCH_NCCL_AVOID_RECORD_STREAMS', '1')
    signature = collective_capture.nccl_environment()
    assert signature['NCCL_P2P_DISABLE'] == '1'
    assert signature['TORCH_NCCL_AVOID_RECORD_STREAMS'] == '1'


def test_backward_communication_policy_change_stays_visible(tmp_path, monkeypatch, mappings):
    monkeypatch.setenv('NCCL_P2P_DISABLE', '0')
    store = CollectiveCapture(tmp_path / 'capture', max_bytes=4096, max_events=10)
    with install_bindings(Inventory(torch, 10, collectives=store), [binding()]):
        value = torch.ones(2, 3, requires_grad=True)
        output = mappings.copy_to_tensor_model_parallel_region(value, group=group())
        monkeypatch.setenv('NCCL_P2P_DISABLE', '1')
        output.sum().backward()
    assert torch.equal(value.grad, torch.full_like(value, 2))
    signature = store.events[-1]['signature']
    assert signature['configuration']['collective']['nccl_environment']['NCCL_P2P_DISABLE'] == '0'
    assert signature['backward_collective']['nccl_environment']['NCCL_P2P_DISABLE'] == '1'


@pytest.mark.parametrize(
    'options',
    [
        None,
        {},
        {'is_high_priority_stream': False, 'config': {}},
        {'is_high_priority_stream': 'false', 'config': {}, 'flags': {}},
    ],
)
def test_incomplete_nccl_options_cannot_match_even_when_both_sides_omit_them(
    tmp_path, monkeypatch, mappings, options
):
    inventories = make_capture(tmp_path, monkeypatch, mappings)
    for inventory in inventories:
        inventory['operations'] = inventory['operations'][:1]
        contract = inventory['operations'][0]['signature']['configuration']['collective']
        contract['backend'] = 'nccl'
        contract['group_options'] = options
    evidence = [
        {
            'schema_version': 1,
            'kind': 'determinism_coverage',
            'context': inventories[0]['context'],
            'run_id': 'incomplete-options-fixture',
            'ranks_present': [0, 1],
            'cases': [
                {
                    'case_id': 'fixture',
                    'status': DETERMINISTIC,
                    'observations': [
                        {
                            'rank': rank,
                            'signature': copy.deepcopy(inventory['operations'][0]['signature']),
                            'protocol': {'replays': 3},
                            'status': DETERMINISTIC,
                        }
                        for rank, inventory in enumerate(inventories)
                    ],
                }
            ],
        }
    ]
    result = build_report(inventories, evidence)
    assert result['counts'][UNVERIFIED] == 2
    assert all('group options' in operation['reason'] for operation in result['operations'])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_rank",
        "incomplete",
        "dirty",
        "context",
        "truncated",
        "gap",
        "order",
        "group",
        "input_shape",
        "gradient_shape",
        "call_id",
        "input_bytes",
        "policy",
        "mixed_policy",
        "higher_order",
    ],
)
def test_replay_preflight_rejects_incomplete_or_changed_contracts(
    tmp_path, monkeypatch, mappings, fault
):
    make_capture(tmp_path, monkeypatch, mappings)
    path = tmp_path / "rank-1/manifest.json"
    report = json.loads(path.read_text())
    if fault == "missing_rank":
        path.unlink()
    elif fault in ("incomplete", "dirty", "context", "truncated", "gap"):
        if fault == "incomplete":
            report["complete"] = False
        elif fault == "dirty":
            report["context"]["dirty"] = True
        elif fault == "context":
            report["context_after"]["revision"] = "b" * 40
        elif fault == "truncated":
            report["truncated"] = True
        else:
            report["capture_issues"] = ["unsupported"]
    elif fault == "order":
        report["events"].reverse()
    else:
        event = report["events"][1]
        signature = event["signature"]
        collective = signature["configuration"]["collective"]
        if fault == "group":
            collective["group_ranks"] = [1, 0]
        elif fault == "input_shape":
            signature["inputs"][0]["shape"] = [3, 2]
        elif fault == "gradient_shape":
            collective["gradient"]["shape"] = [3, 2]
            collective["gradient"]["stride"] = [0, 0]
        elif fault == "call_id":
            event["call_id"] = 9
        elif fault == "input_bytes":
            digest = collective["input"]["sha256"]
            (tmp_path / "rank-1" / (digest + ".bin")).write_bytes(b"x" * 24)
        elif fault == "policy":
            signature["runtime"].pop("fill_uninitialized_memory")
        elif fault == "mixed_policy":
            signature["backward_runtime"] = {
                **runtime_signature(torch),
                "fill_uninitialized_memory": False,
            }
        else:
            collective["backward_grad_enabled"] = True
    if fault != "missing_rank":
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_captures(tmp_path, max_bytes=4096)


@pytest.mark.parametrize("max_bytes,expected_events", [(4096, 2), (1, 0)])
def test_cli_writes_rank_capture_and_propagates_capture_gaps(tmp_path, max_bytes, expected_events):
    module = tmp_path / "collective_probe.py"
    module.write_text(
        "import sys, types, torch\n"
        "from tools.determinism.collective_capture import MAPPINGS\n"
        "module = types.ModuleType('megatron.core.tensor_parallel.mappings')\n"
        "for name in MAPPINGS.values():\n"
        "    setattr(module, name, lambda input_, group=None: input_ * 3)\n"
        "def copy(input_, group=None): return input_ * 2\n"
        "module.copy_to_tensor_model_parallel_region = copy\n"
        "sys.modules[module.__name__] = module\n"
        "torch.distributed.get_process_group_ranks = lambda group: [0, 1]\n"
        "torch.distributed.get_rank = lambda: 0\n"
        "torch.distributed.get_backend = lambda group: 'gloo'\n"
    )
    script = tmp_path / "recipe.py"
    script.write_text(
        "import torch\nfrom types import SimpleNamespace\nfrom collective_probe import copy\n"
        "value = torch.ones(2, 3, requires_grad=True)\n"
        "copy(value, group=SimpleNamespace(rank=lambda: 0, size=lambda: 2)).sum().backward()\n"
        "assert torch.equal(value.grad, torch.full_like(value, 2))\n"
    )
    selected = {**binding(), "target": "collective_probe:copy"}
    bindings = tmp_path / "bindings.json"
    bindings.write_text(json.dumps([selected]))
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(tmp_path), os.getcwd()]),
        "RANK": "0",
        "WORLD_SIZE": "2",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.determinism.capture_recipe",
            "--bindings",
            str(bindings),
            "--output",
            str(tmp_path / "inventory"),
            "--recipe-id",
            "cpu-cli-contract",
            "--collective-capture",
            str(tmp_path / "capture"),
            "--max-collective-bytes",
            str(max_bytes),
            "--",
            str(script),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "inventory/rank-0.json").read_text())
    capture = json.loads((tmp_path / "capture/rank-0/manifest.json").read_text())
    assert report["complete"] and capture["complete"]
    assert len(capture["events"]) == expected_events
    assert report["capture_issues"] == capture["capture_issues"]
    assert bool(report["capture_issues"]) is (max_bytes == 1)
