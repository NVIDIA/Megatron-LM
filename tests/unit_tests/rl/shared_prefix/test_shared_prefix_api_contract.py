# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract for the shipped shared-prefix packing API that NeMo-RL imports.

The ``megatron.rl`` modules sit outside the ``megatron.core`` API-compatibility
gate, so a renamed symbol or keyword would otherwise reach NeMo-RL unnoticed.
Update this contract together with the NeMo-RL adapter.
"""

import importlib
import importlib.util
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

# Found by marker, not by depth, so that moving this file cannot point the probes elsewhere.
REPO = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())

# Module -> {symbol: what NeMo-RL uses}: keyword arguments it passes and ``.attributes`` it
# reads from the symbol's outputs. Every listed symbol must exist.
NEMO_RL_CONTRACT = {
    "shared_prefix_alignment": {
        "align_physical_units": "units costs capacity target_count padding_multiple",
        "align_training_units": "units costs capacity target_count padding_multiple",
        "materialize_alignment": "units costs capacity target_count",
    },
    "shared_prefix_cost": {
        "estimate_shared_prefix_row_work": "group_ids sequence_lengths prompt_lengths"
        " physical_weight expanded_weight row_slot_ids"
    },
    "shared_prefix_dense_bins": {
        "plan_dense_training_bins": "costs bin_capacity dense_packer",
        "share_prefixes_in_dense_training_bins": "rows units costs padding_multiple bin_capacity",
    },
    "shared_prefix_execution": {
        "SharedPrefixExecutionPlan": "units .units .num_units .max_physical_length",
        "SharedPrefixExecutionUnit": "row_indices shared_layout physical_length",
        "plan_shared_prefix_execution_units": "rows row_slots bin_capacity padding_multiple"
        " pack_groups repack_groups pack_dense_fallbacks merge_dense_fallbacks forward_only"
        " evaluation_packing dense_packer largest_first",
        "validate_shared_prefix_execution_units": "units batch_size",
    },
    "shared_prefix_metadata": {
        "FixedExecutionSlotPlan": ".row_slot_ids",
        "GroupCoherentShardPlan": ".shard_indices .rank_order_permutation",
        "get_prescribed_shared_prefix_slots": "group_ids slot_ids",
        "plan_fixed_execution_slots": "group_ids sequence_lengths bin_capacity batch_size"
        " sequence_length_pad_multiple",
        "plan_group_coherent_shards": "group_ids sequence_lengths num_shards batch_size",
    },
    "shared_prefix_packing": {
        "SharedPrefixForestLayout": ".iter_roots .mtp_loss_group_root_counts .row_indices"
        " .completion_lengths .completion_scatter_rows .physical_total_length .total_length"
        " .tree_layout",
        "SharedPrefixLayout": ".iter_roots .prompt_length .row_indices .completion_lengths"
        " .completion_scatter_rows .physical_total_length .total_length .tree_layout",
        "SharedPrefixPlan": "",
        "SharedPrefixRow": "",
    },
    "shared_prefix_tensors": {
        "SharedPrefixContextParallelShard": ".packed_input_ids .position_ids"
        " .global_token_indices .padded_total_length",
        "SharedPrefixTensorBin": ".layout .packed_input_ids .indices",
        "SharedPrefixTensorIndices": ".completion_positions .predecessor_positions"
        " .completion_scatter_columns",
        "build_shared_prefix_rows": "input_ids input_lengths prompt_lengths group_ids",
        "get_shared_prefix_context_parallel_indices": "padded_total_length cp_rank cp_size device",
        "get_shared_prefix_physical_alignment": "tp_size cp_size",
        "materialize_shared_prefix_layout": "input_ids input_lengths layout",
        "materialize_shared_prefix_token_aligned_tensor": "source tensor_bin padding_value",
        "resolve_shared_prefix_parallel_topology": "tp_size cp_size sequence_parallel",
        "resolve_shared_prefix_physical_padding_multiple": "tp_size cp_size padding_multiple",
        "shard_shared_prefix_tensor_bin_for_context_parallel": "tensor_bin cp_rank cp_size tp_size"
        " padding_multiple",
    },
    "tree_layout": {"PackedTreeLayout": ".iter_star_roots .node_len .logical_node_len"},
}


def _packing_modules() -> list[str]:
    """The dependency-free packing modules: ``shared_prefix_*`` and ``tree_layout``."""
    names = [path.stem for path in (REPO / "megatron" / "rl").glob("shared_prefix_*.py")]
    return sorted(f"megatron.rl.{name}" for name in names + ["tree_layout"])


@pytest.mark.parametrize("module_name", [f"megatron.rl.{name}" for name in NEMO_RL_CONTRACT])
def test_nemo_rl_api_contract(module_name):
    """Symbols, keywords and attributes NeMo-RL reads from one packing module."""
    assert module_name in _packing_modules()
    module = importlib.import_module(module_name)
    for name, uses in NEMO_RL_CONTRACT[module_name.rsplit(".", 1)[1]].items():
        assert hasattr(module, name), f"{module_name}.{name} is imported by NeMo-RL"
        symbol, keywords = getattr(module, name), [u for u in uses.split() if u[0] != "."]
        if keywords:
            parameters = inspect.signature(symbol).parameters
            missing = [keyword for keyword in keywords if keyword not in parameters]
            assert not missing, f"{module_name}.{name} lost NeMo-RL keywords {missing}"
        fields = getattr(symbol, "__dataclass_fields__", {})
        attributes = [u[1:] for u in uses.split() if u[0] == "."]
        missing = [a for a in attributes if not (a in fields or hasattr(symbol, a))]
        assert not missing, f"{name} lost attributes {missing} that NeMo-RL reads"


_UNDECLARED_DEPENDENCY_PROBE = """
import importlib, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("pydantic", "nemo_rl", "yaml", "fastapi", "httpx") or (
            fullname.startswith("megatron.") and not fullname.startswith("megatron.rl")
        ):
            raise ImportError(f"undeclared dependency {fullname}")
sys.meta_path.insert(0, Block())
for name in MODULES:
    importlib.import_module(name)
"""


def _undeclared_dependency_probe():
    """The packing modules need only torch, numpy and packaging."""
    modules = _packing_modules()
    # tree_layout is always listed, so only the shared_prefix modules prove that REPO is right.
    assert {f"megatron.rl.{name}" for name in NEMO_RL_CONTRACT} <= set(modules), modules
    return _UNDECLARED_DEPENDENCY_PROBE.replace("MODULES", repr(modules))


_PURE_PACKING_PROBE = """
from megatron.rl.shared_prefix_packing import SharedPrefixRow, plan_shared_prefix_bins
from megatron.rl.shared_prefix_execution import plan_shared_prefix_execution_units
from megatron.rl.shared_prefix_alignment import align_rows_to_count
from megatron.rl.shared_prefix_cost import estimate_shared_prefix_row_work
rows = [SharedPrefixRow(i, "a", (1, 2), 1) for i in range(2)]
assert len(plan_shared_prefix_bins(rows, bin_capacity=4).shared_bins) == 1
assert not any(k == "torch" or k == "pydantic" or k.startswith("nemo_rl")
               or k.startswith("megatron.core") for k in sys.modules)
"""

_GENERATION_API_PROBE = r"""
import inspect, pickle
from megatron.rl import GenericGenerationArgs, Request, TypeLookupable

args = GenericGenerationArgs(temperature=0.8, top_k=16)
updated = args.add(GenericGenerationArgs(max_tokens=32))
assert updated.temperature == 0.8 and updated.top_k == 16 and updated.max_tokens == 32
request = Request(generation_args=updated)
assert pickle.loads(pickle.dumps(request)) == request
# Pickles written before the split name the package path; they still load.
legacy = pickle.dumps(request, protocol=0).replace(
    b"megatron.rl.generation_api\n", b"megatron.rl\n"
)
assert b"megatron.rl.generation_api" not in legacy
assert pickle.loads(legacy) == request
for public_type in (GenericGenerationArgs, Request, TypeLookupable):
    assert "class " + public_type.__name__ in inspect.getsource(public_type)
"""

_SHARED_GENERATION_TYPES_PROBE = """
import megatron.rl
from megatron.rl.inference import api
assert api.InferenceRequest.__mro__[1] is megatron.rl.Request
assert "megatron.rl.__init__" not in sys.modules
"""

_PYDANTIC = pytest.mark.skipif(not importlib.util.find_spec("pydantic"), reason="needs pydantic")
_LOG_ENV = {"LANGRL_LOG_DIR": "{tmp}", "PATH": ""}
# Id -> (probe, interpreter flags, environment, marks).
IMPORT_PROBES = {
    "packing-modules-without-undeclared-deps": (_undeclared_dependency_probe, (), None, ()),
    "packing-imports-without-torch-or-core": (_PURE_PACKING_PROBE, ("-I", "-S"), None, ()),
    # Importing the RL log module must not print or truncate a shared log file.
    "rl-logging-no-import-side-effects": ("import megatron.rl.logging", (), _LOG_ENV, ()),
    "generation-api-pickle-compat": (_GENERATION_API_PROBE, (), None, _PYDANTIC),
    "rl-submodules-share-generation-types": (_SHARED_GENERATION_TYPES_PROBE, (), None, _PYDANTIC),
}


@pytest.mark.parametrize(
    "code, flags, env",
    [pytest.param(*probe[:3], marks=probe[3], id=name) for name, probe in IMPORT_PROBES.items()],
)
def test_import_contracts(code, flags, env, tmp_path):
    """Each probe runs in a fresh interpreter so that this process's imports cannot hide a bug."""
    log_file = tmp_path / "lang_rl.log"
    log_file.write_text("previous run\n")
    if env is not None:
        env = {key: value.format(tmp=tmp_path) for key, value in env.items()}
    code = "import sys\nsys.path.insert(0, sys.argv[1])\n" + (code() if callable(code) else code)
    result = subprocess.run(
        [sys.executable, *flags, "-c", code, str(REPO)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert log_file.read_text() == "previous run\n"
