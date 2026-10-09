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
import inspect
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

# Module -> {symbol: keyword arguments NeMo-RL passes (None for non-callables)}.
NEMO_RL_CONTRACT = {
    "megatron.rl.shared_prefix_alignment": {
        "AlignedUnit": None,
        "align_rows_to_count": ("units", "padded_row_lengths", "bin_capacity", "target_count"),
        "align_physical_units": ("units", "costs", "capacity", "target_count", "padding_multiple"),
        "align_training_units": ("units", "costs", "capacity", "target_count", "padding_multiple"),
        "materialize_alignment": ("units", "costs", "capacity", "target_count"),
        "rebuild_subset": ("unit", "selected", "padding_multiple"),
    },
    "megatron.rl.shared_prefix_cost": {
        "estimate_shared_prefix_row_work": (
            "group_ids",
            "sequence_lengths",
            "prompt_lengths",
            "physical_weight",
            "expanded_weight",
        )
    },
    "megatron.rl.shared_prefix_dense_bins": {
        "plan_dense_training_bins": ("costs", "bin_capacity", "dense_packer"),
        "share_prefixes_in_dense_training_bins": (
            "rows",
            "units",
            "costs",
            "padding_multiple",
            "bin_capacity",
        ),
    },
    "megatron.rl.shared_prefix_execution": {
        "SharedPrefixExecutionPlan": ("units",),
        "SharedPrefixExecutionUnit": ("row_indices", "shared_layout", "physical_length"),
        "plan_shared_prefix_execution_units": (
            "rows",
            "row_slots",
            "bin_capacity",
            "padding_multiple",
            "pack_groups",
            "repack_groups",
            "pack_dense_fallbacks",
            "merge_dense_fallbacks",
            "forward_only",
            "evaluation_packing",
            "dense_packer",
            "largest_first",
        ),
        "validate_shared_prefix_execution_units": ("units", "batch_size"),
    },
    "megatron.rl.shared_prefix_metadata": {
        "FixedExecutionSlotPlan": None,
        "GroupCoherentShardPlan": None,
        "get_prescribed_shared_prefix_slots": ("group_ids", "slot_ids"),
        "plan_fixed_execution_slots": (
            "group_ids",
            "sequence_lengths",
            "bin_capacity",
            "batch_size",
            "sequence_length_pad_multiple",
            "max_rows_per_slot",
        ),
        "plan_group_coherent_shards": ("group_ids", "sequence_lengths", "num_shards", "batch_size"),
    },
    "megatron.rl.shared_prefix_packing": {
        "SharedPrefixForestLayout": None,
        "SharedPrefixLayout": None,
        "SharedPrefixPlan": None,
        "SharedPrefixRow": None,
        "build_shared_prefix_layout": None,
        "pack_shared_prefix_groups": None,
        "plan_shared_prefix_bins": None,
    },
    "megatron.rl.shared_prefix_tensors": {
        "SharedPrefixContextParallelShard": None,
        "SharedPrefixTensorBin": None,
        "SharedPrefixTensorIndices": None,
        "build_shared_prefix_rows": ("input_ids", "input_lengths", "prompt_lengths", "group_ids"),
        "get_shared_prefix_context_parallel_indices": (
            "padded_total_length",
            "cp_rank",
            "cp_size",
            "device",
        ),
        "get_shared_prefix_physical_alignment": ("tp_size", "cp_size"),
        "materialize_shared_prefix_layout": ("input_ids", "input_lengths", "layout"),
        "materialize_shared_prefix_token_aligned_tensor": ("source", "tensor_bin", "padding_value"),
        "resolve_shared_prefix_parallel_topology": ("tp_size", "cp_size", "sequence_parallel"),
        "resolve_shared_prefix_physical_padding_multiple": (
            "tp_size",
            "cp_size",
            "padding_multiple",
        ),
        "shard_shared_prefix_tensor_bin_for_context_parallel": (
            "tensor_bin",
            "cp_rank",
            "cp_size",
            "tp_size",
            "padding_multiple",
        ),
    },
}

# Attributes the NeMo-RL adapter reads from planner and tensor outputs.
NEMO_RL_ATTRIBUTES = {
    ("megatron.rl.shared_prefix_packing", "SharedPrefixLayout"): (
        "iter_roots",
        "prompt_length",
        "row_indices",
        "completion_lengths",
        "completion_scatter_rows",
        "physical_total_length",
        "total_length",
        "tree_layout",
    ),
    ("megatron.rl.shared_prefix_packing", "SharedPrefixForestLayout"): (
        "iter_roots",
        "mtp_loss_group_root_counts",
        "row_indices",
        "completion_lengths",
        "completion_scatter_rows",
        "physical_total_length",
        "total_length",
        "tree_layout",
    ),
    ("megatron.rl.tree_layout", "PackedTreeLayout"): (
        "iter_star_roots",
        "node_len",
        "logical_node_len",
    ),
    ("megatron.rl.shared_prefix_tensors", "SharedPrefixTensorBin"): (
        "layout",
        "packed_input_ids",
        "indices",
    ),
    ("megatron.rl.shared_prefix_tensors", "SharedPrefixTensorIndices"): (
        "completion_positions",
        "predecessor_positions",
        "completion_scatter_columns",
    ),
    ("megatron.rl.shared_prefix_tensors", "SharedPrefixContextParallelShard"): (
        "packed_input_ids",
        "position_ids",
        "global_token_indices",
        "padded_total_length",
    ),
    ("megatron.rl.shared_prefix_metadata", "GroupCoherentShardPlan"): (
        "shard_indices",
        "rank_order_permutation",
    ),
    ("megatron.rl.shared_prefix_metadata", "FixedExecutionSlotPlan"): ("row_slot_ids",),
}


def _shipped_modules() -> list[str]:
    with open(REPO / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["tool"]["setuptools"]["py-modules"]


@pytest.mark.parametrize("module_name", sorted(NEMO_RL_CONTRACT))
def test_nemo_rl_symbols_exist_with_their_keywords(module_name):
    module = importlib.import_module(module_name)
    for name, keywords in NEMO_RL_CONTRACT[module_name].items():
        assert hasattr(module, name), f"{module_name}.{name} is imported by NeMo-RL"
        if keywords is not None:
            parameters = inspect.signature(getattr(module, name)).parameters
            missing = [keyword for keyword in keywords if keyword not in parameters]
            assert not missing, f"{module_name}.{name} lost NeMo-RL keywords {missing}"


def test_nemo_rl_attributes_exist():
    for (module_name, class_name), attributes in NEMO_RL_ATTRIBUTES.items():
        cls = getattr(importlib.import_module(module_name), class_name)
        fields = getattr(cls, "__dataclass_fields__", {})
        missing = [name for name in attributes if not (name in fields or hasattr(cls, name))]
        assert not missing, f"{class_name} lost attributes {missing} that NeMo-RL reads"


def test_wheel_ships_every_contract_module_and_no_rl_runtime():
    shipped = set(_shipped_modules())
    assert set(NEMO_RL_CONTRACT) | {"megatron.rl.tree_layout"} <= shipped
    assert not any(
        name.startswith(("megatron.rl.rl_", "megatron.rl.agent", "megatron.rl.server"))
        for name in shipped
    )


def test_shipped_modules_import_without_undeclared_dependencies():
    """Base dependencies are torch, numpy and packaging; generation_api needs pydantic."""
    modules = [name for name in _shipped_modules() if name != "megatron.rl.generation_api"]
    code = f"""
import importlib, importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("pydantic", "nemo_rl", "yaml", "fastapi", "httpx") or (
            fullname.startswith("megatron.") and not fullname.startswith("megatron.rl")
        ):
            raise ImportError(f"undeclared dependency {{fullname}}")
sys.meta_path.insert(0, Block())
for name in {modules!r}:
    importlib.import_module(name)
"""
    subprocess.run([sys.executable, "-c", code, str(REPO)], check=True)


def test_rl_logging_has_no_import_side_effects(tmp_path):
    """Importing the RL log module must not print or truncate a shared log file."""
    log_file = tmp_path / "lang_rl.log"
    log_file.write_text("previous run\n")
    code = "import sys; sys.path.insert(0, sys.argv[1]); import megatron.rl.logging"
    result = subprocess.run(
        [sys.executable, "-c", code, str(REPO)],
        check=True,
        capture_output=True,
        text=True,
        env={"LANGRL_LOG_DIR": str(tmp_path), "PATH": ""},
    )
    assert result.stdout == ""
    assert log_file.read_text() == "previous run\n"
