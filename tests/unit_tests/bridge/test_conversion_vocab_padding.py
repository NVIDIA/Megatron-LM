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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/conversion/launcher/test_utils.py
# and https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/c209ed397fb36272c2216053058cda4e37802f2d/tests/unit_tests/conversion/launcher/test_gpu_backend.py
# Source license: Apache-2.0.
"""The conversion example pads a vocabulary that the conversion's tensor-parallel size does not divide."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

_SCRIPT_DIR = Path(__file__).resolve().parents[3] / "examples" / "bridge" / "conversion"


def _load_script(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        f"conversion_{name}_under_test", _SCRIPT_DIR / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def conversion_utils() -> types.ModuleType:
    return _load_script("utils")


@pytest.fixture
def gpu_backend(monkeypatch):
    # gpu_backend.py imports its sibling as ``utils``, as it does when run from its own directory.
    monkeypatch.syspath_prepend(str(_SCRIPT_DIR))
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    yield _load_script("gpu_backend")
    sys.modules.pop("utils", None)


def _model_config(*, vocab_size: int, should_pad_vocab: bool = False) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        vocab_size=vocab_size,
        should_pad_vocab=should_pad_vocab,
        pipeline_model_parallel_layout=None,
        transformer=types.SimpleNamespace(),
        finalize=lambda: None,
    )


@pytest.mark.parametrize(
    ("vocab_size", "tp", "padded"),
    [(127, 2, True), (151936, 3, True), (128, 2, False), (32000, 8, False), (50257, 1, False)],
)
def test_configure_model_config_pads_only_a_vocabulary_tp_does_not_divide(
    conversion_utils, vocab_size, tp, padded
):
    model_config = _model_config(vocab_size=vocab_size)

    conversion_utils._configure_model_config(
        model_config, tp=tp, pp=1, ep=1, etp=1, dtype=torch.bfloat16
    )

    assert model_config.should_pad_vocab is padded
    assert model_config.transformer.tensor_model_parallel_size == tp


def test_configure_model_config_keeps_requested_padding(conversion_utils):
    model_config = _model_config(vocab_size=128, should_pad_vocab=True)

    conversion_utils._configure_model_config(
        model_config, tp=2, pp=1, ep=1, etp=1, dtype=torch.bfloat16
    )

    assert model_config.should_pad_vocab is True


@pytest.mark.parametrize(("vocab_size", "padded"), [(127, True), (128, False)])
def test_export_pads_a_vocabulary_that_tp_does_not_divide(
    gpu_backend, monkeypatch, tmp_path, vocab_size, padded
):
    calls = []
    model_config = _model_config(vocab_size=vocab_size)

    class FakeBridge:
        hf_pretrained = types.SimpleNamespace(config=None)

        def get_model_config(self):
            return model_config

        def _get_or_initialize_pg_collection(self, transformer):
            return None

        def load_megatron_model(self, *args, **kwargs):
            calls.append(("load_megatron_model", args, kwargs))
            return ["megatron-model"]

        def save_hf_pretrained(self, *args, **kwargs):
            calls.append(("save_hf_pretrained", args, kwargs))

    monkeypatch.setattr(
        gpu_backend, "_ensure_distributed_initialized", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(gpu_backend, "_prepare_distributed_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        gpu_backend, "resolve_hf_model_revision", lambda model, revision, **kwargs: model
    )
    monkeypatch.setattr(gpu_backend, "_maybe_restore_pipeline_layout", lambda *args, **kwargs: None)
    monkeypatch.setattr(gpu_backend, "is_safe_repo", lambda **kwargs: False)
    monkeypatch.setattr(
        gpu_backend.AutoBridge, "from_hf_pretrained", lambda *args, **kwargs: FakeBridge()
    )
    monkeypatch.setattr(
        gpu_backend.AutoBridge,
        "from_auto_config",
        lambda *args, **kwargs: types.SimpleNamespace(hf_pretrained=None),
    )

    gpu_backend.export_checkpoint.__wrapped__(
        hf_model="hf",
        hf_revision=None,
        megatron_path=str(tmp_path),
        hf_path=str(tmp_path / "hf-export"),
        tp=2,
        pp=1,
        ep=1,
        etp=1,
        torch_dtype="bfloat16",
        trust_remote_code=False,
        strict=True,
        show_progress=False,
        distributed_save=False,
        save_every_n_ranks=1,
        distributed_timeout_minutes=None,
        export_weight_dtype=None,
        overwrite=False,
    )

    # The model is rebuilt from the checkpoint's run config, which an unpadded TP=1 import may have written.
    overrides = next(call for call in calls if call[0] == "load_megatron_model")[2]["mp_overrides"]
    assert overrides.get("should_pad_vocab", False) is padded
