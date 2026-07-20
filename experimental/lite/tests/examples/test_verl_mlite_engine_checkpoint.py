# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

VERL_EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "examples" / "verl"
if str(VERL_EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_EXAMPLE_ROOT))

from verl_mlite.engine.config import MegatronLiteEngineConfig
from verl_mlite.engine.mlite_engine import MegatronLiteEngine


class _Scheduler:
    def __init__(self):
        self.loaded_state = None

    def state_dict(self):
        return {"step": 7, "lr": 0.25}

    def load_state_dict(self, state):
        self.loaded_state = state


@pytest.fixture(autouse=True)
def _single_process_dist(monkeypatch):
    monkeypatch.setattr("verl_mlite.engine.mlite_engine.dist.is_initialized", lambda: False)


def _optimizer_config() -> SimpleNamespace:
    return SimpleNamespace(
        optimizer="adam",
        lr=1e-6,
        min_lr=None,
        min_lr_ratio=None,
        clip_grad=1.0,
        weight_decay=0.1,
        lr_warmup_steps_ratio=0.0,
        total_training_steps=10,
        lr_warmup_steps=0,
        override_optimizer_config={},
    )


def _engine_config(**kwargs) -> MegatronLiteEngineConfig:
    values = {"custom_backend_module": None, "impl_cfg": {"use_thd": True}}
    values.update(kwargs)
    return MegatronLiteEngineConfig(**values)


def _initialized_engine(*, checkpoint_config=None, param_offload=False):
    engine = MegatronLiteEngine(
        model_config=SimpleNamespace(
            local_path="/tmp/qwen35", hf_config={"model_type": "qwen3_5_moe"}, mtp=None
        ),
        engine_config=_engine_config(param_offload=param_offload),
        optimizer_config=_optimizer_config(),
        checkpoint_config=checkpoint_config or {},
    )

    def placement_fn(name):
        return ["placement", name]

    def expert_classifier(name):
        return name.endswith("expert")

    parallel = SimpleNamespace(tp=1, cp=1, pp=1)
    parallel_state = SimpleNamespace(dp_rank=0)
    module = torch.nn.Linear(2, 2)
    optimizer = object()
    scheduler = _Scheduler()
    engine.module = module
    engine.handle = SimpleNamespace(
        _optimizer=optimizer,
        _lr_scheduler=scheduler,
        _config=SimpleNamespace(parallel=parallel),
        _parallel_state=parallel_state,
        _extras={
            "protocol": SimpleNamespace(
                PLACEMENT_FN=placement_fn, EXPERT_CLASSIFIER=expert_classifier
            )
        },
    )
    engine.runtime = object()
    return (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    )


def test_save_checkpoint_forwards_contents_scheduler_and_param_offload_reload(
    tmp_path, monkeypatch
):
    (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    ) = _initialized_engine(checkpoint_config={"save_contents": ["model"]}, param_offload=True)
    to_calls = []
    save_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(tmp_path), global_step=13)

    assert to_calls == [
        {"device": "cuda", "model": True, "optimizer": False, "grad": False},
        {"device": "cpu", "model": True, "optimizer": False, "grad": False},
    ]
    assert sync_calls == [True]
    assert len(save_calls) == 1
    save_args, save_kwargs = save_calls[0]
    assert save_args == (module, optimizer, 13, str(tmp_path), parallel, parallel_state)
    assert save_kwargs["get_placements"] is placement_fn
    assert save_kwargs["is_expert"] is expert_classifier
    assert save_kwargs["save_model"] is True
    assert save_kwargs["save_optimizer"] is False
    assert (
        torch.load(tmp_path / "lr_scheduler.pt", map_location="cpu", weights_only=False)
        == scheduler.state_dict()
    )


def test_save_checkpoint_skips_when_contents_exclude_model_and_optimizer(tmp_path, monkeypatch):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": ["extra"]})
    checkpoint_path = tmp_path / "ckpt"
    save_calls = []
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    engine.save_checkpoint(str(checkpoint_path), global_step=13)

    assert save_calls == []
    assert not checkpoint_path.exists()


def test_load_checkpoint_restores_scheduler_and_param_offload_reload(tmp_path, monkeypatch):
    (
        engine,
        module,
        optimizer,
        scheduler,
        parallel,
        parallel_state,
        placement_fn,
        expert_classifier,
    ) = _initialized_engine(param_offload=True)
    torch.save({"step": 23, "lr": 0.125}, tmp_path / "lr_scheduler.pt")
    to_calls = []
    load_calls = []
    sync_calls = []
    monkeypatch.setattr(engine, "to", lambda **kwargs: to_calls.append(kwargs))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: sync_calls.append(True))
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.load_training_checkpoint",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )

    engine.load_checkpoint(str(tmp_path))

    assert to_calls == [
        {"device": "cuda", "model": True, "optimizer": False, "grad": False},
        {"device": "cpu", "model": True, "optimizer": False, "grad": False},
    ]
    assert sync_calls == [True]
    assert scheduler.loaded_state == {"step": 23, "lr": 0.125}
    assert len(load_calls) == 1
    load_args, load_kwargs = load_calls[0]
    assert load_args == (module, optimizer, str(tmp_path), parallel, parallel_state)
    assert load_kwargs["get_placements"] is placement_fn
    assert load_kwargs["is_expert"] is expert_classifier
    assert load_kwargs["load_model"] is True
    assert load_kwargs["load_optimizer"] is True


def test_hf_model_save_fails_loudly_when_protocol_has_no_export(tmp_path):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": ["hf_model"]})
    engine.handle._model = engine.module
    engine.handle._extras["protocol"] = SimpleNamespace()

    with pytest.raises(RuntimeError, match="does not expose save_hf_weights"):
        engine.save_checkpoint(str(tmp_path), global_step=1)


def test_hf_model_save_fails_loudly_when_model_config_is_missing(tmp_path):
    engine, *_ = _initialized_engine(checkpoint_config={"save_contents": ["hf_model"]})
    engine.handle._model = engine.module
    calls = []
    engine.handle._extras["protocol"] = SimpleNamespace(
        save_hf_weights=lambda *args: calls.append(args)
    )

    with pytest.raises(RuntimeError, match="no model_cfg"):
        engine.save_checkpoint(str(tmp_path), global_step=1)
    assert calls == []


def test_hf_model_only_save_uses_protocol_and_writes_hf_metadata(tmp_path, monkeypatch):
    engine, module, *_ = _initialized_engine(
        checkpoint_config={"save_contents": ["hf_model"]}
    )
    model_cfg = object()
    chunks = [module, object()]
    export_calls = []
    metadata_calls = []

    class Artifact:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, path):
            metadata_calls.append((self.name, path))

    hf_config = Artifact("config")
    hf_config.auto_map = {None: "invalid", "AutoModel": "modeling.Model"}
    engine.model_config.hf_config = hf_config
    engine.model_config.tokenizer = Artifact("tokenizer")
    engine.model_config.processor = Artifact("processor")
    engine.handle._model = module
    engine.handle._extras.update(
        {
            "model_cfg": model_cfg,
            "model_chunks": chunks,
            "protocol": SimpleNamespace(
                save_hf_weights=lambda *args: export_calls.append(args)
            ),
        }
    )
    monkeypatch.setattr(
        "verl_mlite.engine.mlite_engine.save_training_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "hf_model-only save wrote a native checkpoint"
        ),
    )

    engine.save_checkpoint(str(tmp_path), global_step=1)

    hf_path = str(tmp_path / "huggingface")
    assert export_calls == [(chunks, hf_path, model_cfg, engine.handle._parallel_state)]
    assert metadata_calls == [
        ("config", hf_path),
        ("tokenizer", hf_path),
        ("processor", hf_path),
    ]
    assert hf_config.auto_map == {"AutoModel": "modeling.Model"}
