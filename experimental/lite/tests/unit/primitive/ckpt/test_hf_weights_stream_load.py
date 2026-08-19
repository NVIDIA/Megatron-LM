# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from megatron.lite.primitive.ckpt.hf_weights import (  # isort: skip
    SafeTensorReader,
    export_hf_weights,
    load_hf_weights,
)


def _stub_parallel_import(monkeypatch) -> None:
    import megatron.lite.primitive.parallel as parallel

    monkeypatch.setitem(
        parallel.__dict__, "pad_vocab_for_tp", lambda vocab_size, tp_size: vocab_size
    )


def _parallel_state(*, tp_size=1, tp_rank=0):
    return type(
        "ParallelState",
        (),
        {
            "ep_size": 1,
            "ep_rank": 0,
            "tp_size": tp_size,
            "tp_rank": tp_rank,
            "etp_size": 1,
            "etp_rank": 0,
        },
    )()


def test_reader_reuses_cpu_mmap_handle_then_moves_requested_tensor(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "shard.safetensors", "b": "shard.safetensors"}})
    )
    events = []

    class ReadTensor:
        def to(self, device):
            events.append(("to", str(device)))
            return self

    class Handle:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            events.append(("enter", self.device))
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", self.device))

        def get_tensor(self, name):
            events.append(("get", self.device, name))
            return ReadTensor()

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.safe_open",
        lambda *args, device, **kwargs: Handle(device),
    )

    with SafeTensorReader(str(tmp_path)) as reader:
        reader.get_tensor("a", device="cpu")
        reader.get_tensor("b", device="cuda:3")
        reader.get_tensor("a", device=torch.device("cpu"))

    assert events == [
        ("enter", "cpu"),
        ("get", "cpu", "a"),
        ("get", "cpu", "b"),
        ("to", "cuda:3"),
        ("get", "cpu", "a"),
        ("exit", "cpu"),
    ]


def test_reader_checks_unindexed_tensor_without_loading_payload(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "model.safetensors").touch()
    events = []

    class Handle:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

        def keys(self):
            events.append("keys")
            return ["present"]

        def get_tensor(self, name):
            raise AssertionError(f"has_tensor must not load {name}")

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.safe_open",
        lambda *args, **kwargs: Handle(),
    )

    with SafeTensorReader(str(tmp_path), device="cuda:0") as reader:
        assert reader.has_tensor("present")
        assert not reader.has_tensor("missing")

    assert events == ["enter", "keys", "keys", "exit"]


def test_dense_mappings_copy_before_reading_the_next_mapping(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Parameter(torch.zeros(1))
            self.second = nn.Parameter(torch.zeros(1))

    model = Model()
    events = []

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            events.append(("read", name, device))
            if name == "hf_second":
                assert model.first.item() == 1
            return torch.tensor([1 if name == "hf_first" else 2], device=device)

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"first": ["hf_first"], "second": ["hf_second"]}

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    load_hf_weights(model, "unused", Spec(), _parallel_state())

    assert torch.equal(model.first, torch.tensor([1.0]))
    assert torch.equal(model.second, torch.tensor([2.0]))
    assert events == [
        "enter",
        ("read", "hf_first", model.first.device),
        ("read", "hf_second", model.second.device),
        "exit",
    ]


def test_pp_stage_without_final_norm_does_not_match_layer_q_norm(monkeypatch) -> None:
    """A stage-global ``norm.weight`` must not substring-match ``q_norm.weight``."""
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_indices = [0]
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].attn = nn.Module()
            self.layers[0].attn.q_norm = nn.Module()
            self.layers[0].attn.q_norm.weight = nn.Parameter(torch.zeros(128))

    model = Model()

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            assert name == "model.norm.weight"
            return torch.ones(2048, device=device, dtype=target_dtype)

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"norm.weight": ["model.norm.weight"]}

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return None

    ps = _parallel_state()
    ps.pp_size = 2
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )

    load_hf_weights(model, "unused", Spec(), ps)

    assert torch.count_nonzero(model.layers[0].attn.q_norm.weight) == 0


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_dense_fused_gate_up_uses_interleaved_tp2_shard(monkeypatch, tp_rank) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up = nn.Linear(2, 4, bias=False)
            self.gate_up.weight.data.zero_()

    model = Model()
    gate = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    up = torch.arange(100, 108, dtype=torch.float32).reshape(4, 2)

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            assert name == "hf.gate_up.weight"
            return torch.cat([gate, up], dim=0).to(device=device)

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"gate_up.weight": ["hf.gate_up.weight"]}

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return (0, 0)

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    load_hf_weights(
        model,
        "unused",
        Spec(),
        _parallel_state(tp_size=2, tp_rank=tp_rank),
    )

    expected = torch.cat(
        [
            gate.chunk(2, dim=0)[tp_rank],
            up.chunk(2, dim=0)[tp_rank],
        ],
        dim=0,
    )
    assert torch.equal(model.gate_up.weight, expected)


def test_persistent_buffer_is_loaded_by_generic_loader(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("router_expert_bias", torch.zeros(4))

    model = Model()
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0])

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            assert name == "hf.router.expert_bias"
            return expected.to(device=device)

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"router_expert_bias": ["hf.router.expert_bias"]}

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return None

    assert dict(model.named_parameters()) == {}
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    load_hf_weights(model, "unused", Spec(), _parallel_state())

    assert torch.equal(model.router_expert_bias, expected)


def test_mapped_persistent_buffer_missing_from_checkpoint_fails(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("router_expert_bias", torch.zeros(4))

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        @staticmethod
        def first_available(names):
            raise KeyError(names[0])

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"router_expert_bias": ["hf.router_expert_bias"]}

        @staticmethod
        def expert_global_id(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )

    with pytest.raises(
        RuntimeError,
        match=r"Spec.*router_expert_bias.*hf\.router_expert_bias",
    ):
        load_hf_weights(Model(), "unused", Spec(), _parallel_state())


def test_buffer_cannot_be_both_optional_and_expected(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("router_expert_bias", torch.zeros(4))

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"router_expert_bias": ["hf.router.expert_bias"]}

        @staticmethod
        def optional_for_load(name):
            if name == "router_expert_bias":
                return "the constructor owns the default"
            return None

    with pytest.raises(
        ValueError,
        match=(
            r"Spec.*router_expert_bias.*weight_map\(\)/load_weight_map\(\)"
            r".*optional_for_load\(\)"
        ),
    ):
        load_hf_weights(Model(), "unused", Spec(), _parallel_state())


def test_checkpoint_source_without_model_target_fails(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        @staticmethod
        def first_available(names):
            assert names == ["hf.ghost.weight"]
            return names[0]

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"ghost.weight": ["hf.ghost.weight"]}

        @staticmethod
        def expert_global_id(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )

    with pytest.raises(
        RuntimeError,
        match=r"Spec.*hf\.ghost\.weight.*ghost\.weight.*no model target",
    ):
        load_hf_weights(nn.Module(), "unused", Spec(), _parallel_state())


def test_undeclared_buffer_adds_no_warning_or_failure(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("rotary_cache", torch.ones(4))

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

    class LegacySpec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {}

    warnings = []
    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    monkeypatch.setattr("megatron.lite.primitive.utils.log_rank0", warnings.append)

    model = Model()
    load_hf_weights(model, "unused", LegacySpec(), _parallel_state())

    assert warnings == []
    assert torch.equal(model.rotary_cache, torch.ones(4))


def test_nonpersistent_primitive_router_bias_needs_no_checkpoint(
    tmp_path,
    transformer_engine_import_stub,
) -> None:
    from types import SimpleNamespace

    from safetensors.torch import save_file

    transformer_engine_import_stub()
    from megatron.lite.primitive.modules.router import TopKRouter

    router = TopKRouter(
        SimpleNamespace(
            num_experts_per_tok=1,
            num_experts=2,
            router_aux_loss_coef=0.0,
            hidden_size=4,
        ),
        SimpleNamespace(tp_size=1, tp_group=None),
    )

    class Spec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"gate.weight": ["hf.gate.weight"]}

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return None

        @staticmethod
        def is_expert(name):
            return False

        @staticmethod
        def native_to_hf(name, tensor):
            return [("hf.gate.weight", tensor)]

    save_file(
        {"hf.gate.weight": torch.ones_like(router.gate.weight)},
        str(tmp_path / "model.safetensors"),
    )

    load_hf_weights(router, str(tmp_path), Spec(), _parallel_state())

    assert "expert_bias" not in router.state_dict()
    assert torch.equal(router.expert_bias, torch.zeros(2))
    assert torch.equal(router.gate.weight, torch.ones_like(router.gate.weight))

    export_ps = SimpleNamespace(
        pp_size=1,
        tp_size=1,
        tp_group=None,
        ep_size=1,
        ep_group=None,
        etp_size=1,
        etp_group=None,
    )
    exported = dict(export_hf_weights(router, Spec(), export_ps))
    assert set(exported) == {"hf.gate.weight"}


def test_missing_required_hf_tensor_fails_with_spec_and_key_context(
    monkeypatch,
) -> None:
    _stub_parallel_import(monkeypatch)
    model = nn.Linear(1, 1, bias=False)

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        @staticmethod
        def first_available(names):
            raise KeyError(names[0])

    class RequiredSpec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"weight": ["required.hf.weight"]}

        @staticmethod
        def expert_global_id(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )

    with pytest.raises(
        KeyError,
        match=r"RequiredSpec.*weight.*required\.hf\.weight",
    ):
        load_hf_weights(model, "unused", RequiredSpec(), _parallel_state())


def test_mapped_parameter_cannot_be_optional(
    monkeypatch,
) -> None:
    _stub_parallel_import(monkeypatch)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(11)

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        @staticmethod
        def first_available(names):
            raise KeyError(names[0])

    class OptionalSpec:
        num_experts = 0

        @staticmethod
        def weight_map():
            return {"weight": ["optional.hf.weight"]}

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def optional_for_load(name):
            assert name == "weight"
            return "the model constructor owns the default value"

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )

    with pytest.raises(
        ValueError,
        match=r"OptionalSpec.*weight.*weight_map\(\)/load_weight_map\(\)"
        r".*optional_for_load\(\)",
    ):
        load_hf_weights(model, "unused", OptionalSpec(), _parallel_state())
    assert model.weight.item() == 11


def test_missing_required_expert_hf_tensor_fails_with_context(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert0 = nn.Parameter(torch.zeros(1))

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        @staticmethod
        def first_available(names):
            raise KeyError(names[0])

    class RequiredExpertSpec:
        num_experts = 1

        @staticmethod
        def weight_map():
            return {"virtual_expert0": ["required.hf.expert0"]}

        @staticmethod
        def expert_global_id(name):
            return 0

        @staticmethod
        def expert_local_name(name, local_idx):
            assert local_idx == 0
            return "expert0"

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )

    with pytest.raises(
        KeyError,
        match=r"RequiredExpertSpec.*expert0.*required\.hf\.expert0",
    ):
        load_hf_weights(Model(), "unused", RequiredExpertSpec(), _parallel_state())


def test_model_supplies_load_plan_but_primitive_owns_loading(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    events = []

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc, traceback):
            events.append("exit")

        def first_available(self, names):
            events.append(("resolve", tuple(names)))
            return names[-1]

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            events.append(("read", name, device, target_shape))
            return torch.tensor([[3]], device=device, dtype=torch.int32)

    class Spec:
        num_experts = 0

        @staticmethod
        def load_weight_map(base_model, ps, logical_state_keys):
            assert base_model is model
            assert logical_state_keys == ("weight",)
            return {"weight": ["canonical.weight"]}

        @staticmethod
        def hf_name_candidates(native_name, hf_name):
            assert native_name == "weight"
            assert hf_name == "canonical.weight"
            return ["canonical.weight", "alternate.weight"]

        @staticmethod
        def expert_global_id(name):
            return None

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def shard_for_load(name, tensor, ps):
            events.append(("shard", name))
            return tensor + 1

        @staticmethod
        def tp_spec(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    load_hf_weights(model, "unused", Spec(), _parallel_state())

    assert model.weight.item() == 4
    assert model.weight.dtype == torch.float32
    assert events == [
        "enter",
        ("resolve", ("canonical.weight", "alternate.weight")),
        ("read", "alternate.weight", model.weight.device, model.weight.shape),
        ("shard", "weight"),
        "exit",
    ]


def test_expert_mappings_copy_before_reading_the_next_mapping(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert0 = nn.Parameter(torch.zeros(1))
            self.expert1 = nn.Parameter(torch.zeros(1))

    model = Model()

    class Reader:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            if name == "hf_expert1":
                assert model.expert0.item() == 1
            return torch.tensor([1 if name == "hf_expert0" else 2], device=device)

    class Spec:
        num_experts = 2

        @staticmethod
        def weight_map():
            return {"weight0": ["hf_expert0"], "weight1": ["hf_expert1"]}

        @staticmethod
        def expert_global_id(name):
            return int(name.removeprefix("weight"))

        @staticmethod
        def expert_local_name(name, local_idx):
            return f"expert{local_idx}"

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    load_hf_weights(model, "unused", Spec(), _parallel_state())

    assert torch.equal(model.expert0, torch.tensor([1.0]))
    assert torch.equal(model.expert1, torch.tensor([2.0]))


def test_expert_mapping_resolves_qat_parametrized_master(monkeypatch) -> None:
    _stub_parallel_import(monkeypatch)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.expert = nn.Linear(1, 1, bias=False)

    model = Model()
    torch.nn.utils.parametrize.register_parametrization(
        model.expert, "weight", nn.Identity()
    )
    master = model.expert.parametrizations.weight.original
    master.data.zero_()

    class Reader:
        def __init__(self, path):
            assert path == "unused"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            pass

        def get_tensor(self, name, *, device, target_shape=None, target_dtype=None):
            assert name == "hf_expert0"
            return torch.tensor([[7]], device=device, dtype=target_dtype)

    class Spec:
        num_experts = 1

        @staticmethod
        def weight_map():
            return {"virtual_expert0": ["hf_expert0"]}

        @staticmethod
        def expert_global_id(name):
            return 0

        @staticmethod
        def expert_local_name(name, local_idx):
            return "expert.weight"

        @staticmethod
        def hf_to_native(name, tensors):
            return tensors[0]

        @staticmethod
        def tp_spec(name):
            return None

    monkeypatch.setattr(
        "megatron.lite.primitive.ckpt.hf_weights.SafeTensorReader", Reader
    )
    load_hf_weights(model, "unused", Spec(), _parallel_state())

    assert master.item() == 7
