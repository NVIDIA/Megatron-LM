# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU lifecycle controls; these do not establish CUDA reload or RL parity."""
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite import compat


def test_headless_patch_does_not_require_optional_profile(monkeypatch):
    server = ModuleType(compat._VLLM_ASYNC_SERVER_MODULE)
    server.run_headless = lambda args: args.api_server_count
    monkeypatch.setitem(sys.modules, server.__name__, server)
    monkeypatch.delenv('VERL_MLITE_VLLM_SITE', raising=False)
    monkeypatch.setattr(compat, '_vllm_importable', lambda: True)
    assert compat._patch_verl_vllm_headless_api_server_count()
    assert server.run_headless(SimpleNamespace(api_server_count=None)) == 0
    assert server.run_headless(SimpleNamespace(api_server_count=3)) == 3
    assert not compat._patch_verl_vllm_headless_api_server_count()


def test_opaque_nested_hf_config_preserves_foreign_fields(monkeypatch):
    from transformers import AutoConfig, GenerationConfig, PretrainedConfig

    name = 'mlite_cpu_ds41_nested_control'
    monkeypatch.setenv('VERL_MLITE_HF_CONFIG_MODEL_TYPE', name)
    monkeypatch.setattr(compat, '_REGISTERED_HF_CONFIG_TYPES', set())
    assert compat._register_opaque_hf_config()
    raw = {'vocab_size': 32, 'foreign_rope_field': {'literal': 2}, 'bos_token_id': 1}
    config = AutoConfig.for_model(name, text_config=raw)
    assert isinstance(config.get_text_config(), PretrainedConfig)
    assert config.get_text_config().foreign_rope_field == raw['foreign_rope_field']
    assert GenerationConfig.from_model_config(config).bos_token_id == 1


@pytest.fixture
def native_reload(monkeypatch):
    quant = ModuleType('verl.utils.vllm.vllm_quant_utils')
    utils = ModuleType('verl.workers.rollout.vllm_rollout.utils')
    api = ModuleType('vllm.model_executor.model_loader.reload')
    config = ModuleType('vllm.config')
    events = []

    @contextmanager
    def current(value=None):
        yield

    config.set_current_vllm_config = current
    quant.prepare_quanted_weights_for_loading = lambda model: 'legacy-prepare'
    quant.process_quanted_weights_after_loading = lambda model, state: 'legacy-finalize'
    quant.load_quanted_weights = lambda *a, **kw: 'wrong-fallback'
    utils.load_quanted_weights = lambda *a, **kw: 'existing-ds4-wrapper'
    api.initialize_layerwise_reload = lambda model: events.append('initialize')
    api.finalize_layerwise_processing = lambda model, cfg: events.append('finalize')
    mtp = ModuleType('vllm.model_executor.model_loader.mtp_validation')
    mtp.disable_mtp_completeness_check = current
    for module in (quant, utils, api, config, mtp):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(compat, '_vllm_importable', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)

    class DeepseekV41CPUModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vllm_config = SimpleNamespace(model_config=object())
            self.loaded = []

        def load_weights(self, weights):
            self.loaded.extend(weights)
            self.finalize_mhc_broadcast_weights()

        def finalize_mhc_broadcast_weights(self):
            events.append('mhc')

    model = DeepseekV41CPUModel()
    assert compat._patch_verl_dsv41_layerwise_reload()
    return model, quant, utils, events


def test_ds41_reload_clones_buckets_and_finalizes_once(native_reload):
    model, quant, utils, events = native_reload
    runner = SimpleNamespace(model=model)
    state = quant.prepare_quanted_weights_for_loading(model)
    assert state is compat._DSV41_RELOAD_STATE
    first = torch.arange(8)
    utils.load_quanted_weights([('weight', first)], runner)
    first.fill_(99)
    utils.load_quanted_weights([('other', first)], runner)
    assert events == ['initialize']
    assert model.loaded[0][1].tolist() == list(range(8))
    quant.process_quanted_weights_after_loading(model, state)
    assert events == ['initialize', 'finalize', 'mhc']
    assert not model._verl_mlite_dsv41_reload_active
    assert model._verl_mlite_dsv41_deferred == ()
    assert compat._patch_verl_dsv41_layerwise_reload()
    assert utils.load_quanted_weights([], SimpleNamespace(model=object())) == 'existing-ds4-wrapper'


def test_ds41_reload_missing_native_config_fails(native_reload):
    model, quant, _, _ = native_reload
    del model.vllm_config
    with pytest.raises(RuntimeError, match='no vllm_config'):
        quant.prepare_quanted_weights_for_loading(model)
