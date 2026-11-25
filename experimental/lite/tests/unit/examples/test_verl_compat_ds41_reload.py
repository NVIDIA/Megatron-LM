# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU lifecycle controls; these do not establish CUDA reload or RL parity."""
import os
import shlex
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite import compat

pytestmark = pytest.mark.optional


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


def test_sparse_recipe_forwards_contract_to_existing_ray(tmp_path):
    script = Path(__file__).resolve().parents[3] / 'examples/verl/scripts/run_deepseek_v41_dapo.sh'
    env = dict(
        os.environ,
        MODEL_PATH='/unused',
        TRAIN_FILES='/unused/train.parquet',
        VAL_FILES='/unused/val.parquet',
        OUTPUT_ROOT=str(tmp_path),
        DRY_RUN='1',
        NNODES='2',
        NGPUS_PER_NODE='4',
        ROLLOUT_SPEC='False',
        ROLLOUT_SLEEP_LEVEL='1',
        GRAD_OFFLOAD='False',
    )
    argv = shlex.split(subprocess.check_output(['bash', str(script)], env=env, text=True))
    assert 'actor_rollout_ref.rollout.data_parallel_size=4' in argv
    assert 'actor_rollout_ref.rollout.expert_parallel_size=4' in argv
    assert (
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHMLA_SPARSE_DSV41'
        in argv
    )
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides={head_dtype:float32}' in argv
    assert not any(
        arg.startswith('+actor_rollout_ref.rollout.engine_kwargs.vllm.head_dtype=') for arg in argv
    )
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.VERL_MLITE_SKIP_RUNTIME_PATCHES="0"' in argv
    for name in ('MXFP8_GEMM', 'BF16_GEMM', 'MXFP8_PREQUANT', 'MHC_OVERLAP', 'REPLAY_STAGE_LAYERS'):
        assert any(
            arg.startswith('++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BI_' + name + '=')
            for arg in argv
        )
    assert 'actor_rollout_ref.actor.engine.grad_offload=False' in argv
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_BI_MEGA_ATTN="0"' in argv
    env.update(
        ROLLOUT_SPEC='True',
        ROLLOUT_SLEEP_LEVEL='2',
        GRAD_OFFLOAD='True',
        DSV41_ATTENTION_CORE='mega',
        DSV41_HEAD_DTYPE='bfloat16',
    )
    argv = shlex.split(subprocess.check_output(['bash', str(script)], env=env, text=True))
    assert 'actor_rollout_ref.actor.engine.grad_offload=True' in argv
    assert '+actor_rollout_ref.actor.engine.impl_cfg.attention_core=mega' in argv
    assert '+actor_rollout_ref.actor.engine.impl_cfg.head_dtype=bfloat16' in argv
    assert (
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides={head_dtype:bfloat16}' in argv
    )
    assert (
        '+actor_rollout_ref.rollout.engine_kwargs.vllm.attention_backend=FLASHMLA_MEGA_ATTN_DSV41'
        in argv
    )
    assert any('speculative_config={method:dspark,' in a for a in argv)
    assert any(
        'sleep_preserve_parameter_names=' in a and 'draft.*' in a and '*.bias_vl' in a for a in argv
    )
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V2_MODEL_RUNNER="1"' in argv
    assert '++ray_kwargs.ray_init.runtime_env.env_vars.VERL_MLITE_ROLLOUT_SLEEP_LEVEL="2"' in argv
    env['ROLLOUT_SLEEP_LEVEL'] = '3'
    rejected = subprocess.run(['bash', str(script)], env=env, capture_output=True, text=True)
    assert rejected.returncode != 0 and 'must be 1 or 2' in rejected.stderr


def test_retention_range_bisection_is_byte_exact():
    gaps = [(100, 100), (300, 100)]
    owners = [(90, 120), (110, 130), (150, 170), (180, 250), (340, 360)]
    inside = compat._clip_ranges(gaps, owners, True)
    outside = compat._clip_ranges(gaps, owners, False)

    def bytes_of(ranges):
        return {i for start, size in ranges for i in range(start, start + size)}

    original = bytes_of(gaps)
    assert bytes_of(inside) | bytes_of(outside) == original
    assert not bytes_of(inside) & bytes_of(outside)
    assert bytes_of(inside) == original & {i for lo, hi in owners for i in range(lo, hi)}


def test_fingerprint_detects_mutation_and_shared_storage():
    from verl_mlite.c8_fingerprint import fingerprint_model

    model = torch.nn.Linear(3, 2)
    model.register_buffer('shared', model.weight.detach())
    before = fingerprint_model(model)
    assert before['.shared'] == ['alias']
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert before['.weight'] != fingerprint_model(model)['.weight']


def test_level2_retention_is_opt_in(monkeypatch):
    monkeypatch.delenv('VERL_MLITE_L2_RETENTION', raising=False)
    monkeypatch.setattr(
        compat, '_vllm_importable', lambda: pytest.fail('default must not import vllm')
    )
    assert not compat._patch_vllm_sleep_level2_retention()
