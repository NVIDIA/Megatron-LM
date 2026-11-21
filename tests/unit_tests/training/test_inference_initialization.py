# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Inference startup must not depend on training batch or progress state."""

import importlib.util
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.core.transformer import TransformerConfig
from megatron.training import arguments, global_vars, initialize
from megatron.training.argument_utils import inference_cfg_container_from_args
from megatron.training.models import GPTModelConfig


@pytest.mark.parametrize("load_checkpoint", [False, True])
@pytest.mark.parametrize("conditional", [False, True])
def test_generate_samples_uses_inference_startup(monkeypatch, load_checkpoint, conditional):
    # This legacy example still imports the removed generation module. Isolate
    # that unrelated dependency so this test exercises its actual startup path.
    generation = ModuleType("megatron.inference.text_generation")
    generation.generate_and_post_process = Mock()
    monkeypatch.setitem(sys.modules, generation.__name__, generation)
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/academic_paper_scripts/detxoify_lm/generate_samples_gpt.py"
    )
    spec = importlib.util.spec_from_file_location("generate_samples_gpt", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    args = Namespace(
        load="checkpoint" if load_checkpoint else None,
        sample_input_file="prompts.jsonl" if conditional else None,
    )
    model = Mock()
    cfg = Mock()
    monkeypatch.setattr(script, "parse_and_validate_args", Mock(return_value=args))
    monkeypatch.setattr(script, "get_args", lambda: args)
    monkeypatch.setattr(script, "inference_cfg_container_from_args", Mock(return_value=cfg))
    calls = {}
    for name in (
        "set_run_config",
        "initialize_megatron",
        "load_checkpoint",
        "generate_and_write_samples_conditional",
        "generate_and_write_samples_unconditional",
    ):
        calls[name] = Mock()
        monkeypatch.setattr(script, name, calls[name])
    monkeypatch.setattr(script, "get_model", Mock(return_value=[model]))

    script.main()

    calls["set_run_config"].assert_called_once_with(cfg)
    calls["initialize_megatron"].assert_called_once_with()
    if load_checkpoint:
        calls["load_checkpoint"].assert_called_once_with([model], None, None)
    else:
        calls["load_checkpoint"].assert_not_called()
    selected = "conditional" if conditional else "unconditional"
    other = "unconditional" if conditional else "conditional"
    calls[f"generate_and_write_samples_{selected}"].assert_called_once_with(model)
    calls[f"generate_and_write_samples_{other}"].assert_not_called()


@pytest.mark.parametrize("restored_samples", [0, 8, 32])
def test_resume_updates_microbatches_before_setup_validation(
    monkeypatch, restored_samples, run_config
):
    from megatron.training import training
    from megatron.training.state import TrainState

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.skip_train = True  # Evaluation-only setup must also restore the calculator.
    run_config.validation.skip_train = True
    run_config.train.micro_batch_size = 1
    run_config.checkpoint.load = "checkpoint"
    args.load = None  # Deliberately stale; cfg.checkpoint owns the load directory.
    args.data_parallel_size = 1
    args.micro_batch_size = 1
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=16,
        vocab_size=128,
    )
    train_state = TrainState()
    model = [SimpleNamespace()]
    monkeypatch.setattr(training, "get_args", lambda: args)
    monkeypatch.setattr(training, "get_train_state", lambda: train_state)
    monkeypatch.setattr(training, "get_timers", Mock())
    monkeypatch.setattr(training, "get_one_logger", lambda: None)
    monkeypatch.setattr(training, "has_nvidia_modelopt", False)
    monkeypatch.setattr(training, "is_gtp_remat_active", lambda args: False)
    monkeypatch.setattr(training, "get_model", Mock(return_value=model))
    monkeypatch.setattr(training, "unwrap_model", lambda value: value)
    # Memory reporting resolves real process groups, which this stubbed setup never builds.
    monkeypatch.setattr(training, "_get_model_pg_collection", lambda model_chunk: None)
    monkeypatch.setattr(training, "report_memory", lambda *args, **kwargs: None)
    events = []

    def restore(*unused_args, **unused_kwargs):
        assert unused_kwargs["restore_training_state"] is True
        train_state.consumed_train_samples = restored_samples
        events.append("load")
        return 7, 0

    def update(consumed_samples, verbose):
        assert consumed_samples == restored_samples
        assert verbose
        events.append("update")

    class ReachedBatchValidation(Exception):
        pass

    def validate():
        assert events == ["load", "update"]
        raise ReachedBatchValidation

    monkeypatch.setattr(training, "load_checkpoint", restore)
    monkeypatch.setattr(training, "update_num_microbatches", update)
    monkeypatch.setattr(training, "get_num_microbatches", validate)
    with pytest.raises(ReachedBatchValidation):
        training.setup_model_and_optimizer(Mock(), model_provider_func=Mock())


@pytest.mark.parametrize("build_tokenizer", [False, True])
@pytest.mark.parametrize("enable_runtime_flags", [False, True])
@pytest.mark.parametrize("training_kwargs", [{}, {"training": False}, {"training": True}])
def test_runtime_services_initialize_training_only_when_requested(
    monkeypatch, build_tokenizer, enable_runtime_flags, training_kwargs, run_config
):
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=16,
        vocab_size=128,
    )
    args = Namespace(enable_experimental=enable_runtime_flags)
    run_config.dist.disable_jit_fuser = enable_runtime_flags
    experimental = Mock()
    jit = Mock()
    monkeypatch.setattr(global_vars, "set_experimental_flag", experimental)
    monkeypatch.setattr(global_vars, "disable_jit_fuser", jit)
    services = {}
    for name in ("_build_tokenizer", "_set_wandb_writer", "_set_telemetry"):
        services[name] = Mock()
        monkeypatch.setattr(global_vars, name, services[name])
    training_services = Mock()
    monkeypatch.setattr(global_vars, "initialize_training_runtime_services", training_services)
    for name in (
        "init_num_microbatches_calculator",
        "_set_tensorboard_writer",
        "_set_timers",
        "_set_energy_monitor",
        "_set_one_logger",
        "_set_adlr_autoresume",
        "_set_train_state",
        "_set_signal_handler",
    ):
        monkeypatch.setattr(global_vars, name, Mock(side_effect=AssertionError(name)))

    global_vars.initialize_runtime_services(
        args, build_tokenizer=build_tokenizer, **training_kwargs
    )

    assert services["_build_tokenizer"].call_count == int(build_tokenizer)
    services["_set_wandb_writer"].assert_called_once_with(args)
    services["_set_telemetry"].assert_called_once_with(
        args, include_training=training_kwargs.get("training", False)
    )
    if training_kwargs.get("training", False):
        training_services.assert_called_once_with(args)
    else:
        training_services.assert_not_called()
    if enable_runtime_flags:
        experimental.assert_called_once_with(True)
        jit.assert_called_once_with()
    else:
        experimental.assert_not_called()
        jit.assert_not_called()
    assert vars(args) == {"enable_experimental": enable_runtime_flags}


def test_inference_telemetry_does_not_read_training_fields(monkeypatch, run_config):
    class InferenceArgs(Namespace):
        def __getattribute__(self, name):
            if name in {"micro_batch_size", "global_batch_size", "train_iters"}:
                raise AssertionError(f"Inference read training field {name}")
            return super().__getattribute__(name)

    monkeypatch.setattr(global_vars, "_detect_gpu_identity", Mock(return_value={}))
    attrs = global_vars.build_telemetry_resource_attrs(
        InferenceArgs(local_rank=0), include_training=False
    )
    assert "dl.batch_size" not in attrs
    assert "megatron.micro_batch_size" not in attrs
    assert "megatron.train_iters" not in attrs


def test_inference_telemetry_keeps_the_cli_sequence_length(monkeypatch, run_config):
    """Inference has no data section: its sequence-length tag is args.seq_length, whether or not
    the model config carries Megatron Core's per-sample maximum."""
    monkeypatch.setattr(global_vars, "_detect_gpu_identity", Mock(return_value={}))
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=2, hidden_size=32, num_attention_heads=4),
        seq_length=1024,
    )
    assert run_config.model.transformer.max_seq_length_per_sample is None
    attrs = global_vars.build_telemetry_resource_attrs(
        Namespace(local_rank=0, seq_length=32), include_training=False
    )
    assert attrs["dl.sequence_length"] == 32


@pytest.mark.parametrize("training_kwargs", [{}, {"training": False}])
def test_inference_distributed_initialization_without_training_fields(
    monkeypatch, training_kwargs, run_config
):
    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    monkeypatch.setattr(global_vars, '_GLOBAL_RUN_CONFIG', cfg)
    for name in (
        "micro_batch_size",
        "global_batch_size",
        "async_save",
        "use_persistent_ckpt_worker",
        "rerun_mode",
        "error_injection_rate",
        "error_injection_type",
        "result_rejected_tracker_filename",
    ):
        delattr(args, name)
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "initialize_runtime_services", Mock())
    monkeypatch.setattr(initialize, "setup_logging", Mock())
    monkeypatch.setattr(initialize, "print_rank_0", Mock())
    monkeypatch.setattr(initialize, "set_default_log_ranks", Mock())
    distributed = Mock()
    seeds = Mock()
    monkeypatch.setattr(initialize, "_initialize_distributed", distributed)
    monkeypatch.setattr(initialize, "_set_random_seed", seeds)
    for name in (
        "init_persistent_async_worker",
        "initialize_rerun_state_machine",
        "_init_autoresume",
        "_compile_dependencies",
        "_initialize_tp_communicators",
    ):
        monkeypatch.setattr(initialize, name, Mock(side_effect=AssertionError(name)))

    initialize.initialize_megatron(allow_no_cuda=True, **training_kwargs)

    distributed.assert_called_once()
    seeds.assert_called_once()
    assert seeds.call_args.args[0] == cfg.rng.seed


def test_inference_rejects_training_tp_overlap_before_initialization(monkeypatch, run_config):
    args = Namespace(tp_comm_overlap=True)
    cfg = inference_cfg_container_from_args(args, build_model_config=False)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", cfg)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    distributed = Mock()
    monkeypatch.setattr(initialize, "_initialize_distributed", distributed)
    with pytest.raises(ValueError, match="fixed training user buffers"):
        initialize.initialize_megatron(allow_no_cuda=True)
    distributed.assert_not_called()


@pytest.mark.parametrize("native_model", [False, True])
@pytest.mark.parametrize("vlm", [False, True])
def test_distributed_training_services_require_explicit_opt_in(
    monkeypatch, run_config, native_model, vlm
):
    from megatron.core.transformer import TransformerConfig
    from megatron.training.models import GPTModelConfig, LLaVAModelConfig

    parser = ArgumentParser()
    arguments.add_megatron_arguments(parser)
    args = parser.parse_args([])
    args.rank = 0
    args.world_size = 1
    args.async_save = False  # Deliberately stale; cfg.checkpoint owns the worker policy.
    args.use_persistent_ckpt_worker = False
    args.tp_comm_overlap = not native_model
    args.num_layers, args.hidden_size, args.num_attention_heads = 4, 128, 8
    args.params_dtype = torch.float32
    args.seq_length, args.decoder_seq_length = 64, 96
    args.micro_batch_size = 7  # Deliberately stale; cfg.train owns the input.
    args.batch_parallel_size = 1
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", args)
    run_config.train.micro_batch_size = 3
    model_config_cls = LLaVAModelConfig if vlm else GPTModelConfig
    run_config.checkpoint.async_save = True
    run_config.checkpoint.use_persistent_ckpt_worker = True
    if native_model:
        run_config.model = model_config_cls(
            transformer=TransformerConfig(
                num_layers=4,
                hidden_size=128,
                num_attention_heads=8,
                tp_comm_overlap=True,
                max_seq_length_per_sample=96 if vlm else 64,
            ),
            seq_length=1024,  # Position capacity must not be used for buffer sizing.
            tp_comm_overlap_cfg={"qkv_fprop": {"method": "ring_exchange"}},
        )
    else:
        from megatron.training.argument_utils import model_config_from_args

        run_config.model = model_config_from_args(args, model_config_cls=model_config_cls)
    monkeypatch.setattr(initialize, "get_args", lambda: args)
    monkeypatch.setattr(initialize, "setup_logging", Mock())
    monkeypatch.setattr(initialize, "print_rank_0", Mock())
    monkeypatch.setattr(initialize, "set_default_log_ranks", Mock())
    events = []
    finalize = run_config.finalize

    def finalize_config():
        finalize()
        events.append("finalize_config")

    monkeypatch.setattr(run_config, "finalize", finalize_config)
    for name in (
        "initialize_runtime_services",
        "init_persistent_async_worker",
        "initialize_rerun_state_machine",
        "_initialize_distributed",
        "_set_random_seed",
        "_init_autoresume",
        "_compile_dependencies",
        "_initialize_tp_communicators",
    ):
        monkeypatch.setattr(
            initialize,
            name,
            Mock(side_effect=lambda *a, service=name, **kw: events.append(service)),
        )

    initialize.initialize_megatron(allow_no_cuda=True, training=True)

    assert events == [
        "finalize_config",
        "initialize_runtime_services",
        "init_persistent_async_worker",
        "initialize_rerun_state_machine",
        "_initialize_distributed",
        "_set_random_seed",
        "_init_autoresume",
        "_compile_dependencies",
        "_initialize_tp_communicators",
    ]
    tp_call = initialize._initialize_tp_communicators.call_args
    assert isinstance(tp_call.args[0], TransformerConfig)
    assert tp_call.kwargs["sequence_length"] == (96 if vlm else 64)
    assert tp_call.kwargs["micro_batch_size"] == 3
    if native_model:
        assert tp_call.args[0] is run_config.model.transformer
        assert tp_call.kwargs["tp_comm_overlap_cfg"] == run_config.model.tp_comm_overlap_cfg
    else:
        assert tp_call.kwargs["tp_comm_overlap_cfg"] == args.tp_comm_overlap_cfg


@pytest.mark.parametrize("use_mcore,first_rank", [(False, True), (True, True), (True, False)])
def test_vlm_evaluation_uses_resolved_execution_width(
    monkeypatch, run_config, use_mcore, first_rank
):
    from megatron.training.config.training_config import MultiModalTokenizerConfig
    from megatron.training.models import LLaVAModelConfig

    # As in the legacy GPT startup test above, isolate the already-removed
    # generation package and wrapper config; this checks the example's inputs.
    generation = ModuleType("megatron.inference.text_generation")
    api = ModuleType(f"{generation.__name__}.api")
    api.generate_and_post_process = Mock()
    forward = ModuleType(f"{generation.__name__}.forward_step")
    forward.ForwardStep = object
    wrapper_config = ModuleType(
        "megatron.core.inference.model_inference_wrappers.inference_wrapper_config"
    )
    wrapper_config.InferenceWrapperConfig = Mock()
    for module in (generation, api, forward, wrapper_config):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    script_dir = Path(__file__).resolve().parents[3] / "examples/multimodal"
    monkeypatch.syspath_prepend(str(script_dir))
    spec = importlib.util.spec_from_file_location(
        "vlm_generation_inputs", script_dir / "run_text_generation.py"
    )
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    transformer = run_config.model.transformer
    transformer.max_seq_length_per_sample = 128
    run_config.model = LLaVAModelConfig(transformer=transformer, seq_length=256)
    run_config.tokenizer = MultiModalTokenizerConfig()
    # No decoder_seq_length: the canonical execution-width flag alone is supported.
    args = Namespace(
        use_mcore_inference=use_mcore,
        use_tiling=False,
        max_num_tiles=1,
        use_thumbnail=False,
        num_frames=1,
        num_workers=0,
        vision_model_type="clip",
        pixel_shuffle=False,
        use_tile_tags=False,
        hidden_size=32,
        inference_batch_times_seqlen_threshold=512,
        fp32_residual_connection=False,
        params_dtype=torch.float32,
        inference_max_seq_length=128,
    )
    monkeypatch.setattr(script, "get_args", lambda: args)
    monkeypatch.setattr(
        script, "get_evaluation_dataloader", lambda *a: [(Mock(), Mock(), 0, "?", [], {})]
    )
    monkeypatch.setattr(script, "get_num_image_embeddings", lambda *a: 4)
    monkeypatch.setattr(script, "get_conversation", lambda *a: "prompt")
    monkeypatch.setattr(script, "is_first_rank", lambda: first_rank)
    monkeypatch.setattr(script, "StaticInferenceContext", Mock())
    monkeypatch.setattr(script, "get_tokenizer", Mock())
    for name in ("InferenceWrapperConfig", "VLMInferenceWrapper", "VLMTextGenerationController"):
        monkeypatch.setattr(script, name, Mock())
    engine = Mock()
    engine.get_new_request_id.return_value = "0"
    engine.generate.return_value = []
    monkeypatch.setattr(script, "StaticInferenceEngine", Mock(return_value=engine))
    generate = Mock(return_value=([], None, None, None))
    monkeypatch.setattr(script, "generate_and_post_process", generate)

    assert (
        list(script.generate_samples(Mock(), script.EvaluationConfig(task="inference"), False))
        == []
    )
    if use_mcore:
        request = engine.generate.call_args.kwargs["inference_requests"][0]
        assert request.decoder_seq_length == 128
    else:
        forward_step = generate.call_args.kwargs["forward_step"]
        assert forward_step.args[3] == 128


@pytest.mark.parametrize("entrypoint", ["mock_vlm", "energon_vlm"])
@pytest.mark.parametrize("position_type", ["learned_absolute", "rope"])
def test_vlm_length_resolution_does_not_resize_learned_tables(
    monkeypatch, entrypoint, position_type
):
    if entrypoint == "mock_vlm":
        import pretrain_vlm as script

        resolve = script.resolve_vlm_sequence_lengths
        monkeypatch.setattr(script.context_parallel, "get_padding", lambda *a: 0)
    else:
        script_dir = Path(__file__).resolve().parents[3] / "examples/multimodal"
        monkeypatch.syspath_prepend(str(script_dir))
        spec = importlib.util.spec_from_file_location("vlm_model_inputs", script_dir / "model.py")
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        resolve = script.resolve_multimodal_sequence_lengths
    monkeypatch.setattr(script, "get_num_image_embeddings", lambda *a, **kw: 4)
    args = Namespace(
        seq_length=64,
        decoder_seq_length=None,
        max_seq_length_per_sample=128 if entrypoint == "energon_vlm" else None,
        max_position_embeddings=64,
        position_embedding_type=position_type,
        img_h=16,
        img_w=16,
        patch_dim=4,
        disable_vision_class_token=True,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        dynamic_resolution=False,
        pixel_shuffle=False,
        vision_model_type="clip",
        use_tile_tags=False,
        max_num_tiles=1,
        tokenizer_prompt_format="mistral",
        use_thumbnail=False,
        num_frames=1,
    )
    if position_type == "learned_absolute":
        with pytest.raises(
            ValueError, match="--max-position-embeddings.*--max-seq-length-per-sample"
        ):
            resolve(args)
        assert args.max_position_embeddings == 64
        args.max_position_embeddings = 256
        resolve(args)
        assert args.max_position_embeddings == 256
    else:
        resolve(args)
        assert args.max_position_embeddings == (128 if entrypoint == "energon_vlm" else 68)


@pytest.mark.parametrize("is_vlm,is_mimo", [(False, False), (True, False), (True, True)])
def test_dynamic_server_uses_inference_checkpoint_loader(monkeypatch, run_config, is_vlm, is_mimo):
    from examples.inference.advanced import vlm_dynamic_inference

    args = Namespace(inference_ckpt_non_strict=False)
    # The load directory is the section's; inference sets exit_on_missing_checkpoint on the
    # section before loading (the namespace is not written).
    run_config.checkpoint.load = "checkpoint"
    assert run_config.checkpoint.exit_on_missing_checkpoint is False
    if is_mimo:
        args.mimo_checkpoint_prefix_map = {"language_model.": "language_model.module.module."}
    model = Mock()
    provider_module = ModuleType("model")
    provider_module.model_provider = Mock()
    monkeypatch.setitem(sys.modules, "model", provider_module)
    monkeypatch.setitem(sys.modules, "model_provider", provider_module)
    monkeypatch.setitem(sys.modules, "mimo_checkpoint_model", provider_module)
    gpt_builders = ModuleType("gpt_builders")
    gpt_builders.gpt_builder = Mock()
    monkeypatch.setitem(sys.modules, "gpt_builders", gpt_builders)
    monkeypatch.setattr(vlm_dynamic_inference, "get_args", lambda: args)
    monkeypatch.setattr(vlm_dynamic_inference, "_get_model", Mock(return_value=[model]))
    load = Mock()
    monkeypatch.setattr(vlm_dynamic_inference, "load_checkpoint", load)

    def check_mimo(received_args, received_model):
        load.assert_called_once_with(
            ddp_model=[model], optimizer=None, opt_param_scheduler=None, strict=True
        )
        assert received_args is args
        assert received_model is model
        model.eval.assert_not_called()

    check = Mock(side_effect=check_mimo)
    monkeypatch.setattr(vlm_dynamic_inference, "_check_mimo_checkpoint_fully_loaded", check)

    assert vlm_dynamic_inference.get_model(is_vlm=is_vlm) is model
    assert run_config.checkpoint.exit_on_missing_checkpoint is True
    assert not hasattr(args, "exit_on_missing_checkpoint")
    load.assert_called_once_with(
        ddp_model=[model], optimizer=None, opt_param_scheduler=None, strict=True
    )
    if is_mimo:
        check.assert_called_once_with(args, model)
    else:
        check.assert_not_called()
    model.eval.assert_called_once()


@pytest.mark.parametrize("is_mimo", [False, True])
def test_dynamic_server_builds_vlm_shards_with_local_parallel_sizes(
    monkeypatch, run_config, is_mimo
):
    """A TP=1 inference shard of a TP=2 launch builds and loads with shard-local sizes."""
    from examples.inference.advanced import vlm_dynamic_inference

    global_config = TransformerConfig(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        expert_tensor_parallel_size=2,
        sequence_parallel=True,
    )
    run_config.model.transformer = global_config
    run_config.checkpoint.load = "checkpoint"
    args = Namespace(inference_ckpt_non_strict=False)
    if is_mimo:
        args.mimo_checkpoint_prefix_map = {"language_model.": "language_model.module.module."}
    provider_module = ModuleType("model")
    provider_module.model_provider = Mock(return_value=Mock())
    monkeypatch.setitem(sys.modules, "model", provider_module)
    monkeypatch.setitem(sys.modules, "mimo_checkpoint_model", provider_module)
    monkeypatch.setattr(vlm_dynamic_inference, "get_args", lambda: args)
    monkeypatch.setattr(vlm_dynamic_inference, "_check_mimo_checkpoint_fully_loaded", Mock())
    load = Mock()
    monkeypatch.setattr(vlm_dynamic_inference, "load_checkpoint", load)

    def get_model(provider, wrap_with_ddp, config=None, pg_collection=None):
        # Like training.get_model: the provider receives the config passed here.
        assert not wrap_with_ddp
        return [
            provider(
                pre_process=True, post_process=True, config=config, pg_collection=pg_collection
            )
        ]

    monkeypatch.setattr(vlm_dynamic_inference, "_get_model", get_model)
    # Every group of the TP=1 shard, e.g. 'tp=1,role=prefill+tp=1,role=decode', has one rank.
    group_names = ("tp", "pp", "cp", "ep", "expt_tp", "dp_cp", "dp", "expt_dp")
    pg_collection = SimpleNamespace(**{name: object() for name in group_names})
    monkeypatch.setattr(vlm_dynamic_inference, "get_pg_size", lambda group: 1)
    checkpoint_group = object()

    model = vlm_dynamic_inference.get_model(
        is_vlm=True, pg_collection=pg_collection, checkpoint_group=checkpoint_group
    )

    kwargs = provider_module.model_provider.call_args.kwargs
    assert kwargs["pg_collection"] is pg_collection
    shard_config = kwargs["config"]
    assert shard_config is not global_config
    # The sizes the disaggregated coordinator checks against the shard's groups.
    assert (
        shard_config.tensor_model_parallel_size,
        shard_config.pipeline_model_parallel_size,
        shard_config.context_parallel_size,
        shard_config.expert_model_parallel_size,
        shard_config.expert_tensor_parallel_size,
    ) == (1, 1, 1, 1, 1)
    assert not shard_config.sequence_parallel
    assert shard_config.hidden_size == global_config.hidden_size
    # The launch-wide config is unchanged.
    assert run_config.model.transformer is global_config
    assert global_config.tensor_model_parallel_size == 2
    assert global_config.expert_tensor_parallel_size == 2
    assert global_config.sequence_parallel
    load.assert_called_once_with(
        ddp_model=[model],
        optimizer=None,
        opt_param_scheduler=None,
        strict=True,
        tp_group=pg_collection.tp,
        pp_group=pg_collection.pp,
        dp_cp_group=pg_collection.dp_cp,
        dp_group=pg_collection.dp,
        expt_dp_group=pg_collection.expt_dp,
        checkpoint_group=checkpoint_group,
    )
