# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import situlu
from megatron.core.transformer.enums import CudaGraphModule
from megatron.core.transformer.moe.megakernel import (
    megakernel_fuses_shared_expert,
    prepare_megakernel_shared_expert_config,
)
from megatron.core.transformer.moe.megakernel.mok import weights
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version


def _mok_transformer_config(**overrides):
    values = {
        "num_layers": 2,
        "bf16": True,
        "moe_grouped_gemm": True,
        "hidden_size": 128,
        "attention_dropout": 0.0,
        "hidden_dropout": 0.0,
        "num_attention_heads": 4,
        "num_moe_experts": 8,
        "moe_ffn_hidden_size": 256,
        "moe_shared_expert_intermediate_size": 256,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "gated_linear_unit": True,
        "activation_func": F.silu,
        "add_bias_linear": False,
        "gradient_accumulation_fusion": True,
        "moe_megakernel_backend": "mok",
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_mxfp8_shared_expert_config_expresses_bf16_module(monkeypatch, recwarn):
    original_recipe = object()
    config = SimpleNamespace(
        fp8="hybrid", fp8_param=True, quant_recipe=original_recipe, sentinel=object()
    )
    monkeypatch.setattr(weights, "_SHARED_EXPERT_BF16_WARNING_EMITTED", False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    shared_config = weights.prepare_shared_expert_config(config)

    assert shared_config is not config
    assert config.fp8 == "hybrid" and config.fp8_param
    assert shared_config.fp8 is None and not shared_config.fp8_param
    assert shared_config.quant_recipe is original_recipe
    assert shared_config.sentinel is config.sentinel
    assert len(recwarn) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"moe_token_dispatcher_type": "alltoall"},
        {"moe_token_dispatcher_type": "flex", "moe_flex_dispatcher_backend": "hybridep"},
        {
            "moe_token_dispatcher_type": "flex",
            "moe_flex_dispatcher_backend": "hybridep",
            "sequence_packing_scheduler": "dp_balanced",
            "max_seqlen_per_dp_cp_rank": 1024,
            "calculate_per_token_loss": True,
            "variable_seq_lengths": True,
        },
        {"moe_shared_expert_gate": True},
        {"moe_router_dtype": None},
        {"moe_router_dtype": "fp32"},
        {"moe_shared_expert_gate": True, "activation_func": situlu},
        {"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": True, "moe_shared_expert_gate": True},
        {
            "fp8": "hybrid",
            "fp8_recipe": "mxfp8",
            "fp8_param": True,
            "cuda_graph_impl": "full_iteration",
            "cuda_graph_modules": [],
            "moe_layer_recompute": True,
        },
    ],
)
@pytest.mark.parametrize("backend", ["mok", "cutemok"])
def test_mok_accepts_key_supported_configurations(overrides, backend):
    config = _mok_transformer_config(**{"moe_megakernel_backend": backend, **overrides})
    assert config.moe_megakernel_backend == backend


def test_megakernel_backend_name_is_validated():
    with pytest.raises(ValueError, match="must be None, 'mok' or 'cutemok'"):
        _mok_transformer_config(moe_megakernel_backend="cute_mok")


@pytest.mark.parametrize("available", [True, False])
def test_deprecated_mok_runs_as_cutemok_when_available(monkeypatch, available):
    from megatron.core.transformer.moe.megakernel.mok import package

    monkeypatch.setattr(package, "cutemok_available", lambda: available)
    with pytest.warns(FutureWarning, match='"mok" is deprecated'):
        assert package.resolve_mok_backend("mok") == ("cutemok" if available else "mok")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert package.resolve_mok_backend("cutemok") == "cutemok"


def test_cutemok_cannot_run_without_its_cute_dsl(monkeypatch):
    gate = pytest.importorskip("cudnn.moe_ep._cutemok.cutedsl._version_gate")
    from megatron.core.transformer.moe.megakernel.mok import package

    def missing():
        raise gate.CuteDslRequirementError("CuTe DSL is not installed")

    monkeypatch.setattr(gate, "require_cutedsl", missing)
    assert not package.cutemok_available.__wrapped__()


@pytest.mark.parametrize("capability", [(9, 0), (10, 7)])
def test_cutemok_cannot_run_on_other_gpus(monkeypatch, capability):
    from megatron.core.transformer.moe.megakernel.mok import package

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)
    assert not package.cutemok_available.__wrapped__()


@pytest.mark.parametrize("backend", ["mok", "cutemok"])
def test_mok_rejects_megatron_fsdp_in_cli_arguments(monkeypatch, backend):
    from megatron.training.arguments import parse_args, validate_args

    argv = ["test_mok_weights.py", "--num-layers", "2", "--hidden-size", "128"]
    argv += ["--num-attention-heads", "4", "--seq-length", "128", "--max-position-embeddings"]
    argv += ["128", "--micro-batch-size", "1", "--num-experts", "8", "--moe-grouped-gemm"]
    argv += ["--moe-megakernel-backend", backend, "--use-megatron-fsdp"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 10)
    monkeypatch.setattr("megatron.training.arguments._print_args", lambda *_args: None)
    with pytest.raises(ValueError, match="MOK has not been validated with Megatron-FSDP"):
        validate_args(parse_args())


@pytest.mark.parametrize("backend", ["mok", "cutemok"])
def test_mok_rejects_megatron_fsdp_in_typed_config(monkeypatch, backend):
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig
    from megatron.training.config import PretrainConfigContainer
    from megatron.training.config.training_config import (
        CheckpointConfig,
        LoggerConfig,
        SchedulerConfig,
        TrainingConfig,
    )
    from megatron.training.models.gpt import GPTModelConfig

    monkeypatch.delenv("CUDA_DEVICE_MAX_CONNECTIONS", raising=False)
    container = PretrainConfigContainer(
        train=TrainingConfig(micro_batch_size=1, global_batch_size=4, train_iters=2),
        model=GPTModelConfig(
            transformer=_mok_transformer_config(moe_megakernel_backend=backend),
            vocab_size=1024,
            seq_length=512,
        ),
        optimizer=OptimizerConfig(optimizer="adam", lr=1e-4, min_lr=0.0),
        scheduler=SchedulerConfig(lr_decay_iters=2),
        logger=LoggerConfig(),
        checkpoint=CheckpointConfig(ckpt_format="fsdp_dtensor"),
        ddp=DistributedDataParallelConfig(
            use_megatron_fsdp=True, data_parallel_sharding_strategy="optim_grads_params"
        ),
    )
    with pytest.raises(ValueError, match="MOK has not been validated with FSDP"):
        container._validate_ddp_config()


def test_native_moe_still_accepts_fp64_router():
    config = _mok_transformer_config(moe_megakernel_backend=None, moe_router_dtype="fp64")
    assert config.moe_router_dtype == "fp64"


@pytest.mark.skipif(not is_te_min_version("2.14.0"), reason="single grouped weights need TE>=2.14")
def test_native_single_grouped_weights_do_not_require_op_fuser():
    config = _mok_transformer_config(
        moe_megakernel_backend=None,
        moe_single_grouped_weight=True,
        moe_use_grouped_tensor=True,
        use_transformer_engine_op_fuser=False,
    )
    assert config.moe_single_grouped_weight
    assert not config.use_transformer_engine_op_fuser


def test_shared_execution_is_independent_of_activation_and_preserves_native_precision():
    for activation in (F.silu, situlu):
        for options in (
            None,
            {},
            {"fwd_num_comm_sms": 32},
            {"shared_expert_mode": "auto"},
            {"shared_expert_mode": "fused"},
        ):
            config = _mok_transformer_config(
                activation_func=activation, moe_megakernel_backend_config=options
            )
            assert megakernel_fuses_shared_expert(config)
        for overrides in (
            {"moe_megakernel_backend_config": {"shared_expert_mode": "native"}},
            {"moe_latent_size": 64},
            {"moe_shared_expert_intermediate_size": 512},
            {"moe_shared_expert_intermediate_size": None},
            {"tensor_model_parallel_size": 4, "sequence_parallel": True},
        ):
            config = _mok_transformer_config(
                activation_func=activation,
                fp8="hybrid",
                fp8_recipe="mxfp8",
                fp8_param=True,
                **overrides,
            )
            assert not megakernel_fuses_shared_expert(config)
            assert prepare_megakernel_shared_expert_config(config) is config
            assert config.fp8_param and config.fp8 == "hybrid"
    for overrides in (
        {"moe_latent_size": 64},
        {"moe_shared_expert_intermediate_size": 512},
        {"moe_shared_expert_intermediate_size": None},
        {"tensor_model_parallel_size": 4, "sequence_parallel": True},
    ):
        with pytest.raises(ValueError, match="Fused MOK shared execution requires"):
            _mok_transformer_config(
                moe_megakernel_backend_config={"shared_expert_mode": "fused"}, **overrides
            )
    for mode in ("invalid", None):
        with pytest.raises(ValueError, match="must be auto, fused, or native"):
            _mok_transformer_config(moe_megakernel_backend_config={"shared_expert_mode": mode})
    assert not megakernel_fuses_shared_expert(_mok_transformer_config(moe_megakernel_backend=None))
    for options in ({"shared_expert_mode": "auto"}, {"fwd_num_comm_sms": 32}):
        with pytest.raises(ValueError, match="requires moe_megakernel_backend to be set"):
            _mok_transformer_config(
                moe_megakernel_backend=None, moe_megakernel_backend_config=options
            )
    with pytest.raises(ValueError, match="requires sequence_parallel=True"):
        _mok_transformer_config(tensor_model_parallel_size=4)
    with pytest.raises(ValueError, match="requires expert TP=1"):
        _mok_transformer_config(expert_tensor_parallel_size=2)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (
            {"moe_shared_expert_intermediate_size": None, "add_bias_linear": True},
            "MOK requires add_bias_linear=False",
        ),
        ({"bf16": False, "fp16": True}, "FP32, FP16, and FP4 are not supported"),
        ({"fp4": "e2m1"}, "FP32, FP16, and FP4 are not supported"),
        ({"moe_grouped_gemm": False}, "moe_grouped_gemm=True"),
        ({"moe_router_dtype": "fp64"}, "MOK does not support moe_router_dtype='fp64'"),
        ({"moe_virtual_expert_load_balance": True}, "MOK does not support MoonEP"),
        ({"use_transformer_engine_op_fuser": True}, "use_transformer_engine_op_fuser=False"),
        (
            {"overlap_moe_expert_parallel_comm": True},
            "does not support overlap_moe_expert_parallel_comm",
        ),
        ({"gradient_accumulation_fusion": False}, "gradient_accumulation_fusion=True"),
        ({"fp8": "hybrid", "fp8_recipe": "mxfp8", "fp8_param": False}, "fp8_param=True"),
        ({"moe_shortcut_connection": True}, "does not support moe_shortcut_connection"),
        ({"add_bias_linear": True}, "add_bias_linear"),
        ({"moe_apply_probs_on_input": True}, "moe_apply_probs_on_input"),
        ({"glu_linear_offset": 1.0}, "glu_linear_offset"),
        ({"activation_func_tanh_clamp_scale": 1.0}, "activation_func_tanh_clamp_scale"),
        ({"gtp_weight_remat_size": 2}, "GTP"),
        ({"expert_gtp_weight_remat_size": 2}, "GTP"),
        (
            {
                "transformer_impl": "inference_optimized",
                "moe_router_dtype": "fp32",
                "inference_grouped_gemm_backend": "torch",
            },
            "does not support inference_optimized",
        ),
    ],
)
@pytest.mark.parametrize("backend", ["mok", "cutemok"])
def test_mok_rejects_key_incompatible_configurations(overrides, error, backend):
    with pytest.raises(ValueError, match=error):
        _mok_transformer_config(**{"moe_megakernel_backend": backend, **overrides})


@pytest.mark.parametrize("backend", ["mok", "cutemok"])
@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_rejects_per_layer_whole_layer_cuda_graph(cuda_graph_impl, backend):
    with pytest.raises(ValueError, match="whole-layer CUDA Graph capture"):
        _mok_transformer_config(
            moe_megakernel_backend=backend, cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=[]
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
@pytest.mark.parametrize(
    "cuda_graph_modules",
    [
        [CudaGraphModule.moe],
        [CudaGraphModule.moe_router],
        [CudaGraphModule.moe_preprocess],
        [CudaGraphModule.moe_router, CudaGraphModule.moe_preprocess],
    ],
)
@pytest.mark.parametrize("backend", ["mok", "cutemok"])
def test_mok_rejects_per_layer_cuda_graph_covering_moe(
    cuda_graph_impl, cuda_graph_modules, backend
):
    with pytest.raises(ValueError, match="moe/moe_router/moe_preprocess"):
        _mok_transformer_config(
            moe_megakernel_backend=backend,
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_modules=cuda_graph_modules,
        )


@pytest.mark.parametrize("cuda_graph_impl", ["local", "transformer_engine"])
def test_mok_accepts_per_layer_cuda_graph_outside_moe(cuda_graph_impl):
    cuda_graph_modules = [CudaGraphModule.attn]
    config = _mok_transformer_config(
        cuda_graph_impl=cuda_graph_impl, cuda_graph_modules=cuda_graph_modules
    )

    assert config.cuda_graph_modules == cuda_graph_modules
