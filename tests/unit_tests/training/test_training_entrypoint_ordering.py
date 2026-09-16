"""Guard the single preparation path and explicit runtime startup boundaries."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
TRAINING_ENTRYPOINTS = (
    "pretrain_gpt.py",
    "pretrain_hybrid.py",
    "pretrain_vlm.py",
    "examples/bert/pretrain_bert.py",
    "examples/t5/pretrain_t5.py",
    "examples/mimo/train.py",
    "examples/post_training/modelopt/finetune.py",
    "train_rl.py",
    "megatron/elastification/pretrain_hybrid_flex.py",
    "examples/multimodal/train.py",
)
ARGS_ONLY_ENTRYPOINTS = (
    "examples/inference/launch_inference_server.py",
    "examples/inference/offline_inference.py",
    "examples/inference/advanced/gpt_dynamic_inference.py",
    "examples/inference/advanced/gpt_dynamic_inference_with_coordinator.py",
    "examples/inference/advanced/gpt_static_inference.py",
    "examples/multimodal/model_converter/vision_model_tester.py",
    "examples/multimodal/run_text_generation.py",
    "tools/run_vlm_text_generation.py",
    "tools/run_text_generation_server.py",
    "examples/academic_paper_scripts/detxoify_lm/generate_samples_gpt.py",
    "tools/run_dynamic_text_generation_server.py",
    "examples/rl/benchmark_refit.py",
    "tools/run_inference_performance_test.py",
    "examples/post_training/modelopt/validate.py",
    "examples/post_training/modelopt/convert_model.py",
    "examples/post_training/modelopt/offline_feature_extract.py",
    "examples/post_training/modelopt/mmlu.py",
    "examples/post_training/modelopt/quantize.py",
    "examples/post_training/modelopt/prune.py",
    "examples/post_training/modelopt/export.py",
    "examples/post_training/modelopt/generate.py",
)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


@pytest.mark.parametrize("entrypoint", TRAINING_ENTRYPOINTS)
def test_training_entrypoint_initializes_services_after_config(entrypoint: str):
    tree = ast.parse((ROOT / entrypoint).read_text())
    parse_calls = _calls(tree, "parse_and_validate_args")
    config_calls = _calls(tree, "pretrain_cfg_container_from_args")
    runtime_calls = _calls(tree, "initialize_runtime_services")
    vocab_calls = _calls(tree, "resolve_tokenizer_vocab_size")
    pretrain_calls = _calls(tree, "pretrain")
    assert len(parse_calls) == len(config_calls) == len(runtime_calls) == len(vocab_calls) == 1
    assert parse_calls[0].lineno < config_calls[0].lineno < runtime_calls[0].lineno
    assert pretrain_calls
    assert runtime_calls[0].lineno < vocab_calls[0].lineno
    assert all(call.lineno > vocab_calls[0].lineno for call in pretrain_calls)
    assert isinstance(runtime_calls[0].args[0], ast.Name)
    assert runtime_calls[0].args[0].id == "args"
    assert isinstance(vocab_calls[0].args[0], ast.Name)
    assert vocab_calls[0].args[0].id == "full_config"
    assert ast.unparse(vocab_calls[0].args[1]) == "args.padded_vocab_size"


@pytest.mark.parametrize("entrypoint", ARGS_ONLY_ENTRYPOINTS)
def test_args_only_entrypoint_initializes_services_before_distributed(entrypoint: str):
    tree = ast.parse((ROOT / entrypoint).read_text())
    parse_calls = _calls(tree, "parse_and_validate_args")
    runtime_calls = _calls(tree, "initialize_runtime_services")
    distributed_calls = _calls(tree, "initialize_megatron")
    assert len(parse_calls) == len(runtime_calls) == len(distributed_calls) == 1
    assert parse_calls[0].lineno < runtime_calls[0].lineno < distributed_calls[0].lineno
    assert isinstance(runtime_calls[0].args[0], ast.Name)
    assert runtime_calls[0].args[0].id == "args"


@pytest.mark.parametrize(
    "entrypoint,adapter",
    [
        ("pretrain_gpt.py", "gpt_config_from_args"),
        ("pretrain_hybrid.py", "hybrid_config_from_args"),
        ("train_rl.py", "gpt_config_from_args"),
        ("train_rl.py", "hybrid_config_from_args"),
    ],
)
def test_model_adapter_selects_tokenizer_vocabulary(entrypoint: str, adapter: str):
    calls = _calls(ast.parse((ROOT / entrypoint).read_text()), adapter)
    assert calls
    for call in calls:
        kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(kwargs["vocab_size_from_tokenizer"]) is True


def test_parser_has_no_runtime_mode_flag():
    tree = ast.parse((ROOT / "megatron/training/arguments.py").read_text())
    parse = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "parse_and_validate_args"
    )
    assert not parse.args.kwonlyargs
    assert [arg.arg for arg in parse.args.args] == [
        "extra_args_provider",
        "ignore_unknown_args",
        "args_defaults",
    ]
    for name in (
        "set_global_variables",
        "initialize_runtime_services",
        "initialize_training_globals",
    ):
        assert not _calls(parse, name)
    assert len(_calls(parse, "set_args")) == 1


def test_pretrain_does_not_reinitialize_services():
    tree = ast.parse((ROOT / "megatron/training/training.py").read_text())
    pretrain = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pretrain"
    )
    assert not _calls(pretrain, "initialize_training_globals")
    assert not _calls(pretrain, "initialize_runtime_services")


def test_no_training_initialization_wrapper():
    tree = ast.parse((ROOT / "megatron/training/global_vars.py").read_text())
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert "initialize_training_globals" not in definitions
