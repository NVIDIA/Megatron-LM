# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Example script for pruning a GPT / Mamba model using Model Optimizer (ModelOpt).

Read more about ModelOpt pruning at https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/pruning
"""

import functools
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import modelopt.torch.opt as mto
import modelopt.torch.prune as mtp
from modelopt.torch.export import import_mcore_gpt_from_hf
from modelopt.torch.export.plugins.megatron_importer import GPTModelImporter
from modelopt.torch.prune.plugins.mcore_minitron import SUPPORTED_HPARAMS
from modelopt.torch.utils.dataset_utils import (
    get_dataset_samples,
    get_supported_datasets,
)
from modelopt.torch.utils.plugins import megatron_generate, megatron_prefill
from utils import get_hf_tokenizer

from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_tensor_model_parallel_group,
)
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.checkpointing import load_modelopt_checkpoint
from megatron.post_training.model_builder import modelopt_gpt_hybrid_builder
from megatron.post_training.utils import report_current_memory_info
from megatron.training import get_args, get_model, initialize_megatron
from megatron.training.arguments import parse_and_validate_args
from megatron.training.checkpointing import save_checkpoint
from megatron.training.utils import print_rank_0, unwrap_model
from model_provider import model_provider

warnings.filterwarnings("ignore")


# WAR: ModelOpt <= 0.44 `_gated_mlp_merging` calls `module.load_state_dict`
# with a state_dict containing only `weight` (+ optional `weight_quantizer._scale`).
# Strict load into a fused TELayerNormColumnParallelLinear (used under
# --export-default-te-spec) errors with
# `Missing key(s) ... layer_norm_weight, _extra_state`. Wrap load_state_dict
# to seed those keys from the freshly-initialized module state; the `fused_norm`
# rule called later overwrites layer_norm_weight with the real value. Fixed
# upstream in ModelOpt 0.45; safe no-op against that version (the upstream
# call already supplies the keys, so our injection is skipped).
_original_gated_mlp_merging = GPTModelImporter._gated_mlp_merging


def _gated_mlp_merging_war(self, module, *args, **kwargs):
    module_state_dict = module.state_dict()
    if "layer_norm_weight" not in module_state_dict:
        return _original_gated_mlp_merging(self, module, *args, **kwargs)

    original_load = module.load_state_dict

    def _patched_load(state_dict, *load_args, **load_kwargs):
        state_dict.setdefault(
            "layer_norm_weight", module_state_dict["layer_norm_weight"]
        )
        state_dict.setdefault("_extra_state", None)
        return original_load(state_dict, *load_args, **load_kwargs)

    module.load_state_dict = _patched_load
    try:
        return _original_gated_mlp_merging(self, module, *args, **kwargs)
    finally:
        del module.load_state_dict


GPTModelImporter._gated_mlp_merging = _gated_mlp_merging_war


def _load_fused_norms_from_hf_war(model, hf_path, dtype):
    """WAR: modelopt <= 0.44 only loads `fused_norm` for Nemotron-H. For GPT models
    (Qwen3, Llama, ...) under --export-default-te-spec, layer_norm_weight inside the
    fused TELayerNormColumnParallelLinear stays at random init. Walk the model and
    copy the HF norm weights into the fused linears."""

    hf_path = Path(hf_path)
    index_file = hf_path / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:
        weight_map = {}
        for stf in hf_path.glob("*.safetensors"):
            with safe_open(str(stf), framework="pt") as f:
                for k in f.keys():
                    weight_map[k] = stf.name

    def _maybe_load(key):
        # Skip if the key isn't present — e.g. Nemotron-H uses `backbone.layers.{}.norm.weight`,
        # and modelopt already handles those via its `fused_norm` rule. The WAR is a no-op there.
        if key not in weight_map:
            return None
        with safe_open(str(hf_path / weight_map[key]), framework="pt") as f:
            return f.get_tensor(key)

    def _copy_into(param, key):
        t = _maybe_load(key)
        if t is None:
            return
        param.data.copy_(t.to(dtype=dtype, device=param.device))

    for layer in model.decoder.layers:
        i = layer.layer_number - 1
        attn = getattr(layer, "self_attention", None)
        if (
            attn is not None
            and hasattr(attn, "linear_qkv")
            and getattr(attn.linear_qkv, "layer_norm_weight", None) is not None
        ):
            _copy_into(
                attn.linear_qkv.layer_norm_weight,
                f"model.layers.{i}.input_layernorm.weight",
            )
        mlp = getattr(layer, "mlp", None)
        if (
            mlp is not None
            and hasattr(mlp, "linear_fc1")
            and getattr(mlp.linear_fc1, "layer_norm_weight", None) is not None
        ):
            _copy_into(
                mlp.linear_fc1.layer_norm_weight,
                f"model.layers.{i}.post_attention_layernorm.weight",
            )


def add_prune_args(parser):
    """Add additional arguments for ModelOpt pruning."""
    group = parser.add_argument_group(title="ModelOpt pruning")
    group.add_argument(
        "--calib-size",
        type=int,
        default=1024,
        help="Samples to use for pruning calibration.",
    )
    group.add_argument(
        "--calib-dataset",
        type=str,
        default="nemotron-post-training-dataset-v2",
        help=(
            f"HF Dataset name or local .jsonl path for calibration "
            f"(supported options: {', '.join(get_supported_datasets())}). "
            "You can also pass any other dataset and see if auto-detection works."
        ),
    )
    group.add_argument(
        "--calib-max-sequence-length",
        type=int,
        default=512,
        help="Maximum sequence length for calibration samples.",
    )
    group.add_argument(
        "--prompts",
        type=str,
        default=("Hello!|Born in California, Soyer trained as a"),
        help="Input texts. Please use | to separate different batches.",
    )
    group.add_argument(
        "--references",
        type=str,
        default="",
        help="Reference texts. Please use | to separate different batches.",
    )
    group.add_argument(
        "--pretrained-model-path",
        type=str,
        default=None,
        help="HuggingFace pretrained model",
    )
    # Pruning targets
    group.add_argument(
        "--prune-export-config",
        type=str,
        required=True,
        help=(
            'Target pruned config as a JSON object, e.g. \'{"hidden_size": 3584, '
            '"ffn_hidden_size": 9216}\'. '
            f"Supported hyperparameters: {sorted(SUPPORTED_HPARAMS)}."
        ),
    )
    group.add_argument(
        "--prune-intermediate-ckpt",
        type=str,
        default=None,
        help=(
            "Directory to cache and reuse per-rank intermediate pruning scores "
            "for resuming / faster re-runs (e.g. pruning the same model to a different config)."
        ),
    )
    add_modelopt_args(parser)
    return parser


def check_arguments(args):
    """Validate user-provided pruning arguments."""
    try:
        args.prune_export_config = json.loads(args.prune_export_config)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON for --prune-export-config: {args.prune_export_config}"
        ) from exc
    if not isinstance(args.prune_export_config, dict):
        raise ValueError("--prune-export-config must parse to a dictionary.")
    unsupported = set(args.prune_export_config) - set(SUPPORTED_HPARAMS)
    if unsupported:
        raise ValueError(
            f"Unsupported hyperparameters in --prune-export-config: {sorted(unsupported)}. "
            f"Supported: {sorted(SUPPORTED_HPARAMS)}"
        )

    # Default the intermediate-checkpoint location to <save>/modelopt_pruning_scores
    # so that re-running on the same --save target reuses cached per-rank scores
    if args.prune_intermediate_ckpt is None and args.save is not None:
        args.prune_intermediate_ckpt = os.path.join(
            args.save, "modelopt_pruning_scores"
        )
        print_rank_0(
            "No directory provided to cache per-rank intermediate pruning scores. "
            f"Setting to: {args.prune_intermediate_ckpt}"
        )


def get_params(model):
    params = sum(p.numel() for p in model.parameters())
    reduced_params = torch.Tensor([params]).to(device=next(model.parameters()).device)
    torch.distributed.all_reduce(
        reduced_params, group=get_pipeline_model_parallel_group()
    )
    torch.distributed.all_reduce(
        reduced_params, group=get_tensor_model_parallel_group()
    )
    return reduced_params.item()


if __name__ == "__main__":
    parse_and_validate_args(
        extra_args_provider=add_prune_args,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "no_load_rng": True,
            "no_load_optim": True,
        },
    )
    initialize_megatron()

    args = get_args()
    check_arguments(args)

    tokenizer = get_hf_tokenizer()
    # Pruning operates on per-expert linears (which only exist as separate modules under
    # SequentialMLP, not the packed-tensor TEGroupedMLP). `disable_moe_grouped_gemm=True`
    # forces the export spec to SequentialMLP so mtp.prune can act on individual experts.
    # Other example scripts (quantize.py, generate.py, finetune.py) keep the default.
    prune_builder = functools.partial(modelopt_gpt_hybrid_builder, disable_moe_grouped_gemm=True)
    model = get_model(
        functools.partial(model_provider, prune_builder),
        wrap_with_ddp=False,
    )
    unwrapped_model = unwrap_model(model)[0]
    print_rank_0(f"Original Model: {unwrapped_model}")

    report_current_memory_info()

    if args.load is not None:
        load_modelopt_checkpoint(
            model, strict=not args.untie_embeddings_and_output_weights
        )
        print_rank_0("Done loading checkpoint")

    if args.pretrained_model_path is not None:
        import_dtype = torch.float16 if args.fp16 else torch.bfloat16
        workspace_dir = os.environ.get("MLM_WORK_DIR", "/tmp")
        import_kwargs = {
            "dtype": import_dtype,
            "trust_remote_code": args.trust_remote_code,
        }
        import_mcore_gpt_from_hf(
            unwrapped_model, args.pretrained_model_path, workspace_dir, **import_kwargs
        )
        _load_fused_norms_from_hf_war(
            unwrapped_model, args.pretrained_model_path, import_dtype
        )

    def _custom_prompt_forward_loop_func(model):
        all_prompts = args.prompts.split("|")
        if args.references == "":
            all_references = [None] * len(all_prompts)
        else:
            all_references = args.references.split("|")

        for idx, prompt in tqdm(
            enumerate(all_prompts), disable=torch.distributed.get_rank()
        ):
            tokens = tokenizer(prompt, return_tensors="pt")
            # enable_kv_cache=False to skip the static KV-cache pre-allocation; this is a
            # sanity-check generation (32 tokens) and skipping the cache keeps memory headroom.
            generated_ids = megatron_generate(
                model, tokens.input_ids.cuda(), osl=32, enable_kv_cache=False
            )
            generated_texts = tokenizer.batch_decode(generated_ids)
            print_rank_0("{}".format(generated_texts))
            if all_references[idx] is not None:
                assert all_references[idx] == generated_texts[0], all_references[idx]

    def _hf_dataset_forward_loop_func(model):
        if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # WAR for modelopt <= 0.44: pack calibration samples into uniform-length sequences
        # instead of using `get_dataset_dataloader` (which tokenizes each sample independently
        # with truncation + padding, discarding long-document context and feeding padding tokens
        # to the importance estimator). Each calibration sample here is a contiguous slice of
        # the concatenated token stream, matching what Megatron-Bridge's calibration loop does.
        # TODO: revert to `get_dataset_dataloader(..., pack=True, ...)` once modelopt 0.45
        # ships and Megatron-LM bumps the pin (see Model-Optimizer PR #1501).
        seq_len = args.calib_max_sequence_length
        # Pull extra raw samples so we have enough tokens to fill `calib_size` packed chunks.
        # cnn_dailymail articles are ~700-1000 tokens on average; 2x is a safe oversample.
        samples = get_dataset_samples(
            args.calib_dataset, num_samples=args.calib_size * 2
        )
        sep_id = tokenizer.eos_token_id
        token_stream: list[int] = []
        for s in samples:
            token_stream.extend(tokenizer.encode(s, add_special_tokens=False))
            token_stream.append(sep_id)
            if len(token_stream) >= args.calib_size * seq_len:
                break

        n_chunks = min(args.calib_size, len(token_stream) // seq_len)
        print_rank_0(
            f"Calibration packing: {len(samples)} raw samples -> {len(token_stream)} tokens "
            f"-> {n_chunks} chunks of {seq_len} tokens."
        )
        for i in tqdm(range(n_chunks), disable=torch.distributed.get_rank()):
            chunk = token_stream[i * seq_len : (i + 1) * seq_len]
            input_ids = torch.tensor([chunk], dtype=torch.long, device="cuda")
            megatron_prefill(model, input_ids, skip_return_logits=True)

    print_rank_0(f"Pruning model with export_config: {args.prune_export_config}")
    config = {"forward_loop": _hf_dataset_forward_loop_func}
    if args.prune_intermediate_ckpt is not None:
        config["checkpoint"] = args.prune_intermediate_ckpt
    mtp.prune(
        unwrapped_model,
        mode="mcore_minitron",
        constraints={"export_config": args.prune_export_config},
        dummy_input=None,  # Not used
        config=config,
    )
    # Remove unnecessary modelopt_state since ckpt is homogeneous
    if mto.ModeloptStateManager.has_state_for_mode_type("prune", model=unwrapped_model):
        mto.ModeloptStateManager.remove_state(unwrapped_model)

    print_rank_0(f"Pruned Model:\n {unwrapped_model}")
    print_rank_0(f"Pruned Model Params: {get_params(unwrapped_model) / 1e9:.2f}B")

    _custom_prompt_forward_loop_func(unwrapped_model)

    if args.save is not None:
        save_checkpoint(1, model, None, None, 0)

    print_rank_0("Done")
