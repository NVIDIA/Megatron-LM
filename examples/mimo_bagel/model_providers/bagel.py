# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Model provider for a LLaVA-style Vision-Language Model.

This provider assembles a MIMO model that consists of:
• Vicuna-7B language model (Llama-based) built with Transformer-Engine GPT blocks.
• CLIP ViT-L/14 visual encoder (336 px) that produces image patch embeddings.
• A 2-layer MLP projector that maps vision embeddings into Vicuna hidden size.
"""

import copy
import json
import os
from typing import Dict, Optional

import torch
from configs.bagel_configs import (
    get_bagel_projection_config,
    get_bagel_projection_layer_spec,
    get_bagel_language_layer_spec,
    get_bagel_language_model_config,
    get_bagel_language_model_config_qwen3_30b,
)

from megatron.core.process_groups_config import ProcessGroupCollection

from examples.mimo_bagel.diffusion.diffusion_modality_submodule import DiffusionModalitySubmodules
from examples.mimo_bagel.diffusion.embeddings import TimestepEmbedder, PositionEmbedding

from examples.mimo_bagel.vision.hf_bagel_vision_encoder import HFBagelVisionEncoderWrapper
from megatron.core.models.bagel.hf_bagel_llm import BagelLLMHuggingFaceModel
from megatron.core.models.bagel.mcore_bagel_llm import BagelMCoreModel
from megatron.core.models.bagel.bagel_mimo import BagelMimoModel
from examples.mimo.utils.logging import print_mimo_structure
from examples.mimo_bagel.utils.model_helpers import load_submodule_ckpt, load_submodule_ckpt_for_mot
from megatron.core.models.mimo import MimoModelConfig
from examples.mimo_bagel.vision.vision_submodule import BagelVisionSubmodule
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec

from bagel.modeling.bagel import Qwen2Config, SiglipVisionConfig, BagelConfig

from examples.mimo_bagel.diffusion.diffusion_wrapper import build_diffusion_wrapper


def _load_siglip_vision_config(vit_path: str) -> SiglipVisionConfig:
    config_path = os.path.join(vit_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if isinstance(config_dict.get("vit_config"), dict):
            return SiglipVisionConfig.from_dict(config_dict["vit_config"])
    return SiglipVisionConfig.from_pretrained(vit_path)


def _register_te_groups_for_subtree(
    subtree: torch.nn.Module,
    pg_collection: ProcessGroupCollection,
    cp_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    """Register TP/CP groups on every TE-aware module in *subtree*.

    Walks ``subtree.modules()`` (skipping the root) and calls
    ``set_tensor_parallel_group`` / ``set_context_parallel_group`` where the
    child module exposes them. Modules without those setters (HF SigLIP,
    HF VAE, ``torch.nn.Linear``) are silently skipped — that's how vision /
    diffusion stay replicated by default.

    Mirrors gpt_builder.py:337-354 from Megatron-Bridge but parameterised
    over a single subtree so callers can register *different* pg_collections
    on *different* model subtrees (vision vs LLM vs diffusion).
    """
    if pg_collection.tp is not None and torch.distributed.get_world_size(pg_collection.tp) > 1:
        for index, child in enumerate(subtree.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_tensor_parallel_group"):
                child.set_tensor_parallel_group(pg_collection.tp)

    if pg_collection.cp is not None and torch.distributed.get_world_size(pg_collection.cp) > 1:
        if cp_stream is None:
            cp_stream = torch.cuda.Stream()
        cp_global_ranks = torch.distributed.get_process_group_ranks(pg_collection.cp)
        for index, child in enumerate(subtree.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(pg_collection.cp, cp_global_ranks, cp_stream)


def _register_te_parallel_groups(
    model: torch.nn.Module,
    pg_collection: ProcessGroupCollection,
    submodule_pg_collections: Optional[Dict[str, ProcessGroupCollection]] = None,
) -> None:
    """Register TP/CP groups across the BagelMimoModel tree, optionally with
    per-submodule overrides.

    By default (``submodule_pg_collections`` is None or missing entries) every
    subtree uses the top-level ``pg_collection`` — preserves Phase B/C/D
    behaviour exactly. When ``submodule_pg_collections`` provides entries
    keyed by ``"language"`` / modality name (matching MR #2117's convention),
    each subtree is registered with its own pg_collection — this is the
    forward-compatible API that lets vision/diffusion stay TP=1 while the
    LLM is TP>1, etc.
    """
    submodule_pg_collections = submodule_pg_collections or {}
    cp_stream = torch.cuda.Stream() if torch.distributed.get_world_size(pg_collection.cp) > 1 else None

    # Language model subtree.
    lm = getattr(model, "language_model", None)
    if lm is not None:
        lm_pg = submodule_pg_collections.get("language", pg_collection)
        _register_te_groups_for_subtree(lm, lm_pg, cp_stream)

    # Each modality subtree.
    modality_submodules = getattr(model, "modality_submodules", None)
    if modality_submodules is not None:
        for modality_name, submodule in modality_submodules.items():
            mod_pg = submodule_pg_collections.get(modality_name, pg_collection)
            _register_te_groups_for_subtree(submodule, mod_pg, cp_stream)


def _inject_pg_collection_into_modality_spec(
    spec: ModuleSpec,
    pg_collection: ProcessGroupCollection,
) -> ModuleSpec:
    """Deep-copy *spec* and inject ``pg_collection`` into the top-level
    submodule's params.

    Mirrors the pattern from Megatron-Bridge MR #2117
    (``models/megatron_mimo/megatron_mimo_provider.py:_inject_pg_collection_into_modality_spec``):
    the upstream ``MimoModel`` already supports per-submodule pg_collection
    via ``ModuleSpec.params`` + ``build_module``; the provider just needs to
    inject the right pg_collection into the right spec.

    The current Bagel modality submodules (`BagelVisionSubmodule`,
    `DiffusionModalitySubmodules`) accept ``**kwargs`` and silently absorb
    ``pg_collection``; the inner encoders/projections (HF SigLIP,
    ``nn.Linear``, ``MultimodalProjector``) likewise don't require it. So
    today this injection is a no-op behaviour-wise — but it's the API
    surface that lets future submodule classes consume per-module
    pg_collection without provider changes, matching MR #2117's contract.
    """
    spec = copy.deepcopy(spec)
    if spec.params is None:
        spec.params = {}
    spec.params["pg_collection"] = pg_collection
    return spec


def model_provider_bagel(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    is_video_input: bool = False,
    freeze_vit: bool = False,
    freeze_llm: bool = False,
    model_path: str = None,
    llm_path: str = None,
    vit_path: str = None,
    language_use_mcore: bool = False,  # Whether to use Megatron Core GPTModel
    decoder_layer_module: str = "Qwen2DecoderLayer",
    use_moe_mlp: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
    vp_stage: Optional[int] = None,
):
    """
    Build a LLaVA-style Vision-Language MIMO model composed of:
    • Qwen2 language model.
    • Bagel vision encoder.
    • 2-layer MLP vision→language projector.

    Args:
        pre_process: Whether to pre-process the model
        post_process: Whether to post-process the model
        add_encoder: Whether to add an encoder (not used)
        add_decoder: Whether to add a decoder (not used)
        image_special_token_id: Special token ID for images
        is_video_input: Whether input is video
        freeze_vit: Whether to freeze ViT parameters
        freeze_llm: Whether to freeze LLM parameters
        model_path: Path to model checkpoints
        language_use_mcore: If True, use Megatron Core BagelMCoreModel;
                           If False, use HuggingFace BagelLLMHuggingFaceModel
    """
    # NOTE: Pipeline parallelism for the encoder/decoder is not yet supported in this
    # MIMO path, therefore *add_encoder* and *add_decoder* are currently ignored.
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    finetune_from_hf = False
    if finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    else:
        print("llm_path", llm_path, "vit_path", vit_path)
        llm_config = Qwen2Config.from_pretrained(llm_path)
        vit_config = _load_siglip_vision_config(vit_path)
    llm_config.layer_module = decoder_layer_module
    # vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 - 2
    vit_config.rope = False
    bagel_config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=None,
    )

    # Language — dispatch to the right config builder based on model_type.
    # Qwen3 MoE (model_type="qwen3_moe") has a dedicated builder because it differs
    # from Qwen2 in: rope_theta, explicit head_dim, always-on MoE, no QKV bias,
    # and a much larger max_position_embeddings.
    _llm_model_type = getattr(llm_config, 'model_type', 'qwen2')
    if _llm_model_type == 'qwen3_moe':
        language_config = get_bagel_language_model_config_qwen3_30b(hf_config=llm_config)
    else:
        language_config = get_bagel_language_model_config(hf_config=llm_config, use_moe_mlp=use_moe_mlp)
    print("language_config.hidden_size", language_config.hidden_size)

    # Vision→language projection MLP – hidden size follows Qwen2
    projection_config = get_bagel_projection_config(
        hidden_size=language_config.hidden_size,
        ffn_hidden_size=language_config.hidden_size,
    )


    # Sync precision flags from global args (if we're running under Megatron training loop)
    dtype = torch.float32
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if getattr(_args, "bf16", False):
            language_config.bf16 = True
            projection_config.bf16 = True
            dtype = torch.bfloat16
        if getattr(_args, "fp16", False):
            language_config.fp16 = True
            projection_config.fp16 = True
            dtype = torch.float16
        # PP recv-buffer dtype must match the activation dtype shipped between
        # stages. Required by p2p_communication.py:327 when pp_size > 1.
        language_config.pipeline_dtype = dtype
        projection_config.pipeline_dtype = dtype
        language_config.params_dtype = dtype
        projection_config.params_dtype = dtype
        # MoT compact representation has [Lund+Lgen, 1, H] shape which varies
        # per microbatch — PP send/recv must negotiate tensor shapes at run
        # time instead of using args.seq_length/CP.
        language_config.variable_seq_lengths = True
        projection_config.variable_seq_lengths = True
        # VP interleaved schedule (schedules.py:1126-1134) requires
        # microbatch_group_size_per_vp_stage in [PP, num_microbatches].
        # The default is set in __post_init__ to pipeline_model_parallel_size
        # (model_parallel_config.py:447-448), but our TransformerConfig was
        # built without pipeline_model_parallel_size, so post_init left
        # microbatch_group_size_per_vp_stage at 1. Force-set both here.
        # Also propagate virtual_pipeline_model_parallel_size so GPTModel
        # partitions layers correctly across VP chunks — without it, each
        # chunk gets the full PP-stage's layer count and the activation
        # shape between chunks (and across PP stages) doesn't match what
        # the receiver expects, surfacing as `ncclInternalError: Internal
        # check failed` in the shape exchange.
        _pp = getattr(_args, 'pipeline_model_parallel_size', 1)
        _vp = getattr(_args, 'virtual_pipeline_model_parallel_size', None)
        language_config.pipeline_model_parallel_size = _pp
        projection_config.pipeline_model_parallel_size = _pp
        language_config.microbatch_group_size_per_vp_stage = _pp
        projection_config.microbatch_group_size_per_vp_stage = _pp
        if _vp is not None:
            language_config.virtual_pipeline_model_parallel_size = _vp
            projection_config.virtual_pipeline_model_parallel_size = _vp
        # PP=2 (non-interleaved) hangs with batched P2P + variable shapes;
        # _p2p_ops (non-batched) has the pp_size==2 WORLD-group fallback
        # that the batched path lacks (p2p_communication.py:65-76). For the
        # interleaved (VP) schedule, however, batched P2P works fine and the
        # non-batched path triggers a NCCL internal error in
        # _end_coalescing. So only force-disable batched at PP>1 + VP=1.
        if _pp > 1 and (_vp is None or _vp == 1):
            language_config.batch_p2p_comm = False
            projection_config.batch_p2p_comm = False
        # Propagate activation-recompute settings so the MoT block honours them.
        for _attr in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
            _val = getattr(_args, _attr, None)
            if _val is not None:
                setattr(language_config, _attr, _val)
        # MoT und/gen MLP stream-overlap toggle. Read by MoTTransformerLayer via
        # getattr(config, 'mot_stream_overlap', False) — no field on TransformerConfig.
        language_config.mot_stream_overlap = getattr(_args, 'mot_stream_overlap', False)
        # Propagate user-tunable MoE CLI flags into language_config.
        # Architectural fields (num_moe_experts, moe_ffn_hidden_size,
        # moe_router_topk, moe_router_pre_softmax, moe_aux_loss_coeff) are
        # pinned by bagel_configs.py and not in this list. Each entry is
        # overridden only when args has a meaningful value: None / False /
        # store_true-default values are skipped so the TransformerConfig
        # default stays in effect when the user did not set the flag.
        # Knobs pinned by Qwen3 architecture / training recipe (in
        # bagel_configs.py) are intentionally NOT in this list, so CLI args
        # cannot accidentally override them:
        #   moe_router_load_balancing_type, moe_router_enable_expert_bias,
        #   moe_router_bias_update_rate, moe_expert_capacity_factor,
        #   moe_pad_expert_input_to_capacity, moe_token_drop_policy,
        #   moe_shared_expert_overlap.
        _MOE_TUNABLE_ATTRS = (
            "moe_grouped_gemm",
            "moe_use_legacy_grouped_gemm",
            "moe_token_dispatcher_type",
            "moe_flex_dispatcher_backend",
            "moe_deepep_num_sms",
            "moe_hybridep_num_sms",
            "moe_permute_fusion",
            "moe_router_fusion",
            "moe_router_dtype",
            "moe_router_padding_for_fp8",
            "moe_z_loss_coeff",
            "moe_input_jitter_eps",
            "moe_layer_recompute",
        )
        for _attr in _MOE_TUNABLE_ATTRS:
            if not hasattr(_args, _attr):
                continue
            _val = getattr(_args, _attr)
            # Skip Nones and store_true defaults — they mean "user did not set"
            if _val is None or _val is False:
                continue
            setattr(language_config, _attr, _val)
    except (ModuleNotFoundError, AssertionError):
        pass

    # Override any outer torch.device('meta') context (active when
    # --init-model-with-meta-device is passed to training.py) with 'cuda' for
    # all non-TE model construction.  TE LLM modules are unaffected because
    # _get_extra_te_kwargs reads cfg.init_model_with_meta_device=True and
    # passes device='meta' directly to TE constructors regardless of the
    # torch default-device context.  Everything else — VAE, vision encoder,
    # position embeddings, diffusion components — gets real CUDA tensors and
    # does not need reset_parameters().
    # The context must cover both build_diffusion_wrapper() and
    # BagelMimoModel() since both are called inside the meta context.
    _cuda_ctx = torch.device('cuda')
    _cuda_ctx.__enter__()

    #we build diffusion wrapper here for bagel model needs latent channels.
    #but actually we don't want vae on all devices, so we will remove diffusion wrapper on the devices which
    #has no visual data.
    _args.diffusion_wrapper = build_diffusion_wrapper()
    latent_channels = _args.diffusion_wrapper.vae_params.z_channels
    if torch.distributed.get_rank(pg_collection.tp) != 0:
        _args.diffusion_wrapper = None

    ### Diffusion modality submodule
    timestep_embedder = ModuleSpec(
        module=TimestepEmbedder,
        params={
            "hidden_size": language_config.hidden_size,
        },
    )
    pos_embedder = ModuleSpec(
        module=PositionEmbedding,
        params={
            "max_num_patch_per_side": _args.max_latent_size, 
            "hidden_size": language_config.hidden_size,
        }
        
    )
    
    #projector for diffusion
    vae2llm = ModuleSpec(
        module=torch.nn.Linear,
        params={
            "in_features": _args.latent_patch_size**2 * latent_channels,
            "out_features": language_config.hidden_size,
            "dtype": dtype,
        }
    )

    llm2vae = ModuleSpec(
        module=torch.nn.Linear,
        params={
            "in_features": language_config.hidden_size,
            "out_features": _args.latent_patch_size**2 * latent_channels,
            "dtype": dtype,
        }
    )

    # Create modality config for vision
    diffusion_submodule_spec = ModuleSpec(
        module=DiffusionModalitySubmodules,
        params={'timestep_shift': _args.timestep_shift, 'dtype': dtype},
        submodules={
            "encoders": {
                         "timestep": timestep_embedder, 
                         "latent_position_ids": pos_embedder
                         },
            "input_projections": [vae2llm],
            "output_projections": [llm2vae],
        },
    )

    # HF encoder
    vision_encoder = ModuleSpec(
        module=HFBagelVisionEncoderWrapper,
        params={
            "bagel_config": bagel_config,
            "vit_path": vit_path,
            "dtype": dtype,
            "recompute_vit": getattr(_args, "recompute_vit", False),
        },
    )

    # Create projection config for vision to language
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": get_bagel_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": vit_config.hidden_size,  # vision hidden size
        },
    )

    # Create modality config for vision
    vision_submodule_spec = ModuleSpec(
        module=BagelVisionSubmodule,
        params={},
        submodules={
            "encoders": {"vision_encoder": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    # Create language model config (Qwen2-based)
    use_mo = "Mo" in llm_config.layer_module if hasattr(llm_config, 'layer_module') else False
    if language_use_mcore:
        print(f"use_flex_attention={_args.use_flex_attention}")
        if use_mo:
            from megatron.core.models.bagel.transformer_mot_block import get_mot_layer_spec
            transformer_layer_spec = get_mot_layer_spec(num_experts=language_config.num_moe_experts, 
                                                        moe_grouped_gemm=language_config.moe_grouped_gemm, 
                                                        use_flex_attention=_args.use_flex_attention, 
                                                        qk_layernorm=True
                                                        )
        else:
            transformer_layer_spec = get_bagel_language_layer_spec(num_experts=language_config.num_moe_experts, 
                                                                   moe_grouped_gemm=language_config.moe_grouped_gemm, 
                                                                   use_flex_attention=_args.use_flex_attention
                                                                   )


        # Use Megatron Core based BagelMCoreModel
        language_model_spec = ModuleSpec(
            module=BagelMCoreModel,
            params={
                "config": language_config,
                "transformer_layer_spec": transformer_layer_spec,
                "vocab_size": llm_config.vocab_size,
                "max_sequence_length": 32768,
                "pre_process": pre_process,
                "post_process": post_process,
                "position_embedding_type": "rope",
                "rotary_base": getattr(llm_config, 'rope_theta', 1000000),
                "llm_config": llm_config,  # Bagel-specific config
                "use_flex_attention": _args.use_flex_attention,  # Pass flex attention flag
                "pg_collection": pg_collection,
                "vp_stage": vp_stage,
            },
        )
    else:
        # Use HuggingFace based BagelLLMHuggingFaceModel
        language_model_spec = ModuleSpec(
            module=BagelLLMHuggingFaceModel,
            params={"config": language_config, "llm_config": llm_config, "llm_path": llm_path},
        )

    # Per-submodule pg_collection injection (mirrors MR #2117).
    # ``submodule_pg_collections`` keys match MIMO's submodule names: the
    # language model is keyed under "language" (Megatron-Bridge's
    # MIMO_LANGUAGE_MODULE_KEY), and each modality submodule under its key
    # in ``modality_submodules_spec``. Today every entry is the top-level
    # ``pg_collection`` so behaviour is unchanged. When per-submodule
    # parallelism is added (heterogeneous TP/CP), the provider populates
    # this dict with distinct pg_collections per module and the rest of the
    # plumbing — spec injection, post-init TE registration — picks the
    # right group per subtree.
    submodule_pg_collections: Dict[str, ProcessGroupCollection] = {
        "language": pg_collection,
        "images": pg_collection,
        "diffusion": pg_collection,
    }
    vision_submodule_spec = _inject_pg_collection_into_modality_spec(
        vision_submodule_spec, submodule_pg_collections["images"]
    )
    diffusion_submodule_spec = _inject_pg_collection_into_modality_spec(
        diffusion_submodule_spec, submodule_pg_collections["diffusion"]
    )

    # Create MIMO model config
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec,
                                  "diffusion": diffusion_submodule_spec},
        special_token_ids={"images": image_special_token_id,
                           }
    )

    # Create MIMO model (still inside the _cuda_ctx started above).
    mimo_model = BagelMimoModel(
        mimo_model_config,
        pg_collection=pg_collection,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
    _cuda_ctx.__exit__(None, None, None)

    _register_te_parallel_groups(mimo_model, pg_collection,
                                 submodule_pg_collections=submodule_pg_collections)

    print("*" * 100)
    print(f"Using {'Megatron Core BagelMCoreModel' if language_use_mcore else 'HuggingFace BagelLLMHuggingFaceModel'}")
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # load the checkpoint
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if _args.language_model_checkpoint is not None:
            print(f"[model_provider_bagel] Attempting to load checkpoint from {_args.language_model_checkpoint}")
            try:
                if use_mo:
                    # For MoT models, use the special loader that handles _gen parameters
                    load_submodule_ckpt_for_mot(
                        mimo_model.language_model,
                        _args.language_model_checkpoint,
                        init_gen_from_und=True
                    )
                    print(f"Successfully loaded checkpoint (MoT mode) from {_args.language_model_checkpoint}")
                else:
                    load_submodule_ckpt(mimo_model.language_model, _args.language_model_checkpoint)
                    print(f"Successfully loaded checkpoint from {_args.language_model_checkpoint}")
            except Exception as e:
                print(f"[model_provider_bagel] WARNING: Failed to load checkpoint: {e}")
                print("[model_provider_bagel] Continuing with randomly initialized weights.")
    except (ModuleNotFoundError, AssertionError):
        pass

    # Build a BagelMFUTracker so the training loop can report MFU.  Configs are
    # all in scope here: language_config (TransformerConfig), vit_config
    # (SiglipVisionConfig), and llm_config.vocab_size.  The tracker itself is
    # cheap — accumulators only; heavy work happens inside compute_and_reset.
    try:
        from examples.mimo_bagel.utils.mfu import BagelMFUTracker, detect_peak_tflops

        _world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        _args.mfu_tracker = BagelMFUTracker(
            language_config=language_config,
            vit_config=vit_config,
            vocab_size=llm_config.vocab_size,
            world_size=_world_size,
            peak_tflops_per_gpu=detect_peak_tflops(),
            include_diffusion=True,
            latent_patch_size=_args.latent_patch_size,
            latent_channels=latent_channels,
            log_interval=getattr(_args, 'log_interval', 10) or 10,
        )
    except Exception as e:
        print(f"[model_provider_bagel] WARNING: Failed to build MFU tracker: {e}")
        _args.mfu_tracker = None

    # TODO: ykarnati make these configurable and have an API to freeze/unfreeze
    # freeze vision encoder and LLM parameters
    modules_to_freeze = []
    if freeze_vit:
        modules_to_freeze.append(mimo_model.modality_submodules.images.encoders.vision_encoder)
    if freeze_llm:
        modules_to_freeze.append(mimo_model.language_model)
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False

    return mimo_model
