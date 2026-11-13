# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Native Pixtral integration using the existing public ViT implementation."""

from contextlib import nullcontext
from copy import deepcopy

import torch

from examples.mimo.model_providers.radio_encoder import (
    _base_config,
    _disable_gtp,
    _dtype,
    _make_dense_non_hybrid,
)
from examples.mimo.vision import resolve_encoder_spec, vision_initialization_checkpoint
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.models.vision.vit_model import ViTModel
from megatron.core.transformer.spec_utils import ModuleSpec


def load_native_vision_export(model, directory, process_group):
    """Load a complete BF16 architecture export into existing TP1 parameters.

    The export must contain exactly ``model.vision_model.`` followed by every
    named parameter. Transformer Engine's nonparameter runtime state is retained.
    The explicit encoder group prevents collectives involving language ranks.
    Call before DDP and optimizer construction, after parameter materialization.
    """
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    if process_group is None or torch.distributed.get_world_size(process_group) != 1:
        raise ValueError(
            "native vision export initialization requires an explicit encoder TP1 group"
        )
    parameters = dict(model.named_parameters())
    if any(value.is_meta for value in parameters.values()):
        raise ValueError("native vision export initialization requires materialized parameters")
    metadata = FileSystemReader(str(directory)).read_metadata().state_dict_metadata
    prefix = "model.vision_model."
    if set(metadata) != {prefix + name for name in parameters}:
        raise ValueError("vision export must contain exactly every native encoder parameter")
    slots = {}
    for name, parameter in parameters.items():
        entry = metadata[prefix + name]
        if (
            not isinstance(entry, TensorStorageMetadata)
            or tuple(entry.size) != tuple(parameter.shape)
            or entry.properties.dtype != torch.bfloat16
        ):
            raise ValueError(f"vision export shape/dtype disagrees with encoder parameter {name}")
        slots[prefix + name] = torch.empty(
            tuple(parameter.shape), dtype=torch.bfloat16, device="cpu"
        )
    load(slots, checkpoint_id=str(directory), process_group=process_group)
    state = model.state_dict()
    identities = {name: id(parameter) for name, parameter in parameters.items()}
    for name in parameters:
        value = slots[prefix + name]
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite vision export parameter: {name}")
        state[name] = value
    model.load_state_dict(state, strict=True)
    for name, parameter in model.named_parameters():
        if id(parameter) != identities[name] or not torch.equal(
            parameter.detach().cpu(), slots[prefix + name].to(parameter.dtype)
        ):
            raise RuntimeError(f"vision export parameter was not loaded exactly: {name}")
    print(
        f"[pixtral-init] loaded and verified all {len(parameters)} encoder parameters", flush=True
    )


class MimoViTModel(ViTModel):
    """Adapt the MIMO input name and validate native packed-image geometry."""

    def __init__(
        self,
        *,
        encoder_spec,
        force_eval_mode=False,
        pretrained_checkpoint=None,
        pg_collection=None,
        **kwargs,
    ):
        super().__init__(pg_collection=pg_collection, **kwargs)
        self.encoder_spec = encoder_spec
        self.force_eval_mode = force_eval_mode
        if pg_collection is not None:
            self.tp_group = pg_collection.tp
        projection = self.patch_embed.proj
        expected = (encoder_spec.patch_dim, encoder_spec.patch_dim)
        if projection.kernel_size != expected or projection.stride != expected:
            raise ValueError("native ViT patch projection disagrees with encoder geometry")
        if self.spatial_merge_size != encoder_spec.spatial_merge_size:
            raise ValueError("native ViT merger disagrees with encoder geometry")
        if pretrained_checkpoint is not None:
            load_native_vision_export(
                self, pretrained_checkpoint, getattr(pg_collection, "tp", None)
            )

    def forward(self, x, imgs_sizes=None, packed_seq_params=None):
        if imgs_sizes is None or x.ndim != 3:
            raise ValueError("MIMO native Pixtral requires packed images and CPU imgs_sizes")
        if torch.is_tensor(imgs_sizes):
            if imgs_sizes.is_cuda:
                raise ValueError("imgs_sizes must remain on CPU")
            sizes = imgs_sizes.tolist()
        else:
            sizes = imgs_sizes
        spec = self.encoder_spec
        if x.shape[0] != 1 or x.shape[-1] != 3 * spec.patch_dim**2:
            raise ValueError("native Pixtral requires packed RGB patches of width 3 * patch_dim^2")
        if sum((h // spec.patch_dim) * (w // spec.patch_dim) for h, w in sizes) != x.shape[1]:
            raise ValueError("native Pixtral image sizes disagree with packed patch count")
        outputs = 0
        for height, width in sizes:
            if height > spec.default_img_h or width > spec.default_img_w:
                raise ValueError("native Pixtral image exceeds its configured per-axis bounds")
            outputs += spec.num_spatial_embeddings(height, width)
        if self.force_eval_mode:
            self.eval()
        with torch.no_grad() if self.force_eval_mode else nullcontext():
            result = super().forward(x, imgs_sizes=imgs_sizes, packed_seq_params=packed_seq_params)
        if result.shape != (1, outputs, self.out_hidden_size):
            raise ValueError("native Pixtral output disagrees with its planned token count/width")
        return result


def native_vision_config(args, tp_size, pp_size):
    """Build the selected native tower from public registry metadata."""
    spec = resolve_encoder_spec(args)
    config = deepcopy(_base_config(args))
    _make_dense_non_hybrid(config)
    spec.apply_to_config(config)
    config.qk_layernorm = False
    config.mtp_num_layers = None
    config.fp8 = None
    config.fp4 = None
    config.bf16, config.params_dtype = _dtype(args)
    config.pipeline_dtype = config.params_dtype
    config.tensor_model_parallel_size = tp_size
    config.pipeline_model_parallel_size = pp_size
    config.context_parallel_size = 1
    config.sequence_parallel = False
    _disable_gtp(config)
    if pp_size != 1 or config.num_attention_heads % tp_size:
        raise ValueError("native Pixtral requires encoder PP1 and heads divisible by encoder TP")
    return config


def native_vision_encoder_spec(args, config, pg_collection):
    """Reuse native ViT math and keep strict warm-start separate from full resume."""
    spec = resolve_encoder_spec(args)
    checkpoint = vision_initialization_checkpoint(args)
    return ModuleSpec(
        module=MimoViTModel,
        params={
            "encoder_spec": spec,
            "transformer_config": config,
            "transformer_layer_spec": get_vit_layer_with_transformer_engine_spec(),
            "patch_dim": spec.patch_dim,
            "img_h": spec.default_img_h,
            "img_w": spec.default_img_w,
            "add_class_token": False,
            "class_token_len": 0,
            "ln_pre": True,
            "pos_emb_type": "rope2d",
            "rope_theta": 10000.0,
            "use_merger": spec.has_native_spatial_merge,
            "spatial_merge_size": spec.spatial_merge_size,
            "force_eval_mode": bool(getattr(args, "freeze_vit", False)),
            "pretrained_checkpoint": checkpoint,
            "pg_collection": pg_collection,
        },
    )
