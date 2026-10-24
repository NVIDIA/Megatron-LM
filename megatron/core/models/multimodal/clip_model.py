# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
import logging
from collections import namedtuple
from functools import partial
from typing import List, Optional
import numpy as np

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.gpt import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel, get_num_image_embeddings
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.utils import log_single_rank

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.utils import is_te_min_version

    HAVE_TE = True
except:
    HAVE_TE = False


class ClipModel(MegatronModule):
    """CLIP dual-encoder model for image-text contrastive pretraining.

    Vision encoder: CLIPViTModel. Text encoder: GPTModel.
    Both encoders are projected to a shared embedding space via affine projections
    and trained with symmetric InfoNCE (contrastive) loss.

    Only pipeline_model_parallel_size == 1 is supported.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Language model spec.
        language_vocab_size (int): Language model vocabulary size.
        language_max_sequence_length (int): Language model maximum sequence length.
        language_projection_config (TransformerConfig): Language projection config.
        language_projection_layer_spec (ModuleSpec): Language projection spec.
        language_projection_type (str): Type of the language projection. Default: affine.
        vision_transformer_config (TransformerConfig): Transformer config for the vision model.
        vision_transformer_layer_spec (ModuleSpec): Vision model spec.
        drop_vision_class_token (bool): Drop vision class token before the projection.
        vision_projection_config (TransformerConfig): Vision projection config.
        vision_projection_layer_spec (ModuleSpec): Vision projection spec.
        vision_projection_type (str): Type of the vision projection. Default: affine.
        allow_missing_vision_projection_checkpoint (bool): Allow vision projection weights to be
            missing when loading a checkpoint. Default False.
        parallel_output (bool): Keep outputs split across tensor parallel ranks.
        share_embeddings_and_output_weights (bool): Input embedding and output layer share weights.
        language_position_embedding_type (str): Language model position embedding type.
        language_rotary_percent (float): RoPE percent. Defaults to 1.0.
        pre_process (bool): Include embedding layer (used with pipeline parallel).
        post_process (bool): Include output layer (used with pipeline parallel).
        img_h (int): Input image height.
        img_w (int): Input image width.
        patch_dim (int): Image patch size.
        language_rotary_base (int): RoPE base.
        language_rope_scaling (bool): Toggle RoPE scaling.
        language_rope_scaling_factor (float): RoPE scaling factor. Defaults to 8.
        pg_collection (ProcessGroupCollection): Model communication process groups.
        vp_stage (int): Virtual pipeline stage.
        local_loss (bool): Compute cross-entropy only over local DP rows. Default False.
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        language_projection_config: TransformerConfig,
        language_projection_layer_spec: ModuleSpec,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        drop_vision_class_token: bool,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        language_projection_type: str = "affine",
        vision_projection_type: str = "affine",
        allow_missing_vision_projection_checkpoint: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = 'learned_absolute',
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = False,
        img_h: int = 336,
        img_w: int = 336,
        patch_dim: int = 14,
        language_rotary_base: int = 10000,
        language_rope_scaling: bool = False,
        language_rope_scaling_factor: float = 8.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        local_loss: bool = False,
    ) -> None:
        super().__init__(config=language_transformer_config)

        assert language_transformer_config.pipeline_model_parallel_size == 1, (
            "ClipModel supports pipeline_model_parallel_size == 1 only. "
            "PP > 1 is not implemented."
        )

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "ClipModel is work in progress. Features are missing and methods can change.",
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.vp_stage = vp_stage

        self.vision_model = None
        self.vision_projection = None
        self.language_model = None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.tp_comm_overlap_lm = language_transformer_config.tp_comm_overlap
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        if self.sequence_parallel_lm or self.context_parallel_lm > 1:
            assert isinstance(
                language_transformer_layer_spec.submodules, TransformerLayerSubmodules
            )
            assert isinstance(
                language_transformer_layer_spec.submodules.self_attention.submodules,
                SelfAttentionSubmodules,
            )
            attn_submodules = (
                language_transformer_layer_spec.submodules.self_attention.submodules
            )
            assert (
                attn_submodules.core_attention == TEDotProductAttention and HAVE_TE
            ), "Sequence/Context Parallelism is supported only with TE DotProductAttention."
            if self.context_parallel_lm > 1:
                self.cp_group = self.pg_collection.cp
                assert (
                    self.cp_group.size() == self.context_parallel_lm
                ), "CP Group size should match the Language Model CP size"
                assert is_te_min_version(
                    "1.10.0"
                ), "Context Parallelism in ClipModel requires TE v1.10 or higher"
            else:
                self.cp_group = None
        self.tensor_model_parallel_size_lm = language_transformer_config.tensor_model_parallel_size

        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

        self.language_model = GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_vocab_size,
            max_sequence_length=language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type=language_position_embedding_type,
            rotary_percent=language_rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_rotary_base,
            rope_scaling=language_rope_scaling,
            rope_scaling_factor=language_rope_scaling_factor,
            scatter_embedding_sequence_parallel=False,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

        language_projection_input_size = language_transformer_config.hidden_size

        self.language_projection = MultimodalProjector(
            language_projection_config,
            language_projection_layer_spec,
            language_projection_type,
            language_projection_input_size,
            tp_group=self.pg_collection.tp,
        )

        class_token_len = 1
        self._drop_vision_class_token = drop_vision_class_token
        self.vision_model = CLIPViTModel(
            vision_transformer_config,
            vision_transformer_layer_spec,
            img_h=img_h,
            img_w=img_w,
            class_token_len=class_token_len,
            patch_dim=patch_dim,
            model_subtype=vision_transformer_config.vision_model_type,
            add_class_token=True,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

        self.vision_model.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_extra_state
        )

        vision_projection_input_size = vision_transformer_config.hidden_size

        self.vision_projection = MultimodalProjector(
            vision_projection_config,
            vision_projection_layer_spec,
            vision_projection_type,
            vision_projection_input_size,
            tp_group=self.pg_collection.tp,
        )
        if allow_missing_vision_projection_checkpoint:
            vision_projection_param_names = [
                f"vision_projection.{name}"
                for name in self.vision_projection.state_dict().keys()
            ]
            self.vision_projection.register_load_state_dict_post_hook(
                partial(_load_state_dict_hook_ignore_param_names, vision_projection_param_names)
            )

        self.vision_projection.register_load_state_dict_post_hook(
            _load_state_dict_hook_ignore_extra_state
        )

        self.logit_scale = torch.nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=language_transformer_config.params_dtype)
        )

        self.img_seq_len = get_num_image_embeddings(
            img_h,
            img_w,
            patch_dim,
            vision_transformer_config.vision_model_type,
            drop_vision_class_token,
            class_token_len,
            pixel_shuffle=False,
            use_tile_tags=False,
        )

        self._local_loss = local_loss

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_language_projection: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_language_projection (bool): Freeze the language projection module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_language_projection and self.language_projection is not None:
            modules.append(self.language_projection)
        if freeze_vision_projection and self.vision_projection is not None:
            modules.append(self.vision_projection)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def contrastive_loss(self, image_embeddings, language_embeddings, gather_with_grad=True, local_loss=False):
        """Compute contrastive loss between image and language embeddings.

        Two modes:
          local_loss=False (default): each rank gathers all embeddings and computes the full
            [global_batch × global_batch] similarity matrix. Every rank produces the same loss.
          local_loss=True: each rank still gathers all embeddings (so every negative is visible),
            but computes cross-entropy only over its own [local_batch × global_batch] rows.
            Reduces cross-entropy compute by dp_world_size× at the cost of an asymmetric
            gradient graph — DDP averages the gradients across ranks so the result is equivalent.

        Args:
            image_embeddings (torch.Tensor): [local_batch, h_language].
            language_embeddings (torch.Tensor): [local_batch, h_language].
            gather_with_grad (bool): Keep the local rank's own slice in the autograd graph so
                gradients flow back through the gathered tensor. Default True.
            local_loss (bool): Compute cross-entropy only over local rows. Default False.
        Returns:
            torch.Tensor: scalar contrastive loss.
        """
        dp_rank = parallel_state.get_data_parallel_rank()
        dp_world_size = parallel_state.get_data_parallel_world_size()
        dp_group = parallel_state.get_data_parallel_group()

        local_batch_size = image_embeddings.shape[0]

        if dp_world_size > 1:
            all_image_embeddings = [torch.zeros_like(image_embeddings) for _ in range(dp_world_size)]
            torch.distributed.all_gather(all_image_embeddings, image_embeddings, group=dp_group)

            all_language_embeddings = [torch.zeros_like(language_embeddings) for _ in range(dp_world_size)]
            torch.distributed.all_gather(all_language_embeddings, language_embeddings, group=dp_group)

            if gather_with_grad:
                all_image_embeddings[dp_rank] = image_embeddings
                all_language_embeddings[dp_rank] = language_embeddings

            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            all_language_embeddings = torch.cat(all_language_embeddings, dim=0)
        else:
            all_image_embeddings = image_embeddings
            all_language_embeddings = language_embeddings

        logit_scale = self.logit_scale.exp()

        if local_loss and dp_world_size > 1:
            logits_per_image = logit_scale * (image_embeddings @ all_language_embeddings.t())
            logits_per_text  = logit_scale * (language_embeddings @ all_image_embeddings.t())

            offset = dp_rank * local_batch_size
            labels = torch.arange(offset, offset + local_batch_size,
                                  device=image_embeddings.device, dtype=torch.long)
        else:
            logits_per_image = logit_scale * (all_image_embeddings @ all_language_embeddings.t())
            logits_per_text  = logits_per_image.t()

            num_logits = logits_per_image.shape[0]
            labels = torch.arange(num_logits, device=image_embeddings.device, dtype=torch.long)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        return (loss_i + loss_t) / 2

    def set_input_tensor(self, input_tensor):
        """No-op: pipeline parallelism is not supported (PP=1 only)."""
        pass

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the CLIP model.

        Args:
            images (torch.Tensor): Input images [batch, 3, img_h, img_w].
            input_ids (torch.Tensor): Input text token ids [batch, text_seq_len].
            position_ids (torch.Tensor): Input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): Padding mask [batch, text_seq_len] or None.

        Returns:
            torch.Tensor: Scalar contrastive loss.
        """
        batch_size = images.shape[0]

        vision_output = self.vision_model(images)

        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        language_output = self.language_model(
            input_ids,
            position_ids=position_ids,
            attention_mask=None,
        )

        # Extract CLS token from vision (index 0)
        vision_features = vision_output[:, 0, :]

        # GPTModel with post_process=False returns [seq_len, batch, hidden] → transpose
        language_output = language_output.transpose(0, 1)  # [batch, seq_len, hidden]

        # Extract EOS-position feature from text
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=-1) - 1
            language_features = language_output[torch.arange(batch_size, device=input_ids.device), seq_lengths]
        else:
            language_features = language_output[:, -1, :]

        vision_embeddings = self.vision_projection(vision_features)
        language_embeddings = self.language_projection(language_features)

        vision_embeddings = vision_embeddings / vision_embeddings.norm(dim=-1, keepdim=True)
        language_embeddings = language_embeddings / language_embeddings.norm(dim=-1, keepdim=True)

        return self.contrastive_loss(vision_embeddings, language_embeddings,
                                     local_loss=self._local_loss)


def _load_state_dict_hook_ignore_param_names(
    param_names: List[str], module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore missing keys during checkpoint loading.

    Use this if you want to load a checkpoint that contains vision and language model weights
    but not the vision projection weights.

    Args:
        param_names (list str): Parameter names allowed to be missing when calling load_state_dict.
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys.
    """
    for param_name in param_names:
        if param_name in incompatible_keys.missing_keys:
            logging.getLogger(__name__).warning(
                f"{param_name} being removed from incompatible_keys.missing_keys in ClipModel"
            )
            incompatible_keys.missing_keys.remove(param_name)


def _load_state_dict_hook_ignore_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
):
    """Hook to ignore Transformer Engine _extra_state used for FP8.

    Newer TE versions add _extra_state keys to the state dict; older checkpoints may not have
    those keys. They can be ignored when not using FP8.

    Args:
        module (torch.nn.Module): The torch module this hook applies to. Required by the torch API.
        incompatible_keys (namedtuple): Namedtuple with fields missing_keys and unexpected_keys.
    """
    for name, keys in incompatible_keys._asdict().items():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning(
                    f"_extra_state key {key} being removed from {name}"
                )
                keys.remove(key)
