# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.


from typing import Dict, Literal, Optional, Tuple

from megatron.core.transformer.enums import ModelType
import torch
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from einops import rearrange, repeat

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.utils import make_sharded_tensor_for_checkpoint
from megatron.core.models.dit.dit_embeddings import (
    FourierFeatures,
    SinCosPosEmb3D, 
    SDXLTimestepEmbedding, 
    SDXLTimesteps, 
    get_3d_sincos_pos_embed, 
    TimestepLabelEmbedder, 
    get_2d_sincos_pos_embed
)
from megatron.core.models.dit import dit_embeddings
from megatron.core.models.dit.dit_layer_spec import (
    AdaLN,
    get_dit_adaln_block_with_transformer_engine_spec as DiTLayerWithAdaLNspec,
    get_official_dit_adaln_block_with_transformer_engine_spec as OfficialDiTLyaerWithAdaLNspec,
)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    """
    The final layer of video DiT.
    """

    def __init__(self, hidden_size, spatial_patch_size, temporal_patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, spatial_patch_size * spatial_patch_size * temporal_patch_size * out_channels, bias=False
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False))

    def forward(self, x_BT_HW_D, emb_B_D):
        shift_B_D, scale_B_D = self.adaLN_modulation(emb_B_D).chunk(2, dim=1)
        T = x_BT_HW_D.shape[0] // emb_B_D.shape[0]
        shift_BT_D, scale_BT_D = repeat(shift_B_D, "b d -> (b t) d", t=T), repeat(scale_B_D, "b d -> (b t) d", t=T)
        x_BT_HW_D = modulate(self.norm_final(x_BT_HW_D), shift_BT_D, scale_BT_D)
        x_BT_HW_D = self.linear(x_BT_HW_D)
        return x_BT_HW_D

class RMSNorm(nn.Module):
    def __init__(self, channel: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channel))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
class SinCosPosEmb(torch.nn.Module):
    def __init__(
        self,
        model_channels: int,
        len_h: int,
        len_w: int,
        len_t: int,
        is_learnable: bool = False,
        interpolation: str = "crop",
        spatial_interpolation_scale=1.0,
        temporal_interpolation_scale=1.0,
        **kwargs,
    ):
        super().__init__()
        self.len_h = len_h
        self.len_w = len_w
        self.len_t = len_t
        self.interpolation = interpolation
        # h w t
        param = get_3d_sincos_pos_embed(
            model_channels, len_h, len_w, len_t, spatial_interpolation_scale, temporal_interpolation_scale
        )
        
        self.position_embeddings = torch.nn.Embedding(param.shape[0], param.shape[1])
        if not is_learnable:
            self.position_embeddings.weight = torch.nn.Parameter(torch.tensor(param), requires_grad=False)

    def forward(self, pos_ids: torch.Tensor):
        # pos_ids: t h w
        pos_id = pos_ids[..., 1] * self.len_t * self.len_w + pos_ids[..., 2] * self.len_t + pos_ids[..., 0]
        return self.position_embeddings(pos_id)

class DiTCrossAttentionModel(VisionModule):
    """DiT with CrossAttention model.

    Args:
        config (TransformerConfig): transformer config

        transformer_decoder_layer_spec (ModuleSpec): transformer layer customization specs for decoder

        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)

        fp16_lm_cross_entropy (bool, optional): Defaults to False

        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks

        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared. Defaults to False.

        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.

        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.

        seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        additional_timestamp_channels: dict = dict(fps=256, h=256, w=256, org_h=256, org_w=256),
        max_img_h: int = 80,
        max_img_w: int = 80,
        max_frames: int = 34,
        patch_spatial: int = 1,
        patch_temporal: int = 1,
    ):
        super(DiTCrossAttentionModel, self).__init__(config=config)
        # args = get_args()

        self.config: TransformerConfig = config

        self.transformer_decoder_layer_spec = DiTLayerWithAdaLNspec()
        # self.transformer_decoder_layer_spec = getattr(dit_block, args.dit_model_spec)()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False 
        self.concat_padding_mask = True
        self.pos_emb_cls='sincos'

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=False,
            post_layer_norm=False,
        )

        self.t_embedder = torch.nn.Sequential(
            SDXLTimesteps(self.config.hidden_size),
            SDXLTimestepEmbedding(self.config.hidden_size, self.config.hidden_size),
        )

        in_channels = 16
        if self.pre_process:
            in_channels = in_channels + 1 if self.concat_padding_mask else in_channels
            # self.x_embedder = torch.nn.Linear(args.in_channels * args.patch_size **2, self.config.hidden_size)
            # self.x_embedder = torch.nn.Linear(16 * 1 **2, self.config.hidden_size)

            self.x_embedder = dit_embeddings.PatchEmbed(
                spatial_patch_size=1,
                temporal_patch_size=1,
                in_channels=in_channels,
                out_channels=self.config.hidden_size,
                bias=False,
                keep_spatio=True,
                st_conv=True,
            )
            self.pos_embedder = dit_embeddings.SinCosPosEmb3D(
                model_channels=self.config.hidden_size,
                len_h=max_img_h // patch_spatial,
                len_w=max_img_w // patch_spatial,
                len_t=max_frames // patch_temporal,
                max_fps=30,
                min_fps=1,
                is_learnable=False,
                pos_emb_interpolation='crop',
            )
            if parallel_state.get_context_parallel_world_size() > 1:
                cp_group = parallel_state.get_context_parallel_group()
                self.pos_embedder.enable_context_parallel(cp_group)
            # self.pos_embedder = None

        if self.post_process:
            self.final_layer = FinalLayer(
                hidden_size=self.config.hidden_size,
                spatial_patch_size=1,
                temporal_patch_size=1,
                out_channels=16,
            )
            # self.final_layer_adaLN = AdaLN(config=self.config, n_adaln_chunks=2)
            # self.final_layer_linear = torch.nn.Linear(
            #     self.config.hidden_size,
            #     args.patch_size**2 * 1 * args.in_channels,
            # )

            self.logvar = torch.nn.Sequential(
                FourierFeatures(num_channels=128, normalize=True), torch.nn.Linear(128, 1, bias=False)
            )
    #         with torch.no_grad():
    #             self.logvar[0].freqs.copy_(torch.tensor([  7.46875   ,   3.21875   ,   1.71875   ,  -2.328125  ,
    #      4.78125   ,   6.375     ,  -3.703125  ,  -5.21875   ,
    #     -7.53125   ,  -2.65625   ,   2.796875  ,   2.953125  ,
    #     -7.46875   ,   2.078125  ,  -0.24609375,  12.125     ,
    #      1.2578125 ,   2.828125  ,   3.171875  ,   9.8125    ,
    #     -5.71875   ,  -1.3125    ,  -0.28125   ,  -0.23730469,
    #      0.8984375 ,   5.3125    ,   1.40625   ,   0.62890625,
    #     -5.875     ,  14.5       ,  -7.28125   , -13.625     ,
    #     -0.20996094, -12.625     ,   0.16992188,  -4.5       ,
    #      6.75      , -11.        ,   7.3125    ,  -0.41015625,
    #      8.4375    ,   0.09082031,  -1.34375   ,  -4.46875   ,
    #     -6.4375    ,   5.96875   ,   3.125     ,  -1.7578125 ,
    #     -2.5625    ,  -5.90625   ,   7.40625   ,   7.75      ,
    #     -4.96875   ,   1.9765625 ,   1.0390625 ,  -0.671875  ,
    #     -7.6875    ,   4.15625   ,   8.        ,   9.875     ,
    #     -3.40625   ,  -4.09375   ,  -9.3125    ,   3.109375  ,
    #     -8.4375    ,  -4.21875   ,   8.4375    ,   1.71875   ,
    #     -4.5625    ,  -7.53125   ,  -6.3125    , -10.5       ,
    #     -6.59375   ,  -6.59375   ,  -5.375     , -10.625     ,
    #      0.23242188,   3.359375  ,   8.5       ,  11.        ,
    #      6.90625   ,  -0.609375  ,  -3.        ,   0.80078125,
    #     -7.34375   ,  -2.703125  , -10.3125    ,   0.390625  ,
    #      2.015625  ,  -2.28125   ,  -5.4375    ,   6.125     ,
    #     -8.3125    ,   3.078125  ,  12.125     ,   9.625     ,
    #     -6.96875   ,  -1.84375   , -12.375     ,   0.5703125 ,
    #    -10.4375    ,   2.25      ,  -6.34375   ,  -6.3125    ,
    #      4.09375   ,   1.3125    ,   2.46875   ,  -0.24511719,
    #     -3.859375  ,  -4.46875   ,   2.828125  ,   7.375     ,
    #     -3.046875  ,  -7.0625    ,  -3.046875  ,  -1.984375  ,
    #     -3.796875  ,  -9.6875    ,   1.3125    ,   4.        ,
    #     -0.06445312,  -2.5       ,  -5.5       ,  -5.25      ,
    #     -6.125     , -11.625     ,   3.125     ,   4.28125   ], dtype=torch.bfloat16))
    #             self.logvar[0].phases.copy_(torch.tensor([2.71875   , 4.84375   , 1.        , 4.75      , 5.9375    ,
    #    2.78125   , 3.125     , 0.55859375, 1.484375  , 6.25      ,
    #    4.40625   , 0.58984375, 5.15625   , 0.65234375, 4.4375    ,
    #    4.78125   , 2.15625   , 5.375     , 1.84375   , 5.9375    ,
    #    1.515625  , 1.5546875 , 5.8125    , 1.390625  , 6.125     ,
    #    4.78125   , 0.66015625, 3.953125  , 3.625     , 2.578125  ,
    #    0.06225586, 5.4375    , 4.03125   , 2.375     , 2.546875  ,
    #    1.1484375 , 3.734375  , 4.3125    , 4.09375   , 0.77734375,
    #    4.03125   , 0.671875  , 4.09375   , 5.3125    , 2.828125  ,
    #    4.8125    , 4.1875    , 0.7265625 , 5.84375   , 1.9453125 ,
    #    3.28125   , 2.0625    , 3.578125  , 1.984375  , 4.53125   ,
    #    0.84765625, 3.234375  , 0.48046875, 1.5859375 , 2.796875  ,
    #    2.203125  , 0.41796875, 4.        , 5.0625    , 4.21875   ,
    #    5.8125    , 4.8125    , 5.25      , 0.3671875 , 3.203125  ,
    #    2.09375   , 5.625     , 4.96875   , 3.609375  , 3.953125  ,
    #    4.09375   , 4.90625   , 1.625     , 3.953125  , 5.84375   ,
    #    1.28125   , 0.45117188, 3.96875   , 3.515625  , 2.21875   ,
    #    4.53125   , 2.296875  , 1.09375   , 5.34375   , 5.125     ,
    #    4.5       , 5.25      , 0.56640625, 0.6484375 , 4.9375    ,
    #    3.734375  , 4.78125   , 0.8828125 , 2.25      , 1.3671875 ,
    #    6.        , 4.78125   , 3.984375  , 1.4140625 , 1.015625  ,
    #    3.125     , 4.21875   , 5.        , 5.53125   , 6.        ,
    #    1.7734375 , 0.8828125 , 3.265625  , 0.12792969, 2.        ,
    #    3.453125  , 0.55078125, 4.625     , 4.46875   , 5.84375   ,
    #    0.06103516, 4.25      , 1.3203125 , 3.703125  , 6.21875   ,
    #    3.265625  , 5.9375    , 0.9140625 ], dtype=torch.bfloat16))
    #             self.logvar[1].weight.copy_(torch.tensor([[ 4.51707543e-04,  1.07187452e-02, -4.45645675e-02,
    #     -3.78187299e-02, -2.61876127e-03, -3.34550813e-03,
    #      1.65667925e-02,  1.51832937e-03,  1.18261739e-03,
    #     -3.11631616e-02,  3.23928334e-02, -2.78040264e-02,
    #     -1.14706974e-03, -3.98559012e-02,  5.26493937e-02,
    #      8.13660445e-04,  4.62790169e-02, -2.55769808e-02,
    #      9.34263319e-03, -4.94245952e-03,  8.40850361e-03,
    #      3.36295031e-02, -5.31011298e-02, -5.28272949e-02,
    #     -5.03858328e-02,  1.14157330e-02, -4.67478558e-02,
    #      5.28767854e-02, -3.69128468e-03,  2.13332102e-03,
    #      1.63126479e-05, -4.94812429e-03,  5.36172502e-02,
    #     -5.26629994e-03,  5.33446409e-02,  2.85614282e-03,
    #     -1.03186199e-03, -4.08245903e-03,  6.57442783e-04,
    #     -5.28341979e-02,  4.58003953e-03, -5.36197126e-02,
    #      4.53494228e-02, -9.53017734e-03, -2.07500998e-03,
    #      7.78288627e-03,  2.65984051e-02, -4.21635024e-02,
    #     -3.31141502e-02,  1.20253209e-03,  6.79265067e-04,
    #      1.45353039e-03,  6.09619601e-04,  3.63274328e-02,
    #      5.08815870e-02, -5.18652275e-02,  1.82975293e-03,
    #     -1.19612291e-02, -1.16415846e-03,  4.59729321e-03,
    #      2.28881277e-02, -1.01413028e-02,  5.54967625e-03,
    #     -1.50220934e-02,  4.47728252e-03, -1.12233227e-02,
    #      4.63295699e-04, -4.09219377e-02, -4.72241780e-03,
    #      1.00891409e-03, -3.23589833e-04, -1.35878951e-03,
    #     -4.50876541e-04, -3.33954208e-03, -4.12698649e-03,
    #     -2.24468531e-03, -5.30896783e-02, -2.31975168e-02,
    #      4.85839276e-03,  4.58189845e-03, -3.29067116e-04,
    #     -5.16094714e-02,  2.32823137e-02,  5.10993749e-02,
    #      6.78926823e-04,  7.87126645e-03,  2.85282242e-03,
    #     -5.28916419e-02, -3.74817662e-02, -3.77735682e-02,
    #     -1.23734958e-02,  7.31046405e-03, -4.76327352e-03,
    #     -2.66207121e-02,  2.90974719e-03,  5.80254942e-03,
    #     -1.29866926e-03, -3.99429537e-02, -5.59592200e-03,
    #     -5.30329868e-02, -1.53946830e-03,  1.57780387e-02,
    #     -5.41228941e-03,  7.03063374e-03, -1.38788298e-02,
    #      4.72171195e-02,  3.53192985e-02, -5.39102890e-02,
    #     -1.64142009e-02, -7.94644468e-03,  8.53429455e-03,
    #     -8.58757994e-04,  2.49690730e-02,  1.01376197e-03,
    #      2.79847924e-02,  3.96850891e-02, -1.32551305e-02,
    #     -2.18391814e-03,  4.93694656e-02, -1.12386048e-02,
    #     -5.33824563e-02,  2.87868492e-02,  1.17020626e-02,
    #     -1.73278293e-03,  2.64046434e-03, -8.42410047e-03,
    #     -2.37814691e-02, -1.13984533e-02]], dtype=torch.bfloat16))            

        self.additional_timestamp_channels = additional_timestamp_channels
        self.build_additional_timestamp_embedder()
        self.affline_norm = RMSNorm(self.config.hidden_size)
    
    def build_additional_timestamp_embedder(self):
        self.additional_timestamp_embedder = nn.ModuleDict()
        for cond_name, cond_emb_channels in self.additional_timestamp_channels.items():
            self.additional_timestamp_embedder[cond_name] = nn.Sequential(
                SDXLTimesteps(cond_emb_channels),
                SDXLTimestepEmbedding(cond_emb_channels, cond_emb_channels),
            )

    def prepare_additional_timestamp_embedder(self, **kwargs):
        condition_concat = []
        for cond_name, embedder in self.additional_timestamp_embedder.items():
            condition_concat.append(embedder(kwargs[cond_name]))
        embedding = torch.cat(condition_concat, dim=1)
        if embedding.shape[1] < self.config.hidden_size:
            embedding = nn.functional.pad(embedding, (0, self.config.hidden_size - embedding.shape[1]))
        return embedding
    
    def prepare_embedded_sequence(
        self, x_B_C_T_H_W: torch.Tensor, fps: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )

            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )

        x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if "fps_aware" in self.pos_emb_cls:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D, fps=fps)  # [B, T, H, W, D]
        else:
            x_B_T_H_W_D = x_B_T_H_W_D + self.pos_embedder(x_B_T_H_W_D)  # [B, T, H, W, D]
        return x_B_T_H_W_D
    
    def decoder_head(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        original_shape: Tuple[int, int, int, int, int],  # [B, C, T, H, W]
        crossattn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del crossattn_emb, crossattn_mask
        B, C, T, H, W = original_shape
        x_BT_HW_D = rearrange(x_B_T_H_W_D, "B T H W D -> (B T) (H W) D")
        x_BT_HW_D = self.final_layer(x_BT_HW_D, emb_B_D)
        x_B_D_T_H_W = rearrange(
            x_BT_HW_D,
            "(B T) (H W) (p1 p2 t C) -> B C (T t) (H p1) (W p2)",
            p1=1,
            p2=1,
            H=H // 1,
            W=W // 1,
            t=1,
            B=B,
        )
        return x_B_D_T_H_W

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        pos_ids: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): vae encoded videos (b s c)
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """
        ## Decoder forward
        # Decoder embedding.
        fps = kwargs.get('fps', None)
        padding_mask = kwargs.get('padding_mask', None)
        image_size = kwargs.get('image_size', None)
        inference_fwd = kwargs.get('inference_fwd', None)

        if len(x.shape) > 5:
            x = x.squeeze(0)
        if len(x.shape) == 5 and not inference_fwd:
            original_shape = x.shape
            B, C, T, H, W = original_shape
        else:
            cp_size = parallel_state.get_context_parallel_world_size()
            original_shape = kwargs.get('original_shape', [1, 16, 16 // cp_size, 44, 80])
        if len(fps.shape) >= 2:
            fps = fps.squeeze(0)
        if len(padding_mask.shape) >= 5:
                padding_mask = padding_mask.squeeze(0)
        if len(image_size.shape) >= 3:
            image_size = image_size.squeeze(0)

        if self.pre_process and not inference_fwd:
            # transpose to match
            x_B_T_H_W_D = self.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)
            x_S_B_D = rearrange(x_B_T_H_W_D, "B T H W D -> (T H W) B D")
        elif self.pre_process and inference_fwd:
            # already preprocessed for inference
            B, T, H, W, D = x.shape
            x_S_B_D = rearrange(x, "B T H W D -> (T H W) B D")
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states

        timesteps_B_D = self.t_embedder(timesteps.flatten())  # (b d_text_embedding)
        affline_emb_B_D = timesteps_B_D
        # from IPython import embed; embed()
        if self.additional_timestamp_channels:
            if type(image_size) == tuple:
                image_size = image_size[0]

            additional_cond_B_D = self.prepare_additional_timestamp_embedder(
                bs=original_shape[0],
                fps=fps,
                h=image_size[:, 0],
                w=image_size[:, 1],
                org_h=image_size[:, 2],
                org_w=image_size[:, 3],
            )

            affline_emb_B_D += additional_cond_B_D
        
        
        affline_emb_B_D = self.affline_norm(affline_emb_B_D)

        crossattn_emb = rearrange(crossattn_emb, 'B S D -> S B D')

        # [Parth] Enable Sequence Parallelism
        if self.config.sequence_parallel:
            if self.pre_process:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
            crossattn_emb = tensor_parallel.scatter_to_sequence_parallel_region(crossattn_emb)
            # affline_emb_B_D = tensor_parallel.scatter_to_sequence_parallel_region(affline_emb_B_D)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                if self.pre_process:
                    x_S_B_D = x_S_B_D.clone()
                crossattn_emb = crossattn_emb.clone()

        x_S_B_D = self.decoder(
            hidden_states=x_S_B_D,
            attention_mask=affline_emb_B_D,
            context=crossattn_emb,
            context_mask=None,
            packed_seq_params=packed_seq_params,
        )
        # Return if not post_process
        if not self.post_process:
            return x_S_B_D, None
        
        # shift_final, scale_final = self.final_layer_adaLN(timesteps_S_B_D)
        # x_S_B_D = self.final_layer_adaLN.modulate(x_S_B_D, shift_final, scale_final)
        # x_S_B_D = self.final_layer_linear(x_S_B_D)

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        if not inference_fwd:
            D = self.config.hidden_size
            x_B_T_H_W_D = rearrange(x_S_B_D, "(T H W) B D -> B T H W D", B=B, T=T, H=H, W=W, D=D)
        else:
            x_B_T_H_W_D = rearrange(x_S_B_D, "(T H W) B D -> B T H W D", B=B, T=T, H=H, W=W, D=D)

        x_B_T_H_W_D = self.decoder_head(x_B_T_H_W_D, affline_emb_B_D, None, original_shape, None)

        return x_B_T_H_W_D, self.logvar(timesteps)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = 'module.', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """ Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)

        for (param_name, param) in self.t_embedder.named_parameters():
            weight_key = f'{prefix}t_embedder.{param_name}'
            self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)

        for cond_name, embedder in self.additional_timestamp_embedder.items():
            for (param_name, param) in embedder.named_parameters():
                weight_key = f'{prefix}additional_timestamp_embedder.{cond_name}.{param_name}'
                self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)
            
        for (param_name, param) in self.affline_norm.named_parameters():
            weight_key = f'{prefix}affline_norm.{param_name}'
            self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)

        return sharded_state_dict
    
    def tie_embeddings_weights_state_dict(
        self,
        tensor,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if self.pre_process and parallel_state.get_tensor_model_parallel_rank() == 0:
            # Output layer is equivalent to the embedding already
            return

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        last_stage_word_emb_replica_id = (
            0, # copy of first stage embedding
            parallel_state.get_tensor_model_parallel_rank() + parallel_state.get_pipeline_model_parallel_rank() * parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=first_stage_word_emb_key,
            replica_id=last_stage_word_emb_replica_id,
            allow_shape_mismatch=False,
        )

class DiTModel(VisionModule):
    """DiT with CrossAttention model.

    Args:
        config (TransformerConfig): transformer config

        transformer_decoder_layer_spec (ModuleSpec): transformer layer customization specs for decoder

        pre_process (bool): Include embedding layer (used with pipeline parallelism)
        post_process (bool): Include an output layer (used with pipeline parallelism)

        fp16_lm_cross_entropy (bool, optional): Defaults to False

        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks

        share_embeddings_and_output_weights (bool): When True, input embeddings and output logit weights are
            shared. Defaults to False.

        position_embedding_type (string): Position embedding type. Options ['learned_absolute', 'rope'].
            Defaults is 'learned_absolute'.

        rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings.
            Defaults to 1.0 (100%). Ignored unless position_embedding_type is 'rope'.

        seq_len_interpolation_factor (float): scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec = DiTLayerWithAdaLNspec,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
        rotary_percent: float = 1.0,
        seq_len_interpolation_factor: Optional[float] = None,
        max_height=64,
        max_width=80,
        patch_size=1,
        in_channels=16,
        max_t=256,
        t_embed_seed=None,
    ):

        super(DiTModel, self).__init__(config=config)

        self.config: TransformerConfig = config

        self.transformer_decoder_layer_spec = transformer_layer_spec()
        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = True
        self.add_decoder = True
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.share_embeddings_and_output_weights = False 

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # Transformer decoder
        self.decoder = TransformerBlock(
            config=self.config,
            spec=self.transformer_decoder_layer_spec,
            pre_process=self.pre_process,
            post_process=False,
            post_layer_norm=False,
        )

        self.t_embedder = torch.nn.Sequential(
            SDXLTimesteps(self.config.hidden_size),
            SDXLTimestepEmbedding(self.config.hidden_size, self.config.hidden_size, seed=t_embed_seed),
        )

        if self.pre_process:
            self.x_embedder = torch.nn.Linear(in_channels * patch_size **2, self.config.hidden_size)
            self.pos_embedder = SinCosPosEmb3D(
                model_channels=self.config.hidden_size,
                len_h=max_height // patch_size,
                len_w=max_width // patch_size,
                len_t=max_t // 1,
                is_learnable=True,
            )
        if self.post_process:
            self.final_layer_adaLN = AdaLN(config=self.config, n_adaln_chunks=2)
            self.final_layer_linear = torch.nn.Linear(
                self.config.hidden_size,
                patch_size**2 * 1 * in_channels,
            )
        

    def forward(
        self,
        x: Tensor,
        timesteps: Tensor,
        crossattn_emb: Tensor,
        pos_ids: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): vae encoded videos (b s c)
            encoder_decoder_attn_mask (Tensor): cross-attention mask between encoder and decoder
            inference_params (InferenceParams): relevant arguments for inferencing

        Returns:
            Tensor: loss tensor
        """
        # args = get_args()
        # if packed_seq_params and args.packing_algorithm == 'no_packing':
        #     for k in packed_seq_params.keys():
        #         packed_seq_params[k] =  None

        ## Decoder forward
        # Decoder embedding.
        if self.pre_process:
            x = rearrange(x, 'B S D -> S B D')
            x_S_B_D = self.x_embedder(x)
            pos_ids = rearrange(pos_ids, 'B S D -> S B D')
            x_S_B_D = x_S_B_D + self.pos_embedder(pos_ids)
        else:
            # intermediate stage of pipeline
            x_S_B_D = None  ### should it take encoder_hidden_states

        # timesteps_B, timesteps_S = timesteps.shape
        timesteps_B_D = self.t_embedder(timesteps.flatten())
        timesteps_S_B_D = timesteps_B_D.unsqueeze(0) # B D -> S(1) B D
        # timesteps_S_B_D = rearrange(timesteps_B_D, "(S B) D -> S B D", S=timesteps_S, B=timesteps_B)

        crossattn_emb = rearrange(crossattn_emb, 'B S D -> S B D')

        if self.config.sequence_parallel:
            if self.pre_process:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
            crossattn_emb = tensor_parallel.scatter_to_sequence_parallel_region(crossattn_emb)
            
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                if self.pre_process:
                    x_S_B_D = x_S_B_D.clone()
                crossattn_emb = crossattn_emb.clone()
        
        x_S_B_D = self.decoder(
            hidden_states=x_S_B_D,
            attention_mask=timesteps_S_B_D,
            context=crossattn_emb,
            context_mask=None,
            packed_seq_params=packed_seq_params,
        )

        # Return if not post_process
        if not self.post_process:
            return x_S_B_D
        
        shift_final, scale_final = self.final_layer_adaLN(timesteps_S_B_D)
        x_S_B_D = self.final_layer_adaLN.modulate(x_S_B_D, shift_final, scale_final)
        x_S_B_D = self.final_layer_linear(x_S_B_D)

        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)
        x_B_S_D = rearrange(x_S_B_D, 'S B D -> B S D')
        return x_B_S_D

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """ Sharded state dict implementation for GPTModel backward-compatibility (removing extra state).

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        for (param_name, param) in self.t_embedder.named_parameters():
            weight_key = f'{prefix}t_embedder.{param_name}'
            self.tie_embeddings_weights_state_dict(param, sharded_state_dict, weight_key, weight_key)
        
        return sharded_state_dict
    
    def tie_embeddings_weights_state_dict(
        self,
        tensor,
        sharded_state_dict: ShardedStateDict,
        output_layer_weight_key: str,
        first_stage_word_emb_key: str,
    ) -> None:
        """Ties the embedding and output weights in a given sharded state dict.

        Args:
            sharded_state_dict (ShardedStateDict): state dict with the weight to tie
            output_layer_weight_key (str): key of the output layer weight in the state dict.
                This entry will be replaced with a tied version
            first_stage_word_emb_key (str): this must be the same as the
                ShardedTensor.key of the first stage word embeddings.

        Returns: None, acts in-place
        """
        if self.pre_process and parallel_state.get_tensor_model_parallel_rank() == 0:
            # Output layer is equivalent to the embedding already
            return

        # Replace the default output layer with a one sharing the weights with the embedding
        del sharded_state_dict[output_layer_weight_key]
        last_stage_word_emb_replica_id = (
            0, # copy of first stage embedding
            parallel_state.get_tensor_model_parallel_rank() + parallel_state.get_pipeline_model_parallel_rank() * parallel_state.get_pipeline_model_parallel_world_size(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

        sharded_state_dict[output_layer_weight_key] = make_sharded_tensor_for_checkpoint(
            tensor=tensor,
            key=first_stage_word_emb_key,
            replica_id=last_stage_word_emb_replica_id,
            allow_shape_mismatch=False,
        )

class OfficialDiTModel(VisionModule):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        config: TransformerConfig,
        input_size=32,
        patch_size=2,
        in_channels=4,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        pre_process=True,
        post_process=True,
        t_embed_seed=None,
    ):
        super(OfficialDiTModel, self).__init__(config=config)
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        hidden_size = self.config.hidden_size
        self.pre_process = pre_process
        self.post_process = post_process
        self.share_embeddings_and_output_weights = False 

        if pre_process:
            self.x_embedder = torch.nn.Linear(in_channels * patch_size **2, self.config.hidden_size)
            num_patches = (input_size // patch_size)**2
            cp_size = parallel_state.get_context_parallel_world_size()
            cp_rank = parallel_state.get_context_parallel_rank()
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches//cp_size, hidden_size), requires_grad=False)
            pos_embed = get_2d_sincos_pos_embed(hidden_size, int(num_patches**0.5), int(num_patches**0.5))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().reshape(cp_size, -1, hidden_size)[cp_rank].unsqueeze(0)) # for we slice the input along H for cp

        self.t_embedder = TimestepLabelEmbedder(num_classes, hidden_size, class_dropout_prob, seed=t_embed_seed)
 
        self.decoder = TransformerBlock(config=config, 
                                        spec=OfficialDiTLyaerWithAdaLNspec(),
                                        pre_process=pre_process,
                                        post_process=post_process,
                                        post_layer_norm=False,
                                        )
        if post_process:
            self.final_layer = FinalLayer(hidden_size, patch_size, 1, self.out_channels)
        # self.initialize_weights()

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])


    # def initialize_weights(self):
    #     # Initialize transformer layers:
    #     # def _basic_init(module):
    #     #     if isinstance(module, nn.Linear):
    #     #         torch.nn.init.xavier_uniform_(module.weight)
    #     #         if module.bias is not None:
    #     #             nn.init.constant_(module.bias, 0)
    #     # self.apply(_basic_init)

    #     if self.pre_process:
    #         # Initialize (and freeze) pos_embed by sin-cos embedding:
    #         pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
    #         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    #         # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
    #         w = self.x_embedder.proj.weight.data
    #         nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    #         nn.init.constant_(self.x_embedder.proj.bias, 0)

    #         # Initialize label embedding table:
    #         nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

    #         # Initialize timestep embedding MLP:
    #         nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
    #         nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    #     # Zero-out adaLN modulation layers in DiT blocks:
    #     for block in self.blocks.layers:
    #         nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
    #         nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    #     if self.post_process:
    #         # Zero-out output layers:
    #         nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
    #         nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
    #         nn.init.constant_(self.final_layer.linear.weight, 0)
    #         nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        cp_size = parallel_state.get_context_parallel_world_size()
        h = w = int((x.shape[1]*cp_size) ** 0.5)
        assert h//cp_size * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h//cp_size, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h//cp_size * p, w * p))
        return imgs

    def _forward(self, x, t, y, packed_seq_params: PackedSeqParams = None,):
        """
        Forward pass of DiT.
        x: (B, S, C*P^2) tensor of spatial inputs (images or latent representations of images)
        t: (B,) tensor of diffusion timesteps
        y: (B,) tensor of class labels
        """

        c_B_D = self.t_embedder(t, y)  
        c_S_B_D = c_B_D.unsqueeze(0) # B D -> S(1) B D

        if self.pre_process:
            x_B_S_D = self.x_embedder(x) + self.pos_embed  
            x_S_B_D = x_B_S_D.permute(1, 0, 2).contiguous()      
        
            if self.config.sequence_parallel:
                x_S_B_D = tensor_parallel.scatter_to_sequence_parallel_region(x_S_B_D)
                if self.config.clone_scatter_output_in_embedding:
                    x_S_B_D = x_S_B_D.clone()
        else:
            x_S_B_D = None

        x_S_B_D = self.decoder(
                        hidden_states=x_S_B_D,
                        attention_mask=c_S_B_D,
                        context=None,
                        context_mask=None,
                        packed_seq_params=packed_seq_params
                        )                  

        if not self.post_process:
            return x_S_B_D

        x_S_B_D = self.final_layer(x_S_B_D, c_B_D)                # (N, T , patch_size ** 2 * out_channels)
        #To-Do: We can split the target noise to different SP rank and calculate the loss, then do all reduce to gather the loss together.
        if self.config.sequence_parallel:
            x_S_B_D = tensor_parallel.gather_from_sequence_parallel_region(x_S_B_D)

        x_B_S_D = x_S_B_D.permute(1, 0, 2)
        x_B_C_H_W = self.unpatchify(x_B_S_D)                   # (N, out_channels, H, W)
        return x_B_C_H_W 

    def _forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        if self.pre_process:
            half = x[: len(x) // 2]
            combined = torch.cat([half, half], dim=0)
        else:
            combined, t, y = None, None, None

        model_out = self._forward(combined, t, y)

        if not self.post_process:
            return model_out
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward(self, x, t, y, cfg_scale=None):
        if cfg_scale is None:
            return self._forward(x, t, y)
        else:
            return self._forward_with_cfg(x, t, y, cfg_scale=cfg_scale)
