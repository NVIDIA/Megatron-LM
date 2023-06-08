from megatron import get_args
from megatron.model.utils import init_method_normal
import math
import torch
import torch.nn.functional as F
import einops
from .module import MegatronModule
from deepspeed.accelerator import get_accelerator
from megatron.mpu.utils import ClsUtility
from megatron.mpu.initialize import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from megatron.mpu.mappings import reduce_from_tensor_model_parallel_region
from megatron.mpu.layers import ColumnParallelLinear

def twod_interpolate_position_embeddings_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):

    args = get_args()
    num_patches_per_dim = args.img_dim // args.patch_dim
    num_patches = num_patches_per_dim ** 2
    seq_length = num_patches + 1
    hidden_size = args.hidden_size

    key = prefix + "weight"
    # import pdb
    # pdb.set_trace()
    assert key in state_dict
    if key in state_dict:
        input_param = state_dict[key]

        assert input_param.shape[1] == hidden_size
        if input_param.shape[0] != seq_length:
            # update input_param and load it to state_dict[key]

            num_tok_input = input_param.shape[0] - 1
            num_tok_new = seq_length - 1
            input_param_tok, input_param_grid = (
                input_param[:1, :],
                input_param[1:, :],
            )

            gs_input = int(math.sqrt(num_tok_input))
            gs_new = int(math.sqrt(num_tok_new))

            input_param_grid = input_param_grid.transpose(0, 1).contiguous()
            input_param_grid = input_param_grid.reshape(
                (1, -1, gs_input, gs_input)
            )
            input_param_grid = input_param_grid.float()
            scale_factor = gs_new / gs_input

            input_param_grid = F.interpolate(
                input_param_grid, scale_factor=scale_factor, mode="bilinear"
            )

            input_param_grid = input_param_grid.half()
            input_param_grid = input_param_grid.reshape((-1, gs_new * gs_new))
            input_param_grid = input_param_grid.transpose(0, 1).contiguous()

            assert input_param_grid.shape[1] == hidden_size
            input_param = torch.cat((input_param_tok, input_param_grid), dim=0)
            assert (
                input_param.shape[0] == seq_length
                and input_param.shape[1] == hidden_size
            )

            state_dict[key] = input_param

class VitEmbedding(MegatronModule):
    def __init__(self):
        super(VitEmbedding, self).__init__(share_word_embeddings=False)
        args = get_args()
        self.hidden_size = args.hidden_size
        self.patch_dim = args.patch_dim
        self.img_dim = args.img_dim

        assert self.img_dim % self.patch_dim == 0
        self.num_patches_per_dim = self.img_dim // self.patch_dim
        self.num_patches = self.num_patches_per_dim ** 2
        self.seq_length = self.num_patches + 1
        self.flatten_dim = self.patch_dim * self.patch_dim * args.num_channels
        # cls_token
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, self.hidden_size))
        torch.nn.init.zeros_(self.cls_token)

        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()


        # Linear encoder
        self.linear_encoder = ColumnParallelLinear(
            self.flatten_dim, self.hidden_size, gather_output=True
        )

        # embedding
        self.position_embeddings = torch.nn.Embedding(
            self.seq_length, self.hidden_size
        )
        init_method_normal(args.init_method_std)(
            self.position_embeddings.weight
        )
        self.position_ids = torch.arange(self.seq_length).expand(1, -1).to(get_accelerator().device_name())

        self.position_embeddings._register_load_state_dict_pre_hook(
            twod_interpolate_position_embeddings_hook
        )

        self.embedding_dropout = torch.nn.Dropout(args.hidden_dropout)

    def forward(self, x):
        x = einops.rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_dim,
            p2=self.patch_dim,
        )

        assert x.dtype == torch.half
        x, _ = self.linear_encoder(x)
        # Reduce across all the model parallel GPUs.

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.position_embeddings(self.position_ids)
        x = self.embedding_dropout(x)
        return x


class VitEmbeddingPipe(VitEmbedding):

    def forward(self, inputs, **kwargs):
        if not hasattr(self, '_args'):
            self._args = get_args()


        embeddings = super().forward(inputs)
        return embeddings

    @property
    def linear_encoder_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.linear_encoder.weight
