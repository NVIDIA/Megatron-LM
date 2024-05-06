# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.core.models.multimodal import GPTVisionModel
from megatron.training import get_model
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.inference.text_generation_server import MegatronServer
from megatron.inference.text_generation import generate_and_post_process
from megatron.inference.text_generation import beam_search_and_post_process
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)

import torch
from typing import Union
import megatron


from megatron.core.models.vision.eva_clip_model import Eva2ClipModel
from megatron.core.transformer.transformer_config import VisionTransformerConfig
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec_for_eva_clip,
)
from megatron.training.checkpointing import load_checkpoint

class MegatronVisionModel(torch.nn.Module):
    def __init__(self, pre_process):
        super().__init__()
        args = get_args()
        dtype = torch.float32
        if args.bf16:
            dtype = torch.bfloat16
        elif args.fp16:
            dtype = torch.half
        self.dtype = dtype
        eva_args = torch.load(os.path.join(args.vit_load, "iter_0000001/mp_rank_00/model_optim_rng.pt"), map_location="cpu")["args"]
        eva_args.independent_parallel = True
        assert args.tensor_model_parallel_size == eva_args.tensor_model_parallel_size
        print('building EVA model ...')
        config = core_transformer_config_from_args(eva_args, VisionTransformerConfig)
        assert config.independent_parallel
        transformer_layer_spec = get_vit_layer_with_transformer_engine_spec_for_eva_clip()
        self.vit = Eva2ClipModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=1,
            pre_process=pre_process,
        )
        eva_args.load = args.vit_load
        load_checkpoint([self.vit], None, None, args=eva_args)
        self.linear_proj = torch.nn.Linear(eva_args.hidden_size, args.hidden_size)
    
    def forward(self, **kw_args):
        kw_args.pop('indices', None)
        kw_args.pop('pre_len', None)
        external_inputs = {"images": kw_args.pop('images').to(self.dtype)}
        if 'attention_mask' not in kw_args:
            kw_args['attention_mask'] = None
        vit_output = self.vit(**kw_args, external_inputs=external_inputs)
        return self.linear_proj(vit_output.transpose(0, 1))

def eva_model_provider(config):
    model = MegatronVisionModel(True)
    return model

def model_provider(pre_process=True, post_process=True) -> Union[GPTVisionModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """

    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"

    print_rank_0('building GPTVision model ...')

    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
        else:
            transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)

    model = GPTVisionModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        external_feature_model_provider=eva_model_provider
    )

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    group.add_argument("--vit-load", type=str,
                       help='path to load vit model')
    group.add_argument("--image-seq-length", type=int,
                       help='vit image length')
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    from megatron.inference.text_generation import forward_step
    from functools import partial
    from dataset import BlipImageEvalProcessor

    def blip2_image_processor_func_megatron_inference(image_processor, image):
        return {'images': image_processor(image).unsqueeze(0).cuda(), 'input_ids': torch.zeros(1, args.image_seq_length, dtype=torch.long).cuda(), 'position_ids': torch.arange(args.image_seq_length, dtype=torch.long).unsqueeze(0).cuda(), 'pre_len': 0}
    blip2_image_processor_megatron_inference_224 = partial(blip2_image_processor_func_megatron_inference, BlipImageEvalProcessor(224))

    forward_step._IMAGE_PROCESSOR = blip2_image_processor_megatron_inference_224

    from megatron.training.tokenizer.tokenizer import _Llama2Tokenizer
    class _llama2_vision_tokenizer(_Llama2Tokenizer):

        def tokenize(self, s: str, bos=True, eos=False):
            return [0] * args.image_seq_length + super().tokenize(s, bos, eos)

        def detokenize(self, ids):
            return super().detokenize(ids[args.image_seq_length:])

    from megatron.training import global_vars
    global_vars._GLOBAL_TOKENIZER = _llama2_vision_tokenizer(args.tokenizer_model)
    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        server = MegatronServer(model)
        server.run("0.0.0.0",port=args.port)

    while True:
        choice = torch.tensor(1, dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice.item() == 0:
            try:
                generate_and_post_process(model)
            except ValueError as ve:
                pass
        elif choice.item() == 1:
            try:
                beam_search_and_post_process(model)
            except ValueError as ve:
                pass
