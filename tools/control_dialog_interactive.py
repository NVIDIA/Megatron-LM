
"""Sample Generate Controllable Dialog Model"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import argparse
import torch
from transformers import DPRQuestionEncoderTokenizer
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation_utils import dialog_with_gpt_control_interactive, dialog_with_dpr_control_interactive


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False,
                     pre_process=pre_process, post_process=post_process)

    return model


def add_control_dialog_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--out-seq-length", type=int, default=1024,
                       help='Size of the output generated text.')
    group.add_argument("--recompute", action='store_true',
                       help='During generation recompute all attention '
                       'instead of using previously computed keys/values.')
    group.add_argument("--ctrl-type", type=str, default="", 
                        help="Either dpr or gpt")
    group.add_argument("--ctrl-hidden-size", type=int, default=1024, 
                        help="hidden-size of gpt control model")
    group.add_argument("--ctrl-num-layers", type=int, default=24, 
                        help="num-layers of gpt control model")
    group.add_argument("--ctrl-num-attention-heads", type=int, default=16,
                        help="num-attention-heads of gpt control model")
    group.add_argument("--ctrl-gpt-load", type=str, default="",
                        help="checkpoint path of the gpt control model")
    group.add_argument("--ctrl-dpr-load", type=str, default="",
                        help="checkpoint path of the dpr control model")
    group.add_argument("--knowledge-corpus-path", type=str, default="",
                        help="The path for the knowledge corpus")
    group.add_argument("--knowledge-corpus-emb", type=str, default="",
                        help="The path for the knowledge embedding")                 
    group.add_argument('--spec-toks', type=str, default=None,
                        help='additional special tokens')
    group.add_argument('--add-separator', action="store_true",
                        help='Add separator for the inputs')
    
    return parser


def main():
    """Main program."""

    initialize_megatron(extra_args_provider=add_control_dialog_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up conversational model
    conv_model = get_model(model_provider)
    if args.load is not None:
        _ = load_checkpoint(conv_model, None, None)

    assert len(conv_model) == 1, "Above condition should have caught this"
    conv_model = conv_model[0]

    # Set up control model
    assert args.ctrl_type in ["gpt", "dpr"], \
                "please input a correct control model type"
    
    if args.ctrl_type == "gpt":
        args.consumed_train_samples = 0
        args.consumed_valid_samples = 0
        args.hidden_size = args.ctrl_hidden_size
        args.ffn_hidden_size = 4 * args.hidden_size
        args.num_layers = args.ctrl_num_layers
        args.num_attention_heads = args.ctrl_num_attention_heads
        args.load = args.ctrl_gpt_load

        ctrl_model = get_model(model_provider)
        if args.load is not None:
            _ = load_checkpoint(ctrl_model, None, None)
        ctrl_model = ctrl_model[0]
        
        dialog_with_gpt_control_interactive(conv_model, ctrl_model, args.add_separator)

    else:
        print_rank_0("> Loading model from %s" % args.ctrl_dpr_load)
        ctrl_model = torch.load(args.ctrl_dpr_load)
        ctrl_model.cuda()
        ctrl_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        
        print_rank_0("> Loading knowledge corpus and embeddings")
        with open(args.knowledge_corpus_path, "r") as f:
            knowledge_corpus = f.readlines()
        knowledge_corpus_emb = torch.load(args.knowledge_corpus_emb)
        knowledge_corpus_emb = knowledge_corpus_emb.cuda()

        assert knowledge_corpus_emb.size()[0] == len(knowledge_corpus), \
            "The size of knowledge corpus and embeddings should be the same"

        dialog_with_dpr_control_interactive(conv_model, ctrl_model,
                                            ctrl_tokenizer, knowledge_corpus, 
                                            knowledge_corpus_emb, args.add_separator)


if __name__ == "__main__":

    main()
