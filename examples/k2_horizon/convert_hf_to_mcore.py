# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Convert a K2-Horizon Hugging Face checkpoint to Megatron-Core and check logit parity.

The Megatron model is built through the same model_provider/gpt_builder path that
pretrain_gpt.py uses, so a passing parity check validates the model flags in
k2_horizon_0.9b_model_args.sh, not just the weight mapping. Runs on a single GPU
(TP=1, PP=1); the saved torch_dist checkpoint can be resharded to other TP/PP sizes
at load time. Pass the RoPE flags of the stage the checkpoint comes from.

Example (the released main checkpoint):
    source examples/k2_horizon/k2_horizon_0.9b_model_args.sh
    torchrun --nproc-per-node 1 examples/k2_horizon/convert_hf_to_mcore.py \
        "${K2_HORIZON_0P9B_ARCH_ARGS[@]}" "${K2_HORIZON_ROPE_FINAL_ARGS[@]}" \
        --hf-path /path/to/K2-Horizon-0.9B --tokenizer-model /path/to/K2-Horizon-0.9B \
        --bf16 --seq-length 4096 --micro-batch-size 1 --train-iters 1 \
        --save /path/to/k2_horizon_0.9b_mcore
"""

import glob
import os
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from gpt_builders import gpt_builder  # noqa: E402
from model_provider import model_provider  # noqa: E402

from megatron.training import get_args, print_rank_0  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402
from megatron.training.checkpointing import save_checkpoint  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402


def add_conversion_args(parser):
    group = parser.add_argument_group(title='K2-Horizon conversion')
    group.add_argument('--hf-path', type=str, required=True,
                       help='Directory holding the K2-Horizon safetensors, config and modeling code.')
    group.add_argument('--skip-verify', action='store_true',
                       help='Skip the Hugging Face logit-parity check.')
    group.add_argument('--verify-text-file', type=str, default=None,
                       help='Text used for the parity check. Defaults to README.md in --hf-path.')
    return parser


def load_hf_state_dict(hf_path):
    state_dict = {}
    for shard in sorted(glob.glob(os.path.join(hf_path, '*.safetensors'))):
        state_dict.update(load_file(shard))
    assert state_dict, f'no safetensors found in {hf_path}'
    return state_dict


def hf_to_mcore_state_dict(args, hf):
    """Map HF tensor names/layouts onto the Megatron-Core TE layer spec."""
    num_query_groups = args.num_query_groups
    queries_per_group = args.num_attention_heads // num_query_groups
    head_dim = args.kv_channels
    hidden = args.hidden_size

    vocab = hf['model.embed_tokens.weight'].shape[0]
    assert args.padded_vocab_size == vocab, (
        f'padded vocab {args.padded_vocab_size} != HF vocab {vocab}; '
        'vocab padding is not implemented in this converter'
    )

    out = {
        'embedding.word_embeddings.weight': hf.pop('model.embed_tokens.weight'),
        'decoder.final_layernorm.weight': hf.pop('model.norm.weight'),
        'output_layer.weight': hf.pop('lm_head.weight'),
    }
    for i in range(args.num_layers):
        src = f'model.layers.{i}.'
        dst = f'decoder.layers.{i}.'
        # Megatron packs QKV per KV group: [q_1..q_n, k, v] for each group.
        q = hf.pop(src + 'self_attn.q_proj.weight').view(
            num_query_groups, queries_per_group * head_dim, hidden
        )
        k = hf.pop(src + 'self_attn.k_proj.weight').view(num_query_groups, head_dim, hidden)
        v = hf.pop(src + 'self_attn.v_proj.weight').view(num_query_groups, head_dim, hidden)
        out[dst + 'self_attention.linear_qkv.weight'] = torch.cat([q, k, v], dim=1).reshape(
            -1, hidden
        )
        out[dst + 'self_attention.linear_qkv.layer_norm_weight'] = hf.pop(
            src + 'input_layernorm.weight'
        )
        out[dst + 'self_attention.linear_proj.weight'] = hf.pop(src + 'self_attn.o_proj.weight')
        # SwiGLU fc1 stacks [gate; up].
        out[dst + 'mlp.linear_fc1.weight'] = torch.cat(
            [hf.pop(src + 'mlp.gate_proj.weight'), hf.pop(src + 'mlp.up_proj.weight')], dim=0
        )
        out[dst + 'mlp.linear_fc1.layer_norm_weight'] = hf.pop(
            src + 'post_attention_layernorm.weight'
        )
        out[dst + 'mlp.linear_fc2.weight'] = hf.pop(src + 'mlp.down_proj.weight')

    assert not hf, f'unmapped HF tensors: {sorted(hf)}'
    return out


@torch.no_grad()
def load_into_model(model, mcore_state_dict):
    params = dict(model.named_parameters())
    missing = sorted(set(params) - set(mcore_state_dict))
    unexpected = sorted(set(mcore_state_dict) - set(params))
    assert not missing and not unexpected, f'missing={missing} unexpected={unexpected}'
    for name, param in params.items():
        src = mcore_state_dict[name]
        assert param.shape == src.shape, f'{name}: {tuple(param.shape)} vs {tuple(src.shape)}'
        param.copy_(src.to(device=param.device, dtype=param.dtype))


@torch.no_grad()
def verify_parity(args, model):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
    with open(args.verify_text_file or os.path.join(args.hf_path, 'README.md')) as f:
        text = f.read()
    ids = tokenizer(text, add_special_tokens=False)['input_ids'][: args.seq_length]
    input_ids = torch.tensor([ids], device='cuda')
    position_ids = torch.arange(input_ids.shape[1], device='cuda').unsqueeze(0)

    model.eval()
    mcore_logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    mcore_logits = mcore_logits.float()

    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_path, trust_remote_code=True, dtype=args.params_dtype, attn_implementation='sdpa'
    ).cuda().eval()
    hf_logits = hf_model(input_ids=input_ids).logits.float()
    del hf_model

    assert mcore_logits.shape == hf_logits.shape, (mcore_logits.shape, hf_logits.shape)
    labels = input_ids[:, 1:]
    hf_loss = F.cross_entropy(hf_logits[:, :-1].flatten(0, 1), labels.flatten())
    mcore_loss = F.cross_entropy(mcore_logits[:, :-1].flatten(0, 1), labels.flatten())
    diff = (mcore_logits - hf_logits).abs()
    top1 = (mcore_logits.argmax(-1) == hf_logits.argmax(-1)).float().mean()
    kl = F.kl_div(
        F.log_softmax(mcore_logits, -1), F.log_softmax(hf_logits, -1),
        log_target=True, reduction='none',
    ).sum(-1).mean()

    print_rank_0(f'parity check on {input_ids.shape[1]} tokens ({args.params_dtype}):')
    print_rank_0(f'  HF loss {hf_loss.item():.5f} | Megatron loss {mcore_loss.item():.5f}')
    print_rank_0(f'  logits max|diff| {diff.max().item():.4g} | mean|diff| {diff.mean().item():.4g}')
    print_rank_0(f'  top-1 agreement {top1.item():.4%} | KL(HF||Megatron) {kl.item():.3g}')


def main():
    parse_and_validate_args(
        extra_args_provider=add_conversion_args, args_defaults={'save_interval': 1}
    )
    initialize_megatron()
    args = get_args()
    assert args.tensor_model_parallel_size == 1 and args.pipeline_model_parallel_size == 1

    model = model_provider(gpt_builder, pre_process=True, post_process=True)
    model = model.to(device='cuda', dtype=args.params_dtype)

    print_rank_0(f'loading HF checkpoint from {args.hf_path}')
    load_into_model(model, hf_to_mcore_state_dict(args, load_hf_state_dict(args.hf_path)))

    if not args.skip_verify:
        verify_parity(args, model)

    if args.save:
        save_checkpoint(0, [model], None, None, 0)
        print_rank_0(f'saved Megatron checkpoint to {args.save}')


if __name__ == '__main__':
    main()
