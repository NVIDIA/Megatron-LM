"""V7 checkpoint-reshard round-trip for Gemma 4 E4B.

Phase 'save': build the model at --tp, load the source dist-checkpoint (--src,
saved at a DIFFERENT TP), and re-save it as a dist-checkpoint at THIS tp (--save-to).
Phase 'verify': build at --tp, load --src, forward FIXED_TOKENS, compare to a
reference logits dump (--ref-logits). Because dist-checkpoint resharding is lossless,
a TP=1 -> TP=2 -> TP=1 round trip must reproduce the original TP=1 logits BITWISE.

    # save the TP=1 ckpt re-sharded at TP=2:
    torchrun --nproc_per_node=2 tp_reshard.py --mode save --tp 2 --src <tp1_ckpt> --save-to <tp2_ckpt>
    # load that TP=2 ckpt back at TP=1 and check it reproduces the TP=1 oracle logits:
    torchrun --nproc_per_node=1 tp_reshard.py --mode verify --tp 1 --src <tp2_ckpt> --ref-logits logits_tp1.pt
"""
import mlm_env  # noqa: F401  MUST be first

import argparse
import functools
import os

import torch

GELU_TANH = functools.partial(torch.nn.functional.gelu, approximate="tanh")
FIXED_TOKENS = [2, 651, 1234, 99, 17, 8, 200, 9000, 42, 3]


def _init(tp):
    import megatron.core.parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", str(rank))))
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12399")
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    ps.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    return rank


def _build(tp):
    from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model
    from megatron.core.models.gemma4.gemma4_layer_specs import get_gemma4_layer_local_spec
    config = Gemma4TransformerConfig(
        num_layers=42, hidden_size=2560, ffn_hidden_size=10240,
        num_attention_heads=8, num_query_groups=2, layernorm_epsilon=1e-6,
        gated_linear_unit=True, activation_func=GELU_TANH, add_bias_linear=False,
        qk_layernorm=True, bias_activation_fusion=False, bf16=True,
        params_dtype=torch.bfloat16, attention_softmax_in_fp32=True,
        masked_softmax_fusion=False, pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp, sequence_parallel=False,
    )
    spec = get_gemma4_layer_local_spec(config)
    return Gemma4Model(config=config, transformer_layer_spec=spec,
                       vocab_size=262144, max_sequence_length=512).bfloat16().cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["save", "verify"], required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--src", required=True)
    ap.add_argument("--save-to", default=None)
    ap.add_argument("--ref-logits", default=None)
    args = ap.parse_args()

    from megatron.core import dist_checkpointing
    torch.set_grad_enabled(False)
    rank = _init(args.tp)
    model = _build(args.tp)
    model.eval()

    loaded = dist_checkpointing.load(model.sharded_state_dict(), args.src)
    model.load_state_dict(loaded, strict=False)
    if rank == 0:
        print(f"[rank{rank}] loaded src {args.src} into TP={args.tp} model")

    if args.mode == "save":
        os.makedirs(args.save_to, exist_ok=True)
        dist_checkpointing.save(model.sharded_state_dict(), args.save_to)
        if rank == 0:
            print(f"[rank{rank}] re-saved at TP={args.tp} -> {args.save_to}")
        torch.distributed.barrier()
        return

    # verify
    ids = torch.tensor([FIXED_TOKENS], device="cuda")
    logits = model(ids)
    if args.tp > 1:
        from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
        logits = gather_from_tensor_model_parallel_region(logits)
    if rank == 0:
        greedy = logits.argmax(dim=-1).cpu()
        print(f"reshard-verify greedy = {greedy.tolist()}")
        if args.ref_logits and os.path.exists(args.ref_logits):
            ref = torch.load(args.ref_logits, map_location="cpu", weights_only=False)
            rl = ref["logits"].float()
            # ref may be padded; slice to FIXED_TOKENS length.
            n = len(FIXED_TOKENS)
            rl = rl[:, :n]
            dl = logits.float().cpu()[:, :n]
            diff = (rl - dl).abs().max().item()
            bitwise = torch.equal(rl, dl)
            rg, dg = rl.argmax(-1), dl.argmax(-1)
            print(f"vs ref {args.ref_logits}: max_abs={diff:.3e} BITWISE={bitwise} "
                  f"greedy_exact={torch.equal(rg, dg)}")
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
