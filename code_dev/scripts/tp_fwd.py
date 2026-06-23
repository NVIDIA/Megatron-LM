"""TP/SP-parameterized forward harness for Gemma 4 E4B (IMPL-gemma4-tpsp).

Builds Gemma4Model at TP=N (optionally --sequence-parallel), loads the TP=1
converted dist-checkpoint (mlm_ckpt) via Megatron distributed-checkpoint
resharding (sharded_state_dict at TP=N auto-reshards the TP=1 save), runs a
forward on FIXED_TOKENS, and:
  - prints logits shape / finiteness / softcap range / greedy tokens (rank 0);
  - optionally dumps full logits + per-layer hidden states to --out for cross-TP
    parity comparison (compare_logits.py);
  - if --hf-ref given and TP==1, reports max-abs-diff + greedy-exact vs HF.

Launch under torchrun with --nproc_per_node=<TP>:
    torchrun --nproc_per_node=2 tp_fwd.py --tp 2 --sequence-parallel --out out_tp2.pt

Single forward, no_grad. PP=CP=1. Local (non-TE) spec by default (the bitwise
oracle spec); --te selects the TE spec.
"""
import mlm_env  # noqa: F401  MUST be first

import argparse
import functools
import os

import torch

GELU_TANH = functools.partial(torch.nn.functional.gelu, approximate="tanh")

IMPL = (
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_genai/users/ataghibakhsh/"
    "Gemma4_mlm/code_dev/shared-state/implementations/HYBRID-gemma4-mlm"
)
DEFAULT_CKPT = os.path.join(IMPL, "mlm_ckpt")
DEFAULT_HF_REF = os.path.join(IMPL, "refs", "full_e4b.pt")
FIXED_TOKENS = [2, 651, 1234, 99, 17, 8, 200, 9000, 42, 3]


def _init_distributed(tp, sp):
    import megatron.core.parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12399")
        torch.distributed.init_process_group(backend="nccl", world_size=world, rank=rank)
    ps.initialize_model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)
    return rank, world


def _make_config(tp, sp, num_layers=42, hidden=2560, ffn=10240):
    from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig

    return Gemma4TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=8,
        num_query_groups=2,
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=GELU_TANH,
        add_bias_linear=False,
        qk_layernorm=True,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        pipeline_dtype=torch.bfloat16,
        tensor_model_parallel_size=tp,
        sequence_parallel=sp,
    )


def _build_model(config, vocab, te):
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model
    if te:
        from megatron.core.models.gemma4.gemma4_layer_specs import (
            get_gemma4_layer_with_transformer_engine_spec as spec_fn,
        )
    else:
        from megatron.core.models.gemma4.gemma4_layer_specs import (
            get_gemma4_layer_local_spec as spec_fn,
        )
    spec = spec_fn(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab,
        max_sequence_length=512,
    )
    return model.bfloat16().cuda()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--sequence-parallel", dest="sp", action="store_true")
    ap.add_argument("--te", action="store_true", help="use TE layer spec (default: local)")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--hf-ref", default=DEFAULT_HF_REF)
    ap.add_argument("--out", default=None, help="dump logits + per-layer hs to this .pt")
    ap.add_argument("--capture-layers", action="store_true",
                    help="register hooks to capture per-layer hidden states")
    ap.add_argument("--pad-seq-to", type=int, default=0,
                    help="pad FIXED_TOKENS (with pad id 0) up to this length. SP shards "
                         "the sequence dim, so seq_len must be divisible by TP; pass a "
                         "multiple of the largest TP (e.g. 16 for TP up to 8) for ALL runs "
                         "incl. the TP=1 oracle so inputs are identical. Causal masking "
                         "means trailing pad does not change the real-position logits.")
    args = ap.parse_args()

    assert not (args.sp and args.tp == 1), "--sequence-parallel requires --tp>1"

    from megatron.core import dist_checkpointing

    torch.set_grad_enabled(False)
    rank, world = _init_distributed(args.tp, args.sp)
    assert world == args.tp, f"WORLD_SIZE={world} must equal --tp={args.tp} (PP=CP=DP=1)"

    config = _make_config(args.tp, args.sp)
    model = _build_model(config, vocab=262144, te=args.te)
    model.eval()

    # Capture per-layer hidden states (post-layer output) via forward hooks.
    captured = {}
    if args.capture_layers:
        # Under SP the layer output is sequence-sharded [S/TP,B,H]; gather along seq (dim 0)
        # to recover the full-S [S,B,H] for apples-to-apples parity vs the TP=1 oracle.
        if args.sp:
            from megatron.core.tensor_parallel.mappings import (
                gather_from_sequence_parallel_region,
            )
        layers = model.decoder.layers
        for i, layer in enumerate(layers):
            def hook(mod, inp, out, idx=i):
                # TransformerLayer returns (hidden, context); hidden is [S(/TP), B, H]
                h = out[0] if isinstance(out, tuple) else out
                if args.sp:
                    h = gather_from_sequence_parallel_region(h)
                captured[idx] = h.detach().float().cpu()
            layer.register_forward_hook(hook)

    # Load TP=1 dist-checkpoint, auto-reshard into the TP=N model.
    sharded_sd = model.sharded_state_dict()
    loaded = dist_checkpointing.load(sharded_sd, args.ckpt)
    missing = model.load_state_dict(loaded, strict=False)
    if rank == 0:
        print(f"[rank{rank}] loaded ckpt {args.ckpt} (TP={args.tp} SP={args.sp} TE={args.te})")
        print(f"[rank{rank}] missing={list(missing.missing_keys)[:4]} "
              f"unexpected={list(missing.unexpected_keys)[:4]}")

    toks = list(FIXED_TOKENS)
    if args.pad_seq_to and len(toks) < args.pad_seq_to:
        toks = toks + [0] * (args.pad_seq_to - len(toks))
    ids = torch.tensor([toks], device="cuda")
    logits = model(ids)  # [b, s, V/TP] vocab-PARALLEL (parallel_output=True)

    # The output layer is parallel_output=True -> logits are sharded along vocab
    # ([b,s,V/TP]). For cross-TP parity / greedy we need the FULL vocab logits, so
    # all-gather along the last (vocab) dim across the TP group. No-op at TP=1.
    if args.tp > 1:
        from megatron.core.tensor_parallel.mappings import (
            gather_from_tensor_model_parallel_region,
        )
        logits = gather_from_tensor_model_parallel_region(logits)  # [b,s,V]

    if rank == 0:
        print(f"\nFORWARD OK: logits shape={tuple(logits.shape)} dtype={logits.dtype}")
        print(f"  finite={bool(torch.isfinite(logits).all())} "
              f"min={logits.float().min().item():.4f} max={logits.float().max().item():.4f} "
              f"(softcap bounds +/-30)")
        greedy = logits.argmax(dim=-1).cpu()
        print(f"  greedy = {greedy.tolist()}")

        if args.out:
            blob = {"logits": logits.float().cpu(), "greedy": greedy,
                    "tp": args.tp, "sp": args.sp, "te": args.te}
            if args.capture_layers:
                blob["layers"] = {k: captured[k] for k in sorted(captured)}
            torch.save(blob, args.out)
            print(f"  dumped -> {args.out}")

        if args.tp == 1 and args.hf_ref and os.path.exists(args.hf_ref):
            ref = torch.load(args.hf_ref, map_location="cpu", weights_only=False)
            hf_logits = ref["logits"].float()
            diff = (logits.float().cpu() - hf_logits).abs().max().item()
            hf_greedy = hf_logits.argmax(dim=-1)
            print(f"\nvs HF ref: max-abs-diff={diff:.3e} greedy_exact={torch.equal(greedy, hf_greedy)}")

    torch.distributed.barrier()


if __name__ == "__main__":
    main()
