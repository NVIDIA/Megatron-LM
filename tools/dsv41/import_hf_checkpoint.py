# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Import the released DeepSeek-V4.1-Flash weights into a Megatron distributed checkpoint.

The script builds the training model exactly as ``pretrain_hybrid.py`` does (same argument
parsing, same ``hybrid_builder``, same parallel state), fills every parameter of the local
model chunk from the Hugging Face safetensors shards (``real_weights``: FP8 / FP4 / Engram
dequantisation, expert-parallel and pipeline placement) and writes iteration 0 with
Megatron's own ``save_checkpoint``. A training run then loads it with ``--load <dir>
--no-load-optim --no-load-rng --finetune`` at the same or a compatible topology.

Vision, MTP and DSpark tensors of the snapshot are never read (the text model has no
parameter for them); their counts are recorded in ``IMPORT_METADATA.txt`` together with the
snapshot commit, the Megatron commit and the import topology (``key=value`` lines, one per
field) so a training launcher can verify that a checkpoint matches the code it runs with.

Run inside the training container with the full model arguments of the recipe plus::

    torchrun ... tools/dsv41/import_hf_checkpoint.py <model args> \
        --dsv41-hf-snapshot $SNAP --save $OUT --no-save-optim --no-save-rng \
        --engram-token-map-path $TOKEN_MAP --tokenizer-type HuggingFaceTokenizer ...

Independent implementation.
"""

import json
import os
import re
import sys
import time
from functools import partial

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)  # hybrid_builders / model_provider live at the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_weights as rw  # noqa: E402

from hybrid_builders import hybrid_builder  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.enums import ModelType  # noqa: E402
from megatron.core.transformer.experimental_attention_variant.csa2.roles import (  # noqa: E402
    is_attention_position,
    model_layer_id_from_layer_number,
)
from megatron.training import get_args, initialize_megatron, print_rank_0  # noqa: E402
from megatron.training.arguments import parse_and_validate_args  # noqa: E402
from megatron.training.checkpointing import save_checkpoint  # noqa: E402
from megatron.training.training import get_model  # noqa: E402
from megatron.training.utils import unwrap_model  # noqa: E402
from model_provider import model_provider  # noqa: E402

_DROPPED_FAMILIES = ("vision.", "aligner.", "image_", "mtp.", "dspark")


def add_import_args(parser):
    group = parser.add_argument_group("dsv41 import")
    group.add_argument("--dsv41-hf-snapshot", required=True, help="Hugging Face snapshot dir")
    group.add_argument("--dsv41-hf-repo", default="deepseek-ai/DeepSeek-V4.1-Flash")
    group.add_argument(
        "--dsv41-import-metadata",
        default=None,
        help="path of IMPORT_METADATA.txt (default: <save>/IMPORT_METADATA.txt)",
    )
    return parser


@torch.no_grad()
def load_chunk(chunk, snap: rw.Snapshot, log) -> int:
    """Fill one pipeline-stage model chunk from the snapshot. Returns the tensor count."""
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    params = dict(chunk.named_parameters())
    used = set()
    n_experts = int(snap.text_config["n_routed_experts"])
    n_local = n_experts // ep_size

    def put(name: str, value: torch.Tensor) -> None:
        if name not in params:
            raise KeyError(f"parameter missing on this rank: {name}")
        rw._assign(params[name], value, name)
        used.add(name)

    if chunk.pre_process:
        put(
            "embedding.word_embeddings.weight",
            _pad_vocab(snap.dequant("embed.weight"), params["embedding.word_embeddings.weight"]),
        )
    if chunk.post_process:
        put(
            "output_layer.weight",
            _pad_vocab(snap.dequant("head.weight"), params["output_layer.weight"]),
        )
        put("decoder.final_norm.weight", snap.dequant("norm.weight"))

    for local, layer in enumerate(chunk.decoder.layers):
        number = layer.layer_number  # global 1-based pattern position
        i = model_layer_id_from_layer_number(number)
        mg = f"decoder.layers.{local}"
        inner = f"{mg}.inner_layer"
        if is_attention_position(number):
            src_hc = f"layers.{i}.hc_attn"
        else:
            src_hc = f"layers.{i}.hc_ffn"
        put(f"{mg}.hyper_connection.mapping_proj.weight", snap.dequant(f"{src_hc}_fn"))
        put(f"{mg}.hyper_connection.bias", snap.dequant(f"{src_hc}_base"))
        scale = snap.dequant(f"{src_hc}_scale")
        put(f"{mg}.hyper_connection.alpha_pre", scale[0:1])
        put(f"{mg}.hyper_connection.alpha_post", scale[1:2])
        put(f"{mg}.hyper_connection.alpha_res", scale[2:3])

        if is_attention_position(number):
            a = f"layers.{i}.attn"
            put(f"{inner}.input_layernorm.weight", snap.dequant(f"layers.{i}.attn_norm.weight"))
            sa = f"{inner}.self_attention"
            put(f"{sa}.linear_q_down_proj.weight", snap.dequant(f"{a}.wq_a.weight"))
            put(f"{sa}.q_layernorm.weight", snap.dequant(f"{a}.q_norm.weight"))
            put(f"{sa}.linear_q_up_proj.weight", snap.dequant(f"{a}.wq_b.weight"))
            put(f"{sa}.linear_kv_proj.weight", snap.dequant(f"{a}.wkv.weight"))
            put(f"{sa}.kv_layernorm.weight", snap.dequant(f"{a}.kv_norm.weight"))
            put(f"{sa}.linear_o_group_proj", snap.dequant(f"{a}.wo_a.weight"))
            put(f"{sa}.linear_proj.weight", snap.dequant(f"{a}.wo_b.weight"))
            core = f"{sa}.core_attention"
            put(f"{core}.attn_sink", snap.dequant(f"{a}.attn_sink"))
            if snap.has(f"{a}.compressor.wkv.weight"):
                put(
                    f"{core}.compressor.linear_wkv.weight",
                    snap.dequant(f"{a}.compressor.wkv.weight"),
                )
                put(f"{core}.compressor.norm.weight", snap.dequant(f"{a}.compressor.norm.weight"))
                if snap.has(f"{a}.compressor.wgate.weight"):
                    put(
                        f"{core}.compressor.linear_wgate.weight",
                        snap.dequant(f"{a}.compressor.wgate.weight"),
                    )
            if snap.has(f"{a}.indexer.wq_b.weight"):
                put(f"{core}.indexer.linear_wq_b.weight", snap.dequant(f"{a}.indexer.wq_b.weight"))
                put(
                    f"{core}.indexer.linear_weights_proj.weight",
                    snap.dequant(f"{a}.indexer.weights_proj.weight"),
                )
                if snap.has(f"{a}.indexer.wk.weight"):
                    put(f"{core}.indexer.linear_wk.weight", snap.dequant(f"{a}.indexer.wk.weight"))
                    put(f"{core}.indexer.k_norm.weight", snap.dequant(f"{a}.indexer.k_norm.weight"))
            if getattr(layer, "engram", None) is not None:
                eg = f"{mg}.engram"
                target = params[f"{eg}.embedding_rows"]
                rows_per_shard = target.shape[0]
                shard_rank = layer.engram.shard_rank
                r0 = shard_rank * rows_per_shard
                weight_slice = snap.tensor(
                    f"layers.{i}.engram.embed.weight", rows=(r0, r0 + rows_per_shard)
                )
                scale_slice = snap.tensor(
                    f"layers.{i}.engram.embed.scale", rows=(r0, r0 + rows_per_shard)
                )
                table = torch.zeros(rows_per_shard, target.shape[1], dtype=torch.bfloat16)
                chunk_rows = 8_000_000
                for c0 in range(0, weight_slice.shape[0], chunk_rows):
                    w = weight_slice[c0 : c0 + chunk_rows].to(target.device)
                    s = scale_slice[c0 : c0 + chunk_rows].to(target.device)
                    table[c0 : c0 + w.shape[0]] = rw.dequant_fp8_rows(w, s).cpu()
                put(f"{eg}.embedding_rows", table)
                put(f"{eg}.linear_wkv.weight", snap.dequant(f"layers.{i}.engram.wkv.weight"))
                put(f"{eg}.q_weight", snap.dequant(f"layers.{i}.engram.q_weight"))
                put(f"{eg}.k_weight", snap.dequant(f"layers.{i}.engram.k_weight"))
        else:
            f_ = f"layers.{i}.ffn"
            put(f"{inner}.pre_mlp_layernorm.weight", snap.dequant(f"layers.{i}.ffn_norm.weight"))
            mlp = f"{inner}.mlp"
            put(f"{mlp}.router.weight", snap.dequant(f"{f_}.gate.weight"))
            bias_name = f"{mlp}.router.expert_bias"
            if bias_name in params:
                put(bias_name, snap.dequant(f"{f_}.gate.bias"))
            else:  # registered as a buffer
                buffers = dict(chunk.named_buffers())
                rw._assign(buffers[bias_name], snap.dequant(f"{f_}.gate.bias"), bias_name)
            grouped = f"{mlp}.experts.linear_fc1.weight0" in params
            for local_e in range(n_local):
                e = ep_rank * n_local + local_e
                w1 = snap.dequant(f"{f_}.experts.{e}.w1.weight")
                w3 = snap.dequant(f"{f_}.experts.{e}.w3.weight")
                w2 = snap.dequant(f"{f_}.experts.{e}.w2.weight")
                if grouped:
                    put(f"{mlp}.experts.linear_fc1.weight{local_e}", torch.cat([w1, w3], dim=0))
                    put(f"{mlp}.experts.linear_fc2.weight{local_e}", w2)
                else:
                    put(
                        f"{mlp}.experts.local_experts.{local_e}.linear_fc1.weight",
                        torch.cat([w1, w3], dim=0),
                    )
                    put(f"{mlp}.experts.local_experts.{local_e}.linear_fc2.weight", w2)
            w1 = snap.dequant(f"{f_}.shared_experts.w1.weight")
            w3 = snap.dequant(f"{f_}.shared_experts.w3.weight")
            put(f"{mlp}.shared_experts.linear_fc1.weight", torch.cat([w1, w3], dim=0))
            put(
                f"{mlp}.shared_experts.linear_fc2.weight",
                snap.dequant(f"{f_}.shared_experts.w2.weight"),
            )

    missing = sorted(k for k in params if k not in used)
    if missing:
        raise RuntimeError(
            f"parameters not filled on this rank: {missing[:20]} (+{max(0, len(missing) - 20)})"
        )
    return len(used)


def _pad_vocab(weight: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """Vocabulary-padded embedding / head rows (Megatron pads to the divisor; extra rows zero)."""
    if weight.shape == param.shape:
        return weight
    if weight.shape[0] > param.shape[0] or weight.shape[1:] != param.shape[1:]:
        raise ValueError(f"vocab tensor {tuple(weight.shape)} does not fit {tuple(param.shape)}")
    out = torch.zeros(param.shape, dtype=weight.dtype)
    out[: weight.shape[0]] = weight
    return out


def _snapshot_summary(snap: rw.Snapshot) -> dict:
    names = list(snap.weight_map)
    dropped = {
        fam: sum(1 for n in names if n.startswith(fam) or (fam == "dspark" and "dspark" in n))
        for fam in _DROPPED_FAMILIES
    }
    text = [
        n
        for n in names
        if not any(n.startswith(f) for f in ("vision.", "aligner.", "image_", "mtp."))
    ]
    return {
        "total_tensors": len(names),
        "dropped_tensors": {k: v for k, v in dropped.items() if v},
        "text_tensors": len(text),
        "text_layers": len(
            {m.group(1) for n in text for m in [re.match(r"layers\.(\d+)\.", n)] if m}
        ),
    }


def main() -> None:
    args = parse_and_validate_args(extra_args_provider=add_import_args)
    initialize_megatron()
    args = get_args()
    if not args.save:
        raise SystemExit("--save is required")
    if not args.no_save_optim or not args.no_save_rng:
        raise SystemExit("pass --no-save-optim --no-save-rng: the import writes model weights only")

    rank = torch.distributed.get_rank()
    t0 = time.time()
    snap = rw.Snapshot.open(args.dsv41_hf_snapshot)
    model = get_model(
        partial(model_provider, hybrid_builder), ModelType.encoder_or_decoder, wrap_with_ddp=False
    )
    print_rank_0(f"model built in {time.time() - t0:.0f}s; loading weights ...")
    t1 = time.time()
    # get_model wraps every chunk (Float16Module); fill the bare HybridModel underneath.
    counts = [load_chunk(unwrap_model(chunk), snap, print_rank_0) for chunk in model]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    print_rank_0(f"weights loaded in {time.time() - t1:.0f}s ({sum(counts)} tensors on rank 0)")

    t2 = time.time()
    save_checkpoint(0, model, None, None, 0)
    torch.distributed.barrier()
    print_rank_0(f"checkpoint saved in {time.time() - t2:.0f}s to {args.save}")

    if rank == 0:
        commit = os.popen(f"git -C {_REPO_ROOT} rev-parse HEAD").read().strip()
        summary = _snapshot_summary(snap)
        snapshot_commit = os.path.basename(args.dsv41_hf_snapshot.rstrip("/")).split("@")[-1]
        lines = [
            f"hf_repo={args.dsv41_hf_repo}",
            f"hf_commit={snapshot_commit}",
            f"hf_snapshot={args.dsv41_hf_snapshot}",
            "derived_snapshot=text-only (vision/MTP/DSpark tensors never read)",
            f"dropped_tensors={json.dumps(summary['dropped_tensors'], sort_keys=True)}",
            f"kept_tensors={summary['text_tensors']}",
            f"text_layers={summary['text_layers']}",
            f"importer=tools/dsv41/import_hf_checkpoint.py",
            f"mcore_commit={commit}",
            "dtype=bfloat16",
            f"import_topology=tp{args.tensor_model_parallel_size}_pp{args.pipeline_model_parallel_size}"
            f"_ep{args.expert_model_parallel_size}_etp{args.expert_tensor_parallel_size or args.tensor_model_parallel_size}",
            f"hybrid_layer_pattern={args.hybrid_layer_pattern}",
            f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}",
            f"imported_by_user={os.environ.get('USER', '')}",
        ]
        path = args.dsv41_import_metadata or os.path.join(args.save, "IMPORT_METADATA.txt")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(
            "DSV41_IMPORT_METADATA " + json.dumps(dict(l.split("=", 1) for l in lines)), flush=True
        )
    torch.distributed.barrier()


if __name__ == "__main__":
    main()
