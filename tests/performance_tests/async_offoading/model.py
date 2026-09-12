# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sample 50-layer GPT-style transformer for activation-offloading performance tests.

The model is deliberately plain: dense GELU MLP, LayerNorm, RoPE, no dropout, no bias.
It runs on a single GPU (TP=PP=1). The Transformer Engine layer spec is used when TE is
importable, otherwise the pure-PyTorch local spec, so the module also works in a bare venv.
Activation recompute and the two existing CPU-offload mechanisms are exposed through
``SampleModelConfig`` so the same model can be benchmarked under each of them.

Example:

    python tests/performance_tests/async_offoading/model.py --recompute full --steps 3
"""

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class SampleModelConfig:
    """Shape and feature knobs for the sample model.

    Defaults give a ~2.6B parameter model on a single 40K-token sample. Without recompute
    the activations of 50 layers at this length do not fit one 80 GB GPU, so the shape
    exercises full block-wise recompute and CPU offloading of the checkpoint inputs, each
    of which is 160 MiB in bf16.
    """

    # Architecture.
    num_layers: int = 50
    hidden_size: int = 2048
    num_attention_heads: int = 16
    ffn_hidden_size: int = 8192
    seq_length: int = 40960
    micro_batch_size: int = 1
    vocab_size: int = 32000
    bf16: bool = True

    # Activation recompute. ``None`` disables it; ``'full'`` checkpoints whole layers and
    # ``'selective'`` recomputes core attention only.
    recompute_granularity: Optional[str] = None
    recompute_method: str = "uniform"
    # Layers per checkpoint chunk (uniform) or number of checkpointed layers (block).
    # ``None`` means one layer per chunk for uniform and every layer for block.
    recompute_num_layers: Optional[int] = None
    distribute_saved_activations: bool = False

    # Transformer Engine per-layer CPU offloading (requires TE). Works alone or together
    # with full block-wise recompute, where the checkpoint inputs are what gets offloaded
    # and reloads start ``cpu_offloading_prefetch_num_layers`` layers ahead in backward.
    cpu_offloading: bool = False
    cpu_offloading_num_layers: int = 0
    cpu_offloading_prefetch_num_layers: int = 1

    # Layer spec: ``None`` picks TE when available, otherwise the local spec.
    use_te: Optional[bool] = None

    seed: int = 1234

    # Any extra TransformerConfig fields, e.g. fine-grained offloading knobs.
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def uses_te(self) -> bool:
        """Whether the TE layer spec will be used."""
        return HAVE_TE if self.use_te is None else self.use_te


def init_single_process_distributed(seed: int = 1234) -> None:
    """Initialize torch.distributed and Megatron model-parallel state for TP=PP=1.

    Honors torchrun environment variables when present so the module also runs under
    ``torch.distributed.run``; otherwise it stands up a one-process NCCL group.
    """
    if not torch.distributed.is_initialized():
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank % torch.cuda.device_count())
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
    model_parallel_cuda_manual_seed(seed)


def build_transformer_config(cfg: SampleModelConfig) -> TransformerConfig:
    """Translate ``SampleModelConfig`` into a ``TransformerConfig``."""
    if cfg.cpu_offloading and not cfg.uses_te:
        raise ValueError("cpu_offloading requires the Transformer Engine layer spec.")

    recompute_num_layers = cfg.recompute_num_layers
    if cfg.recompute_granularity == "full" and recompute_num_layers is None:
        recompute_num_layers = 1 if cfg.recompute_method == "uniform" else cfg.num_layers

    dtype = torch.bfloat16 if cfg.bf16 else torch.float32
    kwargs: Dict[str, Any] = dict(
        num_layers=cfg.num_layers,
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_attention_heads,
        ffn_hidden_size=cfg.ffn_hidden_size,
        # Plain dense GELU block without bias or dropout.
        activation_func=F.gelu,
        gated_linear_unit=False,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        normalization="LayerNorm",
        # Precision. Parameters are created directly in bf16 on the GPU.
        bf16=cfg.bf16,
        params_dtype=dtype,
        use_cpu_initialization=False,
        attention_softmax_in_fp32=True,
        # Recompute.
        recompute_granularity=cfg.recompute_granularity,
        recompute_method=cfg.recompute_method if cfg.recompute_granularity else None,
        recompute_num_layers=recompute_num_layers,
        distribute_saved_activations=cfg.distribute_saved_activations,
        # TE per-layer CPU offloading of saved activations.
        cpu_offloading=cfg.cpu_offloading,
        cpu_offloading_num_layers=cfg.cpu_offloading_num_layers,
        cpu_offloading_prefetch_num_layers=cfg.cpu_offloading_prefetch_num_layers,
    )
    kwargs.update(cfg.extra_config)
    return TransformerConfig(**kwargs)


def build_layer_spec(cfg: SampleModelConfig) -> ModuleSpec:
    """Pick the TE or local transformer-layer spec."""
    if cfg.uses_te:
        return get_gpt_layer_with_transformer_engine_spec()
    return get_gpt_layer_local_spec()


def build_model(cfg: SampleModelConfig) -> GPTModel:
    """Build the sample GPT model on the current CUDA device in training mode.

    ``init_single_process_distributed`` must have been called first.
    """
    config = build_transformer_config(cfg)
    model = GPTModel(
        config=config,
        transformer_layer_spec=build_layer_spec(cfg),
        vocab_size=cfg.vocab_size,
        max_sequence_length=cfg.seq_length,
        position_embedding_type="rope",
        share_embeddings_and_output_weights=True,
    )
    model.cuda(torch.cuda.current_device())
    model.train()
    return model


def build_random_batch(cfg: SampleModelConfig, device: Optional[torch.device] = None):
    """Random token batch plus causal mask, in the layout ``GPTModel.forward`` expects."""
    device = device or torch.device("cuda", torch.cuda.current_device())
    b, s = cfg.micro_batch_size, cfg.seq_length
    tokens = torch.randint(0, cfg.vocab_size, (b, s), device=device, dtype=torch.long)
    labels = torch.roll(tokens, shifts=-1, dims=1)
    position_ids = torch.arange(s, device=device, dtype=torch.long).unsqueeze(0).expand(b, s)
    if cfg.uses_te:
        # The GPT spec uses a causal mask type, which TE's fused attention applies in-kernel
        # and for which it ignores an explicit mask. Skip building the [s, s] tensor, which
        # would be 1.6 GB at 40K tokens.
        attention_mask = None
    else:
        # True marks positions that may NOT be attended to.
        attention_mask = torch.triu(torch.ones(s, s, device=device, dtype=torch.bool), diagonal=1)
        attention_mask = attention_mask.view(1, 1, s, s)
    return dict(
        input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask, labels=labels
    )


def count_parameters(model: torch.nn.Module) -> int:
    """Total parameter count."""
    return sum(p.numel() for p in model.parameters())


def train_step(model: GPTModel, batch) -> torch.Tensor:
    """One forward/backward pass; returns the scalar loss."""
    per_token_loss = model(**batch)
    loss = per_token_loss.float().mean()
    loss.backward()
    return loss.detach()


def _parse_args() -> tuple[SampleModelConfig, int]:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--num-layers", type=int, default=SampleModelConfig.num_layers)
    parser.add_argument("--hidden-size", type=int, default=SampleModelConfig.hidden_size)
    parser.add_argument(
        "--num-attention-heads", type=int, default=SampleModelConfig.num_attention_heads
    )
    parser.add_argument("--ffn-hidden-size", type=int, default=SampleModelConfig.ffn_hidden_size)
    parser.add_argument("--seq-length", type=int, default=SampleModelConfig.seq_length)
    parser.add_argument("--micro-batch-size", type=int, default=SampleModelConfig.micro_batch_size)
    parser.add_argument("--recompute", choices=["none", "full", "selective"], default="full")
    parser.add_argument("--recompute-method", choices=["uniform", "block"], default="block")
    parser.add_argument("--recompute-num-layers", type=int, default=None)
    parser.add_argument("--cpu-offloading", action="store_true")
    parser.add_argument("--cpu-offloading-num-layers", type=int, default=0)
    parser.add_argument(
        "--cpu-offloading-prefetch-num-layers",
        type=int,
        default=1,
        help="Layers ahead in backward at which an offloaded layer's reload starts.",
    )
    parser.add_argument("--no-te", action="store_true", help="Force the local layer spec.")
    parser.add_argument("--steps", type=int, default=2, help="Timed steps after one warmup.")
    args = parser.parse_args()
    cfg = SampleModelConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        recompute_granularity=None if args.recompute == "none" else args.recompute,
        recompute_method=args.recompute_method,
        recompute_num_layers=args.recompute_num_layers,
        cpu_offloading=args.cpu_offloading,
        cpu_offloading_num_layers=args.cpu_offloading_num_layers,
        cpu_offloading_prefetch_num_layers=args.cpu_offloading_prefetch_num_layers,
        use_te=False if args.no_te else None,
    )
    return cfg, args.steps


def main() -> None:
    """Build the model and time a few forward/backward steps."""
    cfg, steps = _parse_args()
    init_single_process_distributed(cfg.seed)
    model = build_model(cfg)
    batch = build_random_batch(cfg)
    print(
        f"layers={cfg.num_layers} hidden={cfg.hidden_size} seq={cfg.seq_length} "
        f"mbs={cfg.micro_batch_size} spec={'te' if cfg.uses_te else 'local'} "
        f"recompute={cfg.recompute_granularity} params={count_parameters(model) / 1e9:.2f}B"
    )

    # Warmup step, then timed steps.
    train_step(model, batch)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        loss = train_step(model, batch)
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_ms = (time.perf_counter() - start) / steps * 1e3
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(f"loss={loss.item():.4f} step={step_ms:.1f} ms peak_alloc={peak_gib:.2f} GiB")

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
