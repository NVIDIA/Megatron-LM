# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameters, FLOPs and memory per pipeline unit for a GPT/MoE model shape."""

import json
import shlex
from dataclasses import asdict, dataclass, field
from typing import List, Tuple, Union


@dataclass
class ModelShape:
    """Model and batch shape the planner needs; a subset of Megatron's training arguments."""

    num_layers: int = 16
    """Number of decoder layers (``--num-layers``)."""

    hidden_size: int = 2048
    """Hidden size (``--hidden-size``)."""

    ffn_hidden_size: int = 8192
    """Dense FFN width (``--ffn-hidden-size``), used by dense layers."""

    num_experts: int = 0
    """Number of routed experts (``--num-experts``); 0 for a dense model."""

    moe_ffn_hidden_size: int = 1024
    """Width of one routed expert (``--moe-ffn-hidden-size``)."""

    moe_router_topk: int = 2
    """Experts per token (``--moe-router-topk``)."""

    moe_shared_expert_intermediate_size: int = 0
    """Shared expert width (``--moe-shared-expert-intermediate-size``); 0 for none."""

    num_attention_heads: int = 32
    """Attention heads (``--num-attention-heads``)."""

    num_query_groups: int = 32
    """Key/value groups (``--num-query-groups``); equal to heads without GQA."""

    kv_channels: int = 64
    """Channels per head (``--kv-channels``)."""

    vocab_size: int = 128256
    """Padded vocabulary size."""

    seq_length: int = 4096
    """Sequence length (``--seq-length``)."""

    micro_batch_size: int = 1
    """Micro-batch size (``--micro-batch-size``)."""

    global_batch_size: int = 32
    """Global batch size (``--global-batch-size``)."""

    mtp_num_layers: int = 0
    """Number of MTP layers (``--mtp-num-layers``)."""

    data_parallel_size: int = 1
    """Data-parallel size; the distributed optimizer shards optimizer state over it."""

    full_recompute: bool = False
    """Whether ``--recompute-granularity full`` is used."""

    moe_layer_freq: List[int] = field(default_factory=list)
    """Per-layer MoE pattern (1 = MoE, 0 = dense); empty means every layer is MoE."""

    # Calibrated against Megatron's "max allocated" on 40 measured pipeline stages (A100 and H200,
    # BF16, full recompute, micro-batch 1): weights and optimizer state, plus about 10.3 bytes per
    # logit per output head on the stage, plus 0.13 GB. Mean error 2.8 percent; at micro-batch 4
    # the logit term over-estimates (conservative).
    logit_bytes: float = 10.3
    """Bytes per logit per output head used by the memory estimate."""

    memory_overhead_gb: float = 0.13
    """Constant per-stage memory added by the memory estimate."""

    @property
    def num_microbatches(self) -> int:
        """Micro-batches per step on each pipeline."""
        return self.global_batch_size // (self.micro_batch_size * self.data_parallel_size)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Return whether decoder layer ``layer_idx`` is a MoE layer."""
        if self.num_experts == 0:
            return False
        return bool(self.moe_layer_freq[layer_idx]) if self.moe_layer_freq else True

    def _attention_params(self) -> int:
        q_dim = self.num_attention_heads * self.kv_channels
        kv_dim = 2 * self.num_query_groups * self.kv_channels
        return self.hidden_size * (q_dim + kv_dim) + q_dim * self.hidden_size + 2 * self.hidden_size

    def _attention_macs_per_token(self) -> float:
        q_dim = self.num_attention_heads * self.kv_channels
        kv_dim = 2 * self.num_query_groups * self.kv_channels
        projections = self.hidden_size * (q_dim + kv_dim) + q_dim * self.hidden_size
        core = self.seq_length * q_dim  # QK^T and AV with causal masking
        return projections + core

    def _mlp_params(self, moe: bool) -> int:
        if not moe:
            return 3 * self.hidden_size * self.ffn_hidden_size
        params = self.num_experts * 3 * self.hidden_size * self.moe_ffn_hidden_size
        params += self.hidden_size * self.num_experts
        params += 3 * self.hidden_size * self.moe_shared_expert_intermediate_size
        return params

    def _mlp_macs_per_token(self, moe: bool) -> float:
        if not moe:
            return 3 * self.hidden_size * self.ffn_hidden_size
        macs = self.moe_router_topk * 3 * self.hidden_size * self.moe_ffn_hidden_size
        macs += self.hidden_size * self.num_experts
        macs += 3 * self.hidden_size * self.moe_shared_expert_intermediate_size
        return macs

    def unit_params(self, unit: str, layer_idx: int = 0) -> int:
        """Return the parameters held by one unit (``E``, ``t``, ``m`` or ``L``)."""
        hidden, vocab = self.hidden_size, self.vocab_size
        if unit == "E":
            return vocab * hidden
        if unit == "t":
            return self._attention_params() + self._mlp_params(self.is_moe_layer(layer_idx))
        if unit == "m":
            last_moe = self.is_moe_layer(self.num_layers - 1)
            return (
                self._attention_params()
                + self._mlp_params(last_moe)
                + 2 * hidden * hidden
                + 2 * hidden
            )
        if unit == "L":
            # The last stage keeps its own output-layer weights even with tied embeddings.
            return vocab * hidden + hidden
        raise ValueError(f"Unknown unit {unit!r}")

    def unit_macs_per_token(self, unit: str, layer_idx: int = 0) -> float:
        """Return the forward multiply-accumulates per token of one unit."""
        hidden, vocab = self.hidden_size, self.vocab_size
        if unit == "E":
            return 0.0
        if unit == "t":
            return self._attention_macs_per_token() + self._mlp_macs_per_token(
                self.is_moe_layer(layer_idx)
            )
        if unit == "m":
            last_moe = self.is_moe_layer(self.num_layers - 1)
            return (
                self._attention_macs_per_token()
                + self._mlp_macs_per_token(last_moe)
                + 2 * hidden * hidden
                + hidden * vocab
            )
        if unit == "L":
            return float(hidden * vocab)
        raise ValueError(f"Unknown unit {unit!r}")

    def unit_flops_per_microbatch(self, unit: str, layer_idx: int = 0) -> Tuple[float, float]:
        """Return (forward, backward) FLOPs of one unit for one micro-batch.

        With full recompute, transformer layers also pay a recompute forward inside backward.
        """
        tokens = self.seq_length * self.micro_batch_size
        macs = self.unit_macs_per_token(unit, layer_idx) * tokens
        forward, backward = 2 * macs, 4 * macs
        if self.full_recompute and unit == "t":
            backward += 2 * macs
        elif self.full_recompute and unit == "m":
            backward += 2 * (macs - self.hidden_size * self.vocab_size * tokens)
        return forward, backward

    def stage_params(self, stage: List[str], first_layer: int = 0) -> int:
        """Return the parameters of a stage whose first decoder layer is ``first_layer``."""
        total, layer_idx = 0, first_layer
        for unit in stage:
            total += self.unit_params(unit, layer_idx)
            if unit == "t":
                layer_idx += 1
        return total

    def stage_memory_gb(self, stage: List[str], first_layer: int = 0) -> float:
        """Estimate Megatron's peak "max allocated" for one pipeline stage, in GB.

        Counts bf16 weights, fp32 gradients, fp32 master weights and Adam state (sharded over
        data-parallel ranks by the distributed optimizer) and the logits of each output head on
        the stage. Activations under full recompute are small at the calibrated settings and are
        not modeled separately.
        """
        params = self.stage_params(stage, first_layer)
        static_bytes = params * (2 + 4 + 12 / max(1, self.data_parallel_size))
        tokens = self.seq_length * self.micro_batch_size
        heads = stage.count("L") + stage.count("m")
        logit_bytes = tokens * self.vocab_size * self.logit_bytes * heads
        return (static_bytes + logit_bytes) / 1e9 + self.memory_overhead_gb

    def total_params(self) -> int:
        """Return total parameters (embedding, decoder layers, MTP layers, final norm)."""
        total = self.unit_params("E") + self.hidden_size
        total += sum(self.unit_params("t", idx) for idx in range(self.num_layers))
        total += self.mtp_num_layers * self.unit_params("m")
        return total

    def active_params(self) -> int:
        """Return parameters active per token (top-k routed experts only)."""
        total = self.vocab_size * self.hidden_size + self.hidden_size
        for idx in range(self.num_layers):
            total += self._attention_params()
            if self.is_moe_layer(idx):
                total += self.moe_router_topk * 3 * self.hidden_size * self.moe_ffn_hidden_size
                total += self.hidden_size * self.num_experts
                total += 3 * self.hidden_size * self.moe_shared_expert_intermediate_size
            else:
                total += 3 * self.hidden_size * self.ffn_hidden_size
        return total

    def to_json(self) -> str:
        """Return the shape as JSON."""
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(path: str) -> "ModelShape":
        """Load a shape from a JSON file of :class:`ModelShape` fields."""
        with open(path) as handle:
            return ModelShape(**json.load(handle))


_INT_FLAGS = {
    "--num-layers": "num_layers",
    "--hidden-size": "hidden_size",
    "--ffn-hidden-size": "ffn_hidden_size",
    "--num-experts": "num_experts",
    "--moe-ffn-hidden-size": "moe_ffn_hidden_size",
    "--moe-router-topk": "moe_router_topk",
    "--moe-shared-expert-intermediate-size": "moe_shared_expert_intermediate_size",
    "--num-attention-heads": "num_attention_heads",
    "--num-query-groups": "num_query_groups",
    "--kv-channels": "kv_channels",
    "--seq-length": "seq_length",
    "--micro-batch-size": "micro_batch_size",
    "--global-batch-size": "global_batch_size",
    "--mtp-num-layers": "mtp_num_layers",
}


def shape_from_megatron_args(
    args: Union[str, List[str]], vocab_size: int = 128256, data_parallel_size: int = 1
) -> ModelShape:
    """Build a :class:`ModelShape` from a Megatron training command line.

    Unknown flags are ignored. Without ``--num-query-groups`` the model is assumed to use
    multi-head attention.

    Args:
        args: The command line as a string or a list of tokens.
        vocab_size: Padded vocabulary size (not derivable from the flags alone).
        data_parallel_size: Data-parallel size of the planned run.

    Returns:
        The model shape.
    """
    tokens = shlex.split(args) if isinstance(args, str) else list(args)
    shape = ModelShape(vocab_size=vocab_size, data_parallel_size=data_parallel_size)
    seen_query_groups = False
    idx = 0
    while idx < len(tokens):
        flag = tokens[idx]
        if flag in _INT_FLAGS and idx + 1 < len(tokens):
            setattr(shape, _INT_FLAGS[flag], int(tokens[idx + 1]))
            seen_query_groups |= flag == "--num-query-groups"
            idx += 2
            continue
        if flag == "--recompute-granularity" and idx + 1 < len(tokens):
            shape.full_recompute = tokens[idx + 1] == "full"
        idx += 1
    if not seen_query_groups:
        shape.num_query_groups = shape.num_attention_heads
    return shape
