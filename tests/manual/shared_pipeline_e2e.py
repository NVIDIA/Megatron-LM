# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Train identical full and pipeline attention models through production stack loops."""

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.hybrid.hybrid_block import HybridStack, HybridStackSubmodules
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelinePayload,
    PipelinePayloadPlan,
    PipelinePayloadSpec,
    PipelineTensorSpec,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.pipeline_parallel.typed_p2p_communication import TypedP2PCommunicator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig


class SharedAttention(SelfAttention):
    def forward(self, hidden_states, *, shared_state, **kwargs):
        return super().forward(
            hidden_states + shared_state.to(hidden_states.dtype) * 0.25, **kwargs
        )


@dataclasses.dataclass
class SharedState:
    tensor: torch.Tensor

    def attention_kwargs(self, layer_number: int) -> dict[str, torch.Tensor]:
        return {"shared_state": self.tensor}


@dataclasses.dataclass
class Payload(PipelinePayload):
    tensors: tuple[torch.Tensor, ...]
    spec: PipelinePayloadSpec

    @property
    def tensor_specs(self):
        return self.spec.tensor_specs

    @property
    def metadata(self):
        return self.spec.metadata

    @property
    def boundary_id(self):
        return self.spec.boundary_id


def descriptor(dtype, length):
    shape = (length, 1, 64)
    fields = (
        TensorField("hidden", shape, dtype, "SBH", True),
        TensorField("shared", shape, torch.float32, "SBH", True),
        TensorField("unused", shape, torch.float32, "SBH", True),
        TensorField("zero", shape, torch.float32, "SBH", True),
        TensorField("position", (length,), torch.int64, "S", False),
        TensorField("absent", shape, torch.float32, "SBH", True, False),
    )
    return PipelinePayloadSpec(
        tuple(PipelineTensorSpec.from_field(f) for f in fields), (), "shared-v1"
    )


class Model(torch.nn.Module):
    def __init__(self, stack, dtype, groups, stages, rank):
        super().__init__()
        self.model_type = ModelType.encoder_or_decoder
        self.config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            pipeline_model_parallel_size=stages,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            bf16=dtype == torch.bfloat16,
            use_cpu_initialization=True,
            add_bias_linear=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            gradient_accumulation_fusion=False,
            bias_activation_fusion=False,
            bias_dropout_fusion=False,
            masked_softmax_fusion=False,
            deallocate_pipeline_outputs=True,
        )
        self.first, self.last = rank == 0, rank == stages - 1
        self.stack_kind = stack
        if self.first:
            self.embedding = torch.nn.Embedding(128, 64, dtype=dtype)
            self.producer = torch.nn.Linear(64, 64, bias=False, dtype=dtype)
        if self.last:
            self.head = torch.nn.Linear(64, 128, bias=False, dtype=dtype)
        spec = get_gpt_layer_with_transformer_engine_spec()
        spec.submodules.self_attention.module = SharedAttention
        if stack == "transformer":
            self.stack = TransformerBlock(
                self.config,
                spec,
                pre_process=True,
                post_process=False,
                pg_collection=groups,
            )
        else:
            self.stack = HybridStack(
                self.config,
                HybridStackSubmodules(attention_layer=spec),
                pre_process=True,
                post_process=False,
                post_layer_norm=False,
                layer_type_list=["*"] * (4 // stages),
                pp_layer_offset=rank * (4 // stages),
                pg_collection=groups,
            )
        self.input_payload = None
        self.cuda()
        with torch.no_grad():
            for name, param in self.canonical_parameters().items():
                seed = int.from_bytes(
                    hashlib.sha256(name.encode()).digest()[:4], "little"
                )
                generator = torch.Generator(device="cuda").manual_seed(seed)
                if "layer_norm_weight" in name or "layernorm.weight" in name:
                    param.fill_(1)
                else:
                    param.copy_(
                        torch.randn(param.shape, device="cuda", generator=generator)
                        * 0.02
                    )

    def canonical_parameters(self):
        result = {}
        for name, param in self.named_parameters():
            if name.startswith("stack.layers."):
                fields = name.split(".")
                fields[2] = str(self.stack.layers[int(fields[2])].layer_number)
                name = ".".join(fields)
            result[name] = param
        return result

    def set_input_tensor(self, tensors):
        self.input_payload = tensors[0]

    def forward(self, tokens):
        if self.first:
            hidden = self.embedding(tokens).transpose(0, 1)
            shared = self.producer(hidden).float()
            unused, zero = shared * 2, shared * 3
            position = torch.arange(tokens.shape[1], device="cuda")
        else:
            hidden, shared, unused, zero, position = self.input_payload.tensors
            self.input_payload = None
        state = SharedState(shared)
        if self.stack_kind == "transformer":
            output = self.stack(hidden, None, cross_layer_state=state)
        else:
            output = self.stack(hidden, None, cross_layer_state=state)
        if self.last:
            return self.head(output + zero.to(output.dtype) * 0)
        return Payload(
            (output, shared, unused, zero, position),
            descriptor(output.dtype, tokens.shape[1]),
        )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stack", choices=["transformer", "hybrid"], default="transformer")
parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--microbatches", type=int, default=4)
parser.add_argument("--stages", type=int, choices=[2, 4], default=2)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
rank = int(os.environ["RANK"])
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
dist.init_process_group("nccl", timeout=timedelta(seconds=120))
parallel_state.initialize_model_parallel(
    pipeline_model_parallel_size=args.stages, create_gloo_process_groups=False
)
model_parallel_cuda_manual_seed(20260922)
groups = ProcessGroupCollection.use_mpu_process_groups()
local_groups = copy.copy(groups)
local_groups.pp = groups.tp
dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
full = Model(args.stack, dtype, local_groups, 1, 0)
partition = Model(args.stack, dtype, groups, args.stages, rank)
for name, param in partition.canonical_parameters().items():
    torch.testing.assert_close(param, full.canonical_parameters()[name], rtol=0, atol=0)
full_opt = torch.optim.SGD(full.parameters(), lr=0.02)
part_opt = torch.optim.SGD(partition.parameters(), lr=0.02)
generator = torch.Generator().manual_seed(42)
batches = [
    torch.randint(0, 128, (1, length), generator=generator).cuda()
    for length in (8, 12, 8, 16)[: args.microbatches]
]
plans = tuple(descriptor(dtype, b.shape[1]) for b in batches)
plan = PipelinePayloadPlan(
    (None,) * len(batches) if rank == 0 else plans,
    plans if rank < args.stages - 1 else (None,) * len(batches),
)


def loss_for(logits, tokens):
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1).float(), tokens.T.flatten()
    )


def forward_step(iterator, model):
    tokens = next(iterator)
    output = model(tokens)

    def loss_func(logits):
        loss = loss_for(logits, tokens)
        return loss, {"loss": loss.detach().clone()}

    return output, loss_func


results = []
for step in range(args.steps):
    full_opt.zero_grad(set_to_none=True)
    part_opt.zero_grad(set_to_none=True)
    reference_losses = []
    for tokens in batches:
        loss = loss_for(full(tokens), tokens)
        reference_losses.append(float(loss.detach()))
        (loss / len(batches)).backward()
    communicator = TypedP2PCommunicator(
        groups.pp, partition.config, Payload, payload_plan=plan
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    losses = forward_backward_pipelining_without_interleaving(
        forward_step_func=forward_step,
        data_iterator=iter(batches),
        model=partition,
        num_microbatches=len(batches),
        seq_length=16,
        micro_batch_size=1,
        p2p_communicator=communicator,
        pg_collection=groups,
        forward_only=False,
    )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    errors = {}
    for name, param in partition.canonical_parameters().items():
        expected = full.canonical_parameters()[name]
        assert (param.grad is None) == (expected.grad is None), name
        if param.grad is not None:
            errors[name] = float(
                (param.grad.float() - expected.grad.float()).abs().max()
            )
            torch.testing.assert_close(
                param.grad,
                expected.grad,
                rtol=1e-4 if dtype == torch.float32 else 0.03,
                atol=1e-6 if dtype == torch.float32 else 0.001,
                msg=name,
            )
    full_opt.step()
    part_opt.step()
    for name, param in partition.canonical_parameters().items():
        torch.testing.assert_close(
            param,
            full.canonical_parameters()[name],
            rtol=1e-4 if dtype == torch.float32 else 0.03,
            atol=1e-6 if dtype == torch.float32 else 0.001,
            msg=name,
        )
    if rank == args.stages - 1:
        torch.testing.assert_close(
            torch.tensor([float(x["loss"]) for x in losses]),
            torch.tensor(reference_losses),
            rtol=1e-5 if dtype == torch.float32 else 0.01,
            atol=1e-6,
        )
    results.append(
        {
            "step": step,
            "gradient_max_abs": max(errors.values()),
            "schedule_ms": elapsed,
            "peak_allocated": torch.cuda.max_memory_allocated(),
        }
    )
args.out.mkdir(exist_ok=True, parents=True)
(args.out / f"rank-{rank}.json").write_text(
    json.dumps(
        {
            "status": "PASS",
            "stack": args.stack,
            "dtype": args.dtype,
            "steps": results,
            "microbatches": len(batches),
        },
        indent=2,
    )
    + "\n"
)
parallel_state.destroy_model_parallel()
dist.destroy_process_group()
