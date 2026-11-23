# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared typed pipeline stages: CPU transport parity and HybridModel CP2 x PP2/VPP training."""

import gc
import re
import zlib
from functools import partial

import pytest
import torch
from torch import nn

from megatron.core.context_parallel.shared_state import gather_state
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.pipeline_payload import (
    PipelineDataIterator,
    TensorStatePayload,
    backward_pipeline_payload,
)
from megatron.core.pipeline_parallel.pipeline_stage import (
    PipelineStage,
    PipelineStageState,
    prepare_stage_inputs,
)
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.test_stateful_module import (  # pylint: disable=unused-import
    te_rng_tracker,
)


class _Codec:
    """Toy model-specific half of a typed stage.

    With ``regions``, packed prefixes ``cu`` come from the first stage's batch, layer 0
    produces ``memory``/``ids``, every later layer consumes them and the last retires
    them. Without, the batch's tokens are relayed unchanged past ordinary layers.
    """

    name = "toy"

    def __init__(self, num_layers, hidden_size, *, regions=True):
        self.num_layers, self.hidden_size, self.regions = num_layers, hidden_size, regions

    def describe(self, input_ids, packed_seq_params):
        sequences = 0 if packed_seq_params is None else packed_seq_params.cu_seqlens_q.numel() - 1
        return (input_ids.shape[1], input_ids.shape[0], sequences)

    def fields(self, boundary, metadata):
        sequence, batch, sequences = metadata
        shape = (sequence, batch, self.hidden_size)
        hidden = TensorField("hidden", shape, torch.float32, "contiguous", True)
        if boundary == self.num_layers:
            return (hidden,)
        if not self.regions:
            return (hidden, TensorField("tokens", (batch, sequence), torch.int64, "local", False))
        cu = TensorField("cu", (sequences + 1,), torch.int32, "replicated", False)
        if boundary == 0:
            return (hidden, cu)
        return (
            hidden,
            cu,
            TensorField("memory", shape, torch.float32, "contiguous", True),
            TensorField("ids", (sequence,), torch.int64, "contiguous", False),
        )

    def initial_state(self, input_ids, packed_seq_params, metadata):
        return {"cu": packed_seq_params.cu_seqlens_q} if self.regions else {"tokens": input_ids}

    def packed_seq_params(self, state, metadata):
        if "cu" not in state:
            return None
        cu = state["cu"]
        return PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)


class _Region(nn.Module):
    """One layer: the first produces the shared memory, later ones gather and select it."""

    def __init__(self, index, codec, metadata, cp, device):
        super().__init__()
        self.index, self.cp = index, cp
        self.shared = codec.fields(1, metadata)[-2:]
        self.weight = nn.Parameter(torch.zeros((), device=device))

    def forward(self, hidden, state):
        state = dict(state)
        if self.index == 0:
            memory = hidden * self.weight
            ids = torch.arange(hidden.shape[0], device=hidden.device)
            state.update(memory=memory, ids=ids + self.cp.rank() * hidden.shape[0])
            return memory + hidden.sin() * 0.1, state
        gathered, _ = gather_state(state, self.shared, self.cp)
        selected = gathered["memory"].index_select(0, state["ids"])
        output = hidden * self.weight + (selected + gathered["memory"].mean(0)) * self.weight
        if "cu" in state:
            sequence = torch.searchsorted(state["cu"].long(), state["ids"], right=True) - 1
            output = output * (1 + 0.01 * sequence.to(output.dtype)).view(-1, 1, 1)
        return output, state


def _regions(codec, metadata, start, end, cp, device):
    return nn.ModuleList(
        StatefulModule(
            _Region(i, codec, metadata, cp, device),
            codec.fields(i, metadata)[1:],
            codec.fields(i + 1, metadata)[1:],
        )
        for i in range(start, end)
    )


# CPU: the adapter's transport contract, with a simulated wire and no process groups.


class _Rank0:
    def rank(self):
        return 0

    def size(self):
        return 1


class _CpuStage(nn.Module):
    """Uses the adapter exactly as HybridModel does around its decoder."""

    def __init__(self, codec, metadata, start, end, layers):
        super().__init__()
        self.pipeline_stage = PipelineStage(
            codec, start, end, pre_process=start == 0, post_process=end == layers
        )
        self.layers = _regions(codec, metadata, start, end, _Rank0(), "cpu")
        self.hidden = None

    def set_input_tensor(self, tensor):
        self.hidden = self.pipeline_stage.receive(tensor)

    def forward(self, tokens, packed):
        state = self.pipeline_stage.enter(tokens, packed)
        if self.pipeline_stage.pre_process:
            hidden = tokens.t().unsqueeze(-1).float().expand(-1, -1, 3) / 16
        else:
            hidden, self.hidden = self.hidden, None
            torch.testing.assert_close(state.packed_seq_params.cu_seqlens_q, packed.cu_seqlens_q)
        values = state.values
        for layer in self.layers:
            hidden, values = layer.run(hidden, values)
        if self.pipeline_stage.post_process:
            return hidden.square().mean()
        state.values = values
        return self.pipeline_stage.exit(hidden, state)


def _wire(payload):
    """Receive into fresh leaves, as the typed communicator allocates them from its plan."""
    return TensorStatePayload(
        tuple(
            t.detach().clone().requires_grad_(f.differentiable)
            for t, f in zip(payload.tensors, payload.spec.schema.packed_fields)
        ),
        payload.spec,
    )


@pytest.mark.parametrize("pp_size", [2, 4])
def test_cpu_stages_match_unpartitioned_layers(pp_size):
    layers, metadata = 4, (8, 1, 2)
    codec = _Codec(layers, 3)
    tokens = torch.arange(8).view(1, 8)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
    reference = _CpuStage(codec, metadata, 0, layers, layers)
    per = layers // pp_size
    stages = [_CpuStage(codec, metadata, i * per, (i + 1) * per, layers) for i in range(pp_size)]
    with torch.no_grad():
        for i, region in enumerate(reference.layers):
            region.module.weight.fill_(0.3 + 0.1 * i)
            stages[i // per].layers[i % per].module.weight.fill_(0.3 + 0.1 * i)
    expected = reference(tokens, packed)
    expected.backward()

    sent, received, value = [], [None], None
    for i, stage in enumerate(stages):
        plan = stage.pipeline_stage.plan([metadata])
        stage.set_input_tensor(value)
        value = stage(tokens, packed)
        if i < pp_size - 1:
            assert (
                plan.outgoing == (value.spec,) and value.spec.boundary_id == f"toy:{per * i + per}"
            )
            sent.append(value)
            value = _wire(value)
            received.append(value)
        else:
            assert plan.outgoing == (None,)
        assert plan.incoming == ((None,) if i == 0 else (received[i].spec,))
    torch.testing.assert_close(value, expected)
    grads = None
    for i in reversed(range(pp_size)):
        output = value if i == pp_size - 1 else sent[i]
        grads = backward_pipeline_payload(received[i], output, grads)
    for i, region in enumerate(reference.layers):
        actual = stages[i // per].layers[i % per].module.weight.grad
        torch.testing.assert_close(actual, region.module.weight.grad)


def test_cpu_stage_rejects_foreign_or_missing_payloads():
    codec = _Codec(4, 3)
    first = PipelineStage(codec, 0, 2, pre_process=True, post_process=False)
    later = PipelineStage(codec, 2, 4, pre_process=False, post_process=True)
    with pytest.raises(ValueError, match="requires a typed state payload"):
        later.enter(None, None)
    hidden = torch.zeros(8, 1, 3)
    state = {"cu": torch.tensor([0, 8], dtype=torch.int32), "memory": hidden}
    state["ids"] = torch.arange(8)
    stale = PipelineStage(codec, 0, 1, pre_process=True, post_process=False).exit(
        hidden, PipelineStageState((8, 1, 1), state)
    )
    later.receive(stale)
    with pytest.raises(ValueError, match="toy:1 belongs to another boundary"):
        later.enter(None, None)

    class Model(nn.Module):
        pipeline_stage = first

    metadata = [(8, 1, 1), (4, 1, 2)]
    iterator = prepare_stage_inputs(
        iter(metadata * 2), Model(), 2, forward_only=False, describe=lambda batch: batch
    )
    assert isinstance(iterator, PipelineDataIterator)
    assert iterator.pipeline_payload_plan == first.plan(metadata)
    assert list(iterator) == metadata


# GPU: HybridModel stages through the real ordinary and interleaved 1F1B schedules.

COUNT, SEQUENCE, HIDDEN, VOCAB = 4, 8, 16, 32


class _ToyStack(MegatronModule):
    """Decoder stand-in whose layers run as stateful regions in every execution mode."""

    def __init__(
        self,
        config,
        *,
        pre_process,
        post_process,
        layer_config_list,
        pp_layer_offset,
        pg_collection,
        codec,
        metadata,
        mode,
        **kwargs,
    ):
        super().__init__(config)
        self.pre_process, self.post_process = pre_process, post_process
        self.metadata, self.mode, self.cp = metadata, mode, pg_collection.cp
        end = pp_layer_offset + len(layer_config_list)
        self.layers = _regions(
            codec, metadata, pp_layer_offset, end, self.cp, torch.cuda.current_device()
        )
        for layer in self.layers:
            # Schedules read each decoder layer's config, as HybridStack layers carry it.
            layer.config = config
        self.input_tensor, self.state, self.graphs, self.iteration = None, None, None, 0

    def capture(self):
        """Capture each region with one slot per outstanding microbatch."""
        hidden = torch.zeros(SEQUENCE, 1, HIDDEN, device="cuda", requires_grad=True)
        cu = torch.tensor([0, 5, 2 * SEQUENCE], dtype=torch.int32, device="cuda")
        memory = torch.ones_like(hidden, requires_grad=True)
        ids = torch.arange(SEQUENCE, device="cuda") + self.cp.rank() * SEQUENCE
        self.graphs = []
        for layer in self.layers:
            index = layer.module.index
            state = {"cu": cu} if index == 0 else {"cu": cu, "memory": memory, "ids": ids}
            self.graphs.append(StatefulGraphs(layer, hidden, state, slots=COUNT, backend=self.mode))

    def set_input_tensor(self, tensor):
        self.input_tensor = tensor

    def forward(self, hidden_states, packed_seq_params=None, **kwargs):
        state = self.state
        # HybridModel passes the codec-rebuilt prefixes on later stages.
        assert packed_seq_params is state.packed_seq_params
        hidden = hidden_states if self.pre_process else self.input_tensor
        values, slot = state.values, self.iteration % COUNT
        for i, layer in enumerate(self.layers):
            if self.graphs is not None:
                hidden, values = self.graphs[i].run(hidden, values, slot=slot)
            else:
                hidden, values = layer.run(hidden, values, recompute=self.mode == "recompute")
        self.iteration += 1
        state.values = values
        return hidden


class _ToyModel(HybridModel):
    """A subclass whose layers consume the state enters the stage itself, like DSv4.1."""

    def forward(self, input_ids, position_ids, attention_mask=None, **kwargs):
        state = self.pipeline_stage.enter(input_ids, kwargs.pop("packed_seq_params"))
        self.decoder.state = state
        try:
            return super().forward(
                input_ids, position_ids, attention_mask, pipeline_state=state, **kwargs
            )
        finally:
            self.decoder.state = None


def _groups(pp):
    groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp", "dp_cp"])
    return ProcessGroupCollection(tp=groups.tp, cp=groups.cp, pp=pp, embd=None, dp_cp=groups.dp_cp)


def _fill(model, offset):
    """Initialize every parameter from its global name, independent of partitioning."""
    params = {}
    for name, param in model.named_parameters():
        name = re.sub(
            r"decoder\.layers\.(\d+)", lambda m: f"decoder.layers.{int(m[1]) + offset}", name
        )
        generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
        with torch.no_grad():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.5)
        params[name] = param
    return params


def _batches(cp, packed):
    """Contiguous CP shards of 16-token rows, packed as two sequences when requested."""
    batches = []
    for b in range(COUNT):
        tokens = (torch.arange(2 * SEQUENCE, device="cuda") * (b + 3) + 7) % VOCAB
        local = slice(cp.rank() * SEQUENCE, (cp.rank() + 1) * SEQUENCE)
        cu = torch.tensor([0, 5 + b, 2 * SEQUENCE], dtype=torch.int32, device="cuda")
        batches.append(
            dict(
                input_ids=tokens[local].view(1, -1),
                position_ids=torch.arange(2 * SEQUENCE, device="cuda")[local].view(1, -1),
                attention_mask=None,
                labels=tokens.roll(-1)[local].view(1, -1),
                packed_seq_params=(
                    PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu)
                    if packed
                    else None
                ),
            )
        )
    return batches


def _check_against_unpartitioned(build, vp_size, codec):
    """Train COUNT microbatches through CP2 x PP2[/VPP2] and compare with one stage."""
    Utils.initialize_model_parallel(
        1,
        2,
        virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
        context_parallel_size=2,
    )
    rank = torch.distributed.get_rank()
    single = [torch.distributed.new_group([r]) for r in range(Utils.world_size)][rank]
    stage_groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["pp"])
    groups, reference_groups = _groups(stage_groups.pp), _groups(single)
    pp_rank = groups.pp.rank()
    common = dict(
        hidden_size=HIDDEN,
        num_attention_heads=1,
        context_parallel_size=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pipeline_dtype=torch.float32,
    )
    config = TransformerConfig(
        num_layers=codec.num_layers,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=vp_size if vp_size > 1 else None,
        microbatch_group_size_per_vp_stage=2,
        batch_p2p_comm=False,
        overlap_p2p_comm=False,
        deallocate_pipeline_outputs=True,
        **common,
    )
    reference_config = TransformerConfig(num_layers=codec.num_layers, **common)
    per = codec.num_layers // (2 * vp_size)
    pattern = "|".join(["-" * per] * 2 * vp_size)
    models = []
    try:
        for vp in range(vp_size):
            models.append(
                build(
                    config,
                    pattern,
                    groups,
                    reference=False,
                    pre_process=pp_rank == 0 and vp == 0,
                    post_process=pp_rank == 1 and vp == vp_size - 1,
                    vp_stage=vp if vp_size > 1 else None,
                )
            )
        reference = build(
            reference_config,
            "-" * codec.num_layers,
            reference_groups,
            reference=True,
            pre_process=True,
            post_process=True,
            vp_stage=None,
        )
        params = {}
        for model in models:
            params.update(_fill(model, model.pipeline_stage.start))
        expected_params = _fill(reference, 0)
        assert params.keys() <= expected_params.keys()

        batches = _batches(groups.cp, codec.regions)
        expected = []
        for batch in batches:
            value = reference(**batch).float().mean()
            expected.append(value.detach().clone())
            # Match MCore's legacy two-value loss reduction: CP / microbatches.
            (value * groups.cp.size() / COUNT).backward()

        actual = []

        def forward_step(iterator, model):
            output = model(**next(iterator))

            def loss(value):
                value = value.float().mean()
                actual.append(value.detach().clone())
                return value, {"loss": value.detach().clone()}

            return output, loss

        forward_step.prepare_pipeline_inputs = partial(
            prepare_stage_inputs,
            describe=lambda batch: codec.describe(batch["input_ids"], batch["packed_seq_params"]),
        )
        schedule = (
            forward_backward_pipelining_with_interleaving
            if vp_size > 1
            else forward_backward_pipelining_without_interleaving
        )
        schedule(
            forward_step_func=forward_step,
            data_iterator=[iter(batches) for _ in models] if vp_size > 1 else iter(batches),
            model=models if vp_size > 1 else models[0],
            num_microbatches=COUNT,
            seq_length=2 * SEQUENCE,
            micro_batch_size=1,
            p2p_communicator=P2PCommunicator(groups.pp, config),
            pg_collection=groups,
        )
        if pp_rank == 1:
            torch.testing.assert_close(torch.stack(actual), torch.stack(expected))
        for name, param in params.items():
            torch.testing.assert_close(
                param.grad, expected_params[name].grad, rtol=1e-5, atol=1e-6, msg=name
            )
        for model in models:
            assert model.pipeline_stage._incoming is None
    finally:
        for model in models:
            for graph in getattr(model.decoder, "graphs", None) or ():
                graph.close()
        gc.collect()
        Utils.destroy_model_parallel()


def _requires_cp2_pp2():
    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("Requires a multiple of four GPU workers for PP2 x CP2")


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("vp_size", [1, 2])
@pytest.mark.parametrize("mode", ["eager", "recompute", "torch", "transformer_engine"])
def test_hybrid_model_stages_match_unpartitioned_model(vp_size, mode):
    _requires_cp2_pp2()
    codec = _Codec(4 * vp_size, HIDDEN)
    metadata = (SEQUENCE, 1, 2)

    def build(config, pattern, groups, *, reference, **kwargs):
        stack_mode = "eager" if reference else mode
        stack = ModuleSpec(
            module=_ToyStack, params=dict(codec=codec, metadata=metadata, mode=stack_mode)
        )
        model = _ToyModel(
            config=config,
            hybrid_stack_spec=stack,
            vocab_size=VOCAB,
            max_sequence_length=2 * SEQUENCE,
            hybrid_layer_pattern=pattern,
            pg_collection=groups,
            pipeline_state_codec=codec,
            **kwargs,
        ).cuda()
        if stack_mode in ("torch", "transformer_engine"):
            # Graphs keep parameter storage, so _fill may initialize after capture.
            model.decoder.capture()
        return model

    _check_against_unpartitioned(build, vp_size, codec)


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("vp_size", [1, 2])
def test_hybrid_model_enters_its_stage_with_real_layers(vp_size):
    """A plain HybridModel with a codec relays its state through HybridStack MLP layers."""
    _requires_cp2_pp2()
    codec = _Codec(4 * vp_size, HIDDEN, regions=False)

    def build(config, pattern, groups, *, reference, **kwargs):
        return HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=VOCAB,
            max_sequence_length=2 * SEQUENCE,
            hybrid_layer_pattern=pattern,
            pg_collection=groups,
            pipeline_state_codec=codec,
            **kwargs,
        ).cuda()

    _check_against_unpartitioned(build, vp_size, codec)
