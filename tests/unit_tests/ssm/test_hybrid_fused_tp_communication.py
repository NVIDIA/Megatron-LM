# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Hybrid stacks with fused TP communication (``inference_fuse_tp_communication``).

A HybridStack builds attention-only ("*") and MLP-only ("-") TransformerLayers, with an
IdentityOp in place of the other sublayer. Each layer hands its residual add only to an output
projection that can add it (linear_proj, linear_fc2), and the stack links the layers as
TransformerBlock does: the fused reduce-scatter + add + rms-norm + all-gather of each output
projection applies the input norm of the next layer's linear_qkv or linear_fc1, which then reads
its input from symmetric memory. The chain stops at layers that apply their own input norm and
add their own residual: Mamba ("M") and GDN ("G"), whose in_proj and out_proj are their own,
and the MLA ("+") input, a separate input_layernorm (MLA's linear_proj still fuses).
"""

import types
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core import parallel_state
from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
from megatron.core.inference.config import (
    AsyncScheduleMode,
    InferenceConfig,
    MambaInferenceStateConfig,
)
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines import DynamicInferenceEngine
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStack
from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_inference_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.cuda_graphs import delete_cuda_graphs
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.transformer_config import MLATransformerConfig, TransformerConfig
from tests.unit_tests.test_utilities import Utils

_S, _B, _H = 64, 2, 128

# Layer kinds whose input linear applies the layer's input norm, so that the previous output
# projection can fuse it (linear_qkv, linear_fc1), and whose output projection adds the
# residual and can fuse the next layer's norm (linear_proj, linear_fc2).
_FUSABLE_INPUT = {"*", "-"}
_FUSABLE_OUTPUT = {"*", "-", "+"}

# Every pair of output projection and input linear kinds, and the chain stopping at Mamba, GDN
# and the MLA input. A stack cannot mix "*" with MLA, so MLA's linear_proj feeds an MLP layer.
_PATTERNS = {
    "attn_mlp": "*-*-",
    "same_kinds": "**--",
    "mamba": "M*-M*-",
    "gdn": "G*-G-",
    "mla": "+-+-",
}

# (tp, disable_nvls) for each path of the row-parallel layer: no reduce-scatter at TP=1, the
# NCCL reduce-scatter when the Triton NVLS kernels are disabled, and the NVLS reduce-scatter,
# fused with the next layer's input norm.
_PATHS = {"tp1": (1, False), "tp2_nccl": (2, True), "tp2_nvls": (2, False)}

# The residual is N(0, 1) and the sub-layers add a fraction of its norm, so a dropped residual
# is far outside this tolerance; it covers bf16 rounding at |x| ~ 4.
_ATOL = 2e-2
_RTOL = 2e-2


def _config(pattern: str, tp: int, *, fuse: bool, disable_nvls: bool = False, **overrides):
    """The inference-optimized config of a stack of the pattern's layers."""
    options = dict(
        num_layers=len(pattern),
        hidden_size=_H,
        ffn_hidden_size=2 * _H,
        num_attention_heads=4,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        transformer_impl="inference_optimized",
        normalization="RMSNorm",
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        params_dtype=torch.bfloat16,
        bf16=True,
        init_method_std=0.05,
        inference_fuse_tp_communication=fuse,
        inference_disable_triton_nvls_kernels=disable_nvls,
        # Mamba: 8 heads per rank at TP=2.
        mamba_num_heads=16,
        # GDN: 2 key and 2 value heads per rank at TP=2.
        linear_conv_kernel_dim=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    config_class = TransformerConfig
    if "+" in pattern:
        config_class = MLATransformerConfig
        options.update(
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_head_dim=64,
            qk_pos_emb_head_dim=32,
            v_head_dim=64,
            rope_type="rope",
        )
    options.update(overrides)
    return config_class(**options)


def _build(pattern: str, tp: int, *, fuse: bool, disable_nvls: bool = False, **overrides):
    """Smallest inference-optimized hybrid stack of the pattern's layers, without a final
    norm, on the GPU."""
    config = _config(pattern, tp, fuse=fuse, disable_nvls=disable_nvls, **overrides)
    model_parallel_cuda_manual_seed(123)
    stack = HybridStack(
        config,
        hybrid_inference_stack_spec.submodules,
        layer_config_list=validate_segment_layers(pattern, config),
        post_layer_norm=False,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    return stack.cuda().eval()


def _input_linear(symbol, layer):
    """The linear that applies the layer's input norm and all-gather, if it has one."""
    if symbol == "M":
        return layer.mixer.in_proj
    if symbol == "G":
        return layer.self_attention.in_proj
    if symbol == "*":
        return layer.self_attention.linear_qkv
    if symbol == "-":
        return layer.mlp.linear_fc1
    return None  # MLA: a separate input_layernorm


def _output_linear(symbol, layer):
    """The projection that gives the layer's output."""
    if symbol == "M":
        return layer.mixer.out_proj
    if symbol == "G":
        return layer.self_attention.out_proj
    if symbol == "-":
        return layer.mlp.linear_fc2
    return layer.self_attention.linear_proj


def _links(pattern: str) -> list[bool]:
    """Whether each layer's output projection fuses into the next layer's input linear."""
    return [
        a in _FUSABLE_OUTPUT and b in _FUSABLE_INPUT for a, b in zip(pattern, pattern[1:] + " ")
    ]


def _check_wiring(pattern: str, layers, links: list[bool]):
    """Each input linear takes exactly the handoff of a linked previous output projection, and
    each output projection carries exactly the norm weights of a linked next input linear."""
    for i, (symbol, layer) in enumerate(zip(pattern, layers)):
        input_linear = _input_linear(symbol, layer)
        if input_linear is not None:
            expected = None
            if i > 0 and links[i - 1]:
                expected = _output_linear(pattern[i - 1], layers[i - 1]).fused_tp_handoff
            assert input_linear.fused_tp_handoff is expected, f"layer {i} ({symbol}) input"

        norm = _output_linear(symbol, layer).next_layer_norm_weights
        if links[i]:
            next_input = _input_linear(pattern[i + 1], layers[i + 1])
            assert norm is not None, f"layer {i} ({symbol}) output fuses into nothing"
            assert norm.data_ptr() == next_input.layer_norm_weight.data_ptr(), f"layer {i}"
        else:
            assert norm is None, f"layer {i} ({symbol}) output fuses into a layer that cannot"


def _all_ranks_agree(value: bool) -> bool:
    flag = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() == 1.0


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str, atol, rtol):
    """allclose, reduced over all ranks so that every rank passes or fails together."""
    same_shape = actual.shape == expected.shape
    diff = (actual - expected).abs().max() if same_shape and actual.numel() else 0.0
    max_diff = torch.tensor([float(diff)], device="cuda")
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    ok = _all_ranks_agree(same_shape and torch.allclose(actual, expected, atol=atol, rtol=rtol))
    assert ok, f"{what}: max |diff| = {max_diff.item():.4f} (atol={atol}, rtol={rtol})"


def _forward(stack: HybridStack, x: torch.Tensor) -> torch.Tensor:
    """Run the stack on a copy of x; the fused kernel updates the residual in place."""
    with torch.no_grad(), InferenceMode.active():
        out = stack(hidden_states=x.clone(), attention_mask=None)
    torch.cuda.synchronize()
    dist.barrier()
    return out.float()


def _require_nvls():
    buf = SymmetricMemoryManager.get_buffer(
        "tp", process_group=parallel_state.get_tensor_model_parallel_group()
    )  # collective: create on every rank before any skip
    capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    hdl = buf.symm_mem_hdl
    if not _all_ranks_agree(capable and hdl is not None and bool(hdl.multicast_ptr)):
        pytest.skip("NVLS multicast unavailable")


@pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)
class TestHybridFusedTPCommunication:

    def _init(self, tp: int, context_parallel_size: int = 1):
        world = tp * context_parallel_size
        if Utils.world_size % world != 0:
            pytest.skip(f"requires a world size divisible by {world}")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, context_parallel_size=context_parallel_size
        )
        SymmetricMemoryManager.destroy()

    def teardown_method(self, method):
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2])
    @pytest.mark.parametrize("pattern", list(_PATTERNS.values()), ids=list(_PATTERNS))
    def test_stack_wiring(self, pattern, tp):
        """Each output projection that can fuse does so into the next layer's input linear, and
        only there; Mamba, GDN and MLA inputs and the last output are not linked."""
        self._init(tp)
        _check_wiring(pattern, _build(pattern, tp, fuse=True).layers, _links(pattern))

    def test_stack_wiring_stops_at_cp_layout_conversions(self):
        """With context parallelism the stack converts the stream between the attention layers'
        zigzag layout and the MLP layers' contiguous one, so a layer is linked only to a
        neighbor in its own layout."""
        self._init(1, context_parallel_size=2)
        pattern = "*--*"
        stack = _build(
            pattern,
            1,
            fuse=True,
            context_parallel_size=2,
            attention_cp_layout="zigzag",
            linear_cp_layout="contiguous",
        )
        assert stack._cp_layout_manager.requires_conversion
        _check_wiring(pattern, stack.layers, [False, True, False, False])

    @pytest.mark.parametrize("path", list(_PATHS))
    @pytest.mark.parametrize("pattern", list(_PATTERNS.values()), ids=list(_PATTERNS))
    def test_fused_matches_unfused(self, pattern, path):
        """Flag-on output equals flag-off output on every path. With NVLS, the fused kernel
        runs exactly at the links, and only the input linears it feeds skip their own norm."""
        tp, disable_nvls = _PATHS[path]
        self._init(tp)
        if path == "tp2_nvls":
            _require_nvls()

        unfused = _build(pattern, tp, fuse=False, disable_nvls=disable_nvls)
        fused = _build(pattern, tp, fuse=True, disable_nvls=disable_nvls)
        gen = torch.Generator(device="cpu").manual_seed(4321)
        fused_params = dict(fused.named_parameters())
        with torch.no_grad():
            for name, param in unfused.named_parameters():
                if name.endswith(("layer_norm_weight", "layernorm.weight")):
                    # Gains away from 1, so a norm applied with the wrong weights shows.
                    param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=gen))
                # Copy in place: the fused wiring holds the norm weights by reference.
                fused_params.pop(name).copy_(param)
        assert not fused_params, f"parameters only in the fused stack: {sorted(fused_params)}"

        gen = torch.Generator(device="cpu").manual_seed(1234)
        x = torch.randn(_S, _B, _H, generator=gen).to(torch.bfloat16)
        x = x.chunk(tp, dim=0)[parallel_state.get_tensor_model_parallel_rank()].contiguous().cuda()

        expected = _forward(unfused, x)
        fn_fused = inference_layers.fused_multimem_rs_add_norm_ag
        fn_norm = inference_layers._te_rms_norm_kernel
        with (
            mock.patch.object(
                inference_layers, "fused_multimem_rs_add_norm_ag", wraps=fn_fused
            ) as fused_spy,
            mock.patch.object(inference_layers, "_te_rms_norm_kernel", wraps=fn_norm) as norm_spy,
        ):
            actual = _forward(fused, x)
        _assert_close(actual, expected, "fused vs unfused", _ATOL, _RTOL)

        nvls = path == "tp2_nvls"
        links = _links(pattern)
        assert fused_spy.call_count == (sum(links) if nvls else 0)
        applied = [call.kwargs["weight"] for call in norm_spy.call_args_list]
        for i, (symbol, layer) in enumerate(zip(pattern, fused.layers)):
            input_linear = _input_linear(symbol, layer)
            if input_linear is None:
                continue
            fed = nvls and i > 0 and links[i - 1]
            applies_norm = any(w is input_linear.layer_norm_weight for w in applied)
            assert applies_norm != fed, f"layer {i} ({symbol}) applies its own norm: {applies_norm}"


# A Nemotron-H-style model in the dynamic engine: Mamba layers break the fused chain, and
# attention-only and MLP-only layers link across layers.
_ENGINE_PATTERN = "M*-M*-"
_ENGINE_VOCAB, _ENGINE_MAX_SEQ = 128, 1024
# (arrival step, prompt length). Prefill and mixed steps run eagerly, decode steps from graphs.
_ENGINE_PROMPTS = ((0, 9), (0, 37), (1, 130))
_ENGINE_TOKENS_TO_GENERATE = 6
# The decoder outputs are RMS-normalized, of order 1.
_ENGINE_ATOL = 5e-2
_ENGINE_RTOL = 5e-2


@pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)
class TestHybridModelFusedTPEngine:
    """A HybridModel at TP=2 in the dynamic engine with CUDA graphs: the same requests served
    with the flag on and off decode the same tokens through the same decoder outputs."""

    def setup_method(self, method):
        if Utils.world_size % 2 != 0:
            pytest.skip("requires a world size divisible by TP=2")
        Utils.initialize_model_parallel(tensor_model_parallel_size=2)
        SymmetricMemoryManager.destroy()

    def teardown_method(self, method):
        delete_cuda_graphs()
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @staticmethod
    def _build_model(scope: InferenceCudaGraphScope, *, fuse: bool) -> HybridModel:
        model_parallel_cuda_manual_seed(
            123, inference_rng_tracker=True, use_cudagraphable_rng=False, force_reset_rng=True
        )
        config = _config(
            _ENGINE_PATTERN,
            2,
            fuse=fuse,
            # One token per rank in a fused reduce-scatter still fills a warp.
            hidden_size=256,
            ffn_hidden_size=512,
            num_attention_heads=8,
            flash_attention_version=2,
            inference_rng_tracker=True,
            cuda_graph_impl="local",
            inference_cuda_graph_scope=scope,
        )
        model = HybridModel(
            config=config,
            hybrid_stack_spec=hybrid_inference_stack_spec,
            vocab_size=_ENGINE_VOCAB,
            max_sequence_length=_ENGINE_MAX_SEQ,
            hybrid_layer_pattern=_ENGINE_PATTERN,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
        )
        model = model.cuda().eval()
        with torch.no_grad():
            for param in model.parameters():
                # As Float16Module does, but only for parameters not yet in bf16 (Mamba's): the
                # fused wiring holds the bf16 norm weights by reference, so they keep their
                # storage.
                if param.dtype != torch.bfloat16:
                    param.data = param.data.to(torch.bfloat16)
        return model

    @staticmethod
    def _serve(model: HybridModel):
        """Serve the prompts to completion. Returns the generated tokens by request, the decoder
        output of each real token in every forward, and whether each step replayed a graph."""
        context = DynamicInferenceContext(
            model_config=model.config,
            inference_config=InferenceConfig(
                max_sequence_length=_ENGINE_MAX_SEQ,
                buffer_size_gb=0.25,
                block_size_tokens=256,
                max_requests=16,
                materialize_only_last_token_logits=False,
                unified_memory_level=0,
                num_cuda_graphs=4,
                use_cuda_graphs_for_non_decode_steps=False,
                async_sched_mode=AsyncScheduleMode.LEGACY,
                mamba_inference_state_config=MambaInferenceStateConfig.from_model(model),
            ),
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.model_is_pipeline_parallel = False
        controller = TextGenerationController(
            inference_wrapped_model=wrapper,
            tokenizer=types.SimpleNamespace(
                vocab_size=_ENGINE_VOCAB, detokenize=lambda tokens: "tokenized_prompt"
            ),
        )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        outputs, recording = [], [False]

        def record(module, args):
            # The output layer runs eagerly around graphed decoders. Keep the rows of real
            # tokens in this rank's sequence-parallel shard (or in the whole step).
            if not recording[0] or torch.cuda.is_current_stream_capturing():
                return
            hidden = args[0]
            rows = hidden.shape[0]
            offset = 0 if rows == context.padded_active_token_count else tp_rank * rows
            real = max(0, min(rows, context.active_token_count - offset))
            outputs.append(hidden[:real].float().clone())

        hook = model.output_layer.register_forward_pre_hook(record)
        delete_cuda_graphs()
        tokens, graphed = {}, []
        try:
            with torch.inference_mode():
                engine = DynamicInferenceEngine(controller, context)  # captures the graphs
                recording[0] = True
                gen = torch.Generator(device="cpu").manual_seed(77)
                arrivals = [
                    (
                        arrival,
                        DynamicInferenceRequest(
                            request_id=i,
                            prompt_tokens=torch.randint(0, _ENGINE_VOCAB, (length,), generator=gen)
                            .to(torch.int64)
                            .cuda(),
                            sampling_params=SamplingParams(
                                num_tokens_to_generate=_ENGINE_TOKENS_TO_GENERATE,
                                termination_id=-1,
                                top_k=1,
                            ),
                        ),
                    )
                    for i, (arrival, length) in enumerate(_ENGINE_PROMPTS)
                ]
                last_arrival = max(arrival for arrival, _ in arrivals)
                step = 0
                while step <= last_arrival or engine.has_unfinished_requests():
                    for arrival, request in arrivals:
                        if arrival == step:
                            engine._add_request(request)
                    result = engine.step_modern()
                    graphed.append(context.using_cuda_graph_this_step())
                    for request in result["finished_requests"]:
                        tokens[request.request_id] = list(request.generated_tokens)
                    step += 1
        finally:
            hook.remove()
            delete_cuda_graphs()
        return tokens, outputs, graphed

    @pytest.mark.parametrize(
        "scope",
        [InferenceCudaGraphScope.layer, InferenceCudaGraphScope.block],
        ids=["layer_graphs", "block_graphs"],
    )
    def test_engine_matches_unfused(self, scope):
        _require_nvls()
        unfused = self._build_model(scope, fuse=False)
        fused = self._build_model(scope, fuse=True)
        with torch.no_grad():
            # Tied embeddings at about the sub-layers' scale: each request decodes its last
            # prompt token again by a wide logit margin, so both engines decode the same
            # tokens, while the sub-layers still move the decoder outputs visibly.
            embedding = unfused.embedding.word_embeddings.weight
            embedding.mul_(0.3 / embedding.float().square().mean().sqrt())
            fused_params = dict(fused.named_parameters())
            for name, param in unfused.named_parameters():
                # Copy in place: the fused wiring holds the norm weights by reference.
                fused_params.pop(name).copy_(param)
        assert not fused_params, f"parameters only in the fused model: {sorted(fused_params)}"

        expected_tokens, expected, graphed = self._serve(unfused)
        with mock.patch.object(
            inference_layers,
            "fused_multimem_rs_add_norm_ag",
            wraps=inference_layers.fused_multimem_rs_add_norm_ag,
        ) as fused_spy:
            actual_tokens, actual, fused_graphed = self._serve(fused)

        assert fused_spy.call_count > 0, "the fused kernel neither ran nor was captured"
        assert any(graphed) and not all(graphed), f"graphed steps: {graphed}"
        assert fused_graphed == graphed
        assert actual_tokens == expected_tokens
        assert len(actual) == len(expected), f"{len(actual)} vs {len(expected)} forwards"
        for j, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"decoder output, forward {j}", _ENGINE_ATOL, _ENGINE_RTOL)
