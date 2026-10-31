# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Rotary embeddings as inputs of TE per-layer CUDA graphs.

A TE graphed callable copies the inputs it was captured with into the graph at replay and
ignores any other keyword argument. These tests check that the graphed layers replay with the
rotary embedding the model passes, so that graphed training matches eager training bit for bit:
multimodal RoPE ('mrope', whose embedding changes with every batch), YaRN, 'rope', and a vision
encoder's 2D RoPE (VisionTECudaGraphHelper); and that a replay raises instead of silently
dropping a rotary embedding that its graph was captured without.
"""

import gc

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.num_microbatches_calculator import (
    destroy_num_microbatches_calculator,
    init_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.cuda_graphs import (
    HAVE_TE_GRAPHS,
    TECudaGraphHelper,
    VisionTECudaGraphHelper,
)
from megatron.core.transformer.enums import AttnBackend, CudaGraphModule
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import set_attention_backend
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

_SEQ = 64
_MBS = 2
_HIDDEN = 64
_HEADS = 4
_HEAD_DIM = _HIDDEN // _HEADS
_VOCAB = 128
_MROPE_SECTION = [4, 2, 2]  # sums to _HEAD_DIM // 2
_VISION_SEQ = 48
# Large weights give O(1) attention scores, so the rotary embedding changes the outputs.
_INIT_STD = 0.25

pytestmark = pytest.mark.skipif(
    not (HAVE_TE and HAVE_TE_GRAPHS and is_te_min_version("2.10.0")),
    reason="TE per-layer CUDA graph tests require TransformerEngine >= 2.10.0",
)


class _ModelChunk(torch.nn.Module):
    """Stands in for a DDP-wrapped model chunk: TECudaGraphHelper calls zero_grad_buffer()."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def zero_grad_buffer(self):
        self.module.zero_grad(set_to_none=True)


def _config(cuda_graph_impl, **kwargs):
    return TransformerConfig(
        num_layers=2,
        hidden_size=_HIDDEN,
        num_attention_heads=_HEADS,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        use_cpu_initialization=True,
        init_method_std=_INIT_STD,
        # Deterministic attention, so that graphed and eager results match bit for bit.
        attention_backend=AttnBackend.unfused,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_warmup_steps=1,
        **kwargs,
    )


def _build_gpt(cuda_graph_impl, position_embedding_type, seed=1234):
    mrope_section = _MROPE_SECTION if position_embedding_type == "mrope" else None
    config = _config(cuda_graph_impl, mrope_section=mrope_section)
    if position_embedding_type == "yarn":
        # GPTModel reads the YaRN parameters from the config.
        config.yarn_rotary_scaling_factor = 2.0
        config.yarn_original_max_position_embeddings = _SEQ // 2
        config.yarn_beta_fast = 32.0
        config.yarn_beta_slow = 1.0
        config.yarn_mscale = 1.0
        config.yarn_mscale_all_dim = 0.0
        config.yarn_correction_range_round_to_int = True
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
        vocab_size=_VOCAB,
        max_sequence_length=_SEQ,
        position_embedding_type=position_embedding_type,
    ).cuda()
    model.train()
    return model


def _mrope_position_ids(variant):
    """[3, b, s] multimodal position ids: text, an image (fixed t, an h x w grid), then text."""
    position_ids = torch.zeros(3, _MBS, _SEQ, dtype=torch.long)
    for b in range(_MBS):
        start = 5 + 3 * b + 7 * variant
        grid_h, grid_w = (4, 6) if (b + variant) % 2 == 0 else (6, 4)
        num_image = grid_h * grid_w
        position_ids[:, b, :start] = torch.arange(start)
        image = slice(start, start + num_image)
        position_ids[0, b, image] = start
        position_ids[1, b, image] = start + torch.arange(grid_h).repeat_interleave(grid_w)
        position_ids[2, b, image] = start + torch.arange(grid_w).repeat(grid_h)
        rest = _SEQ - start - num_image
        position_ids[:, b, start + num_image :] = start + max(grid_h, grid_w) + torch.arange(rest)
    return position_ids.cuda()


def _gpt_batch(position_embedding_type, variant):
    """The same tokens for every variant; only the (mrope) positions differ."""
    generator = torch.Generator().manual_seed(100)
    input_ids = torch.randint(0, _VOCAB, (_MBS, _SEQ), generator=generator).cuda()
    if position_embedding_type == "mrope":
        position_ids = _mrope_position_ids(variant)
    else:
        position_ids = torch.arange(_SEQ).repeat(_MBS, 1).cuda()
    return input_ids, position_ids


def _run_gpt(model, input_ids, position_ids, weight):
    """One training microbatch: returns the logits and every parameter gradient."""
    model.zero_grad(set_to_none=True)
    logits = model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
    output = logits.detach().float().clone()
    (logits.float() * weight).sum().backward()
    torch.cuda.synchronize()
    grads = {
        name: param.grad.detach().float().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    return output, grads


def _assert_same(actual, expected):
    actual_out, actual_grads = actual
    expected_out, expected_grads = expected
    torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
    assert actual_grads.keys() == expected_grads.keys()
    for name in expected_grads:
        torch.testing.assert_close(actual_grads[name], expected_grads[name], rtol=0, atol=0)


def _assert_different(a, b):
    """The outputs differ by more than 1% of their magnitude (the test inputs matter)."""
    assert (a[0] - b[0]).abs().max() > 1e-2 * b[0].abs().max()


class _CudaGraphTestBase:
    @pytest.fixture(autouse=True)
    def _deterministic_te(self, monkeypatch):
        # Deterministic TE kernels (the configs select the unfused attention backend).
        monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.helper = None

    def teardown_method(self, method):
        if self.helper is not None and self.helper.graphs_created():
            self.helper.delete_cuda_graphs()
        self.helper = None
        destroy_num_microbatches_calculator()
        Utils.destroy_model_parallel()
        gc.collect()

    @staticmethod
    def _init_microbatches(micro_batch_size):
        destroy_num_microbatches_calculator()
        init_num_microbatches_calculator(
            rank=0,
            global_batch_size=micro_batch_size,
            micro_batch_size=micro_batch_size,
            data_parallel_size=1,
            decrease_batch_size_if_needed=False,
        )


class TestLanguageModelRotaryGraphInputs(_CudaGraphTestBase):
    """GPT models: the decoder layers' graphs take the model's rotary embedding as an input."""

    def _graphed_and_reference(self, position_embedding_type):
        self._init_microbatches(_MBS)
        graphed = _build_gpt("transformer_engine", position_embedding_type)
        reference = _build_gpt("none", position_embedding_type, seed=999)
        reference.load_state_dict(graphed.state_dict())
        self.helper = TECudaGraphHelper(
            model=[_ModelChunk(graphed)],
            config=graphed.config,
            seq_length=_SEQ,
            micro_batch_size=_MBS,
            optimizers=[],
        )
        return graphed, reference

    @pytest.mark.parametrize("position_embedding_type", ["mrope", "yarn", "rope"])
    def test_te_graphs_match_eager(self, position_embedding_type):
        """Replays match eager outputs and gradients bit for bit, for batches whose (mrope)
        positions differ from the capture's and from each other."""
        graphed, reference = self._graphed_and_reference(position_embedding_type)
        weight = torch.randn(_MBS, _SEQ, _VOCAB, device="cuda")
        batch_a = _gpt_batch(position_embedding_type, 0)
        batch_b = _gpt_batch(position_embedding_type, 1)
        eager_a = _run_gpt(reference, *batch_a, weight)
        eager_b = _run_gpt(reference, *batch_b, weight)
        _run_gpt(graphed, *batch_a, weight)  # the eager warmup step before the capture
        self.helper.create_cudagraphs()

        _assert_same(_run_gpt(graphed, *batch_a, weight), eager_a)
        _assert_same(_run_gpt(graphed, *batch_b, weight), eager_b)
        _assert_same(_run_gpt(graphed, *batch_a, weight), eager_a)
        if position_embedding_type == "mrope":
            _assert_different(eager_a, eager_b)

    def test_mrope_sample_is_a_buffer_per_layer(self):
        """The mrope input has the [s, b, 1, d] shape and dtype the model passes, and every
        (layer, microbatch) sample has its own buffer."""
        graphed, reference = self._graphed_and_reference("mrope")
        input_ids, position_ids = _gpt_batch("mrope", 0)
        weight = torch.randn(_MBS, _SEQ, _VOCAB, device="cuda")
        _run_gpt(graphed, input_ids, position_ids, weight)
        expected = reference.rotary_pos_emb(position_ids, _MROPE_SECTION)

        _, kwargs = self.helper._get_cuda_graph_input_data()
        samples = [sample["rotary_pos_emb"] for sample in kwargs["sample_kwargs"]]
        assert len(samples) == len(graphed.decoder.layers)
        for sample in samples:
            assert sample.shape == expected.shape == (_SEQ, _MBS, 1, _HEAD_DIM)
            assert sample.dtype == expected.dtype
        assert samples[0].data_ptr() != samples[1].data_ptr()

    def test_rope_sample_unchanged(self):
        """'rope': one embedding from the model's rotary module, shared by every layer."""
        graphed, _ = self._graphed_and_reference("rope")
        weight = torch.randn(_MBS, _SEQ, _VOCAB, device="cuda")
        _run_gpt(graphed, *_gpt_batch("rope", 0), weight)

        _, kwargs = self.helper._get_cuda_graph_input_data()
        samples = [sample["rotary_pos_emb"] for sample in kwargs["sample_kwargs"]]
        assert all(sample is samples[0] for sample in samples)
        torch.testing.assert_close(samples[0], graphed.rotary_pos_emb(_SEQ), rtol=0, atol=0)

    def test_replay_rotary_mismatch_raises(self):
        """Captured without an eager step, the mrope graphs have no rotary input: replaying
        them with the model's embedding raises instead of running without it. A graph
        captured with the input raises when a replay passes none."""
        graphed, _ = self._graphed_and_reference("mrope")
        weight = torch.randn(_MBS, _SEQ, _VOCAB, device="cuda")
        self.helper.create_cudagraphs()
        with pytest.raises(RuntimeError, match="captured without a rotary_pos_emb input"):
            _run_gpt(graphed, *_gpt_batch("mrope", 0), weight)
        for layer in graphed.decoder.layers:
            assert layer._cuda_graph_has_rotary_pos_emb is False

        layer = graphed.decoder.layers[0]
        layer._cuda_graph_has_rotary_pos_emb = True
        hidden_states = torch.zeros(_SEQ, _MBS, _HIDDEN, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(RuntimeError, match="the replay passes none"):
            layer(hidden_states, attention_mask=None, rotary_pos_emb=None)


class _VisionModel(torch.nn.Module):
    """A vision encoder whose layers take a 2D rotary embedding built outside them."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.decoder = TransformerBlock(
            config, get_vit_layer_with_transformer_engine_spec(), post_layer_norm=False
        )


class _VisionLanguageModel(torch.nn.Module):
    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model


def _vision_rotary_pos_emb(grid_h, grid_w):
    """Qwen-VL style 2D RoPE of an h x w patch grid, zero past the image: [s, 1, 1, d] fp32."""
    dim = _HEAD_DIM // 2
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    rows = torch.arange(grid_h).repeat_interleave(grid_w).float()
    cols = torch.arange(grid_w).repeat(grid_h).float()
    freqs = torch.cat([torch.outer(rows, inv_freq), torch.outer(cols, inv_freq)], dim=-1)
    rotary_pos_emb = torch.zeros(_VISION_SEQ, _HEAD_DIM)
    rotary_pos_emb[: freqs.shape[0]] = torch.cat([freqs, freqs], dim=-1)
    return rotary_pos_emb[:, None, None, :].cuda()


def _build_vision(cuda_graph_impl, seed=4321):
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    config = _config(
        cuda_graph_impl,
        cuda_graph_modules=(
            [CudaGraphModule.attn, CudaGraphModule.mlp] if cuda_graph_impl != "none" else []
        ),
    )
    set_attention_backend(config)  # as VisionModule does
    vision_model = _VisionModel(config).cuda()
    vision_model.train()
    return vision_model


def _run_vision(vision_model, hidden_states, rotary_pos_emb, weight):
    vision_model.zero_grad(set_to_none=True)
    hidden_states = hidden_states.detach().clone().requires_grad_(True)
    output = vision_model.decoder(
        hidden_states=hidden_states, attention_mask=None, rotary_pos_emb=rotary_pos_emb
    )
    # Copy the output before the backward: a graph's output is a static buffer that the graphs
    # of the backward pass may reuse.
    output_copy = output.detach().float().clone()
    (output.float() * weight).sum().backward()
    torch.cuda.synchronize()
    grads = {
        name: param.grad.detach().float().clone()
        for name, param in vision_model.named_parameters()
        if param.grad is not None
    }
    grads["hidden_states"] = hidden_states.grad.detach().float().clone()
    return output_copy, grads


class TestVisionRotaryGraphInputs(_CudaGraphTestBase):
    """VisionTECudaGraphHelper captures the 2D rotary embedding the layers saw in eager steps."""

    def _setup(self):
        self._init_microbatches(1)
        graphed = _build_vision("transformer_engine")
        reference = _build_vision("none", seed=77)
        reference.load_state_dict(graphed.state_dict())
        self.helper = VisionTECudaGraphHelper(
            model=[_VisionLanguageModel(graphed)],
            vision_config=graphed.config,
            vision_seq_length=_VISION_SEQ,
            micro_batch_size=1,
            num_microbatches=1,
        )
        generator = torch.Generator().manual_seed(5)
        hidden_states = torch.randn(_VISION_SEQ, 1, _HIDDEN, generator=generator)
        weight = torch.randn(_VISION_SEQ, 1, _HIDDEN, generator=generator)
        return graphed, reference, hidden_states.to(torch.bfloat16).cuda(), weight.cuda()

    def test_vision_graphs_match_eager(self):
        graphed, reference, hidden_states, weight = self._setup()
        rope_a, rope_b = _vision_rotary_pos_emb(4, 8), _vision_rotary_pos_emb(8, 5)
        eager_a = _run_vision(reference, hidden_states, rope_a, weight)
        eager_b = _run_vision(reference, hidden_states, rope_b, weight)
        _run_vision(graphed, hidden_states, rope_a, weight)  # the eager step before the capture
        self.helper.create_cudagraphs()

        _assert_same(_run_vision(graphed, hidden_states, rope_a, weight), eager_a)
        _assert_same(_run_vision(graphed, hidden_states, rope_b, weight), eager_b)
        _assert_different(eager_a, eager_b)
        _, sample_kwargs = self.helper._get_sample_arguments(order=[1, -1])
        for sample in sample_kwargs:
            assert sample["rotary_pos_emb"].shape == (_VISION_SEQ, 1, 1, _HEAD_DIM)
            assert sample["rotary_pos_emb"].dtype == torch.float32

    def test_capture_without_eager_step_raises_on_rotary(self):
        """Without an eager step, the graphs have no rotary input, and a replay with one raises
        instead of silently running without it; a replay without one still works."""
        graphed, _, hidden_states, weight = self._setup()
        self.helper.create_cudagraphs()
        with pytest.raises(RuntimeError, match="captured without a rotary_pos_emb input"):
            _run_vision(graphed, hidden_states, _vision_rotary_pos_emb(4, 8), weight)
        _run_vision(graphed, hidden_states, None, weight)
