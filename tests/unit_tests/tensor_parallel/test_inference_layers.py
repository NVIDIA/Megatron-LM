# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Fused TP communication (``inference_fuse_tp_communication``) in the inference layers.

TransformerLayer hands the residual to ``InferenceRowParallelLinear`` and skips its own
residual add. The row-parallel layer must add the residual whether or not the fused NVLS
reduce-scatter + add + rms-norm + all-gather kernel runs, and the next
``InferenceLayerNormColumnParallelLinear`` may read its input from symmetric memory only when
that kernel ran. Each test compares 2-layer GPT blocks with the flag on and off, using the same
weights, on every path the row-parallel layer can take.
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import megatron.core.tensor_parallel.inference_layers as inference_layers
from megatron.core import parallel_state
from megatron.core.inference.communication.torch_symm_triton.utils import is_device_nvls_capable
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_inference_spec
from megatron.core.tensor_parallel.inference_layers import FusedTPHandoff
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

requires_te = pytest.mark.skipif(
    not inference_layers.HAVE_TE, reason="inference_optimized requires Transformer Engine"
)

_S, _B, _H = 64, 2, 128
_FFN = 512
_HEADS = 4
_NUM_LAYERS = 2

# The residual is N(0, 1) and the sub-layers add about 0.3 of its norm, so a dropped
# residual is far outside this tolerance; it covers bf16 rounding at |x| ~ 4.
_ATOL = 2e-2
_RTOL = 2e-2


def _all_ranks_agree(value: bool) -> bool:
    flag = torch.tensor([float(value)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return flag.item() == 1.0


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str):
    """allclose, reduced over all ranks so that every rank passes or fails together."""
    max_diff = (actual - expected).abs().max().reshape(1)
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    ok = _all_ranks_agree(torch.allclose(actual, expected, atol=_ATOL, rtol=_RTOL))
    assert ok, f"{what}: max |diff| = {max_diff.item():.4f} (atol={_ATOL}, rtol={_RTOL})"


def _build(tp: int, *, fuse: bool, disable_nvls: bool, layer_only: bool = False, **overrides):
    """Smallest inference-optimized GPT block, without a final norm, built on the GPU.

    With ``layer_only``, a single TransformerLayer built the way MultiTokenPredictionLayer
    builds its inner layer; TransformerBlock never configures it for fused TP communication.
    """
    config = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=_H,
        ffn_hidden_size=_FFN,
        num_attention_heads=_HEADS,
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
        use_cpu_initialization=False,
        inference_fuse_tp_communication=fuse,
        inference_disable_triton_nvls_kernels=disable_nvls,
        **overrides,
    )
    model_parallel_cuda_manual_seed(123)
    spec = get_gpt_layer_with_inference_spec()
    if layer_only:
        module = build_module(spec, config=config, layer_number=1, is_mtp_layer=True)
    else:
        module = TransformerBlock(config, spec, post_layer_norm=False)
    module.eval()
    return module


def _build_pair(tp: int, *, disable_nvls: bool, **kwargs):
    """(flag-off, flag-on) modules with identical weights and non-trivial norm gains."""
    unfused = _build(tp, fuse=False, disable_nvls=disable_nvls, **kwargs)
    fused = _build(tp, fuse=True, disable_nvls=disable_nvls, **kwargs)
    gen = torch.Generator(device="cpu").manual_seed(4321)
    fused_params = dict(fused.named_parameters())
    with torch.no_grad():
        for name, param in unfused.named_parameters():
            if name.endswith("layer_norm_weight"):
                # Gains away from 1, so a norm applied with the wrong weights shows.
                param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=gen))
            # Copy in place: the fused wiring holds the norm weights by reference.
            fused_params.pop(name).copy_(param)
    assert not fused_params, f"parameters only in the fused module: {sorted(fused_params)}"
    return unfused, fused


def _input_shard(tp: int) -> torch.Tensor:
    """This rank's sequence-parallel shard of a fixed N(0, 1) [S, B, H] input."""
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(_S, _B, _H, generator=gen).to(torch.bfloat16)
    rank = parallel_state.get_tensor_model_parallel_rank()
    return x.chunk(tp, dim=0)[rank].contiguous().cuda()


def _forward(module, x: torch.Tensor, *, train: bool = False, inference_mode: bool = True):
    """Run the module on a copy of x; the fused kernel updates the residual in place."""
    module.train(train)
    try:
        mode = InferenceMode.active() if inference_mode else contextlib.nullcontext()
        with torch.no_grad(), mode:
            out = module(hidden_states=x.clone(), attention_mask=None)
    finally:
        module.eval()
    if isinstance(out, tuple):  # TransformerLayer returns (hidden_states, context)
        out = out[0]
    out = out.float()
    torch.cuda.synchronize()
    dist.barrier()
    return out


@contextlib.contextmanager
def _count_calls(name: str):
    """Count calls of an inference_layers function without changing its behavior."""
    fn = getattr(inference_layers, name)
    with mock.patch.object(inference_layers, name, wraps=fn) as spy:
        yield spy


def _symm_buffer():
    return SymmetricMemoryManager.get_buffer(
        "tp", process_group=parallel_state.get_tensor_model_parallel_group()
    )


def _require_symm_handle():
    buf = _symm_buffer()  # collective: create on every rank before any skip
    if not _all_ranks_agree(buf.symm_mem_hdl is not None):
        pytest.skip(f"symmetric memory unavailable: {buf.init_failure_reason}")
    return buf


def _require_nvls():
    buf = _symm_buffer()  # collective: create on every rank before any skip
    hdl = buf.symm_mem_hdl
    capable = is_device_nvls_capable(torch.device("cuda", torch.cuda.current_device()))
    if not _all_ranks_agree(capable and hdl is not None and bool(hdl.multicast_ptr)):
        pytest.skip("NVLS multicast unavailable")


# (tp, disable_nvls, needs_nvls) for each path of the row-parallel layer.
_PATHS = {
    # TP=1: no reduce-scatter.
    "tp1": (1, False, False),
    # NCCL reduce-scatter because the Triton NVLS kernels are disabled.
    "tp2_nccl": (2, True, False),
    # NVLS reduce-scatter, fused with the next layer's norm when the wiring allows.
    "tp2_nvls": (2, False, True),
}


@requires_te
class TestFusedTPCommunication:
    """Flag-on output equals flag-off output on every path of the row-parallel layer."""

    def _init(self, path: str):
        tp, disable_nvls, needs_nvls = _PATHS[path]
        if Utils.world_size % tp != 0:
            pytest.skip(f"requires a world size divisible by TP={tp}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        SymmetricMemoryManager.destroy()
        model_parallel_cuda_manual_seed(123)
        if needs_nvls:
            _require_nvls()
        return tp, disable_nvls

    def teardown_method(self, method):
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("path", ["tp1", "tp2_nccl"])
    def test_unfused_inference_matches_training_path(self, path):
        """The flag-off reference equals the TE training forward and keeps the residual."""
        tp, disable_nvls = self._init(path)
        block = _build(tp, fuse=False, disable_nvls=disable_nvls)
        x = _input_shard(tp)
        out = _forward(block, x)
        training = _forward(block, x, train=True, inference_mode=False)
        _assert_close(out, training, "inference vs training path")

        # The sub-layers contribute visibly but the residual dominates the output.
        diff_sq = (out - x.float()).pow(2).sum().reshape(1)
        res_sq = x.float().pow(2).sum().reshape(1)
        dist.all_reduce(diff_sq)
        dist.all_reduce(res_sq)
        ratio = (diff_sq / res_sq).sqrt().item()
        assert 0.01 < ratio < 0.9, f"||out - x|| / ||x|| = {ratio:.3f}"

    @pytest.mark.parametrize(
        "path, batch_invariant, fused_calls, norm_calls",
        [
            # Every column-parallel layer applies its own norm.
            ("tp1", False, 0, 2 * _NUM_LAYERS),
            ("tp2_nccl", False, 0, 2 * _NUM_LAYERS),
            # NCCL reduce-scatter because batch-invariant mode forces it.
            ("tp2_nvls", True, 0, 2 * _NUM_LAYERS),
            # The fused kernel runs in every linear_proj / linear_fc2 and leaves the next
            # layer's input in symmetric memory; only layer 0's linear_qkv applies its norm.
            ("tp2_nvls", False, 2 * _NUM_LAYERS, 1),
        ],
        ids=["tp1", "tp2_nccl", "tp2_batch_invariant", "tp2_nvls"],
    )
    def test_fused_matches_unfused(self, path, batch_invariant, fused_calls, norm_calls):
        tp, disable_nvls = self._init(path)
        unfused, fused = _build_pair(tp, disable_nvls=disable_nvls)
        x = _input_shard(tp)
        mode = (
            set_batch_invariant_mode(True, backend="te_native")
            if batch_invariant
            else contextlib.nullcontext()
        )
        with mode:
            expected = _forward(unfused, x)
            with (
                _count_calls("fused_multimem_rs_add_norm_ag") as fused_spy,
                _count_calls("_te_rms_norm_kernel") as norm_spy,
            ):
                actual = _forward(fused, x)
        _assert_close(actual, expected, "fused TP communication vs unfused")
        assert fused_spy.call_count == fused_calls
        assert norm_spy.call_count == norm_calls

    def test_fusion_is_decided_every_forward(self):
        """A fused step followed by one where linear_proj fuses but linear_fc2 falls back.

        Each linear_fc2 must record that it did not fuse in the second step, or the next
        layer's linear_qkv reads symmetric memory left over from the first step.
        """
        tp, disable_nvls = self._init("tp2_nvls")
        unfused, fused = _build_pair(tp, disable_nvls=disable_nvls)
        x = _input_shard(tp)
        expected = _forward(unfused, x)
        _assert_close(_forward(fused, x), expected, "every linear_proj / linear_fc2 fused")

        for layer in fused.layers:
            # A per-layer runtime fallback, like a buffer too small for this shape.
            layer.mlp.linear_fc2.triton_nvls_kernels_allowed = False
        with (
            _count_calls("fused_multimem_rs_add_norm_ag") as fused_spy,
            _count_calls("_te_rms_norm_kernel") as norm_spy,
        ):
            actual = _forward(fused, x)
        _assert_close(actual, expected, "linear_proj fused, linear_fc2 fell back")
        assert fused_spy.call_count == _NUM_LAYERS
        # Layer 0's linear_qkv and layer 1's linear_qkv after the unfused linear_fc2.
        assert norm_spy.call_count == _NUM_LAYERS

    @pytest.mark.parametrize("path", ["tp1", "tp2_nvls"])
    def test_recompute_mlp_keeps_residual(self, path):
        """linear_fc2 gets the residual on the recompute_mlp branch of _forward_mlp too."""
        tp, disable_nvls = self._init(path)
        unfused, fused = _build_pair(
            tp,
            disable_nvls=disable_nvls,
            recompute_granularity="selective",
            recompute_modules=["mlp"],
        )
        assert all(layer.recompute_mlp for layer in fused.layers)
        x = _input_shard(tp)
        _assert_close(_forward(fused, x), _forward(unfused, x), "recompute_mlp, fused vs unfused")

    @pytest.mark.parametrize("path", ["tp1", "tp2_nccl", "tp2_nvls"])
    def test_unconfigured_layer_keeps_residual(self, path):
        """A layer outside TransformerBlock, like the MTP inner layer, keeps its residual."""
        tp, disable_nvls = self._init(path)
        unfused, fused = _build_pair(tp, disable_nvls=disable_nvls, layer_only=True)
        assert fused.self_attention.linear_proj.next_layer_norm_weights is None
        x = _input_shard(tp)
        with _count_calls("fused_multimem_rs_add_norm_ag") as fused_spy:
            actual = _forward(fused, x)
        _assert_close(actual, _forward(unfused, x), "unconfigured layer, fused vs unfused")
        assert fused_spy.call_count == 0

    @pytest.mark.parametrize("path", ["tp1", "tp2_nccl"])
    def test_training_mode_keeps_bias_dropout_add(self, path):
        """In training mode the layer keeps its residual add, even while InferenceMode is on."""
        tp, disable_nvls = self._init(path)
        unfused, fused = _build_pair(tp, disable_nvls=disable_nvls)
        x = _input_shard(tp)
        actual = _forward(fused, x, train=True)
        expected = _forward(unfused, x, train=True)
        _assert_close(actual, expected, "training mode under InferenceMode, fused vs unfused")
        for layer in fused.layers:
            assert layer.self_attention.linear_proj.residual is None
            assert layer.mlp.linear_fc2.residual is None

    @pytest.mark.parametrize("fuse", [False, True], ids=["unfused", "fused"])
    def test_output_independent_of_stale_symm_buffer(self, fuse):
        """With the NCCL reduce-scatter, no layer may read symmetric memory it did not write."""
        tp, disable_nvls = self._init("tp2_nccl")
        buf = _require_symm_handle()
        block = _build(tp, fuse=fuse, disable_nvls=disable_nvls)
        x = _input_shard(tp)
        outputs = []
        for fill in (0.0, 1.0):
            buf.symm_buffer.view(torch.bfloat16).fill_(fill)
            torch.cuda.synchronize()
            dist.barrier()
            outputs.append(_forward(block, x))
        _assert_close(outputs[1], outputs[0], "outputs after filling the buffer with 1.0 vs 0.0")


class _FakeLayer:
    """Records the fused-TP wiring TransformerBlock gives a layer."""

    def __init__(self, graphed: bool):
        if graphed:
            self.cudagraph_manager = object()
        self.fc2_fused_tp_handoff = FusedTPHandoff()
        self.qkv_fused_tp_handoff = None

    def get_qkv_layer_norm_weights(self):
        return torch.ones(_H)

    def get_fc2_fused_tp_handoff(self):
        return self.fc2_fused_tp_handoff

    def configure_fused_tp_inference(self, qkv_fused_tp_handoff, fc2_next_layer_norm_weights):
        self.qkv_fused_tp_handoff = qkv_fused_tp_handoff


def test_block_links_fused_tp_handoff_only_between_layers_graphed_alike():
    """A QKV trusts the previous FC2's record only if both always replay or run together."""
    layers = [_FakeLayer(graphed=g) for g in (True, True, False, False)]
    TransformerBlock._setup_fused_tp_communication(SimpleNamespace(layers=layers))
    assert layers[0].qkv_fused_tp_handoff is None
    assert layers[1].qkv_fused_tp_handoff is layers[0].fc2_fused_tp_handoff
    assert layers[2].qkv_fused_tp_handoff is None
    assert layers[3].qkv_fused_tp_handoff is layers[2].fc2_fused_tp_handoff
