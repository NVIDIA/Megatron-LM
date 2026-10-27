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
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
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


def _build_block(tp: int, *, fuse: bool, disable_nvls: bool) -> TransformerBlock:
    """Smallest inference-optimized GPT block, without a final norm, built on the GPU."""
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
    )
    model_parallel_cuda_manual_seed(123)
    block = TransformerBlock(config, get_gpt_layer_with_inference_spec(), post_layer_norm=False)
    block.eval()
    return block


def _build_unfused(tp: int, *, disable_nvls: bool) -> TransformerBlock:
    """Flag-off block with non-trivial norm gains, so a norm with the wrong weights shows."""
    block = _build_block(tp, fuse=False, disable_nvls=disable_nvls)
    gen = torch.Generator(device="cpu").manual_seed(4321)
    with torch.no_grad():
        for name, param in block.named_parameters():
            if name.endswith("layer_norm_weight"):
                param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=gen))
    return block


def _build_pair(tp: int, *, disable_nvls: bool):
    """(flag-off block, flag-on block) with identical weights."""
    unfused = _build_unfused(tp, disable_nvls=disable_nvls)
    fused = _build_block(tp, fuse=True, disable_nvls=disable_nvls)
    fused_params = dict(fused.named_parameters())
    with torch.no_grad():
        # Copy in place: the fused wiring holds the norm weights by reference.
        for name, param in unfused.named_parameters():
            fused_params.pop(name).copy_(param)
    assert not fused_params, f"parameters only in the fused block: {sorted(fused_params)}"
    return unfused, fused


def _input_shard(tp: int) -> torch.Tensor:
    """This rank's sequence-parallel shard of a fixed N(0, 1) [S, B, H] input."""
    gen = torch.Generator(device="cpu").manual_seed(1234)
    x = torch.randn(_S, _B, _H, generator=gen).to(torch.bfloat16)
    rank = parallel_state.get_tensor_model_parallel_rank()
    return x.chunk(tp, dim=0)[rank].contiguous().cuda()


def _forward(block: TransformerBlock, x: torch.Tensor, *, inference: bool = True):
    """Run the block on a copy of x; the fused kernel updates the residual in place."""
    with torch.no_grad():
        if inference:
            with InferenceMode.active():
                out = block(hidden_states=x.clone(), attention_mask=None)
        else:
            block.train()
            try:
                out = block(hidden_states=x.clone(), attention_mask=None)
            finally:
                block.eval()
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


class TestFusedTPCommunication:
    """Flag-on output equals flag-off output on every path of the row-parallel layer."""

    def _init(self, tp: int):
        if Utils.world_size % tp != 0:
            pytest.skip(f"requires a world size divisible by TP={tp}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        SymmetricMemoryManager.destroy()
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        torch.cuda.synchronize()
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("tp", [1, 2], ids=["tp1", "tp2"])
    def test_unfused_inference_matches_training_path(self, tp):
        """The flag-off reference equals the TE training forward and keeps the residual."""
        self._init(tp)
        block = _build_unfused(tp, disable_nvls=True)
        x = _input_shard(tp)
        out = _forward(block, x)
        _assert_close(out, _forward(block, x, inference=False), "inference vs training path")

        # The sub-layers contribute visibly but the residual dominates the output.
        diff_sq = (out - x.float()).pow(2).sum().reshape(1)
        res_sq = x.float().pow(2).sum().reshape(1)
        dist.all_reduce(diff_sq)
        dist.all_reduce(res_sq)
        ratio = (diff_sq / res_sq).sqrt().item()
        assert 0.01 < ratio < 0.9, f"||out - x|| / ||x|| = {ratio:.3f}"

    @pytest.mark.parametrize(
        "tp, disable_nvls, batch_invariant, needs_nvls, fused_calls, norm_calls",
        [
            # TP=1: no reduce-scatter; every column-parallel layer applies its own norm.
            (1, False, False, False, 0, 2 * _NUM_LAYERS),
            # NCCL reduce-scatter because the Triton NVLS kernels are disabled.
            (2, True, False, False, 0, 2 * _NUM_LAYERS),
            # NCCL reduce-scatter because batch-invariant mode forces it.
            (2, False, True, False, 0, 2 * _NUM_LAYERS),
            # The fused kernel runs in every linear_proj / linear_fc2 and leaves the next
            # layer's input in symmetric memory; only layer 0's linear_qkv applies its norm.
            (2, False, False, True, 2 * _NUM_LAYERS, 1),
        ],
        ids=["tp1", "tp2_nccl", "tp2_batch_invariant", "tp2_nvls"],
    )
    def test_fused_matches_unfused(
        self, tp, disable_nvls, batch_invariant, needs_nvls, fused_calls, norm_calls
    ):
        self._init(tp)
        if needs_nvls:
            _require_nvls()
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
        assert fused_spy.call_count == fused_calls
        assert norm_spy.call_count == norm_calls
        _assert_close(actual, expected, "fused TP communication vs unfused")

    @pytest.mark.parametrize("fuse", [False, True], ids=["unfused", "fused"])
    def test_output_independent_of_stale_symm_buffer(self, fuse):
        """With the NCCL reduce-scatter, no layer may read symmetric memory it did not write."""
        tp = 2
        self._init(tp)
        buf = _require_symm_handle()
        block = _build_block(tp, fuse=fuse, disable_nvls=True)
        x = _input_shard(tp)
        outputs = []
        for fill in (0.0, 1.0):
            buf.symm_buffer.view(torch.bfloat16).fill_(fill)
            torch.cuda.synchronize()
            dist.barrier()
            outputs.append(_forward(block, x))
        _assert_close(outputs[1], outputs[0], "outputs after filling the buffer with 1.0 vs 0.0")
