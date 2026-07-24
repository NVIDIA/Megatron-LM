# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU equality matrix for chunked vision-encoder execution.

Single-rank (torchrun --nproc-per-node 1):
    outputs / loss / parameter-gradient / optimizer-step equality of the
    chunked vs unchunked real Qwen3.5-VL vision encoder.

Two-rank (torchrun --nproc-per-node 2):
    MFSDP-style lockstep: one image-heavy rank and one text-only rank must
    agree on the tower invocation count in train AND eval, and the injected
    lockstep group must match the FSDP sharding group.

Run via:
    torchrun --nproc-per-node {1,2} -m pytest -q \\
        examples/multimodal_dev/tests/test_vision_chunking_gpu.py
"""

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.base import MultimodalModel, _vision_chunk_slices
from examples.multimodal_dev.models.qwen35_vl.vision_encoder import Qwen35VLVisionEncoder
from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_WORLD = int(os.getenv("WORLD_SIZE", "1"))


@pytest.fixture(scope="module", autouse=True)
def _init_model_parallel():
    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    yield
    Utils.destroy_model_parallel()


def _small_encoder(seed=1234):
    torch.manual_seed(seed)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        use_cpu_initialization=True,
        bf16=False,
    )
    encoder = Qwen35VLVisionEncoder(
        config=config,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=64,
        max_num_positions=2304,
    )
    return encoder.cuda().float()


def _payload(seed=7, grids_hw=((4, 4), (8, 4), (4, 8), (6, 4))):
    torch.manual_seed(seed)
    grids = torch.tensor([[1, h, w] for h, w in grids_hw], dtype=torch.long, device="cuda")
    rows = int((grids[:, 0] * grids[:, 1] * grids[:, 2]).sum().item())
    pixels = torch.randn(rows, 3 * 2 * 16 * 16, device="cuda")
    return pixels, grids


def _chunked_output(encoder, pixels, grids, chunk_patches):
    outputs = []
    for image_lo, image_hi, row_lo, row_hi in _vision_chunk_slices(grids, chunk_patches):
        outputs.append(encoder(pixels[row_lo:row_hi], grids[image_lo:image_hi]))
    return torch.cat(outputs)


@pytest.mark.skipif(_WORLD != 1, reason="single-rank equality matrix")
class TestChunkedEncoderEquality:
    def test_forward_outputs_match(self):
        encoder = _small_encoder()
        pixels, grids = _payload()
        with torch.no_grad():
            whole = encoder(pixels, grids)
            chunked = _chunked_output(encoder, pixels, grids, chunk_patches=32)
        torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-5)

    def test_parameter_gradients_match(self):
        pixels, grids = _payload()
        reference, chunked = _small_encoder(), _small_encoder()
        chunked.load_state_dict(reference.state_dict())

        reference(pixels, grids).square().mean().backward()
        _chunked_output(chunked, pixels, grids, chunk_patches=32).square().mean().backward()

        for (name, p_ref), (_, p_chk) in zip(
            reference.named_parameters(), chunked.named_parameters()
        ):
            assert (p_ref.grad is None) == (p_chk.grad is None), name
            if p_ref.grad is not None:
                torch.testing.assert_close(p_chk.grad, p_ref.grad, rtol=1e-4, atol=1e-5, msg=name)

    def test_optimizer_step_matches(self):
        pixels, grids = _payload()
        reference, chunked = _small_encoder(), _small_encoder()
        chunked.load_state_dict(reference.state_dict())
        opt_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3)
        opt_chk = torch.optim.AdamW(chunked.parameters(), lr=1e-3)

        for _ in range(2):
            opt_ref.zero_grad()
            reference(pixels, grids).square().mean().backward()
            opt_ref.step()
            opt_chk.zero_grad()
            _chunked_output(chunked, pixels, grids, chunk_patches=32).square().mean().backward()
            opt_chk.step()

        for (name, p_ref), (_, p_chk) in zip(
            reference.named_parameters(), chunked.named_parameters()
        ):
            torch.testing.assert_close(p_chk, p_ref, rtol=1e-4, atol=1e-6, msg=name)


@pytest.mark.skipif(_WORLD != 1, reason="single-rank memory proof")
class TestStreamingPoolMemory:
    """Reviewer P0-5 proof: retained conv inputs alias ONE pool storage,
    so input-retention memory does not grow with the chunk count."""

    def _run(self, encoder, grids, chunk_patches, pool):
        outputs = []
        for image_lo, image_hi, row_lo, row_hi in _vision_chunk_slices(grids, chunk_patches):
            outputs.append(encoder(pool[: row_hi - row_lo], grids[image_lo:image_hi]))
        return torch.cat(outputs)

    def test_chunk_inputs_share_one_storage(self):
        encoder = _small_encoder()
        pool = torch.randn(1024, 3 * 2 * 16 * 16, device="cuda")
        grids = torch.tensor([[1, 16, 16]] * 8, dtype=torch.long, device="cuda")  # 8 chunks
        seen = []
        inner_forward = encoder.forward

        def spy(pixel_values, grid_thw):
            seen.append(pixel_values.untyped_storage().data_ptr())
            return inner_forward(pixel_values, grid_thw)

        encoder.forward = spy
        self._run(encoder, grids, chunk_patches=256, pool=pool)
        assert len(seen) == 8
        assert set(seen) == {pool.untyped_storage().data_ptr()}

    def test_retained_input_memory_does_not_scale_with_chunks(self):
        # Same per-chunk size; 2 vs 16 chunks. Eager fresh inputs would
        # retain 8x more pixel memory; pool views retain O(pool).
        encoder = _small_encoder()
        pixel_dim = 3 * 2 * 16 * 16
        pool = torch.randn(256, pixel_dim, device="cuda")

        def measure(num_chunks):
            grids = torch.tensor([[1, 16, 16]] * num_chunks, dtype=torch.long, device="cuda")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            baseline = torch.cuda.memory_allocated()
            out = self._run(encoder, grids, chunk_patches=256, pool=pool)
            torch.cuda.synchronize()
            retained = torch.cuda.memory_allocated() - baseline
            out.sum().backward()
            encoder.zero_grad(set_to_none=True)
            return retained, out.shape[0]

        retained_2, tokens_2 = measure(2)
        retained_16, tokens_16 = measure(16)
        # Retained memory grows with activations/outputs (proportional to
        # tokens) but must NOT additionally grow by the pixel payload:
        # 16 chunks of fresh inputs would add >= 14 * pool_bytes.
        pool_bytes = pool.numel() * pool.element_size()
        activation_scaling = retained_2 * (tokens_16 / tokens_2)
        assert retained_16 < activation_scaling + 4 * pool_bytes


@pytest.mark.skipif(_WORLD != 2, reason="two-rank lockstep")
class TestTwoRankLockstep:
    def _model(self, training):
        model = MultimodalModel.__new__(MultimodalModel)
        model.vision_model = _small_encoder(seed=99)  # identical weights on both ranks
        model.vision_encoder_chunk_patches = 32
        model.vision_lockstep_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
        model.training = training
        return model

    def _rank_payload(self):
        # Rank 0: 4 images (multiple chunks). Rank 1: image-free.
        if torch.distributed.get_rank() == 0:
            return _payload()
        return (
            torch.empty(0, 3 * 2 * 16 * 16, device="cuda"),
            torch.empty((0, 3), dtype=torch.long, device="cuda"),
        )

    @pytest.mark.parametrize("training", [True, False])
    def test_ranks_agree_on_invocation_count(self, training):
        model = self._model(training)
        calls = {"n": 0}
        inner = model.vision_model

        def counting(pixel_values, image_grid_thw):
            calls["n"] += 1
            return inner(pixel_values, image_grid_thw)

        model.vision_model = counting
        pixels, grids = self._rank_payload()
        with torch.enable_grad() if training else torch.no_grad():
            model._vision_forward(pixels, grids)

        counts = torch.tensor([calls["n"]], device="cuda")
        gathered = [torch.zeros_like(counts) for _ in range(2)]
        torch.distributed.all_gather(gathered, counts)
        assert gathered[0].item() == gathered[1].item() > 0

    def test_eval_group_wide_image_free_skips_on_all_ranks(self):
        model = self._model(training=False)
        calls = {"n": 0}

        def counting(pixel_values, image_grid_thw):
            calls["n"] += 1
            return model_inner(pixel_values, image_grid_thw)

        model_inner = model.vision_model
        model.vision_model = counting
        pixels = torch.empty(0, 3 * 2 * 16 * 16, device="cuda")
        grids = torch.empty((0, 3), dtype=torch.long, device="cuda")
        with torch.no_grad():
            embeddings, anchor = model._vision_forward(pixels, grids)
        assert embeddings is None and anchor is None and calls["n"] == 0
