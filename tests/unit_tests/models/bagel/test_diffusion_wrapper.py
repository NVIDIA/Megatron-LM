# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from examples.mimo_bagel.diffusion import diffusion_wrapper as diffusion_wrapper_module
from examples.mimo_bagel.diffusion.diffusion_wrapper import DiffusionWrapper
from examples.mimo_bagel.diffusion.embeddings import TimestepEmbedder
from examples.mimo_bagel.utils.data_helpers import bagel_process_gen_data


class _CpuFakeVAE(torch.nn.Module):
    """Small load_ae stand-in whose ``cuda`` method is safe on CPU CI."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        self.last_input_dtype = None

    def cuda(self):
        return self

    def encode(self, images):
        self.last_input_dtype = images.dtype
        return torch.zeros((images.shape[0], 16, 4, 6), dtype=torch.bfloat16)


def test_init_vae_preserves_native_fp32_parameters(monkeypatch):
    vae = _CpuFakeVAE()
    params = SimpleNamespace(downsample=8, z_channels=16)
    monkeypatch.setattr(diffusion_wrapper_module, "load_ae", lambda _: (vae, params))

    wrapper = DiffusionWrapper()
    wrapper.init_vae("unused.safetensors", latent_patch_size=2, dtype=torch.bfloat16)

    assert {parameter.dtype for parameter in wrapper.vae.parameters()} == {torch.float32}
    assert wrapper.dtype == torch.bfloat16
    assert wrapper.latent_downsample == 16


def test_vae_encode_uses_native_autocast_input_and_preserves_output_contract(monkeypatch):
    vae = _CpuFakeVAE()
    params = SimpleNamespace(downsample=8, z_channels=16)
    monkeypatch.setattr(diffusion_wrapper_module, "load_ae", lambda _: (vae, params))
    autocast_calls = []
    autocast_state = {"active": False}
    einsum_autocast_states = []
    eager_einsum = torch.einsum

    @contextmanager
    def recording_autocast(device_type, *, enabled, dtype):
        autocast_calls.append((device_type, enabled, dtype))
        autocast_state["active"] = True
        try:
            yield
        finally:
            autocast_state["active"] = False

    def recording_einsum(equation, *operands):
        einsum_autocast_states.append(autocast_state["active"])
        return eager_einsum(equation, *operands)

    monkeypatch.setattr(torch.amp, "autocast", recording_autocast)
    monkeypatch.setattr(torch, "einsum", recording_einsum)
    wrapper = DiffusionWrapper()
    wrapper.init_vae("unused.safetensors", latent_patch_size=2, dtype=torch.bfloat16)

    packed = wrapper.vae_encode(
        torch.zeros((1, 3, 32, 48), dtype=torch.float32),
        [(2, 3)],
    )

    assert autocast_calls == [("cuda", True, torch.bfloat16)]
    assert einsum_autocast_states == [True]
    assert vae.last_input_dtype == torch.float32
    assert {parameter.dtype for parameter in vae.parameters()} == {torch.float32}
    assert packed.dtype == torch.bfloat16
    assert packed.shape == (6, 64)


def test_gen_data_casts_timesteps_before_shift_and_noise(monkeypatch):
    """Match native's BF16 model-input boundary for diffusion arithmetic."""

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self)

    class _RecordingWrapper(DiffusionWrapper):
        def shift_timesteps(self, packed_timesteps):
            self.raw_timesteps = packed_timesteps.detach().clone()
            shifted = super().shift_timesteps(packed_timesteps)
            self.shifted_timesteps = shifted.detach().clone()
            return shifted

        def vae_encode(self, padded_images, patchified_shapes):
            return clean.clone()

        def add_noise(self, packed_latents_clean, shifted_timesteps):
            noisy, noise, target = super().add_noise(
                packed_latents_clean, shifted_timesteps
            )
            self.noise = noise.detach().clone()
            return noisy, noise, target

    wrapper = _RecordingWrapper()
    wrapper.dtype = torch.bfloat16
    wrapper.timestep_shift = 1.0
    raw_timesteps = torch.tensor(
        [-0.31815165281295776, 0.1251, float("-inf"), 1.8751],
        dtype=torch.float32,
    )
    clean = torch.tensor(
        [[0.5546875, 1.359375], [-0.65234375, -0.1484375],
         [0.076171875, 0.7578125], [0.44921875, -0.345703125]],
        dtype=torch.bfloat16,
    )
    batch = {
        "packed_timesteps": raw_timesteps,
        "mse_loss_indexes": torch.ones(4, dtype=torch.bool),
        "packed_vae_token_indexes": torch.arange(4),
        "padded_images": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        "patchified_vae_latent_shapes": [(1, 1)],
        "packed_latent_position_ids": torch.arange(4),
    }

    torch.manual_seed(35168)
    rng_state = torch.get_rng_state()
    loss_inputs, modality_inputs = bagel_process_gen_data(batch, wrapper)

    torch.set_rng_state(rng_state)
    native_raw = raw_timesteps.to(torch.bfloat16)
    native_shifted = torch.sigmoid(native_raw)
    native_noise = torch.randn_like(clean)
    native_noisy = (
        (1 - native_shifted[:, None]) * clean
        + native_shifted[:, None] * native_noise
    )
    native_target = (native_noise - clean)[native_shifted > 0]

    assert torch.equal(wrapper.raw_timesteps, native_raw)
    assert torch.equal(wrapper.shifted_timesteps, native_shifted)
    assert torch.equal(wrapper.noise, native_noise)
    assert torch.equal(modality_inputs["latents"], native_noisy)
    assert torch.equal(loss_inputs["vis_gen_target"], native_target)
    assert modality_inputs["latents"].dtype == torch.bfloat16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_timestep_embedder_precast_is_bitwise_native_autocast():
    torch.manual_seed(20260720)
    module = TimestepEmbedder(hidden_size=896).cuda().to(torch.bfloat16)
    reference_mlp = copy.deepcopy(module.mlp)
    timesteps = torch.randn(513, device="cuda", dtype=torch.float32).sigmoid()

    actual = module(timesteps)
    reference_frequency = module.timestep_embedding(
        timesteps, module.frequency_embedding_size
    )
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        expected = reference_mlp(reference_frequency)

    assert torch.equal(actual, expected)

    output_grad = torch.randn_like(actual)
    actual.backward(output_grad)
    expected.backward(output_grad)
    for actual_parameter, expected_parameter in zip(
        module.mlp.parameters(), reference_mlp.parameters()
    ):
        assert torch.equal(actual_parameter.grad, expected_parameter.grad)

    # This rules out the explicit BF16 pre-cast as a source of native/MCore
    # divergence.  PyTorch autocast selects the same Linear inputs here.
    legacy_mlp = copy.deepcopy(module.mlp)
    for parameter in legacy_mlp.parameters():
        parameter.grad = None
    legacy = legacy_mlp(reference_frequency.to(torch.bfloat16))
    assert torch.equal(actual, legacy)
