# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class TestRADIOViTModel:
    """Test RADIO ViT model."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        self.model = RADIOViTModel(
            transformer_config,
            transformer_layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert isinstance(self.model, RADIOViTModel)

        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 1501824

    def test_set_input_tensor(self):
        # [s, b, h] expected to the transformer.
        expected_shape = (256, 2, 64)
        input_tensor = torch.zeros(expected_shape)

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape == torch.Size(expected_shape)

    def test_forward(self):
        self.model.cuda()

        img = torch.zeros((2, 3, 224, 224)).cuda()

        out = self.model.forward(img)
        assert out.shape == torch.Size([2, 256, 64])

    def test_save_load(self, tmp_path):
        path = tmp_path / "model.pt"
        torch.save(self.model.state_dict(), path)

        self.model.load_state_dict(torch.load(path))


class TestRADIOStateDictPreHooks:
    """Unit tests for the 2D↔3D embedder state-dict pre-hooks.

    These hooks only touch ``self.patch_dim`` and ``self.temporal_patch_dim``,
    so we exercise them via a ``SimpleNamespace`` stub instead of constructing
    a full ``RADIOViTModel`` (which would require a parallel-state init and a
    TransformerBlock we don't need).
    """

    PATCH_DIM = 14
    TEMPORAL_PATCH_DIM = 2
    HIDDEN = 32

    def _stub(self):
        return SimpleNamespace(
            patch_dim=self.PATCH_DIM, temporal_patch_dim=self.TEMPORAL_PATCH_DIM
        )

    def _expected_2d_in(self):
        return 3 * self.PATCH_DIM * self.PATCH_DIM

    def _expected_3d_in(self):
        return 3 * self.TEMPORAL_PATCH_DIM * self.PATCH_DIM * self.PATCH_DIM

    @pytest.mark.internal
    def test_init_embedder_expands_2d_to_3d_and_rescales(self):
        weight_2d = torch.ones((self.HIDDEN, self._expected_2d_in())) * 3.0
        state_dict = {"embedder.weight": weight_2d.clone()}

        RADIOViTModel._state_dict_pre_hook_init_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )

        new_weight = state_dict["embedder.weight"]
        assert new_weight.shape == (self.HIDDEN, self._expected_3d_in())
        # Each 2D tile is repeated ``temporal_patch_dim`` times then divided by
        # the same factor ⇒ each entry equals the original value.
        assert torch.allclose(new_weight, weight_2d.repeat(1, self.TEMPORAL_PATCH_DIM) / 2)
        assert torch.allclose(new_weight, torch.full_like(new_weight, 1.5))

    @pytest.mark.internal
    def test_init_embedder_noop_when_already_3d(self):
        weight_3d = torch.full((self.HIDDEN, self._expected_3d_in()), 0.25)
        state_dict = {"embedder.weight": weight_3d.clone()}

        RADIOViTModel._state_dict_pre_hook_init_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )

        assert torch.equal(state_dict["embedder.weight"], weight_3d)

    @pytest.mark.internal
    def test_init_embedder_noop_when_key_missing(self):
        state_dict = {"unrelated": torch.zeros(1)}
        RADIOViTModel._state_dict_pre_hook_init_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )
        assert list(state_dict.keys()) == ["unrelated"]

    @pytest.mark.internal
    def test_init_embedder_respects_prefix(self):
        weight_2d = torch.full((self.HIDDEN, self._expected_2d_in()), 2.0)
        state_dict = {"vision.embedder.weight": weight_2d.clone()}

        RADIOViTModel._state_dict_pre_hook_init_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix="vision."
        )

        assert state_dict["vision.embedder.weight"].shape == (
            self.HIDDEN,
            self._expected_3d_in(),
        )

    @pytest.mark.internal
    def test_init_video_embedder_creates_from_2d_image_weights(self):
        """When the checkpoint has only a 2D embedder, the hook populates
        video_embedder with the 3D-expanded (and rescaled) copy and leaves
        the image embedder untouched."""
        weight_2d = torch.full((self.HIDDEN, self._expected_2d_in()), 4.0)
        state_dict = {"embedder.weight": weight_2d.clone()}

        RADIOViTModel._state_dict_pre_hook_init_video_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )

        assert torch.equal(state_dict["embedder.weight"], weight_2d)
        video_weight = state_dict["video_embedder.weight"]
        assert video_weight.shape == (self.HIDDEN, self._expected_3d_in())
        assert torch.allclose(video_weight, torch.full_like(video_weight, 2.0))

    @pytest.mark.internal
    def test_init_video_embedder_splits_existing_3d_embedder(self):
        """When the checkpoint has a 3D embedder but no video_embedder, the
        hook clones the 3D weight into video_embedder and compresses the
        image embedder back to 2D by averaging across the temporal axis."""
        # Build a 3D weight where the temporal slices have different values
        # so averaging is distinguishable from simple truncation.
        out = self.HIDDEN
        two_d = self._expected_2d_in()
        # Shape: [out, 3*T*P*P] = [out, T, 3*P*P] after a reshape view.
        slice_a = torch.full((out, two_d), 1.0)
        slice_b = torch.full((out, two_d), 3.0)
        weight_3d = torch.cat([slice_a, slice_b], dim=1)  # temporal patch dim = 2
        assert weight_3d.shape == (out, self._expected_3d_in())
        state_dict = {"embedder.weight": weight_3d.clone()}

        RADIOViTModel._state_dict_pre_hook_init_video_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )

        # video_embedder is an exact copy of the original 3D weight.
        assert state_dict["video_embedder.weight"].shape == (out, self._expected_3d_in())
        assert torch.equal(state_dict["video_embedder.weight"], weight_3d)

        # Image embedder is the per-temporal-slice mean ⇒ (1+3)/2 = 2.0.
        image_weight = state_dict["embedder.weight"]
        assert image_weight.shape == (out, two_d)
        assert torch.allclose(image_weight, torch.full_like(image_weight, 2.0))

    @pytest.mark.internal
    def test_init_video_embedder_copies_bias_when_missing(self):
        weight_2d = torch.full((self.HIDDEN, self._expected_2d_in()), 1.0)
        bias = torch.arange(self.HIDDEN, dtype=torch.float32)
        state_dict = {"embedder.weight": weight_2d.clone(), "embedder.bias": bias.clone()}

        RADIOViTModel._state_dict_pre_hook_init_video_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )

        assert "video_embedder.bias" in state_dict
        assert torch.equal(state_dict["video_embedder.bias"], bias)
        # Ensure it's a copy, not an alias (mutating one must not affect the other).
        state_dict["video_embedder.bias"][0] = -999.0
        assert state_dict["embedder.bias"][0] == 0.0

    @pytest.mark.internal
    def test_init_video_embedder_preserves_existing_video_embedder(self):
        weight_2d = torch.full((self.HIDDEN, self._expected_2d_in()), 1.0)
        existing_video = torch.full((self.HIDDEN, self._expected_3d_in()), 7.0)
        state_dict = {
            "embedder.weight": weight_2d.clone(),
            "video_embedder.weight": existing_video.clone(),
        }

        RADIOViTModel._state_dict_pre_hook_init_video_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )

        # Hook must not overwrite an already-present video_embedder.
        assert torch.equal(state_dict["video_embedder.weight"], existing_video)
        assert torch.equal(state_dict["embedder.weight"], weight_2d)

    @pytest.mark.internal
    def test_init_video_embedder_noop_when_embedder_missing(self):
        state_dict = {"unrelated": torch.zeros(1)}
        RADIOViTModel._state_dict_pre_hook_init_video_embedder(
            self._stub(), module=None, state_dict=state_dict, prefix=""
        )
        assert list(state_dict.keys()) == ["unrelated"]
