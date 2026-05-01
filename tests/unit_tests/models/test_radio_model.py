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
        return SimpleNamespace(patch_dim=self.PATCH_DIM, temporal_patch_dim=self.TEMPORAL_PATCH_DIM)

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

        assert state_dict["vision.embedder.weight"].shape == (self.HIDDEN, self._expected_3d_in())

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


class TestRADIOTrainOverride:
    """``RADIOViTModel.train()`` must respect ``force_eval_mode``."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_force_eval_mode_keeps_model_in_eval_after_train(self):
        # Construct with force_eval_mode=True; calling .train() must not flip
        # back to training mode.
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            force_eval_mode=True,
        )
        # eval() was called by the constructor.
        assert model.training is False

        returned = model.train(True)

        # The override returns ``self`` so chaining still works.
        assert returned is model
        # And the model is still in eval mode.
        assert model.training is False

    @pytest.mark.internal
    def test_train_normally_when_force_eval_mode_off(self):
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            force_eval_mode=False,
        )
        model.eval()
        assert model.training is False
        model.train(True)
        assert model.training is True


class TestPixelShuffleNonSquare:
    """``llava_model.pixel_shuffle`` extended to take ``h`` / ``w`` for
    non-square (dynamic-res) patch grids."""

    @pytest.mark.internal
    def test_square_default_path(self):
        from megatron.core.models.multimodal.llava_model import pixel_shuffle

        # 16 patches ⇒ default sq=4. scale=0.5 ⇒ output = 4 patches × 4× hidden.
        x = torch.randn(2, 16, 8)
        out = pixel_shuffle(x)
        assert out.shape == (2, 4, 32)

    @pytest.mark.internal
    def test_non_square_h_w_required_for_dynamic_res(self):
        from megatron.core.models.multimodal.llava_model import pixel_shuffle

        # 12 patches arranged as 3×4 (non-square). With h, w supplied the
        # function must accept this and produce the shuffled output.
        x = torch.randn(1, 12, 4)
        out = pixel_shuffle(x, h=3, w=4)
        # scale=0.5 ⇒ each spatial dim halves ⇒ output area = (3*4)/(2*2) = 3.
        assert out.shape == (1, 3, 16)

    @pytest.mark.internal
    def test_h_w_mismatch_raises(self):
        from megatron.core.models.multimodal.llava_model import pixel_shuffle

        x = torch.randn(1, 12, 4)
        # Mismatch: h*w != patches.
        with pytest.raises(AssertionError):
            pixel_shuffle(x, h=2, w=2)


class TestRADIODynamicResAndTemporal:
    """Forward-path coverage for the new dynamic-resolution and temporal
    compression code paths.

    These exercise model construction with the new flags and the lighter
    branches in ``forward`` / ``_apply_temporal_grouping`` that don't require
    a real image + valid PackedSeqParams round-trip.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=2, hidden_size=64, num_attention_heads=4, use_cpu_initialization=True
        )
        self.layer_spec = get_gpt_layer_with_transformer_engine_spec()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def test_constructor_with_temporal_patch_dim_uses_3d_embedder(self):
        """``temporal_patch_dim > 1`` makes the embedder accept 3*T*P*P inputs."""
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            temporal_patch_dim=2,
        )
        # Embedder input dim = 3 * temporal_patch_dim * patch_dim * patch_dim.
        expected_in = 3 * 2 * 14 * 14
        assert model.embedder.input_size == expected_in
        assert not hasattr(model, "video_embedder")

    @pytest.mark.internal
    def test_constructor_with_separate_video_embedder_creates_both(self):
        """``separate_video_embedder=True`` must construct an extra ``video_embedder``."""
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            temporal_patch_dim=2,
            separate_video_embedder=True,
        )
        # Image embedder is 2D (P*P*3); video embedder is 3D (T*P*P*3).
        assert model.embedder.input_size == 3 * 14 * 14
        assert model.video_embedder.input_size == 3 * 2 * 14 * 14

    @pytest.mark.internal
    def test_constructor_with_temporal_ckpt_compat_registers_pre_hook(self):
        """``temporal_ckpt_compat=True`` registers the 2D→3D state-dict pre-hook."""
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            temporal_patch_dim=2,
            temporal_ckpt_compat=True,
        )
        # PyTorch stores load-state-dict pre-hooks on _load_state_dict_pre_hooks
        # (an OrderedDict). Verify at least one hook was registered.
        assert len(model._load_state_dict_pre_hooks) >= 1

    @pytest.mark.internal
    def test_dynamic_resolution_constructor(self):
        """``dynamic_resolution=True`` toggles the corresponding attribute."""
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            dynamic_resolution=True,
        )
        assert model.dynamic_resolution is True

    @pytest.mark.internal
    def test_align_corners_parameterization_sets_attributes(self):
        """The new align_corners knobs must be stored on the model."""
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            interpolate_align_corners=True,
            grid_sample_align_corners=False,
        )
        assert model.interpolate_align_corners is True
        assert model.grid_sample_align_corners is False

    @pytest.mark.internal
    def test_pg_collection_and_vp_stage_propagate_to_decoder(self):
        """Trintamaki review fix: pg_collection + vp_stage must reach the inner
        TransformerBlock so VP/PP awareness is preserved."""
        model = RADIOViTModel(
            self.transformer_config,
            self.layer_spec,
            img_h=224,
            img_w=224,
            patch_dim=14,
            add_class_token=False,
            vp_stage=0,
        )
        # Both attrs are set on the RADIO model itself...
        assert hasattr(model, "pg_collection")
        assert model.vp_stage == 0
        # ...and the inner TransformerBlock.vp_stage matches.
        assert getattr(model.decoder, "vp_stage", None) == 0


class TestApplyTemporalGrouping:
    """Direct unit tests for ``RADIOViTModel._apply_temporal_grouping``.

    Like the state-dict pre-hook tests, we use a ``SimpleNamespace`` stub
    instead of constructing a full RADIO model.
    """

    PATCH_DIM = 14
    TEMPORAL_PATCH_DIM = 2

    def _stub(self):
        return SimpleNamespace(patch_dim=self.PATCH_DIM, temporal_patch_dim=self.TEMPORAL_PATCH_DIM)

    def _make_global(self, num_frames_list, hidden=8):
        """Build an [1, total_patches, hidden] tensor and matching imgs_sizes."""
        per_img = self.PATCH_DIM * self.PATCH_DIM
        chunks = []
        for i, _ in enumerate(num_frames_list):
            for f in range(num_frames_list[i]):
                chunks.append(torch.full((per_img, hidden), float(i * 100 + f)))
        global_t = torch.cat(chunks, dim=0).unsqueeze(0)
        total_frames = sum(num_frames_list)
        imgs_sizes = torch.tensor([[self.PATCH_DIM, self.PATCH_DIM]] * total_frames)
        return global_t, imgs_sizes

    @pytest.mark.internal
    def test_image_only_replicates_or_skips_per_flag(self):
        """A single-frame "video" (i.e. an image): tubelet collapses to a copy."""
        global_t, imgs_sizes = self._make_global([1])
        # skip_image_duplication=False ⇒ the single frame is replicated T times along last dim.
        x_grouped, new_sizes, new_nf, _packed, is_image = RADIOViTModel._apply_temporal_grouping(
            self._stub(),
            x=global_t,
            imgs_sizes=imgs_sizes,
            num_frames=[1],
            packed_seq_params=None,
            skip_image_duplication=False,
        )
        assert is_image == [True]
        assert new_nf == [1]
        # x_grouped is a tensor (concat), shape: [1, per_img, hidden * T].
        per_img = self.PATCH_DIM * self.PATCH_DIM
        assert x_grouped.shape == (1, per_img, 8 * self.TEMPORAL_PATCH_DIM)
        # The single image "size" entry survives.
        assert new_sizes.shape == (1, 2)

    @pytest.mark.internal
    def test_video_groups_consecutive_frames_into_tubelets(self):
        """A 4-frame video at T=2 yields 2 tubelets, each concatenating 2 frames."""
        global_t, imgs_sizes = self._make_global([4])
        x_grouped, new_sizes, new_nf, _packed, is_image = RADIOViTModel._apply_temporal_grouping(
            self._stub(),
            x=global_t,
            imgs_sizes=imgs_sizes,
            num_frames=[4],
            packed_seq_params=None,
            skip_image_duplication=False,
        )
        assert is_image == [False, False]
        assert new_nf == [2]  # padded_nf // T = 4 // 2 = 2
        per_img = self.PATCH_DIM * self.PATCH_DIM
        # Two tubelets, each of width hidden*T.
        assert x_grouped.shape == (1, 2 * per_img, 8 * self.TEMPORAL_PATCH_DIM)

    @pytest.mark.internal
    def test_partial_tubelet_replicates_last_frame(self):
        """A 3-frame video at T=2 needs the last frame replicated to fill the
        2nd tubelet. Verify the replicated frame's value matches the original."""
        global_t, imgs_sizes = self._make_global([3])
        x_grouped, _new_sizes, new_nf, _packed, is_image = RADIOViTModel._apply_temporal_grouping(
            self._stub(),
            x=global_t,
            imgs_sizes=imgs_sizes,
            num_frames=[3],
            packed_seq_params=None,
            skip_image_duplication=False,
        )
        # padded_nf = 4 ⇒ 2 tubelets.
        assert new_nf == [2]
        assert is_image == [False, False]
        # Second tubelet (positions [per_img:2*per_img]) was built from frames
        # [2, 2] (last frame replicated). The chunk values per frame were
        # 0, 1, 2 (from _make_global with i=0). After concat along hidden,
        # the second half (last 8 of the 16-wide hidden) equals frame index 2.
        per_img = self.PATCH_DIM * self.PATCH_DIM
        second_tubelet = x_grouped[0, per_img : 2 * per_img]  # [per_img, 16]
        # Both halves of the hidden dim should equal 2.0 (the last frame value).
        assert torch.all(second_tubelet[:, :8] == 2.0)
        assert torch.all(second_tubelet[:, 8:] == 2.0)

    @pytest.mark.internal
    def test_assert_on_imgs_sizes_count_mismatch(self):
        """``imgs_sizes`` must have one entry per (ungrouped) frame."""
        global_t, _imgs_sizes = self._make_global([4])
        # Provide a wrong-shape imgs_sizes (only 2 entries instead of 4).
        wrong_sizes = torch.tensor([[self.PATCH_DIM, self.PATCH_DIM]] * 2)
        with pytest.raises(AssertionError, match="one entry per frame"):
            RADIOViTModel._apply_temporal_grouping(
                self._stub(),
                x=global_t,
                imgs_sizes=wrong_sizes,
                num_frames=[4],
                packed_seq_params=None,
                skip_image_duplication=False,
            )

    @pytest.mark.internal
    def test_skip_image_duplication_returns_list_of_chunks(self):
        """``skip_image_duplication=True`` returns a list (not a concat tensor)
        so ``separate_video_embedder`` can route image chunks to a different embedder."""
        global_t, imgs_sizes = self._make_global([1])
        x_grouped, _new_sizes, _new_nf, _packed, is_image = RADIOViTModel._apply_temporal_grouping(
            self._stub(),
            x=global_t,
            imgs_sizes=imgs_sizes,
            num_frames=[1],
            packed_seq_params=None,
            skip_image_duplication=True,
        )
        assert is_image == [True]
        assert isinstance(x_grouped, list)
        assert len(x_grouped) == 1
