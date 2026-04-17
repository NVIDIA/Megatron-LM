# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/test_mimo_model.py
'''

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytest.importorskip("modelopt", minversion="0.25")
# modelopt version < 0.27 breaks HF AutoModel.from_pretrained API
# so we need to skip the tests unitl versions are bumped in pyt LTS CI container


class AudioEncoderWrapper(torch.nn.Module):
    """Generic wrapper for audio encoder models that extracts last_hidden_state."""

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = WhisperModel(WhisperConfig()).encoder

    def forward(self, input_features):
        with torch.no_grad():
            return self.encoder(input_features).last_hidden_state


def get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim):
    """Get the submodule spec for the vision modality."""
    vision_layer_spec = get_gpt_layer_with_transformer_engine_spec()

    vision_config = TransformerConfig(
        num_layers=1, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True
    )
    vision_encoder_spec = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": vision_layer_spec,
            "img_h": img_h,
            "img_w": img_w,
            "patch_dim": patch_dim,
        },
    )

    vision_projection_spec = ModuleSpec(
        module=nn.Linear,
        params={
            "in_features": vision_config.hidden_size,
            "out_features": vision_config.hidden_size,
        },
    )

    return ModuleSpec(
        module=VisionModalitySubmodules,
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )


def get_audio_submodules_spec(hidden_size):
    """Get the submodule spec for the audio modality."""
    audio_encoder_spec = ModuleSpec(module=AudioEncoderWrapper, params={})

    audio_projection_spec = ModuleSpec(
        module=nn.Linear,
        params={"in_features": 384, "out_features": hidden_size},  # Whisper tiny hidden size
    )

    return ModuleSpec(
        module=AudioModalitySubmodules,
        submodules={
            "encoders": {"whisper_encoder": audio_encoder_spec},
            "input_projections": [audio_projection_spec],
        },
    )


def get_language_model_spec(hidden_size, vocab_size, seq_len):
    """Get the language model spec."""
    lm_config = TransformerConfig(
        num_layers=2, hidden_size=hidden_size, num_attention_heads=4, use_cpu_initialization=True
    )
    language_layer_spec = get_gpt_layer_with_transformer_engine_spec()
    return ModuleSpec(
        module=GPTModel,
        params={
            "config": lm_config,
            "transformer_layer_spec": language_layer_spec,
            "vocab_size": vocab_size,
            "max_sequence_length": seq_len,
            "pre_process": True,
            "post_process": True,
        },
    )


def get_avlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):
    mimo_config = MimoModelConfig(
        language_model_spec=get_language_model_spec(hidden_size, vocab_size, seq_len),
        modality_submodules_spec={
            "images": get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim),
            "audio": get_audio_submodules_spec(hidden_size),
        },
        special_token_ids=special_token_ids,
    )
    return MimoModel(mimo_config)


def get_vlm_mimo_model(
    hidden_size, vocab_size, seq_len, img_h, img_w, patch_dim, special_token_ids
):
    mimo_config = MimoModelConfig(
        language_model_spec=get_language_model_spec(hidden_size, vocab_size, seq_len),
        modality_submodules_spec={
            "images": get_vision_submodules_spec(hidden_size, img_h, img_w, patch_dim)
        },
        special_token_ids=special_token_ids,
    )
    return MimoModel(mimo_config)


class TestMimoModel:
    """Test the MimoModel class."""

    def setup_method(self, method):
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception:
            pass

        self.hidden_size = 64
        self.batch_size = 2
        self.seq_len = 2048
        self.img_h = 224
        self.img_w = 224
        self.patch_dim = 16
        self.vocab_size = 48000
        self.special_token_ids = {"images": 50257, "audio": 50258}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
        except Exception:
            pass

    def _make_vlm(self):
        return get_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        ).to(self.device)

    def _make_avlm(self):
        return get_avlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            self.special_token_ids,
        ).to(self.device)

    def _make_input_ids(self):
        return torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
        )

    def _make_position_ids(self):
        return (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )

    def test_constructor(self):
        """Test constructor initialization."""
        mimo_model = self._make_avlm()

        assert "images" in mimo_model.modality_submodules
        assert "audio" in mimo_model.modality_submodules
        assert isinstance(mimo_model.modality_submodules["images"], VisionModalitySubmodules)
        assert isinstance(mimo_model.modality_submodules["audio"], AudioModalitySubmodules)
        assert isinstance(mimo_model.language_model, GPTModel)
        assert mimo_model.special_token_ids == self.special_token_ids

    def test_get_text_embeddings(self):
        """Test getting text embeddings."""
        mimo_model = self._make_avlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        text_embeddings = mimo_model.get_text_embeddings(
            input_ids, position_ids, self.special_token_ids
        )
        assert text_embeddings.shape == (self.batch_size * self.seq_len, self.hidden_size)

    def test_forward_text_only(self):
        """Test forward pass with only text input."""
        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=None
        )
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_with_image_modality(self):
        """Test forward pass with text and image input."""
        expected_img_seq_len = (self.img_h // self.patch_dim) * (
            self.img_w // self.patch_dim
        ) + 1  # +1 for CLS token

        num_images = 5
        images_per_sample = [3, 2]
        images = torch.rand(num_images, 3, self.img_h, self.img_w, device=self.device)
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        # Place image special tokens in each batch sample
        image_token_id = self.special_token_ids["images"]
        start_pos = 5
        for b in range(self.batch_size):
            tokens_in_this_batch = images_per_sample[b] * expected_img_seq_len
            input_ids[b, start_pos : start_pos + tokens_in_this_batch] = image_token_id

        modality_inputs = {"images": {"clip_encoder": {"x": images}}}

        mimo_model = self._make_vlm()
        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=modality_inputs
        )
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_with_image_and_audio_modality(self):
        """Test forward pass with text, image, and audio input."""
        mimo_model = self._make_avlm()

        img_seq_len = (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim) + 1
        encoder_down_sampling = 2
        mel_bins = 80
        time_bins = 3000  # 30 seconds of audio at 10ms per frame
        audio_seq_len = math.ceil(time_bins / encoder_down_sampling)

        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        # Place image and audio special tokens
        start_pos = 5
        image_token_id = self.special_token_ids["images"]
        audio_token_id = self.special_token_ids["audio"]
        for i in range(self.batch_size):
            input_ids[i, start_pos : start_pos + img_seq_len] = image_token_id
            audio_start = start_pos + img_seq_len + 10
            input_ids[i, audio_start : audio_start + audio_seq_len] = audio_token_id

        modality_inputs = {
            "images": {
                "clip_encoder": {"x": torch.rand(2, 3, self.img_h, self.img_w, device=self.device)}
            },
            "audio": {
                "whisper_encoder": {
                    "input_features": torch.rand(2, mel_bins, time_bins, device=self.device)
                }
            },
        }

        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=modality_inputs
        )
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_state_dict(self):
        """Test state dict methods."""
        mimo_model = self._make_avlm()
        state_dict = mimo_model.state_dict()
        assert len(state_dict) > 0
        assert any(k.startswith("language_model.") for k in state_dict)
        assert any(k.startswith("modality_submodules.") for k in state_dict)

        checkpoint_dict = mimo_model.state_dict_for_save_checkpoint()
        assert len(checkpoint_dict) > 0

    def test_pipeline_model_parallel_accepted(self):
        """Test that MimoModel accepts pipeline_model_parallel_size > 1."""
        lm_config_pp2 = TransformerConfig(
            num_layers=2,
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            use_cpu_initialization=True,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
        )
        language_model_spec_pp2 = ModuleSpec(
            module=GPTModel,
            params={
                "config": lm_config_pp2,
                "transformer_layer_spec": get_gpt_layer_with_transformer_engine_spec(),
                "vocab_size": self.vocab_size,
                "max_sequence_length": self.seq_len,
                "pre_process": True,
                "post_process": True,
            },
        )
        mimo_config = MimoModelConfig(
            language_model_spec=language_model_spec_pp2,
            modality_submodules_spec={},
            special_token_ids=self.special_token_ids,
        )

        model = MimoModel(mimo_config)
        assert model is not None

    def test_partition_adapter_none_by_default(self):
        """Test that partition_adapter is None with default config (no CP/SP)."""
        mimo_model = self._make_vlm()
        assert mimo_model.partition_adapter is None

    def test_forward_with_packing_kwargs(self):
        """Test that packing_kwargs builds PackedSeqParams with qkv_format='thd' and int32 seqlens."""
        from megatron.core.packed_seq_params import PackedSeqParams

        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        cu_seqlens = torch.tensor(
            [0, self.seq_len, 2 * self.seq_len], dtype=torch.int64, device=self.device
        )
        packing_kwargs = {"cu_seqlens_q": cu_seqlens.clone(), "cu_seqlens_kv": cu_seqlens.clone()}

        text_emb = torch.zeros(self.batch_size * self.seq_len, self.hidden_size, device=self.device)
        combined_emb = torch.zeros(
            self.seq_len, self.batch_size, self.hidden_size, device=self.device
        )

        captured = {}

        def capture_lm_forward(*args, **kwargs):
            captured['packed_seq_params'] = kwargs.get('packed_seq_params')
            return torch.zeros(self.batch_size, self.seq_len, self.vocab_size, device=self.device)

        with (
            patch.object(mimo_model, 'get_text_embeddings', return_value=text_emb),
            patch.object(
                mimo_model, 'align_embeddings_by_token_positions', return_value=combined_emb
            ),
            patch.object(mimo_model.language_model, 'forward', side_effect=capture_lm_forward),
        ):
            mimo_model(
                input_ids=input_ids,
                position_ids=position_ids,
                modality_inputs=None,
                packing_kwargs=packing_kwargs,
            )

        packed_seq_params = captured['packed_seq_params']
        assert isinstance(packed_seq_params, PackedSeqParams)
        assert packed_seq_params.qkv_format == 'thd'
        assert packed_seq_params.cu_seqlens_q.dtype == torch.int32
        assert packed_seq_params.cu_seqlens_kv.dtype == torch.int32

    def test_forward_with_partition_adapter(self):
        """Test that partition_adapter.shard() is called and embeddings are transposed correctly."""
        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        sharded_seq_len = self.seq_len // 2
        sharded_emb = torch.zeros(
            self.batch_size, sharded_seq_len, self.hidden_size, device=self.device
        )
        mock_adapter = MagicMock()
        mock_adapter.shard.return_value = (sharded_emb, None, None, None, None)
        mimo_model.partition_adapter = mock_adapter

        text_emb = torch.zeros(self.batch_size * self.seq_len, self.hidden_size, device=self.device)
        combined_emb = torch.zeros(
            self.seq_len, self.batch_size, self.hidden_size, device=self.device
        )

        captured = {}

        def capture_lm_forward(*args, **kwargs):
            captured['decoder_input'] = kwargs.get('decoder_input')
            return torch.zeros(
                self.batch_size, sharded_seq_len, self.vocab_size, device=self.device
            )

        with (
            patch.object(mimo_model, 'get_text_embeddings', return_value=text_emb),
            patch.object(
                mimo_model, 'align_embeddings_by_token_positions', return_value=combined_emb
            ),
            patch.object(mimo_model.language_model, 'forward', side_effect=capture_lm_forward),
        ):
            mimo_model(input_ids=input_ids, position_ids=position_ids, modality_inputs=None)

        mock_adapter.shard.assert_called_once()
        shard_kwargs = mock_adapter.shard.call_args[1]
        assert shard_kwargs['embeddings'].shape == (self.batch_size, self.seq_len, self.hidden_size)
        assert captured['decoder_input'].shape == (
            sharded_seq_len,
            self.batch_size,
            self.hidden_size,
        )


class MockProcessGroup:
    """Mock process group for testing."""

    def __init__(self, rank, world_size):
        self._rank = rank
        self._size = world_size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class MockGrid:
    """Mock grid with HyperCommGrid-compatible interface."""

    def __init__(self, rank_offset=0, size=1, dim_names=None, pp_rank=0, pp_size=1):
        self.rank_offset = rank_offset
        self.size = size
        self.dim_names = dim_names or []
        self._pp_group = MockProcessGroup(pp_rank, pp_size)

    def get_pg(self, dims):
        if dims == "pp":
            return self._pp_group
        raise KeyError(f"Process group for {dims} not found")


class TestMimoModelNonColocated:
    """Tests for non-colocated multi-module pipeline parallelism."""

    def setup_method(self, method):
        try:
            Utils.initialize_model_parallel(1, 1)
        except Exception:
            pass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 64
        self.vocab_size = 48000
        self.seq_len = 256
        self.batch_size = 2
        self.img_h = 224
        self.img_w = 224
        self.patch_dim = 16

    def teardown_method(self, method):
        try:
            Utils.destroy_model_parallel()
        except Exception:
            pass

    def _make_config(self, encoder_in_grid=True, language_in_grid=True, pp_rank=0, pp_size=1):
        """Helper to create MimoModelConfig with mock grids."""
        language_model_spec = get_language_model_spec(
            self.hidden_size, self.vocab_size, self.seq_len
        )
        vision_submodule_spec = get_vision_submodules_spec(
            self.hidden_size, self.img_h, self.img_w, self.patch_dim
        )

        world_size = dist.get_world_size()
        encoder_offset = 0 if encoder_in_grid else world_size
        language_offset = 0 if language_in_grid else world_size

        return MimoModelConfig(
            language_model_spec=language_model_spec,
            modality_submodules_spec={"images": vision_submodule_spec},
            special_token_ids={"images": 50257},
            module_to_grid_map={
                "images": MockGrid(
                    rank_offset=encoder_offset,
                    size=world_size,
                    dim_names=["pp"] if pp_size > 1 else [],
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                ),
                MIMO_LANGUAGE_MODULE_KEY: MockGrid(
                    rank_offset=language_offset,
                    size=world_size,
                    dim_names=["pp"] if pp_size > 1 else [],
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                ),
            },
        )

    def test_grid_validation_rejects_mismatched_keys(self):
        """Test validation fails when grid_map keys don't match expected modules."""
        language_model_spec = get_language_model_spec(
            self.hidden_size, self.vocab_size, self.seq_len
        )
        vision_submodule_spec = get_vision_submodules_spec(
            self.hidden_size, self.img_h, self.img_w, self.patch_dim
        )

        mimo_config = MimoModelConfig(
            language_model_spec=language_model_spec,
            modality_submodules_spec={"images": vision_submodule_spec},
            special_token_ids={"images": 50257},
            module_to_grid_map={MIMO_LANGUAGE_MODULE_KEY: MockGrid()},
        )

        with pytest.raises(ValueError, match="module_to_grid_map keys must match"):
            MimoModel(mimo_config)

    def test_role_determination(self):
        """Test role correctly identifies modules and stage positions."""
        # No grid map = colocated role with all modules
        model_no_grid = get_vlm_mimo_model(
            self.hidden_size,
            self.vocab_size,
            self.seq_len,
            self.img_h,
            self.img_w,
            self.patch_dim,
            {"images": 50257},
        )
        assert model_no_grid.role.mode == ModuleLayout.COLOCATED
        assert model_no_grid.role.has_language_module is True
        assert model_no_grid.role.has_modality_modules is True

        # Encoder-only rank
        model_encoder = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=False))
        assert model_encoder.role.has_modality_modules is True
        assert model_encoder.role.has_language_module is False

        # Language-only rank
        model_language = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        assert model_language.role.has_modality_modules is False
        assert model_language.role.has_language_module is True

        # Stage info with PP on a non-colocated layout (encoder and language on
        # different rank ranges, which routes through RankRole.from_grid_map).
        world_size = dist.get_world_size()
        half = max(world_size // 2, 1)
        pp_rank, pp_size = 1, 3
        language_model_spec = get_language_model_spec(
            self.hidden_size, self.vocab_size, self.seq_len
        )
        vision_submodule_spec = get_vision_submodules_spec(
            self.hidden_size, self.img_h, self.img_w, self.patch_dim
        )
        pp_config = MimoModelConfig(
            language_model_spec=language_model_spec,
            modality_submodules_spec={"images": vision_submodule_spec},
            special_token_ids={"images": 50257},
            module_to_grid_map={
                "images": MockGrid(
                    rank_offset=0, size=half, dim_names=["pp"], pp_rank=pp_rank, pp_size=pp_size
                ),
                MIMO_LANGUAGE_MODULE_KEY: MockGrid(
                    rank_offset=half,
                    size=world_size - half,
                    dim_names=["pp"],
                    pp_rank=pp_rank,
                    pp_size=pp_size,
                ),
            },
        )
        model_pp = MimoModel(pp_config)
        assert model_pp.role.is_first_stage("images") is False
        assert model_pp.role.is_last_stage("images") is False

    def test_selective_init_encoder_only(self):
        """Test encoder-only rank initializes encoder but not language model."""
        model = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=False))
        assert "images" in model.modality_submodules
        assert model.language_model is None

    def test_selective_init_language_only(self):
        """Test language-only rank initializes language model but not encoder."""
        model = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        assert "images" not in model.modality_submodules
        assert model.language_model is not None

    def test_forward_encoder_only(self):
        """Test encoder-only forward returns dict of embeddings."""
        model = MimoModel(self._make_config(encoder_in_grid=True, language_in_grid=False))
        model = model.to(self.device)

        images = torch.rand(2, 3, self.img_h, self.img_w, device=self.device)
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
        )

        outputs, _ = model(
            input_ids=input_ids, modality_inputs={"images": {"clip_encoder": {"x": images}}}
        )
        assert isinstance(outputs, dict)
        assert "images" in outputs

    def test_forward_language_only(self):
        """Test language-only forward returns tensor."""
        model = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        model = model.to(self.device)

        img_seq_len = (self.img_h // self.patch_dim) * (self.img_w // self.patch_dim) + 1
        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
        )
        input_ids[:, 5 : 5 + img_seq_len] = 50257
        position_ids = (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )

        encoder_embeddings = torch.randn(
            self.batch_size * img_seq_len, self.hidden_size, device=self.device
        )
        model.set_input_tensor({"images": encoder_embeddings})

        outputs, _ = model(input_ids=input_ids, position_ids=position_ids, modality_inputs=None)
        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)
