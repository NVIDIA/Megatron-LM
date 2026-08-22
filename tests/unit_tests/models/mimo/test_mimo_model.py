# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

'''
WORLD_SIZE=1 LOCAL_RANK=0 python -m pytest tests/unit_tests/models/mimo/test_mimo_model.py
'''

import math
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import WhisperConfig, WhisperModel

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mimo.config.base_configs import MimoModelConfig
from megatron.core.models.mimo.config.role import (
    MIMO_LANGUAGE_MODULE_KEY,
    ModuleLayout,
    ModuleStageInfo,
    RankRole,
)
from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.submodules.audio import AudioModalitySubmodules
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
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

    def test_get_text_embeddings_handles_3d_position_ids(self):
        """3D mRoPE position_ids ``[rope_dim, B, S]`` must produce the same text
        embeddings as the 2D ``[B, S]`` baseline.

        Multimodal RoPE (e.g. Qwen3-VL) carries multiple positional channels.
        ``get_text_embeddings`` should slice the first channel for the absolute
        text-position lookup; otherwise the indexed positions correspond to
        the wrong axis and the language model receives garbage text embeddings.
        ``eval()`` disables embedding dropout so the two calls are
        bit-comparable.
        """
        mimo_model = self._make_avlm().eval()
        input_ids = self._make_input_ids()
        position_ids_2d = self._make_position_ids()
        # Build [rope_dim=3, B, S] by tiling the 2D ids along a new leading
        # axis. Only channel 0 is consumed by the text lookup, so the result
        # must match the 2D baseline exactly.
        position_ids_3d = position_ids_2d.unsqueeze(0).expand(3, -1, -1).contiguous()

        emb_2d = mimo_model.get_text_embeddings(input_ids, position_ids_2d, self.special_token_ids)
        emb_3d = mimo_model.get_text_embeddings(input_ids, position_ids_3d, self.special_token_ids)
        assert emb_3d.shape == emb_2d.shape
        torch.testing.assert_close(emb_3d, emb_2d)

    def test_forward_text_only(self):
        """Test forward pass with only text input."""
        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        outputs, _ = mimo_model(
            input_ids=input_ids, position_ids=position_ids, modality_inputs=None
        )
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)

    def test_forward_threads_position_ids_to_language_model(self):
        """Thread token, position, and MTP-mask inputs to the language model.

        Multimodal RoPE (e.g. Qwen3-VL) consumes ``position_ids`` even when
        the language model receives a pre-combined ``decoder_input``. The main
        embedding lookup ignores ``input_ids`` on this path, but MTP still uses
        them to build shifted-token embeddings and its conditioning mask.
        """
        mimo_model = self._make_vlm()
        mimo_model.language_model.mtp_process = True
        input_ids = self._make_input_ids()
        input_ids[0, 1] = self.special_token_ids['images']
        input_ids[1, 2] = self.special_token_ids['audio']
        position_ids = self._make_position_ids()
        loss_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)

        captured = {}

        def capture_lm_forward(*args, **kwargs):
            captured['input_ids'] = kwargs.get('input_ids')
            captured['position_ids'] = kwargs.get('position_ids')
            captured['decoder_input'] = kwargs.get('decoder_input')
            captured['loss_mask'] = kwargs.get('loss_mask')
            captured['mtp_input_mask'] = kwargs.get('mtp_input_mask')
            return torch.zeros(self.batch_size, self.seq_len, self.vocab_size, device=self.device)

        with patch.object(mimo_model.language_model, 'forward', side_effect=capture_lm_forward):
            mimo_model(
                input_ids=input_ids,
                position_ids=position_ids,
                loss_mask=loss_mask,
                modality_inputs=None,
            )

        assert (
            captured['decoder_input'] is not None
        ), "MimoModel.forward must pass a pre-combined decoder_input to the language model"
        assert (
            captured['input_ids'] is input_ids
        ), "MimoModel.forward must preserve input_ids for MTP"
        assert (
            captured['position_ids'] is not None
        ), "MimoModel.forward must pass position_ids to the language model (got None)"
        torch.testing.assert_close(captured['position_ids'], position_ids)
        assert captured['loss_mask'] is loss_mask
        expected_mtp_input_mask = torch.ones_like(input_ids, dtype=torch.bool)
        for special_token_id in self.special_token_ids.values():
            expected_mtp_input_mask &= input_ids != special_token_id
        torch.testing.assert_close(captured['mtp_input_mask'], expected_mtp_input_mask)

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
        """MTP token metadata must use the same CP-local sequence as hidden states.

        The caller no longer transposes around shard(): it passes the sequence-first
        ``(S, B, H)`` combined embeddings straight in, and shard()'s LM-layout output
        flows straight into the language model as ``decoder_input``. Token and position
        IDs used by MTP must be partitioned to that same local sequence.
        """
        mimo_model = self._make_vlm()
        mimo_model.language_model.mtp_process = True
        input_ids = self._make_input_ids()
        input_ids[0, 1] = self.special_token_ids['images']
        position_ids = self._make_position_ids()
        loss_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)

        sharded_seq_len = self.seq_len // 2
        # shard() returns LM-layout [S/cp, B, H] and a (CP-sharded) loss mask.
        sharded_emb = torch.zeros(
            sharded_seq_len, self.batch_size, self.hidden_size, device=self.device
        )
        sharded_loss_mask = torch.ones(self.batch_size, sharded_seq_len, device=self.device)
        sharded_input_ids = input_ids[:, :sharded_seq_len].clone()
        sharded_position_ids = position_ids[:, :sharded_seq_len].clone()
        mock_adapter = MagicMock()
        mock_adapter.shard.side_effect = [
            (sharded_emb, None, sharded_loss_mask, None),
            (None, sharded_position_ids, sharded_input_ids, None),
        ]
        mimo_model.partition_adapter = mock_adapter

        text_emb = torch.zeros(self.batch_size * self.seq_len, self.hidden_size, device=self.device)
        combined_emb = torch.zeros(
            self.seq_len, self.batch_size, self.hidden_size, device=self.device
        )

        captured = {}

        def capture_lm_forward(*args, **kwargs):
            captured['input_ids'] = kwargs.get('input_ids')
            captured['position_ids'] = kwargs.get('position_ids')
            captured['decoder_input'] = kwargs.get('decoder_input')
            captured['loss_mask'] = kwargs.get('loss_mask')
            captured['mtp_input_mask'] = kwargs.get('mtp_input_mask')
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
            _, out_loss_mask = mimo_model(
                input_ids=input_ids,
                position_ids=position_ids,
                loss_mask=loss_mask,
                modality_inputs=None,
            )

        assert mock_adapter.shard.call_count == 2
        shard_kwargs = mock_adapter.shard.call_args_list[0].kwargs
        # The helper passes sequence-first [S, B, H] embeddings straight to shard().
        assert shard_kwargs['embeddings'].shape == (self.seq_len, self.batch_size, self.hidden_size)
        assert shard_kwargs['loss_mask'] is loss_mask
        mtp_shard_kwargs = mock_adapter.shard.call_args_list[1].kwargs
        assert mtp_shard_kwargs['embeddings'] is None
        assert mtp_shard_kwargs['labels'] is position_ids
        assert mtp_shard_kwargs['loss_mask'] is input_ids
        # shard()'s LM-layout output flows straight into the LM (no extra transpose).
        assert captured['decoder_input'].shape == (
            sharded_seq_len,
            self.batch_size,
            self.hidden_size,
        )
        assert captured['loss_mask'] is sharded_loss_mask
        assert captured['input_ids'] is sharded_input_ids
        assert captured['position_ids'] is sharded_position_ids
        assert captured['mtp_input_mask'].dtype == torch.bool
        assert torch.equal(
            captured['mtp_input_mask'], sharded_input_ids != self.special_token_ids['images']
        )
        # forward() returns the (possibly sharded) loss mask from shard().
        assert out_loss_mask is sharded_loss_mask

    def test_get_text_embeddings_raises_when_sp_and_embedding_scatter_enabled(self):
        """SP must not double-scatter: if the LM embedding still scatters, raise.

        With sequence parallelism active, PartitionAdapter scatters the combined
        embeddings. If the language embedding also scattered, the flat text tokens
        would be split across TP ranks before alignment. ``get_text_embeddings``
        must reject that configuration.
        """
        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        sp_adapter = MagicMock()
        sp_adapter.cfg.seq_parallel = True
        mimo_model.partition_adapter = sp_adapter
        # Embedding layer reports it would scatter for sequence parallelism.
        mimo_model.language_model.embedding.scatter_to_sequence_parallel = True

        with pytest.raises(RuntimeError, match="embedding scatter to be disabled"):
            mimo_model.get_text_embeddings(input_ids, position_ids, self.special_token_ids)

    def test_get_text_embeddings_ok_when_sp_and_embedding_scatter_disabled(self):
        """SP with embedding scatter disabled is the supported configuration."""
        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()

        sp_adapter = MagicMock()
        sp_adapter.cfg.seq_parallel = True
        mimo_model.partition_adapter = sp_adapter
        mimo_model.language_model.embedding.scatter_to_sequence_parallel = False

        text_embeddings = mimo_model.get_text_embeddings(
            input_ids, position_ids, self.special_token_ids
        )
        assert text_embeddings.shape == (self.batch_size * self.seq_len, self.hidden_size)

    def test_forward_language_module_rejects_attention_mask_under_cp(self):
        """A dense attention_mask is rejected under context parallelism.

        Under CP the hidden states are CP-local, so a dense [B, S] mask cannot line up
        with the sharded sequence; _forward_language_module must fail fast.
        """
        mimo_model = self._make_vlm()
        input_ids = self._make_input_ids()
        position_ids = self._make_position_ids()
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)

        cp_adapter = MagicMock()
        cp_adapter.cfg.use_cp = True
        mimo_model.partition_adapter = cp_adapter

        with pytest.raises(RuntimeError, match="context parallelism requires attention_mask=None"):
            mimo_model._forward_language_module(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                loss_mask=None,
                labels=None,
                input_tensors=None,
            )

    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (1, 2), (2, 2)])
    @pytest.mark.parametrize("position_dims", [2, 3])
    def test_mtp_forward_uses_parallel_local_token_metadata(self, tp, cp, position_dims):
        """Run real MIMO+MTP for the supported TP/CP combinations.

        MIMO owns CP partitioning for its pre-combined decoder embeddings, so
        the token and position IDs consumed by MTP must be partitioned to the
        same CP-local sequence. TP additionally exercises MTP's SP scatter.
        """
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        seq_len = 16
        vocab_size = 128
        special_token_ids = {"images": 126, "audio": 127}
        config = TransformerConfig(
            mtp_num_layers=1,
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=tp > 1,
            context_parallel_size=cp,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        language_model_spec = ModuleSpec(
            module=GPTModel,
            params={
                "config": config,
                "transformer_layer_spec": layer_spec,
                "mtp_block_spec": get_gpt_mtp_block_spec(
                    config=config, spec=layer_spec, use_transformer_engine=True
                ),
                "vocab_size": vocab_size,
                "max_sequence_length": seq_len,
                "pre_process": True,
                "post_process": True,
                "position_embedding_type": "rope",
                "scatter_embedding_sequence_parallel": False,
                "share_embeddings_and_output_weights": True,
                "pg_collection": pg_collection,
            },
        )
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=language_model_spec,
                modality_submodules_spec={},
                special_token_ids=special_token_ids,
            ),
            cp_group=pg_collection.cp,
            tp_group=pg_collection.tp,
        ).cuda()

        if cp > 1:
            # The test container has no TE attention backend for CP. Keep the real
            # MIMO partitioning and MTP embedding/projection/loss path, but replace
            # the main decoder and the inner attention layer with identities so the
            # test reaches the CP-local MTP concatenation this regression covers.
            class _IdentityDecoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_tensor = None

                def set_input_tensor(self, input_tensor):
                    self.input_tensor = input_tensor

                def forward(self, hidden_states, **kwargs):
                    return hidden_states if hidden_states is not None else self.input_tensor

            class _IdentityMTPLayer(torch.nn.Module):
                def forward(self, hidden_states, **kwargs):
                    return hidden_states, None

            model.language_model.decoder = _IdentityDecoder()
            for mtp_layer in model.language_model.mtp.layers:
                mtp_layer.mtp_model_layer = _IdentityMTPLayer()

        input_ids = torch.randint(1, min(special_token_ids.values()), (1, seq_len), device="cuda")
        input_ids[0, 3] = special_token_ids["images"]
        input_ids[0, 12] = special_token_ids["audio"]
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        if position_dims == 3:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).contiguous()
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        for special_token_id in special_token_ids.values():
            loss_mask[input_ids == special_token_id] = 0

        output, local_loss_mask = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            modality_inputs=None,
        )

        assert output.shape == local_loss_mask.shape == (1, seq_len // cp)
        output.mean().backward()

    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (1, 2), (4, 1), (1, 4), (2, 2)])
    def test_non_first_pipeline_stage_runs_real_mtp(self, tp, cp):
        """Run real MIMO+MTP on the last stage with combined parallelism."""
        required_world_size = 2 * tp * cp
        if torch.distributed.get_world_size() < required_world_size:
            pytest.skip(f"requires {required_world_size} distributed ranks")

        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=2, context_parallel_size=cp
        )
        model_parallel_cuda_manual_seed(123)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        is_first_stage = parallel_state.is_pipeline_first_stage()
        is_last_stage = parallel_state.is_pipeline_last_stage()

        seq_len = 8
        vocab_size = 128
        invalid_token_id = 127
        config = TransformerConfig(
            mtp_num_layers=1,
            num_layers=2,
            hidden_size=32,
            num_attention_heads=4,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=tp > 1,
            context_parallel_size=cp,
            pipeline_model_parallel_size=2,
            pipeline_dtype=torch.float32,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        language_model_spec = ModuleSpec(
            module=GPTModel,
            params={
                "config": config,
                "transformer_layer_spec": layer_spec,
                "mtp_block_spec": get_gpt_mtp_block_spec(
                    config=config, spec=layer_spec, use_transformer_engine=True
                ),
                "vocab_size": vocab_size,
                "max_sequence_length": seq_len,
                "pre_process": is_first_stage,
                "post_process": is_last_stage,
                "position_embedding_type": "rope",
                "scatter_embedding_sequence_parallel": False,
                "share_embeddings_and_output_weights": True,
                "pg_collection": pg_collection,
            },
        )
        model = MimoModel(
            MimoModelConfig(
                language_model_spec=language_model_spec,
                modality_submodules_spec={},
                special_token_ids={"images": invalid_token_id},
            ),
            cp_group=pg_collection.cp,
            tp_group=pg_collection.tp,
        ).cuda()
        model.role = RankRole(
            modules={
                MIMO_LANGUAGE_MODULE_KEY: ModuleStageInfo(
                    is_first_stage=is_first_stage, is_last_stage=is_last_stage
                )
            },
            mode=ModuleLayout.NON_COLOCATED,
        )

        if not is_last_stage:
            return

        if cp > 1:
            # The validation image has no TE attention backend for CP. Preserve
            # real PP/CP metadata partitioning and the MTP embedding/projection/loss
            # path while replacing only the attention-bearing blocks.
            class _IdentityDecoder(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.input_tensor = None

                def set_input_tensor(self, input_tensor):
                    self.input_tensor = input_tensor

                def forward(self, hidden_states, **kwargs):
                    return hidden_states if hidden_states is not None else self.input_tensor

            class _IdentityMTPLayer(torch.nn.Module):
                def forward(self, hidden_states, **kwargs):
                    return hidden_states, None

            model.language_model.decoder = _IdentityDecoder()
            for mtp_layer in model.language_model.mtp.layers:
                mtp_layer.mtp_model_layer = _IdentityMTPLayer()

        input_ids = torch.randint(1, vocab_size - 1, (1, seq_len), device="cuda")
        input_ids[0, 2] = invalid_token_id
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
        loss_mask[input_ids == invalid_token_id] = 0
        hidden_states = torch.randn(seq_len // (tp * cp), 1, config.hidden_size, device="cuda")

        output, _ = model._forward_language_module(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            loss_mask=loss_mask,
            labels=labels,
            input_tensors={MIMO_LANGUAGE_MODULE_KEY: hidden_states},
        )

        assert output.shape == (1, seq_len // cp)
        output.mean().backward()


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

        with pytest.raises(ValueError, match="module_to_grid_map keys must match"):
            MimoModelConfig(
                language_model_spec=language_model_spec,
                modality_submodules_spec={"images": vision_submodule_spec},
                special_token_ids={"images": 50257},
                module_to_grid_map={MIMO_LANGUAGE_MODULE_KEY: MockGrid()},
            )

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

        # Stage info with PP. language_in_grid=False so encoder and language
        # grids have distinct rank_offsets and role.build dispatches to
        # _from_grid_map (rather than collapsing to the COLOCATED path).
        model_pp = MimoModel(
            self._make_config(encoder_in_grid=True, language_in_grid=False, pp_rank=1, pp_size=3)
        )
        assert model_pp.role.is_first_stage("images") is False
        assert model_pp.role.is_last_stage("images") is False
        assert model_pp.colocated_comms == {}

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
        loss_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)
        loss_mask[input_ids == 50257] = 0
        position_ids = (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )

        encoder_embeddings = torch.randn(
            self.batch_size * img_seq_len, self.hidden_size, device=self.device
        )
        model.set_input_tensor({"images": encoder_embeddings})

        captured = {}

        def capture_language_inputs(module, args, kwargs):
            captured['loss_mask'] = kwargs.get('loss_mask')

        hook = model.language_model.register_forward_pre_hook(
            capture_language_inputs, with_kwargs=True
        )
        try:
            outputs, out_loss_mask = model(
                input_ids=input_ids,
                position_ids=position_ids,
                loss_mask=loss_mask,
                modality_inputs=None,
            )
        finally:
            hook.remove()

        assert isinstance(outputs, torch.Tensor)
        assert outputs.shape == (self.batch_size, self.seq_len, self.vocab_size)
        assert captured['loss_mask'] is loss_mask
        assert out_loss_mask is loss_mask

    def test_forward_language_module_non_first_stage_threads_mtp_mask(self):
        """A non-first PP stage that owns MTP must preserve token IDs and its mask.

        Hidden states arrive through ``set_input_tensor`` while the conditioning
        token IDs remain available for MTP's shifted embedding lookup.
        """
        model = MimoModel(self._make_config(encoder_in_grid=False, language_in_grid=True))
        model = model.to(self.device)
        model.language_model.mtp_process = True

        input_ids = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
        )
        input_ids[0, 1] = model.special_token_ids['images']
        position_ids = (
            torch.arange(self.seq_len, device=self.device).unsqueeze(0).expand(self.batch_size, -1)
        )
        hidden_states = torch.randn(
            self.seq_len, self.batch_size, self.hidden_size, device=self.device
        )
        loss_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)
        loss_mask[:, : self.seq_len // 2] = 0

        captured = {}

        def capture_lm_forward(*args, **kwargs):
            captured.update(kwargs)
            return torch.zeros(self.batch_size, self.seq_len, self.vocab_size, device=self.device)

        with (
            patch.object(model.role, 'is_first_stage', return_value=False),
            patch.object(model.role, 'is_last_stage', return_value=True),
            patch.object(model.language_model, 'forward', side_effect=capture_lm_forward),
        ):
            lm_output, _ = model._forward_language_module(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                loss_mask=loss_mask,
                labels=None,
                input_tensors={MIMO_LANGUAGE_MODULE_KEY: hidden_states},
            )

        assert captured['input_ids'] is input_ids
        assert captured['decoder_input'] is None
        assert captured['loss_mask'] is loss_mask
        assert captured['mtp_input_mask'][0, 1].item() is False
        assert captured['mtp_input_mask'][0, 0].item() is True
        torch.testing.assert_close(captured['position_ids'], position_ids)


class TestMimoModelFanoutHelpers:
    """CPU-only coverage for bridge-fanout helper methods on ``MimoModel``."""

    @staticmethod
    def _stub_model(special_token_ids):
        model = MimoModel.__new__(MimoModel)
        model.special_token_ids = special_token_ids
        # COLOCATED skips the fan-out DP assertion path; this fixture targets
        # the metadata-tagging logic only.
        model.role = SimpleNamespace(mode=ModuleLayout.COLOCATED)
        return model

    def test_attach_modality_split_sizes_tags_output_with_per_sample_counts(self):
        model = self._stub_model({"images": 50257})
        input_ids = torch.tensor([[50257, 50257, 1, 2], [50257, 1, 2, 3]])
        output = torch.zeros(3, 8)

        model._attach_modality_split_sizes(output, input_ids, "images")

        assert output._mimo_bridge_split_sizes == [2, 1]

    def test_attach_modality_split_sizes_skips_when_total_mismatches(self):
        model = self._stub_model({"images": 50257})
        input_ids = torch.tensor([[50257, 1], [1, 1]])
        output = torch.zeros(5, 8)

        model._attach_modality_split_sizes(output, input_ids, "images")

        assert not hasattr(output, "_mimo_bridge_split_sizes")

    def test_attach_modality_split_sizes_skips_when_token_counts_uniform(self):
        model = self._stub_model({"images": 50257})
        # Two samples, two image tokens each — uniform per-sample counts.
        input_ids = torch.tensor([[50257, 50257, 1, 2], [50257, 50257, 3, 4]])
        output = torch.zeros(4, 8)

        model._attach_modality_split_sizes(output, input_ids, "images")

        assert not hasattr(output, "_mimo_bridge_split_sizes")

    def test_attach_modality_split_sizes_skips_for_single_sample_batch(self):
        model = self._stub_model({"images": 50257})
        input_ids = torch.tensor([[50257, 50257, 1]])
        output = torch.zeros(2, 8)

        model._attach_modality_split_sizes(output, input_ids, "images")

        assert not hasattr(output, "_mimo_bridge_split_sizes")

    def test_has_encoder_tokens_detects_presence_and_absence(self):
        model = self._stub_model({"images": 50257, "audio": 50258})

        assert model._has_encoder_tokens(torch.tensor([[50257, 1]]), "images") is True
        assert model._has_encoder_tokens(torch.tensor([[1, 2]]), "images") is False
        assert model._has_encoder_tokens(None, "images") is False
        assert model._has_encoder_tokens(torch.tensor([[1]]), "unknown") is False

    @staticmethod
    def _stub_mimo_config(
        *,
        hidden_size=8,
        language_dtype=torch.float32,
        include_language_dtype=True,
        modality_params=None,
    ):
        language_config = SimpleNamespace(hidden_size=hidden_size)
        if include_language_dtype:
            language_config.params_dtype = language_dtype
        return SimpleNamespace(
            language_model_spec=SimpleNamespace(params={'config': language_config}),
            modality_submodules_spec={
                "images": SimpleNamespace(params={} if modality_params is None else modality_params)
            },
        )

    def test_set_input_tensor_unwraps_outer_list_and_forwards_to_language_model(self):
        lm = MagicMock()
        model = MimoModel.__new__(MimoModel)
        model.language_model = lm
        tensor = torch.zeros(2, 4)

        model.set_input_tensor([tensor])

        assert torch.equal(model.input_tensors, tensor)
        lm.set_input_tensor.assert_called_once_with(tensor)

    def test_set_input_tensor_dict_unwraps_single_element_value_lists(self):
        model = MimoModel.__new__(MimoModel)
        model.language_model = None
        t_lang = torch.zeros(2, 4)
        t_vision = torch.zeros(3, 4)

        model.set_input_tensor({"language": [t_lang], "vision": t_vision})

        assert torch.equal(model.input_tensors["language"], t_lang)
        assert torch.equal(model.input_tensors["vision"], t_vision)

    def test_empty_encoder_output_uses_language_dtype_without_modality_config(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
        model = MimoModel.__new__(MimoModel)
        model.mimo_config = self._stub_mimo_config(
            hidden_size=8, language_dtype=torch.bfloat16, modality_params={}
        )

        output = model._empty_encoder_output("images")

        assert output.shape == (0, 8)
        assert output.dtype == torch.bfloat16
        assert output.device.type == "cpu"
        assert output.requires_grad

    def test_empty_encoder_output_defaults_to_float32_without_language_dtype(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
        model = MimoModel.__new__(MimoModel)
        model.mimo_config = self._stub_mimo_config(
            hidden_size=8, include_language_dtype=False, modality_params={}
        )

        output = model._empty_encoder_output("images")

        assert output.shape == (0, 8)
        assert output.dtype == torch.float32
        assert output.device.type == "cpu"
        assert output.requires_grad

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA for current_device")
    def test_empty_encoder_output_uses_current_cuda_device(self):
        model = MimoModel.__new__(MimoModel)
        model.mimo_config = self._stub_mimo_config(hidden_size=8, language_dtype=torch.bfloat16)

        output = model._empty_encoder_output("images")

        assert output.shape == (0, 8)
        assert output.dtype == torch.bfloat16
        assert output.device.type == "cuda"
        assert output.requires_grad

    def test_empty_encoder_output_raises_when_hidden_size_missing(self):
        model = MimoModel.__new__(MimoModel)
        model.mimo_config = SimpleNamespace(
            language_model_spec=SimpleNamespace(params={'config': SimpleNamespace()})
        )

        with pytest.raises(ValueError, match="hidden_size"):
            model._empty_encoder_output("images")
