# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for tokenizer file saving in checkpointing."""

import os
import shutil
import tempfile
from unittest.mock import Mock, mock_open, patch

import pytest

from megatron.training.checkpointing import save_tokenizer_assets


class TestSaveTokenizerAssets:
    """Test tokenizer file saving functionality."""

    @pytest.fixture
    def checkpoint_path_fixture(self):
        """Create a temporary directory for checkpoint."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_save_tokenizer_assets_none_tokenizer(
        self, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test that function returns early when tokenizer is None."""
        mock_dist_init.return_value = False

        config = Mock()
        config.tokenizer_type = "SentencePieceTokenizer"

        # Should not raise error
        save_tokenizer_assets(None, config, checkpoint_path_fixture)

        # Should not create tokenizer directory
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        assert not os.path.exists(tokenizer_dir)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    def test_save_tokenizer_assets_non_rank0(
        self, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test that non-rank-0 processes don't save files."""
        mock_dist_init.return_value = True
        mock_get_rank.return_value = 1  # Not rank 0

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "SentencePieceTokenizer"

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Should not create tokenizer directory
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        assert not os.path.exists(tokenizer_dir)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_sentencepiece(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test saving SentencePiece tokenizer files."""
        mock_dist_init.return_value = False

        # Create a fake tokenizer model file
        fake_model_path = os.path.join(checkpoint_path_fixture, "tokenizer.model")
        with open(fake_model_path, "wb") as f:
            f.write(b"fake sentencepiece model")

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "SentencePieceTokenizer"
        config.tokenizer_model = fake_model_path

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Check that tokenizer directory was created
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        assert os.path.exists(tokenizer_dir)

        # Check that model file was copied
        copied_file = os.path.join(tokenizer_dir, "tokenizer.model")
        assert os.path.exists(copied_file)

        # Verify content
        with open(copied_file, "rb") as f:
            content = f.read()
        assert content == b"fake sentencepiece model"

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_gpt2bpe(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test saving GPT2BPE tokenizer files (vocab + merges)."""
        mock_dist_init.return_value = False

        # Create fake vocab and merge files
        vocab_path = os.path.join(checkpoint_path_fixture, "vocab.json")
        merge_path = os.path.join(checkpoint_path_fixture, "merges.txt")

        with open(vocab_path, "w") as f:
            f.write('{"test": 0}')
        with open(merge_path, "w") as f:
            f.write("t e\ne s\n")

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "GPT2BPETokenizer"
        config.vocab_file = vocab_path
        config.merge_file = merge_path

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Check files were copied
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        assert os.path.exists(os.path.join(tokenizer_dir, "vocab.json"))
        assert os.path.exists(os.path.join(tokenizer_dir, "merges.txt"))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_bert(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test saving BERT tokenizer files."""
        mock_dist_init.return_value = False

        # Create fake vocab file
        vocab_path = os.path.join(checkpoint_path_fixture, "vocab.txt")
        with open(vocab_path, "w") as f:
            f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n")

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "BertWordPieceLowerCase"
        config.vocab_file = vocab_path

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Check file was copied
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        copied_vocab = os.path.join(tokenizer_dir, "vocab.txt")
        assert os.path.exists(copied_vocab)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_tiktoken(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test saving TikToken tokenizer files."""
        mock_dist_init.return_value = False

        # Create fake tiktoken vocab file
        vocab_path = os.path.join(checkpoint_path_fixture, "tokenizer.json")
        with open(vocab_path, "w") as f:
            f.write('{"vocab": {}}')

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "TikTokenizer"
        config.tokenizer_model = vocab_path

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Check file was copied
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        assert os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json"))

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_null_tokenizer(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test that NullTokenizer doesn't save files."""
        mock_dist_init.return_value = False

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "NullTokenizer"

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Tokenizer directory should be created but empty
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        assert os.path.exists(tokenizer_dir)
        assert len(os.listdir(tokenizer_dir)) == 0

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_relative_path(self, mock_logger, mock_get_rank, mock_dist_init):
        """Test that relative paths are resolved correctly."""
        mock_dist_init.return_value = False

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create file using relative path
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                relative_path = "tokenizer.model"
                with open(relative_path, "wb") as f:
                    f.write(b"test data")

                mock_tokenizer = Mock()
                config = Mock()
                config.tokenizer_type = "SentencePieceTokenizer"
                config.tokenizer_model = relative_path  # Relative path

                checkpoint_path = os.path.join(temp_dir, "checkpoint")
                os.makedirs(checkpoint_path, exist_ok=True)

                save_tokenizer_assets(mock_tokenizer, config, checkpoint_path)

                # Check that file was copied
                tokenizer_dir = os.path.join(checkpoint_path, "tokenizer")
                assert os.path.exists(os.path.join(tokenizer_dir, "tokenizer.model"))
            finally:
                os.chdir(original_cwd)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_huggingface_with_instance(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test saving HuggingFace tokenizer files using provided instance."""
        mock_dist_init.return_value = False

        # Mock tokenizer with save_pretrained method
        # Use spec_set to make hasattr work correctly
        mock_inner_tokenizer = Mock(spec=["save_pretrained"])
        mock_tokenizer = Mock(spec=["_tokenizer"])
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        config = Mock()
        config.tokenizer_type = "HuggingFaceTokenizer"
        config.tokenizer_model = "bert-base-uncased"  # HF model ID

        # Pass tokenizer instance - uses it directly
        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Check that save_pretrained was called
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        mock_tokenizer._tokenizer.save_pretrained.assert_called_once_with(tokenizer_dir)

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_all_sentencepiece_variants(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test all SentencePiece-based tokenizer types."""
        mock_dist_init.return_value = False

        # Create a fake tokenizer model file
        fake_model_path = os.path.join(checkpoint_path_fixture, "tokenizer.model")
        with open(fake_model_path, "wb") as f:
            f.write(b"fake model data")

        tokenizer_types = ["SentencePieceTokenizer", "GPTSentencePieceTokenizer"]

        for tokenizer_type in tokenizer_types:
            # Clear previous tokenizer directory
            tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
            if os.path.exists(tokenizer_dir):
                shutil.rmtree(tokenizer_dir)

            mock_tokenizer = Mock()
            config = Mock()
            config.tokenizer_type = tokenizer_type
            config.tokenizer_model = fake_model_path

            save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

            # Verify file was copied
            copied_file = os.path.join(tokenizer_dir, "tokenizer.model")
            assert os.path.exists(copied_file), f"Failed for {tokenizer_type}"

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_missing_file(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test handling of missing tokenizer files."""
        mock_dist_init.return_value = False

        mock_tokenizer = Mock()
        config = Mock()
        config.tokenizer_type = "SentencePieceTokenizer"
        config.tokenizer_model = "/nonexistent/path/tokenizer.model"

        # Should not raise error
        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Should log debug message about missing file
        assert mock_logger.debug.called

    @patch("megatron.training.checkpointing.MultiStorageClientFeature")
    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_with_msc(
        self, mock_logger, mock_get_rank, mock_dist_init, mock_msc_feature
    ):
        """Test saving tokenizer files with MultiStorageClient enabled."""
        mock_dist_init.return_value = False
        mock_msc_feature.is_enabled.return_value = True

        # Mock MSC
        mock_msc = Mock()
        mock_path = Mock()
        mock_tokenizer_dir = Mock()
        mock_tokenizer_dir.mkdir = Mock()
        mock_path.__truediv__ = Mock(return_value=mock_tokenizer_dir)
        mock_msc.Path = Mock(return_value=mock_path)
        mock_msc.open = mock_open()
        mock_msc_feature.import_package.return_value = mock_msc

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake tokenizer file
            fake_model_path = os.path.join(temp_dir, "tokenizer.model")
            with open(fake_model_path, "wb") as f:
                f.write(b"test data")

            mock_tokenizer = Mock()
            config = Mock()
            config.tokenizer_type = "SentencePieceTokenizer"
            config.tokenizer_model = fake_model_path

            checkpoint_path = os.path.join(temp_dir, "checkpoint")

            save_tokenizer_assets(mock_tokenizer, config, checkpoint_path)

            # Verify MSC methods were called
            mock_msc_feature.is_enabled.assert_called()
            mock_msc_feature.import_package.assert_called()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.get_rank")
    @patch("megatron.training.checkpointing.logger")
    def test_save_tokenizer_assets_huggingface_save_pretrained(
        self, mock_logger, mock_get_rank, mock_dist_init, checkpoint_path_fixture
    ):
        """Test that HuggingFace tokenizer's save_pretrained is called."""
        mock_dist_init.return_value = False

        # Mock tokenizer with save_pretrained on the wrapped _tokenizer
        # Use spec to make hasattr work correctly
        mock_inner_tokenizer = Mock(spec=["save_pretrained"])
        mock_tokenizer = Mock(spec=["_tokenizer"])
        mock_tokenizer._tokenizer = mock_inner_tokenizer

        config = Mock()
        config.tokenizer_type = "HuggingFaceTokenizer"
        config.tokenizer_model = "bert-base-uncased"

        save_tokenizer_assets(mock_tokenizer, config, checkpoint_path_fixture)

        # Verify save_pretrained was called with tokenizer directory
        tokenizer_dir = os.path.join(checkpoint_path_fixture, "tokenizer")
        mock_tokenizer._tokenizer.save_pretrained.assert_called_once_with(tokenizer_dir)
