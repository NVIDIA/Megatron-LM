# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from megatron.core.enums import ModelType

from megatron.training.models.dist_utils import (
    _ddp_wrap,
    _print_num_params,
    _wrap_with_mp_wrapper,
    build_virtual_pipeline_stages,
    to_empty_if_meta_device,
    unimodal_build_distributed_models,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pg():
    """Mock ProcessGroupCollection with dp, cp, tp, pp sub-groups."""
    pg = Mock()
    pg.dp.rank.return_value = 0
    pg.cp.rank.return_value = 0
    pg.tp.rank.return_value = 0
    pg.pp.rank.return_value = 0
    pg.pp.size.return_value = 1
    return pg


def _make_transformer_config(**kwargs):
    cfg = Mock()
    cfg.virtual_pipeline_model_parallel_size = None
    cfg.init_model_with_meta_device = False
    cfg.use_cpu_initialization = False
    cfg.fp16 = False
    cfg.bf16 = False
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_model_module():
    m = Mock()
    m.parameters.return_value = []
    m.modules.return_value = []
    return m


# =============================================================================
# Section 1 — TestToEmptyIfMetaDevice
# =============================================================================


class TestToEmptyIfMetaDevice:
    """to_empty_if_meta_device() materialises meta-device parameters while leaving non-meta parameters unchanged."""

    def test_meta_parameter_becomes_empty_on_target_device(self):
        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.empty(4).to("meta")))
        result = to_empty_if_meta_device(module, device=torch.device("cpu"))
        assert result.weight.device == torch.device("cpu")

    def test_non_meta_parameter_moved_to_device(self):
        module = nn.Module()
        module.register_parameter("weight", nn.Parameter(torch.zeros(4)))
        result = to_empty_if_meta_device(module, device=torch.device("cpu"))
        assert result.weight.device == torch.device("cpu")

    def test_recurse_true_applies_to_submodule_parameters(self):
        parent = nn.Module()
        child = nn.Module()
        child.register_parameter("weight", nn.Parameter(torch.empty(4).to("meta")))
        parent.add_module("child", child)
        to_empty_if_meta_device(parent, device=torch.device("cpu"), recurse=True)
        assert parent.child.weight.device == torch.device("cpu")

    def test_recurse_false_skips_submodule_parameters(self):
        parent = nn.Module()
        child = nn.Module()
        child.register_parameter("weight", nn.Parameter(torch.empty(4).to("meta")))
        parent.add_module("child", child)
        to_empty_if_meta_device(parent, device=torch.device("cpu"), recurse=False)
        assert parent.child.weight.device == torch.device("meta")


# =============================================================================
# Section 2 — TestBuildVirtualPipelineStages
# =============================================================================


class TestBuildVirtualPipelineStages:
    """build_virtual_pipeline_stages() builds one stage without VP or multiple stages with VP, setting model_type on each."""

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage", return_value=True)
    def test_single_stage_when_pp_size_one(self, *_):
        pg = _make_pg()
        pg.pp.size.return_value = 1
        build_fn = Mock(return_value=Mock())
        result = build_virtual_pipeline_stages(build_fn, pg, vp_size=3)
        assert len(result) == 1
        build_fn.assert_called_once()

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage", return_value=True)
    def test_single_stage_when_vp_size_none(self, *_):
        pg = _make_pg()
        pg.pp.size.return_value = 2
        build_fn = Mock(return_value=Mock())
        result = build_virtual_pipeline_stages(build_fn, pg, vp_size=None)
        assert len(result) == 1
        build_fn.assert_called_once()

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage", return_value=True)
    def test_vp_builds_vp_size_stages(self, *_):
        pg = _make_pg()
        pg.pp.size.return_value = 2
        build_fn = Mock(return_value=Mock())
        result = build_virtual_pipeline_stages(build_fn, pg, vp_size=3)
        assert len(result) == 3
        assert build_fn.call_count == 3

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage", return_value=True)
    def test_vp_passes_correct_vp_stage_index(self, *_):
        pg = _make_pg()
        pg.pp.size.return_value = 2
        build_fn = Mock(return_value=Mock())
        build_virtual_pipeline_stages(build_fn, pg, vp_size=3)
        vp_stage_args = [c.kwargs["vp_stage"] for c in build_fn.call_args_list]
        assert vp_stage_args == [0, 1, 2]

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage", return_value=True)
    def test_model_type_set_on_every_stage(self, *_):
        pg = _make_pg()
        pg.pp.size.return_value = 2
        build_fn = Mock(return_value=Mock())
        result = build_virtual_pipeline_stages(build_fn, pg, vp_size=3)
        for model in result:
            assert model.model_type == ModelType.encoder_or_decoder

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage")
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage")
    def test_vp_pre_process_uses_vp_and_pp(
        self, mock_vp_first, mock_vp_last, mock_pp_first, mock_pp_last
    ):
        # First VP stage (i=0): vp_first=True → pre_process = True (and pp_first=True)
        # Other VP stages: vp_first=False → pre_process = False
        pg = _make_pg()
        pg.pp.size.return_value = 2

        # is_vp_first_stage returns True only when vp_stage=0
        mock_vp_first.side_effect = lambda vp_stage, vp_size: vp_stage == 0
        mock_vp_last.side_effect = lambda vp_stage, vp_size: vp_stage == (vp_size - 1)

        build_fn = Mock(return_value=Mock())
        build_virtual_pipeline_stages(build_fn, pg, vp_size=3)

        # Stage 0 should have pre_process=True, stages 1 and 2 should have pre_process=False
        assert build_fn.call_args_list[0].kwargs["pre_process"] is True
        assert build_fn.call_args_list[1].kwargs["pre_process"] is False
        assert build_fn.call_args_list[2].kwargs["pre_process"] is False

        assert build_fn.call_args_list[0].kwargs["post_process"] is False
        assert build_fn.call_args_list[1].kwargs["post_process"] is False
        assert build_fn.call_args_list[2].kwargs["post_process"] is True

    @patch("megatron.core.pipeline_parallel.utils.is_pp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_pp_first_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_last_stage", return_value=True)
    @patch("megatron.core.pipeline_parallel.utils.is_vp_first_stage", return_value=True)
    def test_returns_list_of_models(self, *_):
        pg = _make_pg()
        pg.pp.size.return_value = 1
        build_fn = Mock(return_value=Mock())
        result = build_virtual_pipeline_stages(build_fn, pg, vp_size=None)
        assert isinstance(result, list)


# =============================================================================
# Section 3 — TestPrintNumParams
# =============================================================================


class TestPrintNumParams:
    """_print_num_params() prints parameter counts only on data-parallel and context-parallel rank 0."""

    def test_prints_on_dp0_cp0(self, capsys):
        pg = _make_pg()  # dp.rank()=0, cp.rank()=0 by default
        model = [_make_model_module()]
        _print_num_params(model, pg_collection=pg)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_silent_on_nonzero_dp_rank(self, capsys):
        pg = _make_pg()
        pg.dp.rank.return_value = 1
        model = [_make_model_module()]
        _print_num_params(model, pg_collection=pg)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_silent_on_nonzero_cp_rank(self, capsys):
        pg = _make_pg()
        pg.cp.rank.return_value = 1
        model = [_make_model_module()]
        _print_num_params(model, pg_collection=pg)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_param_count_is_correct(self, capsys):
        pg = _make_pg()
        m1 = Mock()
        m1.parameters.return_value = [torch.zeros(3)]
        m2 = Mock()
        m2.parameters.return_value = [torch.zeros(5)]
        _print_num_params([m1, m2], pg_collection=pg)
        captured = capsys.readouterr()
        assert "8" in captured.out


# =============================================================================
# Section 4 — TestWrapWithMpWrapper
# =============================================================================


class TestWrapWithMpWrapper:
    """_wrap_with_mp_wrapper() applies the mixed-precision wrapper to each stage when fp16 or bf16 is set."""

    def test_no_wrap_when_fp16_false_bf16_false(self):
        cfg = _make_transformer_config(fp16=False, bf16=False)
        stages = [_make_model_module(), _make_model_module()]
        wrapper = Mock()
        result = _wrap_with_mp_wrapper(stages, cfg, wrapper)
        assert result is stages
        wrapper.assert_not_called()

    def test_wraps_each_stage_when_fp16_true(self):
        cfg = _make_transformer_config(fp16=True)
        stage1 = _make_model_module()
        stage2 = _make_model_module()
        wrapped1, wrapped2 = Mock(), Mock()
        wrapped1.modules.return_value = []
        wrapped2.modules.return_value = []
        wrapper = Mock(side_effect=[wrapped1, wrapped2])
        result = _wrap_with_mp_wrapper([stage1, stage2], cfg, wrapper)
        assert wrapper.call_count == 2
        assert wrapper.call_args_list[0].args == (cfg, stage1)
        assert wrapper.call_args_list[1].args == (cfg, stage2)
        assert result == [wrapped1, wrapped2]

    def test_wraps_each_stage_when_bf16_true(self):
        cfg = _make_transformer_config(bf16=True)
        stage = _make_model_module()
        wrapped = Mock()
        wrapped.modules.return_value = []
        wrapper = Mock(return_value=wrapped)
        result = _wrap_with_mp_wrapper([stage], cfg, wrapper)
        wrapper.assert_called_once_with(cfg, stage)
        assert result == [wrapped]

    def test_no_wrap_when_mixed_precision_wrapper_is_none(self):
        cfg = _make_transformer_config(fp16=True)
        stages = [_make_model_module()]
        result = _wrap_with_mp_wrapper(stages, cfg, None)
        assert result is stages

    def test_expert_bias_hook_called_on_qualifying_submodule(self):
        cfg = _make_transformer_config(fp16=True)
        stage = _make_model_module()
        submodule = Mock()
        submodule._maintain_float32_expert_bias = Mock()
        wrapped = Mock()
        wrapped.modules.return_value = [submodule]
        wrapper = Mock(return_value=wrapped)
        _wrap_with_mp_wrapper([stage], cfg, wrapper)
        submodule._maintain_float32_expert_bias.assert_called_once()


# =============================================================================
# Section 5 — TestDdpWrap
# =============================================================================


class TestDdpWrap:
    """_ddp_wrap() wraps each model stage with DDP, Megatron-FSDP, or Torch FSDP2, and optionally broadcasts params."""

    def setup_method(self):
        self.pg = _make_pg()
        self.ddp_config = Mock()
        self.model = [_make_model_module(), _make_model_module()]

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_raises_when_both_fsdp_flags_set(self, mock_stream, mock_curr, mock_ctx, *_):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        with pytest.raises(ValueError):
            _ddp_wrap(
                self.model,
                False,
                self.ddp_config,
                False,
                use_megatron_fsdp=True,
                use_torch_fsdp2=True,
                pg_collection=self.pg,
            )

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_uses_ddp_by_default(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        _ddp_wrap(self.model, False, self.ddp_config, False, pg_collection=self.pg)
        assert mock_ddp.call_count == 2
        mock_fsdp.assert_not_called()
        mock_torch_fsdp.assert_not_called()

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_uses_megatron_fsdp_when_flagged(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        _ddp_wrap(
            self.model, False, self.ddp_config, False, use_megatron_fsdp=True, pg_collection=self.pg
        )
        assert mock_fsdp.call_count == 2
        mock_ddp.assert_not_called()
        mock_torch_fsdp.assert_not_called()

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_uses_torch_fsdp2_when_flagged(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        _ddp_wrap(
            self.model, False, self.ddp_config, False, use_torch_fsdp2=True, pg_collection=self.pg
        )
        assert mock_torch_fsdp.call_count == 2
        mock_ddp.assert_not_called()
        mock_fsdp.assert_not_called()

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_broadcasts_params_when_data_parallel_random_init_true(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        wrapped1 = Mock()
        wrapped2 = Mock()
        mock_ddp.side_effect = [wrapped1, wrapped2]
        _ddp_wrap(self.model, True, self.ddp_config, False, pg_collection=self.pg)
        wrapped1.broadcast_params.assert_called_once()
        wrapped2.broadcast_params.assert_called_once()

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_no_broadcast_when_data_parallel_random_init_false(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        wrapped1 = Mock()
        wrapped2 = Mock()
        mock_ddp.side_effect = [wrapped1, wrapped2]
        _ddp_wrap(self.model, False, self.ddp_config, False, pg_collection=self.pg)
        wrapped1.broadcast_params.assert_not_called()
        wrapped2.broadcast_params.assert_not_called()

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_first_chunk_bucketing_enabled(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        # Use single-element list so only chunk 0 is created
        model = [_make_model_module()]
        _ddp_wrap(model, False, self.ddp_config, False, pg_collection=self.pg)
        call_kwargs = mock_ddp.call_args.kwargs
        assert call_kwargs["disable_bucketing"] is False

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_second_chunk_bucketing_disabled(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        _ddp_wrap(self.model, False, self.ddp_config, False, pg_collection=self.pg)
        # Second call corresponds to chunk index 1
        second_call_kwargs = mock_ddp.call_args_list[1].kwargs
        assert second_call_kwargs["disable_bucketing"] is True

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_overlap_param_gather_disables_bucketing_for_all(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        _ddp_wrap(self.model, False, self.ddp_config, True, pg_collection=self.pg)
        for c in mock_ddp.call_args_list:
            assert c.kwargs["disable_bucketing"] is True

    @patch("megatron.bridge.models.common.unimodal.TorchFullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.FullyShardedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.DistributedDataParallel")
    @patch("megatron.bridge.models.common.unimodal.get_model_config")
    @patch("torch.cuda.stream", new_callable=MagicMock)
    @patch("torch.cuda.current_stream")
    @patch("torch.cuda.Stream")
    def test_returns_list_of_wrapped_modules(
        self, mock_stream, mock_curr, mock_ctx, mock_cfg, mock_ddp, mock_fsdp, mock_torch_fsdp
    ):
        mock_ctx.return_value.__enter__ = Mock(return_value=None)
        mock_ctx.return_value.__exit__ = Mock(return_value=False)
        result = _ddp_wrap(self.model, False, self.ddp_config, False, pg_collection=self.pg)
        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# Section 6 — TestUnimodalBuildDistributedModels
# =============================================================================

_MODULE = "megatron.bridge.models.common.unimodal"


class TestUnimodalBuildDistributedModels:
    """unimodal_build_distributed_models() orchestrates stage building, hooks, parameter setup, GPU allocation, wrapping, and DDP."""

    def setup_method(self):
        self.pg = _make_pg()
        self.mock_model = _make_model_module()
        self.mock_model.parameters.return_value = []
        self.transformer_config = _make_transformer_config()

    # ------------------------------------------------------------------
    # Helpers to build the standard patch stack
    # ------------------------------------------------------------------

    def _standard_patches(self):
        """Returns a dict of started patches that the caller must stop."""
        patches = {
            "bvps": patch(
                f"{_MODULE}.build_virtual_pipeline_stages", return_value=[self.mock_model]
            ),
            "mp_wrap": patch(f"{_MODULE}._wrap_with_mp_wrapper", side_effect=lambda m, *_: m),
            "ddp": patch(f"{_MODULE}._ddp_wrap", side_effect=lambda m, *args, **kwargs: m),
            "print": patch(f"{_MODULE}._print_num_params"),
            "tp_attr": patch(
                f"{_MODULE}.tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes"
            ),
            "cuda_dev": patch("torch.cuda.current_device", return_value=0),
        }
        started = {k: p.start() for k, p in patches.items()}
        # Store originals for teardown
        self._patch_objs = list(patches.values())
        return started

    def _stop_patches(self):
        for p in getattr(self, "_patch_objs", []):
            p.stop()
        self._patch_objs = []

    def teardown_method(self):
        self._stop_patches()

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_raises_when_wrap_with_ddp_true_but_no_ddp_config(self):
        self._standard_patches()
        try:
            with pytest.raises(ValueError):
                unimodal_build_distributed_models(
                    Mock(), self.transformer_config, self.pg, ddp_config=None, wrap_with_ddp=True
                )
        finally:
            self._stop_patches()

    def test_builds_stages_via_build_virtual_pipeline_stages(self):
        mocks = self._standard_patches()
        try:
            build_fn = Mock()
            unimodal_build_distributed_models(
                build_fn, self.transformer_config, self.pg, wrap_with_ddp=False
            )
            mocks["bvps"].assert_called_once_with(
                build_fn,
                self.pg,
                self.transformer_config.virtual_pipeline_model_parallel_size,
                ModelType.encoder_or_decoder,
            )
        finally:
            self._stop_patches()

    def test_meta_device_context_used_when_init_with_meta_device(self):
        transformer_config = _make_transformer_config(init_model_with_meta_device=True)
        mocks = self._standard_patches()
        try:
            # Should complete without error; meta device path goes through build_virtual_pipeline_stages
            unimodal_build_distributed_models(
                Mock(), transformer_config, self.pg, wrap_with_ddp=False
            )
            mocks["bvps"].assert_called_once()
        finally:
            self._stop_patches()

    def test_pre_wrap_hook_applied_and_model_list_updated(self):
        mocks = self._standard_patches()
        try:
            new_list = [_make_model_module()]
            hook = Mock(return_value=new_list)
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False, pre_wrap_hook=hook
            )
            hook.assert_called_once()
            # The mp_wrapper receives the new list
            mp_call_arg = mocks["mp_wrap"].call_args.args[0]
            assert mp_call_arg is new_list
        finally:
            self._stop_patches()

    def test_pre_wrap_hook_returning_none_keeps_original_list(self):
        mocks = self._standard_patches()
        try:
            hook = Mock(return_value=None)
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False, pre_wrap_hook=hook
            )
            # mp_wrapper receives the original list [self.mock_model]
            mp_call_arg = mocks["mp_wrap"].call_args.args[0]
            assert mp_call_arg == [self.mock_model]
        finally:
            self._stop_patches()

    def test_pre_wrap_hook_not_callable_raises_type_error(self):
        self._standard_patches()
        try:
            with pytest.raises(TypeError):
                unimodal_build_distributed_models(
                    Mock(),
                    self.transformer_config,
                    self.pg,
                    wrap_with_ddp=False,
                    pre_wrap_hook="not_callable",
                )
        finally:
            self._stop_patches()

    def test_pre_wrap_hook_none_is_skipped(self):
        self._standard_patches()
        try:
            # Should not raise
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False, pre_wrap_hook=None
            )
        finally:
            self._stop_patches()

    def test_tensor_parallel_attrs_set_for_each_param(self):
        param = Mock()
        self.mock_model.parameters.return_value = [param]
        mocks = self._standard_patches()
        try:
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False
            )
            mocks["tp_attr"].assert_called_once_with(param)
        finally:
            self._stop_patches()

    def test_cuda_called_when_not_fsdp2_not_cpu_not_meta(self):
        self._standard_patches()
        try:
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False, use_torch_fsdp2=False
            )
            self.mock_model.cuda.assert_called_once()
        finally:
            self._stop_patches()

    def test_cuda_not_called_when_use_torch_fsdp2(self):
        self._standard_patches()
        try:
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False, use_torch_fsdp2=True
            )
            self.mock_model.cuda.assert_not_called()
        finally:
            self._stop_patches()

    def test_cuda_not_called_when_use_cpu_initialization(self):
        self.transformer_config = _make_transformer_config(use_cpu_initialization=True)
        self._standard_patches()
        try:
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False
            )
            self.mock_model.cuda.assert_not_called()
        finally:
            self._stop_patches()

    def test_cuda_not_called_when_init_model_with_meta_device(self):
        self.transformer_config = _make_transformer_config(init_model_with_meta_device=True)
        self._standard_patches()
        try:
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False
            )
            self.mock_model.cuda.assert_not_called()
        finally:
            self._stop_patches()

    def test_meta_materialization_called_when_meta_not_fsdp(self):
        self.transformer_config = _make_transformer_config(init_model_with_meta_device=True)
        self._standard_patches()
        try:
            with patch(
                f"{_MODULE}.to_empty_if_meta_device", return_value=self.mock_model
            ) as mock_toempty:
                unimodal_build_distributed_models(
                    Mock(),
                    self.transformer_config,
                    self.pg,
                    wrap_with_ddp=False,
                    use_torch_fsdp2=False,
                    use_megatron_fsdp=False,
                )
                assert mock_toempty.call_count == 1
        finally:
            self._stop_patches()

    def test_meta_materialization_skipped_with_torch_fsdp2(self):
        self.transformer_config = _make_transformer_config(init_model_with_meta_device=True)
        self._standard_patches()
        try:
            with patch(
                f"{_MODULE}.to_empty_if_meta_device", return_value=self.mock_model
            ) as mock_toempty:
                unimodal_build_distributed_models(
                    Mock(),
                    self.transformer_config,
                    self.pg,
                    wrap_with_ddp=False,
                    use_torch_fsdp2=True,
                )
                mock_toempty.assert_not_called()
        finally:
            self._stop_patches()

    def test_meta_materialization_skipped_with_megatron_fsdp(self):
        self.transformer_config = _make_transformer_config(init_model_with_meta_device=True)
        self._standard_patches()
        try:
            with patch(
                f"{_MODULE}.to_empty_if_meta_device", return_value=self.mock_model
            ) as mock_toempty:
                unimodal_build_distributed_models(
                    Mock(),
                    self.transformer_config,
                    self.pg,
                    wrap_with_ddp=False,
                    use_megatron_fsdp=True,
                )
                mock_toempty.assert_not_called()
        finally:
            self._stop_patches()

    def test_mp_wrapper_applied(self):
        mocks = self._standard_patches()
        try:
            mp_wrapper = Mock()
            unimodal_build_distributed_models(
                Mock(),
                self.transformer_config,
                self.pg,
                wrap_with_ddp=False,
                mixed_precision_wrapper=mp_wrapper,
            )
            mocks["mp_wrap"].assert_called_once_with(
                [self.mock_model], self.transformer_config, mp_wrapper
            )
        finally:
            self._stop_patches()

    def test_ddp_wrap_called_when_wrap_with_ddp_true(self):
        mocks = self._standard_patches()
        try:
            ddp_config = Mock()
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, ddp_config=ddp_config, wrap_with_ddp=True
            )
            mocks["ddp"].assert_called_once()
        finally:
            self._stop_patches()

    def test_ddp_wrap_not_called_when_wrap_with_ddp_false(self):
        mocks = self._standard_patches()
        try:
            unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, wrap_with_ddp=False
            )
            mocks["ddp"].assert_not_called()
        finally:
            self._stop_patches()

    def test_returns_final_model_list(self):
        mocks = self._standard_patches()
        try:
            ddp_config = Mock()
            ddp_result = [_make_model_module()]
            mocks["ddp"].side_effect = lambda m, *args, **kwargs: ddp_result
            result = unimodal_build_distributed_models(
                Mock(), self.transformer_config, self.pg, ddp_config=ddp_config, wrap_with_ddp=True
            )
            assert result is ddp_result
        finally:
            self._stop_patches()
