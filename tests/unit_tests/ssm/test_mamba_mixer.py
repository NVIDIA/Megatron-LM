# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm import mamba_mixer as mamba_mixer_module
from megatron.core.ssm.mamba_context_parallel import tex
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import (
    MambaMixer,
    MambaMixerSubmodules,
    _get_varlen_chunk_metadata,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.tuning import policy as tuning_policy
from tests.unit_tests.test_utilities import Utils


@pytest.mark.internal
class TestMambaMixer:

    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def get_mixer(self, tp_size=1, cp_size=1, use_mem_eff_path=True):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,  # The Mamba layer places several constraints on this
            # Need to specify num_attention_heads and num_layers or TransformerConfig
            # will generate errors.
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            use_mamba_mem_eff_path=use_mem_eff_path,
        )
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        assert isinstance(
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            MambaMixerSubmodules,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            transformer_config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            transformer_config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        )
        mixer.cuda()
        return mixer

    @pytest.mark.parametrize(
        "tp_size,cp_size,use_mem_eff_path",
        [
            (1, 1, True),
            (1, 1, False),
            (8, 1, True),
            (4, 2, True),
            (2, 4, True),
            (1, 8, True),
            (1, 8, False),
        ],
    )
    def test_gpu_forward(self, tp_size, cp_size, use_mem_eff_path):
        mixer = self.get_mixer(1, 1, use_mem_eff_path)
        micro_batch_size = 2
        sequence_length = 32
        hidden_states = torch.ones((sequence_length, micro_batch_size, mixer.config.hidden_size))
        hidden_states = hidden_states.cuda()
        output, bias = mixer(hidden_states)
        assert mixer.config.mamba_num_heads == None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == mixer.config.hidden_size
        assert output.dtype == torch.float32

    def test_variable_batch_size_inference(self):
        mixer = self.get_mixer()

        # Test cases where batch size decreases, remains the same, and increases
        micro_batch_sizes = [4, 2, 2, 8]
        sequence_length = 32
        inference_context = StaticInferenceContext(
            max_batch_size=max(micro_batch_sizes), max_sequence_length=sequence_length
        )

        with InferenceMode.active():
            for micro_batch_size in micro_batch_sizes:
                inference_context.max_seqlen = inference_context.max_sequence_length
                inference_context.seqlen_offset = inference_context.sequence_len_offset
                hidden_states = torch.ones(
                    (sequence_length, micro_batch_size, mixer.config.hidden_size)
                )
                hidden_states = hidden_states.cuda()
                output, bias = mixer(hidden_states, inference_context=inference_context)
                assert mixer.config.mamba_num_heads == None
                assert output.shape[0] == sequence_length
                assert output.shape[1] == micro_batch_size
                assert output.shape[2] == mixer.config.hidden_size
                assert output.dtype == torch.float32

    @pytest.mark.parametrize("refresh_on_eval", [False, True], ids=["direct", "eval"])
    def test_refreshes_decode_cache_in_place(self, refresh_on_eval):
        mixer = self.get_mixer()
        cache_ptr = mixer._A_neg_exp_cache.data_ptr()

        with torch.no_grad():
            mixer.A_log.add_(1.0)
        if refresh_on_eval:
            mixer.train()
            mixer.eval()
        else:
            mixer.refresh_cache()

        torch.testing.assert_close(mixer._A_neg_exp_cache, -torch.exp(mixer.A_log.float()))
        assert mixer._A_neg_exp_cache.data_ptr() == cache_ptr
        assert not mixer._A_neg_exp_cache_stale


def _packed_seq_params(seqlens, total_tokens=None):
    cu_seqlens = torch.tensor([0] + seqlens, device="cuda").cumsum(0).to(torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(seqlens),
        max_seqlen_kv=max(seqlens),
        total_tokens=sum(seqlens) if total_tokens is None else total_tokens,
    )


class TestVarlenChunkMetadata:

    def test_chunk_layout(self):
        metadata = _get_varlen_chunk_metadata(_packed_seq_params([300, 1000, 200]), 128, 1500)
        assert metadata.cu_chunk_seqlens.tolist() == (
            [0, 128, 256] + list(range(300, 1300, 128)) + [1300, 1428, 1500]
        )
        assert metadata.seq_idx_for_varlen.tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]
        # The empty extra sequence after cu_seqlens[-1] has no chunk.
        assert metadata.last_chunk_indices.tolist() == [2, 10, 12, 12]

    def test_trailing_tokens_and_empty_sequences(self):
        # Tokens beyond the last boundary form one extra sequence, as in PackedSeqParams.seq_idx.
        metadata = _get_varlen_chunk_metadata(
            _packed_seq_params([5, 0, 130], total_tokens=140), 128, 140
        )
        assert metadata.cu_chunk_seqlens.tolist() == [0, 5, 133, 135, 140]
        assert metadata.seq_idx_for_varlen.tolist() == [0, 2, 2, 3]
        assert metadata.last_chunk_indices.tolist() == [0, 0, 2, 3]

    def test_cache(self):
        params = _packed_seq_params([300, 1000, 200])
        metadata = _get_varlen_chunk_metadata(params, 128, 1500)
        assert _get_varlen_chunk_metadata(params, 128, 1500) is metadata
        # A new pack with the same boundaries is a different microbatch.
        other = _get_varlen_chunk_metadata(_packed_seq_params([300, 1000, 200]), 128, 1500)
        assert other is not metadata
        assert torch.equal(other.cu_chunk_seqlens, metadata.cu_chunk_seqlens)
        assert _get_varlen_chunk_metadata(params, 64, 1500) is not metadata

    def test_cache_is_not_a_field(self):
        # CUDA graphs and TE attention see only the dataclass fields, which must not change.
        params = _packed_seq_params([300, 1000, 200])
        fields = {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}
        _get_varlen_chunk_metadata(params, 128, 1500)
        assert {f.name for f in dataclasses.fields(params)} == fields.keys()
        assert all(getattr(params, name) is value for name, value in fields.items())
        # A copy made with dataclasses.replace starts without the cached layout.
        assert not hasattr(dataclasses.replace(params), "_mamba_varlen_chunk_metadata")

    def test_cache_follows_the_boundaries(self):
        params = _packed_seq_params([300, 1000, 200])
        metadata = _get_varlen_chunk_metadata(params, 128, 1500)
        # Boundaries modified in place, as a reused buffer would be.
        params.cu_seqlens_q.copy_(torch.tensor([0, 1000, 1300, 1500], device="cuda"))
        moved = _get_varlen_chunk_metadata(params, 128, 1500)
        assert moved is not metadata
        assert moved.last_chunk_indices.tolist() == [7, 10, 12, 12]
        # Boundaries replaced, e.g. padded ones set on a shallow copy.
        params.cu_seqlens_q_padded = torch.tensor([0, 384, 1408, 1500], device="cuda")
        padded = _get_varlen_chunk_metadata(params, 128, 1500)
        assert padded is not moved
        assert padded.last_chunk_indices.tolist() == [2, 10, 11, 11]

    def test_inference_mode(self):
        # A pack built under inference_mode has inference-tensor boundaries.
        with torch.inference_mode():
            inference_params = _packed_seq_params([300, 1000, 200])
            _get_varlen_chunk_metadata(inference_params, 128, 1500)
        metadata = _get_varlen_chunk_metadata(inference_params, 128, 1500)
        assert not metadata.cu_chunk_seqlens.is_inference()
        # A layout built under inference_mode is not cached for a later training forward.
        params = _packed_seq_params([300, 1000, 200])
        with torch.inference_mode():
            _get_varlen_chunk_metadata(params, 128, 1500)
        assert not _get_varlen_chunk_metadata(params, 128, 1500).cu_chunk_seqlens.is_inference()


class TestVarlenCudaGraphGuards:

    def test_layer_runs_eagerly_under_training_graphs(self):
        # MambaLayer bypasses its training CUDA graph for the varlen path, like a partial MoE layer.
        layer = SimpleNamespace(
            cudagraph_manager=object(), config=SimpleNamespace(mamba_varlen_ssd_training=True)
        )
        packed_seq_params = _packed_seq_params([300, 200])
        assert not MambaLayer._should_call_local_cudagraph(
            layer, hidden_states=None, packed_seq_params=packed_seq_params
        )
        assert MambaLayer._should_call_local_cudagraph(layer, hidden_states=None)
        layer.config.mamba_varlen_ssd_training = False
        assert MambaLayer._should_call_local_cudagraph(
            layer, hidden_states=None, packed_seq_params=packed_seq_params
        )

    def test_rejects_full_iteration_graphs(self):
        with pytest.raises(AssertionError, match="full_iteration"):
            TransformerConfig(
                hidden_size=256,
                num_layers=1,
                num_attention_heads=1,
                cuda_graph_impl="full_iteration",
                mamba_varlen_ssd_training=True,
            )


@pytest.mark.internal
class TestMambaMixerVarlenSSD:

    @pytest.fixture(autouse=True)
    def deterministic_ssd(self, monkeypatch):
        # The bitwise comparisons need ordered reductions: outside deterministic mode the varlen
        # backward adds ddA_cumsum and ddt with atomics.
        monkeypatch.setattr(tuning_policy, "_deterministic_override", True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def get_mixer(self, mamba_varlen_ssd_training, **config_overrides):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            mamba_varlen_ssd_training=mamba_varlen_ssd_training,
            **config_overrides,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            transformer_config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            transformer_config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        )
        return mixer.cuda().to(torch.bfloat16)

    @staticmethod
    def run_mixer(mixer, hidden_states, grad_output, packed_seq_params=None):
        hidden_states = hidden_states.detach().clone().requires_grad_()
        mixer.zero_grad()
        output, _ = mixer(hidden_states, packed_seq_params=packed_seq_params)
        output.backward(grad_output)
        param_grads = {name: p.grad.clone() for name, p in mixer.named_parameters()}
        return output.detach(), hidden_states.grad, param_grads

    def test_packing_invariance(self):
        mixer = self.get_mixer(mamba_varlen_ssd_training=True)
        hidden_size, target, total = mixer.config.hidden_size, 1000, 1977
        torch.manual_seed(0)
        target_states = torch.randn(target, 1, hidden_size, device="cuda")
        target_grad = torch.randn(target, 1, hidden_size, device="cuda")
        results = []
        # Same total length, so the GEMMs see the same shapes; only the packing differs.
        for before in [300, 977, 0]:
            after = total - target - before
            seqlens = [n for n in (before, target, after) if n > 0]
            hidden_states = torch.randn(total, 1, hidden_size, device="cuda")
            grad_output = torch.randn_like(hidden_states)
            hidden_states[before : before + target] = target_states
            grad_output[before : before + target] = target_grad
            output, dhidden, _ = self.run_mixer(
                mixer, hidden_states.bfloat16(), grad_output.bfloat16(), _packed_seq_params(seqlens)
            )
            results.append((output[before : before + target], dhidden[before : before + target]))
        for output, dhidden in results[1:]:
            assert torch.equal(output, results[0][0])
            assert torch.equal(dhidden, results[0][1])

    def test_empty_sequences(self):
        # Empty sequences get no chunk and tokens beyond the last boundary form one extra
        # sequence; neither may change the result of the same sequences without them.
        mixer = self.get_mixer(mamba_varlen_ssd_training=True)
        torch.manual_seed(0)
        hidden_states = torch.randn(1977, 1, mixer.config.hidden_size, device="cuda").bfloat16()
        grad_output = torch.randn_like(hidden_states)
        expected = self.run_mixer(
            mixer, hidden_states, grad_output, _packed_seq_params([300, 1000, 677])
        )
        for packed_seq_params in [
            _packed_seq_params([0, 300, 0, 1000, 0, 677, 0]),
            _packed_seq_params([300, 0, 0, 1000, 677]),
            _packed_seq_params([300, 1000], total_tokens=1977),
        ]:
            output, dhidden, param_grads = self.run_mixer(
                mixer, hidden_states, grad_output, packed_seq_params
            )
            assert torch.equal(output, expected[0])
            assert torch.equal(dhidden, expected[1])
            for name, grad in param_grads.items():
                if name in ("conv1d_weight", "conv1d_bias"):
                    # The fixture's tuning override does not reach causal_conv1d, which follows
                    # CAUSAL_CONV1D_DETERMINISTIC: it reduces its weight and bias gradients with
                    # atomicAdd, so they can differ even for the same input.
                    torch.testing.assert_close(grad, expected[2][name], atol=4e-3, rtol=1e-2)
                else:
                    assert torch.equal(grad, expected[2][name]), name

    def test_packing_invariance_with_heads_sharing_a_group(self):
        # With several heads per group the backward splits the head reduction of dB and dC by the
        # pack's chunk count, so gradients can differ in the last bits; the output cannot.
        mixer = self.get_mixer(mamba_varlen_ssd_training=True, mamba_num_groups=1)
        hidden_size, target, total = mixer.config.hidden_size, 1000, 4096
        torch.manual_seed(0)
        target_states = torch.randn(target, 1, hidden_size, device="cuda")
        target_grad = torch.randn(target, 1, hidden_size, device="cuda")
        results = []
        for seqlens_before in ([], [1], [300, 1, 7], [129] * 20):
            before = sum(seqlens_before)
            seqlens = seqlens_before + [target, total - target - before]
            hidden_states = torch.randn(total, 1, hidden_size, device="cuda")
            grad_output = torch.randn_like(hidden_states)
            hidden_states[before : before + target] = target_states
            grad_output[before : before + target] = target_grad
            output, dhidden, _ = self.run_mixer(
                mixer, hidden_states.bfloat16(), grad_output.bfloat16(), _packed_seq_params(seqlens)
            )
            results.append((output[before : before + target], dhidden[before : before + target]))
        for output, dhidden in results[1:]:
            assert torch.equal(output, results[0][0])
            torch.testing.assert_close(dhidden, results[0][1])

    def test_rl_packed_bin(self):
        # A bin from the RL packer: slots aligned to 8 tokens (cu_seqlens_q_padded), a trailing
        # ghost slot, and cu_seqlens padded to a fixed length by repeating the last boundary.
        sequence_packing_utils = pytest.importorskip("megatron.rl.sequence_packing_utils")
        seq_lengths, slots = [300, 517, 211], [304, 520, 216]
        bin_size = 1104
        packing_info = sequence_packing_utils.PackingInfo(
            bin_seq_indices=[[0, 1, 2]],
            seq_starts={0: [0, 304, 824, 1040]},
            seq_lengths=seq_lengths,
            seq_to_bin_idx=[0, 0, 0],
            packing_algo='fifo',
        )
        bin_params = sequence_packing_utils.create_packed_seq_params_for_bin(
            packing_info=packing_info,
            bin_idx=0,
            bin_size=bin_size,
            max_sequences_per_bin=6,
            device=torch.device("cuda"),
            seq_length_multiple=8,
        )
        mixer = self.get_mixer(mamba_varlen_ssd_training=True)
        torch.manual_seed(0)
        hidden_states = torch.randn(bin_size, 1, mixer.config.hidden_size, device="cuda").bfloat16()
        grad_output = torch.randn_like(hidden_states)
        got = self.run_mixer(mixer, hidden_states, grad_output, bin_params)
        # The varlen path follows the padded slots; the fixed-length padding adds nothing.
        expected = self.run_mixer(
            mixer, hidden_states, grad_output, _packed_seq_params(slots + [bin_size - 1040])
        )
        assert torch.equal(got[0], expected[0])
        assert torch.equal(got[1], expected[1])
        for name, grad in got[2].items():
            if name in ("conv1d_weight", "conv1d_bias"):
                # causal_conv1d atomics, see test_empty_sequences.
                torch.testing.assert_close(grad, expected[2][name], atol=4e-3, rtol=1e-2)
            else:
                assert torch.equal(grad, expected[2][name]), name

    @pytest.mark.parametrize("mamba_varlen_ssd_training", [True, False])
    def test_rejects_dynamic_context_parallel(self, monkeypatch, mamba_varlen_ssd_training):
        mixer = self.get_mixer(mamba_varlen_ssd_training=mamba_varlen_ssd_training)
        hidden_states = torch.randn(500, 1, 256, device="cuda").bfloat16()
        packed_seq_params = _packed_seq_params([300, 200])
        # The static group and another group object over the same ranks are accepted; a group
        # over other ranks is rejected, whether its size differs or not.
        static_ranks = torch.distributed.get_process_group_ranks(mixer.cp.cp_group)
        same_ranks = SimpleNamespace(size=lambda: len(static_ranks))
        larger = SimpleNamespace(size=lambda: len(static_ranks) + 1)
        other_ranks = SimpleNamespace(size=lambda: len(static_ranks))
        group_ranks = {
            id(mixer.cp.cp_group): static_ranks,
            id(same_ranks): static_ranks,
            id(larger): static_ranks + [max(static_ranks) + 1],
            id(other_ranks): [r + 1 for r in static_ranks],
        }
        monkeypatch.setattr(
            torch.distributed, "get_process_group_ranks", lambda group: group_ranks[id(group)]
        )
        for group, accepted in [
            (mixer.cp.cp_group, True),
            (same_ranks, True),
            (larger, False),
            (other_ranks, False),
        ]:
            packed_seq_params.local_cp_size = group.size()
            packed_seq_params.cp_group = group
            if accepted:
                mixer(hidden_states, packed_seq_params=packed_seq_params)
            else:
                with pytest.raises(AssertionError, match="dynamic context parallelism"):
                    mixer(hidden_states, packed_seq_params=packed_seq_params)

    def test_rejects_cuda_graph_capture(self, monkeypatch):
        mixer = self.get_mixer(mamba_varlen_ssd_training=True)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        zxBCdt = torch.empty(500, 1, 8, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="CUDA graph capture"):
            mixer._ssm_training_varlen(zxBCdt, _packed_seq_params([300, 200]))

    @pytest.mark.parametrize("states_dtype", [None, torch.float32])
    def test_training_ssm_states_dtype(self, monkeypatch, states_dtype):
        # The varlen SSD gets the same state dtype as the fused kernel: the configured one, or
        # params_dtype.
        varlen_fn = mamba_mixer_module.mamba_split_conv1d_scan_combined_varlen
        seen = []

        def record(*args, **kwargs):
            seen.append(kwargs["state_dtype"])
            return varlen_fn(*args, **kwargs)

        monkeypatch.setattr(mamba_mixer_module, "mamba_split_conv1d_scan_combined_varlen", record)
        mixer = self.get_mixer(
            mamba_varlen_ssd_training=True,
            params_dtype=torch.bfloat16,
            mamba_training_ssm_states_dtype=states_dtype,
        )
        torch.manual_seed(0)
        hidden_states = torch.randn(500, 1, 256, device="cuda").bfloat16()
        output, _, _ = self.run_mixer(
            mixer, hidden_states, torch.randn_like(hidden_states), _packed_seq_params([300, 200])
        )
        assert seen == [states_dtype or torch.bfloat16]
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize(
        "states_dtype, seqlens",
        [
            (None, [300, 1000, 1, 128, 200]),
            (torch.bfloat16, [300, 1000, 1, 128, 200]),
            # Packs of a single chunk.
            (None, [100]),
            (None, [128]),
        ],
    )
    def test_matches_unpacked_fused_path(self, states_dtype, seqlens):
        varlen_mixer = self.get_mixer(
            mamba_varlen_ssd_training=True, mamba_training_ssm_states_dtype=states_dtype
        )
        fused_mixer = self.get_mixer(
            mamba_varlen_ssd_training=False, mamba_training_ssm_states_dtype=states_dtype
        )
        fused_mixer.load_state_dict(varlen_mixer.state_dict())
        torch.manual_seed(0)
        hidden_states = torch.randn(sum(seqlens), 1, 256, device="cuda").bfloat16()
        grad_output = torch.randn_like(hidden_states)

        output, dhidden, param_grads = self.run_mixer(
            varlen_mixer, hidden_states, grad_output, _packed_seq_params(seqlens)
        )

        # Reference: every sequence on its own through the fused kernel, so its chunks start at
        # the sequence start as in the varlen path.
        ref_outputs, ref_dhiddens, ref_param_grads = [], [], {}
        start = 0
        for seqlen in seqlens:
            ref_output, ref_dhidden, grads = self.run_mixer(
                fused_mixer,
                hidden_states[start : start + seqlen],
                grad_output[start : start + seqlen],
            )
            ref_outputs.append(ref_output)
            ref_dhiddens.append(ref_dhidden)
            for name, grad in grads.items():
                ref_param_grads[name] = ref_param_grads.get(name, 0) + grad.float()
            start += seqlen

        def assert_close(actual, expected):
            actual, expected = actual.float(), expected.float()
            err = (actual - expected).abs().max() / expected.abs().max()
            assert err < 1e-2, err

        assert_close(output, torch.cat(ref_outputs))
        assert_close(dhidden, torch.cat(ref_dhiddens))
        assert param_grads.keys() == ref_param_grads.keys()
        for name, grad in param_grads.items():
            assert_close(grad, ref_param_grads[name])


def _gather_token_rows(rows, values, group, num_tokens):
    """Places the token rows held by every rank of `group` into one (num_tokens, ...) tensor."""
    world_size = torch.distributed.get_world_size(group)
    all_rows = [torch.empty_like(rows) for _ in range(world_size)]
    all_values = [torch.empty_like(values) for _ in range(world_size)]
    torch.distributed.all_gather(all_rows, rows, group=group)
    torch.distributed.all_gather(all_values, values.contiguous(), group=group)
    output = values.new_zeros((num_tokens,) + tuple(values.shape[1:]))
    for rank_rows, rank_values in zip(all_rows, all_values):
        output[rank_rows] = rank_values
    return output


def _all_ranks(flag):
    flag = torch.tensor([int(flag)], device="cuda")
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


@pytest.mark.internal
class TestMambaMixerVarlenSSDParallel:
    """The varlen SSD path under tensor, context and sequence parallelism."""

    num_tokens = 2048

    @pytest.fixture(autouse=True)
    def deterministic_ssd(self, monkeypatch):
        monkeypatch.setattr(tuning_policy, "_deterministic_override", True)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def get_mixer(self, tp_size, cp_size, sequence_parallel):
        if Utils.world_size < tp_size * cp_size:
            pytest.skip(f"needs {tp_size * cp_size} GPUs")
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)
        transformer_config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=8,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=sequence_parallel,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        mixer = MambaMixer(
            transformer_config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            transformer_config.hidden_size,
            layer_number=1,
            pg_collection=pg_collection,
        )
        return mixer.cuda().to(torch.bfloat16)

    def random_tokens(self, num_tokens, generator):
        tokens = torch.randn(num_tokens, 1, 256, device="cuda", generator=generator).bfloat16()
        torch.distributed.broadcast(tokens, 0)
        return tokens

    def run_mixer(self, mixer, hidden_states, grad_output, seqlens, varlen):
        """Runs the mixer on this rank's share of a packed row and returns, in global token
        order, the out_proj input (this TP rank's slice), the output and the input gradient,
        followed by this rank's parameter gradients."""
        tp_size = mixer.config.tensor_model_parallel_size
        cp_size = mixer.config.context_parallel_size
        cp_group = parallel_state.get_context_parallel_group()
        packed_seq_params = _packed_seq_params(seqlens)
        rows = torch.arange(self.num_tokens, device="cuda")
        if cp_size > 1:
            rows = tex.thd_get_partitioned_indices(
                packed_seq_params.cu_seqlens_q,
                self.num_tokens,
                cp_size,
                parallel_state.get_context_parallel_rank(),
            ).long()
        input_rows = rows
        if mixer.config.sequence_parallel:
            input_rows = rows.chunk(tp_size)[parallel_state.get_tensor_model_parallel_rank()]

        captured = {}
        hook = mixer.out_proj.register_forward_pre_hook(
            lambda module, args: captured.update(out_proj_input=args[0].detach())
        )
        mixer.config.mamba_varlen_ssd_training = varlen
        mixer.zero_grad()
        local_hidden_states = hidden_states[input_rows].detach().clone().requires_grad_()
        output, _ = mixer(local_hidden_states, packed_seq_params=packed_seq_params)
        output.backward(grad_output[input_rows])
        hook.remove()

        def gather(values):
            if mixer.config.sequence_parallel:
                values = _gather_token_rows(
                    input_rows,
                    values,
                    parallel_state.get_tensor_model_parallel_group(),
                    self.num_tokens,
                )[rows]
            return _gather_token_rows(rows, values, cp_group, self.num_tokens)

        out_proj_input = _gather_token_rows(
            rows, captured["out_proj_input"], cp_group, self.num_tokens
        )
        param_grads = {name: p.grad.clone() for name, p in mixer.named_parameters()}
        return (
            out_proj_input,
            gather(output.detach()),
            gather(local_hidden_states.grad),
            param_grads,
        )

    @pytest.mark.parametrize(
        "tp_size,cp_size,sequence_parallel",
        [(2, 1, False), (4, 1, True), (1, 4, False), (2, 4, True), (4, 2, True)],
    )
    def test_matches_fused_path(self, tp_size, cp_size, sequence_parallel):
        # With every length a multiple of the chunk size, the fused kernel's fixed chunk grid
        # coincides with the varlen chunks.
        mixer = self.get_mixer(tp_size, cp_size, sequence_parallel)
        generator = torch.Generator(device="cuda").manual_seed(0)
        hidden_states = self.random_tokens(self.num_tokens, generator)
        grad_output = self.random_tokens(self.num_tokens, generator)
        seqlens = [256, 1024, 384, 384]
        varlen = self.run_mixer(mixer, hidden_states, grad_output, seqlens, varlen=True)
        fused = self.run_mixer(mixer, hidden_states, grad_output, seqlens, varlen=False)

        def close(actual, expected, rtol):
            actual, expected = actual.float(), expected.float()
            return (actual - expected).abs().max() < rtol * expected.abs().max()

        assert _all_ranks(all(close(v, f, 1e-2) for v, f in zip(varlen[:3], fused[:3])))
        # Parameter gradients are bf16 sums over all tokens, e.g. one dD element per local head,
        # which cancel heavily for random inputs, so the summation order shows more.
        assert _all_ranks(all(close(varlen[3][name], fused[3][name], 5e-2) for name in varlen[3]))

    @pytest.mark.parametrize(
        "tp_size,cp_size,sequence_parallel",
        [(2, 1, False), (4, 1, True), (1, 4, False), (2, 4, True), (4, 2, True)],
    )
    def test_packing_invariance(self, tp_size, cp_size, sequence_parallel):
        mixer = self.get_mixer(tp_size, cp_size, sequence_parallel)
        generator = torch.Generator(device="cuda").manual_seed(0)
        hidden_states = self.random_tokens(self.num_tokens, generator)
        grad_output = self.random_tokens(self.num_tokens, generator)
        target = 1000
        target_states = self.random_tokens(target, generator)
        target_grad = self.random_tokens(target, generator)
        results = []
        # Same total length and every length a multiple of 2 * cp_size, as THD CP requires.
        for before in [296, 1048, 0]:
            after = self.num_tokens - target - before
            hidden_states[before : before + target] = target_states
            grad_output[before : before + target] = target_grad
            seqlens = [n for n in (before, target, after) if n > 0]
            result = self.run_mixer(mixer, hidden_states, grad_output, seqlens, varlen=True)
            results.append([t[before : before + target] for t in result[:3]])

        def same(index):
            return _all_ranks(all(torch.equal(r[index], results[0][index]) for r in results[1:]))

        # The SSD output is packing invariant. The mixer output and input gradient also go
        # through TP reductions, whose summation order can depend on the token position for
        # more than two ranks.
        assert same(0)
        if tp_size <= 2:
            assert same(1)
            assert same(2)


class TestMambaMixerErrorChecks:

    @pytest.mark.parametrize(
        "hidden_size, ngroups, tp_size, expected_error_message",
        [
            (65, 8, 1, "d_inner must be evenly divisible by headdim"),
            (96, 8, 2, "nheads must be evenly divisble by tp_size"),  # nheads = 3
            (128, 2, 4, "ngroups must be evenly divisible by tp_size"),
            (128, 8, 4, "nheads must be evenly divisible by ngroups"),  # nheads = 4
        ],
    )
    def test_error_check(self, hidden_size, ngroups, tp_size, expected_error_message):
        Utils.initialize_model_parallel(tp_size)
        transformer_config = TransformerConfig(
            hidden_size=hidden_size,
            num_layers=1,
            num_attention_heads=1,
            use_cpu_initialization=True,
            mamba_num_groups=ngroups,
        )
        assert isinstance(hybrid_stack_spec.submodules, HybridStackSubmodules)
        assert isinstance(hybrid_stack_spec.submodules.mamba_layer.submodules, MambaLayerSubmodules)
        assert isinstance(
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            MambaMixerSubmodules,
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        with pytest.raises(AssertionError, match=expected_error_message):
            MambaMixer(
                transformer_config,
                hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
                transformer_config.hidden_size,
                pg_collection=pg_collection,
            )
        Utils.destroy_model_parallel()
