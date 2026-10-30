# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core.inference.contexts.static_context import StaticInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_block import HybridStackSubmodules
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_layer import MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
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


class TestMambaMixerDecode:
    """``ssm_decode`` against a float64 Mamba-2 decode step written from the recurrence."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1
        )
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def reference_decode_step(mixer, zxBCdt, conv_state, ssm_state):
        """One decode token for batch-first ``zxBCdt`` of shape (b, 1, proj), in float64.

        Returns the output and the updated conv and SSM states; the inputs are not modified.
        """
        d_inner, G, N = mixer.d_inner_local_tp, mixer.ngroups_local_tp, mixer.d_state
        H, P = mixer.nheads_local_tp, mixer.headdim
        b = zxBCdt.shape[0]
        z, xBC, dt = torch.split(zxBCdt[:, 0].double(), [d_inner, d_inner + 2 * G * N, H], dim=-1)

        # Causal depthwise conv over the rolling window: shift left, append the new input.
        conv_state = torch.cat([conv_state.double()[:, :, 1:], xBC.unsqueeze(-1)], dim=-1)
        weight = mixer.conv1d_weight.double()[:, 0, :]
        xBC = F.silu((conv_state * weight).sum(-1) + mixer.conv1d_bias.double())
        x, B, C = torch.split(xBC, [d_inner, G * N, G * N], dim=-1)
        x = x.reshape(b, H, P)
        B = B.reshape(b, G, N).repeat_interleave(H // G, dim=1)
        C = C.reshape(b, G, N).repeat_interleave(H // G, dim=1)

        # h' = exp(dt * A) h + dt * B x, then y = (C h' + D x) * silu(z).
        dt = F.softplus(dt + mixer.dt_bias.double())
        dA = torch.exp(dt * -torch.exp(mixer.A_log.double()))  # (b, H)
        dBx = (dt[..., None] * x)[..., None] * B[:, :, None, :]  # (b, H, P, N)
        ssm_state = ssm_state.double() * dA[..., None, None] + dBx
        D = mixer.D.double().reshape(H, P) if mixer.D_has_hdim else mixer.D.double()[:, None]
        y = ((ssm_state * C[:, :, None, :]).sum(-1) + D * x) * F.silu(z.reshape(b, H, P))
        return y.reshape(b, 1, H * P), conv_state, ssm_state

    @pytest.mark.parametrize(
        "seq_len, slot_map",
        [(1, False), (1, True), (3, True)],
        ids=["batch_order", "slot_map", "speculative"],
    )
    @pytest.mark.parametrize(
        "D_has_hdim, uniform_D",
        [(False, False), (True, True), (True, False)],
        ids=["per_head_D", "per_channel_uniform_D", "per_channel_D"],
    )
    def test_decode_matches_reference(self, D_has_hdim, uniform_D, seq_len, slot_map):
        config = TransformerConfig(
            hidden_size=256, num_layers=1, num_attention_heads=1, use_cpu_initialization=True
        )
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'cp'])
        # Without rmsnorm the z gate is applied inside the decode kernel, so the reference
        # needs no norm.
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            D_has_hdim=D_has_hdim,
            rmsnorm=False,
            layer_number=1,
            pg_collection=pg_collection,
        )
        mixer.cuda().eval()
        H, P = mixer.nheads_local_tp, mixer.headdim
        assert mixer.D.shape == ((H * P,) if D_has_hdim else (H,))

        # The default all-ones D would hide a mis-indexed D.
        generator = torch.Generator(device="cuda").manual_seed(0)
        with torch.no_grad():
            if uniform_D:
                mixer.D.fill_(0.37)
            else:
                mixer.D.copy_(torch.randn(mixer.D.shape, generator=generator, device="cuda"))

        # With a slot map (dynamic batching), requests 0 and 1 own state slots 2 and 0, and
        # slot 1 belongs to a request outside the step. Without one (static batching),
        # request i owns slot i.
        batch = 2
        if slot_map:
            batch_indices = torch.tensor([2, 0], dtype=torch.int64, device="cuda")
            slots, num_slots = batch_indices, 3
        else:
            batch_indices = None
            slots, num_slots = torch.arange(batch, device="cuda"), batch
        proj = 2 * mixer.d_inner_local_tp + 2 * mixer.ngroups_local_tp * mixer.d_state + H
        conv_shape, ssm_shape = mixer.mamba_state_shapes_per_request()
        zxBCdt = torch.randn(batch, seq_len, proj, generator=generator, device="cuda")
        conv_state = torch.randn(num_slots, *conv_shape, generator=generator, device="cuda")
        ssm_state = 0.1 * torch.randn(num_slots, *ssm_shape, generator=generator, device="cuda")
        # Speculative verification decodes several tokens per request and also keeps both
        # states after each token, so a rejected draft can be rolled back.
        speculative = seq_len > 1
        int_conv_state = int_ssm_state = None
        if speculative:
            int_conv_state = torch.zeros(num_slots, seq_len, *conv_shape, device="cuda")
            int_ssm_state = torch.zeros(num_slots, seq_len, *ssm_shape, device="cuda")

        # The reference decodes one token at a time on each request's slot.
        conv_state_ref, ssm_state_ref = conv_state.double(), ssm_state.double()
        y_ref, int_conv_state_ref, int_ssm_state_ref = [], [], []
        for s in range(seq_len):
            y_s, conv_s, ssm_s = self.reference_decode_step(
                mixer, zxBCdt[:, s : s + 1], conv_state_ref[slots], ssm_state_ref[slots]
            )
            conv_state_ref[slots], ssm_state_ref[slots] = conv_s, ssm_s
            y_ref.append(y_s)
            int_conv_state_ref.append(conv_s)
            int_ssm_state_ref.append(ssm_s)

        with torch.no_grad(), InferenceMode.active():
            y = mixer.ssm_decode(
                zxBCdt,
                conv_state=conv_state,
                ssm_state=ssm_state,
                batch_indices=batch_indices,
                intermediate_conv_state=int_conv_state,
                intermediate_ssm_state=int_ssm_state,
            )

        # fp32 kernels against the float64 reference. The whole state pool is compared, so a
        # write to a slot outside the step is caught too.
        tols = dict(atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(conv_state.double(), conv_state_ref, **tols)
        torch.testing.assert_close(ssm_state.double(), ssm_state_ref, **tols)
        if speculative:
            torch.testing.assert_close(
                int_conv_state[slots].double(), torch.stack(int_conv_state_ref, dim=1), **tols
            )
            torch.testing.assert_close(
                int_ssm_state[slots].double(), torch.stack(int_ssm_state_ref, dim=1), **tols
            )
        torch.testing.assert_close(y.double(), torch.cat(y_ref, dim=1), **tols)
