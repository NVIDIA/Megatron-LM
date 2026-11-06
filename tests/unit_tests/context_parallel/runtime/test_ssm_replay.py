# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact runtime CP replay for SSM projections, recurrence, gating and communication."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.tuning import policy as tuning_policy
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    bytes_equal,
    deterministic_algorithms,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

try:
    import triton  # noqa: F401

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAVE_TRITON), reason="needs a GPU and Triton"
    ),
]


@pytest.fixture(autouse=True)
def ssm_deterministic(monkeypatch):
    """Pin runtime SSM workspace selection like ``--deterministic-mode``."""
    monkeypatch.setattr(tuning_policy, "_deterministic_override", True)


@pytest.mark.parametrize("variant", ["gdn", "gdn2"])
def test_runtime_cp_gdn_module_replays(variant):
    """Exercise real projection, CP all-to-all, native recurrence and all gradients.

    The native deterministic recurrence accepts dense sequences, not packed THD.
    Runtime metadata selects CP1/2/4 independently of the build-time CP1 group.
    Packed FLA correctness is covered by the separate parallel GDN tests.
    """
    from megatron.core import parallel_state
    from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
        get_gated_delta_net_module_spec,
    )
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.ssm.gated_delta_net.common import _GDNBase
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer import TransformerConfig

    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires a world size divisible by two")
    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    try:
        model_parallel_cuda_manual_seed(123)
        seeded()
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            linear_conv_kernel_dim=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_num_key_heads=4,
            linear_num_value_heads=8,
            normalization="RMSNorm",
            experimental_attention_variant=variant,
            linear_attention_freq=[1],
            deterministic_mode=True,
            gdn_kernel_backend="torch",
            use_cpu_initialization=True,
            bf16=True,
            params_dtype=torch.bfloat16,
            activation_func=torch.nn.functional.silu,
        )
        spec = get_gated_delta_net_module_spec(config)
        module = (
            spec.module(
                config,
                spec.submodules,
                layer_number=1,
                use_qk_l2norm=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
            .cuda()
            .train()
        )
        assert isinstance(module, _GDNBase)
        build_group = module.pg_collection.cp
        reference = None
        runtime_sizes = (1, 2, 4, 1) if Utils.world_size % 4 == 0 else (1, 2, 1)
        for size in runtime_sizes:
            group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=size)
            packed = PackedSeqParams(qkv_format="sbhd", local_cp_size=size, cp_group=group)
            seeded()
            hidden = torch.randn(
                128, 2, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            with deterministic_algorithms(True):
                result = assert_module_replays_bit_exact(
                    module,
                    {"hidden_states": hidden, "attention_mask": None, "packed_seq_params": packed},
                    replays=3,
                    contention=True,
                    what=f"{variant} runtime CP{size}",
                )
            assert module.pg_collection.cp is build_group and module.cp_size == 1
            assert result[1]
            for tensors in result:
                assert all(torch.isfinite(tensor).all() for tensor in tensors.values())
            if size == 1:
                if reference is None:
                    reference = result
                for expected, actual in zip(reference, result):
                    assert expected.keys() == actual.keys()
                    assert all(bytes_equal(expected[name], actual[name]) for name in expected)
    finally:
        Utils.destroy_model_parallel()


def test_gdp_chunkwise_runtime_cp_full_head_gating_replays():
    """Replay GDP gating with two heads across CP1/4/2/1 without head partitioning."""
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer

    Utils.initialize_distributed()
    seeded()
    mixer = GatedDeltaProductMixer.__new__(GatedDeltaProductMixer)
    torch.nn.Module.__init__(mixer)
    mixer.config = SimpleNamespace(
        linear_cp_mode="chunkwise", linear_cp_layout="contiguous", gdp_cutedsl_kernel=False
    )
    mixer.pg_collection = SimpleNamespace(cp=SimpleNamespace(size=lambda: 4, rank=lambda: 0))
    mixer.cp = None
    mixer.chunkwise_cp_backend = object()
    mixer.d_inner_local_tp, mixer.nheads_local_tp, mixer.ngroups_local_tp = 32, 2, 1
    mixer.d_state, mixer.headdim, mixer.num_householder = 8, 16, 2
    mixer.conv1d = torch.nn.Conv1d(88, 88, 4, groups=88, device="cuda")
    mixer.A_log = torch.nn.Parameter(torch.randn(2, device="cuda"))
    mixer.dt_bias = torch.nn.Parameter(torch.randn(2, device="cuda"))
    mixer.D, mixer.D_has_hdim = None, False

    class Gating(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = mixer

        def forward(self, ba, packed):
            runtime = self.mixer._resolve_runtime_cp(packed)
            assert runtime.nheads == 2
            beta, g = self.mixer._compute_gating(ba, runtime_cp=runtime)
            return torch.cat((beta.flatten(), g.flatten()))

    module = Gating()
    ba = torch.randn(2, 128, 6, device="cuda", requires_grad=True)
    reference = None
    for size in (1, 4, 2, 1):
        # Gating reads CP geometry but performs no communication or sequence sharding.
        group = SimpleNamespace(size=lambda size=size: size, rank=lambda: 0)
        packed = PackedSeqParams(local_cp_size=size, cp_group=group)
        result = assert_module_replays_bit_exact(
            module, (ba, packed), replays=3, contention=True, what=f"GDP full-head gating CP{size}"
        )
        assert mixer.cp is None
        assert {"chunk0.mixer.A_log", "chunk0.mixer.dt_bias"}.issubset(result[1])
        if reference is None:
            reference = result
        for expected, actual in zip(reference, result):
            assert expected.keys() == actual.keys()
            assert all(bytes_equal(expected[name], actual[name]) for name in expected)


@pytest.mark.parametrize("kind", ["mamba", "gdp"])
def test_runtime_cp_ssm_conv_and_collectives_replay(kind):
    """Replay the runtime views consumed by Mamba/GDP: collectives and sliced conv grads.

    This targets the CP wrapper rather than the unchanged external recurrence kernel.
    Every CP size uses real groups, packed boundaries, convolution and backward.
    """
    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.ssm.gdp_context_parallel import GDPContextParallel
    from megatron.core.ssm.mamba_context_parallel import MambaContextParallel

    if Utils.world_size < 4 or Utils.world_size % 4:
        pytest.skip("requires four distributed ranks")
    Utils.initialize_model_parallel(dynamic_context_parallel=True)
    try:
        seeded()
        inner, heads, groups, state, householder = 64, 4, 4, 8, 2
        conv_channels = (
            inner + 2 * groups * state
            if kind == "mamba"
            else inner * householder + groups * state * (householder + 1)
        )
        width = inner + conv_channels + heads * (1 if kind == "mamba" else householder + 1)

        class ConvPath(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(
                    conv_channels, conv_channels, 4, groups=conv_channels, padding=3, device="cuda"
                )
                base_group = parallel_state.get_context_parallel_group()
                common = dict(
                    cp_group=base_group,
                    d_inner_local_tp=inner,
                    nheads_local_tp=heads,
                    ngroups_local_tp=groups,
                    d_state=state,
                    dt_bias_cp1=torch.zeros(heads, device="cuda"),
                    A_log_cp1=torch.zeros(heads, device="cuda"),
                    D_cp1=None,
                    D_has_hdim=False,
                )
                if kind == "mamba":
                    self.helper = MambaContextParallel(
                        **common,
                        conv1d_weight_cp1=self.conv.weight,
                        conv1d_bias_cp1=self.conv.bias,
                        conv1d_padding=3,
                    )
                else:
                    self.helper = GDPContextParallel(
                        **common,
                        conv1d_cp1=self.conv,
                        num_householder=householder,
                        headdim=inner // heads,
                    )

            def forward(self, hidden, packed):
                runtime = self.helper.for_context_parallel_group(packed.cp_group)
                projected = runtime.pre_conv_ssm(hidden, packed)
                begin = runtime.d_inner_local_tpcp
                conv_input = (
                    projected[..., begin : begin + runtime.conv1d_channels()]
                    .permute(1, 2, 0)
                    .contiguous()
                )
                conv_output = runtime.conv1d(conv_input)[..., : projected.shape[0]]
                output = conv_output[:, :begin].permute(2, 0, 1).contiguous()
                return runtime.post_conv_ssm(output, packed)

        module = ConvPath()
        bounds = torch.tensor([0, 64, 192], dtype=torch.int32, device="cuda")
        reference = None
        for size in (1, 2, 4, 1):
            group = parallel_state.get_dynamic_data_context_parallel_groups(group_size=size)
            packed = PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=bounds,
                cu_seqlens_kv=bounds,
                max_seqlen_q=128,
                max_seqlen_kv=128,
                cp_group=group,
                local_cp_size=size,
            )
            seeded()
            hidden = torch.randn(192 // size, 1, width, device="cuda", requires_grad=True)
            with deterministic_algorithms(True):
                result = assert_module_replays_bit_exact(
                    module,
                    (hidden, packed),
                    replays=3,
                    contention=True,
                    what=f"{kind} runtime CP{size} conv/collectives",
                )
            assert module.helper.cp_size == 1
            assert result[1]
            for tensors in result:
                assert all(torch.isfinite(tensor).all() for tensor in tensors.values())
            if size == 1:
                if reference is None:
                    reference = result
                for expected, actual in zip(reference, result):
                    assert expected.keys() == actual.keys()
                    assert all(bytes_equal(expected[name], actual[name]) for name in expected)
    finally:
        Utils.destroy_model_parallel()
