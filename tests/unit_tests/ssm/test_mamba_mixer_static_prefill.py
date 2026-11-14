# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.inference.utils import InferenceMode
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

HIDDEN = 256


def _mixer(nheads, ngroups, use_mem_eff_path):
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        hidden_size=HIDDEN,
        num_layers=1,
        num_attention_heads=1,
        use_cpu_initialization=True,
        use_mamba_mem_eff_path=use_mem_eff_path,
        params_dtype=torch.float32,
        mamba_num_heads=nheads,
        mamba_num_groups=ngroups,
        mamba_head_dim=64,
        mamba_state_dim=64,
    )
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    mixer = MambaMixer(
        config,
        hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
        HIDDEN,
        layer_number=1,
        pg_collection=pg_collection,
    )
    return mixer.cuda().eval()


def _in_proj_width(mixer):
    return 2 * mixer.d_inner + 2 * mixer.ngroups * mixer.d_state + mixer.nheads


def _run(mixer, hidden, entry):
    """Enter `_static_prefill` as a context-free inference call or as the training fallback."""
    with torch.no_grad():
        if entry == "inference":
            with InferenceMode.active():
                return mixer(hidden)[0]
        return mixer(hidden)[0]


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
class TestMambaStaticPrefillConvLayout:
    """`_static_prefill` must hand causal_conv1d a channels-first copy of the conv input.

    It serves static-batching prefill, the context-free MTP draft step of a recurrent head, and
    the `use_mamba_mem_eff_path=False` training path. With one token at batch size 1, a plain
    `contiguous()` kept a view whose token stride is the in_proj width, which causal_conv1d took
    for its channel-last layout and rejected unless that width is a multiple of 8.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("entry", ["inference", "training_fallback"])
    @pytest.mark.parametrize(
        ("nheads", "ngroups", "tokens"),
        [(8, 4, 1), (2, 2, 2), (2, 2, 1)],
        ids=["aligned-1-token", "misaligned-2-tokens", "misaligned-1-token"],
    )
    def test_prefix_matches_longer_run(self, nheads, ngroups, tokens, entry):
        mixer = _mixer(nheads, ngroups, use_mem_eff_path=entry == "inference")
        # 2*512 + 2*4*64 + 8 = 1544 is a multiple of 8; 2*128 + 2*2*64 + 2 = 514 is not.
        assert (_in_proj_width(mixer) % 8 == 0) == (nheads == 8)
        torch.manual_seed(7)
        hidden = torch.randn(4, 1, HIDDEN, device="cuda")
        full = _run(mixer, hidden, entry)
        part = _run(mixer, hidden[:tokens].clone(), entry)
        # The scan's dots may run in TF32; a wrong result is off by O(1).
        scale = float(full.abs().max())
        torch.testing.assert_close(part, full[:tokens], rtol=2e-2, atol=2e-2 * scale)
