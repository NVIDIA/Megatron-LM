# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""
Equivalence tests: GPTModel with DSA vs MambaModel with DSA pattern.

A small DeepSeek-V3.2 proxy model (4 GPT layers / 8 Mamba layers) is built,
weights are remapped GPT→Mamba, and logprobs are compared to verify they are
numerically identical.

Architecture equivalence
------------------------
GPTModel layer N  (combined attention + MLP in one TransformerLayer)
  ≡  MambaModel layer 2N  (S, DSA TransformerLayer: input_layernorm + MLASelfAttention)
   + MambaModel layer 2N+1 (-, MLPLayer: fused-norm MLP)

Run with::

    torchrun --nproc-per-node=2 -m pytest \\
        tests/unit_tests/models/test_dsa_gpt_mamba_equivalence.py -v
"""

import copy
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import megatron.core.parallel_state as mpu
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.rl.rl_utils import selective_log_softmax
from tests.unit_tests.test_utilities import Utils

try:
    from fast_hadamard_transform import hadamard_transform as _hadamard_transform

    HAVE_HADAMARD = True
except ImportError:
    HAVE_HADAMARD = False


# ---------------------------------------------------------------------------
# Hadamard mock (used when the library is not installed)
# ---------------------------------------------------------------------------


def _mock_hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Identity-scale mock for hadamard_transform used in DSA."""
    return x * scale


@pytest.fixture(autouse=True)
def _patch_hadamard_if_needed():
    """Patch hadamard_transform in the DSA module when the library is absent."""
    if not HAVE_HADAMARD:
        with patch(
            'megatron.core.transformer.experimental_attention_variant.dsa.hadamard_transform',
            _mock_hadamard_transform,
        ):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Proxy model constants
# ---------------------------------------------------------------------------

_VOCAB_SIZE = 256
_MAX_SEQ_LEN = 64
_SEQ_LEN = 32
_BATCH_SIZE = 2
_NUM_GPT_LAYERS = 4
_MAMBA_PATTERN = "S-S-S-S-"  # len=8 = 2 * _NUM_GPT_LAYERS


# ---------------------------------------------------------------------------
# Model construction helpers
# ---------------------------------------------------------------------------


def _make_dsa_config(num_layers: int, tp: int = 1) -> MLATransformerConfig:
    """Return a small DeepSeek-V3.2 proxy MLATransformerConfig."""
    return MLATransformerConfig(
        num_layers=num_layers,
        hidden_size=256,
        num_attention_heads=16,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_head_dim=64,
        qk_pos_emb_head_dim=32,
        v_head_dim=64,
        dsa_indexer_n_heads=8,
        dsa_indexer_head_dim=64,
        dsa_indexer_topk=32,
        normalization="RMSNorm",
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        use_cpu_initialization=True,
        rope_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        experimental_attention_variant="dsa",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=tp,
    )


def _make_pg_collection() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'pp', 'cp'])


def _build_gpt_model(
    config: MLATransformerConfig,
    pre_process: bool = True,
    post_process: bool = True,
) -> GPTModel:
    """Build a GPTModel with the DSA transformer block spec."""
    spec = get_transformer_block_with_experimental_attention_variant_spec(config)
    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_MAX_SEQ_LEN,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=False,  # Gather logits across TP for easy comparison
        position_embedding_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        pg_collection=_make_pg_collection(),
    )
    return model.cuda()


def _build_mamba_model(
    config: MLATransformerConfig,
    pattern: str,
    pre_process: bool = True,
    post_process: bool = True,
) -> MambaModel:
    """Build a MambaModel with the given hybrid pattern."""
    mamba_config = copy.deepcopy(config)
    mamba_config.num_layers = len(pattern)
    model = MambaModel(
        config=mamba_config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_MAX_SEQ_LEN,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=False,
        hybrid_override_pattern=pattern,
        position_embedding_type='rope',
        rotary_base=10000,
        rotary_percent=1.0,
        pg_collection=_make_pg_collection(),
    )
    return model.cuda()


# ---------------------------------------------------------------------------
# Weight remapping
# ---------------------------------------------------------------------------


def _remap_gpt_to_mamba_state_dict(
    gpt_sd: Dict[str, torch.Tensor],
    num_local_gpt_layers: int,
) -> Dict[str, torch.Tensor]:
    """Remap a GPTModel state_dict to a MambaModel state_dict.

    GPTModel layer N (combined attention + MLP) maps to:
      * MambaModel layer 2N   – DSA attention (input_layernorm + self_attention)
      * MambaModel layer 2N+1 – MLP           (mlp.*)

    Additionally, ``decoder.final_layernorm.*`` (TransformerBlock naming) is
    remapped to ``decoder.final_norm.*`` (MambaStack naming).

    All other keys (embedding, output_layer, rotary_pos_emb, …) are unchanged.

    Args:
        gpt_sd: State dict obtained from GPTModel.state_dict().
        num_local_gpt_layers: Number of GPT decoder layers on the current
            pipeline stage (i.e. ``len(gpt_model.decoder.layers)``).

    Returns:
        Remapped state dict ready for MambaModel.load_state_dict(strict=True).
    """
    mamba_sd: Dict[str, torch.Tensor] = {}
    layer_prefix = "decoder.layers."
    final_ln_prefix = "decoder.final_layernorm."

    for key, value in gpt_sd.items():
        # ---- final layernorm rename ----
        if key.startswith(final_ln_prefix):
            suffix = key[len(final_ln_prefix):]
            mamba_sd[f"decoder.final_norm.{suffix}"] = value
            continue

        # ---- non-layer keys pass through unchanged ----
        if not key.startswith(layer_prefix):
            mamba_sd[key] = value
            continue

        # ---- parse "decoder.layers.{N}.{rest}" ----
        remainder = key[len(layer_prefix):]
        dot_idx = remainder.index('.')
        layer_n = int(remainder[:dot_idx])
        rest = remainder[dot_idx + 1:]  # e.g. "self_attention.linear_q_proj.weight"

        assert 0 <= layer_n < num_local_gpt_layers, (
            f"Layer index {layer_n} out of range [0, {num_local_gpt_layers}) in key '{key}'"
        )

        if rest.startswith("input_layernorm.") or rest.startswith("self_attention."):
            # Attention sub-module → DSA layer 2N
            mamba_sd[f"{layer_prefix}{2 * layer_n}.{rest}"] = value
        elif rest.startswith("mlp."):
            # MLP sub-module → MLP layer 2N+1
            mamba_sd[f"{layer_prefix}{2 * layer_n + 1}.{rest}"] = value
        else:
            # pre_mlp_layernorm is IdentityOp (no weights); self_attn_bda / mlp_bda
            # are callables (no weights). Anything else is unexpected.
            raise ValueError(
                f"Unexpected sub-key '{rest}' in GPT layer {layer_n} (full key='{key}'). "
                "Expected: input_layernorm.*, self_attention.*, mlp.*"
            )

    return mamba_sd


# ---------------------------------------------------------------------------
# Forward-pass helpers
# ---------------------------------------------------------------------------


def _make_inputs(tokens: torch.Tensor):
    """Return position_ids and attention_mask for a token batch."""
    batch_size, seq_len = tokens.shape
    position_ids = (
        torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, seq_len)
    )
    attention_mask = torch.ones(
        batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=tokens.device
    )
    return position_ids, attention_mask


def _forward_logprobs_pp1(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    """Single-stage (PP=1) forward returning logprobs [batch, seq-1]."""
    position_ids, attention_mask = _make_inputs(tokens)
    with torch.no_grad():
        logits = model(
            input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask
        )
    return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])


def _forward_logprobs_pp2(
    model: torch.nn.Module, tokens: torch.Tensor
) -> Optional[torch.Tensor]:
    """Two-stage (PP=2) forward using point-to-point communication.

    Returns logprobs on the last PP stage; None on the first stage.
    The caller must invoke this function for both GPT and Mamba models in the
    *same order* on all ranks to avoid deadlocks.
    """
    batch_size, seq_len = tokens.shape
    hidden_size = model.config.hidden_size
    position_ids, attention_mask = _make_inputs(tokens)

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    next_rank = mpu.get_pipeline_model_parallel_next_rank()
    prev_rank = mpu.get_pipeline_model_parallel_prev_rank()

    if pp_rank == 0:
        # First stage: embedding + local layers → hidden states [seq, batch, hidden]
        with torch.no_grad():
            hidden = model(
                input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask
            )
        dist.send(hidden.contiguous(), dst=next_rank)
        return None
    else:
        # Last stage: receive hidden states, run remaining layers → logits
        hidden_buf = torch.empty(
            seq_len, batch_size, hidden_size,
            dtype=torch.bfloat16, device=tokens.device,
        )
        dist.recv(hidden_buf, src=prev_rank)
        model.set_input_tensor(hidden_buf)
        with torch.no_grad():
            logits = model(
                input_ids=tokens, position_ids=position_ids, attention_mask=attention_mask
            )
        return selective_log_softmax(logits[:, :-1, :], tokens[:, 1:])


# ---------------------------------------------------------------------------
# Golden-value recording / comparison helpers
# ---------------------------------------------------------------------------


def _save_golden_values(logprobs: torch.Tensor, path: Path) -> None:
    """Save GPTModel logprobs to JSON in the functional test format.

    Format::

        {"0": {"logprobs": [...], "generated_tokens": []}}

    Args:
        logprobs: Tensor of shape [batch, seq-1] on CUDA.
        path: JSON output path (parent directory must exist).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lp_list = logprobs[0].float().tolist()  # first batch item
    data = {"0": {"logprobs": lp_list, "generated_tokens": []}}
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def _compare_against_golden_values(logprobs: torch.Tensor, path: Path, abs_tol: float = 1e-3):
    """Assert logprobs match the golden values JSON within *abs_tol*."""
    with open(path) as fh:
        golden = json.load(fh)
    golden_lp = golden["0"]["logprobs"]
    actual_lp = logprobs[0].float().tolist()
    assert len(actual_lp) == len(golden_lp), (
        f"Logprob length mismatch: actual={len(actual_lp)}, golden={len(golden_lp)}"
    )
    for i, (a, g) in enumerate(zip(actual_lp, golden_lp)):
        assert math.isclose(a, g, abs_tol=abs_tol), (
            f"Logprob mismatch at position {i}: actual={a:.6f}, golden={g:.6f}"
        )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

_GOLDEN_BASE = Path(__file__).parent.parent.parent / (
    "functional_tests/test_cases/hybrid"
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (1, 2)])
class TestDSAGPTMambaEquivalence:
    """Verify logprob equivalence between GPTModel+DSA and MambaModel+DSA.

    For each distributed configuration (TP, PP), the test:
    1. Builds a GPTModel with 4 DSA layers.
    2. Builds a MambaModel with pattern "S-S-S-S-" (8 layers).
    3. Remaps and loads GPT weights into MambaModel (strict=True).
    4. Runs the same random tokens through both models.
    5. Asserts logprob tensors are numerically close.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _skip_if_insufficient_gpus(self, tp: int, pp: int) -> None:
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        required = tp * pp
        if world_size < required:
            pytest.skip(
                f"Test tp={tp} pp={pp} requires {required} GPU(s), "
                f"but WORLD_SIZE={world_size}"
            )

    def test_dsa_logprobs_match(self, tp: int, pp: int) -> None:
        """Build both models, transfer weights, compare logprobs."""
        self._skip_if_insufficient_gpus(tp, pp)
        Utils.initialize_model_parallel(tp, pp)
        model_parallel_cuda_manual_seed(42)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()

        # ---- Build GPTModel ----
        gpt_config = _make_dsa_config(num_layers=_NUM_GPT_LAYERS, tp=tp)
        gpt_model = _build_gpt_model(gpt_config, pre_process=pre_process, post_process=post_process)
        num_local_gpt_layers = len(gpt_model.decoder.layers)
        gpt_sd = gpt_model.state_dict()

        # ---- Build MambaModel ----
        mamba_model = _build_mamba_model(
            gpt_config, _MAMBA_PATTERN, pre_process=pre_process, post_process=post_process
        )

        # ---- Remap GPT weights → Mamba ----
        mamba_sd = _remap_gpt_to_mamba_state_dict(gpt_sd, num_local_gpt_layers)
        missing, unexpected = mamba_model.load_state_dict(mamba_sd, strict=True)
        assert not missing, f"Missing keys after weight remap: {missing}"
        assert not unexpected, f"Unexpected keys after weight remap: {unexpected}"

        # ---- Create identical inputs on all ranks ----
        torch.manual_seed(99)
        tokens = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, _SEQ_LEN), device='cuda')

        # ---- Forward pass ----
        if pp == 1:
            gpt_logprobs = _forward_logprobs_pp1(gpt_model, tokens)
            mamba_logprobs = _forward_logprobs_pp1(mamba_model, tokens)
            # Both models have full logits; compare on all TP ranks
            torch.testing.assert_close(
                gpt_logprobs, mamba_logprobs, atol=1e-5, rtol=1e-5,
                msg=f"Logprob mismatch for tp={tp} pp={pp}",
            )
        else:
            # PP=2: manual pipeline communication
            # Run GPT then Mamba in the same order on all ranks to avoid deadlocks.
            gpt_logprobs = _forward_logprobs_pp2(gpt_model, tokens)
            mamba_logprobs = _forward_logprobs_pp2(mamba_model, tokens)
            if post_process:
                torch.testing.assert_close(
                    gpt_logprobs, mamba_logprobs, atol=1e-5, rtol=1e-5,
                    msg=f"Logprob mismatch for tp={tp} pp={pp}",
                )

    def test_weight_loading_strict(self, tp: int, pp: int) -> None:
        """Verify that strict=True weight loading succeeds (no missing/unexpected keys)."""
        self._skip_if_insufficient_gpus(tp, pp)
        Utils.initialize_model_parallel(tp, pp)
        model_parallel_cuda_manual_seed(42)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()

        gpt_config = _make_dsa_config(num_layers=_NUM_GPT_LAYERS, tp=tp)
        gpt_model = _build_gpt_model(gpt_config, pre_process=pre_process, post_process=post_process)
        mamba_model = _build_mamba_model(
            gpt_config, _MAMBA_PATTERN, pre_process=pre_process, post_process=post_process
        )

        gpt_sd = gpt_model.state_dict()
        num_local_gpt_layers = len(gpt_model.decoder.layers)
        mamba_sd = _remap_gpt_to_mamba_state_dict(gpt_sd, num_local_gpt_layers)
        missing, unexpected = mamba_model.load_state_dict(mamba_sd, strict=True)

        assert not missing, f"Missing keys: {missing}"
        assert not unexpected, f"Unexpected keys: {unexpected}"

    def test_record_and_compare_golden_values(self, tp: int, pp: int) -> None:
        """Record GPTModel logprobs as golden values, then compare MambaModel against them.

        Golden values are written to the functional test directory so they can be
        committed and used by the CI inference golden-value tests (Part 2 of the plan).
        """
        self._skip_if_insufficient_gpus(tp, pp)
        # Only run for TP=1, PP=1 (the canonical golden-value configuration)
        if tp != 1 or pp != 1:
            pytest.skip("Golden-value recording only runs for tp=1, pp=1")

        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(42)

        gpt_config = _make_dsa_config(num_layers=_NUM_GPT_LAYERS, tp=1)
        gpt_model = _build_gpt_model(gpt_config)
        mamba_model = _build_mamba_model(gpt_config, _MAMBA_PATTERN)

        gpt_sd = gpt_model.state_dict()
        mamba_sd = _remap_gpt_to_mamba_state_dict(gpt_sd, len(gpt_model.decoder.layers))
        mamba_model.load_state_dict(mamba_sd, strict=True)

        torch.manual_seed(99)
        tokens = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, _SEQ_LEN), device='cuda')

        gpt_logprobs = _forward_logprobs_pp1(gpt_model, tokens)
        mamba_logprobs = _forward_logprobs_pp1(mamba_model, tokens)

        # Save golden values recorded from GPTModel
        golden_dir = _GOLDEN_BASE / "hybrid_dsa_mamba_logitsmatch_tp1_pp1"
        golden_path = golden_dir / "golden_values_dev_dgx_h100.json"
        _save_golden_values(gpt_logprobs, golden_path)

        # Verify MambaModel matches golden values
        _compare_against_golden_values(mamba_logprobs, golden_path, abs_tol=1e-3)
