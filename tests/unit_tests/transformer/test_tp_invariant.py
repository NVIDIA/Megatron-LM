# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""TP-invariant numerics: TP=1 ≡ TP=2 bitwise on a small TransformerBlock.

Validates the TP-invariant patch stack (TE column/row parallel linear with
all-gather, LayerNormColumnParallelLinear, RowParallelLinear, etc.) produces
forward + backward outputs that are bitwise-identical across tensor-parallel
degrees, given the canonical deterministic config: CPU init, deterministic
mode, unfused attention, and the patched MLM cross_entropy / clip_grads /
batch_invariant_kernels.

Requires:
    - ≥2 GPUs (TP=2/4 cases skipped if not enough)
    - TE TP-invariant patches applied (test skipped otherwise;
      see examples/tp-numerics/README.md "Install TE patches")

All required env vars are set by the test itself — no shell setup needed.

Run:
    torchrun --nproc_per_node=4 -m pytest \\
        tests/unit_tests/transformer/test_tp_invariant.py -v -s

Pass criterion: torch.equal(out_tp1, out_tpN) and torch.equal(grad_tp1, grad_tpN)
for N in {2, 4}, dtype in {fp32, bf16}.
"""
import inspect
import os

# Required env vars for TP-invariant numerics; set BEFORE any TE/Megatron import
# so the values take effect at module init. Self-contained: no shell-level setup
# required to run this test.
os.environ.setdefault("NVTE_TP_INVARIANT_MODE", "1")
os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
# Force unfused attention path. MLM's `attention_backend=AttnBackend.unfused`
# config field does NOT propagate to TE — TE picks backend from these env vars
# (Thread A finding: FA2 on SM_100 leaks 2^-5 in backward; unfused is bitwise).
os.environ.setdefault("NVTE_FLASH_ATTN", "0")
os.environ.setdefault("NVTE_FUSED_ATTN", "0")

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    enable_batch_invariant_mode,
)
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils

# Enable BIK at module-load time so all GEMMs use batch-invariant Triton kernels.
# Required for bitwise TP-invariance: without BIK, GEMMs at different M-sizes
# (full vs sharded) produce ULP-level differences that compound through bwd.
enable_batch_invariant_mode()


# --- Skip-conditions ---------------------------------------------------------

def _te_tp_invariant_patches_applied() -> bool:
    """Return True iff the TE TP-invariant patches have been cp'd into the
    transformer_engine site-packages copy."""
    try:
        import transformer_engine.pytorch.module.linear as _te_linear
        return "NVTE_TP_INVARIANT_MODE" in inspect.getsource(_te_linear)
    except Exception:
        return False


_HAVE_TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
_TE_PATCHED = _te_tp_invariant_patches_applied()


# --- Test config -------------------------------------------------------------

SEQ_LEN = 64
BATCH = 2
HIDDEN = 128
NUM_HEADS = 4
NUM_LAYERS = 1
SEED = 42


def _build_config(tp_size: int, dtype: torch.dtype) -> TransformerConfig:
    bf16 = dtype == torch.bfloat16
    fp16 = dtype == torch.float16
    return TransformerConfig(
        # arch
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=HIDDEN * 4,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_HEADS,  # plain MHA (no GQA); SelfAttention requires this set
        kv_channels=HIDDEN // NUM_HEADS,  # SelfAttention requires this set; default None
        normalization="RMSNorm",
        attention_backend=AttnBackend.unfused,
        # parallelism
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,  # SP=False keeps input replicated; simpler comparison
        # determinism
        use_cpu_initialization=True,
        deterministic_mode=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        # precision
        params_dtype=dtype,
        bf16=bf16,
        fp16=fp16,
    )


def _run_block(
    tp_size: int,
    dtype: torch.dtype,
    input_full: torch.Tensor,
    grad_full: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build a TransformerBlock at the given TP size, run forward + backward
    with the supplied input and grad, return (output, input.grad) gathered to
    rank 0 (None on other ranks).
    """
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
    )
    # Reset both global torch RNG and Megatron MP-cuda seed so weight init is
    # deterministic and identical across TP=1 and TP=2 calls (between the two
    # calls, the prior call's forward/backward has advanced global torch RNG).
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model_parallel_cuda_manual_seed(SEED)

    config = _build_config(tp_size=tp_size, dtype=dtype)
    spec = get_gpt_layer_with_transformer_engine_spec()
    block = TransformerBlock(config=config, spec=spec).cuda()
    block.train()

    inp = input_full.clone().detach().to(dtype=dtype, device="cuda").requires_grad_(True)
    grad = grad_full.clone().detach().to(dtype=dtype, device="cuda")

    out = block(hidden_states=inp, attention_mask=None)
    # TransformerBlock.forward returns hidden_states (Tensor) or (Tensor, None)
    # depending on whether context is used. Normalize.
    if isinstance(out, tuple):
        out = out[0]
    out.backward(grad)

    rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
    if rank0:
        out_cpu = out.detach().to(device="cpu", dtype=torch.float32)
        grad_cpu = inp.grad.detach().to(device="cpu", dtype=torch.float32)
        return out_cpu, grad_cpu
    return None, None


# --- Test --------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_TWO_GPUS, reason="≥2 GPUs required for TP>1 path")
@pytest.mark.skipif(
    not _TE_PATCHED,
    reason=(
        "TE TP-invariant patches not applied. See "
        "examples/tp-numerics/README.md 'Install TE patches' "
        "to apply patches from the jinzex/tp-invariant-numerics TE branch."
    ),
)
class TestTPInvariantTransformerBlock:
    """Bitwise-identical fwd+bwd across TP=1 and TP=N (N in {2, 4})."""

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    @pytest.mark.parametrize("compare_tp", [2, 4])
    def test_tp1_equals_tpN_bitwise(self, dtype, compare_tp):
        if torch.cuda.device_count() < compare_tp:
            pytest.skip(f"{compare_tp} GPUs required for TP={compare_tp}; have {torch.cuda.device_count()}")
        # Also skip if WORLD_SIZE doesn't match what we need for TP=compare_tp.
        # `Utils.initialize_model_parallel` requires world_size to be a multiple of TP.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size < compare_tp:
            pytest.skip(f"WORLD_SIZE={world_size} < compare_tp={compare_tp}; launch torchrun --nproc_per_node={compare_tp}")

        # Deterministic input on CPU (architecture-independent), replicated across ranks
        gen = torch.Generator(device="cpu").manual_seed(SEED)
        input_full = torch.randn(SEQ_LEN, BATCH, HIDDEN, generator=gen, dtype=torch.float32)
        gen2 = torch.Generator(device="cpu").manual_seed(SEED + 1)
        grad_full = torch.randn(SEQ_LEN, BATCH, HIDDEN, generator=gen2, dtype=torch.float32)

        out_tp1, grad_tp1 = _run_block(
            tp_size=1, dtype=dtype, input_full=input_full, grad_full=grad_full
        )
        Utils.destroy_model_parallel()
        out_tpN, grad_tpN = _run_block(
            tp_size=compare_tp, dtype=dtype, input_full=input_full, grad_full=grad_full
        )

        if torch.distributed.get_rank() == 0:
            assert torch.equal(out_tp1, out_tpN), (
                f"[{dtype}, TP=1 vs TP={compare_tp}] forward output differs: "
                f"max_abs={(out_tp1 - out_tpN).abs().max().item():.3e}, "
                f"mean_abs={(out_tp1 - out_tpN).abs().mean().item():.3e}"
            )
            assert torch.equal(grad_tp1, grad_tpN), (
                f"[{dtype}, TP=1 vs TP={compare_tp}] input grad differs: "
                f"max_abs={(grad_tp1 - grad_tpN).abs().max().item():.3e}, "
                f"mean_abs={(grad_tp1 - grad_tpN).abs().mean().item():.3e}"
            )
