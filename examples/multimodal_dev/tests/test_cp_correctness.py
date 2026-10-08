# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed correctness test for Context Parallelism (CP) support.

Verifies that CP>1 produces the same (or numerically close) loss as CP=1
for the Qwen3.5-VL multimodal model by running forward passes with
deterministic data and comparing the per-rank reduced losses.

Launch with torchrun (N must be a multiple of the CP size under test; the
2*cp_size divisibility that zigzag splitting needs is a constraint on the
sequence length, not on N, and is handled by ``aligned_seq_len``):

    # As a pytest module (this is what CI runs; CP sizes that do not
    # divide the world size are skipped):
    torchrun --nproc_per_node=8 -m pytest -q \\
        examples/multimodal_dev/tests/test_cp_correctness.py

    # Test CP=2 on 2 GPUs:
    torchrun --nproc_per_node=2 examples/multimodal_dev/tests/test_cp_correctness.py --cp-size 2

    # Test CP=4 on 4 GPUs:
    torchrun --nproc_per_node=4 examples/multimodal_dev/tests/test_cp_correctness.py --cp-size 4

The test:
  1. Builds a tiny proxy model (2 layers, no MoE, no vision encoder).
  2. Generates a deterministic batch (same seed on all ranks).
  3. Runs forward with CP=1 (each rank processes the full sequence independently).
  4. Re-initialises model-parallel groups with the target CP size.
  5. Runs forward with CP=target (sequence is split across ranks).
  6. Compares the all-reduced loss values.

Exit code 0 = PASS, 1 = FAIL.
"""

import argparse
import math
import os
import sys

import pytest
import torch
import torch.distributed as dist

# Ensure the repo root is on the path so that megatron and examples are importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.unit_tests.test_utilities import Utils  # noqa: E402

# Proxy-model / data sizes and tolerances. Shared by the pytest tests below
# and by the CLI defaults in ``main`` so the two entry points cannot drift.
# ``rtol`` is deliberately tight: CP only redistributes the same tokens, so the
# reduced loss should barely move.  Measured on 8xH100 (bf16, 2-layer proxy):
# CP=2 differs from CP=1 by 2.9e-6 relative and CP=4 by 1.6e-5, so 1e-3 keeps
# ~60x headroom while still failing on a real CP regression.  Grad-level
# sensitivity for the same ``_cp_split_tensor`` helper is covered by
# ``test_cp_thd_correctness.py``, which compares grad norms as well as loss
# (its band is looser -- 2e-3 relative -- sized by its worst case, BSHD
# grad_norm at 3.4e-4; its THD grad_norm is tighter than its loss, so this is
# not a general "grads diverge more" rule. See its TOLERANCES).
DEFAULTS = dict(seq_len=128, seed=42, vocab_size=1024)

# Comparison band, kept out of DEFAULTS because ``run_cp`` never reads it -- only
# the assertion and the CLI do (mirrors ``test_cp_thd_correctness.TOLERANCES``).
TOLERANCES = dict(atol=1e-4, rtol=1e-3)

# CP sizes the pytest entry point compares against the CP=1 baseline. Sizes
# that do not divide the world size are skipped at runtime.
CP_SIZES = [2, 4]


def _parse_args():
    parser = argparse.ArgumentParser(description="CP correctness test")
    parser.add_argument(
        "--cp-size", type=int, default=2,
        help="Target context-parallel size to compare against CP=1 baseline",
    )
    parser.add_argument(
        "--seq-len", type=int, default=DEFAULTS["seq_len"],
        help="Sequence length; rounded up to a multiple of 2*cp_size if needed",
    )
    parser.add_argument(
        "--atol", type=float, default=TOLERANCES["atol"],
        help="Absolute tolerance for loss comparison",
    )
    parser.add_argument(
        "--rtol", type=float, default=TOLERANCES["rtol"],
        help="Relative tolerance for loss comparison (default %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULTS["seed"],
        help="Random seed for reproducibility",
    )
    # Megatron adds extra args; ignore them.
    args, _ = parser.parse_known_args()
    return args


def _init_distributed():
    """Initialise torch.distributed if not already done; return the local rank.

    Goes through ``Utils`` (rather than ``torch.distributed`` directly) so
    that ``Utils.inited`` stays in sync with the real state — the other
    modules in this suite tear down with ``Utils.destroy_model_parallel()``,
    which is a no-op when that flag is stale.
    """
    Utils.initialize_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def _init_megatron_parallel(tp_size=1, pp_size=1, cp_size=1, seed=DEFAULTS["seed"]):
    """(Re-)initialise Megatron model-parallel groups and RNG tracker."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
    )
    model_parallel_cuda_manual_seed(seed)


def _make_deterministic_batch(seed, batch_size, seq_len, vocab_size, device):
    """Create a deterministic batch identical on all ranks."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=rng,
    ).to(device)
    labels = torch.randint(
        0, vocab_size, (batch_size, seq_len), generator=rng,
    ).to(device)
    loss_mask = torch.ones(batch_size, seq_len, device=device)
    # Standard position_ids [B, S]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
    }


def _build_tiny_model(cp_size, device, vocab_size=DEFAULTS["vocab_size"]):
    """Build a minimal GPTModel for testing (no vision, no MoE)."""
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_config import TransformerConfig

    hidden_size = 256
    num_heads = 4
    config = TransformerConfig(
        num_layers=2,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=num_heads,
        kv_channels=hidden_size // num_heads,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=torch.nn.functional.silu,
        bf16=True,
        context_parallel_size=cp_size,
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        sequence_parallel=False,
    )

    spec = get_gpt_layer_with_transformer_engine_spec()

    model = GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab_size,
        max_sequence_length=4096,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=10000,
    )
    model = model.to(device=device, dtype=torch.bfloat16)
    return model, config


def _forward_with_cp(model, batch, cp_size):
    """Run forward pass, handling CP splitting of the batch.

    When cp_size > 1, splits the batch tensors using the same zigzag
    logic as multimodal_dev/models/base.py.
    """
    from examples.multimodal_dev.models.base import _cp_split_tensor
    from megatron.core import parallel_state as ps

    input_ids = batch["input_ids"].clone()
    labels = batch["labels"].clone()
    loss_mask = batch["loss_mask"].clone()
    position_ids = batch["position_ids"].clone()

    if cp_size > 1:
        cp_rank = ps.get_context_parallel_rank()
        input_ids = _cp_split_tensor(input_ids, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
        labels = _cp_split_tensor(labels, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
        loss_mask = _cp_split_tensor(loss_mask, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)
        # position_ids are NOT split — the RoPE layer handles CP slicing internally.

    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            attention_mask=None,
        )

    # output is the per-token loss [B, S/CP]
    masked_loss = (output.float() * loss_mask.float()).sum()
    num_tokens = loss_mask.sum()

    # All-reduce across CP ranks to get global loss
    if cp_size > 1:
        cp_group = ps.get_context_parallel_group()
        dist.all_reduce(masked_loss, group=cp_group)
        dist.all_reduce(num_tokens, group=cp_group)

    avg_loss = masked_loss / num_tokens.clamp(min=1)
    return avg_loss.item()


def cp_skip_reason(cp_size):
    """Why this world size cannot run the given CP size, or ``None``."""
    world_size = dist.get_world_size()
    if world_size < cp_size:
        return f"world_size={world_size} < cp_size={cp_size}; need at least {cp_size} GPUs"
    if world_size % cp_size != 0:
        return f"world_size={world_size} is not divisible by cp_size={cp_size}"
    return None


def aligned_seq_len(seq_len, cp_sizes):
    """Round ``seq_len`` up to a multiple of ``2*cp`` for every CP size given.

    Zigzag CP splitting needs the sequence to divide into ``2*cp_size``
    chunks, so a baseline that is to be reused across several CP sizes must
    satisfy all of them at once.  That is the LCM, not the max: ``max`` only
    happens to work when every size divides the largest one.
    """
    align = 2 * math.lcm(*cp_sizes)
    return ((seq_len + align - 1) // align) * align


# Sequence length shared by the baseline and every trial in the pytest run.
TEST_SEQ_LEN = aligned_seq_len(DEFAULTS["seq_len"], CP_SIZES)


def run_cp(
    cp_size,
    seq_len,
    seed=DEFAULTS["seed"],
    vocab_size=DEFAULTS["vocab_size"],
    state_dict=None,
):
    """Run the comparison at the given CP size.

    Owns the whole leg: initialises the model-parallel groups, builds the
    model and the batch, runs the forward pass and tears the groups back
    down. CP=1 is not a special case — the baseline and every trial go
    through here, differing only in ``cp_size``.

    ``state_dict`` carries the reference weights between calls: the first
    call passes ``None`` and returns the weights it initialised, every later
    call passes them back so all CP sizes are compared on a bitwise
    identical model. The batch needs no such threading — it is a pure
    function of ``(seed, seq_len, vocab_size)`` and independent of CP, so
    rebuilding it here yields the same tensors every time.

    Returns ``(loss, state_dict)``. Leaves the model-parallel groups
    destroyed.
    """
    local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()

    if rank == 0:
        print(f"=== CP={cp_size} (world_size={dist.get_world_size()}) ===", flush=True)

    _init_megatron_parallel(cp_size=cp_size, seed=seed)

    # Set deterministic seed for model init
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, _ = _build_tiny_model(cp_size=cp_size, device=device, vocab_size=vocab_size)
    if state_dict is None:
        state_dict = model.state_dict()
    else:
        model.load_state_dict(state_dict, strict=True)

    batch = _make_deterministic_batch(
        seed=seed + 1, batch_size=1, seq_len=seq_len,
        vocab_size=vocab_size, device=device,
    )

    loss = _forward_with_cp(model, batch, cp_size=cp_size)

    if rank == 0:
        print(f"  CP={cp_size} loss: {loss:.6f}", flush=True)

    del model
    Utils.destroy_model_parallel()
    torch.cuda.empty_cache()

    return loss, state_dict


# ===================================================================
# pytest entry points
# ===================================================================


@pytest.fixture(scope="module")
def cp1_baseline():
    """Memoise the CP=1 run so every CP size under test shares one baseline."""
    return run_cp(1, TEST_SEQ_LEN)


@pytest.mark.parametrize("cp_size", CP_SIZES)
def test_cp_matches_cp1_baseline(request, cp_size):
    """CP>1 must reproduce the CP=1 loss on identical weights and data."""
    _init_distributed()
    reason = cp_skip_reason(cp_size)
    if reason is not None:
        pytest.skip(reason)

    # Requested lazily so an all-skipped world size never pays for the baseline.
    loss_cp1, state_dict = request.getfixturevalue("cp1_baseline")
    loss_cpN, _ = run_cp(cp_size, TEST_SEQ_LEN, state_dict=state_dict)

    atol, rtol = TOLERANCES["atol"], TOLERANCES["rtol"]
    diff = abs(loss_cpN - loss_cp1)
    assert diff <= atol + rtol * abs(loss_cp1), (
        f"CP={cp_size} loss {loss_cpN:.6f} differs from CP=1 loss {loss_cp1:.6f} "
        f"(abs diff {diff:.3e}, atol={atol}, rtol={rtol})"
    )


def main():
    args = _parse_args()
    _init_distributed()
    rank = dist.get_rank()

    target_cp = args.cp_size
    reason = cp_skip_reason(target_cp)
    if reason is not None:
        if rank == 0:
            print(f"SKIP: {reason}.", flush=True)
        dist.destroy_process_group()
        sys.exit(0)

    seq_len = aligned_seq_len(args.seq_len, [target_cp])
    if seq_len != args.seq_len and rank == 0:
        print(f"Adjusted seq_len to {seq_len} for alignment with CP={target_cp}", flush=True)

    loss_cp1, state_dict = run_cp(1, seq_len, seed=args.seed)
    loss_cpN, _ = run_cp(target_cp, seq_len, seed=args.seed, state_dict=state_dict)

    # --- Step 3: Compare ---
    if rank == 0:
        diff = abs(loss_cpN - loss_cp1)
        rel_diff = diff / max(abs(loss_cp1), 1e-10)

        print(f"\n=== Comparison ===", flush=True)
        print(f"  CP=1 loss:         {loss_cp1:.6f}", flush=True)
        print(f"  CP={target_cp} loss:         {loss_cpN:.6f}", flush=True)
        print(f"  Absolute diff:     {diff:.6e}", flush=True)
        print(f"  Relative diff:     {rel_diff:.6e}", flush=True)
        print(f"  Tolerance (atol):  {args.atol:.6e}", flush=True)
        print(f"  Tolerance (rtol):  {args.rtol:.6e}", flush=True)

        passed = diff <= args.atol + args.rtol * abs(loss_cp1)
        if passed:
            print(f"\nPASS: CP={target_cp} matches CP=1 baseline", flush=True)
        else:
            print(f"\nFAIL: CP={target_cp} loss differs from CP=1 beyond tolerance", flush=True)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0 and not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
