# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for pipeline parallelism (PP) support in multimodal_dev.

Covers the PP plumbing added to :class:`MultimodalModel`:

  1. The vision encoder is built on the first PP stage only.
  2. ``pre_process`` / ``post_process`` / ``vp_stage`` reach the wrapped
     ``GPTModel``, and ``shared_embedding_or_output_weight`` is surfaced
     on the outer module for ``finalize_model_grads``.
  3. A real two-stage forward + backward: stage 0 embeds text, runs the
     vision encoder and scatters image embeddings; stage 1 consumes the
     activation via ``set_input_tensor`` and computes the loss.  The
     activation gradient is sent back so stage 0's backward runs too,
     which is what proves the vision encoder is actually in the graph.

These need exactly 2 ranks (PP=2).  Run with::

    torchrun --nproc-per-node 2 -m pytest -q \\
        examples/multimodal_dev/tests/test_pp_support.py
"""

import os
import re
import sys

import pytest
import torch
import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.models.base import MultimodalModel
from megatron.core import parallel_state as ps
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

PP_SIZE = 2
NUM_LAYERS = 2
HIDDEN = 128
HEADS = 4
VOCAB = 128
SEQ = 32
BATCH = 2
IMAGE_TOKEN_ID = 7
# Image-token positions per sample; total visual tokens = BATCH * len(...).
IMAGE_POSITIONS = (0, 1, 2, 3)
NUM_VISUAL_TOKENS = BATCH * len(IMAGE_POSITIONS)


class _CountingVisionEncoder(MegatronModule):
    """Minimal trainable stand-in for the real vision encoder.

    Projects ``pixel_values`` ``[num_visual_tokens, H]`` to embeddings of
    the same shape and records how many times it ran, so a test can
    assert it never executes off the first PP stage.
    """

    def __init__(self, config, hidden_size, dtype=torch.bfloat16):
        """Build the stub with a single square projection."""
        super().__init__(config=config)
        # Must match params_dtype: the surrounding TE layers and the
        # incoming pixel_values share it, and .cuda() does not cast.
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, dtype=dtype)
        self.calls = 0

    def forward(self, pixel_values, image_grid_thw):
        """Project pixel features and count the invocation."""
        self.calls += 1
        return self.proj(pixel_values)


def _make_config(pp_size=PP_SIZE, dtype=torch.bfloat16):
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        ffn_hidden_size=4 * HIDDEN,
        num_attention_heads=HEADS,
        num_query_groups=HEADS,
        bf16=(dtype is torch.bfloat16),
        params_dtype=dtype,
        pipeline_dtype=dtype,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=pp_size,
        sequence_parallel=False,
    )


def _build_model(pre_process, post_process, vp_stage=None, pp_size=PP_SIZE, dtype=torch.bfloat16):
    """Build a PP-stage-aware ``MultimodalModel`` on the current rank."""
    config = _make_config(pp_size, dtype)
    # model_parallel_cuda_manual_seed only seeds the CUDA RNG tracker, but
    # the stub's nn.Linear initialises on CPU from the global RNG.  Without
    # this every rank would build a different vision encoder.
    torch.manual_seed(1234)
    vision = _CountingVisionEncoder(config, HIDDEN, dtype)
    model = MultimodalModel(
        language_config=config,
        language_spec=get_gpt_layer_with_transformer_engine_spec(),
        vision_encoder=vision,
        vocab_size=VOCAB,
        max_sequence_length=SEQ,
        image_token_id=IMAGE_TOKEN_ID,
        position_embedding_type="rope",
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
    return model.cuda()


def _make_batch(seed=1234, dtype=torch.bfloat16):
    """Deterministic batch, identical on every rank."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g, device="cuda")
    # Keep image tokens exactly where we want them.
    input_ids[input_ids == IMAGE_TOKEN_ID] = (IMAGE_TOKEN_ID + 1) % VOCAB
    for pos in IMAGE_POSITIONS:
        input_ids[:, pos] = IMAGE_TOKEN_ID
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=g, device="cuda")
    loss_mask = torch.ones(BATCH, SEQ, device="cuda")
    position_ids = torch.arange(SEQ, device="cuda").unsqueeze(0).expand(BATCH, -1).contiguous()
    pixel_values = torch.randn(NUM_VISUAL_TOKENS, HIDDEN, generator=g, device="cuda", dtype=dtype)
    image_grid_thw = torch.tensor([[1, 2, 2]] * BATCH, device="cuda")
    return input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw


@pytest.fixture(scope="module", autouse=True)
def _init_pp():
    """Initialise PP=2 model-parallel groups for the whole module."""
    # Guard before initialising: Utils.initialize_model_parallel raises on a
    # world size that is not a multiple of PP_SIZE, and this fixture is
    # autouse, so checking any later would turn the intended skip into a
    # collection-time error under the documented 1-rank invocation.
    if Utils.world_size != PP_SIZE:
        pytest.skip(f"needs exactly {PP_SIZE} ranks (PP={PP_SIZE})")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=PP_SIZE
    )
    model_parallel_cuda_manual_seed(1234)
    yield
    Utils.destroy_model_parallel()


@pytest.fixture(scope="module")
def stage_flags():
    """``(is_first, is_last)`` for this rank's PP stage."""
    return ps.is_pipeline_first_stage(), ps.is_pipeline_last_stage()


def test_vision_encoder_only_on_first_stage(stage_flags):
    """The vision encoder is retained on stage 0 and dropped elsewhere."""
    is_first, is_last = stage_flags
    model = _build_model(pre_process=is_first, post_process=is_last)

    if is_first:
        assert model.vision_model is not None, "stage 0 must own the vision encoder"
        assert any(p.requires_grad for p in model.vision_model.parameters())
    else:
        assert model.vision_model is None, "non-first stages must not hold a vision encoder"

    # No rank should carry vision params it does not own.
    vision_param_names = [n for n, _ in model.named_parameters() if n.startswith("vision_model")]
    assert bool(vision_param_names) == is_first


def test_pp_flags_reach_language_model(stage_flags):
    """``pre_process`` / ``post_process`` / ``vp_stage`` are threaded through."""
    is_first, is_last = stage_flags
    model = _build_model(pre_process=is_first, post_process=is_last, vp_stage=0)

    assert model.pre_process is is_first
    assert model.post_process is is_last
    assert model.vp_stage == 0

    assert model.language_model.pre_process is is_first
    assert model.language_model.post_process is is_last
    assert model.language_model.vp_stage == 0

    # Only the first stage builds an embedding; only the last an output layer.
    assert hasattr(model.language_model, "embedding") is is_first
    assert hasattr(model.language_model, "output_layer") is is_last

    # finalize_model_grads inspects the OUTER module for these two.
    assert hasattr(model, "share_embeddings_and_output_weights")
    assert callable(model.shared_embedding_or_output_weight)


def test_pp_two_stage_forward_backward(stage_flags):
    """A real PP=2 forward + backward produces finite loss and grads."""
    is_first, is_last = stage_flags
    model = _build_model(pre_process=is_first, post_process=is_last)
    input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw = _make_batch()

    if is_first:
        hidden = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=None,
            loss_mask=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        # Non-last stage returns hidden states in [S, B, H].
        assert hidden.shape == (SEQ, BATCH, HIDDEN), hidden.shape
        assert model.vision_model.calls == 1, "vision encoder must run exactly once on stage 0"

        dist.send(hidden.detach().contiguous(), dst=1)
        grad = torch.empty_like(hidden)
        dist.recv(grad, src=1)
        hidden.backward(grad)

        vision_grads = [p.grad for p in model.vision_model.parameters() if p.grad is not None]
        assert vision_grads, "vision encoder received no gradient — it is not in the graph"
        assert all(torch.isfinite(g).all() for g in vision_grads)
        assert any(g.abs().sum() > 0 for g in vision_grads)
    else:
        recvd = torch.empty(SEQ, BATCH, HIDDEN, dtype=torch.bfloat16, device="cuda")
        dist.recv(recvd, src=0)
        activation = recvd.clone().requires_grad_(True)
        model.set_input_tensor(activation)

        # Matches forward_step: pixel_values is dropped off the first stage.
        output = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            pixel_values=None,
            image_grid_thw=image_grid_thw,
        )
        # Last stage returns per-token loss [B, S].
        assert output.shape == (BATCH, SEQ), output.shape
        loss = (output.float() * loss_mask).sum() / loss_mask.sum()
        assert torch.isfinite(loss), f"non-finite loss: {loss}"

        loss.backward()
        assert activation.grad is not None, "no gradient flowed back to the stage input"
        assert torch.isfinite(activation.grad).all()
        dist.send(activation.grad.to(torch.bfloat16).contiguous(), dst=0)

        decoder_grads = [p.grad for p in model.language_model.parameters() if p.grad is not None]
        assert decoder_grads, "last stage produced no parameter gradients"
        assert all(torch.isfinite(g).all() for g in decoder_grads)

    dist.barrier()


def _reinit(pp_size):
    """Re-initialise model-parallel groups with a given PP size."""
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=pp_size)
    model_parallel_cuda_manual_seed(1234)


def _remap_to_stage(ref_state, stage_model, layers_per_stage):
    """Pick this PP stage's slice out of a full (PP=1) state dict.

    Local decoder layer ``i`` on stage ``r`` is global layer
    ``i + r * layers_per_stage``; every other key maps one-to-one.
    """
    offset = ps.get_pipeline_model_parallel_rank() * layers_per_stage
    mapped, missing = {}, []
    for key in stage_model.state_dict():
        if "_extra_state" in key:  # TE bookkeeping, not a real weight
            continue
        src = key
        m = re.match(r"(.*decoder\.layers\.)(\d+)(\..*)", key)
        if m:
            src = f"{m.group(1)}{int(m.group(2)) + offset}{m.group(3)}"
        if src in ref_state:
            mapped[key] = ref_state[src]
        else:
            missing.append((key, src))
    assert not missing, f"unmapped stage params: {missing[:5]}"
    return mapped


@pytest.mark.parametrize(
    "dtype,rtol",
    [
        # Both layouts perform the same ops in the same order on the same
        # weights, so in practice this is bit-identical (observed rel diff
        # 0.0 for both dtypes on H100).  The bounds below are headroom for
        # other hardware / kernel selections, not an expected error budget.
        (torch.float32, 1e-5),
        # bf16 keeps ~3.9e-3 relative precision and the stage boundary adds
        # one more rounding of the activation, hence the looser bound.
        (torch.bfloat16, 5e-3),
    ],
    ids=["fp32", "bf16"],
)
def test_pp2_matches_pp1_with_identical_weights(dtype, rtol):
    """PP=2 reproduces the PP=1 loss when both use the same weights.

    This is the parity check that the end-to-end mock-data run cannot
    give: ``MockQwen35VLDataset.__getitem__`` draws from the global RNG
    rather than seeding per index, so a PP=1 job and a PP=2 job do not
    even consume the same samples.  Here the batch is fixed and the PP=2
    stages are loaded from the PP=1 model, so any loss gap is the PP
    plumbing itself.
    """
    # TF32 has only a 10-bit mantissa, so an "fp32" run still rounds every
    # matmul to ~1e-3 relative and the two pipeline layouts issue different
    # GEMM shapes.  Turn it off so the fp32 case really does isolate the PP
    # plumbing rather than measuring TF32 noise.
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    if dtype is torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    try:
        _pp_parity_body(dtype, rtol)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


def _pp_parity_body(dtype, rtol):
    """Body of the PP=1 vs PP=2 parity check (see the test above)."""
    batch = _make_batch(seed=99, dtype=dtype)
    input_ids, labels, loss_mask, position_ids, pixel_values, image_grid_thw = batch

    # --- Phase 1: PP=1 reference (world becomes pure DP, so every rank
    # builds the identical full model and computes the same loss). ---
    _reinit(pp_size=1)
    ref_model = _build_model(pre_process=True, post_process=True, pp_size=1, dtype=dtype)
    with torch.no_grad():
        ref_out = ref_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=None,
            labels=labels,
            loss_mask=loss_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
    ref_loss = ((ref_out.float() * loss_mask).sum() / loss_mask.sum()).item()
    # TE stores `_extra_state` entries that may be None; keep real tensors only.
    ref_state = {
        k: v.detach().clone()
        for k, v in ref_model.state_dict().items()
        if isinstance(v, torch.Tensor)
    }
    del ref_model
    torch.cuda.empty_cache()

    # --- Phase 2: PP=2, same weights, same batch. ---
    _reinit(pp_size=PP_SIZE)
    # _reinit() rebuilt the process groups, so the stage flags must be derived
    # here -- the module-scoped ``stage_flags`` fixture is stale after this.
    is_first, is_last = ps.is_pipeline_first_stage(), ps.is_pipeline_last_stage()
    stage = _build_model(pre_process=is_first, post_process=is_last, dtype=dtype)
    mapped = _remap_to_stage(ref_state, stage, NUM_LAYERS // PP_SIZE)
    stage.load_state_dict(mapped, strict=False)

    # The whole test is meaningless unless the stage really did adopt the
    # reference weights, so prove it rather than assume it.
    loaded = stage.state_dict()
    bad = [k for k, v in mapped.items() if not torch.equal(loaded[k].to(v.dtype).cpu(), v.cpu())]
    assert not bad, f"{len(bad)}/{len(mapped)} params did not take the reference value: {bad[:5]}"

    with torch.no_grad():
        if is_first:
            hidden = stage(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
                loss_mask=None,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            dist.send(hidden.contiguous(), dst=1)
        else:
            recvd = torch.empty(SEQ, BATCH, HIDDEN, dtype=dtype, device="cuda")
            dist.recv(recvd, src=0)
            stage.set_input_tensor(recvd)
            out = stage(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=None,
                labels=labels,
                loss_mask=loss_mask,
                pixel_values=None,
                image_grid_thw=image_grid_thw,
            )
            pp_loss = ((out.float() * loss_mask).sum() / loss_mask.sum()).item()
            rel = abs(pp_loss - ref_loss) / max(abs(ref_loss), 1e-12)
            print(
                f"\n[{dtype}] PP=1 loss {ref_loss:.8f} | "
                f"PP=2 loss {pp_loss:.8f} | rel {rel:.3E}"
            )
            assert rel < rtol, f"PP=2 loss {pp_loss} != PP=1 loss {ref_loss} (rel {rel:.3E})"

    dist.barrier()
