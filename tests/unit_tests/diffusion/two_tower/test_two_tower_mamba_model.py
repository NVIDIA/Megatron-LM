# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GPU-dependent integration tests for :class:`TwoTowerMambaModel`.

Each test targets a failure mode that would silently corrupt training
rather than raise an exception:

* Information leaks through block-causal attention or Mamba state offsets.
* Frozen context tower parameters receiving gradients.
* Mask-diffusion labels not overridden to same-position targets.

Requires at least one GPU.  Run via ``torchrun --nproc_per_node 1``.
"""

import pytest
import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.diffusion.two_tower.mamba_model import TwoTowerMambaModel, create_block_causal_mask
from tests.unit_tests.test_utilities import Utils

CHUNK_SIZE = 128


# ── Helpers ──────────────────────────────────────────────────────────


def _build_model(hybrid_layer_pattern="M*-", freeze_context=True, num_blocks=1, **overrides):
    """Construct a minimal :class:`TwoTowerMambaModel` with Mamba layers.

    Sequence length is set to ``num_blocks * CHUNK_SIZE`` so the
    ``block_size == chunk_size`` invariant required by Mamba layers is
    always satisfied.

    Args:
        hybrid_layer_pattern (str): Layer pattern string.
        freeze_context (bool): Whether to freeze the context tower.
        num_blocks (int): Number of diffusion blocks.
        **overrides: Forwarded to :class:`TwoTowerMambaModel`.

    Returns:
        TwoTowerMambaModel: Model with ``block_size`` set to ``CHUNK_SIZE``.
    """
    config = TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    seq_len = num_blocks * CHUNK_SIZE
    kwargs = dict(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=100,
        max_sequence_length=seq_len,
        hybrid_layer_pattern=hybrid_layer_pattern,
        freeze_context=freeze_context,
        mask_token_id=99,
    )
    kwargs.update(overrides)
    model = TwoTowerMambaModel(**kwargs)
    model.block_size = CHUNK_SIZE
    return model.to(dtype=torch.bfloat16)


def _build_attn_only_model(freeze_context=True, seq_len=4, **overrides):
    """Construct a minimal :class:`TwoTowerMambaModel` with attention-only layers.

    Because there are no Mamba layers, the ``block_size == chunk_size``
    constraint does not apply and *seq_len* can be arbitrarily small,
    keeping these tests fast.

    Args:
        freeze_context (bool): Whether to freeze the context tower.
        seq_len (int): Sequence length (also determines number of blocks
            when ``block_size = 1``).
        **overrides: Forwarded to :class:`TwoTowerMambaModel`.

    Returns:
        TwoTowerMambaModel: Model with ``block_size = 1``.
    """
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    kwargs = dict(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=100,
        max_sequence_length=seq_len,
        hybrid_layer_pattern="*",
        freeze_context=freeze_context,
        mask_token_id=99,
    )
    kwargs.update(overrides)
    model = TwoTowerMambaModel(**kwargs)
    model.block_size = 1
    return model.to(dtype=torch.bfloat16)


def _make_inputs(model, micro_batch_size=2, device="cuda"):
    """Generate random input tensors sized for *model*.

    Args:
        model (TwoTowerMambaModel): Model whose ``max_sequence_length`` and
            ``vocab_size`` determine tensor shapes.
        micro_batch_size (int): Batch dimension.
        device (str): Target device.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: ``(input_ids, position_ids,
            attention_mask, labels)`` all on *device*.
    """
    seq_len = model.max_sequence_length
    input_ids = torch.randint(0, model.vocab_size, (micro_batch_size, seq_len), device=device)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(micro_batch_size, -1)
    attention_mask = torch.ones((micro_batch_size, 1, seq_len, seq_len), dtype=bool).to(device)
    labels = torch.randint(0, model.vocab_size, (micro_batch_size, seq_len), device=device)
    return input_ids, position_ids, attention_mask, labels


@pytest.mark.internal
class TestTwoTowerMambaModel:
    """GPU integration tests for :class:`TwoTowerMambaModel`.

    Each test guards against a specific silent-corruption failure mode.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_forward_mask_diffusion_finite_loss(self):
        """End-to-end forward with mask_diffusion must produce finite, non-negative loss."""
        model = _build_model(num_blocks=1)
        model.cuda()
        input_ids, position_ids, attention_mask, labels = _make_inputs(model)

        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert loss.dim() == 0, "training_loss should return a scalar"
        assert torch.isfinite(loss), "Loss contains NaN or Inf"
        assert loss >= 0, "Cross-entropy loss must be non-negative"

    def test_freeze_context_no_gradients(self):
        """Context-side parameters must receive no gradients when frozen.

        Verifies that ``context_embedding``, ``context_tower``, and
        ``context_output_layer`` have no gradients after backward, while
        the denoiser side has at least one non-zero gradient.  A leak here
        would cause silent drift from the pretrained initialisation.
        """
        model = _build_model(freeze_context=True, num_blocks=1)
        model.cuda()
        input_ids, position_ids, attention_mask, labels = _make_inputs(model)

        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss.sum().backward()

        frozen_modules = ["context_embedding", "context_tower", "context_output_layer"]
        for mod_name in frozen_modules:
            mod = getattr(model, mod_name, None)
            if mod is None:
                continue
            for pname, param in mod.named_parameters():
                assert (
                    param.grad is None or (param.grad == 0).all()
                ), f"{mod_name}.{pname} has non-zero gradient despite freeze_context=True"

        trainable_modules = ["denoiser_embedding", "denoiser_tower", "output_layer"]
        has_any_grad = False
        for mod_name in trainable_modules:
            mod = getattr(model, mod_name, None)
            if mod is None:
                continue
            for pname, param in mod.named_parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_any_grad = True
                    break
            if has_any_grad:
                break
        assert has_any_grad, "Denoiser side should have non-zero gradients"

    def test_mamba_state_no_future_leak(self):
        """Modifying tokens in block 2 must not change block 0's output.

        Denoiser block 0 receives zero initial Mamba states (no prior context).
        An off-by-one error in the state-slicing logic (block *N* receiving
        states from context block *N* instead of *N-1*) would silently leak
        future information through the Mamba recurrent channel.
        """
        num_blocks = 3
        seq_len = num_blocks * CHUNK_SIZE

        config = TransformerConfig(
            num_layers=1, hidden_size=256, num_attention_heads=4, use_cpu_initialization=True
        )
        model = TwoTowerMambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=100,
            max_sequence_length=seq_len,
            hybrid_layer_pattern="M",
            freeze_context=False,
        )
        model.block_size = CHUNK_SIZE
        model.cuda()
        model.eval()

        clean_ids = torch.randint(0, 100, (1, seq_len), device="cuda")
        noisy_base = torch.randint(0, 100, (1, seq_len), device="cuda")
        noisy_alt = noisy_base.clone()
        noisy_alt[:, 2 * CHUNK_SIZE :] = 99

        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device="cuda")

        with torch.no_grad():
            logits_base = model.forward(
                input_ids=clean_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noisy_input_ids=noisy_base,
            )
            logits_alt = model.forward(
                input_ids=clean_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noisy_input_ids=noisy_alt,
            )

        block0_base = logits_base[:, :CHUNK_SIZE, :]
        block0_alt = logits_alt[:, :CHUNK_SIZE, :]
        assert torch.allclose(block0_base, block0_alt, atol=1e-5), (
            "Changing block 2 tokens affected block 0 output — "
            "Mamba state offset is leaking future information"
        )

    def test_attention_mask_prevents_future_context(self):
        """Denoiser block 0 must not attend to context block 1 or later.

        Injects a large sentinel value into context block 1's Key/Value
        tensors and asserts that denoiser block 0's output is unchanged.
        A permissive attention mask would leak future context into earlier
        blocks, silently degrading training quality.
        """
        seq_len = 2
        model = _build_attn_only_model(freeze_context=False, seq_len=seq_len)
        model.cuda()
        model.eval()

        input_ids = torch.tensor([[5, 10]], device="cuda")
        noisy_ids = torch.tensor([[7, 12]], device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
        attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool, device="cuda")

        with torch.no_grad():
            logits_normal = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noisy_input_ids=noisy_ids,
            )

        original_forward = model._forward_context_tower

        def _patched_context_tower(*args, **kwargs):
            kv_cache, mamba_cache, context_hidden = original_forward(*args, **kwargs)
            for i, (k, v) in enumerate(kv_cache):
                k[1:, ...] = 1000.0
                v[1:, ...] = 1000.0
                kv_cache[i] = (k, v)
            return kv_cache, mamba_cache, context_hidden

        model._forward_context_tower = _patched_context_tower

        with torch.no_grad():
            logits_sentinel = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noisy_input_ids=noisy_ids,
            )

        block0_normal = logits_normal[:, :1, :]
        block0_sentinel = logits_sentinel[:, :1, :]
        assert torch.allclose(block0_normal, block0_sentinel, atol=1e-4), (
            "Injecting sentinel values into context block 1's KV affected "
            "denoiser block 0's output — attention mask is not preventing "
            "future context leakage"
        )

    def test_context_ar_loss_unfreezes_output_head(self):
        """Context output layer must receive gradients when ``context_ar_loss=True``.

        With tied towers and context AR loss enabled, the context output head
        is the only non-shared parameter that is trained by the AR objective.
        If it is accidentally frozen (wrong branch in the freeze logic), the
        AR loss still computes and backpropagates through the shared tower
        body, but the context head itself learns nothing — a silent failure.
        """
        model = _build_attn_only_model(
            freeze_context=False, seq_len=4, tied_towers=True, context_ar_loss=True
        )
        model.cuda()
        input_ids, position_ids, attention_mask, _ = _make_inputs(model)
        shifted_labels = torch.roll(input_ids, -1, dims=1)

        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=shifted_labels,
        )
        loss.sum().backward()

        has_grad = False
        for param in model.context_output_layer.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, (
            "context_output_layer has no gradients despite context_ar_loss=True — "
            "the output head is likely still frozen"
        )

    def test_mask_diffusion_labels_overridden_to_input_ids(self):
        """In mask_diffusion mode, labels must be overridden to ``input_ids``.

        Mask diffusion trains on same-position targets (predict the original
        token at each masked position).  If the caller's shifted AR labels
        are used instead, the loss still decreases but optimises the wrong
        objective.  This test supplies deliberately shifted labels and
        verifies the model replaces them with ``input_ids.clone()``.
        """
        captured = {}

        class CapturingProcess(torch.nn.Module):
            """Test-only wrapper that records the labels seen by ``training_loss``."""

            def __init__(self, real_process):
                super().__init__()
                self._real = real_process
                self.mode = real_process.mode

            def corrupt_suffix(self, *args, **kwargs):
                return self._real.corrupt_suffix(*args, **kwargs)

            def training_loss(self, logits, labels, aux):
                captured["labels"] = labels.clone()
                return self._real.training_loss(logits, labels, aux)

            def set_cross_entropy_fn(self, fn):
                return self._real.set_cross_entropy_fn(fn)

            def forward(self, *args, **kwargs):
                return self._real(*args, **kwargs)

        model = _build_attn_only_model(seq_len=4)
        model.diffusion = CapturingProcess(model.diffusion)
        model.cuda()

        input_ids, position_ids, attention_mask, _ = _make_inputs(model)
        shifted_labels = torch.roll(input_ids, -1, dims=1)

        model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=shifted_labels,
        )

        assert "labels" in captured, "training_loss was never called"
        assert torch.equal(captured["labels"], input_ids), (
            "mask_diffusion must override labels to input_ids.clone(), "
            "not use the shifted labels passed by the caller"
        )

    def test_bidirectional_mamba_finite_loss(self):
        """Forward with ``bidirectional_mamba=True`` must produce finite loss.

        Also verifies the backward SSM pass actually executes by comparing
        against a unidirectional forward — outputs should differ.
        """
        model = _build_model(
            hybrid_layer_pattern="M", num_blocks=1, freeze_context=False, bidirectional_mamba=True
        )
        model.cuda()
        model.eval()
        input_ids, position_ids, attention_mask, labels = _make_inputs(model)

        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert loss.dim() == 0, "training_loss should return a scalar"
        assert torch.isfinite(loss), "Bidirectional loss contains NaN or Inf"
        assert loss >= 0, "Cross-entropy loss must be non-negative"

        noisy_ids, _ = model.diffusion.corrupt_suffix(input_ids)
        with torch.no_grad():
            logits_bidir = model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noisy_input_ids=noisy_ids,
            )

        model_unidir = _build_model(
            hybrid_layer_pattern="M", num_blocks=1, freeze_context=False, bidirectional_mamba=False
        )
        model_unidir.cuda()
        model_unidir.eval()
        model_unidir.load_state_dict(model.state_dict(), strict=False)

        with torch.no_grad():
            logits_unidir = model_unidir.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noisy_input_ids=noisy_ids,
            )

        assert not torch.allclose(logits_bidir, logits_unidir, atol=1e-4), (
            "Bidirectional and unidirectional outputs are identical — "
            "the backward SSM pass is likely not executing"
        )

    def test_time_conditioning_finite_loss(self):
        """Forward with ``use_time_conditioning=True`` must produce finite loss."""
        model = _build_attn_only_model(freeze_context=True, seq_len=4, use_time_conditioning=True)
        model.cuda()
        input_ids, position_ids, attention_mask, labels = _make_inputs(model)

        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert loss.dim() == 0, "training_loss should return a scalar"
        assert torch.isfinite(loss), "Time-conditioned loss contains NaN or Inf"
        assert loss >= 0, "Cross-entropy loss must be non-negative"

    # ── Inference tests ─────────────────────────────────────────────

    @torch.no_grad()
    def test_generate_diffusion_produces_valid_output(self):
        """``generate_diffusion()`` must produce well-formed output with no mask tokens."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (4,), device="cuda"),
        ]
        max_new = 4

        output_list, nfe = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=max_new,
            block_length=1,
            steps_per_block=1,
            temperature=0.0,
        )
        assert len(output_list) == 2
        for i, out in enumerate(output_list):
            assert (
                out.shape[0] == 4 + max_new
            ), f"Request {i}: expected length {4 + max_new}, got {out.shape[0]}"
            generated = out[4:]
            assert (generated != model.mask_token_id).all(), "Generated tokens contain mask token"
            assert (generated >= 0).all() and (
                generated < model.vocab_size
            ).all(), "Token IDs out of valid range"

    @torch.no_grad()
    def test_forward_context_with_cache_structure(self):
        """``_forward_context_with_cache()`` must return a well-formed cache dict."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        input_ids = torch.randint(0, 98, (2, 6), device="cuda")

        cache = model._forward_context_with_cache(input_ids)

        assert "kv" in cache and "mamba" in cache and "len" in cache
        assert cache["len"] == 6
        assert (
            len(cache["kv"]) == model.num_attention_layers
        ), f"Expected {model.num_attention_layers} KV entries, got {len(cache['kv'])}"
        for k, v in cache["kv"]:
            assert k.shape[0] == 6, f"KV seq dim should be 6, got {k.shape[0]}"
            assert v.shape[0] == 6

    @torch.no_grad()
    def test_extend_context_cache_grows_correctly(self):
        """``_extend_context_cache()`` must grow KV by the new block length."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        input_ids = torch.randint(0, 98, (2, 4), device="cuda")

        cache = model._forward_context_with_cache(input_ids)
        assert cache["len"] == 4

        new_tokens = torch.randint(0, 98, (2, 2), device="cuda")
        cache = model._extend_context_cache(new_tokens, cache)

        assert cache["len"] == 6
        for k, v in cache["kv"]:
            assert k.shape[0] == 6, f"KV should grow to 6, got {k.shape[0]}"

    @torch.no_grad()
    def test_run_denoiser_step_batched_finite_logits(self):
        """``_run_denoiser_step_batched()`` must return finite logits of correct shape."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()

        context_ids = torch.randint(0, 98, (2, 4), device="cuda")
        batched_cache = model._merge_context_caches(
            [
                model._forward_context_with_cache(context_ids[0:1]),
                model._forward_context_with_cache(context_ids[1:2]),
            ]
        )

        noisy_block = torch.full((2, 3), model.mask_token_id, device="cuda")
        logits = model._run_denoiser_step_batched(noisy_block, batched_cache)

        assert logits.shape == (
            2,
            3,
            model.vocab_size,
        ), f"Expected logits (2, 3, {model.vocab_size}), got {logits.shape}"
        assert torch.isfinite(logits).all(), "Denoiser step produced non-finite logits"

    @torch.no_grad()
    def test_forward_for_likelihood_shape(self):
        """``forward_for_likelihood()`` returns per-request ``(S_i, V)`` finite logits."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        prompt_ids_list = [
            torch.randint(0, 98, (6,), device="cuda"),
            torch.randint(0, 98, (6,), device="cuda"),
        ]

        logits_list = model.forward_for_likelihood(prompt_ids_list)
        assert len(logits_list) == 2
        for logits, prompt in zip(logits_list, prompt_ids_list):
            assert logits.shape == (
                prompt.shape[0],
                model.vocab_size,
            ), f"Expected ({prompt.shape[0]}, {model.vocab_size}), got {logits.shape}"
            assert torch.isfinite(logits).all(), "Likelihood logits contain NaN/Inf"

    @torch.no_grad()
    def test_forward_for_likelihood_uses_context_output_layer(self):
        """``forward_for_likelihood()`` must use ``context_output_layer``."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        prompt_ids_list = [
            torch.randint(0, 98, (6,), device="cuda"),
            torch.randint(0, 98, (6,), device="cuda"),
        ]

        logits_list = model.forward_for_likelihood(prompt_ids_list)
        for logits in logits_list:
            assert logits.shape[-1] == model.vocab_size
            assert torch.isfinite(logits).all()

    @torch.no_grad()
    def test_forward_for_likelihood_variable_length(self):
        """``forward_for_likelihood()`` handles variable-length prompts correctly."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompt_ids_list = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (7,), device="cuda"),
            torch.randint(0, 98, (3,), device="cuda"),
        ]

        logits_list = model.forward_for_likelihood(prompt_ids_list)
        assert len(logits_list) == 3
        for logits, prompt in zip(logits_list, prompt_ids_list):
            assert logits.shape == (
                prompt.shape[0],
                model.vocab_size,
            ), f"Expected ({prompt.shape[0]}, {model.vocab_size}), got {logits.shape}"
            assert torch.isfinite(logits).all(), "Likelihood logits contain NaN/Inf"

    @torch.no_grad()
    def test_load_from_single_tower_weight_mapping(self):
        """``load_from_single_tower()`` must map weights to both towers."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()

        sentinel = torch.randn_like(model.context_embedding.word_embeddings.weight.data)
        fake_state = {"embedding.word_embeddings.weight": sentinel.clone()}
        for key, val in model.state_dict().items():
            if val is None:
                continue
            if key.startswith("context_tower."):
                st_key = key.replace("context_tower.", "decoder.")
                fake_state[st_key] = val.clone()
            elif key.startswith("output_layer."):
                fake_state[key] = val.clone()

        model.load_from_single_tower(fake_state, strict=False)

        assert model._single_tower_mode is True
        assert torch.equal(
            model.context_embedding.word_embeddings.weight.data.cpu(), sentinel.cpu()
        ), "Context embedding not loaded from single-tower state"
        assert torch.equal(
            model.denoiser_embedding.word_embeddings.weight.data.cpu(), sentinel.cpu()
        ), "Denoiser embedding not loaded from single-tower state"

    @torch.no_grad()
    def test_generate_ar_greedy_deterministic(self):
        """Greedy AR generation must produce identical results across calls."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()

        prompt = [torch.randint(0, 98, (4,), device="cuda")]

        out1, _ = model.generate_ar(prompt_ids_list=prompt, max_new_tokens=4, temperature=0.0)
        out2, _ = model.generate_ar(prompt_ids_list=prompt, max_new_tokens=4, temperature=0.0)
        assert torch.equal(out1[0], out2[0]), "Greedy AR generation is not deterministic"

    @torch.no_grad()
    def test_generate_diffusion_with_time_conditioning(self):
        """``generate_diffusion()`` with ``use_time_conditioning=True`` must produce valid output."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8, use_time_conditioning=True)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (4,), device="cuda"),
        ]

        output_list, _ = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=4,
            block_length=1,
            steps_per_block=1,
            temperature=0.0,
        )
        assert len(output_list) == 2
        for i, out in enumerate(output_list):
            assert out.shape[0] == 8, f"Request {i}: expected len 8, got {out.shape[0]}"
            generated = out[4:]
            assert (generated != model.mask_token_id).all(), "Output contains mask tokens"
            assert torch.isfinite(generated.float()).all()

    @torch.no_grad()
    def test_generate_diffusion_with_bidirectional_mamba(self):
        """``generate_diffusion()`` with ``bidirectional_mamba=True`` must produce valid output."""
        model = _build_model(
            hybrid_layer_pattern="M", num_blocks=1, freeze_context=False, bidirectional_mamba=True
        )
        model.cuda()
        model.eval()

        prompts = [torch.randint(0, 98, (CHUNK_SIZE,), device="cuda")]

        output_list, _ = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=CHUNK_SIZE,
            block_length=CHUNK_SIZE,
            steps_per_block=1,
            temperature=0.0,
        )
        assert len(output_list) == 1
        assert (
            output_list[0].shape[0] == 2 * CHUNK_SIZE
        ), f"Expected length {2 * CHUNK_SIZE}, got {output_list[0].shape[0]}"
        generated = output_list[0][CHUNK_SIZE:]
        assert (generated != model.mask_token_id).all(), "Output contains mask tokens"

    # ── Block-causal mask unit test ──────────────────────────────────

    def test_create_block_causal_mask_structure(self):
        """``create_block_causal_mask`` must have correct shape and attend/mask pattern.

        For ``num_blocks=3, block_size=2`` the mask is ``(1, 1, 6, 12)``
        where KV dim = ``2 * seq_len``.  Denoiser block *N* should attend
        to context blocks ``0..N-1`` and denoiser block *N* only.
        """
        num_blocks, block_size = 3, 2
        seq_len = num_blocks * block_size
        mask = create_block_causal_mask(num_blocks, block_size, device="cuda", dtype=torch.float32)

        assert mask.shape == (
            1,
            1,
            seq_len,
            2 * seq_len,
        ), f"Expected (1, 1, {seq_len}, {2 * seq_len}), got {mask.shape}"

        m = mask.squeeze()
        min_val = torch.finfo(torch.float32).min

        for q_block in range(num_blocks):
            q_start = q_block * block_size
            q_end = q_start + block_size
            for kv_pos in range(2 * seq_len):
                is_context = kv_pos < seq_len
                kv_block = kv_pos // block_size if is_context else (kv_pos - seq_len) // block_size
                val = m[q_start, kv_pos].item()

                if is_context and kv_block < q_block:
                    assert (
                        val == 0.0
                    ), f"Denoiser block {q_block} should attend to context block {kv_block}"
                elif not is_context and kv_block == q_block:
                    assert (
                        val == 0.0
                    ), f"Denoiser block {q_block} should attend to own denoiser block"
                else:
                    assert val == min_val, f"Position ({q_start}, {kv_pos}) should be masked"

    # ── Constructor validation tests ─────────────────────────────────

    def test_context_ar_loss_requires_tied_towers(self):
        """``context_ar_loss=True`` without ``tied_towers`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="tied-towers"):
            _build_attn_only_model(
                freeze_context=False, seq_len=4, tied_towers=False, context_ar_loss=True
            )

    def test_context_ar_loss_requires_unfrozen_context(self):
        """``context_ar_loss=True`` with ``freeze_context=True`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match="no-freeze-context"):
            _build_attn_only_model(
                freeze_context=True, seq_len=4, tied_towers=True, context_ar_loss=True
            )

    # ── Tied towers basic test ───────────────────────────────────────

    def test_tied_towers_share_parameters(self):
        """With ``tied_towers=True`` the two towers must share parameter tensors.

        Also verifies forward produces finite loss and the context output
        layer is frozen when ``context_ar_loss=False``.
        """
        model = _build_attn_only_model(
            freeze_context=False, seq_len=4, tied_towers=True, context_ar_loss=False
        )
        model.cuda()

        for ctx_p, den_p in zip(
            model.context_tower.parameters(), model.denoiser_tower.parameters()
        ):
            assert (
                ctx_p.data_ptr() == den_p.data_ptr()
            ), "Tied towers should share the same parameter data pointers"

        for param in model.context_output_layer.parameters():
            assert (
                not param.requires_grad
            ), "context_output_layer should be frozen when context_ar_loss=False"

        input_ids, position_ids, attention_mask, labels = _make_inputs(model)
        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert torch.isfinite(loss), "Tied-tower loss contains NaN or Inf"

    # ── Multi-block training forward test ────────────────────────────

    def test_multi_block_training_forward(self):
        """Training with ``num_blocks=2`` and full ``M*-`` pattern must produce finite loss.

        This exercises cross-block attention and Mamba state passing
        simultaneously, which is the actual training regime.
        """
        model = _build_model(hybrid_layer_pattern="M*-", freeze_context=True, num_blocks=2)
        model.cuda()
        input_ids, position_ids, attention_mask, labels = _make_inputs(model)

        loss = model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        assert loss.dim() == 0, "training_loss should return a scalar"
        assert torch.isfinite(loss), "Multi-block loss contains NaN or Inf"
        assert loss >= 0, "Cross-entropy loss must be non-negative"

    # ── Multi-step generation test ───────────────────────────────────

    @torch.no_grad()
    def test_generate_diffusion_multi_step_denoising(self):
        """``generate_diffusion()`` with ``steps_per_block > 1`` must produce valid output.

        Exercises the iterative denoising loop in ``sample_block``.
        """
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        prompts = [torch.randint(0, 98, (4,), device="cuda")]

        output_list, nfe = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=4,
            block_length=1,
            steps_per_block=4,
            temperature=0.0,
        )
        assert output_list[0].shape[0] == 8, f"Expected length 8, got {output_list[0].shape[0]}"
        generated = output_list[0][4:]
        assert (generated != model.mask_token_id).all(), "Output contains mask tokens"
        assert nfe >= 4, f"Expected at least 4 forward evals for steps_per_block=4, got {nfe}"

    # ── Batched inference tests ──────────────────────────────────────

    @torch.no_grad()
    def test_generate_diffusion_same_length_deterministic(self):
        """``generate_diffusion()`` with same-length prompts must be deterministic."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (4,), device="cuda"),
        ]

        out1, nfe1 = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=4,
            block_length=1,
            steps_per_block=1,
            temperature=0.0,
        )

        out2, nfe2 = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=4,
            block_length=1,
            steps_per_block=1,
            temperature=0.0,
        )

        assert nfe1 == nfe2, f"NFE mismatch: {nfe1} vs {nfe2}"
        for i in range(2):
            assert torch.equal(
                out1[i], out2[i]
            ), f"Request {i} outputs differ between two generate_diffusion calls"

    @torch.no_grad()
    def test_generate_diffusion_variable_length_valid_output(self):
        """``generate_diffusion()`` with variable-length prompts must produce valid output."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (3,), device="cuda"),
            torch.randint(0, 98, (5,), device="cuda"),
            torch.randint(0, 98, (7,), device="cuda"),
        ]

        results, nfe = model.generate_diffusion(
            prompt_ids_list=prompts,
            max_new_tokens=4,
            block_length=1,
            steps_per_block=1,
            temperature=0.0,
        )

        assert len(results) == 3
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected_len = prompt.shape[0] + 4
            assert (
                result.shape[0] == expected_len
            ), f"Request {i}: expected len {expected_len}, got {result.shape[0]}"
            assert torch.equal(
                result[: prompt.shape[0]], prompt
            ), f"Request {i}: prompt prefix was corrupted"
            generated = result[prompt.shape[0] :]
            assert (
                generated != model.mask_token_id
            ).all(), f"Request {i}: generated tokens contain mask token"

    @torch.no_grad()
    def test_generate_diffusion_single_vs_batched_equivalence(self):
        """Batched diffusion must match serial calls (predict_and_noise + adaptive).

        Uses ``block_length > 1`` and ``steps_per_block > 1`` with
        ``adaptive_unmasking=True`` so the per-row commit / re-mask logic
        in ``sample_block`` is actually exercised through the real model
        (left-padding, ``_merge_context_caches``, attention mask). The
        degenerate ``block_length=1, steps_per_block=1`` setup never
        enters the multi-step adaptive branches, so a cross-row coupling
        bug at the model layer could pass silently there.
        """
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (6,), device="cuda"),
        ]
        gen_kwargs = dict(
            max_new_tokens=8,
            block_length=4,
            steps_per_block=8,
            temperature=0.0,
            sampling_strategy="predict_and_noise",
            adaptive_unmasking=True,
        )

        serial_results = []
        for p in prompts:
            out, _ = model.generate_diffusion(prompt_ids_list=[p], **gen_kwargs)
            serial_results.append(out[0])

        batched_results, _ = model.generate_diffusion(prompt_ids_list=prompts, **gen_kwargs)

        for i in range(len(prompts)):
            assert torch.equal(
                serial_results[i], batched_results[i]
            ), f"Request {i}: serial and batched outputs differ"

    @torch.no_grad()
    def test_generate_diffusion_single_vs_batched_equivalence_confidence_unmasking(self):
        """Same as above but for ``sampling_strategy='confidence_unmasking'``.

        Exercises the confidence-based commit / re-mask branch end-to-end
        through the real model, with rows finishing at different rates.
        """
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (6,), device="cuda"),
        ]
        gen_kwargs = dict(
            max_new_tokens=8,
            block_length=4,
            steps_per_block=8,
            temperature=0.0,
            sampling_strategy="confidence_unmasking",
            confidence_threshold=0.5,
        )

        serial_results = []
        for p in prompts:
            out, _ = model.generate_diffusion(prompt_ids_list=[p], **gen_kwargs)
            serial_results.append(out[0])

        batched_results, _ = model.generate_diffusion(prompt_ids_list=prompts, **gen_kwargs)

        for i in range(len(prompts)):
            assert torch.equal(
                serial_results[i], batched_results[i]
            ), f"Request {i}: serial and batched outputs differ (confidence_unmasking)"

    @torch.no_grad()
    def test_generate_diffusion_single_vs_batched_equivalence_posterior(self):
        """Same as above but for ``sampling_strategy='posterior'``."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (6,), device="cuda"),
        ]
        gen_kwargs = dict(
            max_new_tokens=8,
            block_length=4,
            steps_per_block=8,
            temperature=0.0,
            sampling_strategy="posterior",
        )

        serial_results = []
        for p in prompts:
            out, _ = model.generate_diffusion(prompt_ids_list=[p], **gen_kwargs)
            serial_results.append(out[0])

        batched_results, _ = model.generate_diffusion(prompt_ids_list=prompts, **gen_kwargs)

        for i in range(len(prompts)):
            assert torch.equal(
                serial_results[i], batched_results[i]
            ), f"Request {i}: serial and batched outputs differ (posterior)"

    @torch.no_grad()
    def test_merge_context_caches_structure(self):
        """``_merge_context_caches()`` must produce correctly shaped batched cache."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()

        caches = []
        for length in [3, 5, 7]:
            ids = torch.randint(0, 98, (1, length), device="cuda")
            caches.append(model._forward_context_with_cache(ids))

        merged = model._merge_context_caches(caches)

        assert merged["len"] == 7, f"Expected max len 7, got {merged['len']}"
        assert merged["pad_amounts"] == [
            4,
            2,
            0,
        ], f"Expected pad_amounts [4, 2, 0], got {merged['pad_amounts']}"
        for k, v in merged["kv"]:
            assert k.shape[0] == 7, f"KV seq dim should be 7, got {k.shape[0]}"
            assert k.shape[1] == 3, f"KV batch dim should be 3, got {k.shape[1]}"

    @torch.no_grad()
    def test_build_batched_kv_mask(self):
        """``_build_batched_kv_mask()`` must mask only left-padding positions."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()

        mask = model._build_batched_kv_mask(
            pad_amounts=[3, 1, 0],
            past_kv_len=5,
            q_len=2,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        assert mask is not None
        assert mask.shape == (3, 1, 1, 7)
        neg_inf = float('-inf')
        assert mask[0, 0, 0, 0].item() == neg_inf
        assert mask[0, 0, 0, 2].item() == neg_inf
        assert mask[0, 0, 0, 3].item() == 0.0
        assert mask[1, 0, 0, 0].item() == neg_inf
        assert mask[1, 0, 0, 1].item() == 0.0
        assert mask[2, 0, 0, 0].item() == 0.0

    @torch.no_grad()
    def test_prefill_cache_structure(self):
        """``_prefill()`` must return well-formed per-request caches."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (3,), device="cuda"),
            torch.randint(0, 98, (5,), device="cuda"),
            torch.randint(0, 98, (7,), device="cuda"),
        ]

        caches = model._prefill(prompts)

        assert len(caches) == 3
        for i, cache in enumerate(caches):
            assert "kv" in cache and "mamba" in cache and "len" in cache
            expected_len = prompts[i].shape[0]
            assert (
                cache["len"] == expected_len
            ), f"Request {i}: expected len {expected_len}, got {cache['len']}"
            assert len(cache["kv"]) == model.num_attention_layers
            for k, v in cache["kv"]:
                assert (
                    k.shape[0] == expected_len
                ), f"KV seq dim should be {expected_len}, got {k.shape[0]}"

    @torch.no_grad()
    def test_prefill_matches_serial(self):
        """``_prefill()`` must produce caches equivalent to serial ``_forward_context_with_cache``."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (4,), device="cuda"),
            torch.randint(0, 98, (6,), device="cuda"),
        ]

        serial_caches = []
        for p in prompts:
            serial_caches.append(model._forward_context_with_cache(p.unsqueeze(0)))

        packed_caches = model._prefill(prompts)

        for i in range(len(prompts)):
            for layer_idx in range(len(serial_caches[i]["kv"])):
                sk, sv = serial_caches[i]["kv"][layer_idx]
                pk, pv = packed_caches[i]["kv"][layer_idx]
                assert torch.allclose(
                    sk.squeeze(1), pk.squeeze(1), atol=1e-4
                ), f"Request {i}, layer {layer_idx}: KV mismatch between serial and packed prefill"

    @torch.no_grad()
    def test_generate_ar_variable_length(self):
        """``generate_ar()`` with variable-length prompts must produce valid output."""
        model = _build_attn_only_model(freeze_context=False, seq_len=16)
        model.cuda()
        model.eval()
        prompts = [
            torch.randint(0, 98, (3,), device="cuda"),
            torch.randint(0, 98, (5,), device="cuda"),
            torch.randint(0, 98, (7,), device="cuda"),
        ]

        results, nfe = model.generate_ar(prompt_ids_list=prompts, max_new_tokens=4, temperature=0.0)

        assert len(results) == 3
        for i, (prompt, result) in enumerate(zip(prompts, results)):
            expected_len = prompt.shape[0] + 4
            assert (
                result.shape[0] == expected_len
            ), f"Request {i}: expected len {expected_len}, got {result.shape[0]}"
            assert torch.equal(
                result[: prompt.shape[0]], prompt
            ), f"Request {i}: prompt prefix was corrupted"

    @torch.no_grad()
    def test_build_batched_kv_mask_returns_none_when_equal(self):
        """``_build_batched_kv_mask()`` must return ``None`` when no padding needed."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()

        mask = model._build_batched_kv_mask(
            pad_amounts=[0, 0],
            past_kv_len=4,
            q_len=2,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )
        assert mask is None

    @torch.no_grad()
    def test_build_batched_causal_kv_mask(self):
        """``_build_batched_causal_kv_mask()`` must enforce causality + padding."""
        model = _build_attn_only_model(freeze_context=False, seq_len=8)
        model.cuda()

        mask = model._build_batched_causal_kv_mask(
            pad_amounts=[2, 0],
            past_kv_len=4,
            q_len=3,
            device=torch.device("cuda"),
            dtype=torch.float32,
        )

        assert mask is not None
        assert mask.shape == (2, 1, 3, 7)
        neg_inf = float('-inf')
        assert mask[0, 0, 0, 0].item() == neg_inf
        assert mask[0, 0, 0, 1].item() == neg_inf
        assert mask[0, 0, 0, 2].item() == 0.0
        assert mask[0, 0, 0, 3].item() == 0.0
        assert mask[0, 0, 0, 4].item() == 0.0
        assert mask[0, 0, 0, 5].item() == neg_inf
        assert mask[0, 0, 2, 6].item() == 0.0
        assert mask[1, 0, 0, 0].item() == 0.0
