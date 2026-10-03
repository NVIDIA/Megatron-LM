# Two-Tower Mamba Diffusion

Two-Tower Mamba-Hybrid Model for Block-wise Diffusion Language Modeling.

	Architecture:
		- Context Tower: Processes clean sequence, caches Attention KV + Mamba states
		- Denoiser Tower: Processes noisy sequence with block-causal attention to context
		- Backbone: Mamba-hybrid (Mamba SSM + Attention layers via --hybrid-layer-pattern)

	Key Configuration Flags:
		--tt-diffusion-block-size B
			Tokens per diffusion block (must equal --mamba-chunk-size)
		--tt-diffusion-tied-towers
			Share weights between towers (default: separate)
		--tt-diffusion-no-freeze-context
			Unfreeze context tower (default: frozen)
		--tt-diffusion-time-conditioning
			adaLN-single modulation on denoiser (PixArt-α style)
		--tt-diffusion-bidirectional-mamba
			Forward + reverse SSM in denoiser Mamba layers
		--tt-diffusion-context-ar-loss
			Add next-token CE from context head
			(requires --tt-diffusion-tied-towers and --tt-diffusion-no-freeze-context)
		--tt-diffusion-mask-token-id ID
			Absorbing-state token for mask diffusion

	Diffusion Process:
		MaskDiffusionProcess:
			Each token is independently replaced with mask_token_id with
			probability (1 − α_t) where t ~ Uniform[ε, 1].
			Loss = CE at masked positions only.

	Tower Configurations (4 combinations):

		Separate towers + frozen context (default):
			Only denoiser trains; context tower, context embedding, and
			context output head are frozen. Typical setup: denoiser learns
			to predict clean tokens given a fixed context representation.

		Separate towers + unfrozen context (--tt-diffusion-no-freeze-context):
			Not yet implemented. Equivalent to the default (separate + frozen)
			in practice: the context tower is unfrozen but no loss reaches it
			because cached states are detached and context AR loss requires
			tied towers. Needs a standalone context-side loss to be useful.

		Tied towers + frozen context (--tt-diffusion-tied-towers):
			Shared body parameters (one copy in memory); context output head
			frozen. Shared body still trains via the denoiser loss path.

		Tied towers + unfrozen context (--tt-diffusion-tied-towers + --tt-diffusion-no-freeze-context):
			Shared body, context head unfrozen. Required for --tt-diffusion-context-ar-loss,
			which adds next-token CE from the context head to the diffusion loss.

	Training Flow:
		1. Diffusion process corrupts input_ids → noisy_input_ids (per-example t)
		2. Clean tokens → context embedding; noisy tokens → denoiser embedding
		3. Context tower runs full causal forward, caches KV + Mamba states per layer
		4. Block-causal mask is built for the denoiser attention layers
		5. Time embedding encodes t → per-layer (shift, scale, gate) if enabled
		6. Denoiser tower runs with cached context (attention: concat KV; Mamba: initial states)
		7. Loss: CE at masked positions + optional context AR CE vs original labels

	Block-Causal Attention (denoiser block N):
		- Attends to context blocks 0 .. N−1 (strictly past, no context block N)
		- Attends to denoiser block N only (self, bidirectional within block)
		- No cross-block denoiser attention; no future context

## TwoTower: Diffusion LLM with Autoregressive Context

```
      AR / Context Tower            Diffusion / Denoiser Tower

          clean tokens                   noisy token blocks
               │                                 │
       ┌───────▼───────┐                 ┌───────▼───────┐
       │   Embedding   │                 │   Embedding   │
       └───────┬───────┘                 └───────┬───────┘
               │                                 │
       ┌───────▼───────┐    KV + Mamba   ┌───────▼───────┐
       │ Mamba-2/Attn  │─────states─────▶│ Mamba-2/Attn  │
       └───────┬───────┘                 └───────┬───────┘
               │                                 │
       ┌───────▼───────┐                 ┌───────▼───────┐
       │      MoE      │                 │      MoE      │
       └───────┬───────┘                 └───────┬───────┘
               │                                 │
       ┌───────▼───────┐                 ┌───────▼───────┐
       │  Output Head  │                 │  Output Head  │
       └───────┬───────┘                 └───────┬───────┘
               │                                 │
         logits / loss                     logits / loss
          (optional)
```

**Context tower** processes the clean token sequence and caches per-layer
Attention KV pairs and Mamba (conv + SSM) states at every block boundary.

**Denoiser tower** processes the noisy (diffusion-corrupted) sequence.
Each denoiser block *N* attends to context blocks `0 .. N-1` (strict past)
and itself (bidirectional within block) via a block-causal attention mask,
and is initialised with the Mamba states from context block `N-1`.
When time conditioning is enabled, every denoiser layer receives per-timestep
modulation via adaptive layer-norm (adaLN-single, PixArt-alpha style).
When bidirectional Mamba is enabled, each Mamba layer in the denoiser runs the
SSM forward *and* backward (reversed sequence, zero initial states) and
averages the two outputs.  This lets the denoiser attend to both past and
future tokens within each block without adding a second set of Mamba weights.

### Key constraint

`--tt-diffusion-block-size` **must equal** `--mamba-chunk-size`. The Mamba
SSM processes sequences in fixed-size chunks; aligning block boundaries with
chunk boundaries allows exact state extraction at each block transition.

## File layout

```
megatron/diffusion/two_tower/
├── __init__.py              # Public API: TwoTowerMambaModel, create_block_causal_mask
├── README.md                # This file
├── arguments.py             # --tt-diffusion-* CLI argument definitions
├── builder.py               # two_tower_mamba_builder() — model construction from args
├── diffusion_process.py     # DiffusionProcess ABC, MaskDiffusionProcess
├── inference_engine.py      # DiffusionEngine — wraps generate() and forward_for_likelihood()
├── layer_utils.py           # Low-level Mamba layer forwards with explicit state I/O
├── mamba_model.py           # TwoTowerMambaModel, create_block_causal_mask
└── time_conditioning.py     # TimestepEmbedder, modulate(), get_modulation_params()

tools/
└── run_diffusion_text_generation_server.py  # Flask server entry point for inference

pretrain_mamba_tt_diffusion.py   # Training entry point (thin wrapper over pretrain_mamba.py)

tests/unit_tests/diffusion/two_tower/
├── test_block_causal_mask.py        # Element-wise mask correctness (CPU)
├── test_diffusion_process.py        # Corruption, loss, sampling logic (CPU)
├── test_time_conditioning.py        # TimestepEmbedder, modulate, modulation params (CPU)
└── test_two_tower_mamba_model.py    # Integration tests: training + inference (GPU)
```

## Quick start

### Training (8 GPUs, local)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc-per-node=8 \
    pretrain_mamba_tt_diffusion.py \
    --tt-diffusion \
    --tt-diffusion-block-size 128 \
    --tt-diffusion-mask-token-id 3 \
    --tt-diffusion-time-conditioning \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --hybrid-layer-pattern "M*EM*EM*" \
    --mamba-num-heads 16 \
    --mamba-head-dim 64 \
    --mamba-chunk-size 128 \
    --num-layers 8 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 2 \
    --ffn-hidden-size 2048 \
    --kv-channels 64 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-iters 1000 \
    --lr 1e-3 \
    --min-lr 1e-5 \
    --lr-decay-style WSD \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16 \
    --attention-backend flash \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type none \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --sequence-parallel \
    --use-mcore-models \
    --use-distributed-optimizer \
    --tokenizer-type TikTokenizer \
    --tokenizer-model /path/to/tokenizer.json \
    --data-path /path/to/data \
    --save /path/to/save \
    --log-interval 1
```

## CLI flags

Training flags (`--tt-diffusion-*`) are defined in `arguments.py`.

Inference/server flags are defined in `add_inference_args` inside
`tools/run_diffusion_text_generation_server.py`.

## Training flow

1. **Corrupt** — `MaskDiffusionProcess` stochastically masks tokens in the
   input sequence, sampling a per-example timestep *t* that is stored in
   `aux['t']`.
2. **Embed** — clean tokens go through the context embedding; noisy tokens
   through the denoiser embedding.
3. **Context tower** — full causal forward; caches Attention KV pairs and
   Mamba conv/SSM states at every block boundary.
4. **Time embedding** (when `--tt-diffusion-time-conditioning` is set) —
   the sampled timestep *t* is encoded via `TimestepEmbedder` and a global
   modulation MLP (`t_block`) to produce a conditioning vector
   `(B, 3 * hidden_size)`.  Each denoiser layer derives its own
   `(shift, scale, gate)` triple by adding a learned per-layer bias table
   to this vector.
5. **Block-causal mask** — built so denoiser block *N* attends to context
   blocks `0..N-1` and denoiser block *N* (self, bidirectional).
6. **Denoiser tower** — each attention layer concatenates context + denoiser
   KV and applies the block mask; each Mamba layer is initialised with
   the context cache from block `N-1`.  When time conditioning is active,
   hidden states are modulated after each layer norm (`shift`, `scale`)
   and the layer output is gated before the residual add (`gate`).
   When `--tt-diffusion-bidirectional-mamba` is set, each Mamba layer runs the
   SSM in both directions (forward with cached initial states, backward with
   zero states) and averages the outputs.
7. **Loss** — denoiser output is projected to logits; `MaskDiffusionProcess`
   computes per-token cross-entropy at masked positions only.  When
   `--tt-diffusion-context-ar-loss` is set, a next-token-prediction
   cross-entropy from the context tower's output head is added element-wise
   to the diffusion loss (using the dataloader's original shifted labels).
   Megatron's `loss_func` handles the final reduction.

## Known limitation: Mamba state extraction is not differentiable

The custom forwards in `layer_utils.py` slice into internal Triton kernel
outputs to extract conv and SSM states at block boundaries.  Making this
differentiable would require custom Triton backward kernels or a surrogate
gradient strategy, and is left for future work.

Because Mamba states must be detached, attention KV caches are also detached
for consistency.  Even when `--tt-diffusion-no-freeze-context` is set, the
following are **detached** before being passed to the denoiser:

- **Mamba states** (conv + SSM) at each block boundary
- **Attention KV caches** at each layer (could be kept attached, but detached to match)

Since all communication between the context and denoiser towers goes through
these detached caches, no gradient from the denoiser's diffusion loss reaches
any context-tower parameter (including the context embedding).


## Inference / Generation

The model supports two inference modes:

### Block-wise mask diffusion (default)

`model.generate_diffusion()` iterates over blocks:

1. Build per-request context caches via token-packed prefill (`_prefill`).
2. For each block: initialise with mask tokens → denoise via
   `MaskDiffusionProcess.sample_block` (which calls
   `_run_denoiser_step_batched` repeatedly) → extend the batched context
   cache (`_extend_context_cache_batched`).
3. Return the per-request output sequences.

### Single-tower AR (via `--load-single-tower`)

When a single-tower checkpoint is loaded via `load_from_single_tower()`, the
model sets `_single_tower_mode = True` and the engine delegates to
`generate_ar()`, which performs standard autoregressive decoding through the
context tower only (one token at a time, greedy or sampling).

### Loglikelihood evaluation

`model.forward_for_likelihood(input_ids)` runs the context tower and returns
next-token logits `(B, S, V)` for computing log-probabilities — used by
lm-eval-harness for perplexity-style benchmarks.

## Serving

Launch the text generation server with confidence unmasking:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 \
    -m tools.run_diffusion_text_generation_server \
    --tt-diffusion-model-provider mamba_two_tower \
    --tt-diffusion-steps-per-block 32 \
    --tt-diffusion-sampling-strategy confidence_unmasking \
    --use-checkpoint-args \
    --load /path/to/checkpoint \
    --bf16 \
    --port 5000 \
    --temperature 0.0 \
    --top_k 1
```

Generate text (temperature and top_k are set server-side):

```bash
curl -s http://localhost:5000/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["The capital of France is"],
    "tokens_to_generate": 128
  }' | python -m json.tool
```

## Tests

Test files live in `tests/unit_tests/diffusion/two_tower/`:

- `test_block_causal_mask.py` — element-wise mask correctness (CPU)
- `test_diffusion_process.py` — corruption, loss, sampling logic (CPU)
- `test_time_conditioning.py` — TimestepEmbedder, modulate, modulation params (CPU)
- `test_two_tower_mamba_model.py` — integration tests: training + inference (GPU)
