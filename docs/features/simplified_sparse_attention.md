# Simplified sparse attention with CuTe

The CuTe backend implements the learned-Q/K simplified DSA-GQA formulation. An
indexer projects the attention input into separate query and key vectors, scores
causally visible keys, and selects the top-k keys for each query. Main attention
computes softmax attention over that selected support. An auxiliary KL objective
trains the indexer against the detached, head-averaged main-attention probabilities
on the same support.

MCore owns the trainable projections, model construction, checkpoint state and
auxiliary-loss attachment. The optional `simplified_sparse_attention` package owns
the local CuTe compute kernels and their autograd implementation; version
`0.1.0.dev2` or later supplies configurable auxiliary-loss scaling. The adapter
converts ordinary SBHD inputs into the package's token-major views and supplies
causal row bounds. Selecting `dsa_gqa_backend="cute"` explicitly requires that
backend; missing dependencies or unsupported inputs raise an error rather than
silently selecting a different implementation.

## Training semantics

The indexer consumes detached attention inputs: the auxiliary objective updates
the indexer projections without updating the backbone. The main loss propagates
through sparse attention to main Q/K/V and their projections; discrete top-k
selection is not differentiated. The indexer projections use BF16 independently
of the precision selected for main-projection GEMMs.

The configured auxiliary coefficient is applied once to both the KL loss and its
indexer gradients. It may be any finite nonnegative value; zero disables this
training signal while preserving routing and main attention. The denominator
follows MCore's existing indexer-loss convention: average over query tokens, or
retain the token sum when `calculate_per_token_loss` is enabled. The backend
shares DSA-GQA's checkpoint parameter names and opt-in independent indexer
gradient clipping; it introduces no backend-specific checkpoint format.

## Supported integration

| Property | CuTe backend |
| --- | --- |
| GPU architecture | SM103 |
| Layout | Ordinary SBHD, batch size 1 |
| Parallelism | Tensor-parallel size 1; context-parallel size 1 |
| Main Q/K/V | BF16, head dimension 256; 16, 32 or 96 query heads; one K/V head |
| Indexer Q/K | BF16, one head with dimension 128; separate learned Q/K projections |
| Top-k | 512, 1024 or 2048 |
| Masking and scoring | Causal self-attention, NoPE; no indexer ReLU, activation rotation or attention dropout |
| Training | Joint main-attention and sparse indexer-loss training |
| Unsupported modes | Packed THD, context parallelism, decode caching, dense indexer warmup, indexer-only training, shared routing, CUDA Graph capture and deterministic execution |

MCore accepts Q `[S, 1, H, 256]`, K/V `[S, 1, 1, 256]`, and indexer Q/K
`[S, 1, 1, 128]`. The adapter passes token-major views to the package and returns
the ordinary core-attention output `[S, 1, H * 256]` plus an FP32 scalar auxiliary
loss. Support described here is the MCore integration contract; the local package
interface does not perform distributed communication.
