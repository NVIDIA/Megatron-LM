# Simplified sparse attention with CuTe

The CuTe backend implements the learned-Q/K simplified DSA-GQA formulation. An
indexer projects the attention input into separate query and key vectors, scores
causally visible keys, and selects the top-k keys for each query. Main attention
computes softmax attention over that selected support. An auxiliary KL objective
trains the indexer against the detached, head-averaged main-attention probabilities
on the same support.

MCore owns the trainable projections, model construction, checkpoint state and
auxiliary-loss attachment. The optional `simplified_sparse_attention` package owns
the local CuTe compute kernels and their autograd implementation. The integration
requires package version `0.1.0.dev2` or later. The adapter
converts SBHD or packed THD inputs into the package's token-major views and supplies
causal, document-local row bounds. Selecting `dsa_gqa_backend="cute"` explicitly
requires that backend; missing dependencies or unsupported inputs raise an error rather than
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
| Layout | Ordinary SBHD with batch size 1, or packed THD |
| Parallelism | Tensor-parallel size 1; fixed context-parallel groups using all-gather and zigzag layout |
| Main Q/K/V | BF16, head dimension 256; 16, 32 or 96 query heads; one K/V head |
| Indexer Q/K | BF16, one head with dimension 128; separate learned Q/K projections |
| Top-k | 512, 1024 or 2048 |
| Masking and scoring | Causal self-attention, NoPE; no indexer ReLU, activation rotation or attention dropout |
| Training | Joint main-attention and sparse indexer-loss training |
| Unsupported modes | Runtime changes to the context-parallel group, decode caching, dense indexer warmup, indexer-only training, shared routing, CUDA Graph capture and deterministic execution |

For SBHD, each rank supplies Q `[S_local, 1, H, 256]` and K/V
`[S_local, 1, 1, 256]`. Packed THD omits the singleton batch dimension, with
`S_local` counting physical token rows across local document pieces. The indexer
projects each local row to a 128-dimensional query and key. Under CP, the adapter
keeps queries local and supplies globally ordered K/V and indexer keys to the
package. It returns core-attention output `[S_local, 1, H * 256]` and an FP32
scalar auxiliary loss; the outer attention module restores the packed output
layout. The local package does not perform distributed communication.

## Context parallelism and packed sequences

Queries remain local to each CP rank. MCore gathers main K/V and indexer K, then
restores global token order from the rank-major zigzag layout. The local kernels
use each query's global position and document boundaries to select visible keys.
Gather backward sums key-gradient contributions into their original owners with
reduce-scatter. Forward communication and backward reduction preserve the BF16
attention tensor dtype.

Packed inputs must supply shared global Q/K physical cumulative boundaries that
start at zero and cover every physical row, including alignment padding. Each
document's physical length must be divisible by twice the CP size when CP is
enabled. `PackedSeqParams.real_token_mask_q` independently identifies local real
query rows; padded query rows contribute neither attention output nor auxiliary
loss. Producers must retain logical lengths until they build this mask, or pass
an existing mask through. Physical boundaries alone cannot recover padding
validity. The sequence-packing scheduler and fixed-length pretraining path supply
this metadata. Other producers, including legacy SFT inputs, must likewise retain
logical lengths or supply an explicit validity mask.

In mean-loss mode, CP ranks share the global real-query count as the auxiliary
denominator. Per-token-loss mode retains a token sum. This uses MCore's existing
pipeline auxiliary-loss scaling and does not change the training objective when
the same sequence is partitioned across CP ranks.
