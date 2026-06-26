"""Shared setup + model builders for the Gemma 4 E4B example scripts.

Import THIS FIRST in any container-side Gemma4 script (it sets up sys.path for the
Megatron-LM source tree and works around a container<->MLM incompatibility before
megatron is imported). It also provides the three helpers the conversion / parity
scripts share: ``_init_distributed``, ``_make_config`` and ``_build_model``.

Container<->MLM workaround: the container's ``nvidia_resiliency_ext`` ships the
async-ckpt modules but exposes no ``__version__``, so MLM's
``dist_checkpointing/strategies/nvrx.py:is_nvrx_min_version()`` crashes at import
(AttributeError). We populate ``__version__`` before megatron is imported. These
single-device scripts never use async checkpointing, so the value only needs to
satisfy the ``>=0.6.0`` guard.

Run scripts with the CONTAINER python (this module first).
"""
import functools
import os
import sys

# Megatron-LM root is three levels up from this file: examples/gemma4/ -> examples/ -> <repo>.
MLM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if MLM not in sys.path:
    sys.path.insert(0, MLM)

try:
    import nvidia_resiliency_ext as _nvrx  # noqa: E402
    if not hasattr(_nvrx, "__version__"):
        v = None
        try:
            from importlib.metadata import version
            v = version("nvidia_resiliency_ext")
        except Exception:
            v = None
        # Guard requires >= 0.6.0; fall back to that if metadata missing/older.
        try:
            from packaging.version import Version as _V
            if v is None or _V(v) < _V("0.6.0"):
                v = "0.6.0"
        except Exception:
            v = v or "0.6.0"
        _nvrx.__version__ = v
except Exception:
    # nvrx absent entirely -> MLM's HAVE_NVRX=False path handles it fine.
    pass

import torch  # noqa: E402

# Gemma4 MLP/PLE use gelu_pytorch_tanh == F.gelu(approximate="tanh"). Use the full
# sqrt(2/pi) constant (not fast_gelu's truncated 0.7978845608) for bitwise fidelity.
GELU_TANH = functools.partial(torch.nn.functional.gelu, approximate="tanh")


def _init_distributed():
    """Single-process (DDP=TP=PP=CP=SP=1) init for the example scripts."""
    import megatron.core.parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12399")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    ps.initialize_model_parallel(1, 1)
    # Required before building VocabParallelEmbedding / parallel linears (adds the
    # 'model-parallel-rng' cuda rng state used by _initialize_affine_weight_gpu).
    model_parallel_cuda_manual_seed(123)


def _make_config(num_layers=42, hidden=2560, ffn=10240):
    """Real Gemma 4 E4B Megatron config (local-spec bitwise target)."""
    from megatron.core.transformer.gemma4_config import Gemma4TransformerConfig

    return Gemma4TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        ffn_hidden_size=ffn,
        num_attention_heads=8,
        num_query_groups=2,
        layernorm_epsilon=1e-6,
        gated_linear_unit=True,
        activation_func=GELU_TANH,
        add_bias_linear=False,
        qk_layernorm=True,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_softmax_in_fp32=True,
        masked_softmax_fusion=False,
        pipeline_dtype=torch.bfloat16,
    )


def _build_model(spec_fn, config, vocab=262144):
    """Build a bf16/cuda Gemma4Model from a layer-spec factory."""
    from megatron.core.models.gemma4.gemma4_model import Gemma4Model

    spec = spec_fn(config)
    model = Gemma4Model(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=vocab,
        max_sequence_length=512,
    )
    return model.bfloat16().cuda()
