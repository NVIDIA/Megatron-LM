# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FlashAttention-4 must run with the quack-kernels and CuTe DSL it is installed with.

FA4 calls quack-kernels, whose version Megatron's lock sets. FA4 4.0.0b25 and earlier call
``quack.activation.sub_packed_f32x2``, which quack-kernels 0.6.5 removed, so their SM100 forward
cannot compile against it. pyproject.toml therefore locks FA4 together with quack-kernels. FA4
also shares FlashAttention 2's ``flash_attn`` package, so the copy that ``import flash_attn.cute``
loads can differ from the first ``flash-attn-4`` metadata on the path when the two are installed
in layered environments (a venv over a base image); the CI image installs the locked FA4 next to
FA2 for that reason.
"""

import ast
import importlib
import importlib.metadata
import importlib.util
import math
import pathlib

import pytest
import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp

# The SM100 forward runs only on Blackwell; the environment checks belong on both CI images.
pytestmark = pytest.mark.launch_on_gb200


def _imported_fa4():
    """(version, flash_attn/cute directory) of the FA4 that ``import flash_attn.cute`` loads."""
    try:
        importlib.metadata.distribution("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("flash-attn-4 is not installed")
    spec = importlib.util.find_spec("flash_attn.cute")
    assert spec is not None, "flash-attn-4 is installed, but flash_attn.cute is not importable"
    origin = pathlib.Path(spec.origin).resolve()
    for dist in importlib.metadata.distributions(name="flash-attn-4"):
        if pathlib.Path(dist.locate_file("flash_attn/cute/__init__.py")).resolve() == origin:
            return dist.version, origin.parent
    pytest.skip(f"no installed flash-attn-4 distribution owns {origin} (e.g. an editable install)")


def _quack_references(cute_dir):
    """{(quack module, attribute): ['file:line', ...]} for the quack attributes FA4's sources use.

    Covers ``from quack.<module> import <name>``, ``<module>.<name>`` after ``from quack import
    <module>``, and ``quack.<module>.<name>`` after ``import quack.<module>``.
    """
    refs = {}
    for path in sorted(cute_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        where = path.relative_to(cute_dir.parent.parent)
        add = lambda key, node: refs.setdefault(key, {}).setdefault(f"{where}:{node.lineno}")
        modules = {}  # local name -> the quack module it is bound to
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "quack":
                        modules[alias.asname or "quack"] = alias.name if alias.asname else "quack"
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                if node.module.split(".")[0] != "quack":
                    continue
                for alias in node.names:
                    add((node.module, alias.name), node)
                    if node.module == "quack":
                        modules[alias.asname or alias.name] = f"quack.{alias.name}"
        for node in ast.walk(tree):
            chain, base = [], node
            while isinstance(base, ast.Attribute):
                chain.insert(0, base.attr)
                base = base.value
            if not chain or not isinstance(base, ast.Name) or base.id not in modules:
                continue
            module = modules[base.id]
            if module == "quack":
                if len(chain) < 2:
                    continue
                module, chain = f"quack.{chain[0]}", chain[1:]
            add((module, chain[0]), node)
    return {key: list(sites) for key, sites in refs.items()}


def _quack_install():
    spec = importlib.util.find_spec("quack")
    if spec is None:
        return "no importable quack-kernels"
    return f"quack-kernels {importlib.metadata.version('quack-kernels')} at {spec.origin}"


def _resolves(module, name):
    try:
        if hasattr(importlib.import_module(module), name):
            return True
        importlib.import_module(f"{module}.{name}")
        return True
    except ImportError:
        return False


def test_megatron_gates_on_the_fa4_that_imports():
    """The flash-attn-4 version Megatron gates on is that of the copy flash_attn.cute loads."""
    version, cute_dir = _imported_fa4()
    gated = importlib.metadata.version("flash-attn-4")
    assert gated == version, (
        f"Megatron gates FlashAttention-4 on flash-attn-4 {gated}, but `import flash_attn.cute` "
        f"loads {version} from {cute_dir}: an FA4 in another site-packages directory, next to "
        "FlashAttention 2's flash_attn/__init__.py, shadows the installed one"
    )
    assert attention_module.HAVE_FA4, (
        f"flash-attn-4 {version} is installed, but below Megatron's floor "
        f"{attention_module._MIN_FA4_VERSION}"
    )


def test_fa4_quack_references_resolve():
    """Every quack attribute the imported FA4 uses exists in the importable quack-kernels."""
    version, cute_dir = _imported_fa4()
    refs = _quack_references(cute_dir)
    assert refs, f"found no quack references in flash-attn-4 {version} at {cute_dir}"
    missing = [
        f"{module}.{name} ({', '.join(sites[:3])})"
        for (module, name), sites in sorted(refs.items())
        if not _resolves(module, name)
    ]
    assert not missing, (
        f"{len(missing)} of {len(refs)} quack attributes that flash-attn-4 {version} uses do not "
        f"exist in {_quack_install()}: {missing}"
    )


HEADS, KV_HEADS, HEAD_DIM, BLOCK = 8, 2, 128, 256


def _paged_reference(q, k_cache, v_cache, cu_seqlens_q, seqlens_k, block_table, scale):
    """FP32 causal (bottom-right) attention of each request's queries over its paged KV."""
    out = torch.empty(q.shape[0], HEADS, HEAD_DIM, device=q.device)
    for r, length in enumerate(seqlens_k.tolist()):
        s0, s1 = cu_seqlens_q[r].item(), cu_seqlens_q[r + 1].item()
        blocks = block_table[r, : math.ceil(length / BLOCK)].long()
        keys = (
            k_cache[blocks].flatten(0, 1)[:length].float().repeat_interleave(HEADS // KV_HEADS, 1)
        )
        vals = (
            v_cache[blocks].flatten(0, 1)[:length].float().repeat_interleave(HEADS // KV_HEADS, 1)
        )
        scores = torch.einsum("qhd,khd->hqk", q[s0:s1].float(), keys) * scale
        i = torch.arange(s1 - s0, device=q.device)[:, None]
        j = torch.arange(length, device=q.device)[None, :]
        scores = scores.masked_fill(j > i + length - (s1 - s0), float("-inf"))
        out[s0:s1] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals)
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("is_decode_only", [False, True], ids=["prefill", "decode"])
def test_fa4_paged_attention_on_sm100(is_decode_only):
    """FA4's SM100 forward compiles and matches a reference through flash_decode_and_prefill.

    Head dim 128 enables FA4's emulated exp2 on SM100, the path that calls quack-kernels.
    """
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("FA4's SM100 forward needs a Blackwell (SM100) GPU")
    if not attention_module.HAVE_FA4:
        pytest.skip(f"needs flash-attn-4 >= {attention_module._MIN_FA4_VERSION}")
    config = TransformerConfig(
        num_layers=1,
        hidden_size=HEADS * HEAD_DIM,
        num_attention_heads=HEADS,
        num_query_groups=KV_HEADS,
        flash_attention_version=4,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    submodules = SelfAttentionSubmodules(
        linear_qkv=IdentityOp, core_attention=IdentityOp, linear_proj=IdentityOp
    )
    attn = SelfAttention(
        config,
        submodules,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        pg_collection=ProcessGroupCollection(tp=None, cp=None),
    ).eval()

    # A fresh 64-token prompt and a 300-token request spanning two blocks: chunked prefill of
    # its last 40 tokens, or one decode token per request.
    seqlens_k = torch.tensor([64, 300], dtype=torch.int32, device="cuda")
    q_lens = [1, 1] if is_decode_only else [64, 40]
    cu_seqlens_q = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, 64, 364], dtype=torch.int32, device="cuda")
    block_table = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(0)
    k_cache, v_cache = (
        torch.randn(4, BLOCK, KV_HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()
        for _ in range(2)
    )
    q = torch.randn(sum(q_lens), 1, HEADS, HEAD_DIM, generator=g, device="cuda").bfloat16()

    with torch.no_grad():
        out = attn.flash_decode_and_prefill(
            q,
            k_cache,
            v_cache,
            max_seqlen_q=max(q_lens),
            max_seqlen_k=300,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqlens_k=seqlens_k,
            block_table=block_table,
            is_decode_only=is_decode_only,
        )
    expected = _paged_reference(
        q.squeeze(1), k_cache, v_cache, cu_seqlens_q, seqlens_k, block_table, HEAD_DIM**-0.5
    )
    torch.testing.assert_close(out.squeeze(1).float(), expected, atol=2e-2, rtol=2e-2)
