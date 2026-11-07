# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FlashAttention-4 must run with the quack-kernels and CuTe DSL it is installed with.

FA4 calls quack-kernels, whose version Megatron's lock sets. FA4 4.0.0b25 and earlier call
``quack.activation.sub_packed_f32x2``, which quack-kernels 0.6.5 removed, so their SM100 forward
cannot compile against it. pyproject.toml therefore locks FA4 together with quack-kernels. FA4
also shares FlashAttention 2's ``flash_attn`` package, so the copy that ``import flash_attn.cute``
loads can differ from the first ``flash-attn-4`` metadata on the path when the two are installed
in layered environments (a venv over a base image). Megatron gates on the copy that imports, and
the CI image installs the locked FA4 next to FA2 so that copy is the locked one.

FA4 must also work with the Transformer Engine Megatron locks. TE before 2.21 passes FA4 the window
(-1, 0) for causal masks, with -1 meaning unbounded; flash-attn-4 4.0.0b30 and later read -1 as a
literal offset, so TE's FA4 backend returns zeros for causal attention. The lock keeps FA4 below
4.0.0b30 until TE moves, and set_attention_backend refuses the pairing.
"""

import ast
import importlib
import importlib.metadata
import importlib.util
import math
import pathlib
import sys

import pytest
import torch

import megatron.core.transformer.attention as attention_module
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from tests.unit_tests.test_utilities import Utils

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


def _site(root, fa4_version=None, fa2=False, fa4_files=True):
    """A site-packages directory: FA2's ``flash_attn/__init__.py``, FA4's ``flash_attn/cute``, and
    flash-attn-4 metadata, each optional."""
    root.mkdir()
    if fa2:
        (root / "flash_attn").mkdir()
        (root / "flash_attn" / "__init__.py").write_text("")
    if fa4_files:
        (root / "flash_attn" / "cute").mkdir(parents=True)
        (root / "flash_attn" / "cute" / "__init__.py").write_text("")
    if fa4_version is not None:
        info = root / f"flash_attn_4-{fa4_version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Name: flash-attn-4\nVersion: {fa4_version}\n")
    return str(root)


@pytest.mark.parametrize(
    "layout, imported",
    [
        # A venv's FA4 over a base image with FA2 and an older FA4: FA2's regular package makes
        # flash_attn.cute import from the base image, while the first metadata is the venv's.
        ("venv-over-fa2", "4.0.0b24"),
        # Without FA2, flash_attn is a namespace package and the first copy on the path imports.
        ("namespace", "4.0.0b37"),
        # An editable install owns no file next to its metadata; it is the only one installed.
        ("editable", "4.0.0b30"),
    ],
)
def test_fa4_version_is_read_from_the_copy_that_imports(tmp_path, monkeypatch, layout, imported):
    """Megatron gates FA4 on the version of the copy ``import flash_attn.cute`` loads."""
    if layout == "venv-over-fa2":
        path = [
            _site(tmp_path / "venv", "4.0.0b37"),
            _site(tmp_path / "base", "4.0.0b24", fa2=True),
        ]
    elif layout == "namespace":
        path = [_site(tmp_path / "venv", "4.0.0b37"), _site(tmp_path / "base", "4.0.0b24")]
    else:
        path = [_site(tmp_path / "meta", "4.0.0b30", fa4_files=False), _site(tmp_path / "src")]
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] == "flash_attn"}
    monkeypatch.setattr(sys, "path", path)
    importlib.invalidate_caches()
    try:
        assert attention_module._imported_fa4_version() == imported
    finally:
        for k in [k for k in sys.modules if k.split(".")[0] == "flash_attn"]:
            del sys.modules[k]
        sys.modules.update(saved)


# The environment checks below guard the CI images, which install the locked FA4. They fail, rather
# than skip, wherever an FA4 older than Megatron's floor is installed.


def test_megatron_gates_on_the_fa4_that_imports():
    """Megatron gates on the FA4 that ``flash_attn.cute`` loads, and that FA4 meets the floor."""
    version, cute_dir = _imported_fa4()
    assert attention_module._FA4_VERSION == version, (
        f"Megatron gates FlashAttention-4 on flash-attn-4 {attention_module._FA4_VERSION}, but "
        f"`import flash_attn.cute` loads {version} from {cute_dir}"
    )
    assert attention_module.HAVE_FA4, (
        f"{attention_module._FA4_UNAVAILABLE} (loaded from {cute_dir}): in the CI image, the "
        "locked FA4 must replace the base image's copy next to FlashAttention 2"
    )


def test_fa4_quack_references_resolve():
    """Every quack attribute the imported FA4 uses exists in the importable quack-kernels.

    The scan counts every reference as required, including any a future FA4 might guard with a
    fallback; such a release would need an exception here.
    """
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
    # The Blackwell CI image ships FA4, so a missing or too-old one is a failure, not a skip.
    assert attention_module.HAVE_FA4, attention_module._FA4_UNAVAILABLE
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


_NVTE_ATTENTION_ENV = (
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_FLASH_ATTN_V2",
    "NVTE_FLASH_ATTN_V3",
    "NVTE_FLASH_ATTN_V4",
)


def _causal_reference(q, k, v):
    """fp32 causal attention for sbhd q/k/v; returns [s, b, heads * head_dim]."""
    s, b, heads, dim = q.shape
    scores = torch.einsum("sbhd,tbhd->bhst", q.float(), k.float()) * dim**-0.5
    mask = torch.ones(s, s, dtype=torch.bool, device=q.device).triu(1)
    probs = scores.masked_fill(mask, float("-inf")).softmax(-1)
    return torch.einsum("bhst,tbhd->sbhd", probs, v.float()).reshape(s, b, heads * dim)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("batch_invariant", [False, True], ids=["default", "batch_invariant"])
@pytest.mark.parametrize("head_dim", [32, 128])
def test_te_flash_attention_4_causal_matches_reference(monkeypatch, batch_invariant, head_dim):
    """Transformer Engine's FlashAttention-4 backend computes causal attention.

    With TE before 2.21 and flash-attn-4 4.0.0b30 or later, every row comes out zero. The
    batch-invariant configuration is the one Megatron's training path uses next to the dynamic
    engine's FA4 kernel.
    """
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Transformer Engine runs FlashAttention-4 on SM90 and newer")
    import transformer_engine
    from transformer_engine.pytorch.attention.dot_product_attention import (
        dot_product_attention as te_dpa,
    )

    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        set_batch_invariant_mode,
        te_supports_batch_invariant_attention,
    )

    # The CI images ship FA4, so a missing or too-old one is a failure, not a skip. This test uses
    # only what main already has, so it also runs as a negative control against main.
    assert attention_module.HAVE_FA4, "flash-attn-4 is not usable in this environment"
    if batch_invariant and not te_supports_batch_invariant_attention():
        pytest.skip("this Transformer Engine has no batch-invariant attention")

    backends = te_dpa._attention_backends
    Utils.initialize_model_parallel(1, 1)  # Clears the NVTE attention environment.
    try:
        # Only TE's FlashAttention-4 backend, as Megatron configures attention_backend=flash with
        # flash_attention_version=4. TE reads these when it selects a backend and caches its
        # choice, so set them first and make it select again.
        for name, value in zip(_NVTE_ATTENTION_ENV, ("1", "0", "0", "0", "0", "1")):
            monkeypatch.setenv(name, value)
        backends["attention_params"] = None
        backends["backend_selection_requires_update"] = True
        heads, seq, batch = 4, 24, 2
        config = TransformerConfig(
            num_layers=1,
            hidden_size=heads * head_dim,
            num_attention_heads=heads,
            batch_invariant_mode=batch_invariant,
            flash_attention_version=4,
            attention_backend=AttnBackend.flash,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        attn = TEDotProductAttention(
            config=config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
        )
        # q, k and v as Megatron's training forward passes them: views of the fused projection.
        g = torch.Generator(device="cuda").manual_seed(0)
        mixed = torch.randn(seq, batch, heads, 3 * head_dim, generator=g, device="cuda").bfloat16()
        q, k, v = mixed.split(head_dim, dim=-1)
        with torch.no_grad(), set_batch_invariant_mode(batch_invariant):
            out = attn(q, k, v, attention_mask=None, attn_mask_type=AttnMaskType.causal)
        backend = backends.get("flash_attention_backend")
        assert backends.get("use_flash_attention") and backend is not None and backend.major == 4, (
            f"Transformer Engine did not run FlashAttention-4 (flash backend {backend}, "
            f"fused {backends.get('use_fused_attention')}, "
            f"unfused {backends.get('use_unfused_attention')})"
        )
        reference = _causal_reference(q, k, v)
        error = ((out.float() - reference).norm() / reference.norm()).item()
        assert error <= 2e-2, (
            f"Transformer Engine {transformer_engine.__version__} on flash-attn-4 {backend} "
            f"differs from fp32 causal attention by rel {error:.3g}"
        )
    finally:
        backends["backend_selection_requires_update"] = True
        Utils.destroy_model_parallel()
