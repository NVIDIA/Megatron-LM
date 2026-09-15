# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The kernel registry must cover every kernel-bearing file in the tree (CPU only).

This is the in-repo half of the "every kernel has a determinism test" invariant; the CI
half (``tools/check_kernel_determinism_coverage.py``) checks the files a PR changes. Both
use the same manifest and the same notion of "kernel-bearing".
"""

import ast
from pathlib import Path

import pytest

from tests.unit_tests.determinism.kernels import manifest
from tools import check_kernel_determinism_coverage as coverage

REPO_ROOT = Path(__file__).resolve().parents[4]


def _kernel_bearing_files():
    files = []
    for path in sorted((REPO_ROOT / "megatron").rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        if coverage.is_kernel_bearing(rel, manifest, REPO_ROOT):
            files.append(rel)
    return files


def test_entry_names_are_unique():
    names = [entry.name for entry in manifest.KERNELS]
    assert len(names) == len(
        set(names)
    ), f"duplicate names: {sorted(set(n for n in names if names.count(n) > 1))}"


@pytest.mark.parametrize("entry", manifest.KERNELS, ids=lambda e: e.name)
def test_entry_paths_exist(entry):
    for source in entry.sources:
        assert (REPO_ROOT / source).is_file(), f"{entry.name}: source {source} does not exist"
    for test in entry.tests:
        assert (REPO_ROOT / test).is_file(), f"{entry.name}: test {test} does not exist"
        assert test.startswith("tests/unit_tests/"), f"{entry.name}: {test} must be a unit test"


@pytest.mark.parametrize("entry", manifest.KERNELS, ids=lambda e: e.name)
def test_entry_has_test_or_exemption(entry):
    assert (
        entry.tests or entry.exempt_reason
    ), f"{entry.name}: register a bit-exact test or an explicit exempt_reason"
    assert not (
        entry.tests and entry.exempt_reason
    ), f"{entry.name}: an entry is either tested or exempt, not both"


def _imported_modules(test_file: Path) -> set:
    """Dotted module paths imported anywhere in ``test_file`` (module or function level)."""
    tree = ast.parse(test_file.read_text())
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _module_of(source: str) -> str:
    path = Path(source)
    if path.name == "__init__.py":
        path = path.parent
    return path.with_suffix("").as_posix().replace("/", ".")


@pytest.mark.parametrize("entry", manifest.KERNELS, ids=lambda e: e.name)
def test_registered_test_imports_kernel(entry):
    """A kernel-suite test must import at least one of the modules it claims to cover.

    Model-level or pre-existing tests cover kernels indirectly and are not held to this, nor
    are ``kind="dispatch"`` entries: their sources call external kernels that the listed
    tests replay by name (see the entry's ``notes``).
    """
    if entry.kind == "dispatch":
        return
    sources = {_module_of(src) for src in entry.sources if src.endswith(".py")}
    for test in entry.tests:
        if not test.startswith("tests/unit_tests/determinism/kernels/") or not sources:
            continue
        imported = _imported_modules(REPO_ROOT / test)
        assert any(
            module == src or module.startswith(src + ".") for module in imported for src in sources
        ), f"{entry.name}: {test} imports none of {sorted(sources)}"


def test_every_kernel_bearing_file_is_registered():
    unregistered = [path for path in _kernel_bearing_files() if not manifest.entries_for(path)]
    assert not unregistered, (
        "Kernel-bearing files without a manifest entry (add a KernelEntry with a bit-exact "
        "test, or an exempt_reason, in tests/unit_tests/determinism/kernels/manifest.py):\n  "
        + "\n  ".join(unregistered)
    )


def test_ci_tool_accepts_a_registered_kernel_change():
    entry = next(e for e in manifest.KERNELS if e.tests)
    files = [entry.sources[0], entry.tests[0]]
    assert coverage.check(files, manifest, labels=set(), repo_root=REPO_ROOT) == []


def test_ci_tool_requires_test_update_without_label():
    entry = next(e for e in manifest.KERNELS if e.tests)
    violations = coverage.check([entry.sources[0]], manifest, labels=set(), repo_root=REPO_ROOT)
    assert violations and "none of its determinism tests did" in violations[0]
    assert (
        coverage.check(
            [entry.sources[0]], manifest, labels={manifest.EXEMPT_LABEL}, repo_root=REPO_ROOT
        )
        == []
    )


def test_ci_tool_flags_unregistered_kernel_file(tmp_path):
    kernel = tmp_path / "megatron" / "core" / "new_kernel.py"
    kernel.parent.mkdir(parents=True)
    kernel.write_text("import triton\n\n@triton.jit\ndef k(x):\n    pass\n")
    violations = coverage.check(
        ["megatron/core/new_kernel.py"], manifest, labels=set(), repo_root=tmp_path
    )
    assert violations and "not registered" in violations[0]


def test_dispatch_entries_name_their_kernels():
    for entry in manifest.KERNELS:
        if entry.kind == "dispatch":
            assert entry.notes or entry.exempt_reason, f"{entry.name}: say which kernels it calls"


# Representative call sites for every external kernel family the policy covers. Each must be
# detected on its own; a bare import, an isinstance() check, a mention in a comment or a
# docstring, or a plain ``apply_rotary_pos_emb`` caller must not be.
EXTERNAL_DISPATCH_SNIPPETS = {
    "transformer_engine_torch": (
        "import transformer_engine_torch as tex\n\ndef f(x, w):\n    return tex.rmsnorm_fwd(x, w, 1e-5)\n"
    ),
    "te_collective": "def f(x, tp):\n    return gather_along_first_dim(x, tp)\n",
    "te_fp8_cast": "def f(params):\n    cast_master_weights_to_fp8(params)\n",
    "te_cuda_graphs": "def f(fn):\n    return make_graphed_callables(fn, ())\n",
    "fused_rope": "def f(t, freqs):\n    return fused_apply_rotary_pos_emb(t, freqs)\n",
    "flash_attn_rope": "def f(t, cos, sin):\n    return apply_rotary_emb_flash(t, cos, sin)\n",
    "causal_conv1d": "def f(x, w):\n    return causal_conv1d_fn(x, w, None, activation='silu')\n",
    "mamba_ssm": (
        "def f(x, dt, A, B, C):\n    return mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size=256)\n"
    ),
    "mamba_decode": "def f(s, x, dt, A, B, C, D):\n    return selective_state_update(s, x, dt, A, B, C, D)\n",
    "fla": "def f(q, k, v, g, beta):\n    return chunk_gated_delta_rule(q, k, v, g, beta)\n",
    "fla_l2norm": "def f(q):\n    return l2_norm(q)\n",
    "deep_ep": "def f(self, x, handle):\n    return self.buffer.dispatch(x, handle=handle)\n",
    "deep_ep_fused": "def f(x, group):\n    return fused_dispatch(x, group)\n",
    "cutile": "import cuda.tile as ct\n\n@ct.kernel\ndef k(x):\n    pass\n",
    "flashinfer": "def f(p, k):\n    return flashinfer.sampling.top_k_sampling_from_probs(p, k)\n",
    "flashinfer_mxfp8": "def f(a, b):\n    return mm_mxfp8(a, b)\n",
    "apex_multi_tensor": (
        "def f(buf, grads, s):\n    multi_tensor_applier(multi_tensor_scale_impl, buf, [grads, grads], s)\n"
    ),
}

NON_DISPATCH_SNIPPETS = {
    "bare_imports": (
        "import transformer_engine.pytorch as te\n"
        "import transformer_engine_torch as tex\n"
        "from causal_conv1d import causal_conv1d_fn\n"
        "from fla.ops.gated_delta_rule import chunk_gated_delta_rule\n"
        "import flashinfer\n"
    ),
    "isinstance_check": (
        "from transformer_engine.pytorch import Float8Tensor\n\ndef f(t):\n    return isinstance(t, Float8Tensor)\n"
    ),
    "docstring_and_comment": (
        '"""Eventually calls causal_conv1d_fn(x) and tex.rmsnorm_fwd(x)."""\n'
        "# selective_state_update(state, x) happens in the mixer\n"
        "X = 1\n"
    ),
    "plain_rope_caller": "def f(t, freqs):\n    return apply_rotary_pos_emb(t, freqs)\n",
    "class_only": "class Buffer:\n    def dispatch(self, x):\n        return x\n",
}


def _classify(tmp_path, text):
    path = tmp_path / "megatron" / "core" / "candidate.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return coverage.is_kernel_bearing("megatron/core/candidate.py", manifest, tmp_path)


@pytest.mark.parametrize("snippet", sorted(EXTERNAL_DISPATCH_SNIPPETS))
def test_external_dispatch_call_sites_are_detected(tmp_path, snippet):
    assert _classify(tmp_path, EXTERNAL_DISPATCH_SNIPPETS[snippet]), snippet


@pytest.mark.parametrize("snippet", sorted(NON_DISPATCH_SNIPPETS))
def test_non_dispatch_code_is_not_detected(tmp_path, snippet):
    assert not _classify(tmp_path, NON_DISPATCH_SNIPPETS[snippet]), snippet


def test_ci_tool_flags_unregistered_external_dispatch_file(tmp_path):
    """A PR that adds an external-kernel call site in a new file fails the gate."""
    module = tmp_path / "megatron" / "core" / "new_mixer.py"
    module.parent.mkdir(parents=True)
    module.write_text(EXTERNAL_DISPATCH_SNIPPETS["causal_conv1d"])
    violations = coverage.check(
        ["megatron/core/new_mixer.py"], manifest, labels=set(), repo_root=tmp_path
    )
    assert violations and "not registered" in violations[0]


@pytest.mark.parametrize(
    "path",
    [
        "megatron/core/ssm/mamba_mixer.py",
        "megatron/core/ssm/gated_delta_product.py",
        "megatron/core/models/common/embeddings/rope_utils.py",
        "megatron/core/transformer/multi_latent_attention.py",
        "megatron/core/fp8_utils.py",
        "megatron/core/inference/sampling/flashinfer_sampling.py",
    ],
)
def test_known_dispatch_files_stay_kernel_bearing(path):
    """Detector regressions on the real dispatch modules must be visible, not silent."""
    registered = manifest.entries_for(path)
    assert registered, f"{path} must stay registered"
    # Independently of the registration, the content patterns must still recognise them.
    text = (REPO_ROOT / path).read_text()
    assert coverage.matches_kernel_pattern(text, manifest.KERNEL_CONTENT_PATTERNS), path
