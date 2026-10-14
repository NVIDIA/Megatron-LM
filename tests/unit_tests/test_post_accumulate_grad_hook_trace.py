# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for CUDA post-accumulate hooks missed by coverage.py.

The CPU autograd engine invokes the hook on the calling thread, so a tracer
installed there is visible. These tests clear that tracer before backward, which
is the state the CUDA engine actually enters the hook in. The CUDA tests cover
the real engine path: dtypes, the FSDP hook shapes, coverage.py, and Testmon.
"""

import contextlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from tests.unit_tests.post_accumulate_trace import (
    _ORIGINAL,
    install_post_accumulate_trace,
    preserve_caller_trace,
)

install_post_accumulate_trace()

_CUDA_REASON = "CUDA autograd hook coverage requires a GPU"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _line_tracer(function_name: str, seen: list[int]):
    """Return a tracer that records line events from ``function_name``."""

    def tracer(frame, event, arg):
        del arg
        if event == "line" and frame.f_code.co_name == function_name:
            seen.append(frame.f_lineno)
        return tracer

    return tracer


def _call_tracer(function_names: set[str], seen: list[str]):
    """Return a tracer that records which of ``function_names`` executed a line."""

    def tracer(frame, event, arg):
        del arg
        if event == "line" and frame.f_code.co_name in function_names:
            seen.append(frame.f_code.co_name)
        return tracer

    return tracer


@contextlib.contextmanager
def _tracing(*function_names: str):
    """Install a line tracer for ``function_names`` and restore the previous one."""
    seen: list[str] = []
    previous = sys.gettrace()
    sys.settrace(_call_tracer(set(function_names), seen))
    try:
        yield seen
    finally:
        sys.settrace(previous)


def _cuda_device() -> torch.device:
    """Use the current CUDA device so a torchrun rank does not retarget the process."""
    return torch.device("cuda", torch.cuda.current_device())


def _sync() -> None:
    torch.cuda.synchronize()


def _requires_cuda(func):
    func = pytest.mark.launch_on_gb200(func)
    return pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON)(func)


def _skip_unless_primary_process() -> None:
    """Run subprocess GPU checks once when the suite is launched under torchrun."""
    rank = os.environ.get("RANK")
    local_rank = os.environ.get("LOCAL_RANK")
    if rank not in (None, "0") or local_rank not in (None, "0"):
        pytest.skip("CUDA coverage subprocess runs on rank 0")


def test_preserve_is_identity_without_a_tracer():
    previous = sys.gettrace()
    sys.settrace(None)
    try:

        def hook(parameter):
            del parameter
            return None

        assert preserve_caller_trace(hook) is hook
    finally:
        sys.settrace(previous)


def test_wrapper_restores_a_cleared_trace_and_propagates_errors():
    def tracer(frame, event, arg):
        del frame, event, arg
        return tracer

    def boom(parameter):
        del parameter
        raise RuntimeError("boom")

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        wrapped = preserve_caller_trace(boom)
    finally:
        sys.settrace(previous)

    sys.settrace(None)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            wrapped(torch.zeros(1))
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


def test_wrapper_leaves_an_existing_tracer_installed():
    def outer(frame, event, arg):
        del frame, event, arg
        return outer

    def inner(frame, event, arg):
        del frame, event, arg
        return inner

    def hook(parameter):
        del parameter
        return None

    previous = sys.gettrace()
    sys.settrace(outer)
    try:
        wrapped = preserve_caller_trace(hook)
    finally:
        sys.settrace(previous)

    sys.settrace(inner)
    try:
        assert wrapped(torch.zeros(1)) is None
        assert sys.gettrace() is inner
    finally:
        sys.settrace(previous)


def test_cpu_backward_hook_is_traced_after_the_caller_trace_is_cleared():
    seen = []
    weight = torch.nn.Parameter(torch.ones(2, 2))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        sys.settrace(None)
        (weight @ torch.ones(2, 1)).sum().backward()
        assert seen
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.skipif(not torch.cuda.is_available(), reason=_CUDA_REASON),
                pytest.mark.launch_on_gb200,
            ],
        ),
    ],
)
def test_tensor_subclass_hook_registration_does_not_recurse(device):
    class TracedParameter(torch.nn.Parameter):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            del types
            if kwargs is None:
                kwargs = {}
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)

    seen = []
    weight = TracedParameter(torch.ones(2, device=device))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        sys.settrace(None)
        weight.sum().backward()
        if device == "cuda":
            _sync()
        assert seen
    finally:
        sys.settrace(previous)


@_requires_cuda
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"]
)
def test_cuda_backward_hook_is_traced_while_the_caller_trace_stays_installed(dtype):
    seen = []
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, 2, device=device, dtype=dtype))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        (weight @ torch.ones(2, 1, device=device, dtype=dtype)).sum().backward()
        _sync()
        assert seen
    finally:
        sys.settrace(previous)


@_requires_cuda
def test_cuda_backward_hook_is_traced_after_the_caller_trace_is_cleared():
    """The wrapper keeps the tracer captured at registration, not the caller's."""
    seen = []
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, 2, device=device))

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        return None

    previous = sys.gettrace()
    sys.settrace(_line_tracer("user_hook", seen))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        sys.settrace(None)
        (weight @ torch.ones(2, 1, device=device)).sum().backward()
        _sync()
        assert seen
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


@_requires_cuda
def test_cuda_engine_invokes_an_unwrapped_hook_with_no_tracer():
    """The CUDA engine itself still enters the hook with no Python tracer."""
    entered = []
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, 2, device=device))

    def user_hook(parameter):
        entered.append(sys.gettrace())
        if parameter.grad is None:
            raise RuntimeError("grad missing")

    def tracer(frame, event, arg):
        del frame, event, arg
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        _ORIGINAL(weight, user_hook)
        (weight @ torch.ones(2, 1, device=device)).sum().backward()
        _sync()
    finally:
        sys.settrace(previous)
    assert entered == [None]


@_requires_cuda
def test_cuda_wrapper_restores_the_engine_thread_trace():
    """A later unwrapped hook on the same backward must not inherit the tracer."""
    entered = []
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, device=device))

    def wrapped_hook(parameter):
        del parameter
        entered.append(("wrapped", sys.gettrace() is not None))

    def raw_hook(parameter):
        del parameter
        entered.append(("raw", sys.gettrace() is not None))

    previous = sys.gettrace()
    sys.settrace(_call_tracer({"wrapped_hook"}, []))
    try:
        weight.register_post_accumulate_grad_hook(wrapped_hook)
        _ORIGINAL(weight, raw_hook)
        sys.settrace(None)
        weight.sum().backward()
        _sync()
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)
    # ``False`` is ``sys.gettrace() is not None``: the engine thread has no tracer
    # after the wrapped hook returns.
    assert entered == [("wrapped", True), ("raw", False)]


@_requires_cuda
def test_cuda_hook_exception_propagates_and_clears_the_reinstalled_trace():
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, device=device))

    def user_hook(parameter):
        del parameter
        raise RuntimeError("boom")

    previous = sys.gettrace()
    sys.settrace(_call_tracer({"user_hook"}, []))
    try:
        weight.register_post_accumulate_grad_hook(user_hook)
        sys.settrace(None)
        with pytest.raises(RuntimeError, match="boom"):
            weight.sum().backward()
        _sync()
        assert sys.gettrace() is None
    finally:
        sys.settrace(previous)


@_requires_cuda
def test_cuda_module_post_backward_callee_is_traced():
    """The issue repro: the hook body and the method it calls are both traced."""
    device = _cuda_device()

    class HookedLinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False, device=device)
            self.calls = 0
            self.linear.weight.register_post_accumulate_grad_hook(self._make_grad_hook())

        def _make_grad_hook(self):
            def grad_hook(parameter):
                if parameter.grad is None:
                    raise RuntimeError("grad missing")
                self.post_backward()

            return grad_hook

        def forward(self, inputs):
            return self.linear(inputs)

        def post_backward(self) -> None:
            self.calls += 1

    with _tracing("grad_hook", "post_backward") as seen:
        model = HookedLinear()
        model(torch.ones(1, 2, device=device)).sum().backward()
        _sync()
        assert model.calls == 1
    assert "grad_hook" in seen
    assert "post_backward" in seen


@_requires_cuda
def test_cuda_multiple_parameters_are_traced_in_one_backward():
    device = _cuda_device()
    first = torch.nn.Parameter(torch.ones(2, 2, device=device))
    second = torch.nn.Parameter(torch.ones(2, 2, device=device))
    calls = []

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        calls.append(parameter)

    with _tracing("user_hook") as seen:
        first.register_post_accumulate_grad_hook(user_hook)
        second.register_post_accumulate_grad_hook(user_hook)
        (first.sum() + second.sum()).backward()
        _sync()
    assert len(calls) == 2
    assert {call.data_ptr() for call in calls} == {first.data_ptr(), second.data_ptr()}
    assert seen.count("user_hook") >= 2


@_requires_cuda
def test_cuda_repeated_backward_accumulates_and_is_traced():
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, device=device))
    grad_sums = []

    def user_hook(parameter):
        grad_sums.append(float(parameter.grad.sum()))

    with _tracing("user_hook") as seen:
        weight.register_post_accumulate_grad_hook(user_hook)
        weight.sum().backward()
        _sync()
        weight.sum().backward()
        _sync()
        assert grad_sums == [2.0, 4.0]
        weight.grad = None
        weight.sum().backward()
        _sync()
    assert grad_sums == [2.0, 4.0, 2.0]
    assert seen.count("user_hook") >= 3


@_requires_cuda
def test_cuda_removed_hook_does_not_run_on_the_next_backward():
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, device=device))
    calls = []

    def user_hook(parameter):
        del parameter
        calls.append(1)

    with _tracing("user_hook") as seen:
        handle = weight.register_post_accumulate_grad_hook(user_hook)
        weight.sum().backward()
        _sync()
        traced_before_remove = len(seen)
        handle.remove()
        weight.grad = None
        weight.sum().backward()
        _sync()
    assert calls == [1]
    assert traced_before_remove >= 1
    assert len(seen) == traced_before_remove


@_requires_cuda
def test_cuda_hooks_on_one_parameter_run_in_registration_order_and_are_traced():
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, device=device))
    order = []

    def hook_a(parameter):
        del parameter
        order.append("a")

    def hook_b(parameter):
        del parameter
        order.append("b")

    with _tracing("hook_a", "hook_b") as seen:
        weight.register_post_accumulate_grad_hook(hook_a)
        weight.register_post_accumulate_grad_hook(hook_b)
        weight.sum().backward()
        _sync()
    assert order == ["a", "b"]
    assert "hook_a" in seen
    assert "hook_b" in seen


@_requires_cuda
def test_cuda_fsdp_v1_skip_hook_traces_both_branches():
    """Megatron FSDP v1 skips TE delayed weight gradients inside the hook."""
    device = _cuda_device()
    skipped = torch.nn.Parameter(torch.ones(2, device=device))
    kept = torch.nn.Parameter(torch.ones(2, device=device))
    skipped.skip_backward_post_hook = True
    processed = []

    def process(parameter):
        processed.append(parameter)

    def grad_hook(parameter):
        if getattr(parameter, "skip_backward_post_hook", False):
            return None
        process(parameter)
        return None

    with _tracing("grad_hook", "process") as seen:
        skipped.register_post_accumulate_grad_hook(grad_hook)
        kept.register_post_accumulate_grad_hook(grad_hook)
        (skipped.sum() + kept.sum()).backward()
        _sync()
    assert processed == [kept]
    assert seen.count("grad_hook") >= 2
    assert "process" in seen


@_requires_cuda
def test_cuda_parameter_countdown_traces_post_backward():
    """Experimental FSDP runs post_backward once the last parameter accumulates."""
    device = _cuda_device()
    weights = [torch.nn.Parameter(torch.ones(2, device=device)) for _ in range(2)]
    remaining = len(weights)
    posts = []

    def post_backward():
        posts.append("post")

    def grad_hook(parameter):
        nonlocal remaining
        if parameter.grad is None:
            raise RuntimeError("grad missing")
        remaining -= 1
        if remaining == 0:
            post_backward()

    with _tracing("grad_hook", "post_backward") as seen:
        for weight in weights:
            weight.register_post_accumulate_grad_hook(grad_hook)
        sum(weight.sum() for weight in weights).backward()
        _sync()
    assert posts == ["post"]
    assert seen.count("grad_hook") >= 2
    assert "post_backward" in seen


@_requires_cuda
def test_cuda_side_stream_backward_is_traced():
    device = _cuda_device()
    weight = torch.nn.Parameter(torch.ones(2, 2, device=device))
    stream = torch.cuda.Stream(device=device)

    def user_hook(parameter):
        if parameter.grad is None:
            raise RuntimeError("grad missing")

    with _tracing("user_hook") as seen:
        weight.register_post_accumulate_grad_hook(user_hook)
        with torch.cuda.stream(stream):
            (weight @ torch.ones(2, 1, device=device)).sum().backward()
        stream.synchronize()
    assert seen


_COVERAGE_PROBE = textwrap.dedent("""\
    import os
    import sys

    import torch
    from torch import nn


    class HookedLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False, device="cuda")
            self.calls = 0
            self.linear.weight.register_post_accumulate_grad_hook(self._make_grad_hook())

        def _make_grad_hook(self):
            def grad_hook(parameter):
                if parameter.grad is None:
                    raise RuntimeError("grad missing")
                self.post_backward()

            return grad_hook

        def forward(self, inputs):
            return self.linear(inputs)

        def post_backward(self) -> None:
            self.calls += 1


    def main() -> None:
        if os.environ.get("PATCH") == "1":
            from tests.unit_tests.post_accumulate_trace import install_post_accumulate_trace

            install_post_accumulate_trace()
        import coverage

        script = os.path.abspath(__file__)
        cov = coverage.Coverage(
            data_file=os.environ["COVFILE"],
            concurrency=["thread"],
            include=[script],
        )
        cov.start()
        model = HookedLinear()
        model(torch.ones(1, 2, device="cuda")).sum().backward()
        torch.cuda.synchronize()
        assert model.calls == 1
        cov.stop()
        cov.save()
        source = open(script, encoding="utf-8").read().splitlines()

        def line_of(snippet: str) -> int:
            for number, line in enumerate(source, start=1):
                if snippet in line:
                    return number
            raise SystemExit(f"snippet not found: {snippet}")

        _filename, _statements, missing, _formatted = cov.analysis(script)
        hook_line = line_of("self.post_backward()")
        post_line = line_of("self.calls += 1")
        print(f"HOOK_MISSING={hook_line in missing}")
        print(f"POST_MISSING={post_line in missing}")
        print(f"CALLS={model.calls}")


    if __name__ == "__main__":
        main()
    """)

_TESTMON_MODEL = textwrap.dedent("""\
    from torch import nn


    class HookedLinear(nn.Module):
        def __init__(self, device):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False, device=device)
            self.calls = 0
            self.linear.weight.register_post_accumulate_grad_hook(self._make_hook())

        def _make_hook(self):
            def grad_hook(parameter):
                if parameter.grad is not None:
                    self.calls += 1
            return grad_hook

        def forward(self, inputs):
            return self.linear(inputs)
    """)

_TESTMON_TEST = textwrap.dedent("""\
    import pytest
    import torch
    from hook_model import HookedLinear


    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_hook(device):
        model = HookedLinear(device)
        model(torch.ones(1, 2, device=device)).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
        assert model.calls == 1
    """)

_TESTMON_CONFTEST = textwrap.dedent("""\
    import os
    import sys

    if os.environ.get("PATCH") == "1":
        sys.path.insert(0, os.environ["REPO_ROOT"])
        from tests.unit_tests.post_accumulate_trace import install_post_accumulate_trace

        install_post_accumulate_trace()
    """)


def _run_coverage_probe(directory: Path, patch: bool) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    script = directory / "hook_coverage_probe.py"
    script.write_text(_COVERAGE_PROBE)
    env = os.environ.copy()
    env["PATCH"] = "1" if patch else "0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(_REPO_ROOT)]
    )
    env["COVFILE"] = str(directory / ".coverage")
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=directory,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"coverage probe patch={patch} exited {completed.returncode}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    values = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"HOOK_MISSING", "POST_MISSING", "CALLS"}:
            values[key] = value
    return values


def _selected_testmon_nodes(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if "::" in line]


def _run_testmon(directory: Path, patch: bool) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "hook_model.py").write_text(_TESTMON_MODEL)
    (directory / "test_hook.py").write_text(_TESTMON_TEST)
    (directory / "conftest.py").write_text(_TESTMON_CONFTEST)
    env = os.environ.copy()
    env["PATCH"] = "1" if patch else "0"
    env["REPO_ROOT"] = str(_REPO_ROOT)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(_REPO_ROOT)]
    )
    base = [
        sys.executable,
        "-m",
        "pytest",
        "--rootdir",
        str(directory),
        "--confcutdir",
        str(directory),
        "-p",
        "testmon.pytest_testmon",
        "--testmon",
        "-q",
        "--disable-warnings",
        "test_hook.py",
    ]
    baseline = subprocess.run(
        [*base, "--testmon-noselect"],
        cwd=directory,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if baseline.returncode != 0:
        raise AssertionError(
            f"testmon baseline patch={patch} exited {baseline.returncode}\n"
            f"{baseline.stdout}\n{baseline.stderr}"
        )
    model_path = directory / "hook_model.py"
    original = model_path.read_text()
    old = "if parameter.grad is not None:"
    new = "if not (parameter.grad is None):"
    assert original.count(old) == 1
    model_path.write_text(original.replace(old, new))
    selected = subprocess.run(
        [*base, "--testmon-nocollect", "--testmon-forceselect", "--collect-only"],
        cwd=directory,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    # pytest exits 5 when every test is deselected. One selected test exits 0.
    if selected.returncode not in (0, 5):
        raise AssertionError(
            f"testmon select patch={patch} exited {selected.returncode}\n"
            f"{selected.stdout}\n{selected.stderr}"
        )
    return _selected_testmon_nodes(selected.stdout + "\n" + selected.stderr)


@_requires_cuda
def test_coverage_py_records_cuda_hook_bodies_only_with_the_patch(tmp_path):
    """coverage.py marks the CUDA hook and callee only when registration wraps them."""
    _skip_unless_primary_process()
    pytest.importorskip("coverage")
    patched = _run_coverage_probe(tmp_path / "patched", patch=True)
    unpatched = _run_coverage_probe(tmp_path / "unpatched", patch=False)
    assert patched["CALLS"] == "1"
    assert patched["HOOK_MISSING"] == "False"
    assert patched["POST_MISSING"] == "False"
    assert unpatched["CALLS"] == "1"
    assert unpatched["HOOK_MISSING"] == "True"
    assert unpatched["POST_MISSING"] == "True"


@_requires_cuda
def test_testmon_selects_the_cuda_test_only_when_the_hook_is_traced(tmp_path):
    """A hook-body edit selects the CUDA test only after the tracer is preserved."""
    _skip_unless_primary_process()
    pytest.importorskip("testmon")
    patched = _run_testmon(tmp_path / "patched", patch=True)
    unpatched = _run_testmon(tmp_path / "unpatched", patch=False)
    assert any("test_hook[cpu]" in node for node in patched)
    assert any("test_hook[cuda]" in node for node in patched)
    assert any("test_hook[cpu]" in node for node in unpatched)
    assert all("test_hook[cuda]" not in node for node in unpatched)
