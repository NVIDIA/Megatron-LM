# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Oracle for `_multimem_all_gather_kernel`
(megatron/core/inference/communication/torch_symm_triton/collectives.py:68), launched by
`multimem_all_gather` (collectives.py:255): the NVLS all-gather of the inference-optimized
column-parallel linears (hidden states) and of the logits.

Contract (from the launcher's docstring and its callers, inference_layers.py:216-244, 360-385,
569-613): every rank r of a group of ws ranks passes an input x_r of N/ws elements (any dtype;
byte size a multiple of 16). After the call, on every rank, the symmetric buffer bytes
[byte_offset, byte_offset + N * esize) hold concat(x_0, ..., x_{ws-1}) (flat), bit for bit:
a copy, so NaN payloads, -0.0, infinities and subnormals are preserved. `output_tensor` is the
view of the buffer at `byte_offset`. Nothing else is written: buffer bytes before byte_offset
and after the region, and the input. `barrier_before=True` first waits until every rank has
finished its earlier work on the buffer (needed when the previous operation on the buffer was an
all-gather or the fused kernel, whose output peers may still read); on return every rank's
stores are visible on every rank. Preconditions: input bytes % 16 == 0 and output numel ==
ws * input numel (checked), byte_offset % 16 == 0 and a contiguous input (not checked).

Tolerance class: exact (bitwise), against NCCL's all_gather_into_tensor of the same inputs.

Structure-derived inputs: each thread moves one 16-byte chunk (16 / esize elements: the compiled
variants are 2, 4, 8 and 16 elements per thread, for int64/fp64, fp32, bf16 and uint8), blocks of
1024 threads, grids of at most 128 blocks with a grid-stride loop. Per-rank chunk counts: 1,
1023/1024/1025 (one block), 131071/131072/131073 (the 128-block grid, then one more sweep) and
262145; byte_offset in {0, 16, 1 MiB}; barrier_before in {False, True}; ws in {2, 4, 8};
combined pairwise. Inputs are random bit patterns. Memory: a 64 KiB sentinel before (with an
offset) and after the region, a guarded input that must not change. Back-to-back all-gathers on
one buffer with barrier_before, sizes and data changing every step and a random rank delayed
before each, are checked after the loop (the AG -> AG reuse the barrier exists for; the
existing test writes identical data twice, so it cannot see a missing barrier), and, since
the medium-risk batch, a rank held after each all-gather before it copies the result out while
its peers start the next one (`test_barrier_before_waits_for_a_slow_reader`). Production shapes
at the whole node (`test_production_shapes`): Llama-3.1-70B's hidden-state gather of a
2048-token step and its fp32 logits. `test_sanitizer_small_case` is the compute-sanitizer case.
Int32 offsets cannot overflow (no `large` case): indices count 16-byte chunks of the "tp"
buffer, at most 512 MiB (`SymmetricMemoryManager._default_size_mb`).

Covers main and #8709 (n12): n12 makes barrier_before default to True; every call here passes
it explicitly. The RS -> equal-size AG without a barrier, and CUDA-graph replay, are in
test_multimem_reduce_scatter.py; the AG after the fused kernel in
test_multimem_reduce_scatter_residual_add.py.
"""

import pytest
import torch

from megatron.core.inference.communication.torch_symm_triton.collectives import multimem_all_gather
from tests.unit_tests.inference.oracles import oracle_utils as ou
from tests.unit_tests.inference.oracles.nvls import nvls_oracle_utils as nu

CHUNK_BYTES = 16
BLOCK_SIZE = 1024  # threads per block (collectives.py:229-233)
MAX_NUM_BLOCKS = 128  # kernel_config.py
CHUNKS_PER_RANK = (
    [1]
    + ou.boundary_sizes(BLOCK_SIZE, BLOCK_SIZE * MAX_NUM_BLOCKS, multiples=(), degenerate=False)
    + [2 * BLOCK_SIZE * MAX_NUM_BLOCKS + 1]
)
DTYPES = {"int64": torch.int64, "fp32": torch.float32, "bf16": torch.bfloat16, "uint8": torch.uint8}
BYTE_OFFSETS = [0, 16, 1 << 20]


def gpu_test(fn):
    """NVLS needs Hopper or Blackwell; GB200 CI launches the file through launch_on_gb200."""
    return pytest.mark.launch_on_gb200(ou.requires_sm(ou.HOPPER, ou.BLACKWELL)(fn))


def _sweep_params():
    cases = ou.pairwise_cases(
        {
            "chunks": CHUNKS_PER_RANK,
            "dtype": list(DTYPES),
            "byte_offset": BYTE_OFFSETS,
            "barrier_before": [False, True],
            "ws": nu.GROUP_SIZES,
        }
    )
    return [
        pytest.param(
            c["ws"],
            c["chunks"],
            c["dtype"],
            c["byte_offset"],
            c["barrier_before"],
            id=f"ws{c['ws']}-chunks{c['chunks']}-{c['dtype']}-offset{c['byte_offset']}"
            f"-barrier{int(c['barrier_before'])}",
        )
        for c in cases
    ]


def run_all_gather(g, numel, dtype, byte_offset, barrier_before, x, what):
    """All-gather ``x`` (this rank's input, ``numel`` elements) into ``g``'s main buffer at
    ``byte_offset`` and check it on every rank: NCCL's gather bit for bit, the bytes around the
    region and the input unchanged."""
    symm = g.buffer()
    expected = g.gather(x).reshape(-1)
    x_buf = ou.guarded_like(x)
    out = symm.view((g.ws * numel,), dtype, byte_offset)
    region_end = byte_offset + out.numel() * dtype.itemsize
    symm.raw[byte_offset:region_end].fill_(0xFF)  # stale-looking bytes the gather must replace
    before = symm.sentinel_before(byte_offset)
    after = symm.sentinel_after(region_end)
    g.sync()  # no peer's stores may land before this rank's fills
    result = multimem_all_gather(
        out, x_buf.tensor, symm.hdl, byte_offset=byte_offset, barrier_before=barrier_before
    )
    gathered = out.clone()

    def check():
        assert result is out, "multimem_all_gather must return output_tensor"
        ou.assert_exact(gathered, expected, what="gathered region")
        ou.check_guards(input=x_buf)
        ou.assert_exact(x_buf.tensor, x, what="input (read-only)")
        after.check("buffer after the region")
        if before is not None:
            before.check("buffer before byte_offset")

    nu.agree(check, what)


@gpu_test
@pytest.mark.parametrize("ws,chunks,dtype_name,byte_offset,barrier_before", _sweep_params())
def test_all_gather_matches_nccl(nvls_groups, ws, chunks, dtype_name, byte_offset, barrier_before):
    """Every rank's buffer region equals NCCL's all-gather bit for bit; nothing else changes."""
    g = nvls_groups.get(ws)
    dtype = DTYPES[dtype_name]
    numel = chunks * CHUNK_BYTES // dtype.itemsize
    seed = nu.case_seed("all-gather", ws, chunks, dtype_name, byte_offset, barrier_before)
    x = nu.random_bits((numel,), dtype, g.rank_generator(seed))
    run_all_gather(g, numel, dtype, byte_offset, barrier_before, x, "all-gather")


@gpu_test
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_back_to_back_all_gathers_reuse_the_buffer(nvls_groups, ws):
    """All-gather after all-gather on one buffer with barrier_before=True (MTP's eh_proj and
    output-layer gathers): sizes, dtypes and data change every step, and one random rank is
    delayed before each call. Without the barrier a fast rank's stores could overwrite a peer's
    previous result before the peer copied it out; every step is checked after the loop."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    steps = 12
    shared = g.shared_generator(nu.case_seed("all-gather-back-to-back", ws))
    sizes = [1, 1025, 131073, 4097]
    plan = []
    for i in range(steps):
        dtype = list(DTYPES.values())[i % len(DTYPES)]
        chunks = sizes[int(torch.randint(0, len(sizes), (), generator=shared))]
        numel = chunks * CHUNK_BYTES // dtype.itemsize
        x = nu.random_bits((numel,), dtype, g.rank_generator(nu.case_seed("ag-b2b", ws, i)))
        plan.append((x, g.gather(x).reshape(-1)))
    results = []
    for x, _ in plan:
        out = symm.view((ws * x.numel(),), x.dtype)
        g.maybe_sleep(shared)
        multimem_all_gather(out, x, symm.hdl, barrier_before=True)
        results.append(out.clone())

    def check():
        for i, ((_, expected), got) in enumerate(zip(plan, results)):
            ou.assert_exact(got, expected, what=f"step {i} ({got.dtype}, {got.numel()} elements)")

    nu.agree(check, "back-to-back all-gathers")


SLOW_READER_CYCLES = 2_000_000
"""How long the held rank waits before it copies a result out: about 1 ms on H100, a hundred
times the all-gather that its peers start meanwhile."""


@gpu_test
@pytest.mark.parametrize("ws", nu.group_size_params())
def test_barrier_before_waits_for_a_slow_reader(nvls_groups, ws):
    """barrier_before=True makes an all-gather wait until every rank has read the buffer's
    previous contents. After each all-gather one rank (a different one each step) is held for
    SLOW_READER_CYCLES before it copies the result out, while its peers go straight on to the
    next all-gather of the same size on the same buffer, as MTP's consecutive gathers do. With
    the barrier the held rank copies its own step's data; without it the peers' next stores
    land in its buffer first. The back-to-back test above delays a rank before a call, which a
    missing barrier survives (mutant o5-ag-005)."""
    g = nvls_groups.get(ws)
    symm = g.buffer()
    numel = 4097 * CHUNK_BYTES // torch.bfloat16.itemsize
    plan = []
    for i in range(2 * ws):
        gen = g.rank_generator(nu.case_seed("ag-slow-reader", ws, i))
        x = nu.random_bits((numel,), torch.bfloat16, gen)
        plan.append((x, g.gather(x).reshape(-1)))
    out = symm.view((ws * numel,), torch.bfloat16)
    g.sync()
    results = []
    for i, (x, _) in enumerate(plan):
        multimem_all_gather(out, x, symm.hdl, barrier_before=True)
        if g.rank == i % ws:
            torch.cuda._sleep(SLOW_READER_CYCLES)
        results.append(out.clone())

    def check():
        for i, ((_, expected), got) in enumerate(zip(plan, results)):
            held = "held" if g.rank == i % ws else "not held"
            ou.assert_exact(got, expected, what=f"step {i} (this rank {held})")

    nu.agree(check, "all-gather after a slow reader")


PRODUCTION_CASES = {
    # id: (dtype, this rank's input shape at group size ws, barrier_before)
    "llama-70b-hidden-bf16-2048-tokens": lambda ws: (
        torch.bfloat16,
        (nu.PRODUCTION_TOKENS // ws, nu.PRODUCTION_DENSE["hidden"]),
        False,
    ),
    "llama-70b-logits-fp32-64-tokens": lambda ws: (
        torch.float32,
        (64, nu.PRODUCTION_DENSE["vocab"] // ws),
        True,
    ),
}
"""Production gathers at the whole node (``nu.PRODUCTION_GROUP_SIZE``): the column-parallel
input of Llama-3.1-70B (sequence-parallel rows of H = 8192, after the row-parallel
reduce-scatter, so no barrier) for a 2048-token step, and its vocab-parallel logits
[64 tokens, 128256 / ws] in fp32 (``inference_all_gather_from_tensor_model_parallel_region``,
after another gather). 32 MiB gathered in both, at 8 and at 4 ranks."""


@gpu_test
@pytest.mark.parametrize("case", list(PRODUCTION_CASES))
def test_production_shapes(nvls_groups, case):
    """The sweep's checks at production shapes and group size (8 ranks on H100, 4 on GB200)."""
    ws = nu.PRODUCTION_GROUP_SIZE
    g = nvls_groups.get(ws)
    dtype, shape, barrier_before = PRODUCTION_CASES[case](ws)
    x = nu.random_bits(shape, dtype, g.rank_generator(nu.case_seed("ag-production", case)))
    numel = x.numel()
    run_all_gather(g, numel, dtype, 0, barrier_before, x.reshape(-1), f"all-gather {case}")


@gpu_test
@pytest.mark.sanitizer
def test_sanitizer_small_case(nvls_groups):
    """The compute-sanitizer (memcheck) case: ws 2, 1025 16-byte chunks per rank (a full block
    and one lane of a second), bf16, byte_offset 16 (sentinels on both sides) and
    barrier_before, with the sweep's checks. A dedicated function, so the runner selects it by
    name at any world size."""
    g = nvls_groups.get(2)
    numel = 1025 * CHUNK_BYTES // torch.bfloat16.itemsize
    x = nu.random_bits((numel,), torch.bfloat16, g.rank_generator(nu.case_seed("ag-sanitizer")))
    run_all_gather(g, numel, torch.bfloat16, 16, True, x, "all-gather sanitizer case")


@pytest.mark.parametrize("case", ["valid", "output-not-ws-times-input", "input-not-16-bytes"])
def test_launcher_checks_sizes(monkeypatch, case):
    """The launcher refuses an output that is not ws times the input and an input whose byte
    size is not a multiple of 16, before launching (CPU: the kernel is replaced by a recorder).
    Passes on main."""
    pytest.importorskip("triton")
    from types import SimpleNamespace

    from megatron.core.inference.communication.torch_symm_triton import collectives, utils

    launches = []
    monkeypatch.setattr(collectives, "_multimem_all_gather_kernel", _Recorder(launches))
    monkeypatch.setattr(utils, "is_device_nvls_capable", lambda device: True)
    handle = SimpleNamespace(world_size=4, rank=1, multicast_ptr=0, signal_pad_ptrs_dev=0)
    x = torch.zeros(64, dtype=torch.bfloat16)
    out = torch.zeros(4 * 64, dtype=torch.bfloat16)
    if case == "output-not-ws-times-input":
        out = torch.zeros(3 * 64, dtype=torch.bfloat16)
    elif case == "input-not-16-bytes":
        x, out = torch.zeros(4, dtype=torch.bfloat16), torch.zeros(16, dtype=torch.bfloat16)
    if case == "valid":
        collectives.multimem_all_gather(out, x, handle, barrier_before=True)
        assert len(launches) == 1
        return
    with pytest.raises(AssertionError):
        collectives.multimem_all_gather(out, x, handle, barrier_before=True)
    assert not launches


class _Recorder:
    """Stands in for a Triton kernel: records launches instead of running them."""

    def __init__(self, launches):
        self.launches = launches

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.launches.append((grid, kwargs))
