# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Chunked LM head on a GTP-sharded output weight.

``output_layer.weight`` is the one GTP weight that takes several backwards per iteration (one
per MTP head and one for the main head, per microbatch). GTP gathers it once per use and
accumulates its weight gradient across those backwards before a single reduce-scatter.

The chunked head must leave all of that as it is: the weight is still used once per head in
each direction, however many chunks the head is split into, and the reduced ``main_grad`` is
unchanged. A head that called the layer once per chunk would instead issue an on-demand
all-gather of the full weight for every chunk.

Layout: world_size = tp_size x gtp_remat_size. Launch with
``torchrun --nproc-per-node=4 -m pytest <this file>`` (or 2 for the TP=1, GTP=2 case).
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

if not HAVE_GTP:
    pytest.skip("GTP requires TransformerEngine >= 2.19", allow_module_level=True)

from megatron.core import parallel_state as ps
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel import generalized_tensor_parallelism as gtp
from megatron.core.tensor_parallel import layers
from megatron.core.tensor_parallel.gtp_api import (
    classify_gtp_remat_chains,
    finalize_gtp_forward_counts,
)
from megatron.core.tensor_parallel.gtp_symmetric_memory import (
    deregister_and_clear_gtp_symm_pools,
    register_gtp_symm_pool,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import multi_token_prediction as mtp
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _run_distributed,
    _torchrun_dist_init,
    reset_fp8_state,
    reset_gtp_globals,
)
from tests.unit_tests.tensor_parallel.test_chunked_lm_head import (
    HIDDEN,
    MTP_LAYERS,
    VOCAB,
    assert_matches,
    make_batch,
    make_cross_entropy,
    make_head_config,
    run_head_sequence,
)

ITERATIONS = 3
MICROBATCHES = 2
HEADS = 1 + MTP_LAYERS

# How output_layer's main_grad is typed selects the weight-gradient branch of the backward:
#   fp32      -- the GEMM epilogue widens into an fp32 buffer that accumulates across backwards
#                (--ddp-param-name-patterns-for-fp32-local-accumulation *output_layer*).
#   bf16      -- dtypes match but no symmetric pool: a plain GEMM and a reduce-scatter per backward.
#   zero-copy -- dtypes match and the reduce-scatter buffers live in the registered symmetric
#                pool (--grad-reduce-in-bf16 with --gtp-remat-nccl-ub): the GEMM writes straight
#                into the send buffer, accumulating there.
WGRAD_PATHS = ["fp32", "bf16", "zero-copy"]

# (tp_size, gtp_remat_size) the process groups are currently built for.
_current_layout = None


def _ensure_layout(tp_size: int, gtp_remat_size: int) -> None:
    """Build the process groups for a layout once and keep them for every case that uses it.

    Rebuilding them for each case (and each of its two phases) leaves enough NCCL communicators
    behind to exhaust device memory over the whole module.
    """
    global _current_layout
    if _current_layout == (tp_size, gtp_remat_size):
        return
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        gtp_remat_size=gtp_remat_size,
    )
    _current_layout = (tp_size, gtp_remat_size)


@pytest.fixture(scope="module", autouse=True)
def _forget_layout():
    yield
    global _current_layout
    _current_layout = None


def _reset_gtp_between_phases() -> None:
    """Drop the GTP state the previous layer left behind; the process groups stay."""
    gtp.GTPShardedParam._chain_state = {}
    gtp.GTPShardedParam._recompute_chain_state = {}
    gtp.GTPShardedParam._link_tables_flushed = False
    gtp.reset_gtp_state()
    gtp._GTP_PARAMS.clear()


def _worker(
    rank, world_size, port, tp_size, gtp_remat_size, sequence_parallel, wgrad_path, num_chunks
):
    zero_copy = wgrad_path == "zero-copy"
    main_grad_dtype = torch.float32 if wgrad_path == "fp32" else torch.bfloat16
    monkeypatch = pytest.MonkeyPatch()
    counts = {}
    all_gather_weight = gtp.GTPShardedParam._all_gather_weight
    reduce_scatter = gtp.GTPShardedParam._reduce_scatter

    def counting_all_gather_weight(self, async_op, fwd, *args, **kwargs):
        key = "all_gather_fwd" if fwd else "all_gather_bwd"
        counts[key] = counts.get(key, 0) + 1
        return all_gather_weight(self, async_op, fwd, *args, **kwargs)

    def counting_reduce_scatter(self, wgrads, *args, **kwargs):
        counts["reduce_scatter"] = counts.get("reduce_scatter", 0) + 1
        return reduce_scatter(self, wgrads, *args, **kwargs)

    def run_phase(chunks: int):
        """A fresh layer from the same seeds, so both phases start from the same state."""
        monkeypatch.setattr(layers.LinearCrossEntropyChunked, "checks_left", 8)
        _ensure_layout(tp_size, gtp_remat_size)
        _reset_gtp_between_phases()
        model_parallel_cuda_manual_seed(42)
        torch.manual_seed(42)
        if zero_copy:
            register_gtp_symm_pool(ps.get_gtp_weight_remat_group())
        tp_group = ps.get_tensor_model_parallel_group()
        # GTP ranks are separate data lanes; the ranks of one TP group share a lane.
        data_lane = ps.get_gtp_weight_remat_rank()
        assert dist.get_process_group_ranks(tp_group)[0] // tp_size == data_lane

        config = make_head_config(
            tp_size,
            sequence_parallel,
            use_cpu_initialization=False,
            gradient_accumulation_fusion=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            lm_head_loss_num_chunks=chunks,
        )
        head = torch.nn.Module()
        # The name matters: tag_gtp_params_with_names enables weight-gradient accumulation for
        # the parameter called "output_layer.weight" only.
        head.output_layer = tensor_parallel.ColumnParallelLinear(
            HIDDEN,
            VOCAB,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=False,
            output_dtype=torch.float32,
        ).cuda()
        layer = head.output_layer
        weight = layer.weight
        assert isinstance(weight, gtp.GTPShardedParam)
        assert layer.gtp_remat_size == gtp_remat_size
        assert layer.sequence_parallel == (sequence_parallel and tp_size > 1)
        gtp.tag_gtp_params_with_names(head)
        classify_gtp_remat_chains([head])
        assert weight._wgrad_accum_enabled
        weight.main_grad = torch.zeros(weight.shape, dtype=main_grad_dtype, device="cuda")
        # Assert the branch under test is the one that runs, rather than assume it.
        assert weight.use_zero_copy_wgrad(torch.bfloat16) == zero_copy

        compute_loss = make_cross_entropy(tp_group)
        chunked_calls = []
        forward_chunked_loss = layer.forward_chunked_loss

        def counting_forward_chunked_loss(*args, **kwargs):
            chunked_calls.append(1)
            return forward_chunked_loss(*args, **kwargs)

        layer.forward_chunked_loss = counting_forward_chunked_loss

        results = []
        for iteration in range(ITERATIONS):
            # Zeroed once per iteration and drained once after the microbatch loop, as in the
            # schedule: the accumulation window spans the whole iteration.
            weight.main_grad = torch.zeros(weight.shape, dtype=main_grad_dtype, device="cuda")
            counts.clear()
            losses, input_grads = [], []
            for microbatch in range(MICROBATCHES):
                hidden, labels, mask = make_batch(
                    1000 * iteration + 10 * microbatch + data_lane,
                    HEADS,
                    tp_size,
                    sequence_parallel,
                    torch.bfloat16,
                )
                # As in the schedule: once the iteration's last forward has run, the forward
                # counts are sealed, so the last backward can launch the reduce-scatter and a
                # backward without a counted forward is an error.
                last = microbatch == MICROBATCHES - 1
                loss = run_head_sequence(
                    layer,
                    config,
                    compute_loss,
                    hidden,
                    labels,
                    mask,
                    tp_group,
                    before_backward=(lambda: finalize_gtp_forward_counts(head)) if last else None,
                )
                losses.append(loss.detach().clone())
                input_grads.append(hidden.grad.clone())
            gtp.wait_for_gtp_grad_reduction_on_current_stream()
            torch.cuda.synchronize()
            results.append((losses, input_grads, weight.main_grad.clone(), dict(counts)))
        num_chunked_calls = len(chunked_calls)

        dist.barrier()
        del head, layer, weight
        _reset_gtp_between_phases()
        return results, num_chunked_calls

    # The MTP heads log through a tracker that needs the full set of process groups, and their
    # early backward reads the published loss scale.
    monkeypatch.setattr(
        mtp.MTPLossLoggingHelper,
        "save_metrics_to_tracker",
        staticmethod(lambda *args, **kwargs: None),
    )
    monkeypatch.setattr(
        mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.ones(1, device="cuda")
    )
    monkeypatch.setattr(gtp.GTPShardedParam, "_all_gather_weight", counting_all_gather_weight)
    monkeypatch.setattr(gtp.GTPShardedParam, "_reduce_scatter", counting_reduce_scatter)
    try:
        reference, reference_chunked_calls = run_phase(1)
        result, result_chunked_calls = run_phase(num_chunks)
    finally:
        monkeypatch.undo()
        if zero_copy:
            # Registered on a group that outlives this case; the next case must not inherit it.
            deregister_and_clear_gtp_symm_pools()

    # Otherwise this compares the plain path with itself.
    assert reference_chunked_calls == 0
    assert result_chunked_calls == ITERATIONS * MICROBATCHES * HEADS

    uses_per_iteration = MICROBATCHES * HEADS
    for iteration, (ref, res) in enumerate(zip(reference, result)):
        ref_losses, ref_input_grads, ref_main_grad, ref_counts = ref
        res_losses, res_input_grads, res_main_grad, res_counts = res
        where = f"rank {rank}, iteration {iteration}"

        # One gather per head and direction, not one per chunk.
        assert res_counts == ref_counts, f"{where}: GTP calls {res_counts} != {ref_counts}"
        assert res_counts["all_gather_fwd"] == uses_per_iteration, f"{where}: {res_counts}"
        assert res_counts["all_gather_bwd"] == uses_per_iteration, f"{where}: {res_counts}"
        if wgrad_path == "bf16":
            # No accumulation on this branch: every backward reduce-scatters.
            assert res_counts["reduce_scatter"] == uses_per_iteration, f"{where}: {res_counts}"
        else:
            # The iteration's backwards are summed and reduce-scattered once.
            assert res_counts["reduce_scatter"] == 1, f"{where}: {res_counts}"

        assert ref_main_grad.abs().max() > 0, f"{where}: main_grad never received a gradient"
        for microbatch in range(MICROBATCHES):
            assert_matches(
                ref_losses[microbatch], res_losses[microbatch], torch.float32, f"{where}: loss"
            )
            assert_matches(
                ref_input_grads[microbatch],
                res_input_grads[microbatch],
                torch.bfloat16,
                f"{where}: input gradient",
            )
        assert_matches(ref_main_grad, res_main_grad, torch.bfloat16, f"{where}: main_grad")


class TestGTPChunkedLMHead:
    @pytest.mark.parametrize("wgrad_path", WGRAD_PATHS)
    @pytest.mark.parametrize("sequence_parallel", [False, True], ids=["no-sp", "sp"])
    @pytest.mark.parametrize("num_chunks", [5, 8])
    def test_tp_gtp_matches_unchunked(self, wgrad_path, sequence_parallel, num_chunks):
        """TP=2 x GTP=2: vocabulary split over TP, each partition sharded over GTP."""
        _run_distributed(_worker, 4, 2, 2, sequence_parallel, wgrad_path, num_chunks)

    @pytest.mark.parametrize("wgrad_path", WGRAD_PATHS)
    @pytest.mark.parametrize("gtp_remat_size", [2, 4])
    def test_gtp_matches_unchunked(self, wgrad_path, gtp_remat_size):
        """TP=1: the layout of the production runs, where only GTP shards the weight."""
        _run_distributed(_worker, gtp_remat_size, 1, gtp_remat_size, False, wgrad_path, 8)
