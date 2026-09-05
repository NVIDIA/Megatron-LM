# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import gc

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    apply_flex_backend_kwargs,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_dispatcher_configs,
    get_valid_fp8_flags,
)
from tests.unit_tests.test_utilities import Utils

# Transformer Engine 2.17 aborts in the A2A overlap suite with a pybind11 GIL dec_ref failure.
pytestmark = pytest.mark.flaky_in_dev


def build_model(config, use_padding_mask=False, use_mtp_input_mask=False):
    seq_len = 32
    max_seq_len = 300
    # ids = random.sample([i for i in range(max_seq_len)], seq_len)
    ids = [i for i in range(seq_len)]

    # build input tensors
    data = {
        "input_ids": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "labels": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "position_ids": torch.tensor([i for i in range(seq_len)], dtype=torch.int64)
        .repeat((1, 1))
        .cuda(),
        "attention_mask": torch.ones((1, 1, seq_len, seq_len), dtype=bool).cuda(),
    }

    # Optionally add padding_mask with same shape as input_ids
    if use_padding_mask:
        padding_mask = torch.zeros((1, seq_len), dtype=torch.bool).cuda()
        padding_mask[0, -8:] = True
        data["padding_mask"] = padding_mask

    # Optionally add mtp_input_mask, with False holes away from both ends. MTP rolls this
    # mask together with input_ids and then uses it to pick which embedding rows keep a
    # gradient, so the holes must not be symmetric under a one-position shift — otherwise a
    # mask that is rolled one time too many still selects the same rows and the test passes
    # for the wrong reason.
    if use_mtp_input_mask:
        mtp_input_mask = torch.ones((1, seq_len), dtype=torch.bool).cuda()
        mtp_input_mask[0, 5] = False
        mtp_input_mask[0, 11:14] = False
        mtp_input_mask[0, 20] = False
        data["mtp_input_mask"] = mtp_input_mask

    # build layer spec
    transformer_layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec.layer_specs[-1], True)

    # build model
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=128,
        pre_process=True,
        post_process=True,
        max_sequence_length=max_seq_len,
    )
    f_schedule_plan = gpt_model.build_schedule_plan(**data)
    return gpt_model, f_schedule_plan, data


def run_two_chunk_parity(
    layers,
    extra_kwargs,
    microbatches=1,
    use_padding_mask=False,
    use_mtp_input_mask=False,
    on_plans_built=None,
    on_forward_done=None,
):
    """Assert the a2a overlap schedule and the reference forward give the same grads.

    Builds one model per entry in ``layers``, captures reference grads from
    GPTModel.forward/backward, then replays the same inputs through the 1F1B pattern
    (chunk0 fwd -> chunk1 fwd + chunk0 bwd -> chunk1 bwd). ``on_plans_built`` is an
    optional hook for asserting schedule-plan invariants before the overlap run;
    ``on_forward_done`` runs on chunk 0's plan right after its forward-only pass, once per
    microbatch. With ``microbatches > 1`` both paths accumulate grads over that many
    identical microbatches and the plans are rebuilt each round, as in a real 1F1B step.
    """
    assert len(layers) == 2, "the 1F1B pattern below is written for exactly two chunks"

    gpt_models = []
    schedule_plans = []
    ref_captures = []
    datas = []

    with deterministic_mode():
        for layer_num in layers:
            output_tensors = []
            # build config
            config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
            # build model
            gpt_model, schedule_plan, data = build_model(
                config,
                use_padding_mask=use_padding_mask,
                use_mtp_input_mask=use_mtp_input_mask,
            )
            gpt_model.cuda()
            gpt_models.append(gpt_model)
            datas.append(data)
            schedule_plans.append(schedule_plan)

            # run reference
            for _ in range(microbatches):
                loss = gpt_model.forward(**data)
                loss = float16_to_fp32(loss)
                loss.backward(torch.ones_like(loss))
                output_tensors.append(loss)

            capture = {"outputs": output_tensors}
            for name, param in gpt_model.named_parameters():
                capture[name] = param.grad
            ref_captures.append(capture)
            gpt_model.zero_grad()
        assert gpt_models[0].embedding is not None
        assert gpt_models[1].embedding is not None
        if on_plans_built is not None:
            on_plans_built(schedule_plans)
        # run a2a overlap
        capture_0 = {"outputs": []}
        capture_1 = {"outputs": []}
        a2a_captures = [capture_0, capture_1]
        for i in range(microbatches):
            # 1st forward
            if i > 0:
                assert (
                    schedule_plans[0].pre_process is None
                ), "pre_process should be released after backward"
                schedule_plans[0] = gpt_models[0].build_schedule_plan(**datas[0])
                schedule_plans[1] = gpt_models[1].build_schedule_plan(**datas[1])
            f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
            if on_forward_done is not None:
                on_forward_done(schedule_plans[0])
            capture_0["outputs"].append(f_input_0)
            # overlap
            f_input_1 = TransformerModelChunkSchedulePlan.run(
                schedule_plans[1], schedule_plans[0], b_grad=torch.ones_like(f_input_0)
            )
            capture_1["outputs"].append(f_input_1)
            # last backward
            TransformerModelChunkSchedulePlan.run(
                None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
            )
        for i in range(len(gpt_models)):
            for name, param in gpt_models[i].named_parameters():
                a2a_captures[i][name] = param.grad

        # compare results
        for i in range(len(ref_captures)):
            comp_res = compare_captures(ref_captures[i], a2a_captures[i], True, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

        # release resources is necessary, otherwise later testcases will oom
        for i in range(len(schedule_plans)):
            schedule_plans[i] = None
            ref_captures[i] = None
            a2a_captures[i] = None
            for k in datas[i]:
                datas[i][k] = None
            datas[i] = None
            gpt_models[i].zero_grad()
            gpt_models[i] = None
        gc.collect()
        torch.cuda.empty_cache()


def capture_overlap_grads(layers, extra_kwargs, use_mtp_input_mask=False):
    """Run the two-chunk 1F1B overlap pattern once; return one grad dict per chunk.

    Unlike ``run_two_chunk_parity`` this does not involve ``GPTModel.forward`` at all, so
    both sides of a comparison built on it travel the identical schedule. That matters for
    the embedding parameters: overlap-vs-reference does not agree on their gradients even
    with the feature under test disabled (which is why ``run_two_chunk_parity`` passes
    ``skip_embedding=True``), so a reference-based test cannot say anything about them.
    Overlap-vs-overlap can.
    """
    with deterministic_mode():
        gpt_models, schedule_plans, datas = [], [], []
        for layer_num in layers:
            config = get_test_config(num_layers=layer_num, extra_kwargs=extra_kwargs)
            gpt_model, schedule_plan, data = build_model(
                config, use_mtp_input_mask=use_mtp_input_mask
            )
            gpt_model.cuda()
            gpt_models.append(gpt_model)
            schedule_plans.append(schedule_plan)
            datas.append(data)

        f_input_0 = TransformerModelChunkSchedulePlan.run(schedule_plans[0], None)
        f_input_1 = TransformerModelChunkSchedulePlan.run(
            schedule_plans[1], schedule_plans[0], b_grad=torch.ones_like(f_input_0)
        )
        TransformerModelChunkSchedulePlan.run(
            None, schedule_plans[1], b_grad=torch.ones_like(f_input_1)
        )

        captures = []
        for gpt_model in gpt_models:
            captures.append(
                {
                    name: (param.grad.clone() if param.grad is not None else None)
                    for name, param in gpt_model.named_parameters()
                }
            )

        for i in range(len(gpt_models)):
            schedule_plans[i] = None
            for k in datas[i]:
                datas[i][k] = None
            datas[i] = None
            gpt_models[i].zero_grad()
            gpt_models[i] = None
        gc.collect()
        torch.cuda.empty_cache()
    return captures


class TestA2AOverlap:
    """
    Test class for all-to-all overlap optimization in transformer models.

    This class contains tests to verify that the all-to-all overlap optimization
    produces the same results as the reference implementation.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("mtp_layers", [0, 1])
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize("layers", [[2, 1], [1, 2], [1, 1]])
    def test_1f1b_schedule_model_chunk(
        self, mtp_layers, dispatcher_type, flex_backend, fp8_flag, layers
    ):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """
        # create TransformerConfig
        extra_kwargs = {}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        if mtp_layers > 0:
            extra_kwargs["mtp_num_layers"] = mtp_layers
            extra_kwargs["mtp_loss_scaling_factor"] = 1.1
        run_two_chunk_parity(layers, extra_kwargs)

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("layers", [[2, 1], [1, 1]])
    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_1f1b_schedule_model_chunk_with_padding_mask(
        self, dispatcher_type, flex_backend, layers, tp_size
    ):
        """
        Verifies all-to-all overlap optimization with padding_mask produces
        the same results as the reference implementation with various TP/EP/CP combinations.
        """
        # Re-initialize model parallel with the specified configuration
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
            expert_tensor_parallel_size=1,
        )
        set_streams()

        # create TransformerConfig
        extra_kwargs = {"tensor_model_parallel_size": tp_size, "sequence_parallel": tp_size > 1}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        run_two_chunk_parity(layers, extra_kwargs, use_padding_mask=True)

    def _run_full_recompute_parity(
        self,
        mtp_layers,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        recompute_method,
        recompute_num_layers,
        microbatches=1,
    ):
        """Layer-level full recompute matches the non-overlap reference, which checkpoints
        through checkpointed_forward / MTP _checkpointed_forward with the same config.
        """
        if mtp_layers > 0 and not (recompute_method == "uniform" and recompute_num_layers == 1):
            # Otherwise the non-overlap reference warns and skips MTP recompute, so the
            # two paths are not comparable.
            pytest.skip("MTP recompute requires recompute_method='uniform' with num_layers=1")

        extra_kwargs = {
            "recompute_granularity": "full",
            "recompute_method": recompute_method,
            "recompute_num_layers": recompute_num_layers,
        }
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        if mtp_layers > 0:
            extra_kwargs["mtp_num_layers"] = mtp_layers
            extra_kwargs["mtp_loss_scaling_factor"] = 1.1

        def assert_segments_were_built(schedule_plans):
            # Guard against degrading into the plain overlap test.
            for plan in schedule_plans:
                assert len(plan._recompute_segments) > 0, "no recompute segments were built"

        def assert_activations_were_released(plan):
            # The point of the feature: after the chunk forward a recomputed layer holds
            # no activation and only its segment's input survives. Gradient parity alone
            # cannot catch a regression here - recompute is numerically transparent, so a
            # segment that quietly kept its graph would still compare equal. Re-checked
            # every microbatch, since the plans are rebuilt each round.
            assert len(plan._recompute_segments) > 0, "no recompute segments were built"
            for segment in plan._recompute_segments:
                assert segment.input_tensor is not None, "segment input was not retained"
                for layer in segment.layers:
                    for node in layer._iter_recomputed_nodes():
                        assert node.output is None, f"{node.name} kept its forward output"
                        assert node.inputs is None, f"{node.name} kept its forward inputs"

        run_two_chunk_parity(
            [3, 2],
            extra_kwargs,
            microbatches=microbatches,
            on_plans_built=assert_segments_were_built,
            on_forward_done=assert_activations_were_released,
        )

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("mtp_layers", [0, 1])
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize(
        "recompute_method,recompute_num_layers",
        [
            ("uniform", 1),
            ("uniform", 2),
            # 'block' leaves trailing layers eager, pinning the recomputed -> eager
            # hand-off: n=1 keeps layers 1.. eager, n=3 recomputes the whole decoder.
            ("block", 1),
            ("block", 3),
        ],
    )
    def test_1f1b_schedule_model_chunk_full_recompute(
        self,
        mtp_layers,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        recompute_method,
        recompute_num_layers,
    ):
        """Full segmentation matrix, one microbatch."""
        self._run_full_recompute_parity(
            mtp_layers,
            dispatcher_type,
            flex_backend,
            fp8_flag,
            recompute_method,
            recompute_num_layers,
        )

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("mtp_layers", [0, 1])
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize(
        "recompute_method,recompute_num_layers", [("uniform", 1), ("block", 1)]
    )
    def test_1f1b_schedule_model_chunk_full_recompute_multi_microbatch(
        self,
        mtp_layers,
        dispatcher_type,
        flex_backend,
        fp8_flag,
        recompute_method,
        recompute_num_layers,
    ):
        """Same parity, but over four microbatches, as in a real 1F1B steady state.

        One microbatch never runs the same modules through forward -> recompute ->
        backward twice, so it cannot catch state that the replay leaves behind and the
        next microbatch then reads: FP8 scaling metadata, the segments' RNG snapshots, or
        a node whose ``forward_no_grad`` was not restored. Grads are accumulated across
        the four microbatches on both paths and compared bitwise, so any per-round drift
        shows up. The segmentation axis is trimmed to one segment-per-layer case and one
        recomputed -> eager hand-off case; the matrix above covers the rest at
        ``microbatches=1``.
        """
        self._run_full_recompute_parity(
            mtp_layers,
            dispatcher_type,
            flex_backend,
            fp8_flag,
            recompute_method,
            recompute_num_layers,
            microbatches=4,
        )


    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    def test_full_recompute_preserves_mtp_input_mask(self, dispatcher_type, flex_backend):
        """Full recompute is transparent to MTP's input mask.

        ``mtp_input_mask`` is a mutable chunk-state field: MTP's ``_get_embeddings`` rolls
        it together with ``input_ids`` (one concatenated roll, so CP does a single boundary
        exchange) and writes both back. A segment snapshot that omits it lets the replay
        roll an already-rolled mask, shifting it one position against ``input_ids``, and the
        mask then drives
        ``torch.where(valid, decoder_input, decoder_input.detach())`` -- so the detached
        embedding rows are the wrong ones.

        That damage is confined to the embedding gradients: masking never changes a forward
        value, so outputs and every non-embedding parameter still agree. The comparison is
        therefore overlap-with-recompute against overlap-without-recompute, both through
        ``capture_overlap_grads``, rather than against ``GPTModel.forward`` -- the reference
        path does not agree with the overlap path on embedding gradients even without
        recompute, so it cannot arbitrate this.

        The mask's False holes are deliberately not symmetric under a one-position shift;
        otherwise an over-rolled mask would still select the same rows and pass.
        """
        base_kwargs = {"mtp_num_layers": 1, "mtp_loss_scaling_factor": 1.1}
        apply_flex_backend_kwargs(base_kwargs, dispatcher_type, flex_backend)
        recompute_kwargs = dict(
            base_kwargs,
            recompute_granularity="full",
            recompute_method="uniform",
            recompute_num_layers=1,
        )

        eager = capture_overlap_grads([2, 1], base_kwargs, use_mtp_input_mask=True)
        recomputed = capture_overlap_grads([2, 1], recompute_kwargs, use_mtp_input_mask=True)

        for i in range(len(eager)):
            assert "embedding.word_embeddings.weight" in eager[i], "embedding grad not captured"
            comp_res = compare_captures(eager[i], recomputed[i], True, False)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] chunk {i}: {comp_res[1]}"
