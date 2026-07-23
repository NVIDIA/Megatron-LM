# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from contextlib import nullcontext

import pytest
import torch
import torch.nn.functional as F

from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.utils import get_comm_stream, get_comp_stream, set_streams
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    DummyState,
    apply_flex_backend_kwargs,
    build_data,
    compare_captures,
    deterministic_mode,
    get_test_config,
    get_valid_dispatcher_configs,
    get_valid_fp8_flags,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

# Transformer Engine 2.17 aborts in the A2A overlap suite with a pybind11 GIL dec_ref failure.
pytestmark = pytest.mark.flaky_in_dev


def is_nccl_ep_zero_copy_available():
    """Zero-copy needs the newer TE symm-mem APIs (symm_mem_alloc/is_symm_backed), absent in a plain
    NCCL-EP build."""
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    if not HAVE_TE_EP:
        return False
    try:
        from transformer_engine.pytorch.ep import is_symm_backed, symm_mem_alloc  # noqa: F401
    except ImportError:
        return False
    return True


def is_op_fuser_available():
    """The static-shape/zero-copy path runs the TE op-fuser grouped GEMM (needs TE>=2.14 ops)."""
    try:
        from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
    except ImportError:
        return False
    return is_te_min_version("2.14.0")


def run_transformer_layer_ref_with_capture(model, input_tensors, iterations):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    transformer_layer = model.decoder.layers[0]
    output_tensors = []
    for i in range(iterations):
        output = transformer_layer(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in transformer_layer.named_parameters():
        capture[name] = param.grad

    return capture


def run_transformer_layer_a2a_overlap_with_capture(model, input_tensors, microbatches):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    transformer_layer = model.decoder.layers[0]
    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()

    set_streams()
    event = torch.cuda.Event()
    state = DummyState()
    state.is_mtp = False
    state.model = model
    layers = [
        TransformerLayerSchedulePlan(
            transformer_layer,
            event,
            state,
            get_comp_stream,
            get_comm_stream,
            extra_args={"is_moe": True, "enable_deepep": False},
        )
        for _ in range(microbatches)
    ]
    output_tensors = []

    # forward for 1st microbatch
    output, _ = TransformerLayerSchedulePlan.run(
        layers[0], None, f_input=input_tensors[0], b_grad=None
    )
    output_tensors.append(output)
    torch.cuda.synchronize()
    # overlapped forward and backward
    for i in range(1, microbatches):
        f_input, b_grad = TransformerLayerSchedulePlan.run(
            layers[i], layers[i - 1], f_input=input_tensors[i], b_grad=torch.ones_like(output)
        )
        output_tensors.append(f_input)
        torch.cuda.synchronize()
    # backward for last microbatch
    TransformerLayerSchedulePlan.run(None, layers[-1], f_input=None, b_grad=torch.ones_like(output))
    torch.cuda.synchronize()
    capture = {"outputs": output_tensors}
    for name, param in transformer_layer.named_parameters():
        capture[name] = param.grad

    return capture


def run_mtp_layer_ref_with_capture(
    model,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    attention_mask,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    microbatches,
):
    """
    Runs the model in reference mode and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each iteration.
        iterations: Number of iterations to run the model.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    mtp_block = model.mtp

    output_tensors = []
    for i in range(microbatches):
        output = mtp_block(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=hidden_states[i].clone(),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            embedding=model.embedding,
        )
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


def run_mtp_layer_a2a_overlap_with_capture(
    model,
    hidden_states,
    input_ids,
    position_ids,
    labels,
    attention_mask,
    rotary_pos_emb,
    rotary_pos_cos,
    rotary_pos_sin,
    microbatches,
):
    """
    Runs the model with all-to-all overlap optimization and captures outputs and gradients.

    Args:
        model: The transformer model to run.
        input_tensors: List of input tensors for each microbatch.
        microbatches: Number of microbatches to process.

    Returns:
        dict: A dictionary containing model outputs and parameter gradients.
    """
    for i in range(len(hidden_states)):
        hidden_states[i] = hidden_states[i].clone()

    set_streams()
    layers = []
    for _ in range(microbatches):
        state = DummyState()
        state.mtp_labels = labels
        state.input_ids = input_ids
        state.position_ids = position_ids
        state.attention_mask = attention_mask
        state.rotary_pos_emb = rotary_pos_emb
        state.rotary_pos_cos = rotary_pos_cos
        state.rotary_pos_sin = rotary_pos_sin
        state.model = model
        state.is_mtp = True
        event = torch.cuda.Event()
        layers.append(
            TransformerLayerSchedulePlan(
                model.mtp.layers[0],
                event,
                state,
                get_comp_stream,
                get_comm_stream,
                extra_args={
                    "is_moe": True,
                    "enable_deepep": False,
                    "is_first_layer": True,
                    "is_last_layer": True,
                },
            )
        )
    output_tensors = []
    # forward for 1st microbatch
    f_input, _ = TransformerLayerSchedulePlan.run(
        layers[0], None, f_input=hidden_states[0], b_grad=None
    )
    output_tensors.append(f_input)
    torch.cuda.synchronize()
    # overlapped forward and backward
    for i in range(1, microbatches):
        f_input, b_grad = TransformerLayerSchedulePlan.run(
            layers[i], layers[i - 1], f_input=hidden_states[i], b_grad=torch.ones_like(f_input)
        )
        output_tensors.append(f_input)
        torch.cuda.synchronize()
    # backward for last microbatch
    TransformerLayerSchedulePlan.run(
        None, layers[-1], f_input=None, b_grad=torch.ones_like(f_input)
    )
    torch.cuda.synchronize()
    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
        capture[name] = param.grad

    return capture


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

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_dense(self):
        """
        Verifies all-to-all overlap optimization in dense transformer layer produces
        the same results as the reference implementation.
        """
        extra_kwargs = {"moe_token_dispatcher_type": "alltoall"}
        config = get_test_config(num_moe_experts=None, extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, 0) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_shared_expert(self):
        """
        Verifies all-to-all overlap optimization in transformer layer with shared expert produces
        the same results as the reference implement
        ation.
        """
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "moe_shared_expert_intermediate_size": 512,
        }
        overlap_config = get_test_config(extra_kwargs=extra_kwargs)
        extra_kwargs["moe_shared_expert_overlap"] = False
        ref_config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=ref_config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=ref_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(ref_config, 0) if ref_config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            del gpt_model

            gpt_model = GPTModel(
                config=overlap_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_early_attn_memory_release(self):
        """
        Verifies all-to-all overlap optimization in transformer layer with early attn memory release
        produces the same results as the reference implementation.
        """
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "ep_overlap_early_attn_memory_release": True,
            "overlap_moe_expert_parallel_comm": True,
        }
        overlap_config = get_test_config(extra_kwargs=extra_kwargs)
        ref_config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=ref_config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=ref_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(ref_config, 0) if ref_config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            del gpt_model

            gpt_model = GPTModel(
                config=overlap_config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    def test_transformer_layer_overlap(self, dispatcher_type, flex_backend, fp8_flag):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """
        extra_kwargs = {}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, 0) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(
                    gpt_model, input_tensors, microbatches
                )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.skipif(
        not is_nccl_ep_zero_copy_available(), reason="NCCL EP zero-copy TE API is not available"
    )
    @pytest.mark.skipif(
        not is_op_fuser_available(), reason="op-fuser (static-shape/zero-copy) needs TE>=2.14"
    )
    def test_transformer_layer_overlap_zero_copy(self):
        """ncclEP zero-copy under 1F1B a2a overlap must match the non-overlap reference.

        Validates the overlap dispatch-backward path (_StageDispatchBwdGrad staging into the symm
        buffer + the free_input symm guard): zero-copy stays enabled in both runs, so this isolates
        the overlap schedule. bf16 op-fuser (SwiGLU, tp=1) -- no fp8/Blackwell dependency.
        """
        extra_kwargs = {}
        apply_flex_backend_kwargs(extra_kwargs, "flex", "ncclep")
        extra_kwargs.update(
            moe_ncclep_zero_copy=True,
            moe_ncclep_static_shape=True,
            use_transformer_engine_op_fuser=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            # gates the zero-copy dispatch-bwd staging (_StageDispatchBwdGrad); the a2a-overlap run
            # exercises the 1F1B schedule that this flag represents in production.
            overlap_moe_expert_parallel_comm=True,
        )
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config, use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            capture_ref = run_transformer_layer_ref_with_capture(
                gpt_model, input_tensors, microbatches
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                gpt_model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

        # zero-copy sets process-global ncclEP state (ep bootstrap mode + shared symm classvars);
        # reset so later non-zero-copy ncclEP tests in this process are not polluted.
        from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize
        from megatron.core.transformer.moe.token_dispatcher import _NCCLEPManager

        nccl_ep_finalize()
        _NCCLEPManager._zc_fwd_token_buf = None
        _NCCLEPManager._zc_bwd_token_buf = None
        _NCCLEPManager._zc_recv_topk_weights_buf = None

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    @pytest.mark.parametrize("dispatcher_type,flex_backend", get_valid_dispatcher_configs())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    def test_mtp_layer_overlap(self, dispatcher_type, flex_backend, fp8_flag):
        """
        Verifies all-to-all overlap optimization in MTP layer produces
        the same results as the reference implementation.
        """
        extra_kwargs = {"mtp_num_layers": 1, "mtp_loss_scaling_factor": 1.1}
        apply_flex_backend_kwargs(extra_kwargs, dispatcher_type, flex_backend)
        if fp8_flag is not None:
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
            extra_kwargs["fp8"] = fp8_flag[0]
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 1
        seq_len = 32
        with deterministic_mode():
            # init models
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=16,
                moe_grouped_gemm=True,
                qk_layernorm=True,
                multi_latent_attention=True,
            )
            mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, True)
            if mtp_block_spec is None:
                # only last rank has mtp block
                assert True
                return
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                mtp_block_spec=mtp_block_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            gpt_model.decoder.final_layernorm = None
            gpt_model.cuda()
            params = reset_model(gpt_model)

            # build input data
            data = list(range(seq_len))
            hidden_states = [build_data(seq_len) for _ in range(microbatches)]
            input_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            labels = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            position_ids = torch.tensor(data, dtype=torch.int64).repeat((1, 1)).cuda()
            attention_mask = torch.ones((1, 1, seq_len, seq_len), dtype=bool).cuda()
            # get rotary pos emb
            _, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, _, _padding_mask = (
                gpt_model._preprocess(input_ids, position_ids)
            )
            # reset model
            params = reset_model(gpt_model)

            # run reference implementation
            capture_ref = run_mtp_layer_ref_with_capture(
                model=gpt_model,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                microbatches=microbatches,
            )
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_mtp_layer_a2a_overlap_with_capture(
                model=gpt_model,
                hidden_states=hidden_states,
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                microbatches=microbatches,
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
