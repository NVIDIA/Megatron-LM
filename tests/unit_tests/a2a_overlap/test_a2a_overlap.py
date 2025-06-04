# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import pytest
import torch

from contextlib import nullcontext

from megatron.core.models.gpt.fine_grained_schedule import LayerSchedulePlan, schedule_layer_1f1b
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec, get_gpt_decoder_block_spec
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.utils import is_te_min_version
from megatron.core.fp8_utils import get_fp8_context
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.a2a_overlap.utils import build_data, compare_captures, deterministic_mode, reset_model, get_test_config, DummyState


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
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    output_tensors = []
    for i in range(iterations):
        output = model(input_tensors[i].clone())[0]
        output_tensors.append(output)
        output.backward(torch.ones_like(output))

    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
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
    for i in range(len(input_tensors)):
        input_tensors[i] = input_tensors[i].clone()
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    event = torch.cuda.Event()
    comp_stream = torch.cuda.current_stream()
    com_stream = torch.cuda.Stream(device="cuda")
    layers = [
        LayerSchedulePlan(
            model,
            event,
            DummyState(),
            comp_stream,
            com_stream,
            extra_args={"is_moe": True, "enable_deepep": False},
        )
        for _ in range(microbatches)
    ]
    output_tensors = []

    # forward for 1st microbatch
    output, _ = schedule_layer_1f1b(layers[0], None, f_input=input_tensors[0], b_grad=None)
    output_tensors.append(output)
    torch.cuda.synchronize()
    # overlapped forward and backward
    for i in range(1, microbatches):
        f_input, b_grad = schedule_layer_1f1b(
            layers[i], layers[i - 1], f_input=input_tensors[i], b_grad=torch.ones_like(output)
        )
        output_tensors.append(f_input)
        torch.cuda.synchronize()
    # backward for last microbatch
    schedule_layer_1f1b(None, layers[-1], f_input=None, b_grad=torch.ones_like(output))
    torch.cuda.synchronize()
    capture = {"outputs": output_tensors}
    for name, param in model.named_parameters():
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
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    mtp_block = model.mtp

    output_tensors = []
    for i in range(microbatches):
        output = mtp_block(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            hidden_states=hidden_states[i].clone(),
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            embedding=model.embedding,
            output_layer=model.output_layer,
            compute_language_model_loss=model.compute_language_model_loss,
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
    for module in model.modules():
        if hasattr(module, 'fuse_wgrad_accumulation'):
            module.fuse_wgrad_accumulation = False

    comp_stream = torch.cuda.current_stream()
    com_stream = torch.cuda.Stream(device="cuda")
    layers = []
    for _ in range(microbatches):
        state = DummyState()
        state.labels_for_mtp = labels
        state.input_ids = input_ids
        state.position_ids = position_ids
        state.attention_mask = attention_mask
        state.rotary_pos_emb = rotary_pos_emb
        state.rotary_pos_cos = rotary_pos_cos
        state.rotary_pos_sin = rotary_pos_sin
        state.model = model
        event = torch.cuda.Event()
        layers.append(
            LayerSchedulePlan(
                model.mtp.layers[0],
                event,
                state,
                comp_stream,
                com_stream,
                extra_args={"is_moe": True, "enable_deepep": False, "is_first_layer": True, "is_last_layer": True},
            )
        )
    output_tensors = []
    # forward for 1st microbatch
    f_input, _= schedule_layer_1f1b(layers[0], None, f_input=hidden_states[0], b_grad=None)
    output_tensors.append(f_input)
    torch.cuda.synchronize()
    # overlapped forward and backward
    for i in range(1, microbatches):
        f_input, b_grad = schedule_layer_1f1b(
            layers[i], layers[i - 1], f_input=hidden_states[i], b_grad=torch.ones_like(f_input)
        )
        output_tensors.append(f_input)
        torch.cuda.synchronize()
    # backward for last microbatch
    schedule_layer_1f1b(None, layers[-1], f_input=None, b_grad=torch.ones_like(f_input))
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
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
        }
        config = get_test_config(num_moe_experts=None, extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config,
                use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            model = gpt_model.decoder.layers[0]

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, model.layer_number - 1) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(model, input_tensors, microbatches)
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
    
    
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_transformer_layer_overlap_shared_expert(self):
        """
        Verifies all-to-all overlap optimization in transformer layer with shared expert produces
        the same results as the reference implementation.
        """
        extra_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "moe_shared_expert_intermediate_size": 1024,
        }
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config,
                use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            model = gpt_model.decoder.layers[0]

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, model.layer_number - 1) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(model, input_tensors, microbatches)
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"

    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    # TODO: Add flex dispatcher test back in when CI image installs DeepEP.
    @pytest.mark.parametrize("dispatcher_type", ["alltoall"])
    @pytest.mark.parametrize("fp8", ["e4m3", None])
    @pytest.mark.parametrize("fp8_recipe", ["tensorwise"])
    def test_transformer_layer_overlap(self, dispatcher_type, fp8, fp8_recipe):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """
        
        extra_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
        }
        if dispatcher_type == "flex":
            extra_kwargs["moe_enable_deepep"] = True
            extra_kwargs["moe_router_dtype"] = "fp32"
        if fp8 is not None:
            extra_kwargs["fp8_recipe"] = fp8_recipe
            extra_kwargs["fp8"] = fp8
        config = get_test_config(extra_kwargs=extra_kwargs)
        microbatches = 4
        with deterministic_mode():
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config=config,
                use_transformer_engine=True
            )
            gpt_model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=100,
                pre_process=True,
                post_process=True,
                max_sequence_length=300,
            )
            model = gpt_model.decoder.layers[0]

            params = reset_model(gpt_model)
            input_tensors = [build_data() for _ in range(microbatches)]

            fp8_context = get_fp8_context(config, model.layer_number - 1) if config.fp8 else nullcontext()
            with fp8_context:
                capture_ref = run_transformer_layer_ref_with_capture(model, input_tensors, microbatches)
            reset_model(gpt_model, params)
            capture_a2a_overlap = run_transformer_layer_a2a_overlap_with_capture(
                model, input_tensors, microbatches
            )
            comp_res = compare_captures(capture_ref, capture_a2a_overlap, True)
            assert comp_res[0], f"[rank {torch.distributed.get_rank()}] {comp_res[1]}"
