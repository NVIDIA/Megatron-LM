# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import pytest
import torch
import gc

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec, get_gpt_mtp_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.utils import is_te_min_version
from megatron.core.pipeline_parallel.combined_1f1b import schedule_chunk_1f1b
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.transformer.module import float16_to_fp32

from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.a2a_overlap.utils import compare_captures, deterministic_mode, get_test_config

def build_model(config):
    seq_len = 32
    max_seq_len = 300
    # ids = random.sample([i for i in range(max_seq_len)], seq_len)
    ids = [i for i in range(seq_len)]

    # build input tensors
    data = {
        "input_ids": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "labels": torch.tensor(ids, dtype=torch.int64).repeat((1, 1)).cuda(),
        "position_ids": torch.tensor([i for i in range(seq_len)], dtype=torch.int64).repeat((1, 1)).cuda(),
        "attention_mask": torch.ones(
            (1, 1, seq_len, seq_len), dtype=bool
        ).cuda(),
    }
    
    # build layer spec
    transformer_layer_spec = get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True
    )
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec.layer_specs[-1], True)
    
    # build model
    gpt_model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=100,
        pre_process=True,
        post_process=True,
        max_sequence_length=max_seq_len,
    )
    f_schedule_plan = gpt_model.build_schedule_plan(**data)
    return gpt_model, f_schedule_plan, data


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
    # TODO: Add flex dispatcher test back in when CI image installs DeepEP.
    @pytest.mark.parametrize("dispatcher_type", ["alltoall"])
    @pytest.mark.parametrize("fp8", ["e4m3", None])
    @pytest.mark.parametrize("fp8_recipe", ["blockwise"])
    @pytest.mark.parametrize("layers", [[2,1], [1,2], [1,1]])
    def test_1f1b_schedule_mtp_model_chunk(self, dispatcher_type, fp8, fp8_recipe, layers):
        """
        Verifies all-to-all overlap optimization in transformer layer produces
        the same results as the reference implementation.
        """
        microbatches = 1

        gpt_models = []
        schedule_plans = []
        ref_captures = []
        datas = []
        
        # create TransformerConfig
        extra_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
        }
        if dispatcher_type == "flex":
            extra_kwargs["moe_enable_deepep"] = True
            extra_kwargs["moe_router_dtype"] = "fp32"
        if fp8 is not None:
            extra_kwargs["fp8_recipe"] = fp8_recipe
            extra_kwargs["fp8"] = fp8
        with deterministic_mode():
            for layer_num in layers:
                output_tensors = []
                # build config
                config = get_test_config(num_layers = layer_num, extra_kwargs = extra_kwargs)
                # build model
                gpt_model, schedule_plan, data = build_model(config)
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
            # run a2a overlap
            capture_0 = {"outputs": []}
            capture_1 = {"outputs": []}
            a2a_captures = [capture_0, capture_1]
            for i in range(microbatches):
                # 1st forward
                if i > 0:
                    assert schedule_plans[0].pre_process is None, "pre_process should be released after backward"
                    schedule_plans[0] = gpt_models[0].build_schedule_plan(**datas[0])
                    schedule_plans[1] = gpt_models[1].build_schedule_plan(**datas[1])
                f_input_0 = schedule_chunk_1f1b(
                    schedule_plans[0],
                    None
                )
                capture_0["outputs"].append(f_input_0)
                # overlap
                f_input_1 = schedule_chunk_1f1b(
                    schedule_plans[1],
                    schedule_plans[0],
                    b_grad=torch.ones_like(f_input_0)
                )
                capture_1["outputs"].append(f_input_1)
                # last backward
                schedule_chunk_1f1b(
                    None,
                    schedule_plans[1],
                    b_grad=torch.ones_like(f_input_1)
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
    
