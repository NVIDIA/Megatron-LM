# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.

import os
try:
    import habana_frameworks.torch.gpu_migration
except:
    pass

import pytest

# Key in the expected_fail_tests can be an exact node_id or module or directory
def test_in_xfail_dict(test_dict, nodeid):
    for key in test_dict:
        if key.endswith('::') or key.endswith('.py') or key.endswith('/'):
            if nodeid.startswith(key):
                return True
        elif key == nodeid:
            return True

    return False

def get_reason_for_xfail(test_dict, nodeid):
    for key in test_dict:
        if key.endswith('::') or key.endswith('.py') or key.endswith('/'):
            if nodeid.startswith(key):
                return test_dict[key]
        elif key == nodeid:
            return test_dict[key]

    return ""

unit_tests_to_deselect = {
    'https://jira.habana-labs.com/browse/SW-201768': [
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-None]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-9999]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-10000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-10001]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-19999]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[False-True-20000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-True-None]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-True-9999]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-True-10000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-True-10001]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-True-19999]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_bucket_sizes[True-True-20000]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[False-True]',
        'tests/unit_tests/distributed/test_param_and_grad_buffer.py::test_grad_sync[True-True]',
    ],
    'https://jira.habana-labs.com/browse/SW-201767': [
        'tests/unit_tests/models/test_clip_vit_model.py::TestCLIPViTModel::test_constructor',
        'tests/unit_tests/models/test_llava_model.py::TestLLaVAModel::test_constructor',
        'tests/unit_tests/inference/test_modelopt_gpt_model.py::TestModelOptGPTModel::test_load_te_state_dict_pre_hook',
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_build_module',
    ],
    'https://jira.habana-labs.com/browse/SW-202749': [
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_forward_backward[1-8]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_forward_backward[4-2]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[1-8]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[8-1]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[4-2]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_forward_backward[1-1]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[1-8]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[8-1]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[4-2]',
        'tests/unit_tests/transformer/moe/test_a2a_token_dispatcher.py::TestAlltoAllDispatcher::test_capacity_padding_forward_backward[1-1]',
    ],
    'https://jira.habana-labs.com/browse/SW-202752': [
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_sliding_window_attention',
    ],
    'https://jira.habana-labs.com/browse/SW-202755': [
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_constructor',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_weight_init_value_the_same',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gpu_forward',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gpu_forward_with_no_tokens_allocated',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestParallelGroupedMLP::test_gradient_with_no_tokens_allocated',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_constructor',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_gpu_forward_backward',
        'tests/unit_tests/transformer/moe/test_grouped_mlp.py::TestTEGroupedMLP::test_gpu_forward_backward_with_no_tokens_allocated'
    ],
    'https://jira.habana-labs.com/browse/SW-206537': [
        'tests/unit_tests/dist_checkpointing/test_flattened_resharding.py',
    ],
    'https://jira.habana-labs.com/browse/SW-206543': [
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_memory_usage[cuda]',
        'tests/unit_tests/dist_checkpointing/test_fully_parallel.py::TestFullyParallelSaveAndLoad::test_memory_usage[cpu]',
    ],
    'https://jira.habana-labs.com/browse/SW-206546': [
        'tests/unit_tests/dist_checkpointing/test_optimizer.py',
    ],
    'https://jira.habana-labs.com/browse/SW-206557': [
        'tests/unit_tests/transformer/test_rope.py::TestRotaryEmbedding::test_constructor',
        'tests/unit_tests/transformer/test_rope.py::TestRotaryEmbedding::test_gpu_forward',
        'tests/unit_tests/transformer/test_rope.py::TestRotaryEmbedding::test_cpu_forward',
        'tests/unit_tests/transformer/moe/test_sequential_mlp.py::TestParallelSequentialMLP::test_gpu_forward',
        'tests/unit_tests/transformer/test_mlp.py::TestParallelMLP::test_gpu_forward',
    ],
    'https://jira.habana-labs.com/browse/SW-206558': [
        'tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_constructor',
        'tests/unit_tests/models/test_multimodal_projector.py::TestMultimodalProjector::test_constructor',
    ],
    'https://jira.habana-labs.com/browse/SW-206559' : [
        'tests/unit_tests/data/test_preprocess_data.py::test_preprocess_data_bert',
    ],
    'https://jira.habana-labs.com/browse/SW-206560': [
        'tests/unit_tests/transformer/test_spec_customization.py::TestSpecCustomization::test_transformer_block_custom',
    ],
    'https://jira.habana-labs.com/browse/SW-206561':[
        'tests/unit_tests/test_utils.py::test_straggler_detector',
    ],
    'https://jira.habana-labs.com/browse/SW-206562' : [
        'tests/unit_tests/fusions/test_torch_softmax.py::TestTorchSoftmax::test_causal_mask_equal_scores',
    ],
    'https://jira.habana-labs.com/browse/SW-208138' : [
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[8-1-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[4-2-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[1-1-8]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[2-1-4]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_allgather_dispatcher[2-2-2]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[8-1-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[4-2-1]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[1-1-8]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[2-1-4]',
        'tests/unit_tests/transformer/moe/test_aux_loss.py::TestAuxLoss::test_a2a_dispatcher[2-2-2]',
    ]
}

unit_tests_to_deselect_eager_only = {
    'https://jira.habana-labs.com/browse/SW-TODO': [
        'tests/unit_tests/inference/', #Fails to exit gracefully 9/11 passed.
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch', #Fails to exit gracefully
    ],
    'https://jira.habana-labs.com/browse/SW-206331': [
        'tests/unit_tests/transformer/test_module.py::TestFloat16Module::test_fp16_module',
    ],
    'https://jira.habana-labs.com/browse/SW-206335': [
        'tests/unit_tests/transformer/test_module.py::TestFloat16Module::test_bf16_module',
    ],
    'https://jira.habana-labs.com/browse/SW-206337': [
        'tests/unit_tests/transformer/test_attention.py::TestParallelAttention::test_fused_rope_gpu_forward',
    ],
    'https://jira.habana-labs.com/browse/SW-206551' : [
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModel::test_sharded_state_dict_save_load[get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModel::test_sharded_state_dict_save_load[get_gpt_layer_local_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-dp-pp-tp-pp-dp-src_tp_pp5-dest_tp_pp5-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[True-tp-dp-pp-tp-dp-pp-src_tp_pp6-dest_tp_pp6-get_gpt_layer_local_spec-get_gpt_layer_with_transformer_engine_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py::TestGPTModelReconfiguration::test_parallel_reconfiguration_e2e[False-tp-pp-dp-tp-pp-dp-src_tp_pp7-dest_tp_pp7-get_gpt_layer_with_transformer_engine_spec-get_gpt_layer_local_spec]',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-te-local]',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py::TestRetroModel::test_sharded_state_dict_save_load[retro-local-te]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5Model::test_sharded_state_dict_save_load[t5-te-local]',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py::TestT5Model::test_sharded_state_dict_save_load[t5-local-te]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBertModel::test_sharded_state_dict_save_load[dst_layer_spec0-src_layer_spec1]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBertModel::test_sharded_state_dict_save_load[dst_layer_spec1-src_layer_spec0]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBERTModelReconfiguration::test_parallel_reconfiguration_e2e[True-src_tp_pp5-dest_tp_pp5-src_layer_spec5-dst_layer_spec5]',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py::TestBERTModelReconfiguration::test_parallel_reconfiguration_e2e[False-src_tp_pp6-dest_tp_pp6-src_layer_spec6-dst_layer_spec6]',
    ],
    'https://jira.habana-labs.com/browse/SW-206537':[
        'tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py',
        'tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py',
    ],
}


unit_tests_to_deselect_lazy_only = {
    'https://jira.habana-labs.com/browse/SW-206540' : [
        'tests/unit_tests/inference/text_generation_controllers/test_simple_text_generation_controller.py::TestTextGenerationController::test_generate_all_output_tokens_static_batch',
        'tests/unit_tests/dist_checkpointing/test_serialization.py',
        'tests/unit_tests/dist_checkpointing/models/test_t5_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_gpt_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_grouped_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_mlp_glu.py',
        'tests/unit_tests/dist_checkpointing/models/test_retro_model.py',
        'tests/unit_tests/dist_checkpointing/models/test_sequential_mlp.py',
        'tests/unit_tests/dist_checkpointing/models/test_bert_model.py',
    ],
}


all_xfails_dict = {
        node_id: jira for jira in unit_tests_to_deselect for node_id in unit_tests_to_deselect[jira]}

eager_only_xfail_dict = {
        node_id: jira for jira in unit_tests_to_deselect_eager_only for node_id in unit_tests_to_deselect_eager_only[jira]}

lazy_only_xfail_dict = {
        node_id: jira for jira in unit_tests_to_deselect_lazy_only for node_id in unit_tests_to_deselect_lazy_only[jira]}

if os.getenv("PT_HPU_LAZY_MODE") == "0":
    all_xfails_dict.update(eager_only_xfail_dict)
else:
    all_xfails_dict.update(lazy_only_xfail_dict)

def pytest_collection_modifyitems(config, items):
    for item in items:
        if test_in_xfail_dict(all_xfails_dict, item.nodeid):
            reason_str = get_reason_for_xfail(all_xfails_dict, item.nodeid)
            xfail_marker = pytest.mark.xfail(run=False, reason=reason_str)
            item.user_properties.append(("xfail", "true"))
            item.add_marker(xfail_marker)
