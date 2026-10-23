# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from tests.unit_tests.inference.engines import batch_invariant_test_utils as bi


def test_mla_split_device_first_prefill():
    with bi.invariant_runtime(bi.MLA_CASE) as (backend, version):
        engine = bi.build_engine(bi.MLA_CASE, backend, version)
        model = engine.controller.inference_wrapped_model.model
        assert version == 3 and model.config.use_cpu_initialization
        layers = [layer for layer in model.modules() if isinstance(layer, bi.MLASelfAttention)]
        sources = [layer.linear_kv_up_proj for layer in layers]
        assert len(sources) == 2 and all(source.weight.is_cuda for source in sources)
        params = bi.SamplingParams(num_tokens_to_generate=1, top_k=1, termination_id=-1)
        engine.add_request(bi.TARGET, bi.target_prompt(bi.MLA_CASE.prompt_length), params)
        requests = engine.step_modern()["finished_requests"]
        assert len(requests) == 1 and not engine.has_unfinished_requests()
        for layer, source in zip(layers, sources):
            norm, linear = layer.kv_layernorm, layer.linear_kv_up_proj_linear
            pairs = [(norm.weight, source.layer_norm_weight), (linear.weight, source.weight)]
            for actual, expected in pairs:
                assert actual.device == expected.device == source.weight.device
                assert actual.dtype == expected.dtype == bi.torch.bfloat16
                assert bi.torch.equal(actual, expected)
