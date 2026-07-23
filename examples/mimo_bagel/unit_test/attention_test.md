claude code prompt:
reference tests/unit_tests/transformer/test_attention.py，design a unit test for SelfAttentionMoT module in examples/bagel/model/attention_mot.py. 
the main target is compare the output accuracy between SelfAttentionMoT and PackedAttentionMoT in bagel-package/bagel/modeling/bagel/qwen2_navit.py. 
the test python script should put in the folder as same as this file.
Notice:
1. the absolute error diff should keep smaller than 1e-3, best to achieve 1e-4, the relative error diff should be smaller than 1e-3. The input/output tensor's precision should be kept in float16.