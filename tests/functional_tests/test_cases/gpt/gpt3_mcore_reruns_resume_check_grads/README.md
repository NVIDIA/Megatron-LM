## Gradient tests
### Overview of the components of this test
* Trains for one step to collect gradients
* Gradients are extracted from the optimizer state in the checkpoint. You can get
this behavior by setting adam's b1, b2 to 0, so each step will override the previous adam 
state. The gradients for the previous step are stored in adam's first momentum 
state. Make sure to include `--adam-beta1: 0.0` and `--adam-beta2: 0.0`.
* script for comparing grads is currently (2025/09/16) tuned to some degree for
GPT models. The key thing that has to change with other models is the logic for 
determining which layers are row parallel linear. Those layers are sharded 
differently and need a different reshaping function to compare between the
non-model-parallel base case and the model parallel case. The script is located
currently in `tests/functional_tests/python_test_utils/test_optimizer_grads_match.py`.
* The test script currently relies on the older optimizer 
checkpoint format. For now make sure to add 
`--dist-ckpt-save-pre-mcore-014: true` to your test runs.
* You should disable randomization such as dropout which would have different
patterns with a single global batch and/or model/data parallel shards of the 
features.
* You can use this approach to test different configurations that are expected 
to result in the same gradients, not just model parallel configurations.
* To add a new test that follows this pattern, you can copy/modify this directory and
register the test in a similar way into `tests/functional_tests/test_utils/recipes/gpt-grads.yaml`
or something similar.