# Full-size Puzzle checkpoint-resume test

[puzzle.py](puzzle.py) defines the full 75B-A9B model as layer configs, including
heterogeneous MoE layers and one inferred MTP head. It initializes from scratch;
no published weights are downloaded.

The [nightly GB200 recipe](../../../../test_utils/recipes/gb200/nemotron.yaml) runs
one training/resume cycle on four nodes of four GPUs, with TP1/PP1/CP1/EP16/ETP1.
It trains 20 steps, then resumes the step-10 checkpoint through step 20, restoring
optimizer and RNG state. The existing harness compares resumed LM and MTP losses
against the first run. There are no external golden values or performance gates.
