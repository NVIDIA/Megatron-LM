---
name: REGRESSION
about: Report a regression in speed or accuracy due to a Megatron-LM update
title: "[REGRESSION]"
labels: ''
assignees: ''

---

**Describe the regression**
A clear and concise description of what the regression is. Tag @NVIDIA/mcore-oncall
to get oncall's attention to this issue.

**Scenario / affected area**
Tell us what workflow is affected, for example: model family, training vs.
inference, dataset/checkpoint assumptions, precision, and parallelism setup
(TP/PP/DP/CP/EP/FSDP).

**To Reproduce**
Steps to reproduce the behavior. Include the exact command, script, config, and
any required input assumptions. Please also say whether this reproduces on the
latest `main` branch. The easier it is to reproduce the faster it will get
maintainer attention.

**Previous performance**
What speed, memory use, or accuracy did you previously see. Include run-to-run
variance if known.

**New performance**
What speed, memory use, or accuracy do you see after the update. Include
run-to-run variance if known.

**Stack trace/logs**
If applicable, add the stack trace or logs related to the regression.

**Environment (please complete the following information):**
 - Previous Megatron-LM commit ID
 - New Megatron-LM commit ID
 - Container image or OS
 - Previous PyTorch version
 - New PyTorch version
 - Previous CUDA version
 - New CUDA version
 - Previous NCCL version
 - New NCCL version
 - GPU type and number of GPUs/nodes

**Proposed fix**
If you have a proposal for how to fix the issue state it here or link to a PR.

**Additional context**
Add any other context about the problem here.
