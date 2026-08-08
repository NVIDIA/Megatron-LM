---
name: Bug report
about: Create a report to help us improve the repository or project
title: ""
labels: bug
assignees: ''

---

**Describe the bug**

A clear and concise description of what the bug is. Tag @NVIDIA/mcore-oncall
to get oncall's attention to this issue.

**Scenario / affected area**

Tell us what workflow is affected, for example: model family, training vs.
inference, dataset/checkpoint assumptions, precision, and parallelism setup
(TP/PP/DP/CP/EP/FSDP).

**Steps/Code to reproduce bug**

Please list *minimal* steps or a code snippet for us to reproduce the bug.
Include the exact command, script, config, and any required input assumptions.
Please also say whether this reproduces on the latest `main` branch.

A helpful guide on how to craft a minimal bug report http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports.


**Expected behavior**

A clear and concise description of what you expected to happen.


**Actual behavior / logs**

Add the full stack trace, error message, or relevant log excerpt.


**Environment (please complete the following information):**
 - Megatron-LM commit ID or release
 - Container image or OS
 - PyTorch version
 - CUDA version
 - NCCL version
 - GPU type and number of GPUs/nodes


**Additional context**

Add any other context about the problem here.
