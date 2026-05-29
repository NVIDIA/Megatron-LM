## Description: <br>
Bump the NVIDIA PyTorch base image (`nvcr.io/nvidia/pytorch:YY.MM-py3`) used by Megatron-LM CI, covering the two pin sites (GitHub CI and GitLab CI), the post-bump CI loop, and known gotchas. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and CI engineers upgrading the PyTorch base container for Megatron-LM continuous integration, ensuring both GitHub and GitLab CI pins stay in sync and functional tests are refreshed. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Megatron-LM Repository](https://github.com/NVIDIA/Megatron-LM) <br>
- [update-golden-values skill](../update-golden-values/SKILL.md) <br>
- [build-and-dependency skill](../build-and-dependency/SKILL.md) <br>
- [cicd skill](../cicd/SKILL.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML snippets] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
core_v0.15.0rc7 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
