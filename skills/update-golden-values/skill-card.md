## Description: <br>
Refresh golden values from a GitHub Actions workflow run (failing-only or all jobs), score the change with average normalized relative differences, and produce a PR-ready summary. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and CI engineers use this skill to refresh golden-value baselines from GitHub Actions workflow runs and generate per-metric relative-difference summaries for pull request descriptions. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Megatron-LM Repository](https://github.com/NVIDIA/Megatron-LM) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis, Files] <br>
**Output Format:** [Markdown with inline bash code blocks and CSV] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
core_v0.15.0rc7 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
