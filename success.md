# PR #9073 validation report

## Result and scope

No correctness bug was established in the selected coverage for
[PR #9073](https://github.com/NVIDIA/Megatron-LM/pull/9073).
All **16 selected H100 unit buckets and seven selected functional cases** have
successful exact-head upstream CI receipts. Six additional local unit runs
completed successfully; one local transformer run timed out and is **not** a local pass.
This is risk-based validation, not a claim that the full suite passed locally or
that every required PR check is green.

- Tested source: `8248fe1347b021288dc49754b383e33f1bde25ac`.
- Comparison base: `45eec611cc9d1fd89a590c93e034c2f468ec1486`.
- The official CI image was built from merge `5fc7e6172c31768b05097c30241654485951d3c0`,
  whose tree matched the tested source tree. The retained image dependency inputs
  matched this candidate; no new image was built for this report.
- Upstream evidence: [CICD run 42249850591, attempt 1](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591).
- Follow-up scope: this report only, based on `pull-request/9073`; no product,
  test, marker, tolerance, golden value or CI-policy changes.

The requester narrowed the original full-suite request to likely failure paths.
Selection traced TE grouped-MLP precision into MoE/FP8/model consumers, GTP
prefetch into registration and checkpoint paths, MFSDPv1 router-buffer
materialization, the filelock integration, and simplified DSA into kernels,
metadata and shared inference-context/RoPE consumers. The exact H100 inventory
contained 51 unit buckets and 82 functional cases; 110 unselected identities
remain exclusions, not passes. Successful matching CI jobs were reused rather
than rerun locally.

## Exact-head CI evidence

Each row maps to one successful check in the run above.

| Native selector | Upstream check |
| --- | --- |
| `unit:tests/unit_tests/generalized_tensor_parallel/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820973) |
| `unit:tests/unit_tests/transformer/moe/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820727) |
| `unit:tests/unit_tests/transformer/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820408) |
| `unit:tests/unit_tests/distributed/mfsdp_v1/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820877) |
| `unit:tests/unit_tests/distributed/mfsdp_v1/test_mfsdp_fully_shard.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820529) |
| `unit:tests/unit_tests/determinism/kernels/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820592) |
| `unit:tests/unit_tests/models/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820179) |
| `unit:tests/unit_tests/test_fp8_param.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820024) |
| `unit:tests/unit_tests/inference/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820351) |
| `unit:tests/unit_tests/inference/engines/**/*.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820638) |
| `unit:tests/unit_tests/inference/engines/test_dynamic_engine.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820380) |
| `unit:tests/unit_tests/inference/engines/test_dynamic_engine_parallel_and_chunked_prefill.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820505) |
| `unit:tests/unit_tests/inference/test_hybrid_moe.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820279) |
| `unit:tests/unit_tests/inference/engines/test_batch_invariant_handoff.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820003) |
| `unit:tests/unit_tests/inference/engines/test_disagg_coordinator_pairwise.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820302) |
| `unit:tests/unit_tests/inference/test_disagg_transfer_pairwise.py` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820479) |
| `functional:moe/gpt3_mcore_te_tp1_pp1_te_4experts_groupedGEMM_op_fuser` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065678467) |
| `functional:gpt/gpt3_mcore_te_tp2_pp2_dsa` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065671258) |
| `functional:gpt/gpt_dynamic_inference_tp2_pp2_dp2_583m_logitsmatch_zmq` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065671325) |
| `functional:gpt/gpt_dynamic_inference_tp1_pp1_dp8_583m_long_prompt_logitsmatch_zmq` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065671244) |
| `functional:hybrid/hybrid_dynamic_inference_tp1_pp1_dp8_583m_chunked_prefill` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065671409) |
| `functional:moe/gpt_dynamic_inference_tp4_pp1_ep4_16B_mtp_sp_block_cudagraphs` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065678583) |
| `functional:mimo/mimo_hetero_gtp_checkpoint_round_trip` | [Passed](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128065678315) |

## Local H100 results

All local runs used one node with eight H100 GPUs and the unmodified native
recipe launcher. Published manifests and member checksums were validated.
Counts below describe one rank, not the sum of eight copies of the same tests.
Regular and experimental passes are shown separately.

| Unit bucket | Regular pass | Experimental pass |
| --- | --- | --- |
| MFSDPv1 catch-all | 97 passed, 12 skipped, 21 deselected | 130 deselected |
| Inference catch-all | 3950 passed, 113 skipped | 4060 deselected, three collection skips |
| Determinism kernels | 762 passed, 53 skipped, two xfailed, seven non-strict xpassed | 824 deselected |
| Transformer MoE | 899 passed, 104 skipped, 234 deselected | 204 passed, 1033 deselected |
| MFSDPv1 fully-shard | 249 passed, 95 skipped, six xfailed | 350 deselected |
| Models | 1192 passed, 21 skipped, 29 deselected | 24 passed, 1218 deselected |

An empty experimental pass does not negate executed tests in the regular pass.
Skips and expected failures are not passing test cases, and no equivalence to a
protected-base skip baseline is claimed.

### Incomplete local transformer run

The transformer catch-all hit the launcher timeout with a configured 4235-second
limit. All eight published rank logs were inspected: no named pytest failure or
assertion error was found. Rank 0 logged 1891 passing test statuses and 96 skips,
but no final pytest summary; the experimental pass did not complete. These
intermediate statuses do **not** make the local row successful. The launcher sent
SIGTERM, so the local result is infrastructure/no-verdict; the underlying reason
for the delayed completion was not established.

The [matching upstream H100 job](https://github.com/NVIDIA/Megatron-LM/actions/runs/42249850591/job/128003820408)
completed at the same source SHA: 1891 passed, 96 skipped and 94 deselected in the
regular pass, followed by 2081 deselected in the experimental pass. That exact-head
CI receipt satisfies the selected bucket; it does not erase the local timeout.
No unchanged local retry was needed after obtaining the successful native receipt.

## Limits and remaining PR gates

- H100 cannot exercise Blackwell-only fused kernels. No local GB200 or LTS
  coverage is claimed.
- Local inference skips included optional `dynamo` imports and Blackwell-only
  NVFP4 cases. These paths remain unvalidated locally, not silently waived.
- No selectable H100 L1 simplified-DSA inference functional case or MFSDPv1
  functional case was projected. DSA uses the successful exact-head transformer
  unit check plus shared-context functional coverage; MFSDPv1 uses both unit buckets.
- The original PR still lacks CODEOWNERS approvals. Its `codeowners-approval` and
  `multi-approval-bot-summary` checks failed; this is not a test failure. At the
  final check snapshot, two other functional jobs and `cicd-mbridge-testing` were
  still in progress. This report makes no claim that those jobs or the overall
  required-check gate passed.
- Formal reviews and inline review comments were empty at inspection. Existing
  issue comments contained the draft-workflow notice and test-approval commands.
- This report does not authorize merging the original PR or this follow-up.
