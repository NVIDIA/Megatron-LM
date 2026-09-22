# MSA Primitive Skill

Define, implement, use, and validate MiniMax Sparse Attention (per-token block-sparse GQA attention).

## Schema

<!-- MLITE_SKILL_SCHEMA_BEGIN -->
```python
schema = Skill(
    "primitive.module.msa", kind="primitive", purpose="define and validate MiniMax sparse attention primitive",
    imports=["basic.constitution"], calls=["primitive.contract", "primitive.validate", "primitive.module.gqa"],
    inputs=["task", "implementation", "config", "reference", "budget"],
    outputs=["principle", "implementation_contract", "usage_contract", "validation", "risks"],
    exits=["done", "blocked", "out_of_scope"],
)
```
<!-- MLITE_SKILL_SCHEMA_END -->

```python
def msa(task, implementation, config, reference, budget):
    contract = primitive.contract(implementation.msa, scope=task.scope, reference=reference)
    if not contract.done:
        return blocked("MSA contract not satisfied", evidence=contract)
    principle = {
        "semantics": "each query token attends to the top-k 128-token KV blocks chosen by a lightning indexer, "
                     "own block always included, keys j > i masked; selection shared by the query heads of a GQA group",
        "invariants": [
            "selection is per query TOKEN (not per query block); block ids compared as sets",
            "seq_len <= topk * block_size  =>  output == dense causal attention",
            "indexer is a pure selector: no gradient path, no auxiliary loss (frozen, requires_grad=False)",
            "index-Q sharded like KV heads, index-K single head replicated on every TP rank",
        ],
        "reference": reference or "HF MiniMaxM3VLAttention (bf16, same weights) and the fp32 masked-dense attention on the same block selection",
    }
    implementation_contract = {
        "details": ["MSAIndexer (Gemma-norm + partial RoPE + fp32 block max-pool + top-k); MSAIndexer.index_qk = projections only",
                    "flex backend (kernels/msa_kernels.py): torch flex_attention on the per-token selection table; TP + all-gather CP",
                    "magi backend (kernels/magi_msa.py): MagiAttention MSA extension + MM-Sparse-Attention msa_v1 kernels;"
                    " MSAttention.forward bypasses indexer scoring / gather / msa_core_attention and calls calc_msa",
                    "MSAttention = GQAttention projections/QK-norm/RoPE + sparse core"],
        "state": ["index_n_heads == num_key_value_heads", "index_head_dim", "block_size", "topk_blocks", "local_blocks", "backend",
                  "magi: MagiMsaContext (runtime key, dense key, doc-local position_ids) built per micro-batch by the protocol"],
        "boundaries": ["indexer owns selection; kernels own masking; GQA owns projections and TP sharding; "
                       "CP (flex): all-gather K/V + index-K to global order (zigzag shards, any alignment); indices/causal in global positions",
                       "CP (magi): Magi owns the load-balanced token dispatch, required-K / index-K communication, indexer and sparse kernels;"
                       " mlite owns projections, norms, RoPE (by dispatched position), indexer projections (no_grad) and the output linear;"
                       " dense layers run native calc_attn on the same dispatch layout"],
    }
    usage_contract = {
        "config": require_config_keys(config, ["index_n_heads", "index_head_dim", "index_block_size", "index_topk_blocks"]),
        "choose_when": ["layer_types[i] == 'minimax_m3_sparse'"],
        "avoid_when": ["packed/THD sequences on flex (rejected; supported by magi)", "cuDNN BSA-style per-query-block metadata (not equivalent)",
                       "magi with attention TP>1 or shapes other than 64/4/4 heads, D=128, block 128, top-16 (kernel lock)"],
        "compose_with": ["primitive.module.gqa", "primitive.parallel.tp", "primitive.parallel.cp"],
    }
    gqa_validation = primitive.module.gqa(task, implementation=implementation, config=config, reference=reference, budget=budget.gqa)
    if not gqa_validation.done:
        return blocked("MSA depends on a validated GQA projection path", evidence=gqa_validation)
    validation = primitive.validate(task, primitive=implementation.msa, implementation=implementation, budget=budget)
    risks = ["top-k ties flip under kernel/dtype changes (compare sets; bf16 flips 5-17% of rows, so gates are loose and "
             "flip rates are printed as evidence)",
             "flex BlockMask rebuilt per call (perf)",
             "CP (flex) all-gathers full K/V per layer (memory O(S_full) per rank, not overlapped)",
             "indexer scores are pooled chunk-wise and the BlockMask is built from kv blocks (no S_q x S_k tensors)",
             "magi: bf16-only kernels; indexer freeze must be requires_grad=False + no_grad "
             "(kl_loss_coeff=0 is not a freeze) and is asserted bitwise after training; last_block_indices is None; a new cu_seqlens "
             "re-plans the runtime on the host (LRU 16, MAGI_MSA_RUNTIME_CACHE_SIZE); padding up to chunk*cp-1 tokens; "
             "recompute core_attn re-runs Magi's forward communication; dense layers run native calc_attn on FA4 (flash_attn_cute, cutlass-dsl 4.5.2); the pure-torch sdpa_ol backend is test-only (OOM beyond ~8K)",
             "deterministic=True (ImplConfig -> MagiMsaSettings -> MsaConfig; Magi-MSA >= c829982): ordered msa_v1 backward, MSA path bitwise reproducible at CP1/2/4 (kernel bwd ~4x, all-MSA step ~1.5x); the dense layers' fa4 backward stays unordered, so full-model bitwise needs dense sdpa_ol"]
    if not validation.done:
        return blocked("MSA validation failed", evidence=validation)
    return done(principle=principle, implementation_contract=implementation_contract,
                usage_contract=usage_contract, validation=[gqa_validation, validation], risks=risks)
```
