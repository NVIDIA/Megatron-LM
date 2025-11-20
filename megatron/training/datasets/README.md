# Data Pipeline

## FIM dataset

`GPTFIMDataset` extends Megatron-Core’s `GPTDataset` to support **Fill-in-the-Middle (FIM)** data augmentation.
It probabilistically converts samples into FIM format using configurable rates, with support for both PSM and SPM patterns, fragment-level splitting, and length-preserving output.

`GPTFIMDatasetConfig` provides the configuration needed to enable this behavior.
`GPTFIMDatasetConfig` configuration object extending `GPTDatasetConfig` to enable FIM preprocessing.

**Attributes**

- `rate`: Probability of converting a sample into a FIM example. A value of `1.0` means FIM is always applied. a value of `0.0` means FIM is never applied.
- `spm_rate`: Probability of using the SPM FIM pattern (vs PSM).
- `extra_tokens`: Dictionary containing the FIM special tokens: {"prefix", "middle", "suffix", "pad", "eod"}.
- `split_sample`: Optional token around which samples are split before applying FIM.
- `fragment_rate`: Probability of applying FIM to each fragment when split_sample is used.
- `no_prefix`: If the decoded sequence starts with this prefix, FIM is skipped.
`GPTFIMDataset` dataset class that loads token sequences from an `IndexedDataset` and applies FIM transformations before returning each sample.

**PSM Format**
```
[prefix_tok] prefix [suffix_tok] suffix [middle_tok] middle
```

**SPM Format**
```
[prefix_tok, suffix_tok] suffix [middle_tok] prefix middle
```

**Special cases:**

- If the sequence starts with no_prefix, FIM is skipped.
- If FIM is not applied, the sample is returned unchanged.