# Data Pipeline

## FIM dataset

`GPTFIMDataset` extends Megatron-Core’s `GPTDataset` to support **Fill-in-the-Middle (FIM)** data augmentation.
It probabilistically converts samples into FIM format using configurable rates, with support for both PSM and SPM patterns, fragment-level splitting, and length-preserving output.

`GPTFIMDatasetConfig` provides the configuration needed to enable this behavior.
`GPTFIMDatasetConfig` configuration object extending `GPTDatasetConfig` to enable FIM preprocessing.

**Attributes**

- `rate`: Probability of converting a sample into a FIM example. A value of `1.0` means FIM is always applied. a value of `0.0` means FIM is never applied.
- `spm_rate`: Probability of using the SPM FIM pattern (vs PSM). The remaining probability (`1 - spm_rate`) selects the PSM (prefix-suffix-middle) pattern instead. For example, if `spm_rate = 0.3`: 30% SPM, 70% PSM.
- `extra_tokens`: Dictionary containing the FIM special tokens: {"prefix", "middle", "suffix", "pad", "eod"}.
- `split_sample`: Optional token around which samples are split before applying FIM. If provided, the input sequence is divided at every occurrence of this token, and FIM is applied independently to each fragment. `A B C <SPLI_SAMPLE> D E F <SPLIT_SAMPLE> G H` -> `FIM(Fragment 1) <SPLI_SAMPLE> FIM(Fragment 2) <SPLI_SAMPLE> FIM(Fragment 3)`.
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
## Varlen dataset

`VarlenDataset` packs SFT-style instruction data of widely varying lengths into
THD (variable-length) format. It extends the `SFTDataset` family, so it reuses
the same packing / `cu_seqlens` / context-parallel padding logic, and is
selected independently of `--sft`.

This section documents the **input schemas** it accepts. Loading and packing are
described with the dataset classes themselves.

### Schema auto-detection

The input layout is inferred from the dataset's column names, most explicit
first. The first match wins:

| Schema | Detected by | Normalized to |
|---|---|---|
| `openai-messages` | a `messages` column | passed through |
| `sharegpt` | a `conversations` column | messages list |
| `alpaca` / `dolly` | an instruction column **and** an output column | 3-turn messages list |
| `pretrain-text` | a `text` column | raw string (no chat template) |

Column names accepted for the alpaca/dolly layout:

- instruction: `instruction`, `prompt`, `query`, `question`
- output: `output`, `response`, `completion`, `answer`
- optional extra user-turn context: `input` (Stanford Alpaca), `context` (Dolly)

If none of the four layouts match, dataset construction fails with a `ValueError`
listing the columns it saw and the schemas it supports.

### Normalization rules

The three instruction-tuning layouts are converted to the messages list the
parent `SFTDataset` expects:

- **A leading `system` turn is guaranteed.** An empty one is prepended when the
  sample does not already start with a system turn, so
  `SFTDataset._split_conversations` treats each sample as one conversation.
- **ShareGPT speakers are mapped to roles** via the `from` field:
  `human`/`user` → `user`; `gpt`/`assistant`/`model`/`chatgpt`/`bing`/`bard` →
  `assistant`; `system` → `system`; `tool`/`function`/`observation` → `tool`.
  Unrecognized speakers fall back to `user` rather than failing.
- **Alpaca/Dolly context is folded into the user turn**, joined to the
  instruction by a blank line when present.
- **Non-`role`/`content` keys are dropped** from `messages` samples (e.g.
  `name`, `tool_calls`); they are not part of the chat-template input.

`pretrain-text` is the exception: it returns the `text` column unchanged as a
plain string, and the dataset dispatches on that to skip chat templating and
prompt masking. This supports long-context pretraining corpora (Dolma, OLMo
midtraining) packed through the same THD path as SFT.

### Limitations

- Turn content must be a plain string. Multi-modal samples that carry content as
  a list of image/text parts raise a `ValueError`.
- Non-string values in instruction/output fields raise a `ValueError` rather
  than being coerced.
