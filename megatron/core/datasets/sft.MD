# SFT Dataset

The `SFTDataset` class (`sft_dataset.py`) provides supervised fine-tuning over
pretokenized chat conversations stored in Megatron's `IndexedDataset` format
(`.bin` / `.idx` pairs).

---

## 1. `__getitem__` outputs

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `tokens` | `(S-1,)` | `int64` | Input token ids (all but the **last** token of the packed sample) |
| `labels` | `(S-1,)` | `int64` | Target token ids (all but the **first** token — i.e. `tokens` shifted right by one) |
| `loss_mask` | `(S-1,)` | `float32` | Per-token loss mask (`0.0`/`1.0`). Which tokens are trained on depends on config (see §3) |
| `cu_seqlens` | `(D+1,)` | `int32` | Cumulative sequence lengths. `D` is the number of documents packed into this sample. First element is always `0`, last element equals `S` (total tokens). Used for Flash Attention varlen kernels and to reset position ids per document |

`S` is the sum of the token lengths of every document packed into the sample
(always `<= sequence_length`).

---

## 2. Index-building logic

Three numpy arrays are built (or loaded from cache) at construction time:

### 2.1 Document index — `document_index`

A **1-D** `int32` array that maps flat positions to document ids in the
underlying `IndexedDataset`.

```
document_index = [docA, docB, docC, docD, docE, ...]
```

Built by flattening the output of `pack_samples`.

### 2.2 Sample index — `sample_index`

A **1-D** `int32` array of cumulative bin sizes. Its length is
`num_packed_samples + 1`. The documents belonging to packed sample `i` are:

```
document_index[ sample_index[i] : sample_index[i+1] ]
```

**Example** — 6 documents packed into 3 bins `[[0,2], [1,4,5], [3]]`:

```
document_index = [0, 2, 1, 4, 5, 3]
sample_index   = [0, 2, 5, 6]
                  │  │  │  └─ sample 2: document_index[5:6] → [3]
                  │  │  └──── sample 1: document_index[2:5] → [1, 4, 5]
                  │  └─────── sample 0: document_index[0:2] → [0, 2]
                  └────────── always starts at 0
```

### 2.3 Shuffle index — `shuffle_index`

A **1-D** `int32` array that is a random permutation of
`range(num_packed_samples)`, extended across multiple epochs when `num_samples`
exceeds the number of packed samples.

```python
for epoch in range(num_epochs):
    shuffle_index.extend(torch.randperm(num_packed))
```

### 2.4 Sample packing — `pack_samples` (MFFD)

Variable-length documents are packed into fixed-capacity bins using the
**Modified First-Fit Decreasing** algorithm. Documents longer than
`sequence_length` are dropped with a warning.

| Phase | Action |
|-------|--------|
| 0 | Classify items: **large** `(C/2, C]`, **medium** `(C/3, C/2]`, **small** `(C/6, C/3]`, **tiny** `(0, C/6]` |
| 1 | Open one bin per large item |
| 2 | Forward pass: add one medium item to each large bin if it fits |
| 3 | Backward pass: fill large-only bins with two small items |
| 4 | Greedy forward: fit remaining items into existing bins |
| 5 | FFD on leftovers: create new bins for anything still unplaced |

### 2.5 Caching

Indices are saved as `.npy` files under
`<dataset_path>/cache/SFTDataset_indices/`. On subsequent runs, they are
memory-mapped (`mmap_mode="r"`) for fast startup.

### 2.6 Query path (`__getitem__` → indices → tokens)

```
idx
 │
 ▼
shuffle_index[idx]          →  packed_sample_id
 │
 ▼
sample_index[packed_id]     →  (doc_index_beg, doc_index_end)
sample_index[packed_id + 1]
 │
 ▼
for i in range(beg, end):
    document_index[i]       →  doc_id
    dataset.get(doc_id)     →  token array for that document
 │
 ▼
concatenate all token arrays → tokens   (length S)
concatenate all loss masks   → loss_mask (length S)
per-document lengths         → lengths  (used to build cu_seqlens)
```

---

## 3. Loss masking — which tokens do we train on?

### 3.1 Conversation structure

Every pretokenized sample is a complete conversation.  There are two families:

**Chat-only** (no tool use):

```
<|im_start|>system\n ... <|im_end|>\n
<|im_start|>user\n ... <|im_end|>\n
<|im_start|>assistant\n <think>...</think> response <|im_end|>\n
  (possibly more user / assistant turns)
```

**Tool-use** (agentic trajectories):

```
<|im_start|>system\n ... (tool definitions) ... <|im_end|>\n
<|im_start|>user\n ... (task) ... <|im_end|>\n
<|im_start|>assistant\n <think>...</think> text <tool_call>\n ... </tool_call>\n <|im_end|>\n
<|im_start|>user\n <tool_response>\n ... </tool_response>\n <|im_end|>\n
  (repeat tool-call / tool-response turns)
<|im_start|>assistant\n <think>...</think> final answer <|im_end|>\n
```

Key structural observations:

- Every assistant turn starts with `<think>`. The reasoning trace may be empty
  (`<think></think>`) but the tags are always present.
- Tool responses are wrapped inside **user turns**
  (`<|im_start|>user\n <tool_response>...\n </tool_response>\n <|im_end|>\n`).
  They are never a separate role.
- Tool calls live inside **assistant turns**, enclosed by
  `<tool_call>\n ... </tool_call>\n`.
- In tool-use samples, the system prompt contains `<tool_call></tool_call>`
  tokens as part of the format description. These are **not** actual tool calls.

### 3.2 Segment roles

`extract_segments` parses each tokenized conversation and classifies every span
of content tokens into one of these roles:

| Role | Where it appears | Content |
|------|-----------------|---------|
| `system` | System turn content | System prompt / tool definitions |
| `user` | User turn content | User messages |
| `tool_response` | User turn starting with `<tool_response>` | Environment output |
| `reasoning` | Between `<think>` and `</think>` inside an assistant turn | Chain-of-thought trace |
| `assistant` | After `</think>` inside an assistant turn (excluding tool call regions) | Model response text |
| `tool_call` | Between `<tool_call>\n` and `</tool_call>\n` inside an assistant turn | Tool invocations |

### 3.3 What gets `loss_mask = 1`

The loss mask determines which token positions contribute to the training loss.
Three config flags control the behavior (`train_on_tool_calls` and
`train_on_thinking_traces` require `train_on_assistant_responses_only = True`):

| `train_on_assistant_responses_only` | `train_on_tool_calls` | `train_on_thinking_traces` | `loss_mask = 1` on |
|---|---|---|---|
| `False` | — | — | **All** tokens |
| `True` | `False` | `False` | `assistant` only |
| `True` | `True` | `False` | `assistant` + `tool_call` |
| `True` | `False` | `True` | `assistant` + `reasoning` |
| `True` | `True` | `True` | `assistant` + `tool_call` + `reasoning` |

`tool_response` segments (environment output) are **never** trained on — they
are always masked regardless of configuration.

### 3.4 Worked example — chat-only

A 2-turn conversation with reasoning traces. 1 186 tokens total.

```
  POSITION  TOKEN(S)                    ROLE         loss_mask
  ────────  ──────────────────────────  ───────────  ─────────
  [0:3]     <|im_start|>system\n        (template)   0
  [3:3]     (empty system content)      system       0
  [3:5]     <|im_end|>\n                (template)   0
  [5:8]     <|im_start|>user\n          (template)   0
  [8:40]    "Ugh, fill this: ..."       user         0
  [40:42]   <|im_end|>\n                (template)   0
  [42:46]   <|im_start|>assistant\n     (template)   0
  [46]      <think>                     (template)   0
  [47]      </think>                    (template)   0
  [47:173]  "The worst thing ..."       assistant    ← 1
  [173:175] <|im_end|>\n                (template)   0
  [175:178] <|im_start|>user\n          (template)   0
  [178:241] "Okay, my Alexa just ..."   user         0
  [241:243] <|im_end|>\n                (template)   0
  [243:247] <|im_start|>assistant\n     (template)   0
  [247]     <think>                     (template)   0
  [248:965] (717 tokens of reasoning)   reasoning    ← depends on config
  [965]     </think>                    (template)   0
  [966:1184] "The real horror ..."      assistant    ← 1
  [1184:1186] <|im_end|>\n              (template)   0
```

| Setting | Training tokens |
|---------|-----------------|
| assistant only | **343** |
| assistant + reasoning | **1 060** |

### 3.5 Worked example — tool-use

An agentic coding trajectory. 17 955 tokens total, 39 assistant turns, 38 tool
responses, ~39 tool calls.

```
  POSITION     TOKEN(S)                        ROLE           loss_mask
  ──────────   ───────────────────────────────  ─────────────  ─────────
  [0:3]        <|im_start|>system\n             (template)     0
  [3:1107]     system prompt + tool defs        system         0
               (includes <tool_call></tool_call>
                as format examples — NOT real)
  [1107:1109]  <|im_end|>\n                     (template)     0
  [1109:1112]  <|im_start|>user\n               (template)     0
  [1112:2865]  task description                 user           0
  [2865:2867]  <|im_end|>\n                     (template)     0

  ┌─── assistant turn (with tool call) ────────────────────────────────
  │ [2867:2871] <|im_start|>assistant\n         (template)     0
  │ [2871]      <think>                         (template)     0
  │ [2872]      </think>                        (template)     0
  │ [2873:3631] "I'll help you implement ..."   assistant      ← 1
  │ [3631:3633] <tool_call>\n                   (template)     0
  │ [3633:3665] <function=str_replace_editor>   tool_call      ← depends on config
  │             <parameter=command>view ...
  │ [3665:3667] </tool_call>\n                  (template)     0
  │ [3667:3669] <|im_end|>\n                    (template)     0
  └────────────────────────────────────────────────────────────────────

  ┌─── tool response (inside a user turn) ─────────────────────────────
  │ [3669:3672] <|im_start|>user\n              (template)     0
  │ [3672:3674] <tool_response>\n               (template)     0
  │ [3674:4608] OBSERVATION: ...                tool_response  0
  │ [4608:4610] </tool_response>\n              (template)     0
  │ [4610:4612] <|im_end|>\n                    (template)     0
  └────────────────────────────────────────────────────────────────────

  ... (pattern repeats for each tool-call / tool-response pair) ...

  ┌─── final assistant turn (text, no tool call) ──────────────────────
  │ [17855:17859] <|im_start|>assistant\n       (template)     0
  │ [17859]       <think>                       (template)     0
  │ [17860]       </think>                      (template)     0
  │ [17861:17874] "Let me remove the ..."       assistant      ← 1
  │ [17874:17876] <tool_call>\n                 (template)     0
  │ [17876:17910] <function=bash> rm ...        tool_call      ← depends on config
  │ [17910:17912] </tool_call>\n                (template)     0
  │ [17912:17914] <|im_end|>\n                  (template)     0
  └────────────────────────────────────────────────────────────────────
```

| Setting | Training tokens |
|---------|-----------------|
| assistant only | **2 790** |
| assistant + tool calls | **5 732** |

### 3.6 Template tokens and the loss mask

The following structural tokens always have `loss_mask = 0`:

| Token(s) | Id(s) | Purpose |
|----------|-------|---------|
| `<\|im_start\|>` | `10` | Turn boundary |
| `system\n` / `user\n` / `assistant\n` | varies | Role identifier |
| `<\|im_end\|>\n` | `11, 1010` | Turn terminator |
| `<think>` | `12` | Reasoning open |
| `</think>` | `13` | Reasoning close |
| `<tool_call>\n` | `14, 1010` | Tool call open |
| `</tool_call>\n` | `15, 1010` | Tool call close |
| `<tool_response>\n` | `16, 1010` | Tool response open |
| `</tool_response>\n` | `17, 1010` | Tool response close |

These tokens delineate message boundaries and roles. They are part of the chat
template, not model-generated content, so they are always masked out.
