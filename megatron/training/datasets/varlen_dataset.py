# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Variable-length packed (THD) dataset for SFT-style instruction data.

This dataset is the entry point for the ``--use-varlen-dataset`` flag. It is
independent of the ``--sft`` flag (no implicit coupling) but shares the same
THD packing / cu_seqlens / dynamic-CP padding logic by extending the existing
:class:`SFTDataset` family. The variable-length aspect is what matters here:
samples have wildly different lengths and are packed into THD format for
training throughput.

Compared to :class:`SFTDataset`, this dataset adds:

  * **Multi-source loading** — accepts HuggingFace Hub repo ids
    (``owner/repo``), local ``.parquet`` files, and local ``.jsonl/.json``
    files; the latter are read via pandas to sidestep pyarrow's per-chunk
    JSON schema inference which fails when sample fields vary across rows.

  * **Auto schema detection** — four input layouts are auto-detected by column
    name. The three instruction-tuning layouts are normalized to the messages
    list format expected by the parent ``SFTDataset.__getitem__``; the
    ``pretrain-text`` fallback instead returns a raw string handled separately
    in :meth:`VarlenDataset.__getitem__`:

      * **openai-messages** — column ``messages`` (Llama post-training,
        HuggingFaceH4/no_robots, ...)
      * **sharegpt** — column ``conversations`` (OpenOrca, Vicuna, ...)
      * **alpaca / dolly** — at least one of
        ``instruction|prompt|query|question`` + one of
        ``output|response|completion|answer``, plus optional context field
        ``input|context``.
      * **pretrain-text** — column ``text``; returns the raw string (no
        messages list, no role masking), tokenized as plain pretraining text.

  * **Mock variant** — :class:`MockVarlenDataset` mirrors
    :class:`MockSFTDataset` end-to-end (synthetic lognormal sequence-length
    distribution / fixed-length file / verification mode from an
    ``IndexedDataset``), configured via
    ``--varlen-mock-dataset-config-json``.

Limitations (raise a clear ``ValueError`` instead of silently mishandling):

  * Sample content/value must be a plain string — multi-modal content lists
    (image+text parts) are not supported.
  * Tree-structured (OpenAssistant oasst1) and preference (chosen/rejected)
    datasets are out of scope.
  * For HF Hub repos, only ``split="train"`` is loaded.
"""

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Field-name synonyms (probed in order; first non-empty wins).
_INSTRUCTION_FIELDS: Tuple[str, ...] = (
    "instruction", "prompt", "query", "question",
)
_OUTPUT_FIELDS: Tuple[str, ...] = (
    "output", "response", "completion", "answer",
)
# Supplementary user-turn context: Stanford Alpaca's "input", Dolly's "context".
_EXTRA_INPUT_FIELDS: Tuple[str, ...] = ("input", "context")

# ShareGPT "from" value -> chat-template "role". Unknown values fall back to
# "user" so downstream tokenization does not crash on unfamiliar speakers.
_SHAREGPT_ROLE_MAP: Dict[str, str] = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


def _looks_like_hf_id(path: str) -> bool:
    """Heuristic: does ``path`` look like an ``owner/repo`` HF dataset id?

    True iff ``path`` contains ``/``, is not an absolute/relative file path,
    and does not exist on the local filesystem.
    """
    if not path:
        return False
    if os.path.exists(path):
        return False
    if path.startswith(("/", "./", "../")):
        return False
    return "/" in path


def _first_present(
    sample: Dict[str, Any], fields: Iterable[str]
) -> Optional[str]:
    """Return the first non-empty string value among the given fields, or None."""
    for f in fields:
        v = sample.get(f)
        if v in (None, ""):
            continue
        if not isinstance(v, str):
            raise ValueError(
                f"VarlenDataset: field '{f}' must be a string, "
                f"got {type(v).__name__}."
            )
        return v
    return None


def _ensure_str_content(content: Any, where: str) -> str:
    """Validate that a turn's content is a plain string (reject multi-modal lists)."""
    if content is None:
        return ""
    if not isinstance(content, str):
        raise ValueError(
            f"VarlenDataset: {where} content must be a string, "
            f"got {type(content).__name__}. Multi-modal datasets (e.g. "
            "content as a list of image/text parts) are not supported."
        )
    return content


def _alpaca_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert an Alpaca/Dolly-style sample to a 3-turn messages list."""
    instruction = _first_present(sample, _INSTRUCTION_FIELDS) or ""
    extra_input = _first_present(sample, _EXTRA_INPUT_FIELDS) or ""
    output = _first_present(sample, _OUTPUT_FIELDS) or ""
    user_content = (
        f"{instruction}\n\n{extra_input}" if extra_input else instruction
    )
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]


def _sharegpt_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a ShareGPT ``conversations`` sample to a messages list.

    Prepends an empty ``system`` turn unless the conversation already starts
    with one, so ``SFTDataset._split_conversations`` treats the sample as a
    single conversation.
    """
    conv = sample.get("conversations") or []
    out: List[Dict[str, str]] = []
    first_speaker = (conv[0].get("from") or "").lower() if conv else ""
    if first_speaker != "system":
        out.append({"role": "system", "content": ""})
    for turn in conv:
        speaker = (turn.get("from") or "").lower()
        role = _SHAREGPT_ROLE_MAP.get(speaker, "user")
        content = _ensure_str_content(turn.get("value"), f"sharegpt turn role={role}")
        out.append({"role": role, "content": content})
    return out


def _messages_passthrough(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Pass through an OpenAI ``messages`` sample, ensuring a leading system turn.

    Strips any keys other than ``role``/``content`` (e.g. ``name``,
    ``tool_calls``) since they are not part of the chat-template input
    expected by SFTTokenizer.
    """
    raw = list(sample.get("messages") or [])
    if raw and raw[0].get("role") != "system":
        raw = [{"role": "system", "content": ""}] + raw
    out: List[Dict[str, str]] = []
    for m in raw:
        role = m.get("role") or "user"
        content = _ensure_str_content(m.get("content"), f"messages turn role={role}")
        out.append({"role": role, "content": content})
    return out


def _raw_text_loader(sample: Dict[str, Any]) -> str:
    """Return the ``text`` column unchanged for pretrain-style packed runs.

    Unlike the SFT schemas this returns a plain string (no messages list).
    :class:`VarlenDataset.__getitem__` dispatches on the return type to pick
    a tokenization path that skips chat templating and prompt masking.
    """
    text = sample.get("text") or ""
    if not isinstance(text, str):
        raise ValueError(
            f"VarlenDataset (pretrain-text schema): 'text' must be a string, "
            f"got {type(text).__name__}."
        )
    return text


def _select_converter(
    column_names: List[str],
) -> Tuple[Callable[[Dict[str, Any]], Any], str]:
    """Pick a sample converter based on dataset column names.

    Priority (most explicit first): openai-messages > sharegpt > alpaca/dolly
    > pretrain-text. ``pretrain-text`` is the fallback for datasets that
    only carry a single ``text`` column (e.g. Dolma / OLMo midtraining
    corpora) — long-context pretraining packed through the same THD path
    as SFT.
    """
    cols = set(column_names)
    if "messages" in cols:
        return _messages_passthrough, "openai-messages"
    if "conversations" in cols:
        return _sharegpt_to_messages, "sharegpt"
    has_instr = any(f in cols for f in _INSTRUCTION_FIELDS)
    has_out = any(f in cols for f in _OUTPUT_FIELDS)
    if has_instr and has_out:
        return _alpaca_to_messages, "alpaca"
    if "text" in cols:
        return _raw_text_loader, "pretrain-text"
    raise ValueError(
        "VarlenDataset cannot infer schema from columns "
        f"{sorted(cols)}. Supported schemas: "
        f"alpaca/dolly ({'|'.join(_INSTRUCTION_FIELDS)} + "
        f"{'|'.join(_OUTPUT_FIELDS)} [+ optional {'|'.join(_EXTRA_INPUT_FIELDS)}]), "
        "sharegpt (conversations), openai-messages (messages), "
        "pretrain-text (text)."
    )
