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
    files; the latter use stdlib JSON and Dataset.from_list to avoid per-chunk
    schema inference failures when fields vary across rows.
    Hub configs and splits are joined logically without requiring identical Arrow schemas.

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
      * **pretrain-text** — column ``text``, ``raw_content`` (RedPajama V2 /
        RP2), or ``content`` (CodeForge); returns the raw string (no messages
        list, no role masking), tokenized as plain pretraining text.

  * **Mock variant** — :class:`MockVarlenDataset` mirrors
    :class:`MockSFTDataset` end-to-end (synthetic lognormal sequence-length
    distribution / fixed-length file / verification mode from an
    ``IndexedDataset``), configured via
    ``--varlen-mock-dataset-config-json``.

Limitations (raise a clear ``ValueError`` instead of silently mishandling):

  * OpenAI/OpenCode text and tool-result content blocks are supported, but
    image/audio/video content requires a multimodal dataset path.
  * Tree-structured (OpenAssistant oasst1) and preference (chosen/rejected)
    datasets are out of scope.

For HF Hub repos, all configs are selected. Each config uses its ``train``
split when present, otherwise all of its thematic splits. If the Hub Arrow
builder cannot unify heterogeneous JSON rows, the rows are streamed into a
map-style single-payload-column cache so Megatron can still index them.
"""

import json
import logging
import os
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset
from megatron.core.datasets.utils import Split
from megatron.training.datasets.sft_dataset import (
    IGNORE_INDEX,
    MockSFTDataset,
    MockSFTLowLevelDataset,
    SFTDataset,
    SFTLowLevelDataset,
)
from megatron.training.datasets.utils import load_json_arg

logger = logging.getLogger(__name__)

_HF_PAYLOAD_COLUMN = "__varlen_json_payload__"

# Field-name synonyms (probed in order; first non-empty wins).
_INSTRUCTION_FIELDS: Tuple[str, ...] = ("instruction", "prompt", "query", "question")
_OUTPUT_FIELDS: Tuple[str, ...] = ("output", "response", "completion", "answer")
# Supplementary user-turn context: Stanford Alpaca's "input", Dolly's "context".
_EXTRA_INPUT_FIELDS: Tuple[str, ...] = ("input", "context")
# Raw pretraining text: canonical datasets use "text", RedPajama V2 uses
# "raw_content", and CodeForge uses "content".
_PRETRAIN_TEXT_FIELDS: Tuple[str, ...] = ("text", "raw_content", "content")

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


@dataclass
class ChatTemplateSample:
    """Conversation plus keyword arguments consumed by the chat template."""

    messages: List[Dict[str, Any]]
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


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


# Words that mark held-out Hub splits, e.g. ``test_sft`` or ``validation``.
_EVAL_SPLIT_WORDS = frozenset({"test", "validation", "valid", "val", "dev", "eval", "evaluation"})


def _is_eval_split(split_name: str) -> bool:
    """Whether a Hub split name marks evaluation data rather than a training topic."""
    return any(word in _EVAL_SPLIT_WORDS for word in re.split(r"[^0-9a-z]+", split_name.lower()))


def _first_present(sample: Dict[str, Any], fields: Iterable[str]) -> Optional[str]:
    """Return the first non-empty string value among the given fields, or None."""
    for field_name in fields:
        value = sample.get(field_name)
        if value in (None, ""):
            continue
        if not isinstance(value, str):
            raise ValueError(
                f"VarlenDataset: field '{field_name}' must be a string, got {type(value).__name__}."
            )
        return value
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


def _json_text(value: Any, where: str) -> str:
    """Convert a structured text payload to its stable JSON representation."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VarlenDataset: {where} is not JSON serializable.") from exc


def _normalize_openai_content(content: Any, where: str) -> Tuple[str, Dict[str, str]]:
    """Normalize text-only OpenAI/OpenCode content blocks to a string.

    OpenCode stores tool outputs as ``tool-result`` blocks rather than plain
    strings. Text blocks are losslessly flattened for text chat templates;
    image/audio/video blocks remain unsupported by this text-only dataset.
    """
    if content is None:
        return "", {}
    if isinstance(content, str):
        return content, {}
    if not isinstance(content, list):
        raise ValueError(
            f"VarlenDataset: {where} content must be a string or a list of text blocks, "
            f"got {type(content).__name__}."
        )

    pieces: List[str] = []
    metadata: Dict[str, str] = {}
    for index, block in enumerate(content):
        block_where = f"{where} content block {index}"
        if isinstance(block, str):
            pieces.append(block)
            continue
        if not isinstance(block, dict):
            raise ValueError(
                f"VarlenDataset: {block_where} must be a string or object, "
                f"got {type(block).__name__}."
            )

        block_type = block.get("type")
        if block_type in ("text", "input_text", "output_text"):
            pieces.append(_json_text(block.get("text", block.get("value")), block_where))
        elif block_type == "tool-result":
            output = block.get("output")
            if isinstance(output, dict):
                output = output.get("value", output.get("content", output))
            pieces.append(_json_text(output, f"{block_where} output"))
            if block.get("toolCallId"):
                metadata.setdefault("tool_call_id", str(block["toolCallId"]))
            if block.get("toolName"):
                metadata.setdefault("name", str(block["toolName"]))
        else:
            raise ValueError(
                f"VarlenDataset: unsupported {block_where} type {block_type!r}. "
                "Multimodal image/audio/video content requires a multimodal dataset path."
            )

    return "\n".join(pieces), metadata


def _normalize_tools(tools: Any) -> List[Dict[str, Any]]:
    """Convert OpenCode tool definitions; preserve OpenAI definitions."""
    normalized_tools: List[Dict[str, Any]] = []
    for index, tool in enumerate(_ensure_list_field(tools, "tools")):
        if not isinstance(tool, dict):
            raise ValueError(
                f"VarlenDataset: tools[{index}] must be an object, got {type(tool).__name__}."
            )
        if "id" not in tool:
            normalized_tools.append(tool)
            continue
        input_schema = tool.get("inputSchema") or {}
        if not isinstance(input_schema, dict):
            raise ValueError(f"VarlenDataset: tools[{index}].inputSchema must be an object.")
        normalized_tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(tool["id"]),
                    "description": str(tool.get("description") or ""),
                    "parameters": input_schema.get("jsonSchema", input_schema),
                },
            }
        )
    return normalized_tools


def _ensure_list_field(value: Any, field_name: str) -> List[Any]:
    """Return a list field, decoding JSON-encoded parquet columns when needed."""
    if value in (None, ""):
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"VarlenDataset: field '{field_name}' contains invalid JSON.") from exc
    if not isinstance(value, list):
        raise ValueError(
            f"VarlenDataset: field '{field_name}' must be a list or a JSON-encoded list, "
            f"got {type(value).__name__}."
        )
    return value


def _alpaca_to_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert an Alpaca/Dolly-style sample to a 3-turn messages list."""
    instruction = _first_present(sample, _INSTRUCTION_FIELDS) or ""
    extra_input = _first_present(sample, _EXTRA_INPUT_FIELDS) or ""
    output = _first_present(sample, _OUTPUT_FIELDS) or ""
    user_content = f"{instruction}\n\n{extra_input}" if extra_input else instruction
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
    conv = _ensure_list_field(sample.get("conversations"), "conversations")
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


def _messages_passthrough(sample: Dict[str, Any]) -> ChatTemplateSample:
    """Normalize an OpenAI ``messages`` sample without dropping template metadata.

    Fields such as ``reasoning_content``, ``tool_calls``, ``name``, and
    ``tool_call_id`` are interpreted by model-specific Hugging Face chat
    templates. Top-level ``tools`` are forwarded as a template keyword
    argument. This is required for tool-integrated datasets such as
    ``nvidia/Nemotron-SFT-Math-v3``.
    """
    raw = _ensure_list_field(sample.get("messages"), "messages")
    out: List[Dict[str, Any]] = []
    # Legacy OpenAI ``function`` replies and Llama ``ipython`` turns are tool results.
    role_map = {"developer": "system", "function": "tool", "ipython": "tool"}
    supported_roles = {"system", "user", "assistant", "tool"}
    for index, m in enumerate(raw):
        if not isinstance(m, dict):
            raise ValueError(
                f"VarlenDataset: messages[{index}] must be an object, got {type(m).__name__}."
            )
        role = str(m.get("role") or "user").lower()
        role = role_map.get(role, role)
        if role not in supported_roles:
            raise ValueError(f"VarlenDataset: unsupported messages[{index}] role {role!r}.")
        content, content_metadata = _normalize_openai_content(
            m.get("content"), f"messages turn {index} role={role}"
        )
        # Arrow gives every message the union of the fields seen in the column and
        # fills the missing ones with None. Keep only the fields this message sets, so
        # templates that test ``'tool_calls' in message`` see the original row.
        normalized = {key: value for key, value in m.items() if value is not None}
        normalized["role"] = role
        normalized["content"] = content
        for key, value in content_metadata.items():
            normalized.setdefault(key, value)

        tool_calls = normalized.get("tool_calls")
        if tool_calls not in (None, ""):
            normalized["tool_calls"] = _ensure_list_field(
                tool_calls, f"messages[{index}].tool_calls"
            )

        function_call = normalized.get("function_call")
        if function_call and not normalized.get("tool_calls"):
            if isinstance(function_call, str):
                try:
                    function_call = json.loads(function_call)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"VarlenDataset: messages[{index}].function_call contains invalid JSON."
                    ) from exc
            if not isinstance(function_call, dict):
                raise ValueError(
                    f"VarlenDataset: messages[{index}].function_call must be an object."
                )
            normalized["tool_calls"] = [{"type": "function", "function": dict(function_call)}]
        out.append(normalized)

    if out and out[0]["role"] != "system":
        out.insert(0, {"role": "system", "content": ""})

    chat_template_kwargs = sample.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, str):
        try:
            chat_template_kwargs = json.loads(chat_template_kwargs)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "VarlenDataset: field 'chat_template_kwargs' contains invalid JSON."
            ) from exc
    if chat_template_kwargs is None:
        chat_template_kwargs = {}
    if not isinstance(chat_template_kwargs, dict):
        raise ValueError(
            "VarlenDataset: field 'chat_template_kwargs' must be an object or a "
            "JSON-encoded object."
        )
    # A struct column gets the same None padding as messages.
    chat_template_kwargs = {
        key: value for key, value in chat_template_kwargs.items() if value is not None
    }

    tools = _normalize_tools(sample.get("tools"))
    if tools:
        chat_template_kwargs["tools"] = tools
    return ChatTemplateSample(out, chat_template_kwargs)


def _raw_text_loader(sample: Dict[str, Any]) -> str:
    """Return a recognized raw-text column unchanged for pretrain-style packed runs.

    Unlike the SFT schemas this returns a plain string (no messages list).
    :class:`VarlenDataset.__getitem__` dispatches on the return type to pick
    a tokenization path that skips chat templating and prompt masking.
    """
    return _first_present(sample, _PRETRAIN_TEXT_FIELDS) or ""


def _select_converter(column_names: List[str]) -> Tuple[Callable[[Dict[str, Any]], Any], str]:
    """Pick a sample converter based on dataset column names.

    Priority (most explicit first): openai-messages > sharegpt > alpaca/dolly
    > pretrain-text. ``pretrain-text`` is the fallback for datasets whose
    document body is stored in ``text`` (e.g. Dolma / OLMo), ``raw_content``
    (RedPajama V2 / RP2), or ``content`` (CodeForge) — long-context
    pretraining packed through the same THD path as SFT.
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
    if any(field in cols for field in _PRETRAIN_TEXT_FIELDS):
        return _raw_text_loader, "pretrain-text"
    raise ValueError(
        "VarlenDataset cannot infer schema from columns "
        f"{sorted(cols)}. Supported schemas: "
        f"alpaca/dolly ({'|'.join(_INSTRUCTION_FIELDS)} + "
        f"{'|'.join(_OUTPUT_FIELDS)} [+ optional {'|'.join(_EXTRA_INPUT_FIELDS)}]), "
        "sharegpt (conversations), openai-messages (messages), "
        f"pretrain-text ({'|'.join(_PRETRAIN_TEXT_FIELDS)})."
    )


def _iter_hf_data_file_rows(data_file: str) -> Iterator[Dict[str, Any]]:
    """Read raw JSONL or Parquet rows without a declared HF schema."""
    import fsspec

    normalized_path = data_file.lower().split("?", 1)[0]
    if normalized_path.endswith(".parquet"):
        import pyarrow.parquet as pq

        with fsspec.open(data_file, "rb") as stream:
            parquet_file = pq.ParquetFile(stream)
            for batch in parquet_file.iter_batches(batch_size=1024):
                yield from batch.to_pylist()
        return

    with fsspec.open(data_file, "rt", encoding="utf-8", compression="infer") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"VarlenDataset: expected JSON objects in {data_file!r}.")
                yield row


def _iter_hf_rows_as_json(data_files: List[str]):
    """Stream raw heterogeneous Hub rows as one JSON payload column."""
    for data_file in data_files:
        for row in _iter_hf_data_file_rows(data_file):
            yield {_HF_PAYLOAD_COLUMN: json.dumps(row, ensure_ascii=False)}


def _json_payload_to_item(sample: Dict[str, Any]) -> Any:
    """Decode and normalize a row produced by :func:`_iter_hf_rows_as_json`."""
    row = json.loads(sample[_HF_PAYLOAD_COLUMN])
    converter, _ = _select_converter(list(row))
    return converter(row)


class VarlenLowLevelDataset(SFTLowLevelDataset):
    """Low-level loader: HF Hub repo / local parquet / local jsonl, normalized.

    Dataset path interpretation:

      * HF Hub repo id (e.g. ``nvidia/Nemotron-SFT-Math-v4``) — contains ``/``
        and does not exist locally. The repo's default config is used when it
        declares one (its only config, or the one named or marked default);
        otherwise all configs are combined. Each config uses ``train`` when
        available, otherwise all thematic splits; evaluation splits (``test``,
        ``validation``, ``dev``, ...) are skipped.
      * Local ``.parquet`` — loaded via
        ``datasets.load_dataset("parquet", data_files=path, split="all")``;
        parquet's footer schema makes chunked loading safe.
      * Otherwise local jsonl/json — parsed with stdlib JSON and built by
        Dataset.from_list in one pass without a pandas dependency.

    Each config/split remains a separate map-style component and is joined by
    index arithmetic. This avoids requiring Arrow feature compatibility across
    components. With ``shuffle_seed`` (the dataset config's random seed), several
    components are read through one seeded permutation, so contiguous ``--split``
    ranges and sequential samplers mix all of them; a single component keeps its
    row order. A per-component converter is selected from its columns. Hub
    JSON that fails Arrow schema unification is streamed into a cached JSON
    payload column and normalized per row.
    """

    def __init__(self, dataset_path: str, shuffle_seed: Optional[int] = None) -> None:
        try:
            from datasets import (
                Dataset,
                Features,
                Value,
                get_dataset_config_names,
                get_dataset_default_config_name,
                get_dataset_split_names,
                load_dataset,
                load_dataset_builder,
            )
            from datasets.exceptions import DatasetGenerationError
        except ImportError as exc:
            raise ImportError(
                "VarlenDataset requires the `datasets` library " "(pip install datasets)."
            ) from exc
        try:
            from datasets.table import CastError
        except ImportError:
            CastError = DatasetGenerationError

        self._datasets: List[Any] = []
        self._converters: List[Callable[[Dict[str, Any]], Any]] = []
        self._schema_names: List[str] = []
        self._cumulative_sizes: List[int] = []

        def add_component(dataset: Any, payload: bool = False) -> None:
            if payload:
                converter = _json_payload_to_item
                schema_name = "json-payload"
            else:
                converter, schema_name = _select_converter(list(dataset.column_names))
            self._datasets.append(dataset)
            self._converters.append(converter)
            self._schema_names.append(schema_name)
            previous_size = self._cumulative_sizes[-1] if self._cumulative_sizes else 0
            self._cumulative_sizes.append(previous_size + len(dataset))

        if _looks_like_hf_id(dataset_path):
            # A declared default config (the only config, or the one named or marked
            # default) holds the repo's data; the other configs of such repos are
            # usually subsets of it (e.g. fineweb-edu samples and dumps). Combine
            # all configs only when no default is declared.
            default_config = get_dataset_default_config_name(dataset_path)
            if default_config is not None:
                selected_configs = [default_config]
            else:
                selected_configs = get_dataset_config_names(dataset_path)
            selected_sources: List[Tuple[str, str]] = []
            for selected_config in selected_configs:
                available_splits = get_dataset_split_names(dataset_path, selected_config)
                if "train" in available_splits:
                    selected_splits = ["train"]
                else:
                    # Thematic splits (e.g. ``python``, ``cpp``) are training data;
                    # evaluation splits are never trained on.
                    selected_splits = [s for s in available_splits if not _is_eval_split(s)]
                    skipped_splits = [s for s in available_splits if _is_eval_split(s)]
                    if skipped_splits:
                        logger.warning(
                            "VarlenDataset: skipping evaluation splits %s of %s:%s.",
                            skipped_splits,
                            dataset_path,
                            selected_config,
                        )
                selected_sources.extend(
                    (selected_config, selected_split) for selected_split in selected_splits
                )

            if not selected_sources:
                raise ValueError(
                    f"VarlenDataset: no config/split was found in {dataset_path} "
                    "(evaluation splits are not used for training)."
                )
            logger.info(
                "VarlenDataset: loading %s from %s.",
                ", ".join(f"{config}/{split}" for config, split in selected_sources),
                dataset_path,
            )

            for selected_config, selected_split in selected_sources:
                source = f"{dataset_path}:{selected_config}/{selected_split}"
                try:
                    dataset = load_dataset(dataset_path, name=selected_config, split=selected_split)
                    add_component(dataset)
                except (DatasetGenerationError, CastError):
                    builder = load_dataset_builder(dataset_path, name=selected_config)
                    data_files = builder.config.data_files.get(selected_split)
                    if not data_files:
                        raise ValueError(
                            f"VarlenDataset: could not resolve raw files for {source}."
                        )
                    logger.warning(
                        "Arrow schema generation failed for %s; materializing a map-style "
                        "JSON payload cache directly from the raw data files.",
                        source,
                    )
                    dataset = Dataset.from_generator(
                        _iter_hf_rows_as_json,
                        features=Features({_HF_PAYLOAD_COLUMN: Value("string")}),
                        gen_kwargs={"data_files": [str(data_file) for data_file in data_files]},
                    )
                    add_component(dataset, payload=True)
        elif dataset_path.endswith(".parquet"):
            add_component(load_dataset("parquet", data_files=dataset_path, split="all"))
        else:
            # Parse the jsonl with the stdlib and build the table in one
            # pyarrow pass: datasets.load_dataset("json", ...) infers schema
            # per parallel chunk and fails with CastError when the union of
            # fields varies between rows (e.g. LongAlpaca-12k); building from
            # the full record list sidesteps that without a pandas dependency.
            with open(dataset_path) as f:
                records = [json.loads(line) for line in f if line.strip()]
            add_component(Dataset.from_list(records))

        if not self._datasets:
            raise ValueError(f"VarlenDataset: no data was loaded from {dataset_path!r}.")
        self.dataset = self._datasets[0] if len(self._datasets) == 1 else tuple(self._datasets)
        unique_schema_names = list(dict.fromkeys(self._schema_names))
        self._schema_name = (
            unique_schema_names[0]
            if len(unique_schema_names) == 1
            else f"mixed({','.join(unique_schema_names)})"
        )
        # Components are joined end to end, while ``--split`` takes contiguous index
        # ranges and the default sampler reads them in order. A seeded permutation (the
        # same on every rank) makes each range a mix of all components.
        self._order = None
        if shuffle_seed is not None and len(self._datasets) > 1:
            self._order = np.random.default_rng(shuffle_seed).permutation(len(self))

    @property
    def schema_name(self) -> str:
        """Detected schema name: ``alpaca`` / ``sharegpt`` / ``openai-messages`` /
        ``pretrain-text`` / ``json-payload`` (the raw-schema fallback)."""
        return self._schema_name

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Any:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if self._order is not None:
            idx = int(self._order[idx])
        component_index = bisect_right(self._cumulative_sizes, idx)
        component_start = self._cumulative_sizes[component_index - 1] if component_index > 0 else 0
        return self._converters[component_index](
            self._datasets[component_index][idx - component_start]
        )


def _right_padded_ids(
    ids: np.ndarray, keep: int, eod_at: Optional[int], length: int, total: int, eod: int, pad: int
) -> torch.Tensor:
    """Return ``ids[:keep]`` as an int64 tensor of ``total`` tokens.

    EOD is written at ``eod_at`` (overwriting or appending; None writes none), and
    positions from ``length`` on hold ``pad``.
    """
    out = torch.empty(total, dtype=torch.int64)
    buf = out.numpy()
    buf[:keep] = ids[:keep]
    if eod_at is not None:
        buf[eod_at] = eod
    buf[length:] = pad
    return out


class VarlenDataset(SFTDataset):
    """Variable-length single-sample SFT dataset for the packed-sequence path.

    Each ``__getitem__`` returns **one tokenized conversation** in unpacked
    form: ``tokens``/``labels``/``loss_mask``/``position_ids`` whose length
    equals the sample's actual token count (padded to ``pad_granularity``,
    NOT to ``sequence_length``), plus ``original_seq_len``/``padded_seq_len``
    tensors that the upstream packing scheduler consumes directly via
    :func:`get_batch_and_global_seqlens`.

    This is the schema described in :class:`BasePackingScheduler.get_required_sample_keys`.
    It deliberately skips the multi-conversation pre-packing that
    :class:`SFTDataset.__getitem__` does, letting the upstream scheduler
    pack variable-length samples across the DP×CP grid with no per-sample
    padding waste.

    Truncation: samples longer than ``config.sequence_length`` are truncated
    on the right; an EOD token is appended if the truncation removed it.
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return VarlenLowLevelDataset(
            dataset_path, shuffle_seed=getattr(config, "random_seed", None)
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        max_len = self.config.sequence_length
        # HuggingFaceTokenizer returns None for ``pad`` when the underlying
        # tokenizer has no explicit pad token (common for raw pretraining
        # tokenizers like Qwen3). Fall back to eod for padding — irrelevant
        # for loss because loss_mask zeros pad positions out.
        eod = tokenizer.eod
        pad = tokenizer.pad if tokenizer.pad is not None else eod
        assert (
            eod is not None
        ), "VarlenDataset requires the tokenizer to expose an EOD/EOS token id."

        # 1. Pull a single item from the low-level dataset. OpenAI-style data
        #    carries messages plus optional chat-template kwargs (for example,
        #    top-level tool definitions); other SFT schemas return a messages
        #    list, and pretrain-text returns a raw string.
        item = self.dataset[int(self.indices[idx % len(self.indices)])]

        assert not self.config.reset_position_ids
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask

        # 2. Tokenize. SFT schemas go through tokenize_conversation (chat
        #    template + role-aware target masking); pretrain-text bypasses
        #    chat templating and uses the plain ``tokenize`` interface,
        #    treating every token as a target (no prompt masking). Ids stay in
        #    int64 arrays from here on: building the sample as Python lists
        #    costs ~4 ms per 50K tokens, and packing schedulers fetch the whole
        #    global batch at the start of every step.
        if isinstance(item, str):
            tokens = targets = np.array(list(tokenizer.tokenize(item)), dtype=np.int64)
        else:
            if isinstance(item, ChatTemplateSample):
                tokens, targets = tokenizer.tokenize_conversation(
                    item.messages,
                    return_target=True,
                    add_generation_prompt=False,
                    chat_template_kwargs=item.chat_template_kwargs,
                )
            else:
                tokens, targets = tokenizer.tokenize_conversation(
                    item, return_target=True, add_generation_prompt=False
                )
            tokens = np.asarray(tokens, dtype=np.int64)
            targets = np.asarray(targets, dtype=np.int64)

        # 2b. Guard against an empty tokenization (e.g. a blank ``pretrain-text``
        #     row where ``tokenizer.tokenize("")`` returns no ids). Represent it
        #     as a single end-of-document token so the next-token shift still
        #     yields a valid 1-token sample instead of raising on
        #     ``tokens[keep - 1]`` below or producing a zero-length sequence.
        if len(tokens) == 0:
            tokens = targets = np.array([eod, eod], dtype=np.int64)
        assert len(targets) == len(
            tokens
        ), "VarlenDataset requires tokenize_conversation to return one target per token."

        # 3. Right-truncate to ``sequence_length + 1`` (we drop the last token
        #    after the input/label shift below). Keep an EOD at the end so a
        #    truncated assistant turn still has a valid stop token.
        # 4. Ensure EOD is the last token (unconditional for short samples).
        keep = min(len(tokens), max_len + 1)
        if tokens[keep - 1] == eod:
            eod_at = None
        elif keep < len(tokens):
            eod_at = keep - 1  # truncated: overwrite the last kept token
        else:
            eod_at = keep  # append
        length = keep + 1 if eod_at == keep else keep
        valid_len = length - 1

        if self.config.varlen_sbhd_validation:
            # 5a. SBHD validation mode: right-pad to sequence_length + 1, drop
            #     packing metadata, return shape [sequence_length]. Useful as a
            #     numerical reference for THD path verification (no scheduler,
            #     no dynamic-cp).
            total = max(length, max_len + 1)
            assert total == max_len + 1
        else:
            original_seq_len = length - 1  # length after the shift below
            # 5b. THD path: pad to pad_granularity (dp_size * cp_size * 2 * sp),
            #     the minimum alignment required by CP slicing. We deliberately
            #     do NOT pad to sequence_length — the upstream packing scheduler
            #     will combine variable-length samples up to
            #     max_seqlen_per_dp_cp_rank.
            pad_granularity = self._calculate_padding_divisor()
            padded_seq_len = original_seq_len + (-original_seq_len % pad_granularity)
            total = padded_seq_len + 1

        # 6. Apply the next-token shift.
        sequence = _right_padded_ids(tokens, keep, eod_at, length, total, eod, pad)
        if targets is not tokens:
            target_sequence = _right_padded_ids(targets, keep, eod_at, length, total, eod, pad)
        else:
            target_sequence = sequence
        input_ids = sequence[:-1].clone()
        labels = target_sequence[1:].clone()
        loss_mask = torch.ones(total - 1, dtype=torch.float32)
        loss_mask[valid_len:] = 0.0  # mask the right-padded tail by position
        loss_mask[labels == IGNORE_INDEX] = 0.0

        if self.config.varlen_sbhd_validation:
            # Keep physical padding separate from the LM loss mask: prompt
            # tokens may be loss-masked but must still participate in MoE.
            padding_mask = torch.zeros(max_len, dtype=torch.bool)
            padding_mask[valid_len:] = True
            return {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'padding_mask': padding_mask,
                'position_ids': torch.arange(max_len, dtype=torch.int64),
            }

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': torch.arange(padded_seq_len, dtype=torch.int64),
            # The packing scheduler consumes these directly; cu_seqlens /
            # max_seqlen are produced downstream in _pack_sequences.
            'original_seq_len': torch.tensor([original_seq_len], dtype=torch.int32),
            'padded_seq_len': torch.tensor([padded_seq_len], dtype=torch.int32),
        }


class MockVarlenDataset(MockSFTDataset):
    """Mock variable-length dataset for benchmarking the varlen path.

    Uses :class:`MockSFTLowLevelDataset` for sequence-length sampling (lognormal
    distribution / per-line CSV / IndexedDataset verification mode — same JSON
    schema as ``--sft-mock-dataset-config-json``, just consumed via
    ``--varlen-mock-dataset-config-json``).

    Output shape mirrors :class:`VarlenDataset.__getitem__` (not the inherited
    :meth:`MockSFTDataset.__getitem__`) so the mock and real-data paths
    exercise exactly the same downstream pipeline:

      * THD mode: emits **one unpacked sample** padded to ``pad_granularity``
        with ``original_seq_len`` / ``padded_seq_len`` tensors. The upstream
        scheduler packs across the DP×CP grid.

    ``--varlen-sbhd-validation`` is intentionally not implemented for mock
    data; it is guarded against in argument validation.
    """

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        if config.varlen_mock_dataset_config_json is None:
            mock_config = {
                "mode": "distribution",
                "type": "lognormal",
                "min_seq_len": config.sequence_length // 2,
                "max_seq_len": config.sequence_length,
                "mean_seq_len": config.sequence_length // 4 * 3,
                "lognormal_sigma": 1.1,
            }
        else:
            mock_config = load_json_arg(config.varlen_mock_dataset_config_json)
        mock_config.setdefault("vocab_size", getattr(config.tokenizer, "vocab_size", None))
        return MockSFTLowLevelDataset(**mock_config)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        max_len = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad if tokenizer.pad is not None else eod

        # MockSFTLowLevelDataset returns ``length - 1`` token ids; append EOD
        # to make the conversation end on a stop token, mirroring the real
        # VarlenDataset path. Mock data uses ``tokens == targets`` (no role
        # masking), so inputs and labels are both cut from one buffer.
        raw = self.dataset[int(self.indices[idx % len(self.indices)])]

        # MockVarlenDataset only implements the THD (packed) path; SBHD
        # validation is a real-data numerical-reference mode (guarded against
        # --mock-data in validate_args).
        # THD mode: unpacked single sample, pad to pad_granularity only. A sample
        # longer than sequence_length keeps sequence_length - 1 ids plus EOD.
        keep = len(raw) if len(raw) <= max_len else max_len - 1
        original_seq_len = keep

        pad_granularity = self._calculate_padding_divisor()
        padded_seq_len = original_seq_len + (-original_seq_len % pad_granularity)

        sequence = _right_padded_ids(raw, keep, keep, keep + 1, padded_seq_len + 1, eod, pad)
        loss_mask = torch.ones(padded_seq_len, dtype=torch.float32)
        loss_mask[original_seq_len:] = 0.0  # mask the right-padded tail by position
        return {
            'tokens': sequence[:-1].clone(),
            'labels': sequence[1:].clone(),
            'loss_mask': loss_mask,
            'position_ids': torch.arange(padded_seq_len, dtype=torch.int64),
            'original_seq_len': torch.tensor([original_seq_len], dtype=torch.int32),
            'padded_seq_len': torch.tensor([padded_seq_len], dtype=torch.int32),
        }
