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

  * **Auto schema detection** — three common instruction-tuning layouts are
    auto-detected by column name and normalized to the messages list format
    expected by the parent ``SFTDataset.__getitem__``:

      * **openai-messages** — column ``messages`` (Llama post-training,
        HuggingFaceH4/no_robots, ...)
      * **sharegpt** — column ``conversations`` (OpenOrca, Vicuna, ...)
      * **alpaca / dolly** — at least one of
        ``instruction|prompt|query|question`` + one of
        ``output|response|completion|answer``, plus optional context field
        ``input|context``.

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

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


def _select_converter(
    column_names: List[str],
) -> Tuple[Callable[[Dict[str, Any]], List[Dict[str, str]]], str]:
    """Pick a sample->messages converter based on dataset column names.

    Priority: openai-messages > sharegpt > alpaca/dolly.
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
    raise ValueError(
        "VarlenDataset cannot infer schema from columns "
        f"{sorted(cols)}. Supported schemas: "
        f"alpaca/dolly ({'|'.join(_INSTRUCTION_FIELDS)} + "
        f"{'|'.join(_OUTPUT_FIELDS)} [+ optional {'|'.join(_EXTRA_INPUT_FIELDS)}]), "
        "sharegpt (conversations), openai-messages (messages)."
    )


class VarlenLowLevelDataset(SFTLowLevelDataset):
    """Low-level loader: HF Hub repo / local parquet / local jsonl, normalized.

    Dataset path interpretation:

      * HF Hub repo id (e.g. ``Yukang/LongAlpaca-12k``) — contains ``/`` and
        does not exist on the local filesystem; loaded via
        ``datasets.load_dataset(path, split="train")``.
      * Local ``.parquet`` — loaded via
        ``datasets.load_dataset("parquet", data_files=path, split="all")``;
        parquet's footer schema makes chunked loading safe.
      * Otherwise local jsonl/json — loaded via pandas
        ``read_json(lines=True)`` and wrapped in ``Dataset.from_pandas``.
        We avoid ``datasets.load_dataset("json", ...)`` for local files
        because its pyarrow-based JSON reader infers schema per parallel
        chunk and fails with ``CastError`` when the union of fields varies
        between rows (e.g. LongAlpaca-12k).

    A sample->messages converter is selected once at construction time based
    on column names and applied per-sample at access time.
    """

    def __init__(self, dataset_path: str) -> None:
        try:
            from datasets import Dataset, load_dataset
        except ImportError as exc:
            raise ImportError(
                "VarlenDataset requires the `datasets` library "
                "(pip install datasets)."
            ) from exc

        if _looks_like_hf_id(dataset_path):
            self.dataset = load_dataset(dataset_path, split="train")
        elif dataset_path.endswith(".parquet"):
            self.dataset = load_dataset(
                "parquet", data_files=dataset_path, split="all"
            )
        else:
            try:
                import pandas as pd
            except ImportError as exc:
                raise ImportError(
                    "VarlenDataset requires `pandas` to load local jsonl "
                    "files (pip install pandas)."
                ) from exc
            df = pd.read_json(dataset_path, lines=True)
            self.dataset = Dataset.from_pandas(df, preserve_index=False)

        self._converter, self._schema_name = _select_converter(
            list(self.dataset.column_names)
        )

    @property
    def schema_name(self) -> str:
        """Detected schema name: ``alpaca`` / ``sharegpt`` / ``openai-messages``."""
        return self._schema_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> List[Dict[str, str]]:
        return self._converter(self.dataset[idx])


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
    def build_low_level_dataset(
        dataset_path: str, config: GPTDatasetConfig
    ) -> LowLevelDataset:
        return VarlenLowLevelDataset(dataset_path)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        max_len = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad

        # 1. Pull a single conversation (the low-level dataset emits exactly
        #    one messages list per index — see VarlenLowLevelDataset).
        messages = self.dataset[int(self.indices[idx % len(self.indices)])]

        # 2. Tokenize the conversation once; no multi-conv packing here.
        tokens, targets = tokenizer.tokenize_conversation(
            messages, return_target=True, add_generation_prompt=False
        )
        assert not self.config.reset_position_ids
        assert not self.config.create_attention_mask and not self.config.reset_attention_mask

        tokens_list = tokens.tolist()
        targets_list = targets.tolist()

        # 3. Right-truncate to ``sequence_length + 1`` (we drop the last token
        #    after the input/label shift below). Keep an EOD at the end so a
        #    truncated assistant turn still has a valid stop token.
        if len(tokens_list) > max_len + 1:
            tokens_list = tokens_list[: max_len + 1]
            targets_list = targets_list[: max_len + 1]
            if tokens_list[-1] != eod:
                tokens_list[-1] = eod
                targets_list[-1] = eod

        # 4. Ensure EOD is the last token (unconditional for short samples).
        if tokens_list[-1] != eod:
            tokens_list.append(eod)
            targets_list.append(eod)

        # 5a. BSHD validation mode: right-pad to sequence_length + 1, drop
        #     packing metadata, return shape [sequence_length]. Useful as a
        #     numerical reference for THD path verification (no scheduler,
        #     no dynamic-cp).
        if self.config.varlen_bshd_validation:
            pad_len = max_len + 1 - len(tokens_list)
            if pad_len > 0:
                tokens_list.extend([pad] * pad_len)
                targets_list.extend([pad] * pad_len)
            assert len(tokens_list) == max_len + 1
            input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
            labels = torch.tensor(targets_list[1:], dtype=torch.int64)
            loss_mask = torch.ones(max_len, dtype=torch.float32)
            loss_mask[labels == pad] = 0.0
            loss_mask[labels == IGNORE_INDEX] = 0.0
            return {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'position_ids': torch.arange(max_len, dtype=torch.int64),
            }

        original_seq_len = len(tokens_list) - 1  # length after the shift below

        # 5b. THD path: pad to pad_granularity (dp_size * cp_size * 2 * sp),
        #     the minimum alignment required by CP slicing. We deliberately
        #     do NOT pad to sequence_length — the upstream packing scheduler
        #     will combine variable-length samples up to
        #     max_seqlen_per_dp_cp_rank.
        pad_granularity = self._calculate_padding_divisor()
        mod = original_seq_len % pad_granularity
        if mod != 0:
            pad_len = pad_granularity - mod
            tokens_list.extend([pad] * pad_len)
            targets_list.extend([pad] * pad_len)
        padded_seq_len = len(tokens_list) - 1

        # 6. Apply the next-token shift.
        input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
        labels = torch.tensor(targets_list[1:], dtype=torch.int64)
        position_ids = torch.arange(padded_seq_len, dtype=torch.int64)
        loss_mask = torch.ones(padded_seq_len, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0
        loss_mask[labels == IGNORE_INDEX] = 0.0

        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
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
      * BSHD validation mode (``--varlen-bshd-validation``): right-pads to
        ``sequence_length`` with no packing metadata, for THD numerical
        verification against a non-packed reference run.
    """

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: GPTDatasetConfig
    ) -> LowLevelDataset:
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
            mock_config = json.loads(config.varlen_mock_dataset_config_json)
        return MockSFTLowLevelDataset(**mock_config)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenizer = self.config.tokenizer
        max_len = self.config.sequence_length
        eod = tokenizer.eod
        pad = tokenizer.pad

        # MockSFTLowLevelDataset returns ``length - 1`` token ids; append EOD
        # to make the conversation end on a stop token, mirroring the real
        # VarlenDataset path.
        raw = self.dataset[int(self.indices[idx % len(self.indices)])]
        tokens_list = raw.tolist()
        tokens_list.append(eod)
        # Mock data uses ``tokens == targets`` (no role masking).
        targets_list = list(tokens_list)

        # BSHD validation mode: pad to sequence_length + 1, no packing meta.
        if self.config.varlen_bshd_validation:
            if len(tokens_list) > max_len + 1:
                tokens_list = tokens_list[: max_len - 1] + [eod]
                targets_list = targets_list[: max_len - 1] + [eod]
            pad_len = max_len + 1 - len(tokens_list)
            if pad_len > 0:
                tokens_list.extend([pad] * pad_len)
                targets_list.extend([pad] * pad_len)
            assert len(tokens_list) == max_len + 1
            input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
            labels = torch.tensor(targets_list[1:], dtype=torch.int64)
            loss_mask = torch.ones(max_len, dtype=torch.float32)
            loss_mask[labels == pad] = 0.0
            return {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'position_ids': torch.arange(max_len, dtype=torch.int64),
            }

        # THD mode: unpacked single sample, pad to pad_granularity only.
        if len(tokens_list) > max_len + 1:
            tokens_list = tokens_list[: max_len - 1] + [eod]
            targets_list = targets_list[: max_len - 1] + [eod]
        original_seq_len = len(tokens_list) - 1

        pad_granularity = self._calculate_padding_divisor()
        mod = original_seq_len % pad_granularity
        if mod != 0:
            pad_len = pad_granularity - mod
            tokens_list.extend([pad] * pad_len)
            targets_list.extend([pad] * pad_len)
        padded_seq_len = len(tokens_list) - 1

        input_ids = torch.tensor(tokens_list[:-1], dtype=torch.int64)
        labels = torch.tensor(targets_list[1:], dtype=torch.int64)
        loss_mask = torch.ones(padded_seq_len, dtype=torch.float32)
        loss_mask[labels == pad] = 0.0
        return {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': torch.arange(padded_seq_len, dtype=torch.int64),
            'original_seq_len': torch.tensor([original_seq_len], dtype=torch.int32),
            'padded_seq_len': torch.tensor([padded_seq_len], dtype=torch.int32),
        }
