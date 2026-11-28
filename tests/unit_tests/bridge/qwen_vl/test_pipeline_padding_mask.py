# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/b57951ad292e43209f1cc7d4b5500768365a20ac/tests/unit_tests/models/qwen_vl/modelling_qwen3_vl/test_model.py
# Source license: Apache-2.0.

from types import SimpleNamespace

import torch

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.core.packed_seq_params import PackedSeqParams


def test_forward_preserves_collate_packed_layout_for_sequence_parallel(monkeypatch):
    """Packed SP forwards the collator's THD tensors and metadata unchanged.

    The MoE padding mask is the exception: a later pipeline stage receives
    sequence-parallel hidden states, so the mask it builds from the packed
    metadata is scattered the same way before it reaches the routers.
    """

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.torch.cuda.nvtx.range_push",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.torch.cuda.nvtx.range_pop",
        lambda *_args, **_kwargs: None,
    )
    scattered = []

    def scatter_first_half(tensor, group):
        del group
        scattered.append(tensor)
        return tensor[: tensor.shape[0] // 2]

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.tensor_parallel.scatter_to_sequence_parallel_region",
        scatter_first_half,
    )

    class DummyLanguageModel:
        def __init__(self):
            self.rotary_pos_emb = SimpleNamespace(is_thd_format=False)
            self.last_kwargs = None

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return torch.ones(1)

    language_model = DummyLanguageModel()
    model = SimpleNamespace(
        pre_process=False,
        config=SimpleNamespace(sequence_parallel=True),
        pg_collection=SimpleNamespace(
            cp=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
            tp=SimpleNamespace(rank=lambda: 0, size=lambda: 2),
            pp=object(),
        ),
        language_model=language_model,
        use_dist_train=False,
    )
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    input_ids = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 0]])
    position_ids = torch.arange(8).view(1, 1, 8).expand(3, -1, -1).clone()
    labels = input_ids.clone()
    loss_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]])

    output = Qwen3VLModel.forward(
        model,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed_seq_params,
    )

    assert torch.equal(output, torch.ones(1))
    assert language_model.last_kwargs is not None
    assert language_model.last_kwargs["input_ids"] is input_ids
    assert language_model.last_kwargs["position_ids"] is position_ids
    assert language_model.last_kwargs["labels"] is labels
    assert language_model.last_kwargs["loss_mask"] is loss_mask
    assert language_model.last_kwargs["packed_seq_params"] is packed_seq_params
    # Scattered sequence-first like the hidden states this stage receives, then
    # returned batch-first: this TP rank keeps the first half of the 8 tokens.
    assert len(scattered) == 1 and scattered[0].shape == (8, 1)
    assert language_model.last_kwargs["padding_mask"].tolist() == [[False, False, False, True]]


def test_forward_passes_explicit_padding_mask_through_on_later_sequence_parallel_stages(
    monkeypatch,
):
    """A caller-provided mask is already sharded by the step; the model must not scatter it again."""

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.torch.cuda.nvtx.range_push",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.torch.cuda.nvtx.range_pop",
        lambda *_args, **_kwargs: None,
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError(
            "scatter_to_sequence_parallel_region must not run for an explicit mask"
        )

    monkeypatch.setattr(
        "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model.tensor_parallel.scatter_to_sequence_parallel_region",
        fail_if_called,
    )

    class DummyLanguageModel:
        def __init__(self):
            self.rotary_pos_emb = SimpleNamespace(is_thd_format=False)
            self.last_kwargs = None

        def __call__(self, **kwargs):
            self.last_kwargs = kwargs
            return torch.ones(1)

    language_model = DummyLanguageModel()
    model = SimpleNamespace(
        pre_process=False,
        config=SimpleNamespace(sequence_parallel=True),
        pg_collection=SimpleNamespace(
            cp=SimpleNamespace(rank=lambda: 0, size=lambda: 1),
            tp=SimpleNamespace(rank=lambda: 0, size=lambda: 2),
            pp=object(),
        ),
        language_model=language_model,
        use_dist_train=False,
    )
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    input_ids = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 0]])
    position_ids = torch.arange(8).view(1, 1, 8).expand(3, -1, -1).clone()
    sharded_padding_mask = torch.tensor([[False, False, False, True]])

    Qwen3VLModel.forward(
        model,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
        labels=input_ids.clone(),
        loss_mask=torch.ones_like(input_ids, dtype=torch.float),
        packed_seq_params=packed_seq_params,
        padding_mask=sharded_padding_mask,
    )

    assert language_model.last_kwargs["padding_mask"] is sharded_padding_mask
