# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the packed Qwen3.5-VL mock dataset (packed_document)."""

import json
from types import SimpleNamespace

import pytest
import torch

import megatron.training
from examples.multimodal_dev.data.mock_varlen.packed_document import context_scaled_default
from examples.multimodal_dev.data.mock_varlen.qwen35_vl import (
    MOCK_EOD_TOKEN_ID,
    PackedDocumentQwen35VLDataset,
    build_packed_document_plan,
    resolve_varlen_config,
    train_valid_test_varlen_datasets_provider,
)

_IMAGE_TOKEN_ID = 97
_VIDEO_TOKEN_ID = 98
_VISION_START_TOKEN_ID = 96
_VOCAB_SIZE = 100
_PATCH_SIZE = 2
_TEMPORAL_PATCH_SIZE = 2
_SPATIAL_MERGE_SIZE = 2
_PIXEL_DIM = 24


def _bucket_config(*resolutions, weights=None):
    config = {"resolutions": [list(size) for size in resolutions]}
    if weights is not None:
        config["weights"] = list(weights)
    return config


def _assert_samples_equal(lhs, rhs):
    assert lhs.keys() == rhs.keys()
    for key in lhs:
        assert torch.equal(lhs[key], rhs[key]), key


def _assert_vision_contract(sample):
    """Vision starts, image blocks, grids, and pixel rows stay consistent."""
    input_ids = sample["input_ids"]
    grids = sample["image_grid_thw"]
    pixel_values = sample["pixel_values"]
    vision_starts = torch.where(input_ids == _VISION_START_TOKEN_ID)[0].tolist()

    assert tuple(grids.shape) == (len(vision_starts), 3)

    patch_offset = 0
    expected_image_tokens = 0
    block_ends = []
    for vision_start, (t, h, w) in zip(vision_starts, grids.tolist()):
        num_patches = t * h * w
        num_image_tokens = t * (h // _SPATIAL_MERGE_SIZE) * (w // _SPATIAL_MERGE_SIZE)
        image_start = vision_start + 1
        image_end = image_start + num_image_tokens

        assert torch.all(input_ids[image_start:image_end] == _IMAGE_TOKEN_ID)
        assert pixel_values[patch_offset : patch_offset + num_patches].shape == (
            num_patches,
            _PIXEL_DIM,
        )
        patch_offset += num_patches
        expected_image_tokens += num_image_tokens
        block_ends.append(image_end)

    assert patch_offset == pixel_values.shape[0]
    assert expected_image_tokens == int((input_ids == _IMAGE_TOKEN_ID).sum().item())
    assert all(end <= next_start for end, next_start in zip(block_ends, vision_starts[1:]))


def _components(*, length, counts=(0, 1, 2), weights=(1, 1, 1), name="doc", component_weight=1.0):
    """One-component mixture."""
    return [
        {
            "name": name,
            "weight": component_weight,
            "length": dict(length),
            "images_per_document": {"counts": list(counts), "weights": list(weights)},
        }
    ]


def _constant_length(length):
    return {"min": length, "max": length, "mean": length, "sigma": 0}


# Deterministic packed_document geometry (sigma-0 constant): every document
# is one 16-token text-only run — the literal label-semantics tests address
# positions directly.
_CONST_TEXT = 16
_PDOC_CONSTANT_CONFIG = {
    "components": _components(length=_constant_length(_CONST_TEXT), counts=(0,), weights=(1,))
}

# Mixed stochastic packed_document profile: 0/1/2 tiny images inside the
# FINAL logical length — exercises multi-document windows and pixel/grid
# ordering.
_PDOC_MIXED_CONFIG = {
    "components": _components(length={"min": 6, "max": 32, "mean": 14, "sigma": 0.8})
}


def _make_pdoc_dataset(**overrides):
    # Test convenience: window_config/image_size_config overrides merge
    # into the single resolved_profile the constructor takes.
    window_config = overrides.pop("window_config", _PDOC_MIXED_CONFIG)
    image_size_config = overrides.pop("image_size_config", _bucket_config((8, 8), (8, 16)))
    kwargs = {
        "num_samples": 32,
        "seq_length": 128,
        "seed": 1234,
        "vocab_size": _VOCAB_SIZE,
        "image_token_id": _IMAGE_TOKEN_ID,
        "video_token_id": _VIDEO_TOKEN_ID,
        "vision_start_token_id": _VISION_START_TOKEN_ID,
        "resolved_profile": {**window_config, "image_sizes": image_size_config},
        "patch_size": _PATCH_SIZE,
        "temporal_patch_size": _TEMPORAL_PATCH_SIZE,
        "spatial_merge_size": _SPATIAL_MERGE_SIZE,
        "segment_alignment": 1,
    }
    kwargs.update(overrides)
    return PackedDocumentQwen35VLDataset(**kwargs)


def _document_bounds(sample):
    """Per-document (start, end) bounds from the sample's seq_lens."""
    ends = torch.cumsum(sample["seq_lens"], dim=0).tolist()
    return list(zip([0] + ends[:-1], ends))


class TestPackedDocumentDataset:
    def test_window_contract_whole_documents(self):
        # ONE sweep over the default mixed dataset asserts the whole
        # per-sample contract: tensor schema, whole-document seq_lens,
        # loss_mask == (labels != -100) with the plan's supervision count,
        # and pixel/grid rows in the plan's window.images order.
        dataset = _make_pdoc_dataset()
        saw_images = saw_multi_document = saw_multi_image_window = False
        seen_grids = set()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            window = dataset.plan.window(idx % dataset.plan_pool_windows)
            assert sample.keys() == {
                "input_ids",
                "labels",
                "loss_mask",
                "pixel_values",
                "image_grid_thw",
                "seq_lens",
            }
            total = sample["input_ids"].shape[0]
            # NO physical padding in the dataset: T = logical tokens <= S.
            assert total == window.logical_tokens <= 128
            assert sample["labels"].shape == sample["loss_mask"].shape == (total,)
            assert int(sample["seq_lens"].sum().item()) == total
            # Every seq_lens entry is one WHOLE document.
            assert sample["seq_lens"].tolist() == [
                document.logical_length for document in window.documents
            ]
            # loss_mask is exactly the label mask; the window-level
            # supervision count matches the plan accounting.
            assert torch.equal(sample["loss_mask"], (sample["labels"] != -100).float())
            assert int(sample["loss_mask"].sum().item()) == window.supervised_tokens
            # Grid rows in the plan's window.images order == the token
            # placeholder order (checked against input_ids by the contract
            # helper below).
            assert sample["image_grid_thw"].tolist() == [
                list(dataset.grids[image.bucket_index]) for image in window.images
            ]
            _assert_vision_contract(sample)
            saw_images |= bool(sample["image_grid_thw"].shape[0])
            saw_multi_document |= sample["seq_lens"].numel() > 1
            saw_multi_image_window |= len(window.images) > 1
            seen_grids.update(tuple(grid) for grid in sample["image_grid_thw"].tolist())
        assert saw_images and saw_multi_document and saw_multi_image_window
        assert seen_grids == {(1, 4, 4), (1, 4, 8)}

    def test_text_only_document_labels_are_the_shifted_stream_plus_eod(self):
        # Zero-image document: reduces to the pure-text mock exactly —
        # labels = text[1:] + EOD, every position supervised.
        dataset = _make_pdoc_dataset(window_config=_PDOC_CONSTANT_CONFIG)
        for idx in range(4):
            sample = dataset[idx]
            for start, end in _document_bounds(sample):
                assert end - start == _CONST_TEXT
                assert torch.equal(
                    sample["labels"][start : end - 1], sample["input_ids"][start + 1 : end]
                )
                assert sample["labels"][end - 1].item() == MOCK_EOD_TOKEN_ID
                assert (sample["labels"][start:end] != -100).all()
                assert (sample["loss_mask"][start:end] == 1.0).all()

    def test_image_document_targets_walked_position_by_position(self):
        # One (8, 8) image atom (vision_start + 4 placeholders) inside a
        # constant 21-token FINAL length (16 text tokens): every position
        # whose TARGET is an image position is -100; the last atom position
        # targets the first text token (supervised); the last text position
        # targets the (never-input) EOD.
        atom = 1 + 4  # vision_start + merged tokens of the (8, 8) bucket
        config = {
            "components": _components(
                length=_constant_length(atom + _CONST_TEXT), counts=(1,), weights=(1,)
            )
        }
        dataset = _make_pdoc_dataset(window_config=config, image_size_config=_bucket_config((8, 8)))
        sample = dataset[0]
        for start, end in _document_bounds(sample):
            assert end - start == atom + _CONST_TEXT
            assert sample["input_ids"][start].item() == _VISION_START_TOKEN_ID
            assert (sample["input_ids"][start + 1 : start + atom] == _IMAGE_TOKEN_ID).all()
            for position in range(start, end):
                label = int(sample["labels"][position].item())
                if position == end - 1:
                    assert label == MOCK_EOD_TOKEN_ID  # last text position -> EOD
                elif position < start + atom - 1:
                    assert label == -100  # target position+1 is an image position
                else:
                    # Boundary [last IMG] -> first text token, then the
                    # shifted text interior: all supervised.
                    assert label == int(sample["input_ids"][position + 1].item())
                    assert label != -100
            kept = int((sample["labels"][start:end] != -100).sum().item())
            assert kept == _CONST_TEXT + 1  # T + 1 with images
        # Image placeholders and vision_start are never LM targets.
        for special_id in (_IMAGE_TOKEN_ID, _VIDEO_TOKEN_ID, _VISION_START_TOKEN_ID):
            assert not (sample["labels"] == special_id).any()

    def test_full_capacity_document_fills_the_window_alone(self):
        # Directed L_target == S: a constant full-capacity component packs
        # one whole document per window, keeps its images, and the last
        # position still targets the EOD.
        seq_length = 128
        atom = 1 + 4  # the single (8, 8) bucket
        config = {
            "components": _components(
                length=_constant_length(seq_length), counts=(1,), weights=(1,)
            )
        }
        dataset = _make_pdoc_dataset(window_config=config, image_size_config=_bucket_config((8, 8)))
        for idx in range(4):
            sample = dataset[idx]
            assert sample["seq_lens"].tolist() == [seq_length]
            assert sample["input_ids"].shape[0] == seq_length
            assert sample["image_grid_thw"].tolist() == [[1, 4, 4]]  # image kept
            assert sample["labels"][seq_length - 1].item() == MOCK_EOD_TOKEN_ID
            assert sample["loss_mask"][seq_length - 1].item() == 1.0
            # text = S - atoms >= 1; supervised = text + 1 with an image.
            assert int(sample["loss_mask"].sum().item()) == seq_length - atom + 1
            _assert_vision_contract(sample)

    def test_document_boundaries_never_leak(self):
        # The position right before a document boundary targets the EOD,
        # never the next document's first token (EOD is scrubbed from text,
        # so label 13 can never alias a real next-document token).
        dataset = _make_pdoc_dataset()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            assert not (sample["input_ids"] == MOCK_EOD_TOKEN_ID).any()
            bounds = _document_bounds(sample)
            for _, end in bounds:
                assert sample["labels"][end - 1].item() == MOCK_EOD_TOKEN_ID
                assert sample["loss_mask"][end - 1].item() == 1.0
            assert int((sample["labels"] == MOCK_EOD_TOKEN_ID).sum().item()) == len(bounds)

    def test_structural_ids_appear_only_at_planner_positions(self):
        # A random text draw landing on the vision ids or the EOD id would
        # fake a structural marker inside a text run: every occurrence must
        # be exactly one of the planner's image-span positions, and the EOD
        # id never appears as an INPUT token at all.
        dataset = _make_pdoc_dataset(num_samples=16, seq_length=256)
        assert dataset.safe_text_token_id not in dataset.structural_ids
        saw_image_positions = False
        for idx in range(len(dataset)):
            sample = dataset[idx]
            window = dataset.plan.window(idx % dataset.plan_pool_windows)
            expected_starts, expected_placeholders = [], set()
            position = 0
            for document in window.documents:
                for span in document.spans:
                    if span.span_type == "image":
                        expected_starts.append(position)
                        expected_placeholders.update(range(position + 1, position + span.length))
                    position += span.length
            assert (
                torch.where(sample["input_ids"] == _VISION_START_TOKEN_ID)[0].tolist()
                == expected_starts
            )
            assert (
                set(torch.where(sample["input_ids"] == _IMAGE_TOKEN_ID)[0].tolist())
                == expected_placeholders
            )
            assert not (sample["input_ids"] == _VIDEO_TOKEN_ID).any()
            assert not (sample["input_ids"] == MOCK_EOD_TOKEN_ID).any()
            saw_image_positions |= bool(expected_starts)
        assert saw_image_positions

    def test_unusable_id_tables_fail_at_construction(self):
        # The fixed EOD id colliding with a multimodal special id, or falling
        # outside the vocabulary, must fail at construction — as must a
        # special-id table outside the vocabulary itself.
        with pytest.raises(ValueError, match="EOD token id"):
            _make_pdoc_dataset(vision_start_token_id=MOCK_EOD_TOKEN_ID)
        with pytest.raises(ValueError, match="EOD token id"):
            _make_pdoc_dataset(
                vocab_size=13, image_token_id=5, video_token_id=6, vision_start_token_id=7
            )
        with pytest.raises(ValueError, match="token IDs must be in"):
            _make_pdoc_dataset(vocab_size=64)

    def test_windows_are_deterministic_and_access_order_independent(self):
        lhs = _make_pdoc_dataset()
        rhs = _make_pdoc_dataset()
        for idx in reversed(range(8)):
            _assert_samples_equal(lhs[idx], rhs[idx])
        _assert_samples_equal(lhs[3], lhs[3])

    def test_layout_is_fixed_across_content_seeds(self):
        a = _make_pdoc_dataset(seed=1)
        b = _make_pdoc_dataset(seed=2)
        content_differs = False
        for idx in range(len(a)):
            sample_a, sample_b = a[idx], b[idx]
            assert torch.equal(sample_a["seq_lens"], sample_b["seq_lens"])
            assert torch.equal(sample_a["image_grid_thw"], sample_b["image_grid_thw"])
            if not torch.equal(sample_a["input_ids"], sample_b["input_ids"]):
                content_differs = True
        assert content_differs

    def test_plan_seed_overrides_the_layout(self):
        base = _make_pdoc_dataset(num_samples=64)
        other = _make_pdoc_dataset(
            num_samples=64, window_config={**_PDOC_MIXED_CONFIG, "plan_seed": 77}
        )
        assert any(
            not torch.equal(base[idx]["seq_lens"], other[idx]["seq_lens"]) for idx in range(64)
        )

    def test_virtual_length_decouples_from_plan_pool(self):
        # A training-schedule-sized virtual length must not build a
        # training-schedule-sized plan corpus.
        dataset = _make_pdoc_dataset(
            num_samples=1_000_000, window_config={**_PDOC_MIXED_CONFIG, "plan_pool_windows": 8}
        )
        assert len(dataset) == 1_000_000
        assert dataset.plan_pool_windows == 8
        assert len(dataset.plan) == 8
        base, wrapped = dataset[3], dataset[3 + 8]
        # Same pool layout...
        assert torch.equal(base["seq_lens"], wrapped["seq_lens"])
        assert torch.equal(base["image_grid_thw"], wrapped["image_grid_thw"])
        # ...but content stays keyed by the virtual index.
        assert not torch.equal(base["input_ids"], wrapped["input_ids"])

    def test_zero_sample_dataset_reports_no_plan(self):
        # An empty split serves no data: it reports plan None / pool 0 and
        # builds no plan at all.
        dataset = _make_pdoc_dataset(num_samples=0)
        assert len(dataset) == 0
        assert dataset.plan is None
        assert dataset.plan_pool_windows == 0
        with pytest.raises(IndexError):
            dataset[0]

    @pytest.mark.parametrize(
        ("image_size_config", "message"),
        [
            # The bucket table itself is required.
            (None, "resolutions"),
            # Malformed resolutions share one "two positive integers"
            # branch: one wrong-arity row plus the zero boundary.
            ({"resolutions": [[8]]}, "two positive integers"),
            ({"resolutions": [[0, 8]]}, "two positive integers"),
            ({"resolutions": [[8, 8], [8, 16]], "weights": [1]}, "match 'resolutions' in length"),
            ({"resolutions": [[8, 8]], "mode": "buckets"}, "no.*other keys"),
            # Present-but-invalid weights (empty / null / non-number
            # entries — bools are the boundary trap) must never silently
            # become uniform; all hit the same parse branch.
            ({"resolutions": [[8, 8]], "weights": []}, "[Ww]eights"),
            ({"resolutions": [[8, 8]], "weights": None}, "[Ww]eights"),
            ({"resolutions": [[8, 8], [8, 16]], "weights": [True, False]}, "[Ww]eights"),
            # The kernel stays the authority for numeric validity; its
            # single finite/non-negative/positive-sum branch, one row plus
            # the zero-sum boundary (test-scale buckets so weight
            # validation is the only possible failure).
            ({"resolutions": [[8, 8], [8, 16]], "weights": [float("nan"), 1.0]}, "[Ww]eight"),
            ({"resolutions": [[8, 8], [8, 16]], "weights": [0, 0]}, "[Ww]eight"),
        ],
    )
    def test_rejects_missing_or_malformed_bucket_tables(self, image_size_config, message):
        with pytest.raises(ValueError, match=message):
            _make_pdoc_dataset(num_samples=16, image_size_config=image_size_config)

    def test_zero_weight_disables_a_bucket(self):
        dataset = _make_pdoc_dataset(
            num_samples=64,
            window_config={
                "components": _components(
                    length={"min": 12, "max": 32, "mean": 16, "sigma": 0.8},
                    counts=(1,),
                    weights=(1,),
                )
            },
            image_size_config=_bucket_config((8, 8), (8, 16), weights=[0, 1]),
        )
        seen = set()
        for idx in range(64):
            for grid in dataset[idx]["image_grid_thw"].tolist():
                seen.add(tuple(grid))
        assert seen == {(1, 4, 8)}

    def test_plan_pool_and_seed_reject_non_integers(self):
        # bools are the boundary trap (bool is an int subclass); negative
        # seeds would die deep inside numpy's SeedSequence otherwise.
        for key, bad_values in (
            ("plan_pool_windows", (200.0, True, "big", None)),
            ("plan_seed", (True, 12.5, "1234", None, -1)),
        ):
            for bad in bad_values:
                with pytest.raises(ValueError, match=key):
                    _make_pdoc_dataset(window_config={**_PDOC_MIXED_CONFIG, key: bad})

    def test_out_of_range_indices_raise_instead_of_aliasing(self):
        # Silent modulo wrap would duplicate data under a buggy sampler
        # and make plain `for x in dataset` iteration endless (the
        # sequence protocol relies on IndexError). Negative indices keep
        # standard from-the-end semantics; the virtual POOL wrap remains
        # layout-only and in-range.
        dataset = _make_pdoc_dataset(num_samples=6)
        with pytest.raises(IndexError, match="out of range"):
            dataset[6]
        with pytest.raises(IndexError, match="out of range"):
            dataset[-7]
        for key, value in dataset[-1].items():
            assert torch.equal(value, dataset[5][key])
        assert len(list(iter(dataset))) == 6  # iteration terminates


# ---------------------------------------------------------------------------
# Config resolution (context-scaled default, COMPLETE-config contract)
# ---------------------------------------------------------------------------


class TestResolveVarlenConfig:
    def test_omitted_config_resolves_the_context_scaled_default(self):
        # Multi-S default values are already literal-locked by the
        # context_scaled_default tests; this test owns the wiring only.
        expected = context_scaled_default(4096)
        assert resolve_varlen_config(None, seq_length=4096) == expected
        assert resolve_varlen_config("{}", seq_length=4096) == expected

    def test_non_object_json_is_rejected(self):
        with pytest.raises(ValueError, match="JSON object"):
            resolve_varlen_config("[1, 2]", seq_length=4096)

    @pytest.mark.parametrize(
        "spec",
        [
            # There are no migration branches: any unrecognized key hits
            # the same unknown-key error, which carries the resolved default.
            '{"images_per_1k_text_tokens":1.0}'
        ],
    )
    def test_unknown_and_retired_keys_carry_the_resolved_default(self, spec):
        with pytest.raises(ValueError, match="Unknown key") as excinfo:
            resolve_varlen_config(spec, seq_length=4096)
        message = str(excinfo.value)
        for key in json.loads(spec):
            assert key in message
        # "start from the resolved default": the full default JSON is embedded.
        assert json.dumps(context_scaled_default(4096), sort_keys=True) in message

    @pytest.mark.parametrize(
        ("seq_length", "embedded_hint"),
        [
            # In the default's domain the error carries the resolved
            # default JSON to copy and edit; out of domain there is no
            # default to embed and the hint must say so instead.
            (4096, None),
            (64, "unavailable at this seq_length"),
        ],
    )
    def test_partial_config_is_rejected_with_a_default_hint(self, seq_length, embedded_hint):
        # Explicit configs are never merged onto the S-dependent default; the
        # error names the missing keys AND carries the default hint.
        with pytest.raises(ValueError, match="Missing key") as excinfo:
            resolve_varlen_config('{"plan_seed":7}', seq_length=seq_length)
        message = str(excinfo.value)
        for key in ("components", "image_sizes", "plan_pool_windows"):
            assert key in message
        if embedded_hint is None:
            assert json.dumps(context_scaled_default(seq_length), sort_keys=True) in message
        else:
            assert embedded_hint in message

    def test_complete_config_is_returned_verbatim(self):
        config = context_scaled_default(4096)
        config["plan_seed"] = 7
        # seq_length 64 has NO context-scaled default: a complete config must
        # come back verbatim without the default ever being consulted.
        assert resolve_varlen_config(json.dumps(config), seq_length=64) == config


# ---------------------------------------------------------------------------
# Shared plan-construction helper (dataset AND simulator entry)
# ---------------------------------------------------------------------------


class TestBuildPackedDocumentPlan:
    @staticmethod
    def _profile(seq_length, **overrides):
        # Constant full-capacity text-only component: one document per
        # window, so even large pools stay cheap to construct.
        profile = {
            "components": _components(
                length=_constant_length(seq_length), counts=(0,), weights=(1,)
            ),
            "image_sizes": _bucket_config((8, 8), (8, 16)),
            "plan_pool_windows": "auto",
            "plan_seed": 1234,
        }
        profile.update(overrides)
        return profile

    @classmethod
    def _build(cls, seq_length, num_samples, **profile_overrides):
        return build_packed_document_plan(
            cls._profile(seq_length, **profile_overrides),
            seq_length=seq_length,
            num_samples=num_samples,
            segment_alignment=1,
            patch_size=_PATCH_SIZE,
            spatial_merge_size=_SPATIAL_MERGE_SIZE,
        )

    def test_geometry_derivation(self):
        _, geometry = self._build(128, 4)
        assert set(geometry) == {"grids", "weights", "plan_pool_windows", "plan_seed"}
        assert geometry["grids"] == [(1, 4, 4), (1, 4, 8)]
        assert geometry["weights"] == [1.0, 1.0]
        assert geometry["plan_seed"] == 1234

    @pytest.mark.parametrize(
        ("seq_length", "num_samples", "expected_pool", "expected_windows"),
        [
            # The auto pool rule max(2048, ceil(2^26 / S)) with literal
            # anchors at both ends (None = unclamped), then clamped by
            # num_samples.
            (4096, None, 16384, 16384),  # ceil(2^26 / 4096)
            (131072, None, 2048, 2048),  # floor beats ceil(2^26/131072)=512
            (128, 4, 4, 4),  # num_samples clamps the pool
            # An empty split reports a zero pool but still gets a one-window
            # plan built for validation (its caller discards it).
            (128, 0, 0, 1),
        ],
    )
    def test_pool_sizing(self, seq_length, num_samples, expected_pool, expected_windows):
        generator, geometry = self._build(seq_length, num_samples)
        assert geometry["plan_pool_windows"] == expected_pool
        assert len(generator) == expected_windows


# ---------------------------------------------------------------------------
# Provider (runtime contract, migration traps, deterministic startup scan)
# ---------------------------------------------------------------------------


def _pdoc_provider_json(**overrides):
    config = {
        "components": [
            {
                "name": "doc",
                "weight": 1.0,
                "length": {"min": 8, "max": 48, "mean": 16, "sigma": 0},
                "images_per_document": {"counts": [0, 1], "weights": [3, 1]},
            }
        ],
        "image_sizes": {"resolutions": [[32, 32], [64, 32]]},
        "plan_pool_windows": "auto",
        "plan_seed": 1234,
    }
    config.update(overrides)
    return json.dumps(config)


def _provider_args(**overrides):
    args = SimpleNamespace(
        use_varlen_dataset=False,
        sequence_packing_scheduler=None,
        use_packed_sequence=True,
        use_vanilla_collate_fn=True,
        micro_batch_size=1,
        total_seq_length=64,
        seq_length=64,
        varlen_mock_dataset_config_json=None,
        max_vision_patches_per_image=None,
        multimodal_varlen_mock_dataset_config_json=_pdoc_provider_json(),
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=64,
        pad_packed_seq_by_appending_dummy_seq=True,
        padded_vocab_size=248320,
        image_token_id=248056,
        seed=2026,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestPackedDocumentProvider:
    def test_provides_three_splits_with_distinct_seeds(self, monkeypatch, capsys):
        # The pass path in one test: three real splits under a satisfied
        # patch budget, plus the startup-scan launch artifact — the full
        # resolved profile JSON and the per-split pool
        # maxima/conditioning log lines.
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(max_vision_patches_per_microbatch=4096),
        )
        train_ds, val_ds, test_ds = train_valid_test_varlen_datasets_provider((2, 1, 1))
        for dataset in (train_ds, val_ds, test_ds):
            assert isinstance(dataset, PackedDocumentQwen35VLDataset)
        sample = train_ds[0]
        total = sample["input_ids"].shape[0]
        assert int(sample["seq_lens"].sum().item()) == total <= 64
        assert train_ds.seed != val_ds.seed != test_ds.seed
        # The layout is a constant of the profile: all splits share plan_seed.
        assert train_ds.plan_seed == val_ds.plan_seed == test_ds.plan_seed == 1234
        out = capsys.readouterr().out
        assert "resolved profile: " in out
        assert '"components"' in out  # the full resolved JSON is the artifact
        assert out.count("max_raw_patches_per_window=") == 3  # one line per split
        assert "max_images_per_window=" in out
        assert "count_conditioning=" in out
        assert "geometry_conditioning=" in out

    def test_zero_config_resolves_the_context_scaled_default(self, monkeypatch):
        args = _provider_args(
            multimodal_varlen_mock_dataset_config_json=None,
            total_seq_length=None,
            seq_length=4096,
            max_seqlen_per_dp_cp_rank=4096,
        )
        monkeypatch.setattr(megatron.training, "get_args", lambda: args)
        train_ds, _, _ = train_valid_test_varlen_datasets_provider((4, 1, 1))
        assert isinstance(train_ds, PackedDocumentQwen35VLDataset)
        assert train_ds.seq_length == 4096
        assert len(train_ds.grids) == 12  # the DEFAULT_IMAGE_SIZES table
        assert train_ds.plan_seed == 1234
        # auto pool = max(2048, ceil(2^26/4096)) = 16384, clamped by samples.
        assert train_ds.plan_pool_windows == 4
        assert len(train_ds.plan) == 4
        assert train_ds.plan.window_capacity == 4096
        # The per-image guard resolves to the largest drawable default
        # bucket: 1120x896 at patch 16 -> 70 * 56 = 3920 raw patches.
        assert args.max_vision_patches_per_image == 3920

    def test_startup_plan_scan_fails_on_a_heavy_pool(self, monkeypatch, capsys):
        # The plan pool is fully built at construction: a window exceeding
        # --max-vision-patches-per-microbatch is a startup fact, not a
        # mid-epoch surprise. A FAILING launch must still leave the full
        # reproducible artifact: the resolved JSON is printed before any
        # guard verdict and embedded in the guard error itself.
        config = _pdoc_provider_json(
            components=[
                {
                    "name": "dense",
                    "weight": 1.0,
                    "length": {"min": 8, "max": 48, "mean": 16, "sigma": 0},
                    "images_per_document": {"counts": [1], "weights": [1]},
                }
            ]
        )
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(
                multimodal_varlen_mock_dataset_config_json=config,
                max_vision_patches_per_microbatch=1,
            ),
        )
        with pytest.raises(ValueError, match="startup plan scan") as excinfo:
            train_valid_test_varlen_datasets_provider((8, 8, 8))
        assert "Resolved profile: " in str(excinfo.value)
        assert '"components"' in str(excinfo.value)
        out = capsys.readouterr().out
        assert "resolved profile: " in out
        assert '"components"' in out  # the full JSON, not just the hash
        assert "max_raw_patches_per_window=" in out  # the offending split maxima

    @pytest.mark.parametrize(
        ("config_overrides", "message"),
        [
            # Kernel-only failures: these reach neither the bucket-table
            # parser nor the provider matrix, so they prove the zero-sample
            # path still runs the kernel's own validation.
            (
                {
                    "components": [
                        {
                            "name": "doc",
                            "weight": 0,
                            "length": {"min": 8, "max": 48, "mean": 16, "sigma": 0},
                            "images_per_document": {"counts": [0], "weights": [1]},
                        }
                    ]
                },
                r"weight must be > 0",
            ),
            (
                {
                    "components": [
                        {
                            "name": "doc",
                            "weight": 1.0,
                            "length": {"min": 0, "max": 0, "mean": 0, "sigma": 0},
                            "images_per_document": {"counts": [0], "weights": [1]},
                        }
                    ]
                },
                "Invalid truncation window",
            ),
            # Structural failure caught earlier, in the bucket-table parser.
            ({"image_sizes": {"resolutions": [[32, 32]], "weights": [0]}}, "positive finite sum"),
        ],
    )
    def test_zero_sample_splits_still_validate_the_profile(
        self, monkeypatch, config_overrides, message
    ):
        config = _pdoc_provider_json(**config_overrides)
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(multimodal_varlen_mock_dataset_config_json=config),
        )
        with pytest.raises(ValueError, match=message):
            train_valid_test_varlen_datasets_provider((0, 0, 0))

    def test_all_zero_splits_with_a_legal_profile_build_no_plans(self, monkeypatch):
        monkeypatch.setattr(megatron.training, "get_args", lambda: _provider_args())
        datasets = train_valid_test_varlen_datasets_provider((0, 0, 0))
        assert [dataset.plan for dataset in datasets] == [None, None, None]
        assert all(len(dataset) == 0 for dataset in datasets)

    def test_zero_sample_train_split_builds_no_plan(self, monkeypatch, capsys):
        # (0, N, N): the empty train split builds no plan and reports None;
        # the scan log covers only the splits that will actually serve data.
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(max_vision_patches_per_microbatch=4096),
        )
        train_ds, valid_ds, _ = train_valid_test_varlen_datasets_provider((0, 8, 8))
        assert len(train_ds) == 0
        assert train_ds.plan is None
        assert train_ds.plan_pool_windows == 0
        assert len(valid_ds.plan) == 8
        out = capsys.readouterr().out
        assert out.count("max_raw_patches_per_window=") == 2

    @pytest.mark.parametrize(
        ("config", "message"),
        [
            # The resolver owns the branch coverage; this row proves the
            # provider is wired to it.
            ('{"bogus_key":1}', "Unknown key")
        ],
    )
    def test_rejects_removed_legacy_and_malformed_configs(self, monkeypatch, config, message):
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(multimodal_varlen_mock_dataset_config_json=config),
        )
        with pytest.raises(ValueError, match=message):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            # The fixed-physical-target runtime packing contract: all
            # three knobs are startup requirements.
            ({"pad_packed_seq_alignment": None}, "pad-packed-seq-alignment max"),
            ({"pad_packed_seq_alignment": 128}, "pad-packed-seq-alignment max"),
            ({"max_seqlen_per_dp_cp_rank": None}, "max-seqlen-per-dp-cp-rank"),
            # Plan construction is O(S x pool): S outside the supported
            # range is rejected up front (an unbounded S with an explicit
            # profile would otherwise be an hours-long silent startup).
            ({"seq_length": 0, "max_seqlen_per_dp_cp_rank": 0}, "outside the supported range"),
            (
                {"seq_length": 1 << 22, "max_seqlen_per_dp_cp_rank": 1 << 22},
                "outside the supported range",
            ),
            ({"pad_packed_seq_by_appending_dummy_seq": False}, "dummy"),
            # Shared packed-THD requirements apply to packed_document.
            ({"use_packed_sequence": False}, "--use-packed-sequence"),
            ({"use_vanilla_collate_fn": False}, "--use-vanilla-collate-fn"),
            ({"micro_batch_size": 2}, "micro_batch_size == 1"),
            # HybridEP flex dispatch requires variable-token padding.
            (
                {
                    "moe_token_dispatcher_type": "flex",
                    "moe_flex_dispatcher_backend": "hybridep",
                    "moe_hybridep_pad_variable_tokens": False,
                },
                "moe-hybridep-pad-variable-tokens",
            ),
            # Core text-side packing/scheduling paths are incompatible.
            ({"use_varlen_dataset": True}, "--use-varlen-dataset"),
            ({"sequence_packing_scheduler": "dp_balanced"}, "--sequence-packing-scheduler"),
            # 66 % (TP=4 with SP) != 0: the runtime packer
            # (forward_step._pad_multimodal_thd_batch) would reject the
            # very first step; the provider must fail at startup instead.
            (
                {
                    "tensor_model_parallel_size": 4,
                    "sequence_parallel": True,
                    "max_seqlen_per_dp_cp_rank": 66,
                    "total_seq_length": 66,
                    "seq_length": 66,
                },
                "must be a multiple of the CP/SP segment alignment",
            ),
            # The public contract is capacity == S: a CP-local target whose
            # global capacity differs from --seq-length must name both.
            ({"max_seqlen_per_dp_cp_rank": 32}, r"32 \* 1 = 32 must equal seq_length 64"),
            # Legacy flags were folded into the unified config JSON.
            ({"varlen_mock_dataset_config_json": '{"mode":"lognormal"}'}, "no longer reads"),
        ],
    )
    def test_startup_requirement_guards(self, monkeypatch, overrides, message):
        monkeypatch.setattr(megatron.training, "get_args", lambda: _provider_args(**overrides))
        with pytest.raises(ValueError, match=message):
            train_valid_test_varlen_datasets_provider((1, 1, 1))

    @pytest.mark.parametrize(
        ("cp", "tp", "sp", "expected_alignment"),
        [(1, 1, False, 1), (1, 4, True, 4), (1, 4, False, 1), (2, 2, True, 8), (2, 1, False, 4)],
    )
    def test_segment_alignment_and_capacity_derivation(
        self, monkeypatch, cp, tp, sp, expected_alignment
    ):
        # segment_alignment mirrors the runtime packer's per-segment padding
        # multiple; window_capacity is the global (pre-CP-slice) target and
        # must equal --seq-length (T <= S, physically padded to exactly S).
        monkeypatch.setattr(
            megatron.training,
            "get_args",
            lambda: _provider_args(
                context_parallel_size=cp,
                tensor_model_parallel_size=tp,
                sequence_parallel=sp,
                max_seqlen_per_dp_cp_rank=64,
                total_seq_length=64 * cp,
                seq_length=64 * cp,
            ),
        )
        train_ds, _, _ = train_valid_test_varlen_datasets_provider((1, 1, 1))
        assert train_ds.plan.segment_alignment == expected_alignment
        assert train_ds.plan.window_capacity == 64 * cp == train_ds.seq_length

    def test_seq_length_is_the_sole_capacity_authority(self, monkeypatch):
        # The legacy --total-seq-length knob belongs to the fixed-shape
        # providers and is IGNORED here: the physical window capacity is
        # seq_length by construction, whatever total_seq_length says.
        monkeypatch.setattr(
            megatron.training, "get_args", lambda: _provider_args(total_seq_length=128)
        )
        train_ds, _, _ = train_valid_test_varlen_datasets_provider((2, 1, 1))
        assert train_ds.seq_length == 64

    def test_per_image_guard_derivation(self, monkeypatch):
        # Buckets [[32,32],[64,32]] at the real patch size 16: raw patches 4
        # and 8; a zero weight disables the larger bucket; an explicit
        # value is preserved untouched.
        zero_weight_json = _pdoc_provider_json(
            image_sizes={"resolutions": [[32, 32], [64, 32]], "weights": [1, 0]}
        )
        for args, expected in (
            (_provider_args(), 8),
            (_provider_args(multimodal_varlen_mock_dataset_config_json=zero_weight_json), 4),
            (_provider_args(max_vision_patches_per_image=123), 123),
        ):
            monkeypatch.setattr(megatron.training, "get_args", lambda a=args: a)
            train_valid_test_varlen_datasets_provider((1, 1, 1))
            assert args.max_vision_patches_per_image == expected

    def test_help_text_lists_every_packed_document_profile_key(self):
        # The CLI help enumerates the overridable top-level keys by hand;
        # this pins it to the resolved profile schema so the two can never
        # drift.
        import argparse

        from examples.multimodal_dev.arguments import add_multimodal_args

        parser = argparse.ArgumentParser()
        add_multimodal_args(parser)
        (action,) = [
            a for a in parser._actions if a.dest == "multimodal_varlen_mock_dataset_config_json"
        ]
        for key in context_scaled_default(4096):
            assert key in action.help, f"help text is missing profile key {key!r}"
        assert "pad-packed-seq-alignment max" in action.help
