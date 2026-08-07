# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU-only tests for the packed-document plan generation kernel (components v2)."""

import dataclasses

import pytest

from examples.multimodal_dev.data.mock_varlen.packed_document import (
    DEFAULT_IMAGE_SIZES,
    PackedDocument,
    PackedDocumentPlanGenerator,
    context_scaled_default,
)

_PD_BUCKET_MERGED = [4, 16]
_PD_BUCKET_RAW = [16, 64]
_PD_BUCKET_WEIGHTS = [3, 1]
# vision_start + the smallest drawable bucket's merged tokens.
_PD_SMALLEST_ATOM = 5


def _component(**overrides):
    # Small structurally-valid mixture component: tests use tiny geometry
    # instead of the production context-scaled default.
    component = {
        "name": "mix",
        "weight": 1.0,
        "length": {"min": 8, "max": 256, "mean": 48, "sigma": 1.0},
        "images_per_document": {"counts": [0, 1, 2], "weights": [55, 35, 10]},
    }
    component.update(overrides)
    return component


def _pd_config(components=None, **overrides):
    config = {"components": [_component()] if components is None else components}
    config.update(overrides)
    return config


def _pd_text_only(length):
    # sigma=0 constant: every document is a text-only run of *length* tokens.
    return [
        _component(
            length={"min": length, "max": length, "mean": length, "sigma": 0},
            images_per_document={"counts": [0], "weights": [1]},
        )
    ]


def _pd_make(
    seq_length=512,
    num_windows=32,
    seed=1234,
    segment_alignment=1,
    bucket_merged=None,
    bucket_raw=None,
    bucket_weights=None,
    config=None,
    components=None,
    **config_overrides,
):
    if config is None:
        config = _pd_config(components=components, **config_overrides)
    return PackedDocumentPlanGenerator(
        seq_length=seq_length,
        num_windows=num_windows,
        seed=seed,
        config=config,
        bucket_merged_tokens=_PD_BUCKET_MERGED if bucket_merged is None else bucket_merged,
        bucket_raw_patches=_PD_BUCKET_RAW if bucket_raw is None else bucket_raw,
        bucket_weights=_PD_BUCKET_WEIGHTS if bucket_weights is None else bucket_weights,
        segment_alignment=segment_alignment,
    )


def _align_up(value, alignment):
    return -(-int(value) // int(alignment)) * int(alignment)


class TestPackedDocumentValidation:
    """Strict schema surface: retired/unknown keys, malformed components,
    malformed categoricals/lognormals, bools where numbers are expected;
    plus count-feasibility gating, the padding ceiling, the aligned-budget
    gate, and the default-table hash pin."""

    @pytest.mark.parametrize(
        ("config_overrides", "message"),
        [
            # One unknown-key branch: a plain bogus key, and a profile key
            # the build helper is supposed to strip before the kernel.
            ({"bogus": 1}, "unknown key"),
            ({"image_sizes": {"resolutions": [[224, 224]], "weights": [1]}}, "unknown key"),
        ],
    )
    def test_top_level_config_validation(self, config_overrides, message):
        with pytest.raises(ValueError, match=message) as excinfo:
            _pd_make(**config_overrides)
        if message == "unknown key":
            # The allowed-key list is a literal now; keep it honest.
            assert "allowed: ['components']" in str(excinfo.value)

    def test_components_container_validation(self):
        # The components container itself: required, a non-empty list of
        # dicts, every component complete, names unique.
        with pytest.raises(ValueError, match="required key 'components'"):
            _pd_make(config={})
        for components in ({}, [], (), "components", [1], [None]):
            with pytest.raises(ValueError, match="non-empty list|must be a dict"):
                _pd_make(components=components)
        # require_exact_dict owns the is-dict/unknown/missing branches (its
        # own tests cover them); here one row proves this call site passes
        # the right key set and names the offender.
        component = _component()
        del component["length"]
        with pytest.raises(ValueError, match="missing required key") as excinfo:
            _pd_make(components=[component])
        assert "length" in str(excinfo.value)
        with pytest.raises(ValueError, match="not unique"):
            _pd_make(components=[_component(name="dup"), _component(name="dup")])

    @pytest.mark.parametrize(
        ("component_overrides", "message"),
        [
            # One row per hazard family (a few branches carry two
            # exemplars); bools are the boundary hazard for the
            # finite-number checks: bool is an int subclass that
            # require_number must reject by type.
            ({"bogus": 1}, "unknown key"),
            ({"name": ""}, "non-empty string"),
            ({"weight": 0}, "must be > 0"),
            ({"weight": True}, "finite number"),
            ({"length": {"min": 8, "max": 256, "mean": 48}}, "missing required key"),
            ({"length": {"min": 8, "max": 256, "mean": 500, "sigma": 1}}, "must lie in"),
            ({"length": {"min": 8, "max": 256, "mean": 48, "sigma": -1}}, "sigma"),
            ({"length": {"min": "8", "max": 256, "mean": 48, "sigma": 1}}, "must be an integer"),
            (
                {"length": {"min": 0, "max": 256, "mean": 48, "sigma": 1}},
                "Invalid truncation window",
            ),
            (
                {"length": {"min": 64, "max": 8, "mean": 32, "sigma": 1}},
                "Invalid truncation window",
            ),
            ({"length": {"min": 8, "max": 256, "mean": float("inf"), "sigma": 1}}, "finite number"),
            ({"images_per_document": {"counts": [1], "weights": [1], "probs": [1]}}, "unknown key"),
            ({"images_per_document": {"counts": [], "weights": []}}, "non-empty list"),
            (
                {"images_per_document": {"counts": [1, 2], "weights": [1]}},
                "matching counts in length",
            ),
            ({"images_per_document": {"counts": [1], "weights": [True]}}, "finite number"),
            ({"images_per_document": {"counts": [1, 2], "weights": [1, -1]}}, "must be >="),
            # A component with only weight-0 counts is invalid through the
            # Categorical positive-sum rule (disable images with counts [0]).
            ({"images_per_document": {"counts": [1, 2], "weights": [0, 0]}}, "positive finite sum"),
            # Individually finite weights whose sum overflows to inf.
            (
                {"images_per_document": {"counts": [1, 2], "weights": [1e308, 1e308]}},
                "positive finite sum",
            ),
            # A duplicated count is a likely typo: the effective weight
            # would silently be the sum of the duplicate rows.
            ({"images_per_document": {"counts": [1, 1], "weights": [1, 1]}}, "duplicate"),
            ({"images_per_document": {"counts": [-1], "weights": [1]}}, "must be >= 0"),
            ({"images_per_document": {"counts": [True], "weights": [1]}}, "must be an integer"),
        ],
    )
    def test_component_validation(self, component_overrides, message):
        with pytest.raises(ValueError, match=message):
            _pd_make(components=[_component(**component_overrides)])

    def test_constructor_argument_validation(self):
        with pytest.raises(ValueError, match="must be positive"):
            _pd_make(seq_length=0)
        with pytest.raises(ValueError, match="must be positive"):
            _pd_make(num_windows=0)
        with pytest.raises(ValueError, match="segment_alignment"):
            _pd_make(segment_alignment=0)
        with pytest.raises(ValueError, match="must be an integer"):
            _pd_make(segment_alignment=True)
        with pytest.raises(ValueError, match="equal length"):
            _pd_make(bucket_merged=[4], bucket_raw=[16, 64], bucket_weights=[1, 1])
        with pytest.raises(ValueError, match="non-empty"):
            _pd_make(bucket_merged=[], bucket_raw=[], bucket_weights=[])
        for bad_weights in ([float("nan"), 1], [-1, 1], [0, 0], [1e308, 1e308]):
            # The last row: individually finite weights whose SUM overflows
            # to inf would silently corrupt every bucket draw.
            with pytest.raises(ValueError, match="Bucket weights"):
                _pd_make(bucket_weights=bad_weights)

    def test_component_weight_sum_must_be_finite(self):
        # Individually finite weights that overflow the normalizer would
        # zero the CDF and silently pin EVERY document to the last
        # component; only weight ratios matter, so this must fail loudly.
        with pytest.raises(ValueError, match="non-finite"):
            _pd_make(
                components=[_component(name="a", weight=1e308), _component(name="b", weight=1e308)]
            )

    def test_two_layer_count_feasibility_gates_at_startup(self):
        # Layer 1 — the smallest drawable atom is _PD_SMALLEST_ATOM tokens:
        # counts 2 and 3 both overflow a length.min=8 document, and count 0
        # is not in the support.
        assert 2 * _PD_SMALLEST_ATOM + 1 > 8  # even the smallest count fails the floor
        with pytest.raises(ValueError, match="no positive-weight") as excinfo:
            _pd_make(
                components=[_component(images_per_document={"counts": [2, 3], "weights": [1, 1]})]
            )
        assert "'mix'" in str(excinfo.value)
        assert "length.min=8" in str(excinfo.value)
        # Layer 2 — count 60 needs 60 * _PD_SMALLEST_ATOM + 1 = 301 >
        # length.max=256 even with the smallest drawable atom: a dead entry
        # that could never be drawn.
        assert 60 * _PD_SMALLEST_ATOM + 1 == 301 > 256
        with pytest.raises(ValueError, match="dead entry") as excinfo:
            _pd_make(
                components=[_component(images_per_document={"counts": [0, 60], "weights": [1, 1]})]
            )
        assert "'mix'" in str(excinfo.value)
        assert "count 60" in str(excinfo.value)

    def test_component_length_max_above_the_aligned_budget_raises(self):
        with pytest.raises(ValueError, match="aligned window budget") as excinfo:
            _pd_make(
                components=[_component(length={"min": 8, "max": 600, "mean": 48, "sigma": 1.0})]
            )
        assert "'mix'" in str(excinfo.value)
        # max == budget is legal: a drawn document may fill a whole window.
        assert _pd_make(seq_length=256, num_windows=2).window_capacity == 256

    def test_zero_weight_counts_do_not_gate_feasibility(self):
        # A weight-0 count is disabled at parse time: an absurd disabled
        # count must neither fail startup feasibility nor ever be drawn.
        generator = _pd_make(
            num_windows=8,
            components=[_component(images_per_document={"counts": [0, 1000], "weights": [1, 0]})],
        )
        for idx in range(8):
            for document in generator.window(idx).documents:
                assert sum(1 for span in document.spans if span.span_type == "image") == 0

    def test_capacity_must_be_a_multiple_of_segment_alignment(self):
        # Per-segment padding comes in multiples of A: a non-multiple capacity
        # can never be filled exactly and must fail at construction.
        with pytest.raises(ValueError, match="multiple of"):
            _pd_make(seq_length=31, num_windows=2, segment_alignment=16)
        assert _pd_make(seq_length=512, num_windows=2, segment_alignment=16).window_capacity == 512

    def test_default_bucket_table_is_pinned(self):
        # The default bucket table is pinned row by row: any change must be a
        # deliberate re-decision that updates the provenance comment with
        # it. The calibration's extreme
        # single-image tail (1184x960, 960x1184, 1184x1184) is excluded
        # from the zero-config default.
        assert DEFAULT_IMAGE_SIZES["resolutions"] == [
            [224, 224],
            [448, 224],
            [224, 448],
            [352, 448],
            [448, 352],
            [448, 448],
            [672, 448],
            [448, 672],
            [896, 672],
            [672, 896],
            [1120, 896],
            [896, 1120],
        ]
        assert DEFAULT_IMAGE_SIZES["weights"] == [33, 9, 9, 7, 7, 13, 6.5, 6.5, 5.5, 5.5, 3.5, 3.5]
        for removed in ([1184, 960], [960, 1184], [1184, 1184]):
            assert removed not in DEFAULT_IMAGE_SIZES["resolutions"]


class TestContextScaledDefault:

    @pytest.mark.parametrize(
        ("seq_length", "short_max", "long_mean"),
        [
            (4096, 4096, 2867),
            (8192, 4096, 5734),
            (32768, 4096, 22938),
            (65536, 4096, 45875),
            (131072, 4096, 91750),
        ],
    )
    def test_anchor_values_are_literal(self, seq_length, short_max, long_mean):
        # Pure float arithmetic is deterministic: the whole resolved profile
        # is a literal — exactly the four public keys.
        assert context_scaled_default(seq_length) == {
            "components": [
                {
                    "name": "short",
                    "weight": 95.0,
                    "length": {"min": 512, "max": short_max, "mean": 1536, "sigma": 0.30},
                    "images_per_document": {"counts": [0, 1], "weights": [45, 55]},
                },
                {
                    "name": "long",
                    "weight": 5.0,
                    "length": {"min": 2048, "max": seq_length, "mean": long_mean, "sigma": 0.8},
                    "images_per_document": {
                        "counts": [0, 1, 2, 4, 8],
                        "weights": [24, 35, 25, 12, 4],
                    },
                },
            ],
            "image_sizes": {
                "resolutions": DEFAULT_IMAGE_SIZES["resolutions"],
                "weights": DEFAULT_IMAGE_SIZES["weights"],
            },
            "plan_pool_windows": "auto",
            "plan_seed": 1234,
        }

    def test_profile_copies_the_bucket_table(self):
        profile = context_scaled_default(4096)
        assert profile["image_sizes"]["resolutions"] is not DEFAULT_IMAGE_SIZES["resolutions"]
        assert profile["image_sizes"]["weights"] is not DEFAULT_IMAGE_SIZES["weights"]

    @pytest.mark.parametrize("seq_length", [4095, 131073])
    def test_out_of_range_seq_length_raises(self, seq_length):
        with pytest.raises(ValueError, match=r"\[4096, 131072\]"):
            context_scaled_default(seq_length)

    @pytest.mark.parametrize("seq_length", [True, 4096.0])
    def test_non_integer_seq_length_is_rejected(self, seq_length):
        with pytest.raises(ValueError, match="must be an integer"):
            context_scaled_default(seq_length)


class TestPackedDocumentPlanner:
    def test_packed_document_field_schema(self):
        # The v2 public plan schema: the v1 overflow/truncation fields
        # (truncated, text_tokens_removed) are gone, count conditioning is
        # in, and the document ordinal is an RNG key, not a stored field.
        assert {field.name for field in dataclasses.fields(PackedDocument)} == {
            "component_index",
            "spans",
            "logical_length",
            "supervised_tokens",
            "image_count_conditioned",
            "image_geometry_substitutions",
        }

    def test_same_seed_determinism_and_access_order_independence(self):
        lhs, rhs = _pd_make(num_windows=16), _pd_make(num_windows=16)
        forward = [lhs.window(idx) for idx in range(16)]
        backward = [rhs.window(idx) for idx in reversed(range(16))][::-1]
        assert forward == backward
        assert lhs.total_documents == rhs.total_documents
        assert [d.component_index for i in range(len(lhs)) for d in lhs.window(i).documents] == [
            d.component_index for i in range(len(rhs)) for d in rhs.window(i).documents
        ]
        assert lhs.total_padding_fraction == rhs.total_padding_fraction
        assert _pd_make(num_windows=16, seed=999).window(0) != lhs.window(0)

    def test_prefix_stability_including_the_last_window(self):
        short, full = _pd_make(num_windows=12), _pd_make(num_windows=24)
        for idx in range(12):
            assert short.window(idx) == full.window(idx)
        # The last short window is only closed by its overflowing successor;
        # it must still be identical to the longer plan's window.
        assert short.window(11).seq_lens == full.window(11).seq_lens
        assert [document.spans for document in short.window(11).documents] == [
            document.spans for document in full.window(11).documents
        ]

    def test_full_pool_sweep_holds_every_invariant_and_aggregate(self):
        # THE consolidated reviewer sweep: one pass over a 64-window pool
        # asserts every per-window identity (spans conserve the logical
        # length, L <= S, aligned costs fit the capacity) and every pool
        # aggregate (conservation totals,
        # per-component document counts, conditioning counters, maxima)
        # against a hand-computed rescan.
        generator = _pd_make(num_windows=64, segment_alignment=8)
        alignment = generator.segment_alignment
        capacity = generator.window_capacity
        windows = [generator.window(idx) for idx in range(64)]
        for window in windows:
            assert window.seq_lens == tuple(
                document.logical_length for document in window.documents
            )
            aligned = [
                _align_up(document.logical_length, alignment) for document in window.documents
            ]
            for document, cost in zip(window.documents, aligned):
                assert sum(span.length for span in document.spans) == document.logical_length
                assert document.logical_length <= generator.seq_length
                assert cost <= capacity
            assert sum(aligned) <= capacity
            assert window.alignment_padding >= 0
            assert window.tail_padding >= 0
            assert (
                window.logical_tokens + window.alignment_padding + window.tail_padding == capacity
            )
        # Pool aggregates equal the rescan.
        padding = sum(window.alignment_padding + window.tail_padding for window in windows)
        assert generator.total_padding_fraction == pytest.approx(padding / (64 * capacity))
        assert generator.total_documents == sum(len(window.documents) for window in windows)
        documents = [document for window in windows for document in window.documents]
        assert len(documents) == generator.total_documents
        assert generator.image_count_conditioning_events == sum(
            document.image_count_conditioned for document in documents
        )
        assert generator.image_conditioning_events == sum(
            document.image_geometry_substitutions > 0 for document in documents
        )
        # Pool maxima match the hand-computed scan.
        assert generator.pool_max_raw_patches == max(window.raw_patches for window in windows)
        assert generator.pool_max_images == max(len(window.images) for window in windows)
        assert generator.pool_max_logical_tokens == max(window.logical_tokens for window in windows)
        all_images = [image for window in windows for image in window.images]
        assert all_images  # the mixed profile draws images
        assert generator.pool_max_image_raw_patches == max(
            image.raw_patches for image in all_images
        )

    def test_document_layout_is_image_atoms_prefix_then_one_text_run(self):
        generator = _pd_make(num_windows=32)
        saw_images = saw_text_only = False
        for idx in range(32):
            window = generator.window(idx)
            for document in window.documents:
                kinds = [span.span_type for span in document.spans]
                image_count = kinds.count("image")
                # All image atoms at the prefix, then exactly one text run;
                # the terminal EOD is never a span (never an input position).
                assert kinds == ["image"] * image_count + ["text"]
                text_span = document.spans[-1]
                assert text_span.length >= 1  # text = L_target - atoms >= 1
                # L-final-first: the drawn length is the FINAL logical length.
                assert 8 <= document.logical_length <= 256  # component length window
                for span in document.spans[:image_count]:
                    assert span.length == 1 + span.merged_tokens  # vision_start + placeholders
                    assert span.merged_tokens == _PD_BUCKET_MERGED[span.bucket_index]
                    assert span.raw_patches == _PD_BUCKET_RAW[span.bucket_index]
                # Full-sequence supervision: T + 1 with images (the last atom
                # position targets the first text token), T without.
                assert document.supervised_tokens == text_span.length + (1 if image_count else 0)
                saw_images |= image_count > 0
                saw_text_only |= image_count == 0
            # window.images order: document order -> in-document order, with
            # geometry matching the spans; vision/raw totals are their sums.
            expected = [
                (span.bucket_index, span.merged_tokens, span.raw_patches)
                for document in window.documents
                for span in document.spans
                if span.span_type == "image"
            ]
            actual = [
                (image.bucket_index, image.merged_tokens, image.raw_patches)
                for image in window.images
            ]
            assert actual == expected
            assert window.vision_tokens == sum(1 + image.merged_tokens for image in window.images)
            assert window.raw_patches == sum(image.raw_patches for image in window.images)
        # Both supervised branches were exercised: image-bearing documents
        # (T + 1) and text-only documents (exactly T, no images anywhere).
        assert saw_images and saw_text_only

    def test_component_mixture_and_per_component_document_counts(self):
        config = _pd_config(
            components=[
                _component(
                    name="a",
                    weight=3.0,
                    length={"min": 10, "max": 10, "mean": 10, "sigma": 0},
                    images_per_document={"counts": [0], "weights": [1]},
                ),
                _component(
                    name="b",
                    weight=1.0,
                    length={"min": 20, "max": 20, "mean": 20, "sigma": 0},
                    images_per_document={"counts": [0], "weights": [1]},
                ),
            ]
        )
        generator = _pd_make(config=config, seq_length=256, num_windows=64)
        assert [component.name for component in generator.components] == ["a", "b"]
        assert [component.weight for component in generator.components] == [3.0, 1.0]
        constant_length = {"a": 10, "b": 20}
        counted = [0, 0]
        for idx in range(64):
            for document in generator.window(idx).documents:
                component = generator.components[document.component_index]
                # component_index is the drawn mixture component: the drawn
                # constant length identifies it unambiguously.
                assert document.logical_length == constant_length[component.name]
                counted[document.component_index] += 1
        assert sum(counted) == generator.total_documents
        assert all(count > 0 for count in counted)
        # Document-count weights 3:1 -> component "a" share around 0.75.
        share = counted[0] / sum(counted)
        assert 0.6 <= share <= 0.9

    def test_alignment_regression_logical_fit_is_not_physical_fit(self):
        # L=1361 aligns to 1376 at A=16; three documents fit
        # logically (4083 <= 4096) but not physically (4128 > 4096) and must
        # never share a window.
        components = _pd_text_only(1361)
        generator = _pd_make(
            seq_length=4096, num_windows=16, segment_alignment=16, components=components
        )
        for idx in range(16):
            window = generator.window(idx)
            assert window.seq_lens == (1361, 1361)
            assert window.alignment_padding == 30
            assert window.tail_padding == 4096 - 2 * 1376
        # A=1 is pure logical packing: the same documents go three-a-window.
        loose = _pd_make(
            seq_length=4096, num_windows=16, segment_alignment=1, components=components
        )
        for idx in range(16):
            window = loose.window(idx)
            assert window.seq_lens == (1361, 1361, 1361)
            assert window.alignment_padding == 0
            assert window.tail_padding == 4096 - 3 * 1361

    def test_lookahead_overflow_document_opens_the_next_window(self, monkeypatch):
        # Structural proof of ordinal contiguity: record every ordinal the
        # pool build actually draws (value equality alone could not
        # distinguish a skip/duplicate if two ordinals ever produced
        # identical documents).
        drawn: list[int] = []
        original = PackedDocumentPlanGenerator._draw_document
        monkeypatch.setattr(
            PackedDocumentPlanGenerator,
            "_draw_document",
            lambda self, ordinal: drawn.append(ordinal) or original(self, ordinal),
        )
        generator = _pd_make(num_windows=32)
        assert drawn == list(range(len(drawn)))  # each ordinal once, in order
        monkeypatch.undo()
        flat = [document for idx in range(32) for document in generator.window(idx).documents]
        # The document that closed window N is exactly the first document
        # of window N+1: the flattened pool IS the sequential draw
        # (dataclass equality covers spans and accounting).
        assert flat == [generator._draw_document(ordinal) for ordinal in range(len(flat))]
        assert len(flat) <= len(drawn) <= len(flat) + 1  # at most one lookahead overhang

    def test_full_capacity_document_occupies_a_window_alone(self):
        # Directed, not pool luck: a component whose length is constant S
        # legally draws L_target == S and fills a whole window by itself —
        # images kept, text = S - atoms >= 1, zero padding.
        seq_length = 512
        generator = _pd_make(
            seq_length=seq_length,
            num_windows=8,
            segment_alignment=16,
            components=[
                _component(
                    length={"min": seq_length, "max": seq_length, "mean": seq_length, "sigma": 0},
                    images_per_document={"counts": [2], "weights": [1]},
                )
            ],
        )
        for idx in range(8):
            window = generator.window(idx)
            assert window.seq_lens == (seq_length,)
            (document,) = window.documents
            assert document.logical_length == seq_length == generator.window_capacity
            assert _align_up(document.logical_length, 16) == generator.window_capacity
            assert window.alignment_padding == 0
            assert window.tail_padding == 0
            image_spans = [span for span in document.spans if span.span_type == "image"]
            assert len(image_spans) == 2  # the drawn count is preserved
            text_span = document.spans[-1]
            assert text_span.length == seq_length - sum(span.length for span in image_spans)
            assert text_span.length >= 1

    def test_constructor_accepts_length_dependent_counts(self):
        # Two-layer vs static distinction: with a 41-token atom
        # (bucket_merged=[40]), count 2 needs 2*41 + 1 = 83 tokens — it does
        # NOT fit at length.min=32 but does at length.max=4096. Layer 1 is
        # satisfied by count 0 at the floor, layer 2 by every count at the
        # max, so construction must succeed; a static all-counts-at-min
        # filter would wrongly reject this component.
        generator = _pd_make(
            seq_length=4096,
            num_windows=16,
            bucket_merged=[40],
            bucket_raw=[160],
            bucket_weights=[1],
            components=[
                _component(
                    length={"min": 32, "max": 4096, "mean": 512, "sigma": 1.0},
                    images_per_document={"counts": [0, 2], "weights": [1, 1]},
                )
            ],
        )
        assert len(generator) == 16
        # The acceptance is not vacuous: realized draws include both the
        # full count (large L) and count-conditioned documents whose drawn L
        # is too small for count 2 (the categorical renormalizes to count 0).
        documents = [document for idx in range(16) for document in generator.window(idx).documents]
        counts = {
            sum(1 for span in document.spans if span.span_type == "image") for document in documents
        }
        assert 2 in counts
        conditioned = [document for document in documents if document.image_count_conditioned]
        assert conditioned
        assert all(
            sum(1 for span in document.spans if span.span_type == "image") == 0
            and document.logical_length < 83
            for document in conditioned
        )

    def test_tiny_constant_length_conditions_counts_to_zero(self):
        # Constant L_target=40 with a 41-token smallest atom: count 1 fits at
        # length.max (so it is not a dead entry) but never under the drawn
        # length, so every document is conditioned down to 0 images.
        generator = _pd_make(
            seq_length=64,
            num_windows=16,
            bucket_merged=[40],
            bucket_raw=[160],
            bucket_weights=[1],
            components=[
                _component(
                    length={"min": 40, "max": 64, "mean": 40, "sigma": 0},
                    images_per_document={"counts": [0, 1], "weights": [1, 1]},
                )
            ],
        )
        for idx in range(16):
            for document in generator.window(idx).documents:
                assert sum(1 for span in document.spans if span.span_type == "image") == 0
                assert document.image_count_conditioned
                assert document.image_geometry_substitutions == 0
        assert generator.image_count_conditioning_events == generator.total_documents
        assert all(
            d.image_geometry_substitutions == 0 for w in generator._windows for d in w.documents
        )

    def test_forced_geometry_substitution_preserves_k(self):
        # Atoms are 401 or 5 tokens; k is always 2 and the constant L_target
        # of 19 only fits 2*5 + text: every draw that includes a huge atom
        # must substitute its GEOMETRY down to the smallest drawable bucket
        # while the image COUNT never changes and images are never dropped.
        generator = _pd_make(
            seq_length=20,
            num_windows=16,
            bucket_merged=[400, 4],
            bucket_raw=[1600, 16],
            bucket_weights=[1, 1],
            components=[
                _component(
                    length={"min": 19, "max": 19, "mean": 19, "sigma": 0},
                    images_per_document={"counts": [2], "weights": [1]},
                )
            ],
        )
        for idx in range(16):
            for document in generator.window(idx).documents:
                image_spans = [span for span in document.spans if span.span_type == "image"]
                assert len(image_spans) == 2  # k is a contract
                # Post-conditioning every atom fits: only the smallest bucket
                # can coexist with the text remainder at this length.
                assert all(span.merged_tokens == 4 for span in image_spans)
                assert document.logical_length == 19
                assert document.spans[-1].length == 19 - 2 * 5
        assert any(
            d.image_geometry_substitutions > 0 for w in generator._windows for d in w.documents
        )
        assert generator.image_conditioning_events > 0
        # The count itself was always feasible: no count conditioning.
        assert generator.image_count_conditioning_events == 0

    def test_window_capacity_is_the_sequence_length(self):
        generator = _pd_make(seq_length=512, num_windows=8)
        assert generator.window_capacity == 512
        for idx in range(8):
            window = generator.window(idx)
            assert window.logical_tokens + window.alignment_padding + window.tail_padding == 512
