# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.core.inference.engines.abstract_engine import AbstractEngine


class TestAbstractEngine:

    def test_cannot_instantiate_directly(self):
        """AbstractEngine is abstract; instantiation must fail."""
        with pytest.raises(TypeError):
            AbstractEngine()

    def test_subclass_must_implement_generate(self):
        """A subclass that doesn't override generate() cannot be instantiated."""

        class IncompleteEngine(AbstractEngine):
            pass

        with pytest.raises(TypeError):
            IncompleteEngine()

    def test_subclass_with_generate_can_instantiate(self):
        """A subclass that overrides generate() can be instantiated."""

        class ConcreteEngine(AbstractEngine):
            @staticmethod
            def generate() -> dict:
                return {"input_prompt": "p", "generated_text": "x", "generated_tokens": [1]}

        engine = ConcreteEngine()
        out = engine.generate()
        assert out["generated_text"] == "x"


class TestMCoreEngineReexport:

    def test_mcore_engine_aliases_static_inference_engine(self):
        """The mcore_engine module re-exports StaticInferenceEngine as MCoreEngine."""
        from megatron.core.inference.engines.mcore_engine import MCoreEngine
        from megatron.core.inference.engines.static_engine import StaticInferenceEngine

        assert MCoreEngine is StaticInferenceEngine
