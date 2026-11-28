# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/hf_pretrained/base.py
# Source license: Apache-2.0.

#!/usr/bin/env python3
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

import fnmatch
import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin
from transformers.utils import CHAT_TEMPLATE_DIR

from megatron.bridge.models.hf_pretrained.state import (
    SafeTensorsStateSource,
    StateDict,
    StateSource,
)

logger = logging.getLogger(__name__)


def _artifact_fingerprint(artifact: Any) -> str | None:
    """Summarize the state ``save_pretrained`` writes for an artifact, to detect edits made after loading.

    Returns None when the artifact cannot be summarized; saving then re-serializes it.
    """
    try:
        if isinstance(artifact, PreTrainedTokenizerBase):
            backend = getattr(artifact, "backend_tokenizer", None)
            state = (
                len(artifact),
                str(artifact.added_tokens_decoder),
                artifact.special_tokens_map,
                getattr(artifact, "chat_template", None),
                artifact.padding_side,
                artifact.truncation_side,
                artifact.model_max_length,
                # A fast tokenizer's saved pipeline. Encoding changes only its padding and truncation.
                (
                    None
                    if backend is None
                    else [
                        (
                            None
                            if (component := getattr(backend, name)) is None
                            else component.__getstate__()
                        )
                        for name in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
                    ]
                ),
            )
        elif isinstance(artifact, ProcessorMixin):
            # ``to_json_string`` leaves out the processor's tokenizer and chat template.
            state = (
                artifact.to_json_string(),
                getattr(artifact, "chat_template", None),
                _artifact_fingerprint(getattr(artifact, "tokenizer", None)),
            )
        elif hasattr(artifact, "to_json_string"):
            state = artifact.to_json_string()
        else:
            return None
    except Exception:  # e.g. a stand-in object without the full tokenizer API
        return None
    return repr(state)


def _file_mtimes(path: Path) -> Dict[str, int]:
    return {
        file.relative_to(path).as_posix(): file.stat().st_mtime_ns
        for file in path.rglob("*")
        if file.is_file()
    }


class PreTrainedBase(ABC):
    """
    Abstract base class for all pretrained models.

    This class provides a generic mechanism for managing model artifacts
    (e.g., config, tokenizer) with lazy loading. Subclasses that are
    decorated with `@dataclass` can define artifacts as fields with metadata
    specifying a loader method. The `model` itself is handled via a
    dedicated property that relies on the abstract `_load_model` method.

    Example:
        @dataclass
        class MyModel(PreTrainedBase):
            config: AutoConfig = field(
                init=False,
                metadata=artifact(loader="_load_config")
            )

            def _load_model(self) -> "PreTrainedModel":
                # Implementation for the loading logic
                ...
    """

    model_name_or_path: Union[str, Path]
    ARTIFACTS: ClassVar[List[str]] = []
    OPTIONAL_ARTIFACTS: ClassVar[List[str]] = []
    # Source files of each artifact, copied verbatim on save when the artifact is unchanged since it was loaded.
    SOURCE_ARTIFACT_FILES: ClassVar[Dict[str, List[str]]] = {
        "tokenizer": [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
            "tokenizer.model",
            "*.tiktoken",
            "chat_template.jinja",
            "chat_template.json",
            f"{CHAT_TEMPLATE_DIR}/*.jinja",
        ],
        "processor": [
            "processor_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
            "chat_template.json",
            f"{CHAT_TEMPLATE_DIR}/*.jinja",
        ],
        "image_processor": ["preprocessor_config.json"],
    }
    # Loading arguments that do not change what an artifact saves. Any other argument, e.g. a tokenizer's
    # ``model_max_length``, overrides the source's files, so the artifact is re-serialized instead.
    LOAD_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset(
        {
            "revision",
            "cache_dir",
            "token",
            "local_files_only",
            "force_download",
            "proxies",
            "torch_dtype",
            "dtype",
            "device_map",
            "attn_implementation",
            "low_cpu_mem_usage",
        }
    )

    def __init__(self, **kwargs):
        self._state_dict_accessor: Optional[StateDict] = None
        self.init_kwargs = kwargs
        # Artifacts loaded from the source with their fingerprints, to tell on save whether they changed.
        self._loaded_artifacts: Dict[str, tuple[Any, str | None]] = {}

        # File patterns used to copy files used for custom modeling, e.g.
        # modeling_*.py, configuration_*.py, tokenization_*.py,
        # processing_*.py, feature_extraction_*.py. HF repos
        # may have additional python files that are imported in the
        # aforementioned, so we just copy all python files
        # instead of above specific prefixes.
        # Currently, we can capture all json files via ARTIFACTS.
        self.custom_file_patterns = ["*.py"]

    def get_artifacts(self) -> Dict[str, str]:
        """Get the artifacts dictionary mapping artifact names to their attribute names."""
        return {artifact: f"_{artifact}" for artifact in self.ARTIFACTS}

    @staticmethod
    def _cleanup_hf_local_dir_cache(target_path: Path) -> None:
        """Remove Hugging Face Hub metadata created by hf_hub_download(local_dir=...)."""
        huggingface_cache = target_path / ".cache" / "huggingface"
        if huggingface_cache.exists():
            shutil.rmtree(huggingface_cache, ignore_errors=True)

        try:
            (target_path / ".cache").rmdir()
        except OSError:
            pass

    @staticmethod
    def _copy_cached_hub_files(
        repo_id: str, target_path: Path, patterns: List[str], *, revision: str | None
    ) -> List[str]:
        """Copy files matching patterns from the local Hugging Face cache snapshot of a Hub repository.

        Used when the Hub cannot be listed, for example with HF_HUB_OFFLINE=1. Each pattern is
        resolved separately, so a pattern whose files are not all cached does not block the others.
        Files are matched and placed by their repository-relative paths, as Hub downloads are.

        Args:
            repo_id: Hugging Face Hub repository ID.
            target_path: Directory to copy the files into.
            patterns: File names or glob patterns, matched against repository-relative paths.
            revision: Hub revision to read from the cache; None uses the cached default branch.

        Returns:
            List of copied repository-relative file names; empty if nothing matching is cached.
        """
        import fnmatch

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return []

        revision_kwargs = {"revision": revision} if revision is not None else {}
        copied_files = []
        for file_spec in patterns:
            try:
                # allow_patterns limits huggingface_hub's cached-snapshot completeness check to the
                # requested files, so unrelated files missing from the cache do not fail the lookup.
                snapshot_dir = Path(
                    snapshot_download(
                        repo_id=repo_id,
                        local_files_only=True,
                        allow_patterns=[file_spec],
                        **revision_kwargs,
                    )
                )
            except Exception as exc:
                logger.debug("No cached files of %s match %s: %s", repo_id, file_spec, exc)
                continue

            cached_files = sorted(path for path in snapshot_dir.rglob("*") if path.is_file())
            for file_path in cached_files:
                repo_file = file_path.relative_to(snapshot_dir).as_posix()
                if repo_file in copied_files or not fnmatch.fnmatch(repo_file, file_spec):
                    continue
                target_file = target_path / repo_file
                try:
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, target_file)
                    copied_files.append(repo_file)
                except OSError:
                    # Silently skip files that can't be copied
                    pass
        return copied_files

    def _copy_custom_modeling_files(
        self,
        source_path: Union[str, Path],
        target_path: Union[str, Path],
        file_patterns: Optional[List[str]] = None,
        revision: str | None = None,
    ) -> List[str]:
        """Copy files matching patterns from source to target directory.

        This preserves custom modeling files that were used during model loading
        with trust_remote_code=True, or copies additional files specified by patterns.

        Args:
            source_path: Source directory or HuggingFace Hub ID containing files
            target_path: Target directory to copy files to
            file_patterns: Optional list of file names or patterns to copy.
                Supports both exact file names (e.g., "vocab.json") and glob patterns
                (e.g., "*.json"). If not provided, defaults to self.custom_file_patterns.
            revision: Optional Hugging Face Hub revision to use for remote sources.

        Returns:
            Source-relative names of the copied files, placed at the same relative paths in the target.
        """
        source_path = Path(source_path)
        target_path = Path(target_path)
        patterns = file_patterns if file_patterns is not None else self.custom_file_patterns

        copied_files = []

        # First, try to copy from local directory if it exists
        if source_path.exists() and source_path.is_dir():
            for file_spec in patterns:
                for file_path in source_path.glob(file_spec):
                    if file_path.is_file():
                        relative_path = file_path.relative_to(source_path).as_posix()
                        target_file = target_path / relative_path
                        try:
                            target_file.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(file_path, target_file)
                            copied_files.append(relative_path)
                        except (OSError, IOError):
                            # Silently skip files that can't be copied
                            pass

        # If no files were copied and source_path looks like a HuggingFace Hub ID,
        # try to download the files directly from the Hub
        if not copied_files and "/" in str(source_path) and not source_path.exists():
            try:
                from huggingface_hub import hf_hub_download, list_repo_files

                # Get list of all files in the repository
                revision_kwargs = {"revision": revision} if revision is not None else {}
                repo_files = list_repo_files(str(source_path), **revision_kwargs)

                # Use fnmatch for both patterns and exact file names
                import fnmatch

                for file_spec in patterns:
                    matching_files = [f for f in repo_files if fnmatch.fnmatch(f, file_spec)]
                    for repo_file in matching_files:
                        try:
                            _ = hf_hub_download(
                                repo_id=str(source_path),
                                filename=repo_file,
                                local_dir=target_path,
                                local_dir_use_symlinks=False,
                                **revision_kwargs,
                            )
                            copied_files.append(repo_file)
                        except Exception:
                            # Silently skip files that can't be downloaded
                            pass
                self._cleanup_hf_local_dir_cache(target_path)

            except Exception as exc:
                # The Hub cannot be listed (for example with HF_HUB_OFFLINE=1), so use the files
                # that loading this revision already placed in the local Hugging Face cache.
                cached_files = self._copy_cached_hub_files(
                    str(source_path), target_path, patterns, revision=revision
                )
                copied_files.extend(
                    file_name for file_name in cached_files if file_name not in copied_files
                )
                logger.info(
                    "Could not list %s on the Hugging Face Hub (%s); copied %s from the local cache.",
                    source_path,
                    type(exc).__name__,
                    cached_files or "no files",
                )

        return copied_files

    def _load_artifact(self, name: str) -> Any:
        """Load artifact ``name`` with ``_load_<name>`` and record it, so saving can copy its source files."""
        artifact = getattr(self, f"_load_{name}")()
        self._loaded_artifacts[name] = (artifact, _artifact_fingerprint(artifact))
        return artifact

    def _copies_source_files(self, name: str, artifact: Any) -> bool:
        """Whether saving copies ``artifact``'s source files: it was loaded from the source and is unchanged."""
        loaded = self._loaded_artifacts.get(name)
        return (
            name in self.SOURCE_ARTIFACT_FILES
            and getattr(self, "model_name_or_path", None) is not None
            and set(self.init_kwargs) <= self.LOAD_ONLY_KWARGS
            and loaded is not None
            and loaded[0] is artifact
            and loaded[1] is not None
            and loaded[1] == _artifact_fingerprint(artifact)
        )

    def _copy_source_artifact_files(
        self,
        save_path: Path,
        artifacts: Dict[str, Any],
        mtimes: Dict[str, int],
        *,
        revision: str | None,
    ) -> None:
        """Copy the source files of unchanged ``artifacts`` instead of re-serializing them.

        ``save_pretrained`` can change these artifacts, e.g. drop ``vocab.json`` / ``merges.txt`` or switch a fast
        image processor to the slow one. An artifact the source has no files for is saved with ``save_pretrained``.
        Files written since ``mtimes`` by another artifact's ``save_pretrained`` are kept, so edits are not lost.
        """
        if not artifacts:
            return
        patterns = {}
        for name, artifact in artifacts.items():
            # A tokenizer's own vocabulary files, e.g. a SentencePiece model SOURCE_ARTIFACT_FILES lacks.
            vocab_files = getattr(artifact, "vocab_files_names", None)
            vocab_files = list(vocab_files.values()) if isinstance(vocab_files, dict) else []
            patterns[name] = [*self.SOURCE_ARTIFACT_FILES[name], *vocab_files]
        with tempfile.TemporaryDirectory() as source_files:
            copied = self._copy_custom_modeling_files(
                self.model_name_or_path,
                source_files,
                file_patterns=list(
                    dict.fromkeys(pattern for names in patterns.values() for pattern in names)
                ),
                revision=revision,
            )
            for name, artifact in artifacts.items():
                if not any(
                    fnmatch.fnmatch(file, pattern) for file in copied for pattern in patterns[name]
                ):
                    artifact.save_pretrained(save_path)
            written = {
                file for file, mtime in _file_mtimes(save_path).items() if mtimes.get(file) != mtime
            }
            for file in copied:
                if file not in written:
                    (save_path / file).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(Path(source_files) / file, save_path / file)

    def save_artifacts(
        self,
        save_directory: Union[str, Path],
        original_source_path: Optional[Union[str, Path]] = None,
        additional_files: Optional[List[str]] = None,
    ):
        """
        Saves all loaded, generic artifacts that have a `save_pretrained` method
        to the specified directory. Note: This does not save the `model` attribute.

        If the model was loaded with trust_remote_code=True, this method will also
        attempt to preserve any custom modeling files to ensure the saved model
        can be loaded properly.

        Args:
            save_directory: Directory to save artifacts to
            original_source_path: Optional path to the original source for custom modeling files
            additional_files: Optional list of additional file names or patterns to download/copy
                from the source. Supports both exact file names (e.g., "vocab.json") and glob
                patterns (e.g., "*.json"). These files will be copied from the source path.
        """
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        revision = self.init_kwargs.get("revision")
        revision = revision if isinstance(revision, str) else None

        _ = getattr(self, "config")  # trigger lazy loading of config
        if hasattr(self, "_config") and self._config is not None:
            if hasattr(self._config, "quantization_config"):
                # quantized export is not supported currently
                del self._config.quantization_config
            auto_map_missing = object()
            auto_map = vars(self._config).get("auto_map", auto_map_missing)
            strip_auto_map = (
                auto_map is not auto_map_missing
                and getattr(self, "trust_remote_code", False) is not True
            )
            if strip_auto_map:
                delattr(self._config, "auto_map")
            try:
                self._config.save_pretrained(save_path)
            finally:
                if strip_auto_map:
                    self._config.auto_map = auto_map

        # Unchanged artifacts loaded from the source get its files instead, copied after the others are saved.
        source_artifacts = {}
        mtimes = _file_mtimes(save_path)

        # Iterate over required artifacts to save them in a predictable order
        for name in self.ARTIFACTS:
            # Access the public property to trigger lazy loading if needed
            artifact = getattr(self, name)
            attr_name = f"_{name}"
            if hasattr(self, attr_name):
                if artifact is not None and hasattr(artifact, "save_pretrained"):
                    if self._copies_source_files(name, artifact):
                        source_artifacts[name] = artifact
                    else:
                        artifact.save_pretrained(save_path)

        # Iterate over optional artifacts - only save if they exist and have save_pretrained
        for name in self.OPTIONAL_ARTIFACTS:
            artifact = getattr(self, name, None)
            if artifact is not None and hasattr(artifact, "save_pretrained"):
                if self._copies_source_files(name, artifact):
                    source_artifacts[name] = artifact
                    continue
                try:
                    artifact.save_pretrained(save_path)
                except ValueError:
                    if name != "generation_config":
                        raise

                    source_path = original_source_path or getattr(self, "model_name_or_path", None)
                    if source_path is None:
                        raise

                    copied_files = self._copy_custom_modeling_files(
                        source_path=source_path,
                        target_path=save_path,
                        file_patterns=["generation_config.json"],
                        revision=revision,
                    )
                    if "generation_config.json" not in copied_files:
                        raise

                    logger.warning(
                        "GenerationConfig.save_pretrained() rejected the source artifact; "
                        "preserved the original generation_config.json instead."
                    )

        self._copy_source_artifact_files(save_path, source_artifacts, mtimes, revision=revision)

        # Download/copy additional files if specified
        if additional_files:
            self._copy_custom_modeling_files(
                source_path=original_source_path or self.model_name_or_path,
                target_path=save_path,
                file_patterns=additional_files,
                revision=revision,
            )

        # Preserve custom modeling files if trust_remote_code was used
        if hasattr(self, "trust_remote_code") and self.trust_remote_code:
            # Try original source path first, then fallback to model_name_or_path
            source_paths = []

            if original_source_path is not None:
                source_paths.append(original_source_path)
            else:
                # try to automatically determine original source path
                if getattr(self, "auto_map_model_class", None) is not None and hasattr(
                    self, "model_name_or_path"
                ):
                    import sys

                    from transformers.dynamic_module_utils import get_class_from_dynamic_module

                    try:
                        model_class = get_class_from_dynamic_module(
                            self.auto_map_model_class,
                            self.model_name_or_path,
                            trust_remote_code=self.trust_remote_code,
                        )
                        src_file = sys.modules[model_class.__module__].__file__
                        src_path = Path(src_file).parent
                        source_paths.append(src_path)
                    except Exception as e:
                        logger.debug(
                            "Could not load dynamic module for %s: %s", self.model_name_or_path, e
                        )

            if hasattr(self, "model_name_or_path") and self.model_name_or_path:
                source_paths.append(self.model_name_or_path)

            for source_path in source_paths:
                copied_files = self._copy_custom_modeling_files(
                    source_path, save_path, revision=revision
                )
                if copied_files:
                    # Successfully copied files, no need to try other paths
                    break

    @abstractmethod
    def _load_model(self) -> PreTrainedModel:
        """Subclasses must implement this to load the main model."""
        pass

    @abstractmethod
    def _load_config(self) -> AutoConfig:
        """Subclasses must implement this to load the model config."""
        pass

    @property
    def model(self) -> PreTrainedModel:
        """Lazily loads and returns the underlying model."""
        if not hasattr(self, "_model"):
            self._model = self._load_model()
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel):
        """Manually set the model."""
        self._model = value

    @property
    def config(self) -> AutoConfig:
        """Lazy load and return the model config."""
        if not hasattr(self, "_config"):
            self._config = self._load_config()
        return self._config

    @config.setter
    def config(self, value: AutoConfig):
        """Set the config manually."""
        self._config = value

    @property
    def state(self) -> StateDict:
        """
        Get the state dict accessor for pandas-like querying.

        This accessor can be backed by either a fully loaded model in memory
        or a ".safetensors" checkpoint on disk, enabling lazy loading of tensors.

        Examples:
            import re

            model.state()  # Get full state dict
            model.state["key"]  # Get single entry
            model.state[["key1", "key2"]]  # Get multiple entries
            model.state["*.weight"]  # Glob pattern
            model.state[re.compile(r".*\\.bias$")]  # Regex pattern
        """
        if self._state_dict_accessor is None:
            source: Optional[Union[Dict[str, torch.Tensor], StateSource]] = None
            # Prioritize the loaded model's state_dict if available
            if hasattr(self, "_model") and self._model is not None:
                source = self.model.state_dict()
            elif hasattr(self, "model_name_or_path") and self.model_name_or_path:
                revision = self.init_kwargs.get("revision")
                revision = revision if isinstance(revision, str) else None
                source = SafeTensorsStateSource(self.model_name_or_path, revision=revision)

            if source is None:
                raise ValueError(
                    "Cannot create StateDict accessor: model is not loaded and model_name_or_path is not set."
                )
            self._state_dict_accessor = StateDict(source)
        return self._state_dict_accessor
