# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import multiprocessing
import os
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call
from zipfile import ZipFile

import pytest
import requests

from tests.test_utils.python_scripts import download_unit_tests_dataset

CONCURRENT_RANKS = 8


def _archive_bytes(directory: str, content: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(f"{directory}/fixture.txt", content)
    return buffer.getvalue()


def _stage_release_assets(staged_root: Path) -> None:
    staged_dir = staged_root / download_unit_tests_dataset.STAGED_RELEASE_ASSET_DIR
    staged_dir.mkdir(parents=True)
    for asset in download_unit_tests_dataset.ASSETS:
        asset_path = staged_dir / asset["name"]
        asset_path.write_bytes(_archive_bytes(asset_path.stem, asset_path.name))


def _refuse_download(url: str, **_) -> None:
    raise AssertionError(f"GitHub fallback should not be used: {url}")


def _prepare_as_rank(assets_dir: Path, extraction_log_dir: Path, barrier) -> None:
    """Prepare ``assets_dir`` like one pytest rank, logging each extraction it performs."""
    extract_asset = download_unit_tests_dataset.extract_asset
    extraction_log = extraction_log_dir / f"{os.getpid()}.log"

    def logged_extract_asset(asset_path: Path, target_dir: Path) -> None:
        with extraction_log.open("a") as log:
            log.write(f"{asset_path.name}\n")
        extract_asset(asset_path, target_dir)

    download_unit_tests_dataset.extract_asset = logged_extract_asset
    download_unit_tests_dataset.requests.get = _refuse_download
    barrier.wait()
    download_unit_tests_dataset.download_and_extract_asset(assets_dir)


def test_download_and_extract_asset_prefers_staged_assets(monkeypatch, tmp_path):
    staged_root = tmp_path / "staged"
    _stage_release_assets(staged_root)

    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(staged_root))
    get = MagicMock(side_effect=AssertionError("GitHub fallback should not be used"))
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)

    output_dir = tmp_path / "output"
    download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert (output_dir / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_dir / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    get.assert_not_called()


def test_download_and_extract_asset_falls_back_without_github_token(monkeypatch, tmp_path):
    archives = {
        asset["url"]: _archive_bytes(Path(asset["name"]).stem, asset["name"])
        for asset in download_unit_tests_dataset.ASSETS
    }

    class Response:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size: int):
            yield self.content

    get = MagicMock(side_effect=lambda url, **_: Response(archives[url]))
    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(tmp_path / "missing"))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)

    output_dir = tmp_path / "output"
    download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert (output_dir / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_dir / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    assert get.call_args_list == [
        call(asset["url"], stream=True, timeout=60) for asset in download_unit_tests_dataset.ASSETS
    ]


def test_download_and_extract_asset_extracts_once_across_concurrent_ranks(monkeypatch, tmp_path):
    staged_root = tmp_path / "staged"
    _stage_release_assets(staged_root)
    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(staged_root))
    extraction_log_dir = tmp_path / "extractions"
    extraction_log_dir.mkdir()
    output_dir = tmp_path / "output"

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(CONCURRENT_RANKS)
    ranks = [
        context.Process(target=_prepare_as_rank, args=(output_dir, extraction_log_dir, barrier))
        for _ in range(CONCURRENT_RANKS)
    ]
    for rank in ranks:
        rank.start()
    for rank in ranks:
        rank.join(timeout=60)

    assert [rank.exitcode for rank in ranks] == [0] * CONCURRENT_RANKS
    extractions = [log.read_text().split() for log in extraction_log_dir.iterdir()]
    assert extractions == [[asset["name"] for asset in download_unit_tests_dataset.ASSETS]]
    assert (output_dir / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_dir / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        ".output.lock",
        "extractions",
        "output",
        "staged",
    ]


def test_download_and_extract_asset_raises_and_leaves_no_partial_data(monkeypatch, tmp_path):
    get = MagicMock(side_effect=requests.ConnectionError("github.com unreachable"))
    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(tmp_path / "missing"))
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)

    output_dir = tmp_path / "output"
    with pytest.raises(requests.ConnectionError, match="github.com unreachable"):
        download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == [".output.lock"]


def test_download_and_extract_asset_skips_populated_dir_without_locking(monkeypatch, tmp_path):
    get = MagicMock(side_effect=AssertionError("GitHub fallback should not be used"))
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "baked.txt").write_text("baked")

    download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output"]
    assert [path.name for path in output_dir.iterdir()] == ["baked.txt"]
    get.assert_not_called()
