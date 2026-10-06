# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call
from zipfile import ZipFile

from tests.test_utils.python_scripts import download_unit_tests_dataset


def _archive_bytes(directory: str, content: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(f"{directory}/fixture.txt", content)
    return buffer.getvalue()


def test_download_and_extract_asset_prefers_staged_assets(monkeypatch, tmp_path):
    staged_root = tmp_path / "staged"
    staged_dir = staged_root / download_unit_tests_dataset.STAGED_RELEASE_ASSET_DIR
    staged_dir.mkdir(parents=True)
    for asset in download_unit_tests_dataset.ASSETS:
        asset_path = staged_dir / asset["name"]
        asset_path.write_bytes(_archive_bytes(asset_path.stem, asset_path.name))

    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(staged_root))
    get = MagicMock(side_effect=AssertionError("GitHub fallback should not be used"))
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)

    output_dir = tmp_path / "output"
    assert download_unit_tests_dataset.download_and_extract_asset(output_dir)
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
    assert download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert (output_dir / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_dir / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    assert get.call_args_list == [
        call(asset["url"], stream=True, timeout=60) for asset in download_unit_tests_dataset.ASSETS
    ]
