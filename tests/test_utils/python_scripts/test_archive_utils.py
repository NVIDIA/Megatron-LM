# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import tarfile
import zipfile

import pytest

from tests.test_utils.python_scripts.archive_utils import safe_extract_tar, safe_extract_zip


def test_safe_extract_zip_extracts_valid_members(tmp_path):
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("nested/fixture.txt", "fixture")
    archive_bytes.seek(0)

    with zipfile.ZipFile(archive_bytes) as archive:
        safe_extract_zip(archive, tmp_path)

    assert (tmp_path / "nested" / "fixture.txt").read_text() == "fixture"


@pytest.mark.parametrize("member_name", ["../escaped.txt", "/escaped.txt"])
def test_safe_extract_zip_rejects_members_outside_destination(tmp_path, member_name):
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("valid.txt", "valid")
        archive.writestr(member_name, "escaped")
    archive_bytes.seek(0)

    with (
        zipfile.ZipFile(archive_bytes) as archive,
        pytest.raises(ValueError, match="escapes destination"),
    ):
        safe_extract_zip(archive, tmp_path)

    assert not (tmp_path / "valid.txt").exists()


def _tar_bytes(*members: tuple[tarfile.TarInfo, bytes]) -> io.BytesIO:
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode="w") as archive:
        for member, content in members:
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    archive_bytes.seek(0)
    return archive_bytes


def test_safe_extract_tar_extracts_valid_members(tmp_path):
    archive_bytes = _tar_bytes((tarfile.TarInfo("nested/fixture.txt"), b"fixture"))

    with tarfile.open(fileobj=archive_bytes, mode="r") as archive:
        safe_extract_tar(archive, tmp_path)

    assert (tmp_path / "nested" / "fixture.txt").read_text() == "fixture"


@pytest.mark.parametrize("member_name", ["../escaped.txt", "/escaped.txt"])
def test_safe_extract_tar_rejects_members_outside_destination(tmp_path, member_name):
    archive_bytes = _tar_bytes(
        (tarfile.TarInfo("valid.txt"), b"valid"), (tarfile.TarInfo(member_name), b"escaped")
    )

    with (
        tarfile.open(fileobj=archive_bytes, mode="r") as archive,
        pytest.raises(ValueError, match="escapes destination"),
    ):
        safe_extract_tar(archive, tmp_path)

    assert not (tmp_path / "valid.txt").exists()


def test_safe_extract_tar_rejects_links(tmp_path):
    link = tarfile.TarInfo("nested/link")
    link.type = tarfile.SYMTYPE
    link.linkname = "../outside"
    archive_bytes = _tar_bytes((link, b""))

    with (
        tarfile.open(fileobj=archive_bytes, mode="r") as archive,
        pytest.raises(ValueError, match="links are not allowed"),
    ):
        safe_extract_tar(archive, tmp_path)
