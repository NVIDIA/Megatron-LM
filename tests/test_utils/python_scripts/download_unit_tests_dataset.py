# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#!/usr/bin/env python3
"""
Populate unit-test data from staged or public NVIDIA/Megatron-LM v2.5 release assets.
"""

import fcntl
import logging
import os
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import click
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TEST_DATA_ROOT = Path("/home/TestData")
TEST_DATA_ROOT_ENV = "NEMO_TEST_DATA_ROOT"
STAGED_RELEASE_ASSET_DIR = Path("megatron-lm/release-assets/v2.5")
ASSETS = [
    {
        "name": "datasets.zip",
        "url": "https://github.com/NVIDIA/Megatron-LM/releases/download/v2.5/datasets.zip",
    },
    {
        "name": "tokenizers.zip",
        "url": "https://github.com/NVIDIA/Megatron-LM/releases/download/v2.5/tokenizers.zip",
    },
]


def get_test_data_root() -> Path:
    """Return the configured shared TestData root."""
    return Path(os.environ.get(TEST_DATA_ROOT_ENV) or DEFAULT_TEST_DATA_ROOT)


def extract_asset(asset_path: Path, assets_dir: Path) -> None:
    """Extract a release asset into the writable test data directory.

    Args:
        asset_path: Release archive to extract.
        assets_dir: Directory to extract the asset into.

    Raises:
        ValueError: If the archive type is not supported.
    """
    try:
        logger.info(f"  Extracting {asset_path.name} to {assets_dir}...")

        if asset_path.name.endswith('.zip'):
            with zipfile.ZipFile(asset_path, 'r') as zip_ref:
                zip_ref.extractall(assets_dir)
        elif asset_path.name.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(asset_path, 'r:gz') as tar_ref:
                tar_ref.extractall(assets_dir)
        elif asset_path.name.endswith('.tar'):
            with tarfile.open(asset_path, 'r') as tar_ref:
                tar_ref.extractall(assets_dir)
        else:
            raise ValueError(f"Unknown archive type: {asset_path.name}")

        logger.info(f"  Successfully extracted to {assets_dir}")
    except Exception as e:
        logger.error(f"  Error extracting {asset_path.name}: {e}")
        raise


def extract_staged_release_assets(assets_dir: Path) -> bool:
    """Extract staged Megatron-LM v2.5 assets when all of them are available and readable.

    Args:
        assets_dir: Directory to extract the assets into.

    Returns:
        True when every staged asset was extracted; False to fall back to the public download.
    """
    staged_dir = get_test_data_root() / STAGED_RELEASE_ASSET_DIR
    staged_assets = tuple(staged_dir / asset["name"] for asset in ASSETS)
    if not all(asset_path.is_file() for asset_path in staged_assets):
        return False

    logger.info(f"Using staged release assets from {staged_dir}")
    try:
        for asset_path in staged_assets:
            extract_asset(asset_path, assets_dir)
    except Exception:
        logger.warning(f"Staged release assets in {staged_dir} are unusable; downloading instead")
        return False
    return True


def download_release_asset(asset_url: str, asset_name: str, assets_dir: Path) -> None:
    """Download and extract one public GitHub release asset.

    Args:
        asset_url: Public URL of the release asset.
        asset_name: File name of the release asset.
        assets_dir: Directory to extract the asset into.
    """
    temp_file = assets_dir / asset_name
    try:
        logger.info(f"  Downloading {asset_name}...")
        response = requests.get(asset_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        extract_asset(temp_file, assets_dir)
    except Exception as e:
        logger.error(f"  Error downloading/extracting {asset_name}: {e}")
        raise
    finally:
        if temp_file.is_file():
            temp_file.unlink()


def is_populated(assets_dir: Path) -> bool:
    """Return whether ``assets_dir`` is a directory with at least one entry."""
    return assets_dir.is_dir() and any(assets_dir.iterdir())


@contextmanager
def exclusive_lock(lock_path: Path) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``lock_path``, blocking until it is available.

    Args:
        lock_path: Lock file; created if missing and left in place for later callers.
    """
    with open(lock_path, 'a') as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        yield


def download_and_extract_asset(assets_dir: Path) -> None:
    """Populate ``assets_dir`` once: staged v2.5 assets first, then public GitHub downloads.

    Safe to call from every rank of a node at once, before ``torch.distributed`` exists.
    Callers serialize on a lock file next to ``assets_dir``; the first one extracts into a
    sibling staging directory and renames it into place, so ``assets_dir`` is either absent,
    empty, or complete. The others then find it populated and return. A populated
    ``assets_dir`` is detected without the lock, so read-only parents keep working.

    Args:
        assets_dir: Directory to populate; must be absent or empty to be populated.

    Raises:
        Exception: The download or extraction error when no source could populate the data.
    """
    assets_dir = assets_dir.absolute()
    if is_populated(assets_dir):
        logger.info(f"Test data already available at {assets_dir}")
        return

    assets_dir.parent.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(assets_dir.with_name(f".{assets_dir.name}.lock")):
        if is_populated(assets_dir):
            logger.info(f"Test data prepared at {assets_dir} by another process")
            return

        staging_dir = assets_dir.with_name(f".{assets_dir.name}.staging")
        shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir()
        try:
            if not extract_staged_release_assets(staging_dir):
                for asset in ASSETS:
                    download_release_asset(asset["url"], asset["name"], staging_dir)
            os.replace(staging_dir, assets_dir)
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)


@click.command()
@click.option(
    '--repo', default='NVIDIA/Megatron-LM', help='GitHub repository name (format: owner/repo)'
)
@click.option('--assets-dir', default='assets', help='Directory to extract assets to')
def main(repo, assets_dir):
    """Populate unit-test data from staged or public release assets."""
    logger.info(f"Preparing v2.5 release assets for {repo}...")
    logger.info("=" * 80)

    try:
        download_and_extract_asset(Path(assets_dir))
    except Exception as e:
        raise click.ClickException(f"Failed to download and extract release assets: {e!r}") from e


if __name__ == "__main__":
    main()
