# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#!/usr/bin/env python3
"""
Populate unit-test data from staged or public NVIDIA/Megatron-LM v2.5 release assets.
"""

import logging
import os
import tarfile
import zipfile
from pathlib import Path

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


def extract_asset(asset_path: Path, assets_dir: Path) -> bool:
    """Extract a release asset into the writable test data directory.

    Args:
        asset_path: Release archive to extract.
        assets_dir: Directory to extract the asset into.

    Returns:
        True when extraction succeeds.
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
            logger.warning(
                f"  Warning: Unknown file type for {asset_path.name}, skipping extraction"
            )
            return False

        logger.info(f"  Successfully extracted to {assets_dir}")
        return True
    except Exception as e:
        logger.error(f"  Error extracting {asset_path.name}: {e}")
        return False


def extract_staged_release_assets(assets_dir: Path) -> bool:
    """Extract staged Megatron-LM v2.5 assets when all of them are available."""
    staged_dir = get_test_data_root() / STAGED_RELEASE_ASSET_DIR
    staged_assets = tuple(staged_dir / asset["name"] for asset in ASSETS)
    if not all(asset_path.is_file() for asset_path in staged_assets):
        return False

    logger.info(f"Using staged release assets from {staged_dir}")
    return all(extract_asset(asset_path, assets_dir) for asset_path in staged_assets)


def download_release_asset(asset_url: str, asset_name: str, assets_dir: Path) -> bool:
    """Download and extract one public GitHub release asset."""
    temp_file = assets_dir / asset_name
    try:
        logger.info(f"  Downloading {asset_name}...")
        response = requests.get(asset_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return extract_asset(temp_file, assets_dir)
    except Exception as e:
        logger.error(f"  Error downloading/extracting {asset_name}: {e}")
        return False
    finally:
        if temp_file.is_file():
            temp_file.unlink()


def download_and_extract_asset(assets_dir: Path) -> bool:
    """Use staged v2.5 assets first, then fall back to public GitHub downloads."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    if extract_staged_release_assets(assets_dir):
        return True

    return all(download_release_asset(asset["url"], asset["name"], assets_dir) for asset in ASSETS)


@click.command()
@click.option(
    '--repo', default='NVIDIA/Megatron-LM', help='GitHub repository name (format: owner/repo)'
)
@click.option('--assets-dir', default='assets', help='Directory to extract assets to')
def main(repo, assets_dir):
    """Populate unit-test data from staged or public release assets."""
    logger.info(f"Preparing v2.5 release assets for {repo}...")
    logger.info("=" * 80)

    if not download_and_extract_asset(Path(assets_dir)):
        raise click.ClickException("Failed to download and extract release assets")


if __name__ == "__main__":
    main()
