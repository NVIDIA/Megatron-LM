# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

#!/usr/bin/env python3
"""
Script to fetch the oldest release of NVIDIA/Megatron-LM on GitHub and list its assets.
Uses the PyGithub SDK to interact with the GitHub API.
"""

import logging
import tarfile
import zipfile
from pathlib import Path

import click
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def download_and_extract_asset(assets_dir: Path) -> bool:
    """
    Download and extract an asset to the assets directory.

    Args:
        asset_url: URL to download the asset from
        asset_name: Name of the asset file
        assets_dir: Directory to extract the asset to

    Returns:
        bool: True if successful, False otherwise
    """
    for asset in ASSETS:
        asset_name, asset_url = asset.values()
        try:
            # Download the asset
            logger.info(f"  Downloading {asset_name}...")
            response = requests.get(asset_url, stream=True)
            response.raise_for_status()

            # Save to temporary file
            temp_file = assets_dir / asset_name
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"  Extracting {asset_name} to {assets_dir}...")

            # Extract based on file type
            if asset_name.endswith('.zip'):
                with zipfile.ZipFile(temp_file, 'r') as zip_ref:
                    zip_ref.extractall(assets_dir)
            elif asset_name.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(temp_file, 'r:gz') as tar_ref:
                    tar_ref.extractall(assets_dir)
            elif asset_name.endswith('.tar'):
                with tarfile.open(temp_file, 'r') as tar_ref:
                    tar_ref.extractall(assets_dir)
            else:
                logger.warning(
                    f"  Warning: Unknown file type for {asset_name}, skipping extraction"
                )

            # Clean up temporary file
            temp_file.unlink()
            logger.info(f"  Successfully extracted to {assets_dir}")

        except Exception as e:
            logger.error(f"  Error downloading/extracting {asset_name}: {e}")


@click.command()
@click.option(
    '--repo', default='NVIDIA/Megatron-LM', help='GitHub repository name (format: owner/repo)'
)
@click.option('--assets-dir', default='assets', help='Directory to extract assets to')
def main(repo, assets_dir):
    """Fetch the oldest release of a GitHub repository and download its assets."""
    logger.info(f"Fetching oldest release of {repo}...")
    logger.info("=" * 80)

    Path(assets_dir).mkdir(parents=True, exist_ok=True)

    download_and_extract_asset(Path(assets_dir))


if __name__ == "__main__":
    main()
