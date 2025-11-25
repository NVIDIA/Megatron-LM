#!/usr/bin/env python3
"""
Script to fetch the oldest release of NVIDIA/Megatron-LM on GitHub and list its assets.
Uses the PyGithub SDK to interact with the GitHub API.
"""

import os
import sys
import tarfile
import zipfile
from pathlib import Path

import click
import requests
from github import Github


def download_and_extract_asset(asset_url: str, asset_name: str, assets_dir: Path) -> bool:
    """
    Download and extract an asset to the assets directory.

    Args:
        asset_url: URL to download the asset from
        asset_name: Name of the asset file
        assets_dir: Directory to extract the asset to

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Download the asset
        print(f"  Downloading {asset_name}...")
        response = requests.get(asset_url, stream=True)
        response.raise_for_status()

        # Save to temporary file
        temp_file = assets_dir / asset_name
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Extracting {asset_name} to {assets_dir}...")

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
            print(f"  Warning: Unknown file type for {asset_name}, skipping extraction")
            return False

        # Clean up temporary file
        temp_file.unlink()
        print(f"  Successfully extracted to {assets_dir}")
        return True

    except Exception as e:
        print(f"  Error downloading/extracting {asset_name}: {e}")
        return False


def get_oldest_release_and_assets(
    repo_name: str = "NVIDIA/Megatron-LM", assets_dir: str = "assets"
) -> None:
    """
    Fetch the oldest release of a GitHub repository and list its assets.

    Args:
        repo_name: The repository name in format "owner/repo"
        assets_dir: Directory to extract assets to
    """
    try:
        # Initialize GitHub client
        g = Github(login_or_token=os.getenv('GH_TOKEN', None))

        # Get the repository
        repo = g.get_repo(repo_name)
        print(f"Repository: {repo.full_name}")
        print(f"Description: {repo.description}")
        print(f"URL: {repo.html_url}")
        print("-" * 80)

        # Get all releases
        releases = list(repo.get_releases())

        if not releases:
            print("No releases found for this repository.")
            return

        # Sort releases by creation date to find the oldest
        releases.sort(key=lambda x: x.created_at)
        oldest_release = releases[0]

        print(f"Oldest Release:")
        print(f"  Tag: {oldest_release.tag_name}")
        print(f"  Title: {oldest_release.title}")
        print(f"  Created: {oldest_release.created_at}")
        print(f"  Published: {oldest_release.published_at}")
        print(f"  Draft: {oldest_release.draft}")
        print(f"  Prerelease: {oldest_release.prerelease}")
        print(f"  URL: {oldest_release.html_url}")

        if oldest_release.body:
            print(f"  Description: {oldest_release.body[:200]}...")

        print("-" * 80)

        # List assets
        assets = list(oldest_release.get_assets())

        if not assets:
            print("No assets found for this release.")
            return

        print(f"Assets ({len(assets)} total):")
        print("-" * 80)

        for i, asset in enumerate(assets, 1):
            print(f"{i}. {asset.name}")
            print(f"   Size: {asset.size} bytes ({asset.size / 1024 / 1024:.2f} MB)")
            print(f"   Downloads: {asset.download_count}")
            print(f"   Content Type: {asset.content_type}")
            print(f"   URL: {asset.browser_download_url}")
            print(f"   Created: {asset.created_at}")
            print(f"   Updated: {asset.updated_at}")
            print()

        # Summary
        total_size = sum(asset.size for asset in assets)
        total_downloads = sum(asset.download_count for asset in assets)

        print(f"Summary:")
        print(f"  Total assets: {len(assets)}")
        print(f"  Total size: {total_size} bytes ({total_size / 1024 / 1024:.2f} MB)")
        print(f"  Total downloads: {total_downloads}")

        # Download and extract assets if requested
        if assets:
            print("-" * 80)
            print("Downloading and extracting assets...")

            # Create assets directory
            assets_path = Path(assets_dir)
            assets_path.mkdir(parents=True, exist_ok=True)
            print(f"Created assets directory: {assets_path.absolute()}")

            successful_downloads = 0
            for asset in assets:
                print(f"\nProcessing asset: {asset.name}")
                if download_and_extract_asset(asset.browser_download_url, asset.name, assets_path):
                    successful_downloads += 1

            print(f"\nDownload Summary:")
            print(
                f"  Successfully downloaded and extracted: {successful_downloads}/{len(assets)} assets"
            )
            print(f"  Assets directory: {assets_path.absolute()}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


@click.command()
@click.option(
    '--repo', default='NVIDIA/Megatron-LM', help='GitHub repository name (format: owner/repo)'
)
@click.option('--assets-dir', default='assets', help='Directory to extract assets to')
def main(repo, assets_dir):
    """Fetch the oldest release of a GitHub repository and download its assets."""
    print(f"Fetching oldest release of {repo}...")
    print("=" * 80)

    get_oldest_release_and_assets(repo_name=repo, assets_dir=assets_dir)


if __name__ == "__main__":
    main()
