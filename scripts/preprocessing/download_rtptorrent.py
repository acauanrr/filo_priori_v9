#!/usr/bin/env python3
"""
Download RTPTorrent Dataset from Zenodo.

This script downloads and extracts the RTPTorrent dataset for use with Filo-Priori.

Source: https://zenodo.org/records/3712290
Paper: Mattis et al., "RTPTorrent: An Open-source Dataset for Evaluating
       Regression Test Prioritization", MSR 2020

Usage:
    python scripts/preprocessing/download_rtptorrent.py
"""

import os
import sys
import urllib.request
import zipfile
import hashlib
from pathlib import Path
from tqdm import tqdm

# Configuration
ZENODO_URL = "https://zenodo.org/records/3712290/files/rtp-torrent-v1.zip"
EXPECTED_MD5 = None  # Set if known
OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets" / "02_rtptorrent" / "raw"
ZIP_FILENAME = "rtp-torrent-v1.zip"


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path) -> bool:
    """Download a file with progress bar."""
    print(f"\nDownloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def verify_md5(file_path: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of downloaded file."""
    if expected_md5 is None:
        print("MD5 checksum not available, skipping verification")
        return True

    print("Verifying MD5 checksum...")
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)

    actual_md5 = md5_hash.hexdigest()
    if actual_md5 == expected_md5:
        print(f"MD5 verified: {actual_md5}")
        return True
    else:
        print(f"MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """Extract zip file."""
    print(f"\nExtracting {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(f.file_size for f in zip_ref.infolist())

            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, output_dir)
                    pbar.update(file.file_size)

        print(f"Extracted to: {output_dir}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def main():
    """Main download and extraction routine."""
    print("=" * 60)
    print("RTPTorrent Dataset Downloader")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / ZIP_FILENAME

    # Check if already downloaded
    if zip_path.exists():
        print(f"\nZip file already exists: {zip_path}")
        response = input("Re-download? [y/N]: ").strip().lower()
        if response != 'y':
            print("Skipping download.")
        else:
            os.remove(zip_path)
            if not download_file(ZENODO_URL, zip_path):
                sys.exit(1)
    else:
        if not download_file(ZENODO_URL, zip_path):
            sys.exit(1)

    # Verify checksum
    if not verify_md5(zip_path, EXPECTED_MD5):
        print("Warning: Could not verify file integrity")

    # Extract
    if not extract_zip(zip_path, OUTPUT_DIR):
        sys.exit(1)

    # List extracted contents
    print("\nExtracted contents:")
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir():
            file_count = len(list(item.rglob('*')))
            print(f"  {item.name}/ ({file_count} files)")
        else:
            size_mb = item.stat().st_size / (1024 * 1024)
            print(f"  {item.name} ({size_mb:.1f} MB)")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    print("\nNext step: Run the preprocessing script:")
    print("  python scripts/preprocessing/preprocess_rtptorrent.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
