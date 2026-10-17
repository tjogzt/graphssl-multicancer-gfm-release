#!/usr/bin/env python3
"""
@file    10_download_string_ppi.py
@brief   Download and decompress STRING v12 human PPI files for downstream graph construction.
@author  Qingqing Mo, Pingbo Chen, Cheng Xu, Ya Wang, Ting Hu, Qian Sun, Tao Zhu
@date    2026-08 (normalized 2026-09)
@ingroup data_download

Download and decompress the STRING v12 human protein-protein interaction (PPI)
files (links, protein info, aliases, sequences). Downstream graph builders use
these to map STRING protein ids to gene symbols and to filter interactions by
combined score, so this script is the first acquisition step of the pipeline.

Input:  STRING database files fetched over HTTPS; URLs are derived from the
        STRING_VERSION and SPECIES module constants, and files land under
        config.STRING_DIR (overridable via GFM_STRING_DIR / GFM_PROJECT_ROOT).
Output: decompressed .txt files under config.STRING_DIR plus stdout progress logs.
Run:    python scripts/10_download_string_ppi.py
"""
import os
import urllib.request
import gzip
import shutil
import sys

# Add the scripts directory to the import path so config.py is discoverable
# regardless of the current working directory when this script is invoked.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# =========================================================================
# Global constants (concentrated at top for easy maintenance)
# =========================================================================
DATA_DIR = config.STRING_DIR
# Ensure the destination directory exists before attempting any downloads.
os.makedirs(DATA_DIR, exist_ok=True)

# STRING release and organism identifiers used across the whole pipeline.
STRING_VERSION = "12.0"
SPECIES = 9606  # NCBI taxonomy id for Homo sapiens

# Minimum file size (bytes) to consider an extracted file valid (guards against
# truncated or empty downloads that would silently propagate downstream).
MIN_VALID_FILE_SIZE_BYTES = 1024

# Download manifest: local basename -> remote URL. Basenames embed the version
# so a release bump writes fresh files instead of silently overwriting old ones.
DOWNLOAD_MANIFEST = {
    "protein.links.full.v12.0":
        f"https://stringdb-downloads.org/download/protein.links.full.v{STRING_VERSION}/{SPECIES}.protein.links.full.v{STRING_VERSION}.txt.gz",
    "protein.info.v12.0":
        f"https://stringdb-downloads.org/download/protein.info.v{STRING_VERSION}/{SPECIES}.protein.info.v{STRING_VERSION}.txt.gz",
    "protein.aliases.v12.0":
        f"https://stringdb-downloads.org/download/protein.aliases.v{STRING_VERSION}/{SPECIES}.protein.aliases.v{STRING_VERSION}.txt.gz",
    "protein.sequences.v12.0":
        f"https://stringdb-downloads.org/download/protein.sequences.v{STRING_VERSION}/{SPECIES}.protein.sequences.v{STRING_VERSION}.fa.gz",
}


def download_and_extract(name, url, data_dir):
    """
    @brief   Download a gzipped STRING file and extract it to the destination directory.

    @param   name      Local base name for the file (without extension).
    @param   url       Full HTTPS URL of the gzipped file on the STRING server.
    @param   data_dir  Destination directory where files will be written.
    @return  None. Writes files directly to disk; prints progress to stdout.
    @note    Already-extracted files larger than MIN_VALID_FILE_SIZE_BYTES are
             skipped to support idempotent / resume-able pipeline runs.
    """
    # Compose full filesystem paths for the compressed and extracted artifacts.
    if name.endswith(".fa"):
        # Sequence FASTA: the gzipped file itself is kept (not decompressed)
        # because downstream parsers consume the .fa.gz directly.
        gz_path = os.path.join(data_dir, f"{name}.gz")
        txt_path = os.path.join(data_dir, f"{name}.gz")
    else:
        # Tabular PPI files: downloaded as .gz and then decompressed to .txt.
        gz_path = os.path.join(data_dir, f"{name}.txt.gz")
        txt_path = os.path.join(data_dir, f"{name}.txt")

    # Skip files already extracted; a size floor guards against truncated downloads.
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > MIN_VALID_FILE_SIZE_BYTES:
        print(f"  [SKIP] {name} already downloaded ({os.path.getsize(txt_path):,} bytes)")
        return

    # Download the compressed file from the STRING server to local disk.
    print(f"  Downloading {name} ...")
    urllib.request.urlretrieve(url, gz_path)
    filesize = os.path.getsize(gz_path)
    print(f"    Downloaded size: {filesize:,} bytes")

    if name.endswith(".fa"):
        # Sequence files: the gzipped FASTA is the final artifact; rename the
        # download so the extension matches downstream expectations.
        final_fa_gz = txt_path
        shutil.move(gz_path, final_fa_gz)
    else:
        # Tabular files: decompress the gzip payload in place.
        with gzip.open(gz_path, 'rb') as fin, open(txt_path, 'wb') as fout:
            # Stream-copy the decompressed content chunk by chunk to avoid
            # loading the entire file into memory at once.
            shutil.copyfileobj(fin, fout)
        # Remove the temporary gzip archive after successful extraction.
        os.remove(gz_path)

    print(f"    Final extracted size: {os.path.getsize(txt_path):,} bytes")


def main():
    """
    @brief   Main entry point: iterate the download manifest and process each file.

    @return  None. Writes output to DATA_DIR and prints progress/summary.
    """
    print("Phase 1.1: STRING PPI Download")
    print(f"  Version: {STRING_VERSION} | Species: {SPECIES} | Output: {DATA_DIR}")
    print(f"  Files to acquire: {len(DOWNLOAD_MANIFEST)}")

    # Iterate the manifest deterministically (Python 3.7+ dict insertion order).
    for name, url in DOWNLOAD_MANIFEST.items():
        download_and_extract(name, url, DATA_DIR)

    print("Phase 1.1 STRING download complete.")


if __name__ == "__main__":
    main()
