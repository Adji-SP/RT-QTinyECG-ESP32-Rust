# -*- coding: utf-8 -*-
"""
archive_project.py
==================
Automatically archives all project images and data files into a
timestamped ZIP bundle under Product/images/archive/<date>/.

Structure of archive:
  archive/
  └── YYYY-MM-DD/
      └── archive_YYYY-MM-DD_HH-MM-SS.zip
          ├── images/
          │   ├── evaluation/
          │   ├── model_esp32/
          │   └── model_pc/
          ├── data/
          │   ├── plots/
          │   └── *.csv / *.npy / *.npz / *.pkl / *.json
          └── manifest.json

Usage:
  python archive_project.py              # archive everything
  python archive_project.py --dry-run   # preview what will be archived
  python archive_project.py --list      # list existing archives
"""

import os
import sys
import json
import io
import zipfile
import hashlib
import argparse
from datetime import datetime
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

# Root of the Product directory (relative to this script)
PRODUCT_ROOT = Path(__file__).parent.resolve()

# Folders whose contents will be included in the archive
ARCHIVE_SOURCES = {
    "images/evaluation":  ["*.png", "*.jpg", "*.jpeg", "*.svg"],
    "images/model_esp32": ["*.png", "*.jpg", "*.jpeg", "*.svg"],
    "images/model_pc":    ["*.png", "*.jpg", "*.jpeg", "*.svg"],
    "data":               ["*.csv", "*.npy", "*.npz", "*.pkl", "*.json", "*.txt"],
    "data/plots":         ["*.csv"],
}

# Where archives are stored
ARCHIVE_BASE = PRODUCT_ROOT / "images" / "archive"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def sha256_of(path: Path) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files() -> list[dict]:
    """Gather all files that match the archive sources."""
    entries = []
    for rel_folder, patterns in ARCHIVE_SOURCES.items():
        src_dir = PRODUCT_ROOT / rel_folder
        if not src_dir.exists():
            print(f"  [skip] folder not found: {rel_folder}")
            continue
        for pattern in patterns:
            for file in sorted(src_dir.glob(pattern)):
                if file.is_file():
                    entries.append({
                        "abs_path":    file,
                        "archive_path": f"{rel_folder}/{file.name}",
                    })
    return entries


def build_manifest(entries: list[dict], archive_name: str) -> dict:
    """Build a JSON manifest describing the archive."""
    created_at = datetime.now().isoformat(timespec="seconds")
    files = []
    for e in entries:
        p = e["abs_path"]
        files.append({
            "path":   e["archive_path"],
            "size":   p.stat().st_size,
            "sha256": sha256_of(p),
        })
    return {
        "archive":    archive_name,
        "created_at": created_at,
        "file_count": len(files),
        "files":      files,
    }


def _sep(char="-", width=60) -> str:
    return f"  {char * width}"


def list_archives():
    """Print all existing archives."""
    if not ARCHIVE_BASE.exists():
        print("No archives found (archive directory does not exist yet).")
        return

    zips = sorted(ARCHIVE_BASE.rglob("*.zip"))
    if not zips:
        print("No ZIP archives found under:", ARCHIVE_BASE)
        return

    print(f"\n{_sep()}")
    print(f"  Existing archives in: {ARCHIVE_BASE}")
    print(_sep())
    total_bytes = 0
    for z in zips:
        size_kb = z.stat().st_size / 1024
        total_bytes += z.stat().st_size
        print(f"  {z.relative_to(ARCHIVE_BASE)}  ({size_kb:.1f} KB)")
    print(_sep())
    print(f"  Total: {len(zips)} archive(s), {total_bytes/1024:.1f} KB\n")


def dry_run():
    """Preview files that would be archived."""
    entries = collect_files()
    if not entries:
        print("No files found to archive.")
        return
    print(f"\n{_sep()}")
    print(f"  Dry-run: {len(entries)} file(s) would be archived")
    print(_sep())
    total = 0
    for e in entries:
        size = e["abs_path"].stat().st_size
        total += size
        print(f"  {e['archive_path']}  ({size/1024:.1f} KB)")
    print(_sep())
    print(f"  Total: {total/1024:.1f} KB\n")


def create_archive():
    """Create a timestamped ZIP archive of images and data."""
    now = datetime.now()
    date_str      = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Resolve destination
    dest_dir = ARCHIVE_BASE / date_str
    dest_dir.mkdir(parents=True, exist_ok=True)

    archive_name = f"archive_{timestamp_str}.zip"
    archive_path = dest_dir / archive_name

    # Collect files
    entries = collect_files()
    if not entries:
        print("No files matched for archiving. Nothing to do.")
        return

    # Build manifest (pre-zip, so we can include it inside)
    manifest = build_manifest(entries, archive_name)

    print(f"\n{_sep()}")
    print(f"  Creating archive: {archive_path.relative_to(PRODUCT_ROOT)}")
    print(f"  Files to archive: {len(entries)}")
    print(_sep())

    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for e in entries:
            zf.write(e["abs_path"], e["archive_path"])
            size_kb = e["abs_path"].stat().st_size / 1024
            print(f"  + {e['archive_path']}  ({size_kb:.1f} KB)")

        # Write the manifest last
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        print("  + manifest.json")

    zip_size_kb = archive_path.stat().st_size / 1024
    print(_sep())
    print(f"  [OK] Archive saved: {archive_path}")
    print(f"       Size: {zip_size_kb:.1f} KB  |  Files: {len(entries)}")
    print(_sep() + "\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Archive ECG project images and data files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview files that would be archived without creating a ZIP."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all existing archives."
    )
    args = parser.parse_args()

    # Force UTF-8 output on Windows
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("\n  QTinyECG-ESP32 - Project Archiver")
    print(f"  Root: {PRODUCT_ROOT}\n")

    if args.list:
        list_archives()
    elif args.dry_run:
        dry_run()
    else:
        create_archive()


if __name__ == "__main__":
    main()
