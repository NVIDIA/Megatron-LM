#!/usr/bin/env python3
"""One-off script to add NVIDIA copyright header to all .md files under docs/."""

from pathlib import Path

HEADER = """   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

"""

def main():
    docs_dir = Path(__file__).resolve().parent
    already_has = "Copyright (c) 2022-2026, NVIDIA CORPORATION"
    count = 0
    for path in sorted(docs_dir.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if content.strip().startswith(already_has):
            continue
        new_content = HEADER + content
        path.write_text(new_content, encoding="utf-8")
        count += 1
        print(path.relative_to(docs_dir))
    print(f"\nUpdated {count} files.")

if __name__ == "__main__":
    main()
