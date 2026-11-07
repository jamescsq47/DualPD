#!/usr/bin/env python3
"""Write a source-ordered SWE subset whose verifier SIFs already exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def sif_name(instance_id: str) -> str:
    return instance_id.lower().replace("__", "_1776_") + ".sif"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sif-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    rows = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != 500 or len({row["instance_id"] for row in rows}) != 500:
        raise RuntimeError("expected the pinned 500-row source dataset")
    selected = [
        row for row in rows
        if (args.sif_dir / sif_name(row["instance_id"])).is_file()
    ][:args.limit]
    if len(selected) != args.limit:
        raise RuntimeError(
            f"only {len(selected)} source rows have SIFs; need {args.limit}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "count": len(selected),
        "first": selected[0]["instance_id"],
        "last": selected[-1]["instance_id"],
    }))


if __name__ == "__main__":
    main()
