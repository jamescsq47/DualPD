#!/usr/bin/env python3
"""Copy a tar archive while mapping every member owner to uid/gid zero."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    with tarfile.open(args.source, "r:*") as source, tarfile.open(
        args.destination, "w"
    ) as destination:
        for member in source:
            member.uid = 0
            member.gid = 0
            member.uname = "root"
            member.gname = "root"
            payload = source.extractfile(member) if member.isfile() else None
            destination.addfile(member, payload)


if __name__ == "__main__":
    main()
