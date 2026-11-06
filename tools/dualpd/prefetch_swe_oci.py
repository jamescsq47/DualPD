#!/usr/bin/env python3
"""Prefetch the pinned SWE-bench Verified images into one resumable OCI layout.

This deliberately does not claim to install Docker.  The shared OCI store is a
transport cache: once a supported Docker/Podman runtime is available, skopeo
can import each named image without going back to Docker Hub.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


PINNED_DATASET_SHA256 = "f61cd55ceb35b61ad592f645abcbfc8ea4d294c6c9f3c8f15e83211a8e8db98c"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_name(row):
    if row.get("image_name"):
        return str(row["image_name"])
    instance = row["instance_id"].lower().replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{instance}:latest"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--oci-layout", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="diagnostic prefix only")
    parser.add_argument("--skopeo", default="skopeo")
    parser.add_argument("--policy", type=Path,
                        help="explicit containers/image signature policy (TLS remains verified)")
    parser.add_argument("--authfile", type=Path,
                        help="containers/image auth file (keep credentials outside the repository)")
    parser.add_argument("--retry-until-seconds", type=int, default=0,
                        help="keep retrying registry failures for this many seconds")
    args = parser.parse_args()
    if args.retry_until_seconds < 0:
        parser.error("--retry-until-seconds must be nonnegative")
    if args.authfile is not None and not args.authfile.is_file():
        parser.error(f"auth file is unavailable: {args.authfile}")
    if sha256(args.dataset) != PINNED_DATASET_SHA256:
        raise RuntimeError("refusing to prefetch images for a different dataset/order")
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if len(rows) != 500 or len({row["instance_id"] for row in rows}) != 500:
        raise RuntimeError("expected the 500 unique SWE-bench Verified tasks")
    selected = rows[:args.limit] if args.limit is not None else rows
    args.oci_layout.mkdir(parents=True, exist_ok=True)
    state_path = args.oci_layout / "dualpd-prefetch-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "dataset_sha256": PINNED_DATASET_SHA256, "completed": {}, "format": "oci-layout"
    }
    if state.get("dataset_sha256") != PINNED_DATASET_SHA256:
        raise RuntimeError("OCI cache belongs to a different dataset")
    mapping = []
    for index, row in enumerate(selected, 1):
        source = image_name(row)
        tag = row["instance_id"].lower().replace("__", "_1776_")
        mapping.append({"instance_id": row["instance_id"], "source": source, "oci_tag": tag})
        if state["completed"].get(row["instance_id"]) == source:
            print(f"[{index}/{len(selected)}] cached {source}", flush=True)
            continue
        print(f"[{index}/{len(selected)}] copying {source}", flush=True)
        command = [args.skopeo]
        if args.policy is not None:
            command += ["--policy", str(args.policy.resolve())]
        command += ["copy"]
        if args.authfile is not None:
            command += ["--authfile", str(args.authfile.resolve())]
        command += [
            "--retry-times", "5", "--override-os", "linux",
            "--override-arch", "amd64", "docker://" + source.removeprefix("docker://"),
            f"oci:{args.oci_layout}:{tag}",
        ]
        retry_deadline = time.monotonic() + args.retry_until_seconds
        while True:
            try:
                subprocess.run(command, check=True)
                break
            except subprocess.CalledProcessError:
                if time.monotonic() >= retry_deadline:
                    raise
                remaining = max(1, int(retry_deadline - time.monotonic()))
                pause = min(60, remaining)
                print(f"registry copy failed; retrying in {pause}s "
                      f"({remaining}s retry budget remains)", flush=True)
                time.sleep(pause)
        state["completed"][row["instance_id"]] = source
        atomic_json(state_path, state)
    atomic_json(args.oci_layout / "dualpd-image-map.json", mapping)
    print(json.dumps({"requested": len(selected), "completed": len(state["completed"]),
                      "oci_layout": str(args.oci_layout)}, indent=2))


if __name__ == "__main__":
    main()
