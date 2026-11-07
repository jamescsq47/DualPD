#!/usr/bin/env python3
"""Import the pinned shared SWE OCI cache into a local Docker daemon."""

import argparse
import json
from pathlib import Path
import subprocess

from prefetch_swe_oci import PINNED_DATASET_SHA256, image_name, sha256


def docker_target(source):
    return source.removeprefix("docker://").removeprefix("docker.io/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--oci-layout", required=True, type=Path)
    parser.add_argument("--skopeo", default="skopeo")
    args = parser.parse_args()
    if sha256(args.dataset) != PINNED_DATASET_SHA256:
        raise RuntimeError("dataset does not match the pinned Verified500 source order")
    subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    if len(rows) != 500:
        raise RuntimeError("expected all 500 Verified tasks")
    for index, row in enumerate(rows, 1):
        source = image_name(row)
        target = docker_target(source)
        tag = row["instance_id"].lower().replace("__", "_1776_")
        found = subprocess.run(["docker", "image", "inspect", target],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if found.returncode == 0:
            print(f"[{index}/500] present {target}", flush=True)
            continue
        print(f"[{index}/500] importing {target}", flush=True)
        subprocess.run([args.skopeo, "copy", f"oci:{args.oci_layout}:{tag}",
                        "docker-daemon:" + target], check=True)


if __name__ == "__main__":
    main()
