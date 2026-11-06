#!/usr/bin/env python3
"""Run an empty-patch or Oracle-patch SWE-bench verifier smoke test."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


PD_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PD_DIR.parents[1]
for path in (str(PD_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dualpd.data.swe_bench.harness import _create_task  # noqa: E402
from dualpd.data.swe_bench.verifier import (  # noqa: E402
    capture_repository_patch,
    prepare_repository_baseline,
    run_inline_verifier,
)


def _read_instance(path: Path, instance_id: str) -> dict:
    with path.open(encoding="utf-8-sig") as source:
        if path.suffix.lower() == ".jsonl":
            rows = (json.loads(line) for line in source if line.strip())
        else:
            value = json.load(source)
            rows = iter(value if isinstance(value, list) else value["data"])
        for row in rows:
            if str(row.get("instance_id")) == instance_id:
                return dict(row)
    raise ValueError(f"instance {instance_id!r} was not found in {path}")


async def _run(args: argparse.Namespace) -> dict:
    row = _read_instance(args.dataset, args.instance_id)
    options = {
        "sandbox_backend": args.backend,
        "command_timeout_seconds": args.command_timeout,
        "container_start_timeout_seconds": args.start_timeout,
        "container_network": args.network,
    }
    if args.backend == "docker":
        if args.image:
            row["image_name"] = args.image
    else:
        options.update(
            apptainer_binary=args.apptainer_binary,
            apptainer_work_root=args.apptainer_work_root,
            apptainer_fakeroot=args.apptainer_fakeroot,
            apptainer_disable_proc=args.apptainer_disable_proc,
            apptainer_normalize_tool_timeout=True,
            apptainer_state_mode=args.apptainer_state_mode,
            apptainer_overlay_size_mb=args.apptainer_overlay_size_mb,
        )
        if args.image:
            options["apptainer_image_template"] = args.image
        else:
            options["apptainer_image_dir"] = args.apptainer_image_dir
    task = _create_task(row, options)
    try:
        await task.start()
        baseline = await prepare_repository_baseline(task, str(row["base_commit"]))
        if args.patch == "oracle":
            oracle = str(row.get("patch") or "")
            if not oracle:
                raise ValueError(f"{args.instance_id} has no Oracle patch")
            await task.upload_bytes(oracle.encode(), "/tmp/oracle.patch")
            exit_code, output = await task.execute(
                "git apply /tmp/oracle.patch", timeout=args.command_timeout
            )
            if exit_code != 0:
                raise RuntimeError(f"Oracle patch failed to apply: {output[-4000:]}")
        model_patch = await capture_repository_patch(task, baseline.image_commit)
        result = await run_inline_verifier(
            task,
            row,
            model_patch,
            timeout_seconds=args.verifier_timeout,
        )
        return {
            "instance_id": args.instance_id,
            "image": task.image,
            "patch": args.patch,
            "patch_chars": len(model_patch),
            "baseline": {
                "kind": baseline.kind,
                "official_base_commit": baseline.official_base_commit,
                "image_commit": baseline.image_commit,
                "fingerprint": baseline.fingerprint,
            },
            "verifier": result.to_metadata(),
            "task_metrics": task.metrics,
        }
    finally:
        await task.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--patch", choices=("empty", "oracle"), default="empty")
    parser.add_argument("--backend", choices=("docker", "apptainer"),
                        default="docker")
    parser.add_argument("--image", default="")
    parser.add_argument("--network", default="none")
    parser.add_argument("--command-timeout", type=float, default=300)
    parser.add_argument("--start-timeout", type=float, default=600)
    parser.add_argument("--verifier-timeout", type=float, default=2400)
    parser.add_argument("--apptainer-binary", default=(
        "/orcd/software/core/001/pkg/apptainer/1.5.2/bin/apptainer"
    ))
    parser.add_argument("--apptainer-image-dir", default="")
    parser.add_argument("--apptainer-work-root", default="")
    parser.add_argument("--apptainer-fakeroot", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--apptainer-disable-proc",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--apptainer-state-mode", choices=("host_bind", "overlay"),
                        default="host_bind")
    parser.add_argument("--apptainer-overlay-size-mb", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
