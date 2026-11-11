#!/usr/bin/env python3
"""Convert the pinned SWE-bench OCI cache into per-instance Apptainer SIFs.

The conversion is offline: skopeo first writes one temporary docker-archive
from the shared, deduplicated OCI layout.  Apptainer extracts it to a sandbox,
then its bundled mksquashfs and SIF tools package the immutable image.  This
split avoids the proot-wrapped mksquashfs path, which is rejected by the
compute-node ptrace policy.  Final SIFs are published with an atomic rename
only after /testbed is verified.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from .prefetch_oci import PINNED_DATASET_SHA256, image_name, sha256


DEFAULT_APPTAINER = shutil.which("apptainer") or "apptainer"
DEFAULT_MKSQUASHFS = shutil.which("mksquashfs") or "mksquashfs"


def instance_tag(row):
    return str(row["instance_id"]).lower().replace("__", "_1776_")


def select_rows(rows, *, shard_index: int, shard_count: int, limit: int | None):
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    selected = rows[shard_index::shard_count]
    return selected[:limit] if limit is not None else selected


def load_rows(
    dataset: Path,
    limit: int | None,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
):
    if sha256(dataset) != PINNED_DATASET_SHA256:
        raise RuntimeError("dataset does not match the pinned Verified500 source order")
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    if len(rows) != 500 or len({row["instance_id"] for row in rows}) != 500:
        raise RuntimeError("expected the 500 unique SWE-bench Verified tasks")
    return select_rows(
        rows,
        shard_index=shard_index,
        shard_count=shard_count,
        limit=limit,
    )


def validate_inputs(oci_layout: Path, rows) -> None:
    state_path = oci_layout / "dualpd-prefetch-state.json"
    if not state_path.is_file():
        raise RuntimeError(f"missing OCI prefetch state: {state_path}")
    state = json.loads(state_path.read_text())
    if state.get("dataset_sha256") != PINNED_DATASET_SHA256:
        raise RuntimeError("OCI cache belongs to a different dataset")
    missing = [
        row["instance_id"]
        for row in rows
        if state.get("completed", {}).get(row["instance_id"]) != image_name(row)
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"OCI cache is missing {len(missing)} selected images; first: {preview}"
        )


def validate_sif(apptainer: str, path: Path, timeout_seconds: int = 600) -> None:
    subprocess.run(
        [
            apptainer,
            "--silent",
            "exec",
            "--cleanenv",
            "--containall",
            "--no-mount",
            "proc",
            "--net",
            "--network",
            "none",
            str(path),
            "test",
            "-e",
            "/testbed/.git",
        ],
        check=True,
        # Starting a SIF can briefly be I/O-bound while several other workers
        # are publishing images to the shared filesystem.  Keep validation
        # bounded, but allow enough time that transient metadata contention is
        # not mistaken for a corrupt image.
        timeout=timeout_seconds,
    )


def convert_one(
    row,
    *,
    oci_layout: Path,
    sif_dir: Path,
    temporary_root: Path,
    skopeo: str,
    apptainer: str,
    processors: int,
    mksquashfs: str = DEFAULT_MKSQUASHFS,
    policy: Path | None = None,
) -> tuple[Path, bool]:
    tag = instance_tag(row)
    output = sif_dir / f"{tag}.sif"
    lock_path = sif_dir / f".{tag}.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.is_file():
            validate_sif(apptainer, output)
            return output, False

        # A scheduler cancellation or SIGKILL cannot run ``finally``.  Once we
        # hold this image's exclusive lock, only remove temporary artifacts
        # whose exact parent and tag-specific prefix belong to this converter.
        temporary_root = temporary_root.resolve()
        sif_dir = sif_dir.resolve()
        for stale in temporary_root.glob(f"pd-sif-{tag}-*"):
            resolved = stale.resolve()
            if resolved.parent == temporary_root and resolved.name.startswith(
                f"pd-sif-{tag}-"
            ):
                shutil.rmtree(resolved)
        for stale in sif_dir.glob(f".{tag}.*.tmp.sif"):
            resolved = stale.resolve()
            if resolved.parent == sif_dir and resolved.name.startswith(f".{tag}."):
                resolved.unlink()

        temporary_dir = Path(tempfile.mkdtemp(prefix=f"pd-sif-{tag}-", dir=temporary_root))
        archive = temporary_dir / "image.tar"
        apptainer_tmp = temporary_dir / "apptainer-tmp"
        apptainer_cache = temporary_dir / "apptainer-cache"
        rootfs = temporary_dir / "rootfs"
        squashfs = temporary_dir / "rootfs.squashfs"
        temporary_sif = sif_dir / f".{tag}.{uuid.uuid4().hex}.tmp.sif"
        try:
            source = image_name(row).removeprefix("docker://")
            skopeo_command = [skopeo]
            if policy is not None:
                skopeo_command.extend(("--policy", str(policy)))
            skopeo_command.extend((
                "copy",
                f"oci:{oci_layout}:{tag}",
                f"docker-archive:{archive}:{source}",
            ))
            subprocess.run(skopeo_command, check=True)
            environment = os.environ.copy()
            apptainer_tmp.mkdir()
            apptainer_cache.mkdir()
            environment["APPTAINER_TMPDIR"] = str(apptainer_tmp)
            # The default cache lives under $HOME, which is NFS on this
            # cluster.  Concurrent builders can invalidate one another's NFS
            # handles while Apptainer is unpacking a layer.  Keep both cache
            # and temporary state node-local and scoped to this conversion.
            environment["APPTAINER_CACHEDIR"] = str(apptainer_cache)
            # Compute nodes without the setuid path make Apptainer build use
            # proot.  Their kernel rejects proot's seccomp accelerator; this
            # documented fallback affects only the offline build subprocess.
            environment.setdefault("PROOT_NO_SECCOMP", "1")
            subprocess.run(
                [
                    apptainer,
                    "build",
                    "--sandbox",
                    "--userns",
                    str(rootfs),
                    f"docker-archive:{archive}",
                ],
                check=True,
                env=environment,
            )
            # Calling the bundled compressor directly is intentional.  On
            # this cluster Apptainer can extract an OCI image in userns, but
            # its SIF build path wraps mksquashfs in proot; compute nodes deny
            # proot's ptrace(TRACEME).  Compression itself needs no ptrace.
            subprocess.run(
                [
                    mksquashfs,
                    str(rootfs),
                    str(squashfs),
                    "-noappend",
                    "-all-root",
                    "-processors",
                    str(processors),
                ],
                check=True,
            )
            subprocess.run([apptainer, "sif", "new", str(temporary_sif)], check=True)
            subprocess.run(
                [
                    apptainer,
                    "sif",
                    "add",
                    str(temporary_sif),
                    str(squashfs),
                    "--datatype",
                    "4",
                    "--parttype",
                    "2",
                    "--partfs",
                    "1",
                    "--partarch",
                    "2",
                    "--groupid",
                    "1",
                ],
                check=True,
            )
            validate_sif(apptainer, temporary_sif)
            os.replace(temporary_sif, output)
            return output, True
        finally:
            temporary_sif.unlink(missing_ok=True)
            shutil.rmtree(temporary_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--oci-layout", required=True, type=Path)
    parser.add_argument("--sif-dir", required=True, type=Path)
    parser.add_argument("--temporary-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, help="diagnostic source-order prefix only")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="number of deterministic source-order interleaved shards")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="zero-based shard selected by --shard-count")
    parser.add_argument("--skopeo", default="skopeo")
    parser.add_argument("--policy", type=Path,
                        help="explicit containers/image signature policy")
    parser.add_argument("--apptainer", default=DEFAULT_APPTAINER)
    parser.add_argument("--mksquashfs", default=DEFAULT_MKSQUASHFS)
    parser.add_argument("--mksquashfs-processors", type=int, default=4)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    if args.mksquashfs_processors < 1:
        parser.error("--mksquashfs-processors must be positive")
    for executable in (args.skopeo, args.apptainer, args.mksquashfs):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is unavailable: {executable}")
    if args.policy is not None and not args.policy.is_file():
        raise RuntimeError(f"signature policy is unavailable: {args.policy}")
    rows = load_rows(
        args.dataset,
        args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    validate_inputs(args.oci_layout, rows)
    args.sif_dir.mkdir(parents=True, exist_ok=True)
    args.temporary_root.mkdir(parents=True, exist_ok=True)

    built = 0
    for index, row in enumerate(rows, 1):
        output, created = convert_one(
            row,
            oci_layout=args.oci_layout,
            sif_dir=args.sif_dir,
            temporary_root=args.temporary_root,
            skopeo=args.skopeo,
            apptainer=args.apptainer,
            mksquashfs=args.mksquashfs,
            processors=args.mksquashfs_processors,
            policy=args.policy,
        )
        built += int(created)
        action = "built" if created else "validated"
        print(f"[{index}/{len(rows)}] {action} {output}", flush=True)
    print(json.dumps({"selected": len(rows), "built": built,
                      "shard_index": args.shard_index,
                      "shard_count": args.shard_count,
                      "sif_dir": str(args.sif_dir)}, indent=2))


if __name__ == "__main__":
    main()
