import json
from pathlib import Path

import convert_swe_oci_to_sif as c


def test_select_rows_builds_disjoint_complete_interleaved_shards():
    rows = list(range(11))
    shards = [
        c.select_rows(rows, shard_index=index, shard_count=4, limit=None)
        for index in range(4)
    ]
    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]]
    assert sorted(value for shard in shards for value in shard) == rows
    assert c.select_rows(rows, shard_index=2, shard_count=4, limit=2) == [2, 6]


def test_validate_inputs_requires_every_selected_image(tmp_path):
    layout = tmp_path / "oci"
    layout.mkdir()
    row = {"instance_id": "Owner__Repo-1"}
    (layout / "dualpd-prefetch-state.json").write_text(json.dumps({
        "dataset_sha256": c.PINNED_DATASET_SHA256,
        "completed": {},
    }))

    try:
        c.validate_inputs(layout, [row])
    except RuntimeError as exc:
        assert "Owner__Repo-1" in str(exc)
    else:
        raise AssertionError("incomplete OCI cache was accepted")


def test_convert_one_is_atomic_and_validates_existing_sif(tmp_path, monkeypatch):
    layout = tmp_path / "oci"
    sif_dir = tmp_path / "sifs"
    scratch = tmp_path / "scratch"
    for directory in (layout, sif_dir, scratch):
        directory.mkdir()
    stale_dir = scratch / "pd-sif-owner_1776_repo-1-stale"
    stale_dir.mkdir()
    (stale_dir / "image.tar").write_bytes(b"partial")
    stale_sif = sif_dir / ".owner_1776_repo-1.deadbeef.tmp.sif"
    stale_sif.write_bytes(b"partial")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[1] == "copy":
            archive = Path(argv[-1].split(":", 2)[1])
            archive.write_bytes(b"archive")
        elif argv[0] == "mksquashfs":
            Path(argv[2]).write_bytes(b"squashfs")
        elif argv[1:3] == ["sif", "new"]:
            Path(argv[3]).write_bytes(b"sif")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(c.subprocess, "run", run)
    row = {"instance_id": "Owner__Repo-1"}
    output, created = c.convert_one(
        row, oci_layout=layout, sif_dir=sif_dir, temporary_root=scratch,
        skopeo="skopeo", apptainer="apptainer", mksquashfs="mksquashfs",
        processors=1,
    )

    assert created is True and output.read_bytes() == b"sif"
    assert not stale_dir.exists() and not stale_sif.exists()
    assert not list(sif_dir.glob("*.tmp.sif"))
    assert not list(scratch.iterdir())
    assert calls[0][0][2] == f"oci:{layout}:owner_1776_repo-1"
    assert calls[1][0][2:4] == ["--sandbox", "--userns"]
    assert calls[1][1]["env"]["PROOT_NO_SECCOMP"] == "1"
    assert calls[1][1]["env"]["APPTAINER_TMPDIR"].endswith("/apptainer-tmp")
    assert calls[1][1]["env"]["APPTAINER_CACHEDIR"].endswith("/apptainer-cache")
    assert calls[2][0][0] == "mksquashfs"
    assert calls[2][0][-2:] == ["-processors", "1"]
    assert calls[4][0][calls[4][0].index("--parttype") + 1] == "2"

    calls.clear()
    same, created = c.convert_one(
        row, oci_layout=layout, sif_dir=sif_dir, temporary_root=scratch,
        skopeo="skopeo", apptainer="apptainer", mksquashfs="mksquashfs",
        processors=1,
    )
    assert same == output and created is False
    assert len(calls) == 1 and calls[0][0][1] == "--silent"
