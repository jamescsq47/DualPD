"""One-time layout-only move: DualPD itself is the former examples/pd."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def merge(source, target):
    target.mkdir(exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if destination.exists() or destination.is_symlink():
            if item.name == ".gitignore":
                destination.write_text(destination.read_text() + "\n" + item.read_text())
                item.unlink()
            elif item.is_dir() and not item.is_symlink() and destination.is_dir():
                merge(item, destination)
            else:
                raise RuntimeError(f"layout collision: {destination}")
        else:
            item.rename(destination)
    source.rmdir()


def main():
    source = ROOT / "src/dualpd"
    if not source.is_dir():
        raise RuntimeError("source layout already moved or unavailable")
    # These are aliases, not data directories. The real contents are moved.
    for name in ("configs", "dualpd", "examples/pd"):
        path = ROOT / name
        if not path.exists() and not path.is_symlink():
            continue
        if not path.is_symlink():
            raise RuntimeError(f"expected compatibility symlink: {path}")
        path.unlink()
    if (ROOT / "examples").exists():
        (ROOT / "examples").rmdir()
    if (source / "runs-host").is_symlink():
        (source / "runs-host").unlink()
    if (ROOT / "results").exists():
        (ROOT / "results").rename(ROOT / "runs-host")
    if (source / "README.md").exists():
        (source / "README.md").rename(ROOT / "docs/ORIGINAL_PD_GUIDE.md")
    # Keep the repository-wide development instructions, already read.
    if (source / "AGENTS.md").exists():
        (source / "AGENTS.md").rename(ROOT / "docs/ORIGINAL_EXTRACTION_GATE.md")
    merge(source, ROOT)
    (ROOT / "tools/dualpd").rename(ROOT / "scripts/cluster")
    (ROOT / "tools").rmdir()
    (ROOT / "validation").rename(ROOT / "tests/reports")
    for path in ROOT.rglob("*"):
        if any(p in {".git", ".venv", "runs-host", "build", "__pycache__", "reports"} for p in path.parts):
            continue
        if not path.is_file() or path.suffix not in {".py", ".sh", ".sbatch", ".toml", ".json", ".yaml", ".yml"}:
            continue
        if path.name == "flatten_project.py":
            continue
        text = path.read_text()
        text = text.replace("src/dualpd/", "")
        text = text.replace('"src/dualpd"', '"."')
        text = text.replace("tools/dualpd", "scripts/cluster")
        text = text.replace("tools.dualpd", "dualpd.scripts.cluster")
        text = text.replace("examples/pd/", "")
        text = text.replace('"examples/pd"', '"."')
        text = text.replace("${PD_DIR}/../../..", "${PD_DIR}/..")
        text = text.replace('$(cd -- "${PD_DIR}/../.." && pwd)', '${PD_DIR}')
        text = text.replace("${PD_DIR}/../..", "${PD_DIR}")
        text = text.replace("${WORKSPACE_ROOT}/DualPD/src", "${WORKSPACE_ROOT}/DualPD")
        path.write_text(text)
    # Build products and old editable metadata are outside the runtime tree.
    # Preserve them outside the visible project instead of removing data.
    archive = ROOT / ".git/layout-before-flat"
    archive.mkdir(exist_ok=True)
    for name in ("src", "build"):
        path = ROOT / name
        if path.exists():
            shutil.move(str(path), str(archive / name))


if __name__ == "__main__":
    main()
