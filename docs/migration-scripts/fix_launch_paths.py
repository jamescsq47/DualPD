"""Mechanical correction of site-root references in copied launch scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for path in (ROOT / "scripts/cluster").glob("*"):
        if path.suffix not in {".py", ".sh", ".sbatch"}:
            continue
        text = path.read_text()
        text = text.replace('ROOT = Path("/home/shangy/dualpd")', 'ROOT = Path(__file__).resolve().parents[2]')
        text = text.replace('SLIME = REPO / "slime"', 'SLIME = Path(__file__).resolve().parents[2]')
        text = text.replace('"slime/tools/', '"tools/').replace("'slime/tools/", "'tools/")
        text = text.replace('"slime/examples/', '"examples/').replace("'slime/examples/", "'examples/")
        text = text.replace('/home/shangy/dualpd/sglang-src/pd"', '/home/shangy/dualpd/sglang_pd"')
        path.write_text(text)


if __name__ == "__main__":
    main()
