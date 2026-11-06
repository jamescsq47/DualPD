"""Mechanical import/path migration, restricted to the extracted checkout."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PD = ROOT / "src/dualpd"


def main():
    modules = {p.stem for p in PD.glob("*.py")} - {"__init__", "cli"}
    modules.add("data")
    pattern = "|".join(sorted(modules, key=len, reverse=True))
    for tree in (PD, ROOT / "tools/dualpd"):
        for path in tree.rglob("*"):
            if not path.is_file() or "runs-host" in path.parts or path.suffix not in {".py", ".sh", ".sbatch"}:
                continue
            text = path.read_text()
            if path.suffix == ".py":
                text = re.sub(rf"(?m)^(\s*from )({pattern})(?=[.\s])", r"\1dualpd.\2", text)
                text = re.sub(rf"(?m)^(\s*import )({pattern})(\s+as\s+)", r"\1dualpd.\2\3", text)
                # Imports without alias retain their old local binding.
                text = re.sub(rf"(?m)^(\s*import )({pattern})(\s*(?:#.*)?$)", r"\1dualpd.\2 as \2\3", text)
                # Harness discovery and pytest monkeypatch dotted paths.
                text = re.sub(r'(?<=["\x27])data\.', 'dualpd.data.', text)
                text = text.replace('f"data.{name}"', 'f"dualpd.data.{name}"')
                for module in modules - {"data"}:
                    text = re.sub(rf'(?<=["\x27]){re.escape(module)}\.', f'dualpd.{module}.', text)
            text = text.replace("/home/shangy/dualpd/slime", str(ROOT))
            text = text.replace("/orcd/home/002/shangy/dualpd/slime", str(ROOT))
            text = text.replace('"slime/examples/pd/', '"examples/pd/')
            text = text.replace('"sglang-src/pd"', '"sglang_pd"')
            path.write_text(text)


if __name__ == "__main__":
    main()
