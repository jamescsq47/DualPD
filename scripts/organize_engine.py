"""Move the custom engine extension together, retaining public module names."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT.parent / "sglang_pd"
DISAGG = ENGINE / "python/sglang/srt/disaggregation"


def main():
    target = DISAGG / "dualpd"
    target.mkdir(exist_ok=True)
    manifest = []
    paths = sorted(set(DISAGG.glob("agentic_*.py")) | set(DISAGG.glob("test_agentic_*.py")) | {DISAGG / "p2d_host_staging.py"})
    for path in paths:
        if not path.exists():
            continue
        destination = target / path.name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        path.rename(destination)
        manifest.append({"original": str(path.relative_to(ENGINE)),
                         "path": str(destination.relative_to(ENGINE)), "sha256": digest})
    output = ENGINE / "dualpd-source-map.json"
    if manifest:
        output.write_text(json.dumps({"source_commit": subprocess.check_output(
            ["git", "-C", str(ENGINE), "rev-parse", "HEAD"], text=True).strip(),
            "files": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
