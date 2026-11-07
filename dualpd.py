"""Expose this flat source directory as the ``dualpd`` package.

This permits direct scripts and ``python -m dualpd`` without a nested source
directory or an examples/pd compatibility link.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent)]
__version__ = "0.1.0"

if __name__ == "__main__":
    from dualpd.runtime.cli import main
    main()
