# Third-party engine

`sglang/` is a Git submodule of
[jamescsq47/sglang](https://github.com/jamescsq47/sglang), tracking its organized
`main` branch. The parent DualPD commit pins an exact engine commit, not a
floating checkout. See `sglang/README_DUALPD.md` for engine organization and
verification scope; upstream and dependency licenses remain in the engine tree.

Initialize the recorded version from the DualPD repository root:

```bash
git submodule update --init --recursive
```

To deliberately adopt a newer engine version, update the submodule, validate
the application/engine pair and commit the changed gitlink:

```bash
git submodule update --remote 3rdparty/sglang
git add 3rdparty/sglang
git commit -m "Update pinned SGLang engine"
```

Do not run the remote-update command implicitly during experiment startup.
Baseline experiments must continue using their separately selected baseline
engine and Python environment.
