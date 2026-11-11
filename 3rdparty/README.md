# Pinned engine

`sglang/` is a Git submodule of
[jamescsq47/sglang](https://github.com/jamescsq47/sglang/tree/main).
The DualPD commit pins an exact engine revision rather than a floating branch.
Third-party licenses and the engine's upstream tests remain in the submodule.

Initialize the recorded revision from the repository root:

```bash
git submodule update --init --recursive
```

To intentionally adopt a new engine revision, run
`git submodule update --remote 3rdparty/sglang`, validate the pair, and commit
the updated gitlink. Do not update it implicitly during experiment startup.
Baseline experiments must use their separately selected clean engine.
