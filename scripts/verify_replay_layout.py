"""Complete immutable replay-address proof, not runtime/model acceptance.

Reconstruct every migrated file from6bc1523 using separately reviewable pure
address/import rules plus a finite pinned literal edit list. Never import a
historical controller or execute its entrypoint to verify source equivalence.
"""
from functools import lru_cache
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = '6bc15235174dee9eb5f4b331d558b777ee031d9b'
PINNED = {
    'scripts/replay/manifest.json': 'ec6eea3e344ea8f1ccd1d74b0693f507acb1881f945495c63f819b0d35817331',
    'scripts/replay/address-edits.json': 'de9f58f859a71e5b0242183016ce11abc1041e7e363b7c1ecfa672fba15b9fd7',
    'scripts/replay/layout_rules.py': '6c297a69e064be10eb91919ab399cd3ccb9a0084cc017075e81d547e33cf4e4a',
    'scripts/replay/cluster/_core.py': '4ae8b4ad23a3cf472088117aa7e0a4384605b3e2fce55c97b508fa13f9d2439d',
}
# Stage8's explicitly reviewed installed-package entry fix, separate from
# unchanged stage7b archival. These are not blanket retained-file waivers.
ROUTER_ENTRY_UPDATES = {
    'scripts/new_method/internal/launch_late_binding_router.py':
        ('current', 'launch_late_binding_router.py'),
    'scripts/new_method/internal/launch_mamba_late_binding_router.py':
        ('legacy_mamba', 'legacy_mamba_router.py'),
}
POST_ARCHIVE_HELPERS = {
    'runtime/engine_entry.py': '976877dd18509a8e2c55cbd448a5503f1fd085f3052195fa2c474e51b4f99402',
}


def router_entry_update(relative, source):
    kind, filename = ROUTER_ENTRY_UPDATES[relative]
    assert source.count('import os\n') == 1
    before = (
        '    engine = Path(os.environ.get("DUALPD_ENGINE_ROOT", Path(__file__).resolve().parents[4] / "sglang_pd"))\n'
        f'    runpy.run_path(str(engine / "python/sglang/srt/disaggregation/dualpd/router/{filename}"), run_name="__main__")\n'
    )
    after = (
        '    helper = Path(__file__).resolve().parents[3] / "runtime/engine_entry.py"\n'
        f'    runpy.run_path(str(helper))["launch_router"]("{kind}")\n'
    )
    assert source.count(before) == 1
    return source.replace('import os\n', '', 1).replace(before, after, 1)


def sha(value):
    return hashlib.sha256(value).hexdigest()


@lru_cache(maxsize=1)
def originals():
    raw = subprocess.check_output([
        'git', '-C', str(ROOT), 'archive', BASE, 'scripts/cluster', 'scripts/new_method',
    ])
    # Read members in memory, never extract an archive onto the workspace.
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {member.name: archive.extractfile(member).read()
                for member in archive.getmembers() if member.isfile()}


@lru_cache(maxsize=1)
def executable_modes():
    raw = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '-z', BASE])
    modes = {}
    for entry in raw.split(b'\0'):
        if entry:
            info, name = entry.split(b'\t', 1)
            modes[name.decode()] = info.split()[0] == b'100755'
    return modes


@lru_cache(maxsize=None)
def original(relative):
    if relative in originals():
        return originals()[relative].decode()
    return subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', BASE + ':' + relative], text=True)


@lru_cache(maxsize=1)
def rules():
    path = ROOT / 'scripts/replay/layout_rules.py'
    assert sha(path.read_bytes()) == PINNED['scripts/replay/layout_rules.py'], 'proof rules changed'
    spec = importlib.util.spec_from_file_location('_reviewed_replay_layout_rules', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(root=ROOT):
    root = Path(root)
    for relative, digest in PINNED.items():
        assert sha((root / relative).read_bytes()) == digest, (relative, 'reviewed source/data changed')
    for relative, digest in POST_ARCHIVE_HELPERS.items():
        path = root / relative
        assert path.is_file() and not path.is_symlink(), (relative, 'regular helper required')
        assert sha(path.read_bytes()) == digest, (relative, 'reviewed package helper changed')
    manifest = json.loads((root / 'scripts/replay/manifest.json').read_text())
    edits = json.loads((root / 'scripts/replay/address-edits.json').read_text())
    assert manifest['immutable_revision'] == edits['immutable_revision'] == BASE
    rows = manifest['files']
    names = [row['original'] for row in rows]
    destinations = [row['path'] for row in rows]
    assert len(rows) == len(set(names)) == len(set(destinations)) == 201
    assert set(names) == set(originals()), 'incomplete immutable inventory'
    mapping = {row['original']: row['path'] for row in rows if row['path'] != row['original']}
    assert len(mapping) == 142
    for old, new in mapping.items():
        assert new == old.replace('scripts/', 'scripts/replay/', 1), (old, 'unexpected destination')
    actual_files = set()
    for path in (root / 'scripts/replay').rglob('*'):
        # Check links before excluding caches: a cache-shaped symlink must not
        # provide a route to unreviewed executable sources or external trees.
        assert not path.is_symlink(), (str(path), 'unreviewed replay symlink')
        if '__pycache__' in path.parts:
            continue
        mode = path.lstat().st_mode
        assert stat.S_ISDIR(mode) or stat.S_ISREG(mode), (str(path), 'unsupported replay file type')
        if stat.S_ISREG(mode):
            actual_files.add(str(path.relative_to(root)))
    assert actual_files == set(mapping.values()) | set(PINNED) | {
        'scripts/replay/README.md',
    }, 'unreviewed replay source inventory'
    siblings = {Path(name).stem for name in mapping
                if name.startswith('scripts/cluster/') and name.endswith('.py')}
    special = {row['original']: row for row in edits['files']}
    assert len(special) == len(edits['files']) == 24
    assert set(special) <= set(mapping)
    moved = retained = updated_entries = docs = 0
    for row in rows:
        old, new = row['original'], row['path']
        assert (root / new).is_file() and not (root / new).is_symlink(), (new, 'source must be a regular file')
        assert bool((root / new).stat().st_mode & stat.S_IXUSR) == executable_modes()[old], \
            (old, 'executable mode differs')
        source_bytes = originals()[old]
        assert sha(source_bytes) == row['original_sha256'], (old, 'immutable hash differs')
        if old not in mapping:
            if old.endswith('.md'):
                # The three navigation documents are not serving code. Their
                # source hashes remain recorded, not falsely called unchanged.
                docs += 1
                assert (root / old).is_file()
                continue
            if old in ROUTER_ENTRY_UPDATES:
                expected = router_entry_update(old, source_bytes.decode())
                assert (root / old).read_bytes() == expected.encode(), (old, 'whole package entry differs')
                updated_entries += 1
                continue
            assert (root / old).read_bytes() == source_bytes, (old, 'maintained code changed')
            retained += 1
            continue
        assert not (root / old).exists(), (old, 'old source remains')
        expected = rules().relocate(old, source_bytes.decode(), mapping)
        if old.startswith('scripts/cluster/') and old.endswith('.py'):
            expected = rules().package_imports(expected, siblings)
        if old in special:
            change = special[old]
            assert change['path'] == new
            for edit in change['edits']:
                before = edit['before']
                assert before and expected.count(before) == 1, (old, 'ambiguous reviewed edit')
                expected = expected.replace(before, edit['after'], 1)
        assert (root / new).read_bytes() == expected.encode(), (old, 'whole replay file differs')
        moved += 1
    # Actual runtime/test callers outside the original two launcher directories
    # are checked completely too, with exactly one declared address replacement.
    callers = {
        'scripts/tools/run_swe_aiohttp_matrix.py':
            ("launcher=PD/'scripts/new_method'/filename", "launcher=PD/'scripts/replay/new_method'/filename"),
        'tests/test_qwen35_fused_swe_topology.py':
            ('scripts/new_method/run_qwen35_fused_swe500_2p6d.sh',
             'scripts/replay/new_method/run_qwen35_fused_swe500_2p6d.sh'),
    }
    for relative, (before, after) in callers.items():
        source = original(relative)
        assert source.count(before) == 1
        assert (root / relative).read_text() == source.replace(before, after, 1), (relative, 'caller differs')
    assert (moved, retained, updated_entries, docs) == (142, 54, 2, 3)
    return {
        'immutable_revision': BASE, 'inventory_entries_verified': 201,
        'executable_modes_verified': 201,
        'whole_replay_files_reconstructed': moved, 'byte_identical_retained_code': retained,
        'post_archive_whole_router_entries_reconstructed': updated_entries,
        'post_archive_helper_hashes_verified': len(POST_ARCHIVE_HELPERS),
        'navigation_documents_not_called_byte_identical': docs,
        'finite_special_edit_files': len(special), 'complete_outside_callers_verified': len(callers),
        'new_core_bootstrap_hash_verified': True,
        'scope': 'Complete source-address/import reconstruction only. New bootstrap/import selection needs independent contract/fault review. Not runtime replay, native/GPU/performance, TCP/TP2/hybrid or complete-goal acceptance.',
    }


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
