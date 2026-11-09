"""Disposable stdlib fixture tests; do not execute operational site scripts."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from dualpd.scripts.replay.cluster._core import CoreLoader, DEPENDENCIES, PREFIX


@contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='dualpd-core-loader-') as directory:
        root = Path(directory) / 'dualpd'
        core = root / 'scripts/cluster'
        core.mkdir(parents=True)
        (root / '__init__.py').write_text('')
        for name, dependencies in DEPENDENCIES.items():
            imports = ''.join('import ' + dependency + '\n' for dependency in dependencies)
            (core / (name + '.py')).write_text(imports + '''
VALUE = object()
import logging
LOGGER = logging.getLogger(__name__)
class Marker:
    pass
def value():
    return VALUE
''')
        names = [*DEPENDENCIES, *(PREFIX + name for name in DEPENDENCIES),
                 'dualpd', 'dualpd.scripts', 'dualpd.scripts.cluster']
        # Remove only test aliases temporarily, preserving the interpreter's
        # prior state. The real package comes from a disposable fixture tree.
        with patch.dict(sys.modules), patch.object(sys, 'path', [str(root.parent), *sys.path]):
            for name in names:
                sys.modules.pop(name, None)
            importlib.invalidate_caches()
            yield root, core, CoreLoader(root)


class CoreLoaderTests(unittest.TestCase):
    def test_separate_helper_copies_share_lock_and_module_identity(self):
        helper = Path(__file__).resolve().parents[1] / 'scripts/replay/cluster/_core.py'
        spec = importlib.util.spec_from_file_location('_fixture_second_core_helper', helper)
        second = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(second)
        with fixture() as (root, core, loader):
            other = second.CoreLoader(root)
            with ThreadPoolExecutor(max_workers=4) as workers:
                modules = list(workers.map(
                    lambda index: (loader if index % 2 else other).load('qwen35_native_pd_slurm'),
                    range(20)))
            self.assertTrue(all(module is modules[0] for module in modules))
            self.assertIs(modules[0].qwen35_slurm, sys.modules['qwen35_slurm'])

    def test_parent_only_binding_is_reused_without_reexecution(self):
        with fixture() as (root, core, loader):
            original = loader.load('qwen35_slurm')
            sys.modules.pop('qwen35_slurm')
            sys.modules.pop(PREFIX + 'qwen35_slurm')
            self.assertIs(loader.load('qwen35_slurm'), original)

    def test_bare_loading_without_importable_package_does_not_fabricate_parent(self):
        with fixture() as (root, core, loader):
            before = list(sys.path)
            with patch('importlib.util.find_spec', return_value=None):
                actual = loader.load('qwen35_slurm')
            self.assertEqual(sys.path, before)
            self.assertNotIn('dualpd', sys.modules)
            self.assertNotIn('dualpd.scripts.cluster', sys.modules)
            self.assertIs(actual, sys.modules['qwen35_slurm'])

    def test_pickle_and_logger_module_identity_remain_stable(self):
        for public_first in (False, True):
            with self.subTest(public_first=public_first), fixture() as (root, core, loader):
                if public_first:
                    module = importlib.import_module(PREFIX + 'qwen35_slurm')
                else:
                    module = loader.load('qwen35_slurm')
                old_module_name = module.Marker.__module__
                original_logger = module.LOGGER
                value = module.Marker()
                loader.load('qwen35_slurm')
                self.assertEqual(module.Marker.__module__, old_module_name)
                self.assertIs(module.LOGGER, original_logger)
                self.assertEqual(module.LOGGER.name, old_module_name)
                self.assertIs(type(pickle.loads(pickle.dumps(value))), module.Marker)

    def test_missing_and_invalid_source_leave_no_partial_aliases(self):
        for invalid in (False, True):
            with self.subTest(invalid=invalid), fixture() as (root, core, loader):
                path = core / 'qwen35_slurm.py'
                if invalid:
                    path.write_text('def broken(:\n')
                else:
                    path.unlink()
                with self.assertRaises(SyntaxError if invalid else FileNotFoundError):
                    loader.load('qwen35_slurm')
                self.assertNotIn('qwen35_slurm', sys.modules)
                self.assertNotIn(PREFIX + 'qwen35_slurm', sys.modules)
                self.assertFalse(hasattr(sys.modules['dualpd.scripts.cluster'], 'qwen35_slurm'))

    def test_dependency_exception_is_not_replaced_and_target_never_published(self):
        with fixture() as (root, core, loader):
            error = ImportError('core dependency failure')
            hook = ModuleType('_fixture_dependency_error')
            hook.error = error
            sys.modules[hook.__name__] = hook
            (core / 'qwen35_slurm.py').write_text(
                'from _fixture_dependency_error import error\nraise error\n')
            try:
                loader.load('qwen35_native_pd_slurm')
            except ImportError as caught:
                self.assertIs(caught, error)
            else:
                self.fail('dependency failure was swallowed')
            for name in ('qwen35_slurm', 'qwen35_native_pd_slurm'):
                self.assertNotIn(name, sys.modules)
                self.assertNotIn(PREFIX + name, sys.modules)
            self.assertIs(sys.modules['process_supervisor'],
                          sys.modules[PREFIX + 'process_supervisor'])

    def test_all_names_bare_first_and_public_first(self):
        for name in DEPENDENCIES:
            for public_first in (False, True):
                with self.subTest(name=name, public_first=public_first), fixture() as (root, core, loader):
                    before_path = list(sys.path)
                    if public_first:
                        for dependency in DEPENDENCIES[name]:
                            loader.load(dependency)
                        old = importlib.import_module(PREFIX + name)
                    else:
                        # Recreate an ordinary old direct-script bare import,
                        # then restore the search path before invoking loader.
                        sys.path.insert(0, str(core))
                        try:
                            old = importlib.import_module(name)
                        finally:
                            sys.path[:] = before_path
                    original_spec, original_name = old.__spec__, old.__name__
                    actual = loader.load(name)
                    self.assertIs(actual, old)
                    self.assertIs(importlib.import_module(name), old)
                    self.assertIs(importlib.import_module(PREFIX + name), old)
                    self.assertIs(getattr(importlib.import_module('dualpd.scripts.cluster'), name), old)
                    self.assertIs(actual.__spec__, original_spec)
                    self.assertEqual(actual.__name__, original_name)
                    self.assertIs(actual.value.__globals__, actual.__dict__)
                    marker = object()
                    actual.VALUE = marker
                    self.assertIs(sys.modules[PREFIX + name].value(), marker)
                    self.assertIs(actual.Marker, old.Marker)
                    self.assertEqual(sys.path, before_path)
                    for dependency in DEPENDENCIES[name]:
                        self.assertIs(getattr(actual, dependency), sys.modules[dependency])

    def test_actual_callers_preserve_aliases_and_import_error_identity(self):
        root = Path(__file__).resolve().parents[1]
        code = r'''
import importlib, os, pathlib, socket, subprocess, sys
from unittest.mock import patch
root = pathlib.Path(sys.argv[1])
name, mode, fault = sys.argv[2:]
if mode == 'direct':
    sys.path.insert(0, str(root / 'scripts/replay/cluster'))
helper = importlib.import_module('_core' if mode == 'direct' else
                                'dualpd.scripts.replay.cluster._core')
original_load = helper.CoreLoader.load
sentinel = ImportError('actual caller core import sentinel')
def load(self, name):
    target = ('dram_resource_sampler' if sys.argv[2].startswith('browsecomp_')
              else 'qwen35_slurm')
    if fault == 'fault' and name == target:
        raise sentinel
    return original_load(self, name)
def forbidden(*args, **kwargs):
    raise AssertionError('operational action during import')
before = dict(os.environ)
with patch.object(helper.CoreLoader, 'load', load), \
     patch.object(subprocess, 'Popen', forbidden), \
     patch.object(os, 'kill', forbidden), patch.object(os, 'killpg', forbidden), \
     patch.object(socket.socket, 'bind', forbidden), \
     patch.object(socket.socket, 'connect', forbidden):
    try:
        module = importlib.import_module(name if mode == 'direct' else
                                        'dualpd.scripts.replay.cluster.' + name)
    except ImportError as error:
        assert fault == 'fault' and error is sentinel
    else:
        assert fault == 'normal', 'caller swallowed core import exception'
        assert pathlib.Path(module.__file__).resolve() == (
            root / 'scripts/replay/cluster' / (name + '.py')).resolve()
        for core in helper.DEPENDENCIES:
            if core in sys.modules:
                assert sys.modules[core] is sys.modules[helper.PREFIX + core]
        if name == 'minimax_m27_three_node_autopilot':
            assert module.native.site is module.site
            assert module.multinode.process_control is sys.modules['process_supervisor'].control
assert os.environ == before
assert not any(n == 'sglang' or n.startswith(('sglang.', 'torch', 'slime'))
               for n in sys.modules)
'''
        for caller in ('minimax_m27_three_node_autopilot',
                       'qwen38_27b_hicache_ratio12_single_run',
                       'browsecomp_qwen3_8b_dram_global64_128_h100'):
            for mode in ('direct', 'package'):
                for fault in ('normal', 'fault'):
                    with self.subTest(caller=caller, mode=mode, fault=fault):
                        result = subprocess.run(
                            [sys.executable, '-B', '-c', code, str(root), caller, mode, fault],
                            cwd=root.parent, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrong_origin_not_overwritten(self):
        with fixture() as (root, core, loader):
            module = ModuleType('qwen35_slurm')
            module.__file__ = '/not-the-selected-checkout/qwen35_slurm.py'
            sys.modules['qwen35_slurm'] = module
            with self.assertRaisesRegex(ImportError, 'unexpected source'):
                loader.load('qwen35_slurm')
            self.assertIs(sys.modules['qwen35_slurm'], module)
            self.assertNotIn(PREFIX + 'qwen35_slurm', sys.modules)

    def test_split_aliases_rejected_without_overwrite(self):
        with fixture() as (root, core, loader):
            objects = [ModuleType('first'), ModuleType('second')]
            for alias, module in zip(('qwen35_slurm', PREFIX + 'qwen35_slurm'), objects):
                module.__file__ = str(core / 'qwen35_slurm.py')
                sys.modules[alias] = module
            with self.assertRaisesRegex(ImportError, 'split core module'):
                loader.load('qwen35_slurm')
            self.assertIs(sys.modules['qwen35_slurm'], objects[0])
            self.assertIs(sys.modules[PREFIX + 'qwen35_slurm'], objects[1])

    def test_real_import_error_identity_and_alias_rollback(self):
        with fixture() as (root, core, loader):
            error = ImportError('dependency really failed')
            hook = ModuleType('_fixture_import_error')
            hook.error = error
            sys.modules[hook.__name__] = hook
            (core / 'qwen35_native_pd_slurm.py').write_text(
                'from _fixture_import_error import error\nraise error\n')
            try:
                loader.load('qwen35_native_pd_slurm')
            except ImportError as caught:
                self.assertIs(caught, error)
            else:
                self.fail('original import exception swallowed')
            self.assertNotIn('qwen35_native_pd_slurm', sys.modules)
            self.assertNotIn(PREFIX + 'qwen35_native_pd_slurm', sys.modules)
            # Dependencies were successfully imported and are not rolled back.
            self.assertIs(sys.modules['qwen35_slurm'], sys.modules[PREFIX + 'qwen35_slurm'])

    def test_unknown_module_rejected(self):
        with fixture() as (root, core, loader):
            for name in ('os', '../qwen35_slurm', 'historical_matrix'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    loader.load(name)

    def test_conflicting_real_parent_attribute_rejected(self):
        with fixture() as (root, core, loader):
            original = loader.load('qwen35_slurm')
            parent = importlib.import_module('dualpd.scripts.cluster')
            wrong = ModuleType('conflicting_parent_value')
            wrong.__file__ = str(core / 'qwen35_slurm.py')
            parent.qwen35_slurm = wrong
            with self.assertRaisesRegex(ImportError, 'split core parent'):
                loader.load('qwen35_slurm')
            self.assertIs(sys.modules['qwen35_slurm'], original)
            self.assertIs(parent.qwen35_slurm, wrong)

    def test_foreign_importable_package_rejected(self):
        with fixture() as (root, core, loader):
            package = ModuleType('dualpd')
            package.__path__ = ['/wrong/checkout']
            sys.modules['dualpd'] = package
            with self.assertRaisesRegex(ImportError, 'not the selected checkout'):
                loader.load('qwen35_slurm')
            self.assertNotIn('qwen35_slurm', sys.modules)
            self.assertNotIn(PREFIX + 'qwen35_slurm', sys.modules)


if __name__ == '__main__':
    unittest.main()
