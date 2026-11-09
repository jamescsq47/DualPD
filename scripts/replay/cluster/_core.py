"""External migration draft, never used by a serving model or controller.

Only seven canonical stdlib-only site/shared tools may be loaded. Historical
siblings are not handled here. Bootstrap callers must not race unrelated
ordinary imports of these modules; this is not a general import-hook service.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType


DEPENDENCIES = {
    'process_supervisor': (),
    'qwen35_slurm': (),
    'multinode': ('process_supervisor',),
    'qwen35_native_pd_slurm': ('process_supervisor', 'qwen35_slurm'),
    'qwen35_colocated_slurm': ('process_supervisor', 'qwen35_slurm'),
    'dram_resource_sampler': (),
    'dram_complete_frame_analysis': (),
}
PREFIX = 'dualpd.scripts.cluster.'

# A direct-script and package import of this helper must share the same lock.
# This private state is not a stand-in for any real DualPD package.
_candidate_state = ModuleType('_dualpd_replay_core_loader_state')
_candidate_state.lock = threading.RLock()
_state = sys.modules.setdefault(_candidate_state.__name__, _candidate_state)


class CoreLoader:
    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.core_root = self.project_root / 'scripts/cluster'

    def _validate_origin(self, module, path):
        origin = getattr(module, '__file__', None)
        if origin is None or Path(origin).resolve() != path.resolve():
            raise ImportError('loaded core alias has an unexpected source origin')

    def _parent(self):
        # No fake parent tree and no sys.path edits. Public/package support is
        # conditional on the actual selected DualPD package being importable.
        if 'dualpd' not in sys.modules and importlib.util.find_spec('dualpd') is None:
            return None
        package = importlib.import_module('dualpd')
        paths = tuple(Path(path).resolve() for path in getattr(package, '__path__', ()))
        if self.project_root not in paths:
            raise ImportError('importable DualPD package is not the selected checkout')
        parent = importlib.import_module('dualpd.scripts.cluster')
        if self.core_root.resolve() not in tuple(
                Path(path).resolve() for path in getattr(parent, '__path__', ())):
            raise ImportError('core parent package has an unexpected source origin')
        return parent

    def load(self, name):
        if name not in DEPENDENCIES:
            raise ValueError('unknown maintained core/shared module')
        path = self.core_root / (name + '.py')
        aliases = (name, PREFIX + name)
        with _state.lock:
            existing = [sys.modules[alias] for alias in aliases if alias in sys.modules]
            for module in existing:
                self._validate_origin(module, path)
            if len({id(module) for module in existing}) > 1:
                raise ImportError('split core module identities; use a fresh process')
            parent = self._parent()
            if parent is not None and hasattr(parent, name):
                bound = getattr(parent, name)
                self._validate_origin(bound, path)
                if existing and bound is not existing[0]:
                    raise ImportError('split core parent binding; use a fresh process')
                if not existing:
                    existing = [bound]
            for dependency in DEPENDENCIES[name]:
                self.load(dependency)
            if existing:
                module = existing[0]
                for alias in aliases:
                    sys.modules[alias] = module
                if parent is not None:
                    setattr(parent, name, module)
                return module
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError('cannot load maintained core/shared source')
            module = importlib.util.module_from_spec(spec)
            for alias in aliases:
                sys.modules[alias] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                # Preserve the original exception and pre-existing dependency
                # modules. Only aliases created for this failed load are ours.
                for alias in aliases:
                    if sys.modules.get(alias) is module:
                        del sys.modules[alias]
                raise
            if parent is not None:
                setattr(parent, name, module)
            return module


_DEFAULT = CoreLoader(Path(__file__).resolve().parents[3])

def load_core(name):
    return _DEFAULT.load(name)
