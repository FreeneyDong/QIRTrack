"""Read trusted upstream pickle checkpoints after isolating their Python namespaces.

This does not bypass strict state-dict loading or change tensor values. As with
torch.load in upstream code, only use checkpoints from trusted sources.
"""
import pickle
from types import SimpleNamespace
import torch

class LegacyTrainingEnvironment:
    """Inert container for unused machine-local metadata in upstream checkpoints."""
    pass

def load_checkpoint(path, kind, map_location='cpu', **kwargs):
    if kind not in ('artrackv2', 'ostrack', 'sutrack'):
        raise ValueError(kind)
    class NamespacedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'lib.train.admin.local' and name == 'EnvironmentSettings':
                return LegacyTrainingEnvironment
            if module.startswith('lib.') and not module.startswith('lib.trackers.'):
                module = 'lib.trackers.' + kind + '.native.' + module[4:]
            return super().find_class(module, name)
    compat = SimpleNamespace(__name__='pickle', Unpickler=NamespacedUnpickler,
                             load=pickle.load, loads=pickle.loads)
    return torch.load(path, map_location=map_location, pickle_module=compat, **kwargs)
