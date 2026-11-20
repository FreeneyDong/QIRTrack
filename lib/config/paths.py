import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = 'repository'


def configured_path(key):
    return os.environ.get(key, str(Path("__UNCONFIGURED__") / key))


def portable_path(value):
    """Store paths within this checkout relative to its root."""
    path = Path(value).expanduser().resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_config_paths(config, keys):
    """Resolve marked repository paths; retain cwd semantics for old configs."""
    resolved = dict(config)
    base = ROOT if config.get('path_base') == REPOSITORY else Path.cwd()
    for key in keys:
        if config.get(key):
            resolved[key] = str((base / Path(config[key]).expanduser()).resolve())
    return resolved
