import os
from pathlib import Path

def configured_path(key):
    # Explicitly invalid placeholder: never reaches an author server by default.
    return os.environ.get(key, str(Path("__UNCONFIGURED__") / key))
