from pathlib import Path

# Project-root data directory (independent of Jupyter/process cwd)
_DATA_ROOT = Path(__file__).resolve().parent
DATA_PATH = _DATA_ROOT / "data"
