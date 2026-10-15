"""Compatibility entry point; the application implementation lives in app.cli.

Both direct execution and imports of this historical module remain supported.
The module alias also keeps patches of CLI helpers attached to their globals.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import cli


if __name__ == "__main__":
    cli.main()
else:
    sys.modules[__name__] = cli
