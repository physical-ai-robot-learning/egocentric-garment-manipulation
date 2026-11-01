"""Compatibility wrapper: the code lives in ``egogarment.io.trajio`` (src/egogarment/io/trajio.py).

Kept so that ``import trajio`` (with scripts/ on sys.path) works without installing the package; after
``pip install -e .`` use ``from egogarment.io import trajio``. Library only (no command line).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.io.trajio import *  # noqa: E402,F401,F403
