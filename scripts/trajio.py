"""Compatibility wrapper: the code lives in ``egogarment.io.trajio`` (src/egogarment/io/trajio.py).

Kept so that ``python scripts/trajio.py ...`` and ``import trajio`` (with scripts/ on sys.path) work
without installing the package. Equivalent: ``python -m egogarment.io.trajio ...`` after ``pip install -e .``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.io.trajio import *  # noqa: E402,F401,F403
