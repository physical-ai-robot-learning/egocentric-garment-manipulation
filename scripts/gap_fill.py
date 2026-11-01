"""Compatibility wrapper: the code lives in ``egogarment.pose.gap_fill`` (src/egogarment/pose/gap_fill.py).

Kept so that ``python scripts/gap_fill.py ...`` and ``import gap_fill`` (with scripts/ on sys.path) work
without installing the package. Equivalent: ``python -m egogarment.pose.gap_fill ...`` after ``pip install -e .``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.pose.gap_fill import *  # noqa: E402,F401,F403
from egogarment.pose.gap_fill import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
