"""Compatibility wrapper: the code lives in ``egogarment.pose.relative_se3`` (src/egogarment/pose/relative_se3.py).

Kept so that ``python scripts/relative_se3.py ...`` and ``import relative_se3`` (with scripts/ on sys.path) work
without installing the package. Equivalent: ``python -m egogarment.pose.relative_se3 ...`` after ``pip install -e .``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.pose.relative_se3 import *  # noqa: E402,F401,F403
from egogarment.pose.relative_se3 import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
