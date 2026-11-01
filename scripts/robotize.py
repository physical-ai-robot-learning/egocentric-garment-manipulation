"""Compatibility wrapper: the code lives in ``egogarment.retarget.robotize`` (src/egogarment/retarget/robotize.py).

Kept so that ``python scripts/robotize.py ...`` and ``import robotize`` (with scripts/ on sys.path) work
without installing the package. Equivalent: ``python -m egogarment.retarget.robotize ...`` after ``pip install -e .``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.retarget.robotize import *  # noqa: E402,F401,F403
from egogarment.retarget.robotize import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
