"""Compatibility wrapper: the code lives in ``egogarment.evaluation.parity_metrics`` (src/egogarment/evaluation/parity_metrics.py).

Kept so that ``python scripts/parity_metrics.py ...`` and ``import parity_metrics`` (with scripts/ on sys.path) work
without installing the package. Equivalent: ``python -m egogarment.evaluation.parity_metrics ...`` after ``pip install -e .``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.evaluation.parity_metrics import *  # noqa: E402,F401,F403
from egogarment.evaluation.parity_metrics import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
