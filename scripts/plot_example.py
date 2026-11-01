"""Compatibility wrapper: the code lives in ``egogarment.viz.plot_example`` (src/egogarment/viz/plot_example.py).

Kept so that ``python scripts/plot_example.py ...`` and ``import plot_example`` (with scripts/ on sys.path) work
without installing the package. Equivalent: ``python -m egogarment.viz.plot_example ...`` after ``pip install -e .``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egogarment.viz.plot_example import *  # noqa: E402,F401,F403
from egogarment.viz.plot_example import main  # noqa: E402,F401

if __name__ == "__main__":
    main()
