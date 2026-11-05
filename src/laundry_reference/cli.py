"""Console-script entry points (``[project.scripts]`` in pyproject.toml names these functions; keep their names).

Each wraps the module's ``main()``, which returns the exit status (0) as setuptools' ``sys.exit(func())`` expects. Behaviour and
printed output are identical to ``python -m laundry_reference.<module>``.
"""
from __future__ import annotations


def gap_fill() -> int:
    from .gap_fill import main
    return main()


def relative_se3() -> int:
    from .relative_se3 import main
    return main()


def robotize() -> int:
    from .robotize import main
    return main()


def parity_metrics() -> int:
    from .parity_metrics import main
    return main()


def plot_example() -> int:
    from .plot_example import main
    return main()
