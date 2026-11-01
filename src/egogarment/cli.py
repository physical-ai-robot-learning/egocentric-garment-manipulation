"""Console-script entry points (see ``[project.scripts]`` in pyproject.toml).

Each module's ``main()`` returns its report (dict / float) for programmatic use. setuptools wraps an entry point
as ``sys.exit(func())``, which would treat that return value as an error status, so these thin functions call
``main()`` and discard the return value. Behaviour and printed output are otherwise identical to
``python -m egogarment.<component>.<module>``.
"""
from __future__ import annotations


def gap_fill() -> None:
    from egogarment.pose.gap_fill import main
    main()


def relative_se3() -> None:
    from egogarment.pose.relative_se3 import main
    main()


def robotize() -> None:
    from egogarment.retarget.robotize import main
    main()


def parity_metrics() -> None:
    from egogarment.evaluation.parity_metrics import main
    main()


def plot_example() -> None:
    from egogarment.viz.plot_example import main
    main()
