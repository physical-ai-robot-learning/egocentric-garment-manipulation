"""Make the src/ packages (egogarment, visual_teleop, rebot_prep) importable from a plain checkout,
without ``pip install -e .``."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
