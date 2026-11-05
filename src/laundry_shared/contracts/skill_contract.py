"""Skill and garment-state vocabulary shared by the orchestrator, the state machine and the low-level policy.

There is deliberately NO recovery skill: a failed skill is handled by re-assessing the garment state and
transitioning back to an earlier skill (see orchestrator/state_machine.py, the ONLY place that maps
garment state -> skill).
"""
from enum import Enum


class Skill(str, Enum):
    UNFOLD_ALIGN = "UNFOLD_ALIGN"
    FOLD = "FOLD"
    STACK = "STACK"
    DONE = "DONE"


class GarmentState(str, Enum):
    CRUMPLED = "CRUMPLED"
    PARTIAL_FLAT = "PARTIAL_FLAT"
    FLAT_ALIGNED = "FLAT_ALIGNED"
    PARTIAL_FOLD = "PARTIAL_FOLD"
    FOLDED = "FOLDED"
    STACKED = "STACKED"
    UNKNOWN = "UNKNOWN"


# Skills that the low-level policy executes.  DONE is terminal and has no policy.
POLICY_SKILLS = (Skill.UNFOLD_ALIGN, Skill.FOLD, Skill.STACK)

# Short instruction passed with the skill id to the policy (Multi-Task DiT conditions on a frozen CLIP text
# embedding of this string).  Agent B trains with exactly these strings; change them only here.
SKILL_INSTRUCTION = {
    Skill.UNFOLD_ALIGN: "unfold and align the shirt",
    Skill.FOLD: "fold the shirt",
    Skill.STACK: "place the folded shirt on the stack",
}

# Integer ids for datasets / logs (stable; never reorder, only append).
SKILL_INDEX = {Skill.UNFOLD_ALIGN: 0, Skill.FOLD: 1, Skill.STACK: 2, Skill.DONE: 3}
