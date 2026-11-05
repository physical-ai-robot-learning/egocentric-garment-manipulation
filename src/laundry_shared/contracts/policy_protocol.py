"""Low-level policy API.  The execution loop sees ONLY this protocol, so a real Multi-Task DiT, MockPolicy, a recorded
action player or any future policy are interchangeable without touching deployment/robot_loop.py.

Agent B's deliverable for deployment is a factory `make_policy(**kwargs) -> Policy` importable as
`package.module:make_policy` (see deployment/policy_loader.py).
"""
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .observation_contract import Observation
from .skill_contract import Skill


@dataclass
class PolicyInput:
    observation: Observation          # images under CAMERA_KEYS + base-frame TCP poses
    proprio: np.ndarray               # (PROPRIO_DIM,) built by observation_contract.proprio_vector
    skill: Skill                      # one of POLICY_SKILLS
    instruction: str                  # SKILL_INSTRUCTION[skill] unless overridden
    episode_id: str = ""
    call_index: int = 0


@dataclass
class PolicyOutput:
    action_chunk: np.ndarray          # (H, ACTION_DIM) relative to the observed TCPs (action_contract.py)
    checkpoint_id: str
    confidence: Optional[float] = None
    completion_candidate: bool = False  # policy-side hint that the skill may be finished -> orchestrator check
    diagnostics: dict = field(default_factory=dict)  # free-form; "timing_ms": {stage: ms} is aggregated by benchmarks


@runtime_checkable
class Policy(Protocol):
    checkpoint_id: str
    horizon: int

    def reset(self) -> None:
        """called at episode start and on every skill switch"""

    def predict(self, inp: PolicyInput) -> PolicyOutput:
        ...
