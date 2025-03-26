from dataclasses import dataclass
from typing import Dict


@dataclass
class Observation:
    obs_type: str
    roles: Dict[str, str]
    base_logit: float
    outcome: int
