from datetime import datetime
from typing import Dict
from core_engine.latent_state import LatentState
import numpy as np


def inflate_variance(
    latent_states: Dict[str, LatentState],
    match_date: datetime,
    inflation_rate: float,
) -> Dict[str, LatentState]:
    updated = {}
    for player_id, state in latent_states.items():
        days_passed = (match_date - state.last_updated).days
        inflation = np.sqrt(1 + inflation_rate * days_passed)
        new_std = state.std * inflation
        updated[player_id] = LatentState(
            mean=state.mean, std=new_std, last_updated=state.last_updated
        )
    return updated
