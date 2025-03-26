from datetime import datetime
from typing import Dict, List
from core_engine.latent_state import LatentState
import numpy as np


def inflate_variance(
    latent_states: Dict[str, List[LatentState]],
    match_date: datetime,
    inflation_rate: float,
) -> Dict[str, List[LatentState]]:
    updated = {}
    for player_id, state_list in latent_states.items():
        new_state_list = []
        for state in state_list:
            days_passed = (match_date - state.last_updated).days
            inflation = np.sqrt(1 + inflation_rate * days_passed)
            new_std = state.std * inflation
            new_state_list.append(
                LatentState(
                    mean=state.mean, std=new_std, last_updated=state.last_updated
                )
            )
        updated[player_id] = new_state_list
    return updated
