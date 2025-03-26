from core_engine.match_csv_loader import load_matches_from_csv
from core_engine.latent_state import LatentState
from core_engine.pre_match_updater import inflate_variance
from core_engine.post_match_updater import bayesian_update

import numpy as np
from datetime import datetime

# --- Config ---
csv_path = "data/mlb_2019_may.csv"
observation_role_config = {
    "IsSteal": ["ManOnBase1", "ManOnBase2", "ManOnBase3"],
}

latent_dim = 1
starting_variance = 1.0
inflation_rate = 0.05

# --- Initialization ---
player_latent_states = {
    "dummy": [
        LatentState(
            mean=np.zeros(latent_dim),
            std=np.ones(latent_dim),
            last_updated=datetime(1970, 1, 1),
        )
    ]
}

weights = {
    "IsSteal": {
        "ManOnBase1": np.array([0.9]),
        "ManOnBase2": np.array([0.25]),
        "ManOnBase3": np.array([0.05]),
    },
}

# --- Load matches ---
matches = load_matches_from_csv(csv_path, observation_role_config)

# --- Process each match ---
for match in matches:
    players_in_match = set()
    for obs in match.observations:
        for pid in obs.roles.values():
            players_in_match.add(pid)
            if pid not in player_latent_states:
                player_latent_states[pid] = [
                    LatentState(
                        mean=np.zeros(latent_dim),
                        std=np.ones(latent_dim) * starting_variance,
                        last_updated=match.date,
                    )
                ]

    match_latent_states = {
        pid: player_latent_states[pid] for pid in players_in_match.union({"dummy"})
    }
    pre_match_states = inflate_variance(match_latent_states, match.date, inflation_rate)

    updated_states = bayesian_update(
        latent_states=pre_match_states,
        match_data=match,
        weights=weights,
        model_config={"latent_dim": latent_dim},
    )

    for pid, state in updated_states.items():
        player_latent_states[pid] = state
