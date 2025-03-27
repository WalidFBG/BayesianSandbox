from core_engine.match_csv_loader import load_matches_from_csv
from core_engine.latent_state import LatentState
from core_engine.pre_match_updater import inflate_variance
from core_engine.post_match_updater import bayesian_update
from core_engine.match_data_preprocessor import preprocess_match

import numpy as np
from datetime import datetime

# --- Config ---
csv_path = "data/mlb_2019_may.csv"

training_config = {
    "latent_dim": 2,
    "inflation_rate": 0.05,
    "observation_role_config": {
        "IsSteal": ["ManOnBase1", "ManOnBase2", "ManOnBase3"],
    },
}

# --- Initialization ---
latent_dim = training_config["latent_dim"]
player_latent_states = {
    "dummy": LatentState(
        mean=np.zeros(latent_dim),
        std=np.ones(latent_dim),
        last_updated=datetime(1970, 1, 1),
    )
}

weights = {
    "IsSteal": {
        "ManOnBase1": np.array([0.9, 0.1]),
        "ManOnBase2": np.array([0.25, 0.1]),
        "ManOnBase3": np.array([0.05, 0.1]),
    },
}

# --- Load matches ---
matches = load_matches_from_csv(csv_path, training_config["observation_role_config"])

# --- Process each match ---
inflation_rate = training_config["inflation_rate"]

for match in matches:
    players_in_match = set()
    for obs in match.observations:
        for pid in obs.roles.values():
            players_in_match.add(pid)
            if pid not in player_latent_states:
                player_latent_states[pid] = LatentState(
                    mean=np.zeros(latent_dim),
                    std=np.ones(latent_dim),
                    last_updated=match.date,
                )

    match_latent_states = {
        pid: player_latent_states[pid] for pid in players_in_match.union({"dummy"})
    }

    pre_match_states = inflate_variance(match_latent_states, match.date, inflation_rate)

    player_id_to_idx = {
        pid: idx for idx, pid in enumerate(list(pre_match_states.keys()))
    }

    generic_logits, outcomes, player_idxs, weight_vectors = preprocess_match(
        match,
        player_id_to_idx,
        pre_match_states,
        weights,
        latent_dim,
    )

    updated_states = bayesian_update(
        latent_states=pre_match_states,
        match_date=match.date,
        model_config={"latent_dim": latent_dim},
        generic_logits=generic_logits,
        outcomes=outcomes,
        player_idxs=player_idxs,
        weight_vectors=weight_vectors,
    )

    for pid, state in updated_states.items():
        player_latent_states[pid] = state
