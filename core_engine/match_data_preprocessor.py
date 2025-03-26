from typing import Dict, Tuple
import numpy as np
from core_engine.data_match.match_data import MatchData
from core_engine.latent_state import LatentState


def preprocess_match(
    match_data: MatchData,
    player_id_to_idx: Dict[str, int],
    latent_states: Dict[str, LatentState],
    weights: Dict[str, Dict[str, np.ndarray]],
    latent_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generic_logits = []
    outcomes = []
    player_idxs = []
    weight_vectors = []

    max_roles = max(len(obs.roles) for obs in match_data.observations)

    for obs in match_data.observations:
        obs_type = obs.obs_type
        roles = obs.roles

        assert obs_type in weights, f"Weights missing for observation type '{obs_type}'"

        role_weights = weights[obs_type]
        role_ids = sorted(roles.keys())

        row_player_idxs = []
        row_weight_vectors = []

        for role in role_ids:
            pid = roles[role]
            assert role in role_weights, (
                f"Missing weight for role '{role}' in '{obs_type}'"
            )
            assert pid in player_id_to_idx, f"Unknown player ID '{pid}'"
            assert pid in latent_states, f"Missing latent state for player '{pid}'"

            row_player_idxs.append(player_id_to_idx[pid])
            row_weight_vectors.append(role_weights[role])

        # Pad with dummy player (index 0) and zero weights
        while len(row_player_idxs) < max_roles:
            row_player_idxs.append(0)
            row_weight_vectors.append(np.zeros(latent_dim))

        generic_logits.append(obs.base_logit)
        outcomes.append(obs.outcome)
        player_idxs.append(row_player_idxs)
        weight_vectors.append(row_weight_vectors)

    generic_logits = np.array(generic_logits)
    outcomes = np.array(outcomes)
    player_idxs = np.array(player_idxs, dtype=int)
    weight_vectors = np.array(weight_vectors)

    return generic_logits, outcomes, player_idxs, weight_vectors
