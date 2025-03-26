import numpy as np
from typing import Dict, List
from core_engine.latent_state import LatentState
from core_engine.base_model import build_model
from core_engine.inference_engine_svi import run_svi
from core_engine.match_data_preprocessor import preprocess_match
from core_engine.data_match.match_data import MatchData


def bayesian_update(
    latent_states: Dict[str, List[LatentState]],
    match_data: MatchData,
    weights: Dict[str, Dict[str, np.ndarray]],
    model_config: Dict,
) -> Dict[str, List[LatentState]]:
    player_ids = list(latent_states.keys())
    player_id_to_idx = {pid: idx for idx, pid in enumerate(player_ids)}

    latent_dim = model_config["latent_dim"]
    num_players = len(player_ids)

    prior_means = np.zeros((num_players, latent_dim))
    prior_stds = np.ones((num_players, latent_dim))
    for pid, idx in player_id_to_idx.items():
        for dim, state in enumerate(latent_states[pid]):
            prior_means[idx, dim] = state.mean[dim]
            prior_stds[idx, dim] = state.std[dim]

    generic_logits, outcomes, player_idxs, weight_vectors = preprocess_match(
        match_data, player_id_to_idx, latent_states, weights, latent_dim
    )
    data = {
        "generic_logits": generic_logits,
        "player_idxs": player_idxs,
        "weight_vectors": weight_vectors,
        "outcomes": outcomes,
    }

    model = build_model(prior_means, prior_stds)
    posterior_mean, posterior_std, _ = run_svi(model, data)

    updated = {}
    for pid, idx in player_id_to_idx.items():
        if pid == "dummy":
            continue
        updated[pid] = [
            LatentState(
                mean=posterior_mean[idx],
                std=posterior_std[idx],
                last_updated=match_data.date,
            )
        ]

    return updated
