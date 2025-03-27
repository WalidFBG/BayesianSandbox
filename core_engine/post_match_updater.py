import numpy as np
from datetime import datetime
from typing import Dict, List
from core_engine.latent_state import LatentState
from core_engine.base_model import build_model
from core_engine.inference_engine_mc import run_mcmc


def bayesian_update(
    latent_states: Dict[str, LatentState],
    match_date: datetime,
    model_config: Dict,
    generic_logits: np.ndarray,
    outcomes: np.ndarray,
    player_idxs: np.ndarray,
    weight_vectors: np.ndarray,
) -> Dict[str, List[LatentState]]:
    player_ids = list(latent_states.keys())
    player_id_to_idx = {pid: idx for idx, pid in enumerate(player_ids)}

    latent_dim = model_config["latent_dim"]
    num_players = len(player_ids)

    prior_means = np.zeros((num_players, latent_dim))
    prior_stds = np.ones((num_players, latent_dim))
    for pid, idx in player_id_to_idx.items():
        prior_means[idx] = latent_states[pid].mean
        prior_stds[idx] = latent_states[pid].std

    data = {
        "generic_logits": generic_logits,
        "player_idxs": player_idxs,
        "weight_vectors": weight_vectors,
        "outcomes": outcomes,
    }

    model = build_model(prior_means, prior_stds)
    posterior_mean, posterior_std, _ = run_mcmc(model, data)

    updated = {}
    for pid, idx in player_id_to_idx.items():
        if pid == "dummy":
            continue
        updated[pid] = LatentState(
            mean=posterior_mean[idx],
            std=posterior_std[idx],
            last_updated=match_date,
        )

    return updated
