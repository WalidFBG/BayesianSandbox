import numpy as np
from datetime import datetime
import copy
from typing import Dict, Tuple

from core_engine.match_csv_loader import load_matches_from_csv
from core_engine.latent_state import LatentState
from core_engine.pre_match_updater import inflate_variance
from core_engine.post_match_updater import bayesian_update
from core_engine.data_match.match_data import MatchData
from core_engine.match_data_preprocessor import preprocess_match


def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def calculate_gradients_and_loglik(
    match: MatchData,
    pre_match_states: Dict[str, LatentState],
    current_weights: Dict[str, Dict[str, np.ndarray]],
    latent_dim: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], float]:
    weight_gradients = {
        obs_type: {role: np.zeros(latent_dim) for role in roles_weights}
        for obs_type, roles_weights in current_weights.items()
    }
    match_log_likelihood = 0.0

    for obs in match.observations:
        obs_type = obs.obs_type
        outcome = obs.outcome
        latent_vectors_involved = {}

        logit = obs.base_logit

        for role, pid in obs.roles.items():
            w = current_weights[obs_type][role]
            latent_means = pre_match_states[pid].mean
            logit += np.dot(w, latent_means)
            latent_vectors_involved[role] = latent_means

        probability = sigmoid(logit)

        epsilon = 1e-9
        log_lik_contrib = outcome * np.log(probability + epsilon) + (
            1 - outcome
        ) * np.log(1 - probability + epsilon)

        match_log_likelihood += log_lik_contrib

        gradient_multiplier = outcome - probability

        for role, latent_vector in latent_vectors_involved.items():
            grad_w = gradient_multiplier * latent_vector
            weight_gradients[obs_type][role] += grad_w

    return weight_gradients, match_log_likelihood


def train_weights(
    matches, training_config, num_epochs=10, learning_rate=0.001, prior_variance=10.0
):
    latent_dim = training_config["latent_dim"]
    inflation_rate = training_config["inflation_rate"]
    observation_role_config = training_config["observation_role_config"]

    # --- Initialize Weights ---
    weights = {}
    for obs_type, roles in observation_role_config.items():
        weights[obs_type] = {}
        for role in roles:
            weights[obs_type][role] = np.random.randn(latent_dim) * 0.01

    # --- Training Loop ---
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        total_epoch_log_likelihood = 0.0

        player_latent_states = {
            "dummy": LatentState(
                mean=np.zeros(latent_dim),
                std=np.ones(latent_dim),
                last_updated=datetime(1970, 1, 1),
            )
        }

        current_player_states = copy.deepcopy(player_latent_states)

        for i, match in enumerate(matches):
            if i % 100 == 0 and i > 0:
                print(f"  Processed {i}/{len(matches)} matches...")

            players_in_match = set()
            for obs in match.observations:
                for pid in obs.roles.values():
                    players_in_match.add(pid)
                    if pid not in current_player_states:
                        current_player_states[pid] = LatentState(
                            mean=np.zeros(latent_dim),
                            std=np.ones(latent_dim),
                            last_updated=match.date,
                        )

            match_participants_states = {
                pid: current_player_states[pid]
                for pid in players_in_match.union({"dummy"})
                if pid in current_player_states
            }

            pre_match_states = inflate_variance(
                copy.deepcopy(match_participants_states), match.date, inflation_rate
            )

            local_player_ids = list(pre_match_states.keys())
            player_id_to_idx = {pid: idx for idx, pid in enumerate(local_player_ids)}

            weight_gradients, match_log_likelihood = calculate_gradients_and_loglik(
                match, pre_match_states, weights, latent_dim
            )
            total_epoch_log_likelihood += match_log_likelihood

            for obs_type, roles_grads in weight_gradients.items():
                for role, grad in roles_grads.items():
                    w = weights[obs_type][role]
                    prior_grad = -w / prior_variance
                    weights[obs_type][role] += learning_rate * (grad + prior_grad)

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
                current_player_states[pid] = state

        print(
            f"Epoch {epoch + 1} finished. Total Log Likelihood: {total_epoch_log_likelihood:.2f}"
        )
        # TODO: Implement learning rate decay
        # learning_rate *= 0.95 # Example decay

    print("\nTraining complete.")
    print("Final Learned Weights:")
    for obs_type, roles_weights in weights.items():
        print(f"  {obs_type}:")
        for role, w in roles_weights.items():
            print(f"    {role}: {np.round(w, 3)}")

    return weights, current_player_states


# --- Config ---
training_config = {
    "latent_dim": 1,
    "inflation_rate": 0.05,
    "observation_role_config": {
        "IsSteal": ["ManOnBase1", "ManOnBase2", "ManOnBase3"],
    },
}

# --- Load matches ---
matches = load_matches_from_csv(
    "data/mlb_2019_may.csv", training_config["observation_role_config"]
)
matches.sort(key=lambda m: m.date)
print(f"Loaded {len(matches)} matches.")

# --- Run Training ---
learned_weights, final_states = train_weights(
    matches=matches,
    training_config=training_config,
    num_epochs=5,
    learning_rate=0.05,
    prior_variance=1000.0,  # regularization
)
