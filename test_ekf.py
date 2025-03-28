from numba import njit
import numpy as np
import time
from data_match.match_csv_loader import load_matches_from_csv


def create_obs_role_maps(observation_role_config):
    obs_type_map = {}
    obs_role_maps = {}
    next_type_idx = 0

    for obs_type, role_list in observation_role_config.items():
        obs_type_map[obs_type] = next_type_idx
        next_type_idx += 1

        role_map = {}
        for i, role_name in enumerate(role_list):
            role_map[role_name] = i
        obs_role_maps[obs_type] = role_map

    return obs_type_map, obs_role_maps


def create_weights_array(
    weights_dict,
    obs_type_map,
    obs_role_maps,
    latent_dim,
):
    n_obs_types = len(obs_type_map)
    max_roles = max(len(r_map) for r_map in obs_role_maps.values())

    weights_array = np.zeros((n_obs_types, max_roles, latent_dim), dtype=np.float64)

    for obs_type, subdict in weights_dict.items():
        ot_idx = obs_type_map[obs_type]
        role_map = obs_role_maps[obs_type]
        for role_name, vector in subdict.items():
            r_idx = role_map[role_name]
            weights_array[ot_idx, r_idx, :] = vector

    return weights_array


def initialize_player_latents_and_variances(matches, latent_dim):
    player_ids = set()
    for match in matches:
        for obs in match.observations:
            for player_id in obs.roles.values():
                player_ids.add(player_id)

    player_ids_list = sorted(player_ids)
    player_idx_map = {pid: i for i, pid in enumerate(player_ids_list)}

    num_players = len(player_ids_list)
    player_latent_states = np.zeros((num_players, latent_dim), dtype=np.float64)
    player_variances = np.zeros((num_players, latent_dim), dtype=np.float64)
    player_last_update = np.array([None] * num_players, dtype=object)

    return (player_latent_states, player_variances, player_last_update, player_idx_map)


def preprocess_single_match(match, obs_type_map, obs_role_maps, player_idx_map):
    observations = match.observations
    num_obs = len(observations)
    base_estimates = np.empty(num_obs, dtype=np.float64)
    outcomes = np.empty(num_obs, dtype=np.int8)
    roles_arrays = []

    for i, obs in enumerate(observations):
        base_estimates[i] = obs.base_logit
        outcomes[i] = obs.outcome

        obs_type_idx = obs_type_map[obs.obs_type]
        role_map = obs_role_maps[obs.obs_type]

        row_data = []
        for role_name, player_id in obs.roles.items():
            r_idx = role_map[role_name]
            p_idx = player_idx_map[player_id]
            row_data.append([obs_type_idx, r_idx, p_idx])

        arr = np.array(row_data, dtype=np.int64).reshape(-1, 3)
        roles_arrays.append(arr)

    return {
        "base_estimates": base_estimates,
        "outcomes": outcomes,
        "roles_per_obs": roles_arrays,
        "unique_players": {
            player_id for obs in observations for player_id in obs.roles.values()
        },
        "match_date": match.date,
    }


def preprocess_all_matches(matches, obs_type_map, obs_role_maps, player_idx_map):
    match_preproc_list = []
    for match in matches:
        match_preproc = preprocess_single_match(
            match, obs_type_map, obs_role_maps, player_idx_map
        )
        match_preproc_list.append(match_preproc)
    return match_preproc_list


def update_player_variances_for_match(
    match, player_variances, player_last_update, player_idx_map, process_noise=0.05
):
    unique_players = match["unique_players"]
    match_date = match["match_date"]

    for pid in unique_players:
        idx = player_idx_map[pid]
        if player_last_update[idx] is None:
            player_variances[idx].fill(1.0)
        else:
            days_since = (match_date - player_last_update[idx]).days
            player_variances[idx, :] += process_noise * days_since

        player_last_update[idx] = match_date


def compute_likelihood_for_match(
    match_preproc, player_latent_states, weights, epsilon=1e-12
):
    base_estimates = match_preproc["base_estimates"]
    outcomes = match_preproc["outcomes"]
    roles_per_obs = match_preproc["roles_per_obs"]

    logits = base_estimates.copy()

    num_obs = len(base_estimates)
    for i in range(num_obs):
        obs_roles = roles_per_obs[i]
        dot_sum = 0.0
        for obs_type, role_name, player_index in obs_roles:
            w = weights[obs_type][role_name]
            lat = player_latent_states[player_index]
            dot_sum += np.dot(lat, w)

        logits[i] += dot_sum

    probs = 1.0 / (1.0 + np.exp(-logits))
    ll_obs = outcomes * np.log(probs + epsilon) + (1 - outcomes) * np.log(
        1.0 - probs + epsilon
    )
    return -ll_obs.sum()


@njit
def run_ekf_for_match_jit(
    base_estimates,
    outcomes,
    roles_per_obs,
    player_latent_states,
    player_variances,
    weights_array,
    measurement_noise,
):
    num_obs = len(base_estimates)

    for i in range(num_obs):
        obs_outcome = outcomes[i]
        obs_base = base_estimates[i]
        obs_roles = roles_per_obs[i]

        if len(obs_roles) == 0:
            continue

        R = len(obs_roles)
        d = player_latent_states.shape[1]

        latents_matrix = np.empty((R, d), dtype=np.float64)
        var_matrix = np.empty((R, d), dtype=np.float64)
        weight_matrix = np.empty((R, d), dtype=np.float64)
        players_in_obs = np.empty(R, dtype=np.int64)

        for r in range(R):
            obs_type_idx, role_idx, p_idx = obs_roles[r]
            latents_matrix[r, :] = player_latent_states[p_idx, :]
            var_matrix[r, :] = player_variances[p_idx, :]
            weight_matrix[r, :] = weights_array[obs_type_idx, role_idx, :]
            players_in_obs[r] = p_idx

        contrib = np.sum(weight_matrix * latents_matrix)
        logit = obs_base + contrib
        p = 1.0 / (1.0 + np.exp(-logit))
        residual = obs_outcome - p
        sigmoid_derivative = p * (1.0 - p)
        H = sigmoid_derivative * weight_matrix
        S = np.sum(H * var_matrix * H) + measurement_noise
        K = (var_matrix * H) / S

        x_updated = latents_matrix + (K * residual)
        var_updated = var_matrix - (K * H * var_matrix)

        for r in range(R):
            idx = players_in_obs[r]
            player_latent_states[idx, :] = x_updated[r, :]
            player_variances[idx, :] = var_updated[r, :]


def run_ekf_for_match(
    match_preproc,
    player_latent_states,
    player_variances,
    measurement_noise,
    weights_array,
):
    base_estimates = match_preproc["base_estimates"]
    outcomes = match_preproc["outcomes"]
    roles_per_obs = match_preproc["roles_per_obs"]

    run_ekf_for_match_jit(
        base_estimates,
        outcomes,
        roles_per_obs,
        player_latent_states,
        player_variances,
        weights_array,
        measurement_noise,
    )


def run_match_by_match_likelihood(
    match_preproc_list,
    player_latent_states,
    player_variances,
    player_last_update,
    player_idx_map,
    weights_array,
    process_noise=0.001,
    measurement_noise=0.005,
):
    total_ll = 0.0

    for match_preproc in match_preproc_list:
        update_player_variances_for_match(
            match_preproc,
            player_variances,
            player_last_update,
            player_idx_map,
            process_noise=process_noise,
        )

        match_ll = compute_likelihood_for_match(
            match_preproc, player_latent_states, weights_array
        )
        total_ll += match_ll

        run_ekf_for_match(
            match_preproc,
            player_latent_states,
            player_variances,
            measurement_noise,
            weights_array,
        )

    return total_ll


# ----- Config
training_config = {
    "latent_dim": 1,
    "num_epochs": 10,
    "observation_role_config": {
        "IsSteal": ["ManOnBase1", "ManOnBase2", "ManOnBase3"],
    },
}

weights = {
    "IsSteal": {
        "ManOnBase1": np.array([0.267]),
        "ManOnBase2": np.array([0.083]),
        "ManOnBase3": np.array([0.012]),
    },
}
# ----- Load Matches
matches = load_matches_from_csv(
    "data/mlb_2019.csv", training_config["observation_role_config"]
)
matches.sort(key=lambda m: m.date)
print(f"Loaded {len(matches)} matches.")

# ----- Pre process data
obs_type_map, obs_role_maps = create_obs_role_maps(
    training_config["observation_role_config"]
)

weights_array = create_weights_array(
    weights, obs_type_map, obs_role_maps, training_config["latent_dim"]
)

player_latent_states, player_variances, player_last_update, player_idx_map = (
    initialize_player_latents_and_variances(matches, training_config["latent_dim"])
)

matches_preprocessed = preprocess_all_matches(
    matches, obs_type_map, obs_role_maps, player_idx_map
)

# ----- Run iterations
original_latent_states = player_latent_states.copy()
original_variances = player_variances.copy()
original_last_update = player_last_update.copy()

num_epochs = training_config["num_epochs"]
for epoch in range(num_epochs):
    start_time = time.perf_counter()
    total_ll = run_match_by_match_likelihood(
        matches_preprocessed,
        player_latent_states,
        player_variances,
        player_last_update,
        player_idx_map,
        weights_array,
    )
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    print(f"Epoch {epoch}, likelihood = {total_ll:.4f}, elapsed = {elapsed_ms:.2f} ms")

    if epoch < num_epochs - 1:
        player_latent_states[:] = original_latent_states
        player_variances[:] = original_variances
        player_last_update[:] = original_last_update
