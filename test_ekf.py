from numba import njit
import numpy as np
from data_match.match_csv_loader import load_matches_from_csv
from scipy.optimize import minimize


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


def create_weights_array(weights_dict, obs_type_map, obs_role_maps, latent_dim):
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
    player_last_update = np.zeros(num_players, dtype=np.int64)

    return player_latent_states, player_variances, player_last_update, player_idx_map


def preprocess_flat_roles(matches, obs_type_map, obs_role_maps, player_idx_map):
    total_obs = sum(len(m.observations) for m in matches)
    total_roles = sum(len(obs.roles) for m in matches for obs in m.observations)

    base_estimates = np.empty(total_obs, dtype=np.float64)
    outcomes = np.empty(total_obs, dtype=np.int8)
    match_dates = np.empty(total_obs, dtype=np.int64)
    role_ptrs = np.empty((total_obs, 2), dtype=np.int64)
    roles_data = np.empty((total_roles, 3), dtype=np.int64)

    current_obs = 0
    current_role_idx = 0
    match_obs_ranges = []

    for match in matches:
        match_start = current_obs
        match_date_int = np.datetime64(match.date, "D").astype(np.int64)

        for obs in match.observations:
            base_estimates[current_obs] = obs.base_logit
            outcomes[current_obs] = obs.outcome
            match_dates[current_obs] = match_date_int

            obs_type_idx = obs_type_map[obs.obs_type]
            role_map = obs_role_maps[obs.obs_type]

            roles = list(obs.roles.items())
            role_len = len(roles)

            for role_name, player_id in roles:
                role_idx = role_map[role_name]
                player_idx = player_idx_map[player_id]
                roles_data[current_role_idx] = [obs_type_idx, role_idx, player_idx]
                current_role_idx += 1

            role_ptrs[current_obs] = [current_role_idx - role_len, role_len]
            current_obs += 1

        match_end = current_obs
        unique_players = {
            player_idx_map[pid]
            for obs in match.observations
            for pid in obs.roles.values()
        }
        match_obs_ranges.append(
            (match_start, match_end, match_date_int, unique_players)
        )

    return (
        base_estimates,
        outcomes,
        role_ptrs,
        roles_data,
        match_dates,
        match_obs_ranges,
    )


@njit
def compute_likelihood(
    base_estimates,
    outcomes,
    role_ptrs,
    roles_data,
    player_latents,
    weights_array,
    epsilon=1e-12,
):
    n_obs = base_estimates.shape[0]
    total_ll = 0.0

    for i in range(n_obs):
        logit = base_estimates[i]
        start_idx, length = role_ptrs[i]
        for j in range(start_idx, start_idx + length):
            obs_type, role_idx, player_idx = roles_data[j]
            w = weights_array[obs_type, role_idx]
            lat = player_latents[player_idx]
            logit += np.dot(w, lat)

        p = 1.0 / (1.0 + np.exp(-logit))
        y = outcomes[i]
        total_ll += -(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))

    return total_ll


@njit
def run_ekf(
    base_estimates,
    outcomes,
    role_ptrs,
    roles_data,
    player_latents,
    player_vars,
    weights_array,
    measurement_noise,
):
    n_obs = base_estimates.shape[0]
    latent_dim = player_latents.shape[1]

    for i in range(n_obs):
        logit = base_estimates[i]
        y = outcomes[i]
        start_idx, length = role_ptrs[i]

        if length == 0:
            continue

        R = length
        latents_matrix = np.empty((R, latent_dim))
        var_matrix = np.empty((R, latent_dim))
        weight_matrix = np.empty((R, latent_dim))
        players = np.empty(R, dtype=np.int64)

        for r in range(R):
            obs_type, role_idx, player_idx = roles_data[start_idx + r]
            latents_matrix[r] = player_latents[player_idx]
            var_matrix[r] = player_vars[player_idx]
            weight_matrix[r] = weights_array[obs_type, role_idx]
            players[r] = player_idx

        contrib = np.sum(weight_matrix * latents_matrix)
        logit += contrib
        p = 1.0 / (1.0 + np.exp(-logit))
        residual = y - p
        H = p * (1 - p) * weight_matrix
        S = np.sum(H * var_matrix * H) + measurement_noise
        K = (var_matrix * H) / S
        updated_latents = latents_matrix + K * residual
        updated_vars = var_matrix - K * H * var_matrix

        for r in range(R):
            player_latents[players[r]] = updated_latents[r]
            player_vars[players[r]] = updated_vars[r]


# ----- Config
training_config = {
    "latent_dim": 1,
    "num_epochs": 10,
    "observation_role_config": {
        "IsSteal": ["ManOnBase1", "ManOnBase2", "ManOnBase3"],
    },
}

# ----- Load Matches
matches = load_matches_from_csv(
    "data/mlb_2019.csv", training_config["observation_role_config"]
)
matches.sort(key=lambda m: m.date)
print(f"Loaded {len(matches)} matches.")

# ----- Setup
obs_type_map, obs_role_maps = create_obs_role_maps(
    training_config["observation_role_config"]
)
player_latent_states, player_variances, player_last_update, player_idx_map = (
    initialize_player_latents_and_variances(matches, training_config["latent_dim"])
)

base_estimates, outcomes, role_ptrs, roles_data, match_dates, match_obs_ranges = (
    preprocess_flat_roles(matches, obs_type_map, obs_role_maps, player_idx_map)
)

# ----- Optimize
flat_weights_init = np.array([0.0, 0.0, 0.0])
init_process_noise = 0.001
init_measurement_noise = 0.005


def unpack_params(params):
    weights = {
        "IsSteal": {
            "ManOnBase1": np.array([params[0]]),
            "ManOnBase2": np.array([params[1]]),
            "ManOnBase3": np.array([params[2]]),
        }
    }
    process_noise = max(params[3], 1e-6)
    measurement_noise = max(params[4], 1e-6)
    return weights, process_noise, measurement_noise


def objective(params):
    weights, process_noise, measurement_noise = unpack_params(params)
    weights_array = create_weights_array(weights, obs_type_map, obs_role_maps, 1)

    latents = player_latent_states.copy()
    variances = player_variances.copy()
    last_update = player_last_update.copy()

    total_ll = 0.0

    for match_start, match_end, match_date, unique_players in match_obs_ranges:
        for pid in unique_players:
            last = last_update[pid]
            if last == 0:
                variances[pid].fill(1.0)
            else:
                days_since = match_date - last
                variances[pid] += process_noise * days_since
            last_update[pid] = match_date

        total_ll += compute_likelihood(
            base_estimates[match_start:match_end],
            outcomes[match_start:match_end],
            role_ptrs[match_start:match_end],
            roles_data,
            latents,
            weights_array,
        )

        run_ekf(
            base_estimates[match_start:match_end],
            outcomes[match_start:match_end],
            role_ptrs[match_start:match_end],
            roles_data,
            latents,
            variances,
            weights_array,
            measurement_noise,
        )
    print(f"likelihood: {total_ll}")
    return total_ll


x0 = np.concatenate([flat_weights_init, [init_process_noise, init_measurement_noise]])
res = minimize(
    objective,
    x0,
    bounds=[(-5, 5)] * len(flat_weights_init) + [(1e-6, 0.1), (1e-6, 0.1)],
)

opt_weights, opt_process_noise, opt_measurement_noise = unpack_params(res.x)
print("Optimized Weights:", opt_weights)
print("Process Noise:", opt_process_noise)
print("Measurement Noise:", opt_measurement_noise)
