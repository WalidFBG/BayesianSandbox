import pandas as pd
from typing import List, Dict
from core_engine.data_match.observation import Observation
from core_engine.data_match.match_data import MatchData


def load_matches_from_csv(
    csv_path: str, observation_role_config: Dict[str, List[str]]
) -> List[MatchData]:
    df = pd.read_csv(csv_path)
    matches = {}

    for _, row in df.iterrows():
        obs_type = row["ObservationType"]
        if obs_type not in observation_role_config:
            continue

        match_id = str(row["MatchId"])
        timestamp = pd.to_datetime(row["DateTime"])
        outcome = int(row["Outcome"])
        base_logit = float(row["BaseLogit"])

        roles_config = observation_role_config[obs_type]
        roles = {}

        for role in roles_config:
            player_id = row.get(role)
            if pd.notna(player_id):
                roles[role] = str(int(player_id))

        obs = Observation(
            obs_type=obs_type,
            roles=roles,
            base_logit=base_logit,
            outcome=outcome,
        )

        if match_id not in matches:
            matches[match_id] = {
                "date": timestamp,
                "observations": [obs],
            }
        else:
            matches[match_id]["observations"].append(obs)

    match_objects = [
        MatchData(match_id=mid, date=data["date"], observations=data["observations"])
        for mid, data in matches.items()
    ]
    return match_objects
