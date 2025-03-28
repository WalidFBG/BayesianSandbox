import pandas as pd
from typing import List, Dict
from data_match.observation import Observation
from data_match.match_data import MatchData


def load_matches_from_csv(
    csv_path: str, observation_role_config: Dict[str, List[str]]
) -> List[MatchData]:
    df = pd.read_csv(csv_path)
    df = df[df["ObservationType"].isin(observation_role_config.keys())]

    df["DateTime"] = pd.to_datetime(df["DateTime"], dayfirst=True)
    df["MatchId"] = df["MatchId"].astype(str)

    matches: Dict[str, Dict] = {}

    for row in df.itertuples(index=False):
        obs_type = row.ObservationType
        match_id = row.MatchId
        timestamp = row.DateTime
        outcome = int(row.Outcome)
        base_logit = float(row.BaseEstimate)

        roles_config = observation_role_config[obs_type]
        roles = {
            role: str(int(getattr(row, role)))
            for role in roles_config
            if pd.notna(getattr(row, role))
        }

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

    return [
        MatchData(match_id=mid, date=data["date"], observations=data["observations"])
        for mid, data in matches.items()
    ]
