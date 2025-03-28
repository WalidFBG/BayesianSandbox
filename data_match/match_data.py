from dataclasses import dataclass
from datetime import datetime
from typing import List
from .observation import Observation


@dataclass
class MatchData:
    match_id: str
    date: datetime
    observations: List[Observation]

    def get_all_player_ids(self) -> List[str]:
        return list({pid for obs in self.observations for pid in obs.roles.values()})
