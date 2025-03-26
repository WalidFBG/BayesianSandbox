from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class LatentState:
    mean: np.ndarray
    std: np.ndarray
    last_updated: datetime
