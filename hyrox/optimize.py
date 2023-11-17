"""Optimize the performance of an individual."""
from itertools import permutations
from typing import List

import numpy as np
import pandas as pd


exercises = [
    "Running 1",
    "1000m SkiErg",
    "Running 2",
    "50m Sled Push",
    "Running 3",
    "50m Sled Pull",
    "Running 4",
    "80m Burpee Broad Jump",
    "Running 5",
    "1000m Row",
    "Running 6",
    "200m Farmers Carry",
    "Running 7",
    "100m Sandbag Lunges",
    "Running 8",
    "Wall Balls",
    "Roxzone Time",
]


def create_template() -> pd.DataFrame:
    effort_levels = ["Maintenance", "Priority", "All-In"]
    return pd.DataFrame(index=exercises, columns=effort_levels)


def index_values(counts: List[int]):
    """Return all possible index values for a given set of counts."""
    total = sum(counts)

    index_values = np.arange(total)

    for perm in permutations(index_values, total):
        splits = []
        start = 0
        for count in counts:
            splits.append(np.array(perm[start : start + count]))
            start += count

        yield splits
