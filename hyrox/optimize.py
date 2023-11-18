"""Optimize the performance of an individual."""
from itertools import permutations
from typing import List
import warnings

import numpy as np
import pandas as pd


exercises = [
    "1000m SkiErg",
    "50m Sled Push",
    "50m Sled Pull",
    "80m Burpee Broad Jump",
    "1000m Row",
    "200m Farmers Carry",
    "100m Sandbag Lunges",
    "Wall Balls",
    "Running 1000m",
]


def create_template() -> pd.DataFrame:
    effort_levels = ["Maintenance", "Priority", "All-In"]
    return pd.DataFrame(index=exercises, columns=effort_levels)


def index_values(counts: List[int]):
    """Return all possible index values for a given set of counts."""
    total = sum(counts)

    if total > 10:
        warnings.warn(
            "This may take a while. Consider reducing the number of exercises.",
            UserWarning,
            stacklevel=2,
        )

    index_values = np.arange(total)

    for perm in permutations(index_values, total):
        splits = []
        start = 0
        for count in counts:
            splits.append(np.array(perm[start : start + count]))
            start += count

        yield splits
