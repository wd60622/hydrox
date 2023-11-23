"""Optimize the performance of an individual."""
from itertools import combinations
from typing import Tuple

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


def combinations_split(idx, r):
    for comb in combinations(idx, r):
        comb_array = np.array(comb)
        mask = np.isin(idx, comb_array)

        yield idx[mask], idx[~mask]


def index_values(counts: Tuple[int, int, int]):
    total = sum(counts)

    idx = np.arange(total)

    for lit, lit_rest in combinations_split(idx, counts[0]):
        for lit2, lit2_rest in combinations_split(lit_rest, counts[1]):
            yield lit, lit2, lit2_rest
