"""Optimize the performance of an individual."""
from itertools import combinations, islice
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


def _index_helper(idx, counts):
    for lit, lit_rest in combinations_split(idx, counts[0]):
        for lit2, lit2_rest in combinations_split(lit_rest, counts[1]):
            yield lit, lit2, lit2_rest


def brute_force_index_values(counts: Tuple[int, int, int]):
    total = sum(counts)

    idx = np.arange(total)

    yield from _index_helper(idx, counts)


def random_index_values(counts: Tuple[int, int, int]):
    total = sum(counts)

    idx = np.arange(total)

    np.random.shuffle(idx)

    yield from _index_helper(idx, counts)


def first_n_index_values(counts: Tuple[int, int, int], n: int):
    yield from islice(brute_force_index_values(counts=counts), n)


def calculate_total_time(df: pd.DataFrame, main, prio, ai):
    return df.iloc[main, 0].sum() + df.iloc[prio, 1].sum() + df.iloc[ai, 2].sum()


def brute_force(df: pd.DataFrame, counts: Tuple[int, int, int]):
    if df.isnull().any().any():
        raise ValueError("The exercises must be filled out completely.")

    total = sum(counts)
    if total != len(df):
        raise ValueError(
            f"The number of exercises must equal the sum of the effort levels. i.e. {total} != {len(df)}"
        )

    times = []
    min_time = np.inf
    best = None

    iteration_values = brute_force_index_values(counts=counts)
    for idx, (main, prio, ai) in enumerate(iteration_values):
        total_time = calculate_total_time(df, main, prio, ai)
        times.append(total_time)

        if total_time < min_time:
            min_time = total_time
            best = (idx, main, prio, ai)

    print("The best happens when you do the following effort levels:")
    print("Maintenance:")
    print(df.index[best[1]].tolist())
    print("Priority:")
    print(df.index[best[2]].tolist())
    print("All-In:")
    print(df.index[best[3]].tolist())

    return times, best
