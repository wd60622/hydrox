from typing import List
from pathlib import Path

import pickle

import typer

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from hydrox.data import IndividualDetails, Details, highlight_some


def highlight_callback(values: List[int]) -> List[int]:
    if len(values) == 0:
        return list(range(5))

    return [val - 1 for val in values]


HIGHLIGHT = typer.Option([], callback=highlight_callback)


def details_callback(paths: List[Path]) -> Details:
    details = []
    for path in paths:
        with path.open(mode="rb") as f:
            details.append(pickle.load(f))

    return details[0]


DETAILS = typer.Option(..., callback=details_callback)


def normalize(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    transform = np.log if log else lambda x: x
    return df.pipe(transform).pipe(lambda df: (df - df.mean()) / df.std())


app = typer.Typer()


@app.command()
def save_data(
    url: List[str] = typer.Option(...), path: Path = typer.Option(...)
) -> None:
    """Save data from a list of urls to a pickle file."""
    details = Details.from_urls(url)

    with path.open(mode="wb") as f:
        pickle.dump(details, f)


@app.command()
def individual_profile(url: str) -> None:
    """Load from the URL and display some figures for the individual"""
    individual = IndividualDetails.from_url(url)

    print(individual.participant)

    fig, axes = plt.subplots(ncols=2)

    ax = axes[0]
    runs = individual.get_runs()
    runs.plot(ax=ax)

    ax = axes[1]
    other = individual.get_other_exercises()
    other.plot(ax=ax)
    ax.axhline(individual.get_roxzone_time(), color="black", linestyle="--")

    name = individual.participant["Name"]
    fig.suptitle(f"Hydrox results for {name}")
    plt.show()

    fig, ax = plt.subplots()
    times = other.copy()
    times["Run"] = runs.sum()
    times["Roxzone"] = individual.get_roxzone_time()
    times.sort_values().plot.bar(ax=ax)

    fig.suptitle(f"Hydrox results for {name}")


def load_details(paths: List[Path]) -> Details:
    details = []
    for path in paths:
        with path.open(mode="rb") as f:
            details.append(pickle.load(f))

    details = details[0]
    return details


@app.command()
def results(
    path: List[Path] = typer.Option(...), highlight: List[int] = HIGHLIGHT
) -> None:
    """High level overview of the results."""
    details = load_details(path)

    details.plot_splits(highlight=highlight)
    plt.show()

    details.plot_overall_times()
    plt.show()

    details.plot_cummlative_splits(highlight=highlight)
    plt.show()


@app.command()
def individual_comparison(
    path: List[Path] = typer.Option(...), highlight: List[int] = HIGHLIGHT
) -> None:
    """Load the results and compare the individuals."""
    details = load_details(path)

    exercises = details.get_exercises()

    print(exercises)

    exercises_norm = exercises.pipe(normalize, log=True)

    exercises_norm.T.pipe(highlight_some, highlight_idx=highlight)
    plt.show()

    df_stats = pd.concat(
        [
            exercises.mean().rename("mean"),
            exercises.std().rename("std"),
            exercises_norm.pipe(
                lambda df: (df.iloc[highlight] - df.mean()) / df.std()
            ).T,
        ],
        axis=1,
    )

    ys = exercises.index[highlight]
    for y in ys:
        for x in ["mean", "std"]:
            df_plot = df_stats.iloc[:-3].loc[:, [x, y]]
            print(df_plot.sort_values(x))
            ax = df_plot.plot.scatter(x=x, y=y)
            ax.axhline(0, color="black", linestyle="--")
            plt.show()
