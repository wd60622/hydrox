from itertools import islice
from typing import List, Optional
from pathlib import Path

import pickle

import typer

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from hyrox.data import IndividualDetails, Details, highlight_some, normalize
from hyrox.optimize import create_template, index_values


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

app = typer.Typer()


@app.command()
def save_data(
    url: List[str] = typer.Option(...), path: Path = typer.Option(...)
) -> None:
    """Save data from a list of urls to a pickle file."""
    if path.exists():
        raise FileExistsError(f"File {path} already exists.")

    if path.suffix not in (".pickle", ".pkl"):
        raise ValueError(f"File {path} must have a pickle extension.")

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
    fig.suptitle(f"Hyrox results for {name}")
    plt.show()

    rest_times = individual.get_rest_times().reorder_levels([1, 0])

    print(rest_times)
    print(rest_times.loc["Pre-Exercise"].sort_index())

    fig, axes = plt.subplots(ncols=2)
    ax = axes[0]
    rest_times.loc["Pre-Exercise"].sort_index().iloc[:-1].plot(ax=ax)
    ax.set(title="Rest time before exercise")

    ax = axes[1]
    rest_times.loc["Recovery"].sort_index().plot(ax=ax)
    ax.set(title="Recovery time after exercise")
    plt.show()

    fig, ax = plt.subplots()
    times = other.copy()
    times["Run"] = runs.sum()
    times["Roxzone"] = individual.get_roxzone_time()
    times = times / times.sum()
    times.sort_values().plot.barh(ax=ax)
    ax.set(title="% Time spent on each exercise")

    fig.suptitle(f"Hyrox results for {name}")
    plt.show()


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

    other = details.get_other_exercises().index[1:]
    ax = (
        details.get_runs()
        .pct_change()[1:]
        .set_index(other)
        .pipe(highlight_some, highlight_idx=highlight)
    )
    ax.set(
        ylim=(0.9 - 1, 1.20 - 1),
        title="Time relative to previous",
        xlabel="Previous Exercise",
    )
    plt.show()

    details.plot_splits(highlight=highlight)
    plt.show()

    details.plot_overall_times()
    plt.show()

    exercises = details.get_exercises()

    exercises.iloc[:, :-2].reset_index(drop=True).plot()
    plt.show()

    exercises["rank"] = range(1, len(details.individuals) + 1)
    print(exercises.corr()["rank"].sort_values())

    # details.plot_cummlative_splits(highlight=highlight)
    # plt.show()


@app.command()
def workout_correlation(
    path: List[Path] = typer.Option(...), exercise: List[str] = typer.Option(...)
) -> None:
    details = load_details(path)
    details.individuals = [
        individual for individual in details.individuals if individual is not None
    ]
    details.sort_by_rank()

    exercises = details.get_exercises()
    for ex in exercise:
        if ex not in exercises.columns:
            raise ValueError(
                f"Exercise {ex} not found in the data. Available exercises are {exercises.columns}"
            )

    exercises["rank"] = range(1, len(details.individuals) + 1)

    for ex in exercise:
        print(ex)
        xlim = exercises[ex].quantile(0.99) * 1.1
        ax = exercises.plot.scatter(x=ex, y="rank")
        ax.set(title=f"Correlation between {ex} and rank")
        ax.set_xlim(None, xlim)
        plt.show()


@app.command()
def individual_comparison(
    path: List[Path] = typer.Option(...), highlight: List[int] = HIGHLIGHT
) -> None:
    """Load the results and compare the individuals."""
    details = load_details(path)

    exercises = details.get_exercises()

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


@app.command()
def optimize_template(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"File {path} already exists.")

    template = create_template()
    template.to_csv(path, index=True)


@app.command()
def optimize_template_average(
    path: Path,
    results_path: Path = typer.Option(...),
    individual: Optional[int] = typer.Option(None),
) -> None:
    if results_path.exists():
        raise FileExistsError(f"File {path} already exists.")

    details = load_details([results_path])
    df = details.get_other_exercises()
    df["Running 1000m"] = details.get_runs().sum(axis=1) / 8
    template = create_template()
    fill_values = (
        df.T.mean(axis=1).iloc[:-3] if individual is None else df.iloc[individual, :-3]
    )
    template.loc[:, "All-In"] = fill_values
    print(template)
    template.round(2).to_csv(results_path, index=True)


@app.command()
def optimize(
    path: Path, maintenance: int = 4, prioritize: int = 2, all_in: int = 2
) -> None:
    """Brute force the optimization of the exercises based on the template and different effort levels."""
    df = pd.read_csv(path, index_col=0).iloc[:8]

    if df.isnull().any().any():
        raise ValueError("The template must be filled out completely.")

    total = maintenance + prioritize + all_in
    if total != len(df):
        raise ValueError(
            f"The number of exercises must equal the sum of the effort levels. i.e. {total} != {len(df)}"
        )

    counts = [maintenance, prioritize, all_in]
    times = []
    min_time = np.inf
    best = None

    iteration_values = islice(index_values(counts=counts), 10_000_000)
    for idx, (main, prio, ai) in enumerate(iteration_values):
        total_time = (
            df.iloc[main, 0].sum() + df.iloc[prio, 1].sum() + df.iloc[ai, 2].sum()
        )

        times.append(total_time)

        if total_time < min_time:
            min_time = total_time
            best = (idx, main, prio, ai)

    print("The best happens when you do the following exercises:")
    print("Maintenance:")
    print(df.index[best[1]])
    print("Priority:")
    print(df.index[best[2]])
    print("All-In:")
    print(df.index[best[3]])

    _, axes = plt.subplots(ncols=2)

    ax = axes[0]
    times = pd.Series(times).pipe(lambda ser: ser / ser.min())
    times.plot(ax=ax, alpha=0.25)
    ax.axvline(best[0], color="black", linestyle="--")

    ax = axes[1]
    times.hist(ax=ax, bins=30, edgecolor="black")
    plt.show()
