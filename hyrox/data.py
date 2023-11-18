from dataclasses import dataclass
from typing import List, Optional, Union

import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


EXERCISES = [
    "1000m SkiErg",
    "50m Sled Push",
    "50m Sled Pull",
    "80m Burpee Broad Jump",
    "1000m Row",
    "200m Farmers Carry",
    "100m Sandbag Lunges",
    "Wall Balls",
]

SPLIT_INDEX = pd.MultiIndex.from_product(
    [EXERCISES, ["Pre-Exercise", "Exercise", "Recovery", "Run After"]]
)


def normalize(df: pd.DataFrame, log: bool = True) -> pd.DataFrame:
    transform = np.log if log else lambda x: x
    return df.pipe(transform).pipe(lambda df: (df - df.mean()) / df.std())


@dataclass
class IndividualDetails:
    """Data for an individual athlete.

    Example : https://results.hyrox.com/season-6/index.php?content=detail&fpid=list&pid=list&idp=JGDMS4JI7BE9F&lang=EN_CAP&event=HPRO_JGDMS4JI5C9&num_results=100&pidp=ranking_nav&ranking=time_finish_netto&search%5Bsex%5D=M&search%5Bage_class%5D=%25&search%5Bnation%5D=%25&search_event=HPRO_JGDMS4JI5C9

    """

    participant: pd.Series
    scoring: pd.DataFrame
    workout_result: pd.DataFrame
    judge_decision: pd.DataFrame
    overall_time: pd.DataFrame
    splits: pd.DataFrame

    def get_rank(self) -> int:
        return int(self.overall_time[1].iloc[0])

    def get_name(self, with_rank: bool = False) -> str:
        name = self.participant["Name"]
        if with_rank:
            name = f"{ordinal(self.get_rank())} " + name

        return name

    def get_exercises(self) -> pd.Series:
        return self.workout_result["seconds"]

    def get_runs(self) -> pd.Series:
        ser = self.workout_result.pipe(filter_running)["seconds"]
        ser.index = range(1, len(ser) + 1)
        ser.index.name = "Run"

        return ser

    def get_other_exercises(self) -> pd.Series:
        ser = self.workout_result.iloc[1:-2:2]["seconds"]
        ser.index.name = "Exercise"

        return ser

    def get_roxzone_time(self) -> float:
        return self.workout_result.loc["Roxzone Time", "seconds"]

    def get_splits(self) -> pd.DataFrame:
        if "seconds" not in self.splits.columns:
            self.splits["seconds"] = time_to_seconds(self.splits["Time"])
            self.splits["diff"] = self.splits["seconds"].diff().shift(-1)

        return self.splits.set_index(SPLIT_INDEX[:-2])

    def get_rest_times(self) -> pd.Series:
        return self.get_splits()["diff"].rename("seconds")

    @classmethod
    def from_url(cls, individual_url: str) -> "IndividualDetails":
        try:
            dfs = pd.read_html(individual_url)
        except Exception:
            return None

        individual = cls(
            participant=dfs[0].set_index(0).squeeze(),
            scoring=dfs[1],
            workout_result=dfs[2].set_index("Split"),
            judge_decision=dfs[3],
            overall_time=dfs[4],
            splits=dfs[5],
        )

        individual.workout_result["seconds"] = time_to_seconds(
            individual.workout_result["Time"]
        )

        individual.splits["seconds"] = time_to_seconds(individual.splits["Time"])
        individual.splits["diff"] = individual.splits["seconds"].diff().shift(-1)

        return individual


@dataclass
class Details:
    individuals: List[IndividualDetails]

    @classmethod
    def from_urls(cls, urls: List[str]) -> "Details":
        hrefs = []
        for url in urls:
            hrefs.extend(get_all_hrefs(url))

        hrefs = list(set(hrefs))

        # TODO: Parallelize this
        individuals = []
        for url in hrefs:
            try:
                individual = IndividualDetails.from_url(url)
            except Exception as e:
                print(f"Error loading {url}: {e}")
            else:
                individuals.append(individual)

        return cls(individuals=individuals)

    def sort_by_rank(self) -> "Details":
        self.individuals.sort(key=lambda individual: individual.get_rank())
        return self

    @classmethod
    def from_list(cls, details: List["Details"]) -> "Details":
        individuals = []
        for detail in details:
            individuals.extend(detail.individuals)

        return cls(individuals=individuals)

    def get_exercises(self, with_rank: bool = True) -> pd.Series:
        return pd.concat(
            [
                individual.get_exercises().rename(
                    individual.get_name(with_rank=with_rank)
                )
                for individual in self.individuals
            ],
            axis=1,
        ).T

    def get_runs(self, with_rank: bool = True) -> pd.DataFrame:
        return pd.concat(
            [
                individual.get_runs().rename(individual.get_name(with_rank=with_rank))
                for individual in self.individuals
            ],
            axis=1,
        )

    def get_other_exercises(self, with_rank: bool = True) -> pd.DataFrame:
        return pd.concat(
            [
                individual.get_other_exercises().rename(
                    individual.get_name(with_rank=with_rank)
                )
                for individual in self.individuals
            ],
            axis=1,
        )

    def get_rest_times(self, with_rank: bool = True) -> pd.DataFrame:
        return pd.concat(
            [
                individual.get_rest_times().rename(
                    individual.get_name(with_rank=with_rank)
                )
                for individual in self.individuals
            ],
            axis=1,
        )

    def plot_splits(
        self,
        highlight: Union[int, List[int]] = 5,
        location: Optional[str] = None,
        fig: Optional[plt.Figure] = None,
    ) -> None:
        plot_splits(
            self.individuals,
            highlight=highlight,
            location=location,
            fig=fig,
        )

    def plot_cummlative_splits(
        self,
        highlight: int = 5,
        location: Optional[str] = None,
        fig: Optional[plt.Figure] = None,
    ) -> None:
        plot_cummlative_splits(
            self.individuals, highlight=highlight, location=location, fig=fig
        )

    def plot_overall_times(
        self, ax: Optional[plt.Axes] = None, **plot_kwargs
    ) -> plt.Axes:
        return plot_overall_times(self.individuals, ax=ax, **plot_kwargs)


def load_data(individual_url: str) -> IndividualDetails:
    try:
        dfs = pd.read_html(individual_url)
    except Exception:
        return None

    individual = IndividualDetails(
        participant=dfs[0].set_index(0).squeeze(),
        scoring=dfs[1],
        workout_result=dfs[2].set_index("Split"),
        judge_decision=dfs[3],
        overall_time=dfs[4],
        splits=dfs[5],
    )

    individual.workout_result["seconds"] = time_to_seconds(
        individual.workout_result["Time"]
    )

    return individual


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


def time_to_seconds(times: pd.Series) -> pd.Series:
    null_rows = times == "–"

    split = times.loc[~null_rows].str.split(":", expand=True).astype(int)

    multiplier = np.array([60 * 60, 60, 1])
    result = split @ multiplier

    return result.reindex(times.index)


def filter_running(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index.str.contains("Running")

    return df.loc[idx]


def get_base_url(url) -> str:
    base_url = url.split("?")[0]

    if not base_url.endswith("index.php"):
        base_url += "index.php"

    return base_url


def get_all_hrefs(url: str) -> List[str]:
    """Extract all the individual results URLs"""
    # creating requests object
    html = requests.get(url).content

    # creating soup object
    data = BeautifulSoup(html, "html.parser")

    div = data.find("div", {"class": "col-sm-12 row-xs"})
    rows = div.find_all("li")

    def get_href(row):
        a = row.find("a")
        if a is None:
            return None

        return a["href"]

    base_url = get_base_url(url)
    return [f"{base_url}{get_href(row)}" for row in rows]


def highlight_some(
    df: pd.DataFrame,
    highlight_idx: List[int],
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> pd.DataFrame:
    ax = ax or plt.gca()
    plot_kwargs = plot_kwargs or {
        "legend": False,
        "color": "black",
        "alpha": 0.10,
    }

    df.plot(ax=ax, **plot_kwargs)
    df.iloc[:, highlight_idx].plot(ax=ax, legend=True)
    return ax


def plot_splits(
    results: List[IndividualDetails],
    highlight: Union[int, List[int]] = 5,
    location: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
) -> None:
    running_times = pd.concat(
        [
            individual.get_runs().rename(individual.get_name(with_rank=True))
            for individual in results
        ],
        axis=1,
    )

    other_exercises = pd.concat(
        [
            individual.get_other_exercises().rename(individual.get_name(with_rank=True))
            for individual in results
        ],
        axis=1,
    )

    NCOLS = 2
    if fig is None:
        fig, axes = plt.subplots(ncols=NCOLS)
    else:
        axes = np.array(fig.axes)
        assert len(axes) == NCOLS

    suptitle = f"Top {len(results)} athletes"
    if location is not None:
        suptitle = f"{suptitle} for {location}"
    fig.suptitle(suptitle)

    if isinstance(highlight, int):
        highlight_idx = list(range(highlight))
    else:
        highlight_idx = highlight

    highlight_some(running_times, highlight_idx, ax=axes[0])
    axes[0].set_title("Running times")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Time (seconds)")

    highlight_some(other_exercises, highlight_idx, ax=axes[1])
    axes[1].set_title("Other exercises")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")


def plot_cummlative_splits(
    results: List[IndividualDetails],
    highlight: int = 5,
    location: Optional[str] = None,
    fig: Optional[plt.Figure] = None,
) -> None:
    df_splits = pd.concat(
        [
            result.splits.set_index("Split")["Time"]
            .pipe(time_to_seconds)
            .rename(f"{ordinal(i + 1)} " + result.participant["Name"])
            for i, result in enumerate(results)
        ],
        axis=1,
    )

    if isinstance(highlight, int):
        highlight_idx = list(range(highlight))
    else:
        highlight_idx = highlight

    ax = fig or plt.gca()
    ax = highlight_some(df_splits, highlight_idx, ax=ax)
    ax.set_title("Cumulative splits")
    ax.set_xlabel("")
    ax.set_ylabel("Time (seconds)")

    return ax


def plot_overall_times(
    results, ax: Optional[plt.Axes] = None, **plot_kwargs
) -> plt.Axes:
    overall_times = (
        pd.concat(
            [
                result.overall_time.set_index(0)
                .loc["Overall Time"]
                .rename(result.get_name(with_rank=False))
                for i, result in enumerate(results)
            ],
            axis=1,
        )
        .T.squeeze()
        .pipe(time_to_seconds)
        .rename("Overall Time")
    )

    if ax is None:
        ax = plt.gca()

    ax = overall_times.reset_index(drop=True).mul(1 / 60).plot(ax=ax, **plot_kwargs)

    title = "Overall time"
    ax.set(
        xlabel="Rank",
        ylabel="Overall Time (minutes)",
        title=title,
    )

    return ax


if __name__ == "__main__":
    file = "./data/munich-2023.pkl"
    details = pd.read_pickle(file)

    details.individuals = [
        individual for individual in details.individuals if individual is not None
    ]

    details.sort_by_rank()

    from functools import partial

    log = False
    transform = partial(normalize, log=True) if log else lambda x: x
    df_plot = details.get_rest_times().T.pipe(transform).T.reorder_levels([1, 0])
    fig, axes = plt.subplots(ncols=2)

    ax = axes[0]
    df_plot.loc["Pre-Exercise"].pipe(
        highlight_some, highlight_idx=[0, 1, 2, 149], ax=ax
    )
    ax.set(ylabel="Rank (person - mean) / std", title="Rest time before exercise")

    ax = axes[1]
    df_plot.loc["Recovery"].pipe(highlight_some, highlight_idx=[0, 1, 2, 149], ax=ax)
    ax.set(ylabel="", title="Recovery time after exercise")
    if log:
        for ax in axes:
            ax.set_ylim(-3, 3)
    plt.show()
