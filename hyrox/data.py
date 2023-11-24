from dataclasses import dataclass
from typing import List, Optional, Union

import requests
from bs4 import BeautifulSoup

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from hyrox.plot import highlight_some
from hyrox.utils import ordinal, time_to_seconds


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
EXERCISES = [f"{i + 1:02} {exercise}" for i, exercise in enumerate(EXERCISES)]

SPLIT_INDEX = pd.MultiIndex.from_product(
    [EXERCISES[:-1], ["Pre-Exercise", "Exercise", "Recovery", "Run After"]]
).append(
    pd.MultiIndex.from_tuples(
        [
            (EXERCISES[-1], "Exercise"),
            (EXERCISES[-1], "Recovery"),
        ]
    )
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
        idx = self.workout_result.index.str.contains("Running")
        ser = self.workout_result.loc[idx, "seconds"]
        ser.index = range(1, len(ser) + 1)
        ser.index.name = "Run"

        return ser

    def get_other_exercises(self) -> pd.Series:
        ser = self.workout_result.iloc[1:-2:2]["seconds"]
        ser.index.name = "Exercise"

        return ser

    def percent_running(self) -> float:
        runs = self.get_runs()
        other = self.get_other_exercises()

        return runs.sum() / (runs.sum() + other.sum())

    def get_roxzone_time(self) -> float:
        return self.workout_result.loc["Roxzone Time", "seconds"]

    def get_splits(self) -> pd.DataFrame:
        if "seconds" not in self.splits.columns:
            self.splits["seconds"] = time_to_seconds(self.splits["Time"])
            self.splits["diff"] = self.splits["seconds"].diff().shift(-1)

        return self.splits.set_index(SPLIT_INDEX)

    def get_rest_times(self) -> pd.Series:
        return self.get_splits()["diff"].rename("seconds")

    @classmethod
    def from_url(cls, individual_url: str) -> "IndividualDetails":
        try:
            dfs = pd.read_html(individual_url)
        except Exception as e:
            print(f"Error loading {individual_url}: {e}")
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

    def __getitem__(self, idx: int) -> IndividualDetails:
        return self.individuals[idx]

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
                if individual is not None:
                    individuals.append(individual)

        return cls(individuals=individuals)

    def sort_by_rank(self) -> "Details":
        self.individuals.sort(key=lambda individual: individual.get_rank())
        return self

    def total_running(self) -> pd.Series:
        index = [individual.get_name(with_rank=True) for individual in self.individuals]
        return pd.Series(
            [individual.get_runs().sum() for individual in self.individuals],
            name="Total Running",
            index=index,
        )

    def percent_running(self) -> pd.Series:
        index = [individual.get_name(with_rank=True) for individual in self.individuals]
        return pd.Series(
            [individual.percent_running() for individual in self.individuals],
            name="% Running",
            index=index,
        ).mul(100)

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

    def plot_rest_times(self, highlight, ymax: int = 60) -> None:
        df_rest = self.get_rest_times().reorder_levels([1, 0])

        fig, axes = plt.subplots(ncols=2)
        fig.suptitle("Rest times")

        ax = axes[0]
        df_rest.loc["Pre-Exercise"].pipe(highlight_some, highlight_idx=highlight, ax=ax)
        ax.set(
            ylim=(0, ymax),
            title="Rest time before exercise",
            ylabel="Time (seconds)",
        )
        ax = axes[1]
        df_rest.loc["Recovery"].pipe(highlight_some, highlight_idx=highlight, ax=ax)
        ax.set(
            ylim=(0, ymax),
            title="Recovery time after exercise",
            ylabel="",
        )


def get_base_url(url) -> str:
    base_url = url.split("?")[0]

    if not base_url.endswith("index.php"):
        base_url += "index.php"

    return base_url


def get_all_hrefs(url: str) -> List[str]:
    """Extract all the individual results URLs

    Example page:
    https://results.hyrox.com/season-6/?page=2&event=HPRO_JGDMS4JI619&pid=list&pidp=ranking_nav&ranking=time_finish_netto&search%5Bsex%5D=M&search%5Bage_class%5D=%25&search%5Bnation%5D=%25

    """
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

    highlight_some(running_times, highlight_idx=highlight, ax=axes[0])
    axes[0].set_title("Running times")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Time (seconds)")

    highlight_some(other_exercises, highlight_idx=highlight, ax=axes[1])
    axes[1].set_title("Other exercises")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")


def plot_cummlative_splits(
    results: List[IndividualDetails],
    highlight: Union[int, List[int]] = 5,
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

    ax = fig or plt.gca()
    ax = highlight_some(df_splits, highlight_idx=highlight, ax=ax)
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
                for result in results
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
