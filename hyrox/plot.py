from typing import List, Optional, Union

import pandas as pd

import matplotlib.pyplot as plt


def highlight_some(
    df: pd.DataFrame,
    highlight_idx: Union[int, List[int]],
    ax: Optional[plt.Axes] = None,
    **plot_kwargs,
) -> plt.Axes:
    """Plot the dataframe columns along index with some highlighted."""
    ax = ax or plt.gca()
    plot_kwargs = {
        "legend": False,
        "color": "black",
        "alpha": 0.10,
    } | plot_kwargs

    if isinstance(highlight_idx, int):
        highlight_idx = list(range(highlight_idx))

    df.plot(ax=ax, **plot_kwargs)
    df.iloc[:, highlight_idx].plot(ax=ax, legend=True)
    return ax
