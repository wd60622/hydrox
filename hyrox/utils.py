import pandas as pd
import numpy as np


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
