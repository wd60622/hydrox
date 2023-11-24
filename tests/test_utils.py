import pytest

import pandas as pd
import numpy as np

from hyrox.utils import ordinal, time_to_seconds


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (10, "10th"),
        (11, "11th"),
        (20, "20th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (24, "24th"),
        (100, "100th"),
        (101, "101st"),
        (102, "102nd"),
        (103, "103rd"),
    ],
)
def test_ordinal(n, expected):
    assert ordinal(n) == expected


def test_time_to_seconds():
    times = pd.Series(["–", "00:01:00", "01:00:00", "01:00:00"])
    actual = time_to_seconds(times)

    expected = pd.Series([np.nan, 60, 60 * 60, 60 * 60])
    pd.testing.assert_series_equal(actual, expected)
