import pytest

import math
import numpy as np

from hyrox.optimize import brute_force_index_values, combinations_split


def total_counts(counts):
    return math.factorial(sum(counts)) / np.prod([math.factorial(c) for c in counts])


@pytest.mark.parametrize(
    "idx, r, expected",
    [
        (
            np.arange(4),
            2,
            [
                ([0, 1], [2, 3]),
                ([0, 2], [1, 3]),
                ([0, 3], [1, 2]),
                ([1, 2], [0, 3]),
                ([1, 3], [0, 2]),
                ([2, 3], [0, 1]),
            ],
        ),
        (
            np.arange(5),
            3,
            [
                ([0, 1, 2], [3, 4]),
                ([0, 1, 3], [2, 4]),
                ([0, 1, 4], [2, 3]),
                ([0, 2, 3], [1, 4]),
                ([0, 2, 4], [1, 3]),
                ([0, 3, 4], [1, 2]),
                ([1, 2, 3], [0, 4]),
                ([1, 2, 4], [0, 3]),
                ([1, 3, 4], [0, 2]),
                ([2, 3, 4], [0, 1]),
            ],
        ),
    ],
)
def test_combination_split(idx, r, expected) -> None:
    actual = list(combinations_split(idx, r))

    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        for a, e in zip(act, exp):
            np.testing.assert_array_equal(a, np.array(e))


@pytest.mark.parametrize(
    "counts",
    [
        (1, 2, 3),
        (1, 2, 2),
        (5, 3, 1),
        (8, 3, 2),
    ],
)
def test_index_values(counts):
    actual = list(brute_force_index_values(counts))

    assert len(actual) == total_counts(counts)
