import pytest

import math
import numpy as np

from hyrox.optimize import index_values, combinations_split


def total_counts(counts):
    return math.factorial(sum(counts)) / np.prod([math.factorial(c) for c in counts])


@pytest.mark.parametrize(
    "idx, r, expected",
    [
        (
            np.arange(4),
            2,
            [
                (np.array([0, 1]), np.array([2, 3])),
                (np.array([0, 2]), np.array([1, 3])),
                (np.array([0, 3]), np.array([1, 2])),
                (np.array([1, 2]), np.array([0, 3])),
                (np.array([1, 3]), np.array([0, 2])),
                (np.array([2, 3]), np.array([0, 1])),
            ],
        ),
    ],
)
def test_combination_split(idx, r, expected) -> None:
    actual = list(combinations_split(idx, r))

    assert len(actual) == len(expected)
    for act, exp in zip(actual, expected):
        for a, e in zip(act, exp):
            np.testing.assert_array_equal(a, e)


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
    actual = list(index_values(counts))

    assert len(actual) == total_counts(counts)
