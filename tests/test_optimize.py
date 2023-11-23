import pytest

import math
import numpy as np

from hyrox.optimize import index_values, combinations_split


def total_counts(counts):
    return math.factorial(sum(counts)) / np.prod([math.factorial(c) for c in counts])


def test_combination_split() -> None:
    idx = np.arange(4)

    actual = list(combinations_split(idx, 2))

    assert len(actual) == 6
    np.testing.assert_array_equal(actual[0][0], np.array([0, 1]))
    np.testing.assert_array_equal(actual[0][1], np.array([2, 3]))
    np.testing.assert_array_equal(actual[1][0], np.array([0, 2]))
    np.testing.assert_array_equal(actual[1][1], np.array([1, 3]))


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
