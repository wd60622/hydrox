import numpy as np

from hydrox.optimize import index_values


def test_index_values():
    counts = [1, 2]
    expected = [
        [np.array([0]), np.array([1, 2])],
        [np.array([0]), np.array([2, 1])],
        [np.array([1]), np.array([0, 2])],
        [np.array([1]), np.array([2, 0])],
        [np.array([2]), np.array([0, 1])],
        [np.array([2]), np.array([1, 0])],
    ]

    actual = list(index_values(counts))

    for act, exp in zip(actual, expected):
        for a, e in zip(act, exp):
            assert np.array_equal(a, e)
