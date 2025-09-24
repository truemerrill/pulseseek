import numpy as np
from qsail.types import (
    is_matrix,
    is_square_matrix,
    is_hermitian,
    is_anti_hermitian,
)


def test_is_matrix():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([1, 2])
    assert is_matrix(X)
    assert not is_matrix(Y)
    assert not is_matrix(X, shape=(2, 3))
    assert is_matrix(X, shape=(2, 2))


def test_is_square_matrix():
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([[1, 2, 3], [4, 5, 6]])
    assert is_square_matrix(X)
    assert not is_square_matrix(Y)


def test_is_hermitian():
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1], [1, 0]])
    assert is_hermitian(X)
    assert not is_hermitian(Y)


def test_is_anti_hermitian():
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1], [1, 0]])
    assert not is_anti_hermitian(X)
    assert is_anti_hermitian(Y)
