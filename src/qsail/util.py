import numpy as np
from .types import Hermitian, is_hermitian


def pauli() -> tuple[Hermitian, Hermitian, Hermitian]:
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    assert is_hermitian(X)
    assert is_hermitian(Y)
    assert is_hermitian(Z)

    return X, Y, Z