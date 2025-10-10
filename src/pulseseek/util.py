import jax.numpy as jnp
from jax import Array
from itertools import combinations
from typing import Iterator
from .types import Hermitian, is_hermitian


def hash_array(x: Array) -> int:
    return hash((x.shape, x.dtype, x.tobytes()))


def pauli() -> tuple[Hermitian, Hermitian, Hermitian]:
    """Return the Pauli matrices

    Returns:
        tuple[Hermitian, Hermitian, Hermitian]: the Pauli matrices.
    """
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    assert is_hermitian(X)
    assert is_hermitian(Y)
    assert is_hermitian(Z)

    return X, Y, Z


def sumsto(total: int, degree: int) -> Iterator[tuple[int, ...]]:
    """Iterate over tuples of length `degree` that sum to `total`.

    Args:
        total (int): the total sum of the tuple
        degree (int): the length of the tuple

    Yields:
        Iterator[tuple[int, ...]]: the sequence of tuples
    """
    if degree <= 0 or total <= 0:
        return
    
    for cuts in combinations(range(1, total), degree - 1):
        parts = [a - b for a, b in zip(cuts + (total,), (0,) + cuts)]
        yield tuple(parts)
