import jax.numpy as jnp

from .types import ArrayLike, Hermitian, is_hermitian


def hash_array(x: ArrayLike) -> int:
    return hash((x.shape, x.dtype, x.tobytes()))


def pauli() -> tuple[Hermitian, Hermitian, Hermitian]:
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    assert is_hermitian(X)
    assert is_hermitian(Y)
    assert is_hermitian(Z)

    return X, Y, Z