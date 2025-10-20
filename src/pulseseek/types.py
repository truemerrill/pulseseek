from typing import Any, TypeGuard
import numpy as np
import jax


Scalar = float | complex | np.floating[Any] | np.complexfloating[Any, Any] | jax.Array

def is_scalar(x: Any) -> TypeGuard[Scalar]:
    """Check if x is a Scalar

    Args:
        x (Any): the object to check

    Returns:
        TypeGuard[Scalar]: whether the object is a scalar
    """
    if hasattr(x, "shape"):
        return x.shape == (1,)
    if isinstance(x, (float, complex)):
        return True
    return False


LieVector = jax.Array


def is_vector(x: Any, *, dimension: int | None = None) -> TypeGuard[LieVector]:
    """Check if x is a vector

    Args:
        x (Any): the object to check
        dimension (int | None): the vector dimension. Defaults to None.

    Returns:
        TypeGuard[Vector]: whether x is a vector
    """
    if not isinstance(x, jax.Array):
        return False
    if x.ndim != 1:
        return False
    m = x.shape[0]
    if dimension and m != dimension:
        return False
    return True


Matrix = jax.Array
SquareMatrix = Matrix
Hermitian = Matrix
SquareMatrix = Matrix


def is_matrix(x: Any, *, shape: tuple[int, int] | None = None) -> TypeGuard[Matrix]:
    """Check if x is a matrix

    Args:
        x (Any): the object to check
        shape (tuple[int, int] | None): the required matrix shape. Defaults to None.

    Returns:
        TypeGuard[Matrix]: whether the object is a matrix
    """
    if not isinstance(x, jax.Array):
        return False
    if x.ndim != 2:
        return False
    if shape and x.shape != shape:
        return False
    return True


def is_square_matrix(
    x: Any, *, dimension: int | None = None
) -> TypeGuard[SquareMatrix]:
    if not is_matrix(x):
        return False
    n, m = x.shape
    if n != m:
        return False
    if dimension and (n != dimension or m != dimension):
        return False
    return True


def is_hermitian(
    x: Any, *, dimension: int | None = None, atol: float = 1e-12
) -> TypeGuard[Hermitian]:
    if not is_square_matrix(x, dimension=dimension):
        return False
    X = np.asarray(x, dtype=np.complex64)
    return np.allclose(X, X.conj().T, atol=atol)


def is_anti_hermitian(
    x: Any, *, dimension: int | None = None, atol: float = 1e-12
) -> TypeGuard[SquareMatrix]:
    if not is_square_matrix(x, dimension=dimension):
        return False
    X = np.asarray(x, dtype=np.complex64)
    return np.allclose(X, -X.conj().T, atol=atol)


Tensor = jax.Array
AntiSymmetricTensor = jax.Array


def is_tensor(x: Any, *, shape: tuple[int, ...] | None = None) -> TypeGuard[Tensor]:
    """Check if x is a tensor

    Args:
        x (Any): the object to check
        shape (tuple[int, ...] | None): the required tensor shape. Defaults to None.

    Returns:
        TypeGuard[Tensor]: whether the object is a tensor
    """
    if not isinstance(x, jax.Array):
        return False
    if shape and x.shape != shape:
        return False
    return True


def is_antisymmetric_tensor(
    x: Any, *, dimension: int | None = None, atol: float = 1e-12
) -> TypeGuard[AntiSymmetricTensor]:
    """Check if x is an anti-symmetric rank-3 tensor

    Args:
        x (Any): the object to check
        dimension (int | None): the tensor dimension. Defaults to None.
        atol (float, optional): the absolute tolerance. Defaults to 1e-12.

    Returns:
        TypeGuard[AntiSymmetricTensor]: whether the object is an anti-symmetric
            tensor
    """
    if not isinstance(x, jax.Array):
        return False
    if x.ndim != 3:
        return False
     
    m = dimension if dimension else x.shape[0]
    if not x.shape == (m, m, m):
        return False
    for a in range(m):
        if not np.allclose(x[a], - x[a].T, atol=atol):
            return False
    return True
