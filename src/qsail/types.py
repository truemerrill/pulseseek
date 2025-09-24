from typing import Any, Protocol, Self, TypeGuard, runtime_checkable
import numpy as np
import numpy.typing as npt


Array = npt.NDArray[Any]


class SupportsAddition(Protocol):
    def __add__(self, other: Any, /) -> Self: ...
    def __radd__(self, other: Any, /) -> Self: ...
    def __sub__(self, other: Any, /) -> Self: ...
    def __rsub__(self, other: Any, /) -> Self: ...


Scalar = float | complex | np.floating[Any] | np.complexfloating[Any, Any]


class SupportsScalarMultiplication(Protocol):
    def __mul__(self, other: Scalar) -> Self: ...
    def __rmul__(self, other: Scalar) -> Self: ...
    def __truediv__(self, other: Scalar) -> Self: ...
    def __neg__(self) -> Self: ...


class SupportsMatrixMultiplication(Protocol):
    def __matmul__(self, other: Any) -> Self: ...
    def __rmatmul__(self, other: Any) -> Self: ...
    def __imatmul__(self, other: Any) -> Self: ...


@runtime_checkable
class Vector(
    SupportsAddition,
    SupportsScalarMultiplication,
    Protocol
):
    @property
    def shape(self) -> tuple[int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def ndim(self) -> int: ...

    def copy(self) -> Self: ...
    
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]: ...
    def __getitem__(self, key: Any) -> Any: ...


def is_vector(x: Any, *, dimension: int | None = None) -> TypeGuard[Vector]:
    """Check if x is a vector

    Args:
        x (Any): the object to check
        dimension (int | None): the vector dimension. Defaults to None.

    Returns:
        TypeGuard[Vector]: whether x is a vector
    """
    if not isinstance(x, Vector):
        return False
    if x.ndim != 1:
        return False
    m = x.shape[0]
    if dimension and m != dimension:
        return False
    return True


@runtime_checkable
class Matrix(
    SupportsAddition,
    SupportsScalarMultiplication,
    SupportsMatrixMultiplication,
    Protocol,
):
    @property
    def shape(self) -> tuple[int, int]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def ndim(self) -> int: ...

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]: ...
    def __getitem__(self, key: Any) -> Any: ...


@runtime_checkable
class SquareMatrix(Matrix, Protocol): ...


@runtime_checkable
class Hermitian(SquareMatrix, Protocol): ...


@runtime_checkable
class AntiHermitian(SquareMatrix, Protocol): ...


def is_matrix(x: Any, *, shape: tuple[int, int] | None = None) -> TypeGuard[Matrix]:
    """Check if x is a matrix

    Args:
        x (Any): the object to check
        shape (tuple[int, int] | None): the required matrix shape. Defaults to None.

    Returns:
        TypeGuard[Matrix]: whether the object is a matrix
    """
    if not isinstance(x, Matrix):
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
    X = np.asarray(x, dtype=np.complex128)
    return np.allclose(X, X.conj().T, atol=atol)


def is_anti_hermitian(
    x: Any, *, dimension: int | None = None, atol: float = 1e-12
) -> TypeGuard[AntiHermitian]:
    if not is_square_matrix(x, dimension=dimension):
        return False
    X = np.asarray(x, dtype=np.complex128)
    return np.allclose(X, -X.conj().T, atol=atol)


@runtime_checkable
class Tensor(SupportsAddition, SupportsScalarMultiplication, Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def ndim(self) -> int: ...

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]: ...
    def __getitem__(self, key: Any) -> Any: ...


@runtime_checkable
class AntiSymmetricTensor(Tensor, Protocol):
    @property
    def shape(self) -> tuple[int, int, int]: ...


def is_tensor(x: Any, *, shape: tuple[int, ...] | None = None) -> TypeGuard[Tensor]:
    """Check if x is a tensor

    Args:
        x (Any): the object to check
        shape (tuple[int, ...] | None): the required tensor shape. Defaults to None.

    Returns:
        TypeGuard[Tensor]: whether the object is a tensor
    """
    if not isinstance(x, Tensor):
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
    if not isinstance(x, AntiSymmetricTensor):
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
