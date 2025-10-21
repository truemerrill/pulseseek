from dataclasses import dataclass
from typing import Any, Generator, Mapping, overload

import jax.numpy as jnp
import numpy as np
import functools

from .types import (
    LieVector,
    SquareMatrix,
    is_anti_hermitian,
    is_square_matrix,
    is_vector,
)
from .util import hash_array, pauli


@overload
def basis_vector(b: int, index: int) -> LieVector: ...
@overload
def basis_vector(b: "LieBasis", index: int) -> LieVector: ...


def basis_vector(b: "int | LieBasis", index: int) -> LieVector:
    """Construct a Lie basis vector

    Args:
        b (int | LieBasis): the basis or the rank of the algebra
        index (int): the basis vector index

    Returns:
        Vector: the basis vector
    """
    dim = b.dim if isinstance(b, LieBasis) else b
    e = np.zeros((dim,), dtype=float)
    e[index] = 1.0
    ev = jnp.array(e)
    assert is_vector(ev)
    return ev


@dataclass(frozen=True)
class LieBasis:
    ndim: int
    _labels: tuple[str, ...]
    _elements: tuple[SquareMatrix, ...]
    _index: Mapping[str, int]

    @overload
    def __getitem__(self, key: int) -> SquareMatrix: ...
    @overload
    def __getitem__(self, key: str) -> SquareMatrix: ...

    def __getitem__(self, key) -> SquareMatrix:
        if isinstance(key, str):
            idx = self._index[key]
            return self._elements[idx]
        elif isinstance(key, int):
            return self._elements[key]
        else:
            raise KeyError(f"Invalid key: {key}")

    def __hash__(self) -> int:
        return hash(
            (self.ndim, self._labels, tuple([hash_array(el) for el in self._elements]))
        )

    @classmethod
    def new(
        cls,
        elements: Mapping[str, Any],
        *,
        dimension: int | None = None,
    ):
        matrices: list[SquareMatrix] = []
        for mat in elements.values():
            if dimension is None:
                dimension = mat.shape[0]
            if not is_square_matrix(mat, dimension=dimension):
                raise ValueError(f"Not a square matrix of dimension {dimension}: {mat}")
            matrices.append(mat)

        if dimension is None:
            raise ValueError("Could not infer dimension")

        labels = tuple(elements.keys())
        index = {label: idx for idx, label in enumerate(labels)}
        return cls(dimension, labels, tuple(matrices), index)

    @property
    def elements(self) -> tuple[SquareMatrix, ...]:
        """The Lie algebra basis vectors in their matrix representation

        Returns:
            tuple[SquareMatrix, ...]: the matrix representations
        """
        return self._elements

    @property
    def vectors(self) -> tuple[LieVector, ...]:
        """The Lie algebra basis vectors

        Returns:
            tuple[LieVector, ...]: the basis vectors
        """
        return tuple([basis_vector(self, i) for i in range(self.dim)])

    @property
    def zero(self) -> LieVector:
        """The zero vector on the Lie algebra

        Returns:
            LieVector: the zero vector
        """
        return jnp.zeros((self.dim,))

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def dim(self) -> int:
        """The dimension (rank) of the Lie algebra

        Returns:
            int: the dimension
        """
        return len(self._elements)

    def items(self) -> Generator[tuple[str, SquareMatrix], None, None]:
        for label, element in zip(self.labels, self.elements):
            yield label, element
    
    def matrix(self, x: LieVector) -> SquareMatrix:
        """The matrix representation of a Lie algebra vector

        Args:
            x (LieVector): the Lie algebra vector

        Returns:
            SquareMatrix: the matrix representation of xs
        """
        def matrix_sum(a: SquareMatrix, b: SquareMatrix) -> SquareMatrix:
            return a + b
  
        return functools.reduce(
            matrix_sum,
            (xi * ei for xi, ei in zip(x, self.elements))
        )


def special_unitary_basis(ndim: int) -> LieBasis:
    if ndim < 2:
        raise ValueError("Dimension must be >= to 2")

    def E(i: int, j: int) -> SquareMatrix:
        M = np.zeros((ndim, ndim), dtype=complex)
        M[i, j] = 1.0
        Ma = jnp.array(M)
        assert is_square_matrix(Ma)
        return Ma

    def S(i: int, j: int) -> SquareMatrix:
        M = 1j * (E(i, j) + E(j, i))
        assert is_anti_hermitian(M)
        return M

    def A(i: int, j: int) -> SquareMatrix:
        M = E(i, j) - E(j, i)
        assert is_anti_hermitian(M)
        return M

    def H(k: int) -> SquareMatrix:
        if not (1 <= k <= ndim - 1):
            raise ValueError(f"k must be in [1, {ndim - 1}]")
        d = np.zeros(ndim, dtype=complex)
        d[:k] = 1.0
        d[k] = -k
        d = d * np.sqrt(2) / np.sqrt(k * (k + 1))
        M = 1j * jnp.diag(d)
        assert is_anti_hermitian(M)
        return M

    elements: dict[str, SquareMatrix] = {}
    for i in range(ndim):
        for j in range(i + 1, ndim):
            s_label = f"S({i + 1},{j + 1})"
            a_label = f"A({i + 1},{j + 1})"

            elements[s_label] = S(i, j)
            elements[a_label] = A(i, j)

    for k in range(1, ndim):
        h_label = f"H({k})"
        elements[h_label] = H(k)

    return LieBasis.new(elements, dimension=ndim)


def pauli_basis() -> LieBasis:
    """The Pauli operator basis for the su(2) algebra

    Returns:
        LieBasis: the Pauli basis
    """
    X, Y, Z = pauli()
    return LieBasis.new({"iX": 1j * X, "iY": 1j * Y, "iZ": 1j * Z})


def heisenburg_basis() -> LieBasis:
    """Heisenburg algebra using traditional basis.

    !!! note

        See the Wikipedia page on the [Heisenburg algebra](https://en.wikipedia.org/wiki/Heisenberg_group#Heisenberg_algebra)

    Returns:
        LieBasis: the Lie basis for the Heisenburg algebra
    """
    X = jnp.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    Y = jnp.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    Z = jnp.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    elements = {"X": X, "Y": Y, "Z": Z}
    return LieBasis.new(elements)


def fock_basis(ndim: int = 15) -> LieBasis:
    """Heisenburg algebra using a truncated Fock state representation.

    Args:
        ndim (int, optional): Dimension of the truncated Fock state. Defaults
            to 15.

    Returns:
        LieBasis: the Lie basis for the Heisenburg algebra
    """

    def annihilation() -> SquareMatrix:
        a = np.zeros((ndim, ndim))
        for n in range(1, ndim):
            a[n - 1, n] = np.sqrt(n)
        A = jnp.array(a)
        assert is_square_matrix(A)
        return A

    def identity() -> SquareMatrix:
        I = jnp.array(np.eye(ndim))
        assert is_square_matrix(I)
        return I

    A = annihilation()
    Id = identity()
    elements = {"a": A, "a†": A.T.conj(), "I": Id}
    return LieBasis.new(elements)
