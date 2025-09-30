import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Any, Generator, Mapping, overload
from .types import AntiHermitian, SquareMatrix, Vector, is_anti_hermitian, is_square_matrix, is_vector
from .util import hash_array


@overload
def basis_vector(b: int, index: int) -> Vector: ...
@overload
def basis_vector(b: "LieBasis", index: int) -> Vector: ...

def basis_vector(b: "int | LieBasis", index: int) -> Vector:
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
    e = jnp.array(e)
    assert is_vector(e)
    return e


@dataclass(frozen=True)
class LieBasis:
    ndim: int
    _labels: tuple[str, ...]
    _elements: tuple[AntiHermitian, ...]
    _index: Mapping[str, int]

    @overload
    def __getitem__(self, key: int) -> AntiHermitian: ...
    @overload
    def __getitem__(self, key: str) -> AntiHermitian: ...

    def __getitem__(self, key) -> AntiHermitian:
        if isinstance(key, str):
            idx = self._index[key]
            return self._elements[idx]
        elif isinstance(key, int):
            return self._elements[key]
        else:
            raise KeyError(f"Invalid key: {key}")

    def __hash__(self) -> int:
        return hash((
            self.ndim,
            self._labels,
            tuple([hash_array(el) for el in self._elements])
        ))

    @classmethod
    def new(
        cls,
        elements: Mapping[str, Any],
        *,
        dimension: int | None = None,
        atol: float = 1e-12,
    ):
        matrices: list[AntiHermitian] = []
        for mat in elements.values():
            if dimension is None:
                dimension = mat.shape[0]
            if not is_anti_hermitian(mat, dimension=dimension, atol=atol):
                raise ValueError(
                    f"Not an anti-Hermitian matrix of dimension {dimension}: {mat}"
                )
            matrices.append(mat)
            
        if dimension is None:
            raise ValueError("Could not infer dimension")

        labels = tuple(elements.keys())
        index = {label: idx for idx, label in enumerate(labels)}
        return cls(dimension, labels, tuple(matrices), index)

    @property
    def elements(self) -> tuple[AntiHermitian, ...]:
        return self._elements
    
    @property
    def vectors(self) -> tuple[Vector, ...]:
        return tuple([basis_vector(self, i) for i in range(self.dim)])
    
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

    def items(self) -> Generator[tuple[str, AntiHermitian], None, None]:
        for label, element in zip(self.labels, self.elements):
            yield label, element


def special_unitary_basis(ndim: int) -> LieBasis:
    if ndim < 2:
        raise ValueError("Dimension must be >= to 2")

    def E(i: int, j: int) -> SquareMatrix:
        M = np.zeros((ndim, ndim), dtype=complex)
        M[i, j] = 1.0
        M = jnp.array(M)
        assert is_square_matrix(M)
        return M
    
    def S(i: int, j: int) -> AntiHermitian:
        M = 1j * (E(i, j) + E(j, i))
        assert is_anti_hermitian(M)
        return M

    def A(i: int, j: int) -> AntiHermitian:
        M = E(i, j) - E(j, i)
        assert is_anti_hermitian(M)
        return M
    
    def H(k: int) -> AntiHermitian:
        if not (1 <= k <= ndim - 1):
            raise ValueError(f"k must be in [1, {ndim - 1}]")
        d = np.zeros(ndim, dtype=complex)
        d[:k] = 1.0
        d[k] = - k
        d = d * np.sqrt(2) / np.sqrt(k * (k + 1))
        M = 1j * jnp.diag(d)
        assert is_anti_hermitian(M)
        return M

    elements: dict[str, AntiHermitian] = {}
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
