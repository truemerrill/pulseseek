from dataclasses import dataclass
from math import factorial
from typing import Any, Callable, Mapping, Generator, Iterable, overload

import numpy as np

from .basis import LieBasis
from .types import (
    AntiHermitian,
    AntiSymmetricTensor,
    Hermitian,
    SquareMatrix,
    Vector,
    is_antisymmetric_tensor,
    is_hermitian,
    is_square_matrix,
    is_vector,
)

InnerProduct = Callable[[AntiHermitian, AntiHermitian], float]
Bracket = Callable[[AntiHermitian, AntiHermitian], AntiHermitian]


def hilbert_schmidt_inner_product(X: AntiHermitian, Y: AntiHermitian) -> float:
    return float((-np.trace(X @ Y)).real)


def matrix_commutator(X: SquareMatrix, Y: SquareMatrix) -> SquareMatrix:
    return X @ Y - Y @ X


@overload
def basis_vector(b: int, index: int) -> Vector: ...
@overload
def basis_vector(b: LieBasis, index: int) -> Vector: ...


def basis_vector(b: int | LieBasis, index: int) -> Vector:
    """Construct a Lie basis vector

    Args:
        b (int | LieBasis): the basis or the rank of the algebra
        index (int): the basis vector index

    Returns:
        Vector: the basis vector
    """
    dim = b.ndim if isinstance(b, LieBasis) else b
    e = np.zeros((dim,), dtype=float)
    e[index] = 1.0
    assert is_vector(e)
    return e


NameElement = Callable[[int], str]


def lie_closure(
    elements: Mapping[str, Any],
    bracket: Bracket = matrix_commutator,
    name_element: NameElement = lambda idx: f"_A{idx}",
    max_rank: int = 100
) -> LieBasis:

    def in_span(x: AntiHermitian, vectors: Iterable[AntiHermitian], atol=1e-12) -> bool:
        vecs = tuple(vectors)
        if len(vecs) == 0:
            return False

        V = np.column_stack([M.reshape(-1) for M in vecs])
        X = x.reshape(-1)

        rank_V = np.linalg.matrix_rank(V, tol=atol)
        rank_aug = np.linalg.matrix_rank(np.column_stack([V, X]), tol=atol)
        return rank_V == rank_aug
    
    def brackets(vectors: Iterable[AntiHermitian]) -> Generator[AntiHermitian, None, None]:
        for a, x in enumerate(vectors):
            for b, y in enumerate(vectors):
                if a != b:
                    yield bracket(x, y)
    
    def new_elements(vectors: Iterable[AntiHermitian], max_rank: int) -> dict[str, AntiHermitian]:
        v = list(vectors)
        elements: dict[str, AntiHermitian] = {}
        idx = 0
        
        for E in brackets(v):
            if not in_span(E, v):
                v.append(E)
                name = name_element(idx)
                idx += 1
                elements[name] = E

            if len(v) > max_rank:
                raise ValueError("Maximum rank exceeded, Lie algebra may be infinite rank")
        return elements
                
    def linearly_independent_elements(elements: Mapping[str, AntiHermitian]) -> dict[str, AntiHermitian]:
        independent: dict[str, AntiHermitian] = {}
        for name, E in elements.items():
            if not in_span(E, independent.values()):
                independent[name] = E
        return independent

    U = linearly_independent_elements(elements)
    V = new_elements(U.values(), max_rank)
    closure = {**U, **V}

    while True:
        V = new_elements(closure.values(), max_rank)
        if len(V) == 0:
            break
        closure = {**closure, **V}
    
    return LieBasis.new(closure)


def gram_matrix(
    basis: LieBasis, inner_product: InnerProduct = hilbert_schmidt_inner_product
) -> Hermitian:
    """Calculate the Gram matrix of a Lie basis

    Args:
        basis (LieBasis): the Lie basis
        inner_product (InnerProduct, optional): the inner product. Defaults to
            hilbert_schmidt_inner_product.

    Returns:
        Hermitian: the Gram matrix
    """
    m = basis.dim
    G = np.zeros((m, m), dtype=float)
    for a in range(m):
        for b in range(m):
            A = basis[a]
            B = basis[b]
            G[a, b] = inner_product(A, B)
    assert is_hermitian(G, dimension=m)
    return G


def structure_constants(
    basis: LieBasis,
    inner_product: InnerProduct = hilbert_schmidt_inner_product,
    bracket: Bracket = matrix_commutator,
) -> AntiSymmetricTensor:
    """Calculate the structure constant tensor for a Lie basis

    Note:
        The elements of the structure constant tensor are defined as

        .. math::

            F_{a b c} = \\langle E_a, [ E_b, E_c ] \\rangle

        where E_a, E_b, and E_c are Lie basis elements.  Using the Einstein
        summation notation, if we expand two arbitrary elements `P`, `Q` in
        terms of the Lie basis `P = P_i E_i` and `Q = Q_j E_j`, then their
        bracket may be written as

        .. math::

            [P, Q] = P_i Q_i F_{k i j} E_k

    Args:
        basis (LieBasis): the Lie basis
        inner_product (InnerProduct, optional): the inner product. Defaults to
            hilbert_schmidt_inner_product.
        bracket (Bracket, optional): the Lie bracket. Defaults to
            matrix_commutator.

    Returns:
        AntiSymmetricTensor: the structure constant tensor
    """
    m = basis.dim
    S = np.zeros((m, m, m), dtype=float)
    G = gram_matrix(basis, inner_product)
    Ginv = np.linalg.inv(G)

    for b, B in enumerate(basis.elements):
        for c, C in enumerate(basis.elements):
            ad_B_C = bracket(B, C)
            for a, A in enumerate(basis.elements):
                S[a, b, c] = inner_product(A, ad_B_C)

    F = np.einsum("ad,dbc->abc", Ginv, S, optimize=True)
    assert is_antisymmetric_tensor(F, dimension=m)
    return F


@dataclass(frozen=True)
class LieAlgebra:
    _G: Hermitian
    _F: AntiSymmetricTensor

    @classmethod
    def new(
        cls,
        basis: LieBasis,
        inner_product: InnerProduct = hilbert_schmidt_inner_product,
        bracket: Bracket = matrix_commutator,
    ):
        G = gram_matrix(basis, inner_product)
        F = structure_constants(basis, inner_product, bracket)
        return cls(G, F)

    @property
    def gram_matrix(self) -> Hermitian:
        return self._G

    @property
    def structure_constants(self) -> AntiSymmetricTensor:
        return self._F

    def inner_product(self, a: Vector, b: Vector) -> float:
        """Compute the inner product of two vectors using the Gram matrix

        Args:
            a (Vector): a Lie algebra vector
            b (Vector): a Lie algebra vector

        Returns:
            float: the inner product
        """
        p = np.einsum("i,ij,j->", a, self.gram_matrix, b, optimize=True)
        assert p.ndim == 0
        return float(p)

    def bracket(self, a: Vector, b: Vector) -> Vector:
        """Compute the Lie bracket of two vectors using the structure constants

        Args:
            a (Vector): a Lie algebra vector
            b (Vector): a Lie algebra vector

        Returns:
            Vector: the Lie bracket
        """
        P = np.einsum("kij,i,j->k", self.structure_constants, a, b, optimize=True)
        assert is_vector(P)
        return P

    def adjoint(self, a: Vector) -> SquareMatrix:
        """Compute the matrix representation of the adjoint endomorphism ad_{a}

        Args:
            a (Vector): a Lie algebra vector

        Returns:
            SquareMatrix: the adjoint matrix.  Usually this is anti-Hermitian
                but that may depend on the inner product.
        """
        A = np.einsum("kij,i->kj", self.structure_constants, a)
        assert is_square_matrix(A)
        return A

    def adjoint_action(
        self, a: Vector, b: Vector, atol: float = 1e-12, max_order: int = 1000
    ) -> Vector:
        """Compute Ad_{exp(a)}(b) via a truncated Lie bracket expansion

        Note:
            If `a`, `b` are Lie algebra vectors and `A = exp(a)`, then

            .. math::

                A^\\dagger b A = Ad_{A}(b) = Ad_{exp(a)}(b)

        Args:
            a (Vector): Lie algebra vector
            b (Vector): Lie algebra vector
            atol (float, optional): the absolute tolerance for convergence.
                Defaults to 1e-12.
            max_order (int, optional): the maximum order of the Lie bracket
                series. Defaults to 1000.

        Raises:
            ValueError: if the series fails to converge

        Returns:
            Vector: the result of the adjoint action
        """
        x = b.copy()
        term = b.copy()
        converged = False

        for r in range(1, max_order + 1):
            term = self.bracket(a, term)
            incr = term / float(factorial(r))
            norm = float(np.sqrt(self.inner_product(incr, incr)))
            x = x + incr

            if norm < atol:
                converged = True
                break

        if not converged:
            raise ValueError(f"Failed to converge after {max_order} iterations")
        return x
