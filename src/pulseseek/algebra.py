import functools
from typing import (
    Any,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    overload
)

import jax
import jax.numpy as jnp
import numpy as np

from .basis import LieBasis, fock_basis, heisenburg_basis, special_unitary_basis
from .types import (
    SquareMatrix,
    AntiSymmetricTensor,
    Hermitian,
    Scalar,
    SquareMatrix,
    LieVector,
    is_antisymmetric_tensor,
    is_hermitian,
    is_square_matrix,
)
from .util import hash_array

# Required to increase numerical precision
jax.config.update("jax_enable_x64", True)


# --- Matrix representation ---------------------------------------------------

MatrixInnerProduct = Callable[[SquareMatrix, SquareMatrix], Scalar]
MatrixBracket = Callable[[SquareMatrix, SquareMatrix], SquareMatrix]


def hilbert_schmidt_inner_product(X: SquareMatrix, Y: SquareMatrix) -> Scalar:
    return jnp.trace(X.T.conj() @ Y)


def matrix_commutator(X: SquareMatrix, Y: SquareMatrix) -> SquareMatrix:
    return X @ Y - Y @ X


NameElement = Callable[[int], str]


def lie_closure(
    elements: Mapping[str, Any],
    bracket: MatrixBracket = matrix_commutator,
    name_element: NameElement = lambda idx: f"_A{idx}",
    max_rank: int = 100,
) -> LieBasis:
    """Construct a basis closed under the Lie bracket that spans the elements.

    Args:
        elements (Mapping[str, Any]): the elements to span.  The keys should be
            the names of the elements and the values should be the elements
            themselves.  The elements must be square matrices.
        bracket (MatrixBracket, optional): the Lie bracket. Defaults to
            matrix_commutator.
        name_element (NameElement, optional): A callback function that is used
            to generate names of created algebra elements. Defaults to
            `lambda idx: f"_A{idx}"`.
        max_rank (int, optional): the maximum rank of the generated Lie
            algebra.  If the rank exceeds max_rank, the function raises a
            ValueError.  Defaults to 100.

    Raises:
        ValueError: if the rank of the generated algebra exceeds max_rank

    Returns:
        LieBasis: the spanning basis
    """

    def in_span(x: SquareMatrix, vectors: Iterable[SquareMatrix], atol=1e-12) -> bool:
        """Check if x is in the span of vectors."""
        vecs = tuple(vectors)
        if len(vecs) == 0:
            return False

        V = np.column_stack([M.reshape(-1) for M in vecs])
        X = x.reshape(-1)

        rank_V = np.linalg.matrix_rank(V, tol=atol)
        rank_aug = np.linalg.matrix_rank(np.column_stack([V, X]), tol=atol)
        return rank_V == rank_aug

    def brackets(
        vectors: Iterable[SquareMatrix],
    ) -> Generator[SquareMatrix, None, None]:
        """Iterate over Lie brackets."""
        for a, x in enumerate(vectors):
            for b, y in enumerate(vectors):
                if a != b:
                    yield bracket(x, y)

    def new_elements(
        vectors: Iterable[SquareMatrix], max_rank: int
    ) -> dict[str, SquareMatrix]:
        """Generate new elements in one round of repeated Lie brackets."""
        v = list(vectors)
        elements: dict[str, SquareMatrix] = {}
        idx = 0

        for E in brackets(v):
            if not in_span(E, v):
                v.append(E)
                name = name_element(idx)
                idx += 1
                elements[name] = E

            if len(v) > max_rank:
                raise ValueError(
                    "Maximum rank exceeded, Lie algebra may be infinite rank"
                )
        return elements

    def linearly_independent_elements(
        elements: Mapping[str, SquareMatrix],
    ) -> dict[str, SquareMatrix]:
        """Select the linearly independent elements."""
        independent: dict[str, SquareMatrix] = {}
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
    basis: LieBasis, inner_product: MatrixInnerProduct = hilbert_schmidt_inner_product
) -> Hermitian:
    """Calculate the Gram matrix of a Lie basis

    Args:
        basis (LieBasis): the Lie basis
        inner_product (MatrixInnerProduct, optional): the inner product.
            Defaults to hilbert_schmidt_inner_product.

    Returns:
        Hermitian: the Gram matrix
    """
    m = basis.dim
    G = np.zeros((m, m), dtype=complex)
    for a in range(m):
        for b in range(m):
            A = basis[a]
            B = basis[b]
            G[a, b] = inner_product(A, B)
    Gr = jnp.array(G)
    assert is_hermitian(Gr, dimension=m)
    return Gr


def structure_constants(
    basis: LieBasis,
    inner_product: MatrixInnerProduct = hilbert_schmidt_inner_product,
    bracket: MatrixBracket = matrix_commutator,
) -> AntiSymmetricTensor:
    """Calculate the structure constant tensor for a Lie basis

    !!! note

        The elements of the structure constant tensor are defined as

        $$ F_{a b c} = \\langle E_a, [ E_b, E_c ] \\rangle $$

        where E_a, E_b, and E_c are Lie basis elements.  Using the Einstein
        summation notation, if we expand two arbitrary elements `P`, `Q` in
        terms of the Lie basis `P = P_i E_i` and `Q = Q_j E_j`, then their
        bracket may be written as

        $$ [P, Q] = P_i Q_i F_{k i j} E_k $$

    Args:
        basis (LieBasis): the Lie basis
        inner_product (MatrixInnerProduct, optional): the inner product.
            Defaults to hilbert_schmidt_inner_product.
        bracket (MatrixBracket, optional): the Lie bracket. Defaults to
            matrix_commutator.

    Returns:
        AntiSymmetricTensor: the structure constant tensor
    """
    m = basis.dim
    S = np.zeros((m, m, m), dtype=complex)
    G = gram_matrix(basis, inner_product)
    Ginv = jnp.linalg.inv(G)

    for b, B in enumerate(basis.elements):
        for c, C in enumerate(basis.elements):
            ad_B_C = bracket(B, C)
            for a, A in enumerate(basis.elements):
                S[a, b, c] = inner_product(A, ad_B_C)

    F = jnp.einsum("ad,dbc->abc", Ginv, S, optimize=True)
    assert is_antisymmetric_tensor(F, dimension=m)
    return F


# --- Lie algebra representation ----------------------------------------------

LieInnerProduct = Callable[[LieVector, LieVector], Scalar]
LieBracket = Callable[[LieVector, LieVector], LieVector]
LieProjection = Callable[[SquareMatrix], LieVector]
LieExponential = Callable[[LieVector], SquareMatrix]
LieLogarithm = Callable[[SquareMatrix], LieVector]


class LieAlgebra(NamedTuple):
    basis: LieBasis
    G: Hermitian
    F: AntiSymmetricTensor

    def __hash__(self) -> int:
        return hash((self.basis, hash_array(self.G), hash_array(self.F)))

    def __eq__(self, other: Hashable) -> bool:
        return hash(self) == hash(other)

    @property
    def dim(self) -> int:
        return self.G.shape[0]



def _lie_algebra_explicit_su2() -> LieAlgebra:
    return _lie_algebra_implicit(special_unitary_basis(2))


def _lie_algebra_explicit_heisenberg() -> LieAlgebra:
    return _lie_algebra_implicit(heisenburg_basis())


def _lie_algebra_explicit_heisenberg_fock(ndim: int = 15) -> LieAlgebra:
    basis = fock_basis(ndim=ndim)
    m = ndim * (ndim - 1) / 2
    G = jnp.array([[m, 0, 0], [0, m, 0], [0, 0, ndim]])
    F = jnp.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        ]
    )
    return LieAlgebra(basis, G, F)


def _lie_algebra_implicit(
    basis: LieBasis,
    *,
    inner_product: MatrixInnerProduct = hilbert_schmidt_inner_product,
    bracket: MatrixBracket = matrix_commutator,
) -> LieAlgebra:
    """Construct a LieAlgebra from a basis, inner_product, and bracket

    Args:
        basis (LieBasis): the Lie basis set in the matrix representation.
        inner_product (MatrixInnerProduct, optional): the inner product.
            Defaults to hilbert_schmidt_inner_product.
        bracket (MatrixBracket, optional): the bracket. Defaults to
            matrix_commutator.

    Returns:
        LieAlgebra: the Lie algebra
    """
    G = gram_matrix(basis, inner_product)
    F = structure_constants(basis, inner_product, bracket)
    return LieAlgebra(basis, G, F)


@overload
def lie_algebra(
    basis: LieBasis,
    *,
    inner_product: MatrixInnerProduct = hilbert_schmidt_inner_product,
    bracket:MatrixBracket = matrix_commutator
) -> LieAlgebra: ...

@overload
def lie_algebra(
    basis: Literal["su2"],
) -> LieAlgebra: ...

@overload
def lie_algebra(
    basis: Literal["heisenberg"]
) -> LieAlgebra: ...

@overload
def lie_algebra(
    basis: Literal["heisenberg-fock"],
    *,
    ndim: int = 15
) -> LieAlgebra: ...

def lie_algebra(
    basis,
    **kwargs
) -> LieAlgebra:
    if isinstance(basis, LieBasis):
        return _lie_algebra_implicit(basis, **kwargs)
    else:
        if basis == "su2":
            return _lie_algebra_explicit_su2()
        elif basis == "heisenberg":
            return _lie_algebra_explicit_heisenberg()
        elif basis == "heisenberg-fock":
            return _lie_algebra_explicit_heisenberg_fock(**kwargs)
        raise ValueError(f"Unknown Lie algebra: {str(basis)}")


@functools.cache
def lie_projection(
    algebra: LieAlgebra,
    inner_product: MatrixInnerProduct = hilbert_schmidt_inner_product,
) -> LieProjection:
    Ginv = jnp.linalg.inv(algebra.G)
    assert is_square_matrix(Ginv)

    @jax.jit
    def project(matrix: SquareMatrix) -> LieVector:
        h = jnp.array([inner_product(matrix, E) for E in algebra.basis.elements])
        r = Ginv @ h
        return r

    return project


@functools.cache
def lie_exponential(algebra: LieAlgebra) -> LieExponential:
    E = jnp.array(algebra.basis.elements)

    @jax.jit
    def exp(x: LieVector) -> SquareMatrix:
        X = jnp.einsum("i,ijk->jk", x, E)
        return jax.scipy.linalg.expm(X)

    return exp


@functools.cache
def lie_inner_product(algebra: LieAlgebra) -> LieInnerProduct:
    """Construct the inner product function for the algebra

    Args:
        algebra (LieAlgebra): the Lie algebra

    Returns:
        LieInnerProduct: the inner product
    """
    G = algebra.G

    @jax.jit
    def inner_product(x: LieVector, y: LieVector) -> Scalar:
        ip = x @ G @ y
        return ip

    return inner_product


@functools.cache
def lie_bracket(algebra: LieAlgebra) -> LieBracket:
    """Construct the Lie bracket function for the algebra

    Args:
        algebra (LieAlgebra): the Lie algebra

    Returns:
        LieBracket: the lie bracket
    """
    F = algebra.F

    @jax.jit
    def bracket(x: LieVector, y: LieVector) -> LieVector:
        z = jnp.einsum("kij,i,j->k", F, x, y, optimize=True)
        return z

    return bracket


class LieAdjointCarry(NamedTuple):
    result: LieVector
    term: LieVector
    n: jax.Array


@functools.cache
def lie_adjoint_action(
    algebra: LieAlgebra, atol: float = 1e-12, max_terms: int = 1000
) -> LieBracket:
    """Construct the adjoint action of a Lie algebra

    !!! note

        If `a`, `b` are Lie algebra vectors then

        $$ \\exp(a) b \\exp(-a) = Ad_{\\exp(a)}(b) $$

        is the adjoint action.

    Args:
        algebra (LieAlgebra): the Lie algebra
        atol (float, optional): the absolute tolerance. Defaults to 1e-12.
        max_terms (int, optional): the maximum number of terms. Defaults to
            1000.

    Returns:
        LieBracket: the adjoint action function
    """
    inner_product = lie_inner_product(algebra)
    bracket = lie_bracket(algebra)

    @jax.jit
    def adjoint_action_horner(x: LieVector, y: LieVector) -> LieVector:
        """Computes the Taylor series of Ad_{exp(x)} y using Horner's method.

        Args:
            x (Vector): Lie algebra vector
            y (Vector): Lie algebra vector

        Returns:
            Vector: the adjoint action
        """
        eps = jnp.array(atol)

        def body(carry: LieAdjointCarry) -> LieAdjointCarry:
            n = carry.n + 1
            term = bracket(x, carry.term) / n
            result = carry.result + term
            return LieAdjointCarry(result, term, n)

        def cond(carry: LieAdjointCarry) -> jax.Array:
            norm = jnp.sqrt(inner_product(carry.term, carry.term))
            is_not_converged = jnp.array(norm >= eps).reshape(())
            is_not_done = jnp.array(carry.n < max_terms).reshape(())
            return jnp.logical_and(is_not_converged, is_not_done)

        n = jnp.array([0])
        initial = LieAdjointCarry(y, y, n)
        final = jax.lax.while_loop(cond, body, initial)
        return final.result

    return adjoint_action_horner
