import numpy as np
import jax.numpy as jnp
from scipy.linalg import expm

from pulseseek.types import is_vector, is_anti_hermitian, is_hermitian
from pulseseek.basis import special_unitary_basis, basis_vector
from pulseseek.algebra import (
    hilbert_schmidt_inner_product,
    matrix_commutator,
    lie_algebra,
    lie_closure,
    lie_inner_product,
    lie_bracket,
    lie_adjoint_action
)


def test_algebra_inner_product():
    su2_basis = special_unitary_basis(2)
    su2 = lie_algebra(su2_basis)
    inner_product = lie_inner_product(su2)

    E1, E2, E3 = su2_basis.elements
    e1, e2, e3 = tuple([basis_vector(3, i) for i in range(3)])

    assert inner_product(e1, e1) == hilbert_schmidt_inner_product(E1, E1)
    assert inner_product(e1, e2) == hilbert_schmidt_inner_product(E1, E2)

    X = E1 + 2 * E3
    Y = E2 - E3

    x = e1 + 2 * e3
    y = e2 - e3

    assert inner_product(x, y) == hilbert_schmidt_inner_product(X, Y)


def test_algebra_bracket():
    su2_basis = special_unitary_basis(2)
    su2 = lie_algebra(su2_basis)
    inner_product = lie_inner_product(su2)
    bracket = lie_bracket(su2)

    E1, E2, E3 = su2_basis.elements
    e1, e2, e3 = tuple([basis_vector(3, i) for i in range(3)])

    assert np.isclose(
        inner_product(e3, bracket(e1, e2)),
        hilbert_schmidt_inner_product(E3, matrix_commutator(E1, E2)),
    )

    # Try to take the bracket of completely random matrices
    np.random.seed(42)

    for _ in range(20):
        a = jnp.array(np.random.normal(size=3))
        b = jnp.array(np.random.normal(size=3))

        assert is_vector(a)
        assert is_vector(b)

        A = a[0] * E1 + a[1] * E2 + a[2] * E3
        B = b[0] * E1 + b[1] * E2 + b[2] * E3

        assert np.isclose(
            inner_product(e1, bracket(a, b)),
            hilbert_schmidt_inner_product(E1, matrix_commutator(A, B)),
        )
        assert np.isclose(
            inner_product(e2, bracket(a, b)),
            hilbert_schmidt_inner_product(E2, matrix_commutator(A, B)),
        )
        assert np.isclose(
            inner_product(e3, bracket(a, b)),
            hilbert_schmidt_inner_product(E3, matrix_commutator(A, B)),
        )


def test_algebra_adjoint_action():
    su3_basis = special_unitary_basis(3)
    su3 = lie_algebra(su3_basis)
    m = su3_basis.dim
    Ad = lie_adjoint_action(su3)

    E = np.array(su3_basis.elements)

    # Take the adjoint action on random members of SU(3)
    np.random.seed(42)

    for _ in range(20):
        a = jnp.array(np.random.normal(size=m))
        b = jnp.array(np.random.normal(size=m))

        assert is_vector(a)
        assert is_vector(b)

        # Generate the matrix representation of a, b as anti-hermitian matrices
        A = jnp.einsum("kij,k->ij", E, a)
        B = jnp.einsum("kij,k->ij", E, b)
        assert is_anti_hermitian(A)
        assert is_anti_hermitian(B)

        # The Hamiltonian-like operator
        H = 1j * B
        assert is_hermitian(H)

        # The propagator-like operator
        U = expm(A)

        # The toggled Hamiltonian
        Ht = U.conj().T @ H @ U
        assert is_hermitian(Ht)

        # Compute the same toggling transformation using the adjoint action
        ad = Ad(-a, b)
        C = jnp.einsum("kij,k->ij", E, ad)
        assert is_anti_hermitian(C)

        assert np.isclose(1j * C, Ht, atol=1e-12).all()


def test_lie_closure():
    # Generate su2
    X = 1j * jnp.array([[0, 1], [1, 0]])
    Y = 1j * jnp.array([[0, -1j], [1j, 0]])

    closure_basis = lie_closure({"X": X, "Y": Y})
    assert closure_basis.dim == 3

    # Generate su4
    X = 1j * jnp.array(np.diag([-3 / 2, -1 / 2, 0, 2]))
    Y = 1j * jnp.array(
        [[0, 1, 0, 0], [1, 0, 1 + 1j, 0], [0, 1 - 1j, 0, 1], [0, 0, 1, 0]]
    )

    closure_basis = lie_closure({"X": X, "Y": Y})
    assert closure_basis.dim == 15
    for x in closure_basis.elements:
        assert np.trace(x) == 0
