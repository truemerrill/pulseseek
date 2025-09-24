import numpy as np
from scipy.linalg import expm

from qsail.types import is_vector, is_anti_hermitian, is_hermitian
from qsail.basis import special_unitary_basis
from qsail.algebra import LieAlgebra, basis_vector, hilbert_schmidt_inner_product, matrix_commutator, lie_closure


def test_algebra_inner_product():
    su2_basis = special_unitary_basis(2)
    su2 = LieAlgebra.new(su2_basis)

    E1, E2, E3 = su2_basis.elements
    e1, e2, e3 = tuple([basis_vector(3, i) for i in range(3)])

    assert su2.inner_product(e1, e1) == hilbert_schmidt_inner_product(E1, E1)
    assert su2.inner_product(e1, e2) == hilbert_schmidt_inner_product(E1, E2)

    X = E1 + 2 * E3
    Y = E2 - E3

    x = e1 + 2 * e3
    y = e2 - e3

    assert su2.inner_product(x, y) == hilbert_schmidt_inner_product(X, Y)


def test_algebra_bracket():
    su2_basis = special_unitary_basis(2)
    su2 = LieAlgebra.new(su2_basis)

    E1, E2, E3 = su2_basis.elements
    e1, e2, e3 = tuple([basis_vector(3, i) for i in range(3)])

    assert np.isclose(
        su2.inner_product(e3, su2.bracket(e1, e2)),
        hilbert_schmidt_inner_product(E3, matrix_commutator(E1, E2))
    )

    # Try to take the bracket of completely random matrices
    np.random.seed(42)

    for _ in range(20):
        a = np.random.normal(size=3)
        b = np.random.normal(size=3)

        assert is_vector(a)
        assert is_vector(b)

        A = a[0] * E1 + a[1] * E2 + a[2] * E3
        B = b[0] * E1 + b[1] * E2 + b[2] * E3

        assert np.isclose(
            su2.inner_product(e1, su2.bracket(a, b)),
            hilbert_schmidt_inner_product(E1, matrix_commutator(A, B))
        )
        assert np.isclose(
            su2.inner_product(e2, su2.bracket(a, b)),
            hilbert_schmidt_inner_product(E2, matrix_commutator(A, B))
        )
        assert np.isclose(
            su2.inner_product(e3, su2.bracket(a, b)),
            hilbert_schmidt_inner_product(E3, matrix_commutator(A, B))
        )


def test_algebra_adjoint():
    su2_basis = special_unitary_basis(2)
    su2 = LieAlgebra.new(su2_basis)

    e1, e2, e3 = tuple([basis_vector(3, i) for i in range(3)])

    ad_e1 = su2.adjoint(e1)
    assert (ad_e1 @ e1 == 0).all()  # type: ignore
    assert (ad_e1 @ e2 == - 2 * e3).all()  # type: ignore
    assert (ad_e1 @ e3 == 2 * e2).all()  # type: ignore


def test_algebra_adjoint_action():
    su3_basis = special_unitary_basis(3)
    su3 = LieAlgebra.new(su3_basis)
    m = su3_basis.dim

    E = np.array(su3_basis.elements)

    # Take the adjoint action on random members of SU(3)
    np.random.seed(42)

    for _ in range(20):
        a = np.random.normal(size=m)
        b = np.random.normal(size=m)

        assert is_vector(a)
        assert is_vector(b) 

        # Generate the matrix representation of a, b as anti-hermitian matrices
        A = np.einsum("kij,k->ij", E, a)
        B = np.einsum("kij,k->ij", E, b)
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
        ad = su3.adjoint_action(- a, b)
        C = np.einsum("kij,k->ij", E, ad)
        assert is_anti_hermitian(C)

        assert np.isclose(1j * C, Ht, atol=1e-12).all()


def test_lie_closure():
    # Generate su2
    X = 1j * np.array([[0, 1], [1, 0]])
    Y = 1j * np.array([[0, -1j], [1j, 0]])

    closure_basis = lie_closure({"X": X, "Y": Y})
    assert closure_basis.dim == 3
    
    # Generate su4
    X = 1j * np.diag([-3/2, -1/2, 0, 2])
    Y = 1j * np.array([[0, 1, 0, 0],
                       [1, 0, 1 + 1j, 0],
                       [0, 1 - 1j, 0, 1],
                       [0, 0, 1, 0]])
    
    closure_basis = lie_closure({"X": X, "Y": Y})
    assert closure_basis.dim == 15
    for x in closure_basis.elements:
        assert np.trace(x) == 0
