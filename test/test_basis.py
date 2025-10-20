import numpy as np
import jax.numpy as jnp
from pulseseek.basis import LieBasis, special_unitary_basis, fock_basis
from pulseseek.algebra import gram_matrix, structure_constants


def test_su2_basis():
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])
    su2 = LieBasis.new({
        "iX": 1j * X,
        "iY": 1j * Y,
        "iZ": 1j * Z
    })
    G = gram_matrix(su2)

    assert G.ndim == 2
    assert G.shape == (3, 3)
    assert np.allclose(G, np.eye(3) * 2)


def test_special_unitary_basis():
    
    for n in range(2, 5):
        m = (n * n) - 1
        su_n = special_unitary_basis(n)
        assert su_n.dim == m
        G = gram_matrix(su_n)
        assert np.allclose(G, np.eye(m) * 2)


def levi_civita(*idx: int) -> int:
    # scalar ε_{i1...in} (0-based indices)
    if len(set(idx)) != len(idx):
        return 0
    inv = 0
    for a in range(len(idx)):
        for b in range(a+1, len(idx)):
            inv += idx[a] > idx[b]
    return -1 if (inv % 2) else 1


def test_su2_structure_constants():
    su2 = special_unitary_basis(2)
    F = structure_constants(su2)
    m = su2.dim

    for a in range(m):
        for b in range(m):
            for c in range(m):
                 f = F[a, b, c]
                 np.allclose(f, - 2 * levi_civita(a, b, c))


def test_su2_non_orthanormal_structure_constants():
    X = jnp.array([[0, 1], [1, 0]])
    Y = jnp.array([[0, -1j], [1j, 0]])
    Z = jnp.array([[1, 0], [0, -1]])

    A1 = 1j * X
    A2 = 1j * Y
    A3 = 1j * (Z + X)

    su2 = LieBasis.new({
        'A1': A1,
        'A2': A2,
        'A3': A3
    })

    F = structure_constants(su2)
    assert F[0, 1, 2] == - 4.0


def test_fock_basis():
    ndim = 5
    basis = fock_basis(ndim=ndim)
    a, ad, identity = basis.elements
    N = np.diag(range(ndim))
    
    assert np.isclose(ad @ a, N).all()
    assert np.trace(identity) == ndim
