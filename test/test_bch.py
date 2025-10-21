from pulseseek.basis import special_unitary_basis
from pulseseek.algebra import lie_algebra, lie_projection, lie_bracket
from pulseseek.bch import (
    baker_campbell_hausdorff,
    baker_campbell_hausdorff_series,
    _baker_campbell_hausdorff_series_direct,
)
from pulseseek.types import is_anti_hermitian

import jax.numpy as jnp
from scipy.linalg import expm, logm


def test_baker_campbell_hausdorff():
    su2 = lie_algebra(special_unitary_basis(2))
    project = lie_projection(su2)
    iX, iY, iZ = su2.basis.elements

    # Compute the product e^x e^y using the BCH expansion
    eps = 1e-1
    ex, ey, ez = su2.basis.vectors
    terms = baker_campbell_hausdorff(su2)(eps * ex, eps * ey)

    # We can compute the first few terms analytically
    assert jnp.allclose(terms[0], eps * (ex + ey))
    assert jnp.allclose(terms[1], -(eps**2) * ez)
    assert jnp.allclose(terms[2], -(eps**3) / 3 * (ex + ey))
    assert jnp.allclose(terms[3], 0 * ez)
    assert jnp.allclose(terms[4], -(eps**5) / 15 * (ex + ey))
    assert jnp.allclose(terms[5], 2 * eps**6 / 45 * ez)

    er = sum(terms)

    # compute the product e^x e^y using the Pauli matrix representation
    R = jnp.array(logm(expm(iX * eps) @ expm(iY * eps)))
    assert is_anti_hermitian(R)
    r = project(R)
    assert jnp.allclose(er, r, atol=eps**10)


def test_baker_campbell_hausdorff_series_direct():
    su2 = lie_algebra("su2")
    bracket = lie_bracket(su2)
    series = baker_campbell_hausdorff_series(bracket, max_order=5)
    eps = 1e-1
    ex, ey, ez = su2.basis.vectors

    terms = [fn(eps * ex, eps * ey) for fn in series]
    direct = [fn(eps * ex, eps * ey) for fn in _baker_campbell_hausdorff_series_direct(bracket)]

    for a, b in zip(terms, direct):
        assert jnp.allclose(a, b)


def test_baker_campbell_hausdorff_series_lifting():
    su2 = lie_algebra(special_unitary_basis(2))
    bracket = lie_bracket(su2)
    ex, ey, ez = su2.basis.vectors

    def lifting(x, degree: int):
        return jnp.column_stack([x for _ in range(degree)])

    eps = 1e-1
    series = baker_campbell_hausdorff_series(bracket, mode="lifting")

    terms = []
    for i, fn in enumerate(series):
        print(f"Order: {i + 1}")
        terms.append(fn(lifting(eps * ex, i + 1), lifting(eps * ey, i + 1)))

    # terms = tuple(fn(lifting(eps * ex, i + 1), lifting(eps * ey, i + 1)) for i, fn in enumerate(series))

    # We can compute the first few terms analytically
    assert jnp.allclose(terms[0], eps * (ex + ey))
    assert jnp.allclose(terms[1], -(eps**2) * ez)
    assert jnp.allclose(terms[2], -(eps**3) / 3 * (ex + ey))
    assert jnp.allclose(terms[3], 0 * ez)
    assert jnp.allclose(terms[4], -(eps**5) / 15 * (ex + ey))
    assert jnp.allclose(terms[5], 2 * eps**6 / 45 * ez)
