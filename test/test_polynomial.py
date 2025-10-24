import numpy as np
import pytest
from pulseseek.basis import special_unitary_basis
from pulseseek.algebra import lie_algebra, lie_projection
from pulseseek.polynomial import (
    LiePolynomial,
    lie_polynomial_bch_product,
    lie_polynomial_adjoint_action,
)

import jax.numpy as jnp
from scipy.linalg import expm, logm


def test_polynomial_bch_product_analytic():
    su2 = lie_algebra("su2")
    bch_product = lie_polynomial_bch_product(su2)

    ex, ey, ez = su2.basis.vectors
    x = LiePolynomial.new(ex, ey, ez)
    y = LiePolynomial.new(ey, ez, ex)

    # Compute the product
    w = bch_product(x, y)

    # Compare term by term to analytic formula
    assert np.allclose(w.coeffs[0], ex + ey)
    assert np.allclose(w.coeffs[1], ey)
    assert np.allclose(w.coeffs[2], 2 / 3 * (ex + ey) + ez)


def test_polynomial_bch_product_numerical():
    su2 = lie_algebra("su2")
    project = lie_projection(su2)
    bch_product = lie_polynomial_bch_product(su2)

    # Construct two polynomials that in general do not commute
    vectors = su2.basis.vectors
    x = LiePolynomial.new(*vectors)
    y = LiePolynomial.new(*reversed(vectors))

    # Compute the product
    w = bch_product(x, y)

    # Starting at the identity (t = 0) and exploring the curves x(t) and y(t),
    # compute the BCH product and compare to the terms of the polynomial w(t).
    for t in map(float, np.arange(0.01, 0.1, 0.01)):
        iXt = su2.basis.matrix(x(t))
        iYt = su2.basis.matrix(y(t))

        # Compute product e^iXt e^iYt using matrix representation
        R = jnp.array(logm(expm(iXt) @ expm(iYt)))
        r = project(R)
        wt = w(t)

        assert jnp.allclose(r, wt, atol=1e-12)


@pytest.mark.bench
def test_polynomial_adjoint_action_benchmark(compile_then_benchmark):
    su2 = lie_algebra("su2")
    polynomial_ad = lie_polynomial_adjoint_action(su2)
    ex, ey, ez = su2.basis.vectors

    y = LiePolynomial.new(ex, 2 * ey, 3 * ez)
    compile_then_benchmark(polynomial_ad, ex, y)


@pytest.mark.bench
@pytest.mark.parametrize("n", [2, 4, 8])
def test_polynomial_bch_product_scan_dimension_benchmark(compile_then_benchmark, n):
    algebra = lie_algebra(special_unitary_basis(n))

    x = LiePolynomial.new(*algebra.basis.vectors)
    y = LiePolynomial.new(*reversed(algebra.basis.vectors))

    bch_product = lie_polynomial_bch_product(algebra)
    compile_then_benchmark(bch_product, x, y)


@pytest.mark.bench
@pytest.mark.parametrize("product_degree", [1, 2, 3, 4, 5, 6])
def test_polynomial_bch_product_scan_product_degree_benchmark(
    compile_then_benchmark, product_degree
):
    su2 = lie_algebra("su2")

    x = LiePolynomial.new(*su2.basis.vectors)
    y = LiePolynomial.new(*reversed(su2.basis.vectors))

    bch_product = lie_polynomial_bch_product(
        su2, left_degree=3, right_degree=3, product_degree=product_degree
    )
    compile_then_benchmark(bch_product, x, y)
