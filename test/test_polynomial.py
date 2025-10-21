import numpy as np

from pulseseek.algebra import lie_algebra, lie_projection
from pulseseek.polynomial import LiePolynomial, lie_polynomial_bch_product

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
    bch_product = lie_polynomial_bch_product(su2, max_order=6)

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

