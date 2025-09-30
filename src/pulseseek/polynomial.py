import numpy as np
import jax.numpy as jnp
from math import factorial
from dataclasses import dataclass

from typing import Any, Iterable
from .algebra import LieAlgebra, lie_bracket
from .types import Vector, is_vector


#
# Let Y(t) = t Y_1 + t^2 Y_2 + t^3 Y_3 + ... be a polynomial where the
# coefficients Y_n are terms in a Lie algebra.  The exponential map of
# Y(t) is an element of the group.  We are interested in computing
# `toggling` transformations of exp(Y(t)) of the form
#
#       exp(tX) exp(Y(t)) exp(-tX) = exp(Z(t))
#
# where Z(t) is another Lie polynomial in the same form as Y(t). Recognize that
# the toggling transformation is a special case of the transformation
#
#       g exp(Y) g^{-1} = exp( Ad_g Y )
#
# which in this case gives
#
#       exp(tX) exp(Y(t)) exp(-tX) = exp( Ad_{e^{tX}} Y(t) )
#
#                                  = exp( exp( t ad_X ) ) Y(t) ).
#
# Taking the logarithm of both sides gives an expression for Z(t)
#
#       Z(t) = exp(t ad_X ) Y(t)
#
#       Z(t) = ( \sum_n \frac{t^n}{n!} (ad_X)^n ) Y(t).
#
# Expanding this product and grouping the terms by powers of t gives us an
# expression for the terms of Z(t)
#
#       Z_m = \sum_{n = 1}^{m} \frac{1}{(m - n)!} (ad_{X})^{m - n} (Y_n)
#
# This last expression for the polynomial terms Z_m is essentially the secret
# sauce for the whole technique.  If e^{Y(t)} is a toggling frame propagator,
# and in the toggling frame I accumulate error from a new pulse e^{tX}, the
# propagator after applying the new pulse is exactly e^{Z(t)}, which we can
# compute analytically and to arbitrary accuracy with the above polynomial
# expansion.
#

def _lie_polynomial_toggle_term(y: "LiePolynomial", x: Vector, m: int) -> Vector:
    algebra = y.algebra
    dim = algebra.dim
    bracket = lie_bracket(algebra)
    
    def ad_x_n(x: Vector, n: int, y: Vector) -> Vector:
        """Compute ad_X^n (y)"""
        p = y.copy()
        assert is_vector(p)
        for _ in range(n):
            p = bracket(x, p)
        return p

    zero = jnp.zeros((dim,), dtype=float)
    assert is_vector(zero, dimension=dim)
    
    z_m = zero
    for n in range(1, m + 1):
        if (n - 1) < len(y.coeffs):
            yn = y.coeffs[n - 1]
            ad_x_y = ad_x_n(x, m - n, yn)
            z_m += ad_x_y / factorial(m - n)
        
        # Otherwise the yn coefficient is zero and doesn't contribute

    return z_m


@dataclass(frozen=True)
class LiePolynomial:
    """A Lie algebra polynomial representation."""

    algebra: LieAlgebra
    degree: int
    coeffs: tuple[Vector, ...]

    @classmethod
    def new(cls, algebra: LieAlgebra, degree: int, coefficients: Iterable[Any]):
        dim = algebra.dim
        coeffs: list[Vector] = []

        for zn in coefficients:
            if not is_vector(zn, dimension=dim):
                raise ValueError(f"Not a {dim}-d vector {zn}")
            coeffs.append(zn)
        if len(coeffs) < 1:
            raise ValueError("No coefficients provided")

        return LiePolynomial(algebra, degree, tuple(coeffs))

    def evaluate(self, t: float) -> Vector:
        dim = self.algebra.dim
        z = np.zeros((dim,), dtype=float)
        assert is_vector(z, dimension=dim)
        for zn in reversed(self.coeffs):
            z = zn + t * z
        return t * z

    def __call__(self, t: float) -> Vector:
        return self.evaluate(t)
    
    def toggle(self, x: Vector) -> "LiePolynomial":
        coeffs = tuple([_lie_polynomial_toggle_term(self, x, m) for m in range(1, self.degree + 1)])
        return LiePolynomial.new(self.algebra, self.degree, coeffs)



# BilinearMap = Callable[[Vector, Vector], Vector]


# def baker_campbell_hausdorff(bracket: BilinearMap) -> tuple[BilinearMap, ...]:
#     """Build functions for the first five terms of the BCH formula

#     Args:
#         bracket (BilinearMap): the Lie bracket

#     Returns:
#         tuple[BilinearMap, ...]: tuple of functions implementing the terms
#     """

#     def bch_1(x: Vector, y: Vector) -> Vector:
#         return x + y
    
#     def bch_2(x: Vector, y: Vector) -> Vector:
#         return 1.0 / 2 * bracket(x, y)
    
#     def bch_3(x: Vector, y: Vector) -> Vector:
#         return 1.0 / 12 * (
#             bracket(x, bracket(x ,y)) + bracket(y, bracket(y, x))
#         )
    
#     def bch_4(x: Vector, y: Vector) -> Vector:
#         return - 1.0 / 24 * (
#             bracket(y, bracket(x, bracket(x, y)))
#         )
    
#     def bch_5(x: Vector, y: Vector) -> Vector:
#         return (
#             - 1.0 / 720 * (
#                 bracket(y, bracket(y, bracket(y, bracket(y, x)))) +
#                 bracket(x, bracket(x, bracket(x, bracket(x, y))))
#             )
#             + 1.0 / 360 * (
#                 bracket(x, bracket(y, bracket(y, bracket(y, x)))) +
#                 bracket(y, bracket(x, bracket(x, bracket(x, y))))
#             )
#             + 1.0 / 120 * (
#                 bracket(y, bracket(x, bracket(y, bracket(x, y)))) +
#                 bracket(x, bracket(y, bracket(x, bracket(y, x))))
#             )
#         )
    
#     return bch_1, bch_2, bch_3, bch_4, bch_5