import jax
import jax.numpy as jnp
import functools

from typing import Callable, NamedTuple
from .algebra import LieAlgebra, lie_adjoint_action
from .types import LieVector, is_vector


class LiePolynomial(NamedTuple):
    """A polynomial with coefficients on the Lie algebra

    !!! note

        A Lie polynomial is a polynomial of the form 
        \\( Y(t) = \\sum_{n > 0} t^n Y_n \\), where the coefficients `Y_n` are
        members of the Lie algebra.
    """

    coeffs: tuple[LieVector, ...]

    def evaluate(self, t: float) -> LieVector:
        """Evaluate the polynomial using a Horner scheme

        Args:
            t (float): the independent variable

        Returns:
            Vector: the evaluated value
        """
        if len(self.coeffs) < 1:
            raise ValueError("No coefficients")
        shape = self.coeffs[0].shape

        z = jnp.zeros(shape, dtype=float)
        assert is_vector(z)
        for zn in reversed(self.coeffs):
            z = zn + t * z
        return t * z

    def __call__(self, t: float) -> LieVector:
        return self.evaluate(t)


@functools.cache
def lie_polynomial_adjoint_action(
    algebra: LieAlgebra, atol: float = 1e-12, max_terms: int = 1000
) -> Callable[[LieVector, LiePolynomial], LiePolynomial]:
    """Compute the adjoint action / toggling transformation of a polynomial

    !!! note
        If `x` is a Lie algebra vector and `y(t)` is a Lie polynomial then

        $$ \\exp(x) y(t) \\exp(-x) = \\sum_n t^n Ad_{\\exp(x)} y_n = z(t) $$
        
        is the adjoint action of the polynomial.  The adjoint action of a
        polynomial is a polynomial formed from the adjoint action of each
        of the coefficients `y_n`.

    Args:
        algebra (LieAlgebra): the Lie algebra
        atol (float, optional): the absolute tolerance. Defaults to 1e-12.
        max_terms (int, optional): the maximum number of terms. Defaults to
            1000.

    Returns:
        LiePolynomial: the adjoint action
    """
    ad = lie_adjoint_action(algebra, atol=atol, max_terms=max_terms)

    @jax.jit
    def adjoint_action_polynomial(x: LieVector, y: LiePolynomial) -> LiePolynomial:
        coeffs = tuple(ad(x, c) for c in y.coeffs)
        return LiePolynomial(coeffs)
    
    return adjoint_action_polynomial


"""Lie polynomial Baker-Campbell-Hausdorff (BCH) products.


Let `x` be a Lie algebra vector and `y(t)` a Lie polynomial.  The left BCH
product of `x` and `y(t)` is given by

.. math::

    w_l(t) = log(exp(t^k x) exp(y(t))).

Similarly the right BCH product is

.. math::

    w_r(t) = log(exp(y(t) exp(t^k x))).

The functions `w_r(t)` and `w_l(t)` are themselves Lie polynomials. We can see
this from the BCH series

.. math::

    w_l(t) = \\sum_{n > 1} Z_n ( t^k x, y(t) ) 
    
           = \\sum_{n > 1} \\sum_{m > 1} Z_n ( t^k x, t^m y_m )

           = \\sum_{n > 1} \\sum_{m > 1} \\sum_{p + q = n} t^{m + k} Z_{(p, q)}( x, y_m ).

Then grouping terms by powers of `t`

.. math::

    w_l(t) = \\sum_{m > 1} t^{m + k} \\left[ \\sum_{n > 1} \\sum_{p + q = n} Z_{(p, q)}( x, y_m ) \\right].

The terms in the brackets are precisely the coefficients of the polynomial
`w_l(t)`.  Following similar arguments, we see the coefficients of both
polynomials are

.. math::

    w_{l,(m + k)} = \\sum_{n > 1} \\sum_{p + q = n} Z_{(p, q)}( x, y_m )

    w_{r,(m + k)} = \\sum_{n > 1} \\sum_{p + q = n} Z_{(p, q)}( y_m, x )  
"""


def lie_polynomial_lmult(algebra: LieAlgebra):

    @jax.jit
    def lmult(x: LieVector, y: LiePolynomial) -> LiePolynomial:
        ...

    return lmult


def lie_polynomial_rmult(algebra: LieAlgebra):

    @jax.jit
    def rmult(y: LiePolynomial, z: LieVector) -> LiePolynomial:
        ...
    
    return rmult