import jax
import jax.numpy as jnp
import functools

from typing import Callable, NamedTuple, Iterator
from .algebra import LieAlgebra, lie_bracket, lie_adjoint_action
from .bch import BCH_MAX_ORDER, baker_campbell_hausdorff_compile, BCHLifting
from .util import sumsto
from .types import LieVector, is_vector


class LiePolynomial(NamedTuple):
    """A polynomial with coefficients on the Lie algebra

    !!! note

        A Lie polynomial is a polynomial of the form
        \\( Y(t) = \\sum_{n > 0} t^n Y_n \\), where the coefficients `Y_n` are
        members of the Lie algebra.
    """

    coeffs: tuple[LieVector, ...]

    def degree(self) -> int:
        return len(self.coeffs)

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


@jax.jit
def _lifting(polynomial: LiePolynomial, indices: tuple[int, ...]) -> BCHLifting:
    coeffs = polynomial.coeffs
    if len(coeffs) == 0:
        return jnp.zeros((0, 0), dtype=jnp.float32)

    mat = jnp.stack(coeffs, axis=1)
    idx = jnp.asarray(indices, dtype=jnp.int32) - 1
    lifted = jnp.take(mat, idx, axis=1, fill_value=0)
    return lifted


def _iter_lifting_indices(order: int) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    for degree in range(1, order + 1):
        for indices in sumsto(order, degree):
            for k in range(degree + 1):
                yield tuple(indices[:k]), tuple(indices[k:])


LiePolynomialProduct = Callable[[LiePolynomial, LiePolynomial], LiePolynomial]


def lie_polynomial_bch_product(
    algebra: LieAlgebra,
    max_order: int = BCH_MAX_ORDER
) -> LiePolynomialProduct:
    bracket = lie_bracket(algebra)
    zero = algebra.basis.zero
    Z = baker_campbell_hausdorff_compile(bracket, max_order, "lifting")

    # @jax.jit
    def product_term(x: LiePolynomial, y: LiePolynomial, n: int) -> LieVector:
        wn = zero
        for r, s in _iter_lifting_indices(n):
            p, q = len(r), len(s)
            
            # Check for the F_pq function
            F_pq = Z.get((p, q))
            if not F_pq:
                continue

            xl, yl = _lifting(x, r), _lifting(y, s)
            term = F_pq(xl, yl)
            wn += term
        return wn
    

    def product(x: LiePolynomial, y: LiePolynomial) -> LiePolynomial:
        coeffs = tuple(product_term(x, y, n) for n in range(1, max_order + 1))
        return LiePolynomial(coeffs)

    return product
