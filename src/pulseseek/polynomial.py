import jax
import jax.numpy as jnp
import functools

from dataclasses import dataclass
from typing import Callable, NamedTuple, Iterator
from .algebra import LieAlgebra, lie_bracket, lie_adjoint_action
from .bch import BCH_MAX_ORDER, baker_campbell_hausdorff_compile, BCHLifting, BCHLiftingMap
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


@dataclass(frozen=True)
class BCHLiftingPlanTerm:
    """A single BCH multilinear lifting term \\( F_{p, q}(X_r, Y_s) \\)
    
    Attributes:
        r (jax.Array): indices for the multilinear lifting of x
        s (jax.Array): indices for the multilinear lifting of y
        p (int): the degree of the multilinear lifting of x
        q (int): the degree of the multilinear lifting of y
        F (BCHLiftingMap): the lifted BCH term function
    """
    r: jax.Array
    s: jax.Array
    p: int
    q: int
    F: BCHLiftingMap


@dataclass(frozen=True)
class BCHLiftingPlanOrder:
    """Collection of all terms that contribute to order"""
    order: int
    terms: tuple[BCHLiftingPlanTerm, ...]


@dataclass(frozen=True)
class BCHLiftingPlan:
    """Collection of all terms up to and including max_order"""
    max_order: int
    orders: tuple[BCHLiftingPlanOrder, ...]

    @classmethod
    def new(
        cls,
        Z: dict[tuple[int, int], BCHLiftingMap],
        max_order: int
    ):
        orders: list[BCHLiftingPlanOrder] = []
        for n in range(1, max_order + 1):
            terms: list[BCHLiftingPlanTerm] = []
            for r, s in _iter_lifting_indices(n):
                p, q = len(r), len(s)
                F_pq = Z.get((p, q))
                if F_pq is None:
                    continue

                ra = jnp.asarray(r, dtype=jnp.int32)
                sa = jnp.asarray(s, dtype=jnp.int32)
                terms.append(BCHLiftingPlanTerm(ra, sa, p, q, F_pq))
            orders.append(BCHLiftingPlanOrder(n, tuple(terms)))
        return cls(max_order, tuple(orders))


LiePolynomialProduct = Callable[[LiePolynomial, LiePolynomial], LiePolynomial]

DEFAULT_LIE_POLYNOMIAL_PRODUCT_ORDER = 6


@functools.cache
def lie_polynomial_bch_product(
    algebra: LieAlgebra,
    max_order: int = DEFAULT_LIE_POLYNOMIAL_PRODUCT_ORDER
) -> LiePolynomialProduct:
    """Compile the BCH product between Lie polynomials

    !!! note
        Given two Lie polynomials `x(t)` and `y(t)`, their BCH product is
        defined as

        $$ log(exp(x(t)) exp(y(t))) = Z(x(t), y(t)) $$

        where `Z` is defined by the Baker-Campbell-Hausdorff series.

    Args:
        algebra (LieAlgebra): the Lie algebra
        max_order (int, optional): the truncation order of the BCH series. 
            Defaults to DEFAULT_LIE_POLYNOMIAL_PRODUCT_ORDER.

    Returns:
        LiePolynomialProduct: the function computing the BCH product.
    """
    bracket = lie_bracket(algebra)
    zero = algebra.basis.zero
    Z = baker_campbell_hausdorff_compile(bracket, max_order, "lifting")

    # Build a static plan so there are no dict lookups inside jit
    plan = BCHLiftingPlan.new(Z, max_order)

    @jax.jit
    def product(x: LiePolynomial, y: LiePolynomial) -> LiePolynomial:
        coeffs: list[LieVector] = []
        for order_plan in plan.orders:
            wn = zero
            for term_plan in order_plan.terms:
                xl = _lifting(x, term_plan.r)
                yl = _lifting(y, term_plan.s)
                F = term_plan.F
                wn = wn + F(xl, yl)
            coeffs.append(wn)
        return LiePolynomial(tuple(coeffs))

    return product
