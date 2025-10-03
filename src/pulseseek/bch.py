import csv
import functools
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from typing import Callable, Iterable, Iterator, Literal

import jax
from jax import numpy as jnp

from .algebra import LieBracket, LieAlgebra, lie_bracket
from .types import Vector

BilinearMap = Callable[[Vector, Vector], Vector]
BCHSeries = tuple[BilinearMap, ...]
BCHTerms = tuple[Vector, ...]
BCHOperation = Literal["X", "Y", "BR"]


"""
Baker-Campbell-Hausdorff (BCH) series

The BCH series is a series expansion for `z = log(exp(x) exp(y))` in terms of
x, y, and their repeated Lie brackets.  The series is given by

.. math::

    z = \\sum_{n > 1} Z_n(x, y)

where the index `n` runs over the expansion terms.  Although a closed form
expression for the `Z_n(x, y)` was discovered by Dykin, this form is difficult
to use in practice.  More commonly, the expansion is truncated after only a few
terms (perhaps as little as 4) since higher-order terms grow quickly in
complexity with the expansion order.

In `pulseseek`, we take a slightly different approach for the BCH expansion.
First, we decompose each term into a sum of terms that are each homogeneous
in the input parameters,

.. math::

    Z_n(x, y) = \\sum_{p + q = n} Z_{(p, q)}(x, y)

where for all `Z_{(p, q)}`

.. math::

    Z_{(p, q)}(a x, b y) = a^p b^q Z_{(p, q)}(x, y).

The advantage of this decomposition is it permits certain algebraic
rearrangements of the expansion terms that are not possible in the standard BCH
expansion.

To manage the extensive book-keeping required to write each `Z_{(p, q)}`
function, we use an auxiliary computer algebra system (Sage) to produce a CSV
file which encodes each `Z_{(p, q)}` up to and including BCH order n = 15.
This data file is included in the `pulseseek` package.

Later during runtime, when the user supplies a Lie bracket function, this
module's `baker_campbell_hausdorff_compile` function parses the CSV data and
produces a sequence set of `Z_{(p, q)}` functions.  Each `Z_{(p, q)}` function
is `jax.jit` compiled for efficiency and speed.
"""

BCH_MAX_ORDER = 15


@dataclass
class BCHMonomial:
    order: int
    degree_x: int
    degree_y: int
    coeff: Fraction
    term: str


def _iter_bch_monomial(max_order: int = BCH_MAX_ORDER) -> Iterator[BCHMonomial]:
    """Iterate over the BCH monomial terms in the data file

    Args:
        max_order (int, optional): the maximum expansion order. Defaults to 15.

    Yields:
        Iterator[BCHMonomial]: the BCHMonomial terms.
    """
    if max_order > BCH_MAX_ORDER:
        raise ValueError(
            "Expansion order is greater than the maximum allowed order "
            f"{BCH_MAX_ORDER}."
        )

    data = files("pulseseek").joinpath("bch.csv")
    with data.open("r", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            monomial = BCHMonomial(
                order=int(row["order"]),
                degree_x=int(row["degree_x"]),
                degree_y=int(row["degree_y"]),
                coeff=Fraction(row["coefficient"]),
                term=row["term"],
            )

            if monomial.order > max_order:
                break

            yield monomial


def _iter_ops(term: str) -> Iterator[BCHOperation]:
    depth = 0
    for char in term:
        if char == "[":
            depth += 1
        if char == "]":
            depth += -1
            yield "BR"
        if char == "X":
            yield "X"
        if char == "Y":
            yield "Y"
    if depth != 0:
        raise ValueError("Unclosed bracket")


def _get_ops(term: str) -> tuple[BCHOperation, ...]:
    """Parse BCH term into a sequence of RPN operations"""
    return tuple(_iter_ops(term))


@functools.cache
def _compile_ops(bracket: LieBracket, ops: Iterable[BCHOperation]) -> BilinearMap:
    """Parse the RPN operations, building a callable."""

    def x(x: Vector, _: Vector) -> Vector:
        return x

    def y(_: Vector, y: Vector) -> Vector:
        return y

    def br(left: BilinearMap, right: BilinearMap) -> BilinearMap:
        def apply(x: Vector, y: Vector) -> Vector:
            return bracket(left(x, y), right(x, y))

        return apply

    stack: list[BilinearMap] = []
    for op in ops:
        if op == "X":
            stack.append(x)
        elif op == "Y":
            stack.append(y)
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(br(a, b))
    if len(stack) != 1:
        raise RuntimeError("BCH stack not singular.")
    return stack[0]


BCHPolynomial = tuple[BCHMonomial, ...]


def _get_polynomial_order_degree(polynomial: BCHPolynomial) -> tuple[int, int, int]:
    if len(polynomial) < 1:
        raise RuntimeError("No terms in BCH polynomial")
    term = polynomial[0]

    for monomial in polynomial:
        if (
            (monomial.order != term.order)
            or (monomial.degree_x != term.degree_x)
            or (monomial.degree_y != monomial.degree_y)
        ):
            raise RuntimeError("Inconsistent BCH polynomial terms")
    return term.order, term.degree_x, term.degree_y


def _compile_polynomial(bracket: LieBracket, polynomial: BCHPolynomial) -> BilinearMap:
    # Reduce the polynomial by combining coefficients on like terms
    reduced: dict[tuple[BCHOperation, ...], Fraction] = {}
    for monomial in polynomial:
        ops = _get_ops(monomial.term)
        reduced[ops] = reduced.get(ops, Fraction(0, 1)) + monomial.coeff

    monomial_fns = [
        (coeff, _compile_ops(bracket, ops))
        for ops, coeff in reduced.items()
        if coeff != 0
    ]

    @jax.jit
    def poly_fn(x: Vector, y: Vector) -> Vector:
        z = jnp.zeros(x.shape)
        for coeff, term_fn in monomial_fns:
            term = float(coeff) * term_fn(x, y)
            z += term
        return z

    return poly_fn


def _iter_bch_polynomial(max_order: int = BCH_MAX_ORDER) -> Iterator[BCHPolynomial]:
    current_order = 0
    monomials: dict[tuple[int, int], list[BCHMonomial]] = {}

    for monomial_term in _iter_bch_monomial(max_order):
        if monomial_term.order != current_order:
            if current_order > 0:
                yield from [tuple(t) for t in monomials.values()]
                monomials = {}

        degree = (monomial_term.degree_x, monomial_term.degree_y)
        if degree not in monomials:
            monomials[degree] = []
        monomials[degree].append(monomial_term)

    yield from [tuple(t) for t in monomials.values()]


@functools.cache
def baker_campbell_hausdorff_compile(
    bracket: LieBracket, max_order: int = BCH_MAX_ORDER
) -> dict[tuple[int, int], BilinearMap]:
    """Compile BCH Z_{(p, q)} functions using the Lie bracket

    Args:
        bracket (LieBracket): the Lie bracket operation.
        max_order (int, optional): the maximum BCH expansion order. Defaults to
           BCH_MAX_ORDER.

    Returns:
        dict[tuple[int, int], BilinearMap]: A dictionary of JIT compiled
            functions. The keys are tuples `(p, q)` and the values are bilinear
            mapping functions on the Lie algebra.
    """
    fns: dict[tuple[int, int], BilinearMap] = {}
    _compile_ops.cache_clear()

    for polynomial in _iter_bch_polynomial(max_order):
        _, degree_x, degree_y = _get_polynomial_order_degree(polynomial)
        polynomial_fn = _compile_polynomial(bracket, polynomial)
        fns[(degree_x, degree_y)] = polynomial_fn

    # Clear the cache after the compiling pass
    _compile_ops.cache_clear()
    return fns


@functools.cache
def baker_campbell_hausdorff_series(
    bracket: LieBracket, max_order: int = BCH_MAX_ORDER
) -> BCHSeries:

    Z = baker_campbell_hausdorff_compile(bracket, max_order)

    def iter_pq(n: int) -> Iterator[tuple[int, int]]:
        for p in range(n + 1):
            q = n - p
            yield p, q

    def bch_fn(n: int) -> BilinearMap:
        terms: list[BilinearMap] = []
        for pq in iter_pq(n):
            if pq in Z:
                terms.append(Z[pq])

        @jax.jit
        def fn(x: Vector, y: Vector) -> Vector:
            z = 0 * x
            for z_pq in terms:
                z += z_pq(x, y)
            return z
        return fn
    
    return tuple(bch_fn(n) for n in range(1, max_order + 1))


def baker_campbell_hausdorff(
    algebra: LieAlgebra, order: int = 8
) -> Callable[[Vector, Vector], BCHTerms]:
    """Generate a function that computes the Baker-Campbell-Hausdorff series.

    Args:
        algebra (LieAlgebra): the Lie algebra
        order (int, optional): the maximum order. Defaults to 8.

    Raises:
        ValueError: raised if the series order is invalid

    Returns:
        Callable[[Vector, Vector], BCHTerms]: function computing the series
    """
    bracket = lie_bracket(algebra)
    series = baker_campbell_hausdorff_series(bracket)

    if order + 1 > len(series):
        raise ValueError("order is greater than the maximum series order")

    @jax.jit
    def bch(x: Vector, y: Vector) -> BCHTerms:
        return tuple(Zm(x, y) for Zm in series[0:order])

    return bch
