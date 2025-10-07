import csv
import functools
from dataclasses import dataclass
from fractions import Fraction
from importlib.resources import files
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    TypeVar,
    Generic,
    overload,
)

import jax
from jax import numpy as jnp

from .algebra import LieBracket, LieAlgebra, lie_bracket
from .types import Vector, Matrix


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

BilinearMap = Callable[[Vector, Vector], Vector]
BCHTerms = tuple[Vector, ...]
BCHOperationFlag = Literal["X", "Y", "BR"]
BCHSlots = Matrix
BilinearSlotsMap = Callable[[BCHSlots, BCHSlots], Vector]
BCHCompiledSeries = tuple[BilinearSlotsMap, ...]

BCH_MAX_ORDER = 15


@dataclass
class BCHMonomial:
    order: int
    degree_x: int
    degree_y: int
    coeff: Fraction
    term: str


@dataclass
class BCHOperation:
    flag: BCHOperationFlag
    index: int | None

    def __hash__(self) -> int:
        return hash((self.flag, self.index))


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
    num_x, num_y = 0, 0

    for char in term:
        if char == "[":
            depth += 1
        if char == "]":
            depth += -1
            yield BCHOperation("BR", None)
        if char == "X":
            yield BCHOperation("X", num_x)
            num_x += 1
        if char == "Y":
            yield BCHOperation("Y", num_y)
            num_y += 1
    if depth != 0:
        raise ValueError("Unclosed bracket")


def _get_ops(term: str) -> tuple[BCHOperation, ...]:
    """Parse BCH term into a sequence of RPN operations"""
    return tuple(_iter_ops(term))


T = TypeVar("T", BilinearMap, BilinearSlotsMap)


class BCHCompilerPrimitives(NamedTuple, Generic[T]):
    compile_x: Callable[[BCHOperation], T]
    compile_y: Callable[[BCHOperation], T]
    compile_br: Callable[[BCHOperation, T, T], T]


def _bch_primitives(bracket: BilinearMap) -> BCHCompilerPrimitives[BilinearMap]:
    def compile_x(op: BCHOperation) -> BilinearMap:
        assert op.flag == "X"

        def x(x: Vector, _: Vector) -> Vector:
            return x

        return x

    def compile_y(op: BCHOperation) -> BilinearMap:
        assert op.flag == "Y"

        def y(_: Vector, y: Vector) -> Vector:
            return y

        return y

    def compile_br(
        op: BCHOperation, left: BilinearMap, right: BilinearMap
    ) -> BilinearMap:
        assert op.flag == "BR"

        def br(x: Vector, y: Vector) -> Vector:
            return bracket(left(x, y), right(x, y))

        return br

    return BCHCompilerPrimitives(compile_x, compile_y, compile_br)


def _bch_slots_primitives(
    bracket: BilinearMap,
) -> BCHCompilerPrimitives[BilinearSlotsMap]:
    def compile_x(op: BCHOperation) -> BilinearSlotsMap:
        assert op.flag == "X" and op.index is not None
        index = op.index

        def x(x: BCHSlots, y: BCHSlots) -> Vector:
            xi = x[index]
            return xi

        return x

    def compile_y(op: BCHOperation) -> BilinearSlotsMap:
        assert op.flag == "Y" and op.index is not None
        index = op.index

        def y(_: BCHSlots, y: BCHSlots) -> Vector:
            yi = y[index]
            return yi

        return y

    def compile_br(
        op: BCHOperation, left: BilinearSlotsMap, right: BilinearSlotsMap
    ) -> BilinearSlotsMap:
        assert op.flag == "BR" and op.index is None

        def br(x: BCHSlots, y: BCHSlots) -> Vector:
            return bracket(left(x, y), right(x, y))

        return br

    return BCHCompilerPrimitives(compile_x, compile_y, compile_br)


def _compile_ops(
    primitives: BCHCompilerPrimitives[T], ops: Iterable[BCHOperation]
) -> T:
    """Parse the RPN operations, building a callable."""
    compile_x, compile_y, compile_br = primitives

    stack: list[T] = []
    for op in ops:
        if op.flag == "X":
            stack.append(compile_x(op))
        elif op.flag == "Y":
            stack.append(compile_y(op))
        else:
            b = stack.pop()
            a = stack.pop()
            stack.append(compile_br(op, a, b))
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


def _compile_polynomial(
    primitives: BCHCompilerPrimitives[T], polynomial: BCHPolynomial
) -> T:
    # Reduce the polynomial by combining coefficients on like terms
    reduced: dict[tuple[BCHOperation, ...], Fraction] = {}
    for monomial in polynomial:
        ops = _get_ops(monomial.term)
        reduced[ops] = reduced.get(ops, Fraction(0, 1)) + monomial.coeff

    monomial_fns = [
        (coeff, _compile_ops(primitives, ops))
        for ops, coeff in reduced.items()
        if coeff != 0
    ]

    @jax.jit
    def poly_fn(x, y):
        z = jnp.zeros(x.shape)
        for coeff, term_fn in monomial_fns:
            term = float(coeff) * term_fn(x, y)
            z += term
        return z

    return poly_fn  # type: ignore


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


@overload
def baker_campbell_hausdorff_compile(
    bracket: LieBracket,
    max_order: int = ...,
    mode: Literal["standard"] = ...,
) -> dict[tuple[int, int], BilinearMap]: ...


@overload
def baker_campbell_hausdorff_compile(
    bracket: LieBracket,
    max_order: int = ...,
    mode: Literal["slot"] = ...,
) -> dict[tuple[int, int], BilinearSlotsMap]: ...


@functools.cache
def baker_campbell_hausdorff_compile(
    bracket: LieBracket,
    max_order: int = BCH_MAX_ORDER,
    mode: Literal["standard", "slot"] = "standard",
) -> dict[tuple[int, int], T]:
    """Compile BCH Z_{(p, q)} functions using the Lie bracket

    Args:
        bracket (LieBracket): the Lie bracket.
        max_order (int, optional): the maximum BCH expansion order. Defaults to
            BCH_MAX_ORDER.
        mode (str, optional): the mode of compilation. Defaults to "standard".

    Returns:
        dict[tuple[int, int], T]: A dictionary of JIT compiled
            functions. The keys are tuples `(p, q)` and the values are bilinear
            mapping functions on the Lie algebra.
    """
    primitives = (
        _bch_primitives(bracket)
        if mode == "standard"
        else _bch_slots_primitives(bracket)
    )
    fns = {}

    for polynomial in _iter_bch_polynomial(max_order):
        _, degree_x, degree_y = _get_polynomial_order_degree(polynomial)
        polynomial_fn = _compile_polynomial(primitives, polynomial)
        fns[(degree_x, degree_y)] = polynomial_fn

    return fns


@overload
def baker_campbell_hausdorff_series(
    bracket: BilinearMap,
    max_order: int = ...,
    mode: Literal["standard"] = ...,
) -> tuple[BilinearMap, ...]: ...


@overload
def baker_campbell_hausdorff_series(
    bracket: BilinearSlotsMap,
    max_order: int = ...,
    mode: Literal["slot"] = ...,
) -> tuple[BilinearSlotsMap, ...]: ...


@functools.cache
def baker_campbell_hausdorff_series(
    bracket: LieBracket,
    max_order: int = BCH_MAX_ORDER,
    mode: Literal["standard", "slot"] = "standard",
) -> tuple:
    Z = baker_campbell_hausdorff_compile(bracket, max_order, mode)

    def iter_pq(n: int) -> Iterator[tuple[int, int]]:
        for p in range(n + 1):
            q = n - p
            yield p, q

    def bch_fn_standard(n: int) -> BilinearMap:
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

    def bch_fn_slot(n: int) -> BilinearSlotsMap:
        terms: list[BilinearSlotsMap] = []
        for pq in iter_pq(n):
            if pq in Z:
                terms.append(Z[pq])

        @jax.jit
        def fn(x: BCHSlots, y: BCHSlots) -> Vector:
            z = 0 * x[:, 0]
            for z_pq in terms:
                z += z_pq(x, y)
            return z

        return fn

    if mode == "standard":
        return tuple(bch_fn_standard(n) for n in range(1, max_order + 1))
    else:
        return tuple(bch_fn_slot(n) for n in range(1, max_order + 1))


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


def _baker_campbell_hausdorff_series_direct(
    bracket: LieBracket,
) -> tuple[BilinearMap, ...]:
    """Build functions for the first five terms of the BCH formula

    Note:
        This function is used for benchmarking and should not be directly used.

    Args:
        bracket (LieBracket): the Lie bracket

    Returns:
        tuple[BilinearMap, ...]: tuple of functions implementing the terms
    """

    @jax.jit
    def bch_1(x: Vector, y: Vector) -> Vector:
        return x + y

    @jax.jit
    def bch_2(x: Vector, y: Vector) -> Vector:
        return 1.0 / 2 * bracket(x, y)

    @jax.jit
    def bch_3(x: Vector, y: Vector) -> Vector:
        return 1.0 / 12 * (bracket(x, bracket(x, y)) + bracket(y, bracket(y, x)))

    @jax.jit
    def bch_4(x: Vector, y: Vector) -> Vector:
        return -1.0 / 24 * (bracket(y, bracket(x, bracket(x, y))))

    @jax.jit
    def bch_5(x: Vector, y: Vector) -> Vector:
        return (
            -1.0
            / 720
            * (
                bracket(y, bracket(y, bracket(y, bracket(y, x))))
                + bracket(x, bracket(x, bracket(x, bracket(x, y))))
            )
            + 1.0
            / 360
            * (
                bracket(x, bracket(y, bracket(y, bracket(y, x))))
                + bracket(y, bracket(x, bracket(x, bracket(x, y))))
            )
            + 1.0
            / 120
            * (
                bracket(y, bracket(x, bracket(y, bracket(x, y))))
                + bracket(x, bracket(y, bracket(x, bracket(y, x))))
            )
        )

    return bch_1, bch_2, bch_3, bch_4, bch_5
