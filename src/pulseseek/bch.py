from typing import Callable
import jax

from .algebra import LieAlgebra, lie_bracket
from .types import Vector


BilinearMap = Callable[[Vector, Vector], Vector]
BCHSeries = tuple[BilinearMap, ...]
BCHTerms = tuple[Vector, ...]


def baker_campbell_hausdorff_series(br: BilinearMap) -> BCHSeries:
    """Generate first nine terms in the Baker-Campbell-Hausdorff expansion

    Note:
        For two Lie algebra elements X and Y, the BCH expansion is

        .. math::

            Z = log(e^X e^Y) = \\sum_{n = 1}^{\\infty} Z_n(X, Y)

        where the Z_n are functions that depend only on X, Y and their Lie
        brackets.  Each Z_n is homogeneous of degree n, meaning

        .. math::

            Z_n(X, Y) = a^n Z_n(X/a, Y/a)

        This scaling property allows control of truncation error in certain
        numerical methods.

    Note:
        The code below was generated algorithmically from a symbolic
        representation of the BCH expansion terms using `sagemath`.

    Args:
        br (BilenearMap): the Lie bracket function.

    Returns:
        tuple[BilinearMap, ...]: the terms in the series
    """

    @jax.jit
    def bch_1(X: Vector, Y: Vector) -> Vector:
        term = 1 * X + 1 * Y
        return term

    @jax.jit
    def bch_2(X: Vector, Y: Vector) -> Vector:
        term = 1.0 / 2 * br(X, Y)
        return term

    @jax.jit
    def bch_3(X: Vector, Y: Vector) -> Vector:
        term = 1.0 / 12 * br(X, br(X, Y)) + 1.0 / 12 * br(br(X, Y), Y)
        return term

    @jax.jit
    def bch_4(X: Vector, Y: Vector) -> Vector:
        term = 1.0 / 24 * br(X, br(br(X, Y), Y))
        return term

    @jax.jit
    def bch_5(X: Vector, Y: Vector) -> Vector:
        term = (
            -1.0 / 720 * br(X, br(X, br(X, br(X, Y))))
            + 1.0 / 180 * br(X, br(X, br(br(X, Y), Y)))
            + 1.0 / 360 * br(br(X, br(X, Y)), br(X, Y))
            + 1.0 / 180 * br(X, br(br(br(X, Y), Y), Y))
            + 1.0 / 120 * br(br(X, Y), br(br(X, Y), Y))
            + -1.0 / 720 * br(br(br(br(X, Y), Y), Y), Y)
        )
        return term

    @jax.jit
    def bch_6(X: Vector, Y: Vector) -> Vector:
        term = (
            -1.0 / 1440 * br(X, br(X, br(X, br(br(X, Y), Y))))
            + 1.0 / 720 * br(X, br(br(X, br(X, Y)), br(X, Y)))
            + 1.0 / 360 * br(X, br(X, br(br(br(X, Y), Y), Y)))
            + 1.0 / 240 * br(X, br(br(X, Y), br(br(X, Y), Y)))
            + -1.0 / 1440 * br(X, br(br(br(br(X, Y), Y), Y), Y))
        )
        return term

    @jax.jit
    def bch_7(X: Vector, Y: Vector) -> Vector:
        term = (
            1.0 / 30240 * br(X, br(X, br(X, br(X, br(X, br(X, Y))))))
            + -1.0 / 5040 * br(X, br(X, br(X, br(X, br(br(X, Y), Y)))))
            + 1.0 / 10080 * br(X, br(X, br(br(X, br(X, Y)), br(X, Y))))
            + 1.0 / 3780 * br(X, br(X, br(X, br(br(br(X, Y), Y), Y))))
            + 1.0 / 10080 * br(br(X, br(X, br(X, Y))), br(X, br(X, Y)))
            + 1.0 / 1680 * br(X, br(X, br(br(X, Y), br(br(X, Y), Y))))
            + 1.0 / 1260 * br(X, br(br(X, br(br(X, Y), Y)), br(X, Y)))
            + 1.0 / 3780 * br(X, br(X, br(br(br(br(X, Y), Y), Y), Y)))
            + 1.0 / 2016 * br(br(X, br(X, Y)), br(X, br(br(X, Y), Y)))
            + -1.0 / 5040 * br(br(br(X, br(X, Y)), br(X, Y)), br(X, Y))
            + 13.0 / 15120 * br(X, br(br(X, Y), br(br(br(X, Y), Y), Y)))
            + 1.0 / 10080 * br(br(X, br(br(X, Y), Y)), br(br(X, Y), Y))
            + -1.0 / 1512 * br(br(X, br(br(br(X, Y), Y), Y)), br(X, Y))
            + -1.0 / 5040 * br(X, br(br(br(br(br(X, Y), Y), Y), Y), Y))
            + 1.0 / 1260 * br(br(X, Y), br(br(X, Y), br(br(X, Y), Y)))
            + -1.0 / 2016 * br(br(X, Y), br(br(br(br(X, Y), Y), Y), Y))
            + -1.0 / 5040 * br(br(br(X, Y), Y), br(br(br(X, Y), Y), Y))
            + 1.0 / 30240 * br(br(br(br(br(br(X, Y), Y), Y), Y), Y), Y)
        )
        return term

    @jax.jit
    def bch_8(X: Vector, Y: Vector) -> Vector:
        term = (
            1.0 / 60480 * br(X, br(X, br(X, br(X, br(X, br(br(X, Y), Y))))))
            + -1.0 / 15120 * br(X, br(X, br(X, br(br(X, br(X, Y)), br(X, Y)))))
            + -1.0 / 10080 * br(X, br(X, br(X, br(X, br(br(br(X, Y), Y), Y)))))
            + 1.0 / 20160 * br(X, br(br(X, br(X, br(X, Y))), br(X, br(X, Y))))
            + -1.0 / 20160 * br(X, br(X, br(X, br(br(X, Y), br(br(X, Y), Y)))))
            + 1.0 / 2520 * br(X, br(X, br(br(X, br(br(X, Y), Y)), br(X, Y))))
            + 23.0 / 120960 * br(X, br(X, br(X, br(br(br(br(X, Y), Y), Y), Y))))
            + 1.0 / 4032 * br(X, br(br(X, br(X, Y)), br(X, br(br(X, Y), Y))))
            + -1.0 / 10080 * br(X, br(br(br(X, br(X, Y)), br(X, Y)), br(X, Y)))
            + 13.0 / 30240 * br(X, br(X, br(br(X, Y), br(br(br(X, Y), Y), Y))))
            + 1.0 / 20160 * br(X, br(br(X, br(br(X, Y), Y)), br(br(X, Y), Y)))
            + -1.0 / 3024 * br(X, br(br(X, br(br(br(X, Y), Y), Y)), br(X, Y)))
            + -1.0 / 10080 * br(X, br(X, br(br(br(br(br(X, Y), Y), Y), Y), Y)))
            + 1.0 / 2520 * br(X, br(br(X, Y), br(br(X, Y), br(br(X, Y), Y))))
            + -1.0 / 4032 * br(X, br(br(X, Y), br(br(br(br(X, Y), Y), Y), Y)))
            + -1.0 / 10080 * br(X, br(br(br(X, Y), Y), br(br(br(X, Y), Y), Y)))
            + 1.0 / 60480 * br(X, br(br(br(br(br(br(X, Y), Y), Y), Y), Y), Y))
        )
        return term

    @jax.jit
    def bch_9(X: Vector, Y: Vector) -> Vector:
        term = (
            -1.0 / 1209600 * br(X, br(X, br(X, br(X, br(X, br(X, br(X, br(X, Y))))))))
            + 1.0 / 151200 * br(X, br(X, br(X, br(X, br(X, br(X, br(br(X, Y), Y)))))))
            + -1.0 / 100800 * br(X, br(X, br(X, br(X, br(br(X, br(X, Y)), br(X, Y))))))
            + -1.0 / 56700 * br(X, br(X, br(X, br(X, br(X, br(br(br(X, Y), Y), Y))))))
            + -1.0 / 43200 * br(X, br(X, br(X, br(X, br(br(X, Y), br(br(X, Y), Y))))))
            + 1.0 / 75600 * br(X, br(X, br(X, br(br(X, br(br(X, Y), Y)), br(X, Y)))))
            + 1.0 / 75600 * br(X, br(X, br(X, br(X, br(br(br(br(X, Y), Y), Y), Y)))))
            + 1.0 / 302400 * br(br(X, br(X, br(X, br(X, Y)))), br(X, br(X, br(X, Y))))
            + 1.0 / 67200 * br(X, br(X, br(br(X, br(X, Y)), br(X, br(br(X, Y), Y)))))
            + 1.0 / 43200 * br(X, br(X, br(br(br(X, br(X, Y)), br(X, Y)), br(X, Y))))
            + 11.0 / 302400 * br(X, br(X, br(X, br(br(X, Y), br(br(br(X, Y), Y), Y)))))
            + 1.0 / 25200 * br(X, br(br(X, br(X, br(br(X, Y), Y))), br(X, br(X, Y))))
            + 11.0 / 201600 * br(X, br(X, br(br(X, br(br(X, Y), Y)), br(br(X, Y), Y))))
            + 11.0 / 151200 * br(X, br(X, br(br(X, br(br(br(X, Y), Y), Y)), br(X, Y))))
            + 1.0 / 75600 * br(X, br(X, br(X, br(br(br(br(br(X, Y), Y), Y), Y), Y))))
            + 1.0 / 43200 * br(br(X, br(X, br(X, Y))), br(X, br(X, br(br(X, Y), Y))))
            + 1.0 / 37800 * br(X, br(br(X, br(X, Y)), br(br(X, br(X, Y)), br(X, Y))))
            + 23.0 / 302400 * br(X, br(br(X, br(X, Y)), br(X, br(br(br(X, Y), Y), Y))))
            + -1.0 / 30240 * br(br(X, br(br(X, br(X, Y)), br(X, Y))), br(X, br(X, Y)))
            + 1.0 / 100800 * br(X, br(X, br(br(X, Y), br(br(X, Y), br(br(X, Y), Y)))))
            + 1.0 / 33600 * br(X, br(br(X, br(br(X, Y), br(br(X, Y), Y))), br(X, Y)))
            + 1.0 / 20160 * br(X, br(X, br(br(X, Y), br(br(br(br(X, Y), Y), Y), Y))))
            + 1.0 / 67200 * br(br(X, br(X, br(br(X, Y), Y))), br(X, br(br(X, Y), Y)))
            + -17.0 / 100800 * br(X, br(br(br(X, br(br(X, Y), Y)), br(X, Y)), br(X, Y)))
            + 1.0 / 30240 * br(X, br(X, br(br(br(X, Y), Y), br(br(br(X, Y), Y), Y))))
            + -1.0 / 21600 * br(br(X, br(X, br(br(br(X, Y), Y), Y))), br(X, br(X, Y)))
            + -1.0 / 15120 * br(X, br(br(X, br(br(br(X, Y), Y), Y)), br(br(X, Y), Y)))
            + -1.0 / 7560 * br(X, br(br(X, br(br(br(br(X, Y), Y), Y), Y)), br(X, Y)))
            + -1.0 / 56700 * br(X, br(X, br(br(br(br(br(br(X, Y), Y), Y), Y), Y), Y)))
            + 1.0 / 25200 * br(br(X, br(X, Y)), br(X, br(br(X, Y), br(br(X, Y), Y))))
            + -1.0 / 8400 * br(br(X, br(X, Y)), br(br(X, br(br(X, Y), Y)), br(X, Y)))
            + -1.0 / 17280 * br(br(X, br(X, Y)), br(X, br(br(br(br(X, Y), Y), Y), Y)))
            + -1.0 / 25200 * br(br(br(X, br(X, Y)), br(X, Y)), br(X, br(br(X, Y), Y)))
            + 1.0 / 50400 * br(br(br(br(X, br(X, Y)), br(X, Y)), br(X, Y)), br(X, Y))
            + 1.0 / 6048 * br(X, br(br(X, Y), br(br(X, Y), br(br(br(X, Y), Y), Y))))
            + -1.0 / 20160 * br(X, br(br(br(X, Y), br(br(X, Y), Y)), br(br(X, Y), Y)))
            + -1.0 / 10080 * br(br(X, br(br(X, Y), br(br(br(X, Y), Y), Y))), br(X, Y))
            + -23.0 / 302400 * br(X, br(br(X, Y), br(br(br(br(br(X, Y), Y), Y), Y), Y)))
            + 1.0 / 60480 * br(br(X, br(br(X, Y), Y)), br(X, br(br(br(X, Y), Y), Y)))
            + 1.0 / 20160 * br(br(X, br(br(X, Y), Y)), br(br(X, Y), br(br(X, Y), Y)))
            + 1.0 / 20160 * br(br(br(X, br(br(X, Y), Y)), br(br(X, Y), Y)), br(X, Y))
            + -11.0 / 120960 * br(X, br(br(br(X, Y), Y), br(br(br(br(X, Y), Y), Y), Y)))
            + 1.0 / 10080 * br(br(br(X, br(br(br(X, Y), Y), Y)), br(X, Y)), br(X, Y))
            + 1.0 / 90720 * br(br(X, br(br(br(X, Y), Y), Y)), br(br(br(X, Y), Y), Y))
            + 1.0 / 60480 * br(br(X, br(br(br(br(X, Y), Y), Y), Y)), br(br(X, Y), Y))
            + 1.0 / 21600 * br(br(X, br(br(br(br(br(X, Y), Y), Y), Y), Y)), br(X, Y))
            + 1.0 / 151200 * br(X, br(br(br(br(br(br(br(X, Y), Y), Y), Y), Y), Y), Y))
            + 1.0 / 10080 * br(br(X, Y), br(br(X, Y), br(br(X, Y), br(br(X, Y), Y))))
            + -1.0 / 7560 * br(br(X, Y), br(br(X, Y), br(br(br(br(X, Y), Y), Y), Y)))
            + -1.0 / 15120 * br(br(X, Y), br(br(br(X, Y), Y), br(br(br(X, Y), Y), Y)))
            + 1.0 / 15120 * br(br(br(X, Y), br(br(br(X, Y), Y), Y)), br(br(X, Y), Y))
            + 1.0 / 43200 * br(br(X, Y), br(br(br(br(br(br(X, Y), Y), Y), Y), Y), Y))
            + 1.0 / 33600 * br(br(br(X, Y), Y), br(br(br(br(br(X, Y), Y), Y), Y), Y))
            + 1.0 / 60480 * br(br(br(br(X, Y), Y), Y), br(br(br(br(X, Y), Y), Y), Y))
            + -1.0 / 1209600 * br(br(br(br(br(br(br(br(X, Y), Y), Y), Y), Y), Y), Y), Y)
        )
        return term

    return (bch_1, bch_2, bch_3, bch_4, bch_5, bch_6, bch_7, bch_8, bch_9)


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

