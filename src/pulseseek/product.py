from typing import NamedTuple, cast
import jax
import jax.numpy as jnp
import functools

from .algebra import LieAlgebra, lie_bracket, lie_projection, MatrixInnerProduct, hilbert_schmidt_inner_product
from .bch import BilinearMap, baker_campbell_hausdorff_series
from .types import LieVector



class BCHScalingCarry(NamedTuple):
    x: LieVector
    y: LieVector
    scale: float


def lie_product_baker_campbell_hausdorff(
    algebra: LieAlgebra, atol: float = 1e-12, order: int = 8
) -> BilinearMap:
    bracket = lie_bracket(algebra)
    series = baker_campbell_hausdorff_series(bracket)

    if order + 1 > len(series):
        raise ValueError("order is greater than the maximum series order")

    # The next term in the series, calculating to bound truncation error
    z_tail = series[order]

    @jax.jit
    def bch_scaling(x: LieVector, y: LieVector) -> float:
        """Find scaling factor so that the truncation error is less than atol

        Args:
            x (Vector): Lie algebra element
            y (Vector): Lie algebra element

        Returns:
            int: scale power
        """
        eps = jnp.array(atol)

        def body(carry: BCHScalingCarry) -> BCHScalingCarry:
            return BCHScalingCarry(carry.x, carry.y, carry.scale * 2)

        def cond(carry: BCHScalingCarry) -> jax.Array:
            xa = carry.x / carry.scale
            ya = carry.y / carry.scale
            za = z_tail(xa, ya)
            norm = jnp.linalg.norm(za)
            is_not_converged = jnp.array(carry.scale * norm >= eps).reshape(())
            return is_not_converged

        initial = BCHScalingCarry(x, y, 1.0)
        final = jax.lax.while_loop(cond, body, initial)
        return final.scale

    @jax.jit
    def product(x: LieVector, y: LieVector) -> LieVector:
        scale = bch_scaling(x, y)
        xa = x / scale
        ya = y / scale
        terms = tuple(scale * Zm(xa, ya) for Zm in series[0:order])
        return cast(LieVector, sum(terms))

    return product
