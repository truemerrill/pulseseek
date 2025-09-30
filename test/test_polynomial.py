import numpy as np

from pulseseek.algebra import lie_algebra
from pulseseek.basis import special_unitary_basis
from pulseseek.polynomial import LiePolynomial


def test_polynomial_toggle():
    su2 = lie_algebra(special_unitary_basis(2))
    Ex, Ey, Ez = su2.basis.vectors
    Y = LiePolynomial.new(su2, degree=30, coefficients=[Ey])
    
    # Note:
    #
    #   - The minus sign is the same minus sign that appears in the equation
    #     for the propagator, U = exp(- i H t ).  So we have to put it in
    #     in order to have a positive sense of rotation.
    #
    #   - This basis for the su2 Lie algebra uses the Pauli matrices, so we
    #     still need to divide by 2 in the exponent, as in 
    #     U = exp(- i \theta X / 2)
    #
    Z = Y.toggle(- (np.pi / 2) * Ex / 2)

    assert np.allclose(Y(1), Ey)        # Initially pointed along Y direction
    assert np.allclose(Z(1), Ez)        # After rotate by pi/2 around X, points at Z