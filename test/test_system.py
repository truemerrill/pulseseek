import numpy as np
from pulseseek import pauli
from pulseseek.basis import special_unitary_basis
from pulseseek.system import ControlSystem


def test_control_system():
    su2_basis = special_unitary_basis(2)
    X, _, Z = pauli()
    system = ControlSystem.new(
        control_hamiltonians=(X,),
        error_hamiltonian=X,
        drift_hamiltonian=Z,
        basis=su2_basis
    )

    H = system.control_hamiltonian([1.0])
    assert np.isclose(H, X + Z).all()

    h = system.control_lie_vector([1.0])
    assert np.isclose(h, np.array([1.0, 0.0, 1.0])).all()


def test_control_system_without_basis():
    X, _, Z = pauli()
    system = ControlSystem.new(
        control_hamiltonians=(X,),
        error_hamiltonian=X,
        drift_hamiltonian=Z
    )

    H = system.control_hamiltonian([1.0])
    assert np.isclose(H, X + Z).all()

    h = system.control_lie_vector([1.0])
    assert np.isclose(h, np.array([1.0, 1.0, 0.0])).all()