import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Any, Iterable

from .algebra import LieAlgebra, lie_algebra, lie_closure, MatrixInnerProduct, hilbert_schmidt_inner_product, MatrixBracket, matrix_commutator
from .basis import LieBasis
from .types import Hermitian, AntiHermitian, is_anti_hermitian, is_hermitian, Vector, is_vector


@dataclass
class ControlSystem:
    # Matrix representation
    _H_drift: Hermitian | None
    _H_controls: tuple[Hermitian, ...]
    _H_error: Hermitian

    # Lie algebra representation
    _E_drift: Vector | None
    _E_controls: tuple[Vector, ...]
    _E_error: Vector

    basis: LieBasis
    algebra: LieAlgebra

    @classmethod
    def new(
        cls,
        control_hamiltonians: Iterable[Hermitian],
        error_hamiltonian: Hermitian,
        drift_hamiltonian: Hermitian | None = None,
        basis: LieBasis | None = None,
        inner_product: MatrixInnerProduct = hilbert_schmidt_inner_product,
        bracket: MatrixBracket = matrix_commutator,
    ):
        """Construct a new control system

        !!! note
            A control system is a set of Hamiltonian operators of the form

            $$ H(u) = H_d + \\sum_i u_i H_i + \\epsilon H_e $$

            where `H_d`, `H_i`, and `H_e` are the drift Hamiltonian, control
            Hamiltonians, and error Hamiltonian respectively.  The strength
            of the error Hamiltonian `\\epsilon` is unknown.

            Alternatively, it is possible to represent the control system as
            a set of vectors on a Lie algebra

            $$ V(u) = V_d + \\sum_i u_i V_i + \\epsilon V_e $$
 
            where `V_d`, `V_i`, and `V_e` are the images of the Hamiltonains
            represented in the Lie algebra.

        Args:
            control_hamiltonians (Iterable[Hermitian]): the control
                Hamiltonians
            error_hamiltonian (Hermitian): the error Hamiltonian
            drift_hamiltonian (Hermitian | None): the drift Hamiltonian.  If
                not set, no drift is assumed.  Defaults to None.
            basis (LieBasis | None): the Lie basis to use for the Lie algebra
                representation. If not provided, this method computes a basis
                from the Lie closure of the provided Hamiltonians. Defaults to
                None.
            inner_product (InnerProduct, optional): the inner product. Defaults
                to hilbert_schmidt_inner_product.
            bracket (Bracket, optional): the Lie bracket. Defaults to
                matrix_commutator.

        Returns:
            ControlSystem: the control system
        """
        H_controls = tuple(control_hamiltonians)
        
        if basis is None:
            # Construct a Lie basis from the Lie closure
            Hd = {} if drift_hamiltonian is None else {"Hd": 1j * drift_hamiltonian}
            Hc = {f"Hc_{i}": 1j * H for i, H in enumerate(H_controls)}
            He = {"He": 1j * error_hamiltonian}
            elements = {**Hd, **Hc, **He}
            basis = lie_closure(elements, bracket=bracket)

        algebra = lie_algebra(basis, inner_product, bracket)

        def decomposition(x: Any) -> Vector:
            def project(x: Any, y: AntiHermitian) -> float:
                assert is_anti_hermitian(x)
                return inner_product(x, y)

            g = jnp.array([project(x, y) for y in basis.elements])
            v = jnp.linalg.solve(algebra.G, g)
            assert is_vector(v)
            return v
        
        E_drift = None if drift_hamiltonian is None else decomposition(1j * drift_hamiltonian)
        E_controls = tuple([decomposition(1j * H) for H in H_controls])
        E_error = decomposition(1j * error_hamiltonian)

        return cls(
            _H_drift = drift_hamiltonian,
            _H_controls = H_controls,
            _H_error = error_hamiltonian,
            _E_drift = E_drift,
            _E_controls = E_controls,
            _E_error = E_error,
            basis = basis,
            algebra = algebra
        )  

    def control_hamiltonian(self, control: Iterable[float]) -> Hermitian:
        """Calculate the ideal (error free) Hamiltonian under the control

        Args:
            control (Vector): the control vector

        Returns:
            Hermitian: the ideal Hamiltonian
        """
        H = np.zeros(self._H_error.shape, dtype=complex)

        if self._H_drift is not None:
            H += self._H_drift
        for u, Hu in zip(control, self._H_controls):
            H += u * Hu

        assert is_hermitian(H)
        return H

    def control_lie_vector(self, control: Iterable[float]) -> Vector:
        """Calculate the image of the ideal Hamiltonian in the Lie algebra
        
        Args:
            control (Vector): the control vector
        
        Returns:
            Vector: the Lie algebra representation of the ideal Hamiltonian
        """
        m = self.basis.dim
        v = np.zeros((m,), dtype=float)

        if self._E_drift is not None:
            v += self._E_drift

        for u, Eu in zip(control, self._E_controls):
            v += u * Eu
        assert is_vector(v, dimension=m)
        return v