"""Toy ASE calculators with known physics, for testing phonons and mode mapping without ML models."""

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class DoubleWell(Calculator):
    """Independent atoms in an on-site potential ``A d² + B d⁴`` per Cartesian direction.

    ``d`` is the displacement from the nearest site of an orthorhombic grid with spacings
    ``a0``. ``A < 0`` along a direction makes it unstable; the well depth per atom is
    ``A² / (4B)`` eV, so the depth sets whether mode mapping calls the instability real.
    """

    implemented_properties = ["energy", "forces", "stress"]  # noqa: RUF012

    def __init__(self, a0: tuple[float, float, float], A: tuple[float, float, float], B: float = 1.0) -> None:
        """Set the grid and the per-direction quadratic coefficients.

        Parameters
        ----------
        a0 : tuple of float
            Grid spacing per direction, Å.
        A : tuple of float
            Quadratic coefficients per direction, eV/Å².
        B : float
            Quartic coefficient, eV/Å⁴.
        """
        super().__init__()
        self.a0 = np.asarray(a0, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.B = B

    def calculate(
        self, atoms: Any = None, properties: Any = ("energy",), system_changes: Any = all_changes
    ) -> None:
        """Compute energy, forces and (zero) stress.

        Parameters
        ----------
        atoms : ase.Atoms
            Structure.
        properties : sequence of str
            Requested properties.
        system_changes : sequence of str
            Changes since the last call.
        """
        super().calculate(atoms, properties, system_changes)
        pos = self.atoms.get_positions()
        d = pos - np.round(pos / self.a0) * self.a0
        self.results["energy"] = float(np.sum(self.A * d**2 + self.B * d**4))
        self.results["forces"] = -(2 * self.A * d + 4 * self.B * d**3)
        self.results["stress"] = np.zeros(6)


def toy_cell() -> Atoms:
    """One Cu atom in an orthorhombic cell matching the DoubleWell grid.

    Returns
    -------
    ase.Atoms
        The cell (3.0 by 3.1 by 3.2 Å).
    """
    return Atoms("Cu", cell=[3.0, 3.1, 3.2], positions=[[0, 0, 0]], pbc=True)


A0 = (3.0, 3.1, 3.2)
