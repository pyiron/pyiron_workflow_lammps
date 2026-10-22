import unittest

import numpy as np
from ase import units

from pyiron_workflow_lammps.lammps import arrays_to_ase_atoms


class TestVelocityCapture(unittest.TestCase):
    def test_arrays_to_ase_atoms_attaches_velocities(self):
        velocities = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])  # Å/fs

        atoms = arrays_to_ase_atoms(
            cells=np.eye(3) * 5.0,
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            indices=np.array([1, 1]),
            species_lists=[["Al", "Al"]],
            pbc=True,
            velocities=velocities,
        )

        # stored in ASE velocity units = (Å/fs) / units.fs
        self.assertTrue(np.allclose(atoms.get_velocities(), velocities / units.fs))

    def test_arrays_to_ase_atoms_without_velocities_is_zero(self):
        atoms = arrays_to_ase_atoms(
            cells=np.eye(3) * 5.0,
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            indices=np.array([1, 1]),
            species_lists=[["Al", "Al"]],
            pbc=True,
        )

        self.assertTrue(np.allclose(atoms.get_velocities(), 0.0))


if __name__ == "__main__":
    unittest.main()
