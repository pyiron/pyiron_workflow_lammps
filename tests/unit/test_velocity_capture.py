import numpy as np
from ase import units

from pyiron_workflow_lammps.lammps import arrays_to_ase_atoms


def test_arrays_to_ase_atoms_attaches_velocities():
    species_lists = [["Al", "Al"]]
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    cells = np.eye(3) * 5.0
    indices = np.array([1, 1])
    velocities = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]])  # pyiron units (Å/fs)
    atoms = arrays_to_ase_atoms(
        cells=cells,
        positions=positions,
        indices=indices,
        species_lists=species_lists,
        pbc=True,
        velocities=velocities,
    )
    assert atoms.get_velocities() is not None
    # stored in ASE velocity units = (Å/fs) / units.fs
    assert np.allclose(atoms.get_velocities(), velocities / units.fs)


def test_arrays_to_ase_atoms_without_velocities_is_zero():
    atoms = arrays_to_ase_atoms(
        cells=np.eye(3) * 5.0,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        indices=np.array([1, 1]),
        species_lists=[["Al", "Al"]],
        pbc=True,
    )
    assert np.allclose(atoms.get_velocities(), 0.0)
