"""Real LAMMPS runs through pyiron_workflow_atomistics' ``calculate``.

Needs a LAMMPS binary with the MANYBODY package (``pair_style eam/fs``):
``lmp`` on ``PATH`` or ``$LAMMPS_COMMAND``. Skipped without one.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase import Atoms, units
from ase.build import bulk
from pyiron_workflow_atomistics.engine import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
    EngineOutput,
    calculate,
)

from pyiron_workflow_lammps.engine import LammpsEngine

pytestmark = pytest.mark.integration

MD_STEPS = 40


def _structure() -> Atoms:
    """54-atom bcc Fe with two Al, slightly rattled so forces are non-zero."""
    structure = bulk("Fe", "bcc", a=2.87, cubic=True).repeat((3, 3, 3))
    structure[0].symbol = "Al"
    structure[13].symbol = "Al"
    structure.rattle(stdev=0.05, seed=1)
    return structure


def _engine(engine_input, tmp_path, lammps_executable, eam_potential, **kwargs):
    settings = {
        "working_directory": str(tmp_path / "calc"),
        "command": f"{lammps_executable} -in in.lmp -log log.lammps",
        "input_script_pair_style": "eam/fs",
        "path_to_model": str(eam_potential),
        **kwargs,
    }
    return LammpsEngine(EngineInput=engine_input, **settings)


def _last_thermo_row(log_path) -> dict[str, float]:
    """The last thermo line of the last run in a LAMMPS log, by column name."""
    header, row = None, None
    with open(log_path) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        tokens = line.split()
        if tokens[:1] == ["Step"]:
            header = tokens
            for following in lines[i + 1 :]:
                values = following.split()
                if following.startswith("Loop time") or len(values) != len(header):
                    break
                row = values
    assert header is not None and row is not None, f"no thermo output in {log_path}"
    return dict(zip(header, map(float, row), strict=True))


def _dump_frames(dump_path) -> list[dict]:
    """Frames of a ``dump custom`` file: step, box bounds and columns by name."""
    with open(dump_path) as f:
        lines = f.read().splitlines()
    frames = []
    i = 0
    while i < len(lines):
        assert lines[i] == "ITEM: TIMESTEP", lines[i]
        step = int(lines[i + 1])
        n_atoms = int(lines[i + 3])
        bounds = np.array([line.split()[:2] for line in lines[i + 5 : i + 8]], float)
        columns = lines[i + 8].split()[2:]
        rows = np.array(
            [line.split() for line in lines[i + 9 : i + 9 + n_atoms]], dtype=float
        )
        data = dict(zip(columns, rows.T, strict=True))
        order = np.argsort(data["id"])
        frames.append(
            {
                "step": step,
                "bounds": bounds,
                **{name: values[order] for name, values in data.items()},
            }
        )
        i += 9 + n_atoms
    return frames


def _check_output(out, structure, engine, converged=True) -> dict[str, float]:
    """Sanity checks shared by every mode; returns the log's last thermo row."""
    assert isinstance(out, EngineOutput)
    assert out.converged is converged
    assert type(out.final_energy) is float
    assert np.isfinite(out.final_energy)
    assert type(out.n_ionic_steps) is int
    assert out.n_ionic_steps >= 0

    n_atoms = len(structure)
    assert len(out.final_structure) == n_atoms
    assert (
        out.final_structure.get_chemical_symbols() == structure.get_chemical_symbols()
    )
    assert np.shape(out.final_forces) == (n_atoms, 3)
    assert np.all(np.isfinite(out.final_forces))
    assert np.shape(out.final_stress) == (3, 3)
    assert out.final_volume == pytest.approx(out.final_structure.get_volume())
    assert len(out.energies) >= 1
    assert out.energies[-1] == out.final_energy
    # one structure per thermo row: the dump is written on the thermo steps
    assert len(out.structures) == len(out.energies) == len(out.forces)
    np.testing.assert_array_equal(
        out.structures[-1].get_positions(), out.final_structure.get_positions()
    )

    thermo = _last_thermo_row(os.path.join(engine.working_directory, "log.lammps"))
    # The potential energy, consistent with the forces - not the total energy.
    assert out.final_energy == pytest.approx(thermo["PotEng"], rel=1e-10, abs=1e-8)
    assert out.n_ionic_steps == int(thermo["Step"])
    return thermo


def test_static(tmp_path, lammps_executable, eam_potential):
    structure = _structure()
    engine = _engine(CalcInputStatic(), tmp_path, lammps_executable, eam_potential)

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    assert out.n_ionic_steps == 0
    np.testing.assert_allclose(
        out.final_structure.get_positions(wrap=True),
        structure.get_positions(wrap=True),
        atol=1e-6,
    )
    # rattled, so there are forces to report
    assert np.abs(out.final_forces).max() > 1e-3


@pytest.mark.parametrize("relax_cell", [False, True], ids=["fixed-cell", "relax-cell"])
def test_minimize(tmp_path, lammps_executable, eam_potential, relax_cell):
    structure = _structure()
    static = calculate(
        structure=structure,
        engine=_engine(
            CalcInputStatic(), tmp_path / "static", lammps_executable, eam_potential
        ),
    )
    engine = _engine(
        CalcInputMinimize(
            energy_convergence_tolerance=0.0,
            force_convergence_tolerance=1e-4,
            max_iterations=2000,
            relax_cell=relax_cell,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    assert 0 < out.n_ionic_steps <= 2000
    assert out.final_energy < static.final_energy
    volume_change = abs(out.final_volume - structure.get_volume())
    if relax_cell:
        assert volume_change > 1e-3
        # box/relax drives the pressure to input_script_relax_pressure (bar);
        # final_stress is lammpsparser's pressure in GPa. The default is a
        # zero-pressure relaxation (0.1.0 defaulted to 1e4 bar = 1 GPa).
        assert engine.input_script_relax_pressure == 0.0
        target_gpa = engine.input_script_relax_pressure / 1e4
        assert np.mean(np.diag(out.final_stress)) == pytest.approx(target_gpa, abs=1e-2)
    else:
        assert volume_change < 1e-8
        assert np.abs(out.final_forces).max() < np.abs(static.final_forces).max()


MD_CASES = {
    "nve": {"mode": "NVE"},
    "nvt-langevin": {"mode": "NVT", "thermostat": "langevin"},
    "nvt-nose-hoover": {"mode": "NVT", "thermostat": "nose-hoover"},
    "npt-nose-hoover": {"mode": "NPT", "thermostat": "nose-hoover", "pressure": 1e5},
}


@pytest.mark.parametrize("case", MD_CASES.values(), ids=MD_CASES.keys())
def test_md(tmp_path, lammps_executable, eam_potential, case):
    structure = _structure()
    engine = _engine(
        CalcInputMD(
            temperature=300.0,
            n_ionic_steps=MD_STEPS,
            n_print=10,
            time_step=1.0,
            seed=42,
            **case,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    thermo = _check_output(out, structure, engine)
    assert out.n_ionic_steps == MD_STEPS
    # Kinetic energy is not part of final_energy.
    assert thermo["Temp"] > 0
    assert thermo["TotEng"] - thermo["PotEng"] > 1e-3
    assert out.final_energy < thermo["TotEng"]
    assert len(out.structures) == len(out.energies) == MD_STEPS // 10 + 1
    # a timestep in fs -> metal ps conversion gone wrong would blow up here
    assert np.abs(out.final_forces).max() < 10.0
    if case["mode"] == "NPT":
        assert out.final_volume != pytest.approx(structure.get_volume(), abs=1e-6)
    else:
        assert out.final_volume == pytest.approx(structure.get_volume())


@pytest.mark.parametrize(
    "case",
    [
        {"mode": "NVE", "seed": None},
        # fix langevin's Marsaglia RNG accepts seeds up to 900000000
        {"mode": "NVT", "thermostat": "langevin", "seed": 900_000_000},
        {"mode": "NVE", "initial_temperature": 0.0},
    ],
    ids=["nve-seed-none", "langevin-largest-seed", "nve-initial-0K"],
)
def test_md_seed_and_initial_temperature(
    tmp_path, lammps_executable, eam_potential, case
):
    structure = _structure()
    engine = _engine(
        CalcInputMD(temperature=300.0, n_ionic_steps=10, n_print=5, **case),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    thermo = _check_output(out, structure, engine)
    assert out.n_ionic_steps == 10
    with open(os.path.join(engine.working_directory, "in.lmp")) as f:
        script = f.read()
    assert "None" not in script
    if case.get("initial_temperature") == 0.0:
        assert "velocity" not in script
        # started at rest, so the kinetic energy only comes from the forces
        assert out.structures[0].get_velocities() == pytest.approx(0.0)
        assert thermo["Temp"] < 300.0
    else:
        assert "velocity all create 300.0 " in script


def _minimum_image_displacements(final: Atoms, initial: Atoms) -> np.ndarray:
    """Per-atom displacement lengths from ``initial``, modulo the (fixed) cell."""
    scaled = final.get_scaled_positions(wrap=False) - initial.get_scaled_positions(
        wrap=False
    )
    scaled -= np.rint(scaled)
    return np.linalg.norm(scaled @ np.asarray(initial.cell), axis=1)


# CalcInputMD's thermostat Literal lists the first four; the dataclass does
# not enforce it and the engine also implements temp/rescale and temp/csvr.
NVT_THERMOSTATS = [
    "nose-hoover",
    "langevin",
    "andersen",
    "berendsen",
    "temp/rescale",
    "temp/csvr",
]


@pytest.mark.parametrize("thermostat", NVT_THERMOSTATS)
def test_md_nvt_thermostat_moves_atoms(
    tmp_path, lammps_executable, eam_potential, thermostat
):
    """Every NVT thermostat runs and integrates the equations of motion.

    berendsen, temp/rescale and temp/csvr only rescale velocities; without
    an integrator next to them LAMMPS warns "No fixes with time integration,
    atoms won't move" and the displacement is exactly zero.
    """
    structure = _structure()
    engine = _engine(
        CalcInputMD(
            mode="NVT",
            thermostat=thermostat,
            temperature=300.0,
            n_ionic_steps=50,
            n_print=10,
            seed=11,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    thermo = _check_output(out, structure, engine)
    assert out.n_ionic_steps == 50
    assert len(out.structures) == 6
    with open(os.path.join(engine.working_directory, "log.lammps")) as f:
        assert "atoms won't move" not in f.read()
    assert thermo["Temp"] > 10.0
    # ~1 A in 50 fs at 300 K; the rattle alone is 0.05 A
    assert _minimum_image_displacements(out.final_structure, structure).max() > 0.1
    for previous, current in zip(out.structures, out.structures[1:], strict=False):
        assert _minimum_image_displacements(current, previous).max() > 1e-3
    np.testing.assert_allclose(
        np.asarray(out.final_structure.cell), np.asarray(structure.cell), atol=1e-10
    )
    if thermostat == "temp/rescale":
        # window 0, fraction 1: rescaled exactly to the target every n_print
        assert thermo["Temp"] == pytest.approx(300.0, rel=1e-8)


def test_md_langevin_seed_none_runs(tmp_path, lammps_executable, eam_potential):
    """seed=None draws seeds fix langevin's Marsaglia RNG accepts.

    With the old [1, 2**31 - 1] range ~58% of these runs failed with
    "Invalid seed for Marsaglia random # generator"; ten in a row passed
    with probability ~2e-4.
    """
    structure = _structure()
    seeds = set()
    for i in range(10):
        engine = _engine(
            CalcInputMD(
                mode="NVT",
                thermostat="langevin",
                temperature=300.0,
                n_ionic_steps=2,
                n_print=1,
            ),
            tmp_path / f"run{i}",
            lammps_executable,
            eam_potential,
        )
        assert engine.EngineInput.seed is None

        out = calculate(structure=structure, engine=engine)

        _check_output(out, structure, engine)
        with open(os.path.join(engine.working_directory, "in.lmp")) as f:
            (langevin,) = [
                line for line in f.read().splitlines() if " langevin " in line
            ]
        seed = int(langevin.split()[-1])
        assert 1 <= seed <= 900_000_000
        seeds.add(seed)
    assert len(seeds) > 1


def test_md_n_print_zero(tmp_path, lammps_executable, eam_potential):
    """thermo 0 prints the first and last step; the dump follows it."""
    structure = _structure()
    engine = _engine(
        CalcInputMD(mode="NVE", temperature=300.0, n_ionic_steps=20, n_print=0, seed=5),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    assert out.n_ionic_steps == 20
    assert len(out.structures) == len(out.energies) == len(out.forces) == 2
    assert len(out.stresses) == 2
    frames = _dump_frames(os.path.join(engine.working_directory, "dump.out"))
    assert [frame["step"] for frame in frames] == [0, 20]
    np.testing.assert_allclose(
        out.final_forces,
        np.column_stack([frames[-1]["fx"], frames[-1]["fy"], frames[-1]["fz"]]),
        atol=1e-10,
    )


def test_npt_without_pressure_raises(tmp_path, lammps_executable, eam_potential):
    engine = _engine(
        CalcInputMD(mode="NPT", thermostat="nose-hoover", n_ionic_steps=2, n_print=1),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    with pytest.raises(ValueError, match="NPT mode requires CalcInputMD.pressure"):
        calculate(structure=_structure(), engine=engine)


def test_minimize_dumps_on_the_thermo_interval(
    tmp_path, lammps_executable, eam_potential
):
    """Frames follow input_script_thermo_every, even if dump_every differs."""
    structure = _structure()
    engine = _engine(
        CalcInputMinimize(
            energy_convergence_tolerance=0.0,
            force_convergence_tolerance=1e-8,
            max_iterations=12,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
        input_script_thermo_every=5,
        input_script_dump_every=3,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine, converged=False)
    assert out.n_ionic_steps == 12
    assert len(out.structures) == len(out.energies) == len(out.forces) == 4
    frames = _dump_frames(os.path.join(engine.working_directory, "dump.out"))
    assert [frame["step"] for frame in frames] == [0, 5, 10, 12]
    for frame, energy in zip(frames, out.energies, strict=True):
        assert np.sum(frame["c_eng"]) == pytest.approx(energy, abs=1e-8)


def _primitive_bcc_feal() -> Atoms:
    """Primitive bcc supercell: LAMMPS must rotate it and change its basis."""
    structure = bulk("Fe", "bcc", a=2.85).repeat(3)
    structure[0].symbol = "Al"
    structure.rattle(stdev=0.05, seed=2)
    return structure


def _sheared_cubic_feal() -> Atoms:
    """Cubic cell under a small shear: not lower triangular, so rotated."""
    structure = bulk("Fe", "bcc", a=2.85, cubic=True).repeat(3)
    structure[0].symbol = "Al"
    strain = np.eye(3)
    strain[1, 2] = strain[2, 1] = 0.01
    structure.set_cell(np.asarray(structure.cell) @ strain, scale_atoms=True)
    structure.rattle(stdev=0.05, seed=2)
    return structure


ROTATED_CELLS = {
    "primitive-bcc": _primitive_bcc_feal,
    "sheared-cubic": _sheared_cubic_feal,
}


def _ase_eam(structure: Atoms, eam_potential) -> Atoms:
    from ase.calculators.eam import EAM

    reference = structure.copy()
    reference.calc = EAM(potential=str(eam_potential), form="fs")
    return reference


def _ase_stress(atoms: Atoms, eps: float = 1e-6) -> np.ndarray:
    """ASE stress (eV/A^3, tension positive) as a 3x3 matrix.

    The EAM calculator of older ASE (e.g. 3.24, the floor) has no analytic
    stress, and ASE's own finite differences need a free energy it does not
    provide either, so fall back to central differences of the energy under
    symmetric strains (agrees with the analytic stress to ~3e-8 GPa here).
    """
    from ase.calculators.calculator import PropertyNotImplementedError

    try:
        return atoms.get_stress(voigt=False)
    except PropertyNotImplementedError:
        pass
    strained = atoms.copy()
    strained.calc = atoms.calc
    cell = np.asarray(atoms.cell)
    volume = atoms.get_volume()
    stress = np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            energies = []
            for sign in (1.0, -1.0):
                deformation = np.eye(3)
                deformation[i, j] += sign * eps / 2
                deformation[j, i] += sign * eps / 2
                strained.set_cell(cell @ deformation, scale_atoms=True)
                strained.set_scaled_positions(atoms.get_scaled_positions())
                energies.append(strained.get_potential_energy())
            stress[i, j] = stress[j, i] = (energies[0] - energies[1]) / (
                2 * eps * volume
            )
    return stress


def _assert_matches_ase_eam(out, structure, eam_potential):
    reference = _ase_eam(structure, eam_potential)
    assert out.final_energy == pytest.approx(reference.get_potential_energy(), abs=1e-8)
    np.testing.assert_allclose(out.final_forces, reference.get_forces(), atol=1e-5)
    # final_stress is a pressure tensor in GPa, compression positive
    np.testing.assert_allclose(
        out.final_stress,
        -_ase_stress(reference) / units.GPa,
        atol=1e-5,
    )


@pytest.mark.parametrize("make_structure", ROTATED_CELLS.values(), ids=ROTATED_CELLS)
def test_rotated_cell_results_are_in_the_input_frame(
    tmp_path, lammps_executable, eam_potential, make_structure
):
    """Forces, stress, cell and positions come back in the input's frame.

    LAMMPS needs a lower-triangular cell with bounded tilts, so it rotates
    these cells (and re-chooses the primitive cell's lattice vectors). The
    results are compared with ASE's own EAM calculator on the input.
    """
    structure = make_structure()
    # not lower triangular, so LAMMPS really has to rotate it
    assert not np.allclose(np.triu(np.asarray(structure.cell), 1), 0.0)
    engine = _engine(CalcInputStatic(), tmp_path, lammps_executable, eam_potential)

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    _assert_matches_ase_eam(out, structure, eam_potential)
    np.testing.assert_allclose(
        np.asarray(out.final_structure.cell), np.asarray(structure.cell), atol=1e-10
    )
    scaled = out.final_structure.get_scaled_positions(
        wrap=False
    ) - structure.get_scaled_positions(wrap=False)
    np.testing.assert_allclose(scaled - np.rint(scaled), 0.0, atol=1e-10)
    # the trajectory is in the same frame
    np.testing.assert_allclose(
        np.asarray(out.structures[0].cell), np.asarray(structure.cell), atol=1e-10
    )
    np.testing.assert_allclose(out.forces[-1], out.final_forces)
    np.testing.assert_allclose(out.stresses[-1], out.final_stress)


@pytest.mark.parametrize("make_structure", ROTATED_CELLS.values(), ids=ROTATED_CELLS)
def test_rotated_cell_minimize_is_in_the_input_frame(
    tmp_path, lammps_executable, eam_potential, make_structure
):
    """After relaxing positions the cell, forces and stress are still consistent."""
    structure = make_structure()
    engine = _engine(
        CalcInputMinimize(
            energy_convergence_tolerance=0.0,
            force_convergence_tolerance=1e-3,
            max_iterations=1000,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    np.testing.assert_allclose(
        np.asarray(out.final_structure.cell), np.asarray(structure.cell), atol=1e-10
    )
    for frame in out.structures:
        np.testing.assert_allclose(
            np.asarray(frame.cell), np.asarray(structure.cell), atol=1e-10
        )
    _assert_matches_ase_eam(out, out.final_structure, eam_potential)


def test_md_dumps_on_thermo_steps(tmp_path, lammps_executable, eam_potential):
    """final_structure/final_forces are from the same (last) step as final_energy."""
    structure = _structure()
    engine = _engine(
        CalcInputMD(mode="NVE", temperature=300.0, n_ionic_steps=53, n_print=7, seed=7),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    assert out.n_ionic_steps == 53
    assert len(out.structures) == len(out.energies) == 9
    frames = _dump_frames(os.path.join(engine.working_directory, "dump.out"))
    assert [frame["step"] for frame in frames] == [0, 7, 14, 21, 28, 35, 42, 49, 53]
    last = frames[-1]
    # orthogonal box: scaled unwrapped coordinates times the box lengths;
    # the parser wraps atoms back into the cell, so compare modulo the box
    lengths = last["bounds"][:, 1] - last["bounds"][:, 0]
    np.testing.assert_allclose(out.final_structure.cell.lengths(), lengths)
    dump_positions = np.column_stack([last["xsu"], last["ysu"], last["zsu"]]) * lengths
    difference = out.final_structure.get_positions() - dump_positions
    difference -= np.round(difference / lengths) * lengths
    np.testing.assert_allclose(difference, 0.0, atol=1e-8)
    # the previous frame (step 49) is a different structure
    previous = np.column_stack(
        [frames[-2]["xsu"], frames[-2]["ysu"], frames[-2]["zsu"]]
    )
    assert np.abs(previous * lengths - dump_positions).max() > 1e-4
    np.testing.assert_allclose(
        out.final_forces,
        np.column_stack([last["fx"], last["fy"], last["fz"]]),
        atol=1e-10,
    )
    # and the energy of that frame is the last thermo energy
    assert np.sum(last["c_eng"]) == pytest.approx(out.final_energy, abs=1e-8)


def test_minimize_max_iterations_is_not_converged(
    tmp_path, lammps_executable, eam_potential
):
    structure = _structure()
    engine = _engine(
        CalcInputMinimize(
            energy_convergence_tolerance=0.0,
            force_convergence_tolerance=1e-8,
            max_iterations=3,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine, converged=False)
    assert out.n_ionic_steps == 3
    assert len(out.structures) == len(out.energies) == 2
    frames = _dump_frames(os.path.join(engine.working_directory, "dump.out"))
    assert [frame["step"] for frame in frames] == [0, 3]
    with open(os.path.join(engine.working_directory, "log.lammps")) as f:
        assert "Stopping criterion = max iterations" in f.read()


def test_converged_minimize_ends_on_a_tolerance(
    tmp_path, lammps_executable, eam_potential
):
    structure = _structure()
    engine = _engine(
        CalcInputMinimize(
            energy_convergence_tolerance=0.0,
            force_convergence_tolerance=1e-4,
            max_iterations=2000,
        ),
        tmp_path,
        lammps_executable,
        eam_potential,
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine, converged=True)
    with open(os.path.join(engine.working_directory, "log.lammps")) as f:
        assert "Stopping criterion = force tolerance" in f.read()
    frames = _dump_frames(os.path.join(engine.working_directory, "dump.out"))
    assert frames[-1]["step"] == out.n_ionic_steps
    assert len(frames) == len(out.energies)


def test_failed_rerun_raises_instead_of_returning_stale_output(
    tmp_path, lammps_executable, eam_potential
):
    first = _engine(CalcInputStatic(), tmp_path, lammps_executable, eam_potential)
    out = calculate(structure=_structure(), engine=first)
    assert out.converged is True
    log = os.path.join(first.working_directory, "log.lammps")
    assert os.path.isfile(log)

    # a different structure, same directory, a launcher that does not exist
    second = _engine(
        CalcInputStatic(),
        tmp_path,
        "pwl-no-such-lammps",
        eam_potential,
    )
    with pytest.raises(RuntimeError, match="return code 127") as info:
        calculate(structure=bulk("Fe", "bcc", a=2.95, cubic=True), engine=second)

    assert "pwl-no-such-lammps" in str(info.value)
    assert first.working_directory in str(info.value)
    assert os.path.isfile(log)  # the stale output was there to be (mis)read


def test_failing_lammps_raises_with_its_error(
    tmp_path, lammps_executable, eam_potential
):
    """LAMMPS starts but fails: its own error message reaches the caller."""
    engine = _engine(
        CalcInputStatic(),
        tmp_path,
        lammps_executable,
        eam_potential,
        path_to_model=str(tmp_path / "missing.eam.fs"),
    )

    with pytest.raises(RuntimeError, match="LAMMPS command exited") as info:
        calculate(structure=_structure(), engine=engine)

    assert "missing.eam.fs" in str(info.value)


@pytest.mark.parametrize("mode", ["static", "md"])
def test_non_default_data_and_dump_file_names(
    tmp_path, lammps_executable, eam_potential, mode
):
    structure = _structure()
    engine_input = (
        CalcInputStatic()
        if mode == "static"
        else CalcInputMD(mode="NVE", temperature=300.0, n_ionic_steps=10, n_print=5)
    )
    default = calculate(
        structure=structure,
        engine=_engine(
            engine_input, tmp_path / "default", lammps_executable, eam_potential
        ),
    )
    engine = _engine(
        engine_input,
        tmp_path,
        lammps_executable,
        eam_potential,
        input_script_read_data_file="struct.data",
        input_script_dump_filename="traj.dump",
    )

    out = calculate(structure=structure, engine=engine)

    _check_output(out, structure, engine)
    files = os.listdir(engine.working_directory)
    assert "struct.data" in files and "traj.dump" in files
    assert "lammps.data" not in files and "dump.out" not in files
    assert len(out.structures) == len(default.structures)
    if mode == "static":
        assert out.final_energy == pytest.approx(default.final_energy, abs=1e-10)


def test_engine_is_reusable(tmp_path, lammps_executable, eam_potential):
    """One engine, several structures and sub-directories, no state carried over."""
    engine = _engine(CalcInputStatic(), tmp_path, lammps_executable, eam_potential)
    fe = bulk("Fe", "bcc", a=2.87, cubic=True).repeat((2, 2, 2))
    al_first = fe.copy()
    al_first[0].symbol = "Al"

    out_fe = calculate(structure=fe, engine=engine.with_working_directory("fe"))
    out_al = calculate(structure=al_first, engine=engine.with_working_directory("al"))

    assert out_fe.final_structure.get_chemical_symbols() == ["Fe"] * 16
    assert out_al.final_structure.get_chemical_symbols() == (["Al"] + ["Fe"] * 15)
    assert out_fe.final_energy != pytest.approx(out_al.final_energy)
    assert engine.potential_elements is None
    assert engine.calc_fn is None
    assert sorted(os.listdir(engine.working_directory)) == ["al", "fe"]


def test_calculate_as_workflow_node(tmp_path, lammps_executable, eam_potential):
    pwf = pytest.importorskip("pyiron_workflow")
    structure = _structure()
    engine = _engine(CalcInputStatic(), tmp_path, lammps_executable, eam_potential)

    run = pwf.run(calculate, structure=structure, engine=engine)

    _check_output(run.outputs.engine_output, structure, engine)
