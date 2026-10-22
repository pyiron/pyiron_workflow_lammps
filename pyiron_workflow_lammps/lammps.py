# Standard library imports
import os
import warnings
from collections.abc import Callable
from typing import Any

import flowrep as fr
import numpy as np
from ase import Atoms
from pyiron_snippets.logger import logger

from pyiron_workflow_lammps.generic import (
    ShellOutput,
    create_WorkingDirectory,
    isLineInFile,
    shell,
)

DEFAULT_COMMAND = "lmp -in in.lmp -log log.lammps"

# lammpsparser warns on every parse that it has no unit conversion for the
# per-atom energy compute (``c_eng``) in our dump; the values are already eV.
warnings.filterwarnings(
    "ignore",
    message=".*unit conversion type of quantity eng.*",
    category=UserWarning,
)

# LAMMPS minimizer stopping criteria that mean the tolerances were met.
# Anything else ("max iterations", "max force evaluations", "linesearch alpha
# is zero", ...) stopped the minimization before convergence.
CONVERGED_STOPPING_CRITERIA = ("energy tolerance", "force tolerance", "forces are zero")


@fr.atomic("working_directory")
def write_LammpsStructure(
    structure,
    working_directory,
    potential_elements,
    units="metal",
    file_name="lammps.data",
):
    from lammpsparser import write_lammps_structure

    write_lammps_structure(
        structure=structure,
        potential_elements=potential_elements,
        units=units,
        file_name=file_name,
        working_directory=working_directory,
    )
    return working_directory


@fr.atomic("filepath")
def write_LammpsInput(
    lammps_input: str, filename: str, working_directory: str | None = None
) -> str:
    """
    Write a LAMMPS input script to disk and return the full file path.

    Parameters
    ----------
    lammps_input
        An instance of LammpsInput (your dataclass) containing all settings.
        If `lammps_input.raw_script` is set, that script is written verbatim.
    filename
        Name of the file to write (e.g. "in.lammps").
    working_directory
        Directory in which to place the file. If None, writes to the current
        working directory. The directory will be created if it doesn't exist.

    Returns
    -------
    filepath : str
        The full path to the file that was written.
    """
    script = str(lammps_input)

    if working_directory is not None:
        os.makedirs(working_directory, exist_ok=True)
        path = os.path.join(working_directory, filename)
    else:
        path = filename

    with open(path, "w") as f:
        f.write(script)

    return path


@fr.atomic("lammps_output")
def parse_LammpsOutput(
    working_directory: str,
    potential_elements: list[str],
    units: str,
    prism: list[list[float]] | None = None,
    lammps_structure_filepath: str = "lammps.data",
    dump_out_file_name: str = "dump.out",
    log_lammps_file_name: str = "log.lammps",
    log_lammps_convergence_printout: str = "Total wall time:",
    _parser_fn=None,
    _parser_fn_kwargs=None,
    after: object = None,
):
    """
    Parse a finished LAMMPS run into an ``EngineOutput``.

    ``final_energy`` and ``energies`` are the potential energy (thermo
    ``pe``/``PotEng``), the energy the forces are the negative gradient of.
    The total energy (``etotal``) additionally contains the kinetic energy
    of an MD run. ``final_stress`` and ``stresses`` are lammpsparser's
    pressure tensors (GPa, compression positive), not ASE stress.
    ``n_ionic_steps`` is the last LAMMPS step number of the run.

    ``converged`` requires the log's ``log_lammps_convergence_printout`` line
    and, for a minimization, a stopping criterion that means the tolerances
    were met (see ``CONVERGED_STOPPING_CRITERIA``).

    If ``_parser_fn`` is given it is called as ``_parser_fn(**_parser_fn_kwargs)``
    instead, and its return value is passed through unchanged.

    ``after`` orders this node after the LAMMPS run. When it is the run's
    :class:`generic.ShellOutput` (as in :func:`lammps_job`) and the command
    failed, a ``RuntimeError`` with its output is raised instead of parsing
    whatever files an earlier run left in ``working_directory``.
    """
    from ase.io.lammpsdata import read_lammps_data
    from lammpsparser import parse_lammps_output_files
    from pyiron_workflow_atomistics.engine import EngineOutput

    if isinstance(after, ShellOutput) and after.return_code != 0:
        raise RuntimeError(
            f"LAMMPS command exited with return code {after.return_code} in "
            f"{working_directory}.\n--- stdout (tail) ---\n{after.stdout[-2000:]}"
            f"\n--- stderr (tail) ---\n{after.stderr[-2000:]}"
        )
    if _parser_fn is None:
        try:
            lammps_EngineOutput = read_lammps_data(
                os.path.join(working_directory, lammps_structure_filepath)
            )
            logger.info("parse_LammpsOutput: Successfully read LAMMPS structure file")
        except Exception as e:
            raise Exception(
                f"Error parsing LAMMPS output: {e}, you must provide a parser function (_parser_fn) and kwargs (_parser_fn_kwargs) if you want to use a custom parser."
            )

        lammpsparser_output = parse_lammps_output_files(
            working_directory=working_directory,
            structure=read_lammps_data(
                os.path.join(working_directory, lammps_structure_filepath)
            ),
            potential_elements=potential_elements,
            units=units,
            prism=prism,
            dump_out_file_name=dump_out_file_name,
            log_lammps_file_name=log_lammps_file_name,
        )
        species_lists = get_structure_species_lists(
            lammps_data_filepath=os.path.join(
                working_directory, lammps_structure_filepath
            ),
            lammps_dump_filepath=os.path.join(working_directory, dump_out_file_name),
        )
        # Walk the per-step lammpsparser_output and build trajectory + finals.
        # Attach per-atom velocities when present so each trajectory frame carries
        # momenta (needed for kinetic temperature in coexistence analyses).
        generic_traj = lammpsparser_output["generic"]
        velocities_traj = generic_traj.get("velocities")
        atoms_list = []
        for i in range(len(generic_traj["cells"])):
            atoms_list.append(
                arrays_to_ase_atoms(
                    cells=generic_traj["cells"][i],
                    positions=generic_traj["positions"][i],
                    indices=generic_traj["indices"][i],
                    species_lists=species_lists,
                    velocities=(
                        velocities_traj[i] if velocities_traj is not None else None
                    ),
                )
            )

        log_path = os.path.join(working_directory, log_lammps_file_name)
        converged = isLineInFile(
            filepath=log_path,
            line=log_lammps_convergence_printout,
            exact_match=False,
        ) and _minimization_converged(log_path)

        final_atoms = atoms_list[-1]
        energies_traj = generic_traj["energy_pot"]
        forces_traj = generic_traj["forces"]
        stresses_traj = generic_traj["pressures"]

        # lammpsparser's "generic" dict holds the step numbers of the thermo
        # output under "steps" (older versions: "step"); the last one is the
        # number of steps LAMMPS took. Fall back to the trajectory length.
        steps = generic_traj.get("steps", generic_traj.get("step"))
        n_ionic_steps = (
            int(np.asarray(steps).reshape(-1)[-1])
            if steps is not None and np.size(steps) > 0
            else len(atoms_list) - 1
        )

        lammps_EngineOutput = EngineOutput(
            final_structure=final_atoms,
            final_energy=float(energies_traj[-1]),
            converged=bool(converged),
            final_forces=forces_traj[-1],
            final_stress=stresses_traj[-1],  # 3x3 stress tensor
            final_volume=final_atoms.get_volume(),
            energies=energies_traj,
            forces=forces_traj,
            stresses=stresses_traj,
            structures=atoms_list,
            n_ionic_steps=n_ionic_steps,
        )
    else:
        try:
            lammps_EngineOutput = _parser_fn(**(_parser_fn_kwargs or {}))
        except Exception as e:
            raise Exception(
                f"Error parsing LAMMPS output: {e}, you must provide a parser function (_parser_fn) and kwargs (_parser_fn_kwargs) if you want to use a custom parser."
            )

    return lammps_EngineOutput


def _minimization_converged(log_path: str) -> bool:
    """False if the log's last minimizer stopping criterion is not a converged one.

    Logs without a minimization (static ``run 0``, MD) have no stopping
    criterion and count as converged.
    """
    criterion = None
    with open(log_path) as f:
        for line in f:
            if "Stopping criterion =" in line:
                criterion = line.split("=", 1)[1].strip()
    return criterion is None or criterion in CONVERGED_STOPPING_CRITERIA


def get_structure_species_lists(
    lammps_data_filepath="lammps.data", lammps_dump_filepath="dump.out"
):
    """
    Prompt for a LAMMPS data file path, parse its Masses section,
    and return a dict mapping atom-type indices to element symbols.
    """
    from pymatgen.io.lammps.outputs import parse_lammps_dumps

    species_map = get_species_map(lammps_data_filepath)
    species_lists = []
    logger.info(f"Get structure species lists operating in {lammps_dump_filepath}")
    for dump in parse_lammps_dumps(lammps_dump_filepath):
        species_lists.append([species_map[idx] for idx in dump.data.type])
    return species_lists


def get_species_map(lammps_data_filepath="lammps.data"):
    species_map = {}
    with open(lammps_data_filepath) as f:
        # find the "Masses" section
        for line in f:
            if line.strip() == "Masses":
                # skip the blank line immediately after
                next(f)
                break
        # read until the next blank line
        for line in f:
            stripped = line.strip()
            if not stripped:
                # end of the Masses block
                break
            # split off any comment after '#'
            parts = stripped.split("#", 1)
            # first token is "<index> <mass>"
            idx = int(parts[0].split()[0])
            # if there's a comment like "(Fe)", strip parentheses
            symbol = parts[1].strip().strip("()") if len(parts) > 1 else None
            species_map[idx] = symbol
    return species_map


def arrays_to_ase_atoms(
    cells: np.ndarray,  # (n_frames, 3, 3)
    positions: np.ndarray,  # (n_frames, n_atoms, 3)
    indices: np.ndarray,  # (n_frames, n_atoms)
    species_lists: list[list[str]],  # e.g. {0:'Fe',1:'C'} or {1:'Fe',2:'C'}
    pbc: bool = True,
    velocities: np.ndarray | None = None,  # (n_atoms, 3), optional per-atom velocities
) -> Atoms:
    """
    Convert the final frame of LAMMPS‐style arrays into an ASE Atoms.

    If ``velocities`` is given, attach them so the frame carries momenta (used by
    kinetic-temperature analyses of MD trajectories).

    Raises KeyError if any type in `indices` isn't a key in `species_map`.
    """

    # last frame
    cell = cells
    pos = positions
    types = indices

    atoms = Atoms(symbols=species_lists[-1], positions=pos, cell=cell, pbc=pbc)
    if velocities is not None:
        from ase import units as _u

        # lammpsparser reports velocities in Å/fs; ASE Atoms use Å/(ASE time unit),
        # so convert (else get_temperature() is ~1/units.fs**2 ≈ 103.6x too low).
        atoms.set_velocities(np.asarray(velocities) / _u.fs)
    return atoms


@fr.workflow("lammps_output")
def lammps_job(
    working_directory: str,
    structure: Atoms,
    lammps_input: str,
    units: str,
    potential_elements: list[str],
    lammps_structure_filepath: str = "lammps.data",
    input_filename: str = "in.lmp",
    dump_out_file_name: str = "dump.out",
    command: str = DEFAULT_COMMAND,
    lammps_log_filepath: str = "log.lammps",
    lammps_log_convergence_printout: str = "Total wall time:",
    _lammps_parser_function: Callable[..., Any] | None = None,
    _lammps_parser_args: dict[str, Any] | None = None,
):
    """
    Run a LAMMPS calculation end-to-end:
      1) create working_directory
      2) write structure → data file
      3) write LammpsInput → input script
      4) shell out to LAMMPS
      5) parse results with an (optional) custom parser

    pyiron_workflow 0.19 removed execution signals (``>>``/``starting_nodes``),
    so the order is expressed as data edges: each step consumes the previous
    step's output (``after=`` arguments are ordering tokens that are never
    read).

    Args:
        working_directory: Path to the working directory.
        structure: ASE Atoms object to simulate.
        lammps_input: str representation of the lammps input script.
        potential_elements: List of element symbols in the potential.
        input_filename: Name of the input script (default "in.lmp").
        command: Full MPI/launcher command to run LAMMPS.
        _lammps_parser_function: If provided, use this instead of the default parser.
        _lammps_parser_args: Dict of kwargs to pass to the parser function.

    Returns:
        Parsed LAMMPS output (whatever the parser returns).
    """
    wd_created = create_WorkingDirectory(working_directory=working_directory)
    wd_structure = write_LammpsStructure(
        structure=structure,
        working_directory=wd_created,
        potential_elements=potential_elements,
        units=units,
        file_name=lammps_structure_filepath,
    )
    input_path = write_LammpsInput(
        lammps_input=lammps_input,
        filename=input_filename,
        working_directory=wd_structure,
    )
    job = shell(command=command, working_directory=wd_structure, after=input_path)
    lammps_output = parse_LammpsOutput(
        working_directory=wd_structure,
        potential_elements=potential_elements,
        units=units,
        lammps_structure_filepath=lammps_structure_filepath,
        dump_out_file_name=dump_out_file_name,
        log_lammps_file_name=lammps_log_filepath,
        log_lammps_convergence_printout=lammps_log_convergence_printout,
        _parser_fn=_lammps_parser_function,
        _parser_fn_kwargs=_lammps_parser_args,
        after=job,
    )
    return lammps_output


def lammps_calculator_fn(
    working_directory: str,
    structure: Atoms,
    lammps_input: str,
    units: str,
    potential_elements: list[str],
    input_filename: str = "in.lmp",
    command: str = DEFAULT_COMMAND,
    lammps_log_filepath: str = "log.lammps",
    lammps_log_convergence_printout: str = "Total wall time:",
    _lammps_parser_function: Callable[..., Any] | None = None,
    _lammps_parser_args: dict[str, Any] | None = None,
    lammps_structure_filepath: str = "lammps.data",
    dump_out_file_name: str = "dump.out",
):
    """The ``calc_fn`` LammpsEngine hands to ``pyiron_workflow_atomistics``.

    Calls :func:`lammps_job` as plain Python (flowrep-decorated functions stay
    directly callable), so it runs inside whatever node invoked the engine.
    """
    return lammps_job(
        working_directory=working_directory,
        structure=structure,
        lammps_input=lammps_input,
        units=units,
        potential_elements=potential_elements,
        lammps_structure_filepath=lammps_structure_filepath,
        input_filename=input_filename,
        dump_out_file_name=dump_out_file_name,
        command=command,
        lammps_log_filepath=lammps_log_filepath,
        lammps_log_convergence_printout=lammps_log_convergence_printout,
        _lammps_parser_function=_lammps_parser_function,
        _lammps_parser_args=_lammps_parser_args,
    )
