# Standard library imports
import os
from collections.abc import Callable
from typing import Any

import numpy as np

# Local imports
import pyiron_workflow as pwf
from ase import Atoms
from pyiron_snippets.logger import logger


@pwf.as_function_node("working_directory")
def write_LammpsStructure(
    structure,
    working_directory,
    potential_elements,
    units="metal",
    file_name="lammps.data",
):
    from pyiron_lammps import write_lammps_structure

    write_lammps_structure(
        structure=structure,
        potential_elements=potential_elements,
        units=units,
        file_name=file_name,
        working_directory=working_directory,
    )
    return working_directory


@pwf.as_function_node("filepath")
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


@pwf.as_function_node("lammps_output")
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
):
    import warnings

    from ase.io.lammpsdata import read_lammps_data
    from pyiron_lammps import parse_lammps_output_files
    from pyiron_workflow_atomistics.engine import EngineOutput

    from pyiron_workflow_lammps.generic import isLineInFile

    warnings.filterwarnings("ignore")
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

        pyiron_lammps_output = parse_lammps_output_files(
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
        # Walk the per-step pyiron_lammps_output and build trajectory + finals.
        atoms_list = []
        for i in range(len(pyiron_lammps_output["generic"]["cells"])):
            atoms_list.append(
                arrays_to_ase_atoms(
                    cells=pyiron_lammps_output["generic"]["cells"][i],
                    positions=pyiron_lammps_output["generic"]["positions"][i],
                    indices=pyiron_lammps_output["generic"]["indices"][i],
                    species_lists=species_lists,
                )
            )

        converged = isLineInFile.node_function(
            filepath=os.path.join(working_directory, log_lammps_file_name),
            line=log_lammps_convergence_printout,
            exact_match=False,
        )

        final_atoms = atoms_list[-1]
        energies_traj = pyiron_lammps_output["generic"]["energy_tot"]
        forces_traj = pyiron_lammps_output["generic"]["forces"]
        stresses_traj = pyiron_lammps_output["generic"]["pressures"]

        lammps_EngineOutput = EngineOutput(
            final_structure=final_atoms,
            final_energy=energies_traj[-1],
            converged=bool(converged),
            final_forces=forces_traj[-1],
            final_stress=stresses_traj[-1],          # 3x3 stress tensor
            final_volume=final_atoms.get_volume(),
            energies=energies_traj,
            forces=forces_traj,
            stresses=stresses_traj,
            structures=atoms_list,
            n_ionic_steps=pyiron_lammps_output["generic"]["steps"],
        )
    else:
        try:
            lammps_EngineOutput = _parser_fn(**_parser_fn_kwargs)
        except Exception as e:
            raise Exception(
                f"Error parsing LAMMPS output: {e}, you must provide a parser function (_parser_fn) and kwargs (_parser_fn_kwargs) if you want to use a custom parser."
            )

    return lammps_EngineOutput


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
) -> Atoms:
    """
    Convert the final frame of LAMMPS‐style arrays into an ASE Atoms.

    Raises KeyError if any type in `indices` isn't a key in `species_map`.
    """

    # last frame
    cell = cells
    pos = positions
    types = indices

    return Atoms(symbols=species_lists[-1], positions=pos, cell=cell, pbc=pbc)


@pwf.as_macro_node("lammps_output")
def lammps_job(
    self,
    working_directory: str,
    structure: Atoms,
    lammps_input: str,
    units: str,
    potential_elements: list[str],
    lammps_structure_filepath: str = "lammps.data",
    input_filename: str = "in.lmp",
    dump_out_file_name: str = "dump.out",
    command: str = "mpirun -np 40 --bind-to none /cmmc/ptmp/hmai/LAMMPS/lammps_grace/build/lmp -in in.lmp -log log.lammps",
    lammps_log_filepath: str = "log.lammps",
    lammps_log_convergence_printout: str = "Total wall time:",
    _lammps_parser_function: Callable[..., Any] | None = None,
    _lammps_parser_args: dict[str, Any] = {},
):
    """
    Run a LAMMPS calculation end-to-end:
      1) create working_directory
      2) write structure → data file
      3) write LammpsInput → input script
      4) shell out to LAMMPS
      5) parse results with an (optional) custom parser

    Args:
        working_directory: Path to the working directory.
        structure: ASE Atoms object to simulate.
        lammps_input: str representation of the lammps input script.
        potential_elements: List of element symbols in the potential.
        input_filename: Name of the input script (default "in.lmp").
        command: Full MPI/launcher command to run LAMMPS.
        lammps_parser_function: If provided, use this instead of the default parse_lammps_output.
        lammps_parser_args: Dict of kwargs to pass to the parser function.

    Returns:
        Parsed LAMMPS output (whatever the parser returns).
    """
    from pyiron_workflow_lammps.generic import create_WorkingDirectory, shell

    # 1) Create the directory
    self.working_dir = create_WorkingDirectory(working_directory=working_directory)

    # 2) Write the ASE structure to LAMMPS data
    self.structure_writer = write_LammpsStructure(
        structure=structure,
        potential_elements=potential_elements,
        units=units,
        file_name=lammps_structure_filepath,
        working_directory=working_directory,
    )

    # 3) Write the input script
    self.input_writer = write_LammpsInput(
        lammps_input=lammps_input,
        filename=input_filename,
        working_directory=working_directory,
    )

    # 4) Run LAMMPS
    self.job = shell(command=command, working_directory=working_directory)

    # 5) Parse output (using custom or default parser)
    self.lammps_output = parse_LammpsOutput(
        working_directory=working_directory,
        potential_elements=potential_elements,
        units=units,
        prism=None,
        lammps_structure_filepath=lammps_structure_filepath,
        dump_out_file_name=dump_out_file_name,
        log_lammps_file_name=lammps_log_filepath,
        log_lammps_convergence_printout=lammps_log_convergence_printout,
        _parser_fn=_lammps_parser_function,
        _parser_fn_kwargs=_lammps_parser_args,
    )

    # Wire up the flow
    (
        self.working_dir
        >> self.structure_writer
        >> self.input_writer
        >> self.job
        >> self.lammps_output
    )

    # Register start
    self.starting_nodes = [self.working_dir]

    return self.lammps_output


def lammps_calculator_fn(
    working_directory: str,
    structure: Atoms,
    lammps_input: str,
    units: str,
    potential_elements: list[str],
    input_filename: str = "in.lmp",
    command: str = "mpirun -np 40 --bind-to none /cmcm/ptmp/hmai/LAMMPS/lammps_grace/build/lmp -in in.lmp -log minimize.log",
    lammps_log_filepath: str = "log.lammps",
    lammps_log_convergence_printout: str = "Total wall time:",
    _lammps_parser_function: Callable[..., Any] | None = None,
    _lammps_parser_args: dict[str, Any] = {},
):
    output = lammps_job(
        working_directory=working_directory,
        structure=structure,
        lammps_input=lammps_input,
        units=units,
        potential_elements=potential_elements,
        input_filename=input_filename,
        command=command,
        lammps_log_filepath=lammps_log_filepath,
        lammps_log_convergence_printout=lammps_log_convergence_printout,
        _lammps_parser_function=_lammps_parser_function,
        _lammps_parser_args=_lammps_parser_args,
    )()
    return output["lammps_output"]
