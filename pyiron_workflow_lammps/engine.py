import os
import textwrap
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ase import Atoms
from pyiron_workflow_atomistics.dataclass_storage import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
    Engine,
)


@dataclass
class LammpsEngine(Engine):
    """
    Unified LAMMPS Engine using InputCalc dataclasses directly to build scripts.
    Mode is inferred from EngineInput by checking key attributes; boilerplate defaults
    are defined on the engine via `input_script_*` attributes.
    Thermostat and ensemble logic is implemented directly here.
    """

    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    mode: Literal["static", "minimize", "md"] = field(init=False)
    working_directory: str = field(default_factory=os.getcwd)
    input_filename: str = "in.lmp"
    command: str = "lmp -in in.lmp -log log.lammps"
    calc_fn: Callable = None
    calc_fn_kwargs: dict[str, Any] = None
    parse_fn: Callable = None
    parse_fn_kwargs: dict[str, Any] = None
    lammps_log_filepath: str = "log.lammps"
    lammps_log_convergence_printout: str = "Total wall time:"
    raw_script: str | None = None
    potential_elements: list[str] = None
    path_to_model: str = "/path/to/model"
    # Default boilerplate fields for the input script
    input_script_units: str = "metal"
    input_script_dimension: int = 3
    input_script_boundary: str | tuple[str, str, str] = ("p", "p", "p")
    input_script_atom_style: str = "atomic"
    input_script_read_data_file: str = "lammps.data"
    input_script_pair_style: str = "grace"
    input_script_compute_id: str = "eng"
    input_script_dump_every: int = 10
    input_script_dump_filename: str = "dump.out"
    input_script_dump_modify: str = (
        'dump_modify 1 sort id format line "'
        "%d %d "
        "%20.15g %20.15g %20.15g %20.15g %20.15g "
        '%20.15g %20.15g %20.15g %20.15g %20.15g %20.15g %20.15g"'
    )
    input_script_thermo_style_fields: tuple[str, ...] = (
        "step",
        "temp",
        "pe",
        "etotal",
        "pxx",
        "pxy",
        "pxz",
        "pyy",
        "pyz",
        "pzz",
        "vol",
    )
    input_script_thermo_format: str = "%20.15g"
    input_script_thermo_every: int = 10
    input_script_min_style: str = "cg"
    input_script_relax_pressure: float = 1e4
    input_script_relax_vmax: float = 0.001
    input_script_relax_type: Literal["iso", "aniso", "tri"] = "iso"

    def __post_init__(self):
        # Ensure the attribute exists and is consistent with EngineInput
        self.toggle_mode()
        # Normalize kwargs dicts
        if self.calc_fn_kwargs is None:
            self.calc_fn_kwargs = {}
        if self.parse_fn_kwargs is None:
            self.parse_fn_kwargs = {}

    def get_lammps_element_order(self, atoms: Atoms) -> list[str]:
        return list(dict.fromkeys(atoms.get_chemical_symbols()))

    def toggle_mode(self):
        # Infer mode from EngineInput type
        if isinstance(self.EngineInput, CalcInputMinimize):
            inferred = "minimize"
        elif isinstance(self.EngineInput, CalcInputMD):
            inferred = "md"
        elif isinstance(self.EngineInput, CalcInputStatic):
            inferred = "static"
        else:
            raise TypeError(f"Unsupported EngineInput type: {type(self.EngineInput)}")

        # If mode not present (or explicitly None), assign it
        if not hasattr(self, "mode") or getattr(self, "mode", None) is None:
            self.mode = inferred
            return self.mode

        # mode already present -> warn and do NOT overwrite
        current = self.mode
        if current != inferred:
            warnings.warn(
                f"'mode' already set to '{current}', but EngineInput implies '{inferred}'. "
                "Keeping existing 'mode' (not overwriting). Ensure they agree.",
                RuntimeWarning,
            )
        else:
            warnings.warn(
                f"'mode' already set to '{current}' and also implied by EngineInput; no change.",
                RuntimeWarning,
            )
        return current

    def _build_script(self, structure: Atoms) -> str:
        # Boilerplate header
        self.potential_elements = self.get_lammps_element_order(structure)

        pair_coeff = ("*", "*", self.path_to_model, " ".join(self.potential_elements))
        boundary = (
            self.input_script_boundary
            if isinstance(self.input_script_boundary, str)
            else " ".join(self.input_script_boundary)
        )
        lines = [
            f"units {self.input_script_units}",
            f"dimension {self.input_script_dimension}",
            f"boundary {boundary}",
            f"atom_style {self.input_script_atom_style}",
            "",
            f"read_data {self.input_script_read_data_file}",
            "",
            f"pair_style {self.input_script_pair_style}",
            f"pair_coeff {' '.join(pair_coeff)}",
            "",
            f"compute {self.input_script_compute_id} all pe/atom",
            f"dump 1 all custom {self.input_script_dump_every} {self.input_script_dump_filename} "
            f"id type xsu ysu zsu fx fy fz vx vy vz c_{self.input_script_compute_id}",
            self.input_script_dump_modify,
            "",
            "thermo_style custom " + " ".join(self.input_script_thermo_style_fields),
            f"thermo_modify format float {self.input_script_thermo_format}",
            f"thermo {self.input_script_thermo_every}",
            "",
        ]
        self.toggle_mode()
        print("current mode ", self.mode)
        # Mode-specific sections
        if self.mode == "static":
            lines += [f"min_style {self.input_script_min_style}", "minimize 0 0 0 0"]

        elif self.mode == "minimize":
            energy_convergence_tolerance = self.EngineInput.energy_convergence_tolerance
            force_convergence_tolerance = self.EngineInput.force_convergence_tolerance
            max_iterations = self.EngineInput.max_iterations
            max_evaluations = self.EngineInput.max_evaluations
            if self.EngineInput.relax_cell:
                lines.append(
                    f"fix 1 all box/relax {self.input_script_relax_type} {self.input_script_relax_pressure:.6f} vmax {self.input_script_relax_vmax}"
                )
            lines += [
                f"min_style {self.input_script_min_style}",
                f"minimize {energy_convergence_tolerance} {force_convergence_tolerance} {max_iterations} {max_evaluations}",
            ]

        elif self.mode == "md":
            # MD run: velocities & thermostat/integrator
            md = self.EngineInput
            T0 = md.initial_temperature or md.temperature
            # Velocity init
            lines.append(
                f"velocity all create {T0} {md.seed} mom yes rot yes dist gaussian"
            )
            # Thermostat/integrator logic
            if md.mode == "NVE":
                lines.append("fix 1 all nve")

            elif md.mode == "NVT":
                if md.thermostat == "nose-hoover":
                    lines.append(
                        f"fix 1 all nvt temp {md.temperature} {md.temperature} {md.temperature_damping_timescale}"
                    )
                elif md.thermostat == "berendsen":
                    lines.append(
                        f"fix 1 all temp/berendsen {md.temperature} {md.temperature} {md.temperature_damping_timescale}"
                    )
                elif md.thermostat == "andersen":
                    lines.append(
                        f"fix 1 all langevin {md.temperature} {md.temperature} {md.temperature_damping_timescale} {md.seed}"
                    )
                    lines.append("fix 2 all nve")
                elif md.thermostat == "temp/rescale":
                    delta = md.delta_temp or 0.0
                    lines.append(
                        f"fix 1 all temp/rescale {md.n_print} {md.temperature} {md.temperature} {delta} units box"
                    )
                elif md.thermostat == "temp/csvr":
                    lines.append(
                        f"fix 1 all temp/csvr {md.temperature} {md.temperature} {md.temperature_damping_timescale}"
                    )
                elif md.thermostat == "langevin":
                    lines.append(
                        f"fix 1 all langevin {md.temperature} {md.temperature} "
                        f"{md.temperature_damping_timescale} {md.seed}"
                    )
                    lines.append("fix 2 all nve")
                else:
                    raise ValueError(f"Unsupported thermostat: {md.thermostat}")

            elif md.mode == "NPT":
                pressure_bar = md.pressure / 1e5  # 1 bar = 1e5 Pa
                if md.thermostat != "nose-hoover":
                    raise ValueError("NPT mode supports only 'nose-hoover' thermostat")
                lines.append(
                    f"fix 1 all npt temp {md.temperature} {md.temperature} {md.temperature_damping_timescale} "
                    f"iso {pressure_bar} {pressure_bar} {md.pressure_damping_timescale}"
                )
            else:
                raise ValueError(f"Unknown MD mode: {md.mode}")

            # Timestep, thermo frequency, and run
            lines.append(f"timestep {md.time_step}")
            lines.append(f"thermo {md.n_print}")
            lines.append(f"run {md.n_ionic_steps}")
            # print(lines)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return textwrap.dedent("\n".join(lines))

    def write_input_file(self, structure: Atoms) -> str:
        """
        Write the built LAMMPS script to `working_directory`/`input_filename`.
        Returns the path to the written file.
        """
        script = self._build_script(structure)
        path = os.path.join(self.working_directory, self.input_filename)
        with open(path, "w") as f:
            f.write(script)
        return path

    def get_calculate_fn(self, structure: Atoms) -> tuple[Callable, dict[str, Any]]:
        if self.calc_fn is None:
            from pyiron_workflow_lammps.lammps import lammps_calculator_fn

            self.potential_elements = self.get_lammps_element_order(structure)
            self.calc_fn = lammps_calculator_fn
            self.calc_fn_kwargs = {
                "working_directory": self.working_directory,
                "lammps_input": self._build_script(structure),
                "potential_elements": self.potential_elements,
                "input_filename": self.input_filename,
                "command": self.command,
                "lammps_log_filepath": self.lammps_log_filepath,
                "units": self.input_script_units,
                "lammps_log_convergence_printout": self.lammps_log_convergence_printout,
            }
        return self.calc_fn, self.calc_fn_kwargs

    def get_parse_fn(self) -> Callable:
        if self.parse_fn is None:
            from pyiron_workflow_lammps.lammps import parse_LammpsOutput

            self.parse_fn = parse_LammpsOutput
        return self.parse_fn, self.parse_fn_kwargs
