import os
import random
import textwrap
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ase import Atoms
from pyiron_workflow_atomistics.engine import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
)


@dataclass
class LammpsEngine:
    """
    Unified LAMMPS Engine using InputCalc dataclasses directly to build scripts.
    Mode is inferred from EngineInput by checking key attributes; boilerplate defaults
    are defined on the engine via `input_script_*` attributes.
    Thermostat and ensemble logic is implemented directly here.

    The engine is never mutated by use: ``get_calculate_fn`` builds the input
    script for each structure it is given, so one engine (and copies made with
    ``with_working_directory`` or ``dataclasses.replace``) can be reused across
    structures and threads. ``potential_elements=None`` means "the structure's
    elements in order of first occurrence".
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
    max_evaluations: int = 10000  # New: Engine-specific, NOT taken from dataclass
    max_iterations: int | None = None  # New: Engine-specific, NOT taken from dataclass

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
    # 12 dump columns: id type (ints) + xsu ysu zsu fx fy fz vx vy vz c_eng (10 floats).
    # The format line MUST have exactly 12 tokens (2 %d + 10 %20.15g). A previous
    # default had 12 %20.15g (14 tokens total) — 2 too many — which makes LAMMPS
    # SEGFAULT the first time a dump frame is written (i.e. at step 0 of any MD run;
    # minimize escaped it only when it converged before a dump trigger fired).
    input_script_dump_modify: str = (
        'dump_modify 1 sort id format line "'
        "%d %d "
        "%20.15g %20.15g %20.15g %20.15g %20.15g "
        '%20.15g %20.15g %20.15g %20.15g %20.15g"'
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

    def with_working_directory(self, subdir: str) -> "LammpsEngine":
        """Return a copy of this engine with working_directory composed.

        Pure — never mutates self. Nothing is cached on the engine, so the
        copy builds its script against the new directory, and a custom
        ``calc_fn``/``parse_fn`` carries over.
        """
        return replace(
            self,
            working_directory=os.path.join(self.working_directory, subdir),
        )

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
        return current

    def _potential_elements(self, structure: Atoms) -> list[str]:
        if self.potential_elements is not None:
            return list(self.potential_elements)
        return self.get_lammps_element_order(structure)

    def _build_script(self, structure: Atoms) -> str:
        # Boilerplate header
        potential_elements = self._potential_elements(structure)

        pair_coeff = ("*", "*", self.path_to_model, " ".join(potential_elements))
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
        # Mode-specific sections
        if self.mode == "static":
            lines += ["run 0"]

        elif self.mode == "minimize":
            energy_convergence_tolerance = self.EngineInput.energy_convergence_tolerance
            force_convergence_tolerance = self.EngineInput.force_convergence_tolerance
            if self.max_iterations is not None:
                warnings.warn(
                    f"'max_iterations' provided to Engine takes precedence over EngineInput.max_iterations. Overwriting EngineInput.max_iterations {self.EngineInput.max_iterations} with {self.max_iterations}",
                    RuntimeWarning,
                )
                max_iterations = self.max_iterations
            else:
                max_iterations = self.EngineInput.max_iterations
            max_evaluations = (
                self.max_evaluations
            )  # Because counting force calls is a lammps specific thing
            if self.EngineInput.relax_cell:
                lines.append(
                    f"fix 1 all box/relax {self.input_script_relax_type} {self.input_script_relax_pressure:.6f} vmax {self.input_script_relax_vmax}"
                )
            # Dump on the thermo steps so trajectory frames and energies align
            lines += self._final_frame_dump(
                self.input_script_thermo_every, max_iterations
            )
            lines += [
                f"min_style {self.input_script_min_style}",
                f"minimize {energy_convergence_tolerance} {force_convergence_tolerance} {max_iterations} {max_evaluations}",
            ]

        elif self.mode == "md":
            # MD run: velocities & thermostat/integrator
            md = self.EngineInput
            # seed=None means non-deterministic (CalcInputMD contract); LAMMPS
            # needs an explicit positive seed, so draw one per script. The
            # Marsaglia RNG behind fix langevin accepts at most 900000000.
            seed = (
                md.seed
                if md.seed is not None
                else random.SystemRandom().randint(1, 900_000_000)
            )
            T0 = (
                md.initial_temperature
                if md.initial_temperature is not None
                else md.temperature
            )
            # CalcInputMD time quantities are in FEMTOSECONDS (ASE/pwa convention), but
            # LAMMPS interprets times in the `units` system's native unit: ps for `metal`,
            # fs for `real`. Without this conversion `timestep 2.0` (meant as 2 fs) becomes
            # 2 ps under metal -> a 1000x-too-large step that explodes the run after a few
            # steps. Convert fs -> native time unit here.
            _time_fac = {"metal": 1e-3, "real": 1.0, "si": 1e-15, "cgs": 1e-15}.get(
                self.input_script_units, 1.0
            )
            dt = md.time_step * _time_fac
            tdamp = md.thermostat_time_constant * _time_fac
            pdamp = md.pressure_damping_timescale * _time_fac
            # Velocity init (LAMMPS cannot create velocities at 0 K)
            if T0 > 0:
                lines.append(
                    f"velocity all create {T0} {seed} mom yes rot yes dist gaussian"
                )
            # Thermostat/integrator logic
            if md.mode == "NVE":
                lines.append("fix 1 all nve")

            elif md.mode == "NVT":
                if md.thermostat == "nose-hoover":
                    lines.append(
                        f"fix 1 all nvt temp {md.temperature} {md.temperature} {tdamp}"
                    )
                elif md.thermostat == "berendsen":
                    # temp/berendsen only rescales velocities; it does not integrate
                    lines.append(
                        f"fix 1 all temp/berendsen {md.temperature} {md.temperature} {tdamp}"
                    )
                    lines.append("fix 2 all nve")
                elif md.thermostat == "andersen":
                    lines.append(
                        f"fix 1 all langevin {md.temperature} {md.temperature} {tdamp} {seed}"
                    )
                    lines.append("fix 2 all nve")
                elif md.thermostat == "temp/rescale":
                    delta = getattr(md, "delta_temp", None) or 0.0
                    lines.append(
                        f"fix 1 all temp/rescale {max(md.n_print, 1)} {md.temperature} {md.temperature} {delta} 1.0"
                    )
                    lines.append("fix 2 all nve")
                elif md.thermostat == "temp/csvr":
                    lines.append(
                        f"fix 1 all temp/csvr {md.temperature} {md.temperature} {tdamp} {seed}"
                    )
                    lines.append("fix 2 all nve")
                elif md.thermostat == "langevin":
                    lines.append(
                        f"fix 1 all langevin {md.temperature} {md.temperature} "
                        f"{tdamp} {seed}"
                    )
                    lines.append("fix 2 all nve")
                else:
                    raise ValueError(f"Unsupported thermostat: {md.thermostat}")

            elif md.mode == "NPT":
                if md.pressure is None:
                    raise ValueError("NPT mode requires CalcInputMD.pressure (in Pa)")
                pressure_bar = md.pressure / 1e5  # 1 bar = 1e5 Pa
                if md.thermostat != "nose-hoover":
                    raise ValueError("NPT mode supports only 'nose-hoover' thermostat")
                lines.append(
                    f"fix 1 all npt temp {md.temperature} {md.temperature} {tdamp} "
                    f"iso {pressure_bar} {pressure_bar} {pdamp}"
                )
            else:
                raise ValueError(f"Unknown MD mode: {md.mode}")

            # Timestep, thermo frequency, and run
            lines.append(f"timestep {dt}")
            lines.append(f"thermo {md.n_print}")
            lines += self._final_frame_dump(md.n_print, md.n_ionic_steps)
            lines.append(f"run {md.n_ionic_steps}")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return textwrap.dedent("\n".join(lines))

    @staticmethod
    def _final_frame_dump(every: int, last_step: int) -> list[str]:
        """Dump every ``every`` steps *and* on ``last_step``.

        Thermo output always includes the last step of a run, a plain
        ``dump ... N`` only multiples of N, so the final structure/forces
        could come from an earlier step than the final energy. Driving the
        dump with a variable that also returns ``last_step`` keeps the dump
        frames on the thermo steps (``first yes`` keeps the step-0 frame).
        ``every <= 0`` (``thermo 0``: only first and last step) dumps those two.
        """
        if every <= 0:
            every = max(last_step, 1)
        return [
            f'variable pwl_next_dump equal "ternary(step+{every}<{last_step},'
            f'step+{every},ternary(step<{last_step},{last_step},step+{every}))"',
            "dump_modify 1 every v_pwl_next_dump first yes",
        ]

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
        """Return ``(calc_fn, kwargs)``; the caller runs ``calc_fn(structure=..., **kwargs)``.

        A user-supplied ``calc_fn`` is returned with a copy of ``calc_fn_kwargs``.
        Otherwise the default LAMMPS runner is returned with kwargs built for
        this ``structure``; a custom ``parse_fn`` is passed on to it.
        """
        if self.calc_fn is not None:
            return self.calc_fn, dict(self.calc_fn_kwargs)

        from pyiron_workflow_lammps.lammps import lammps_calculator_fn

        calc_fn_kwargs = {
            "working_directory": self.working_directory,
            "lammps_input": self._build_script(structure),
            "potential_elements": self._potential_elements(structure),
            "input_filename": self.input_filename,
            "command": self.command,
            "lammps_log_filepath": self.lammps_log_filepath,
            "units": self.input_script_units,
            "lammps_log_convergence_printout": self.lammps_log_convergence_printout,
            "lammps_structure_filepath": self.input_script_read_data_file,
            "dump_out_file_name": self.input_script_dump_filename,
        }
        if self.parse_fn is not None:
            calc_fn_kwargs["_lammps_parser_function"] = self.parse_fn
            calc_fn_kwargs["_lammps_parser_args"] = dict(self.parse_fn_kwargs)
        return lammps_calculator_fn, calc_fn_kwargs

    def get_parse_fn(self) -> tuple[Callable, dict[str, Any]]:
        if self.parse_fn is None:
            from pyiron_workflow_lammps.lammps import parse_LammpsOutput

            return parse_LammpsOutput, self.parse_fn_kwargs
        return self.parse_fn, self.parse_fn_kwargs
