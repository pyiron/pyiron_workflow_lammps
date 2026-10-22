import copy
import inspect
import os
import pickle
import re
import shutil
import tempfile
import unittest
import warnings
from unittest.mock import patch

from ase import Atom
from ase.build import bulk
from pyiron_workflow_atomistics.engine import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
)

from pyiron_workflow_lammps.engine import LammpsEngine
from pyiron_workflow_lammps.lammps import lammps_calculator_fn, parse_LammpsOutput


class TestLammpsEngine(unittest.TestCase):
    """Test the LammpsEngine class."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_lammps_engine_init_static(self):
        """Test LammpsEngine initialization with static calculation."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        self.assertEqual(engine.mode, "static")
        self.assertEqual(engine.working_directory, self.temp_dir)
        self.assertEqual(engine.input_filename, "in.lmp")
        self.assertEqual(engine.command, "lmp -in in.lmp -log log.lammps")

    def test_lammps_engine_init_minimize(self):
        """Test LammpsEngine initialization with minimization calculation."""
        engine_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=1000,
            relax_cell=False,
        )
        engine = LammpsEngine(
            EngineInput=engine_input,
            working_directory=self.temp_dir,
            max_evaluations=2000,
        )

        self.assertEqual(engine.mode, "minimize")
        self.assertEqual(engine.EngineInput.energy_convergence_tolerance, 1e-6)
        self.assertEqual(engine.EngineInput.force_convergence_tolerance, 1e-5)

    def test_lammps_engine_init_md(self):
        """Test LammpsEngine initialization with MD calculation."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="nose-hoover",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        self.assertEqual(engine.mode, "md")
        self.assertEqual(engine.EngineInput.mode, "NVT")
        self.assertEqual(engine.EngineInput.temperature, 300)

    def test_lammps_engine_init_unsupported_type(self):
        """Test LammpsEngine initialization with unsupported input type."""
        with self.assertRaises(TypeError):
            LammpsEngine(EngineInput="invalid_input", working_directory=self.temp_dir)

    def test_get_lammps_element_order(self):
        """Test element order extraction from Atoms object."""
        # Single element
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        elements = engine.get_lammps_element_order(self.structure)
        self.assertEqual(elements, ["Fe"])

        # Multiple elements
        multi_structure = bulk("Fe", "bcc", a=2.87)
        multi_structure.append(Atom("C", (1.0, 1.0, 1.0)))
        elements = engine.get_lammps_element_order(multi_structure)
        self.assertEqual(elements, ["Fe", "C"])

        # Preserve order
        multi_structure = bulk("C", "diamond", a=3.57)
        multi_structure.append(Atom("Fe", (1.0, 1.0, 1.0)))
        elements = engine.get_lammps_element_order(multi_structure)
        self.assertEqual(elements, ["C", "Fe"])

    def test_toggle_mode(self):
        """Test mode detection from EngineInput type."""
        engine_input = CalcInputMinimize()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        # Mode should be set during initialization
        self.assertEqual(engine.mode, "minimize")

        # Test manual toggle
        engine.toggle_mode()
        self.assertEqual(engine.mode, "minimize")

    def test_build_script_static(self):
        """Test script building for static calculation."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check basic boilerplate
        self.assertIn("units metal", script)
        self.assertIn("dimension 3", script)
        self.assertIn("boundary p p p", script)
        self.assertIn("atom_style atomic", script)
        self.assertIn("read_data lammps.data", script)
        self.assertIn("pair_style grace", script)

        # Check static-specific content: a 0-step run, not 'minimize 0 0 0 0'
        # (which LAMMPS reports as stopped on max iterations)
        self.assertEqual(script.splitlines()[-1], "run 0")
        self.assertNotIn("minimize", script)
        self.assertNotIn("min_style", script)
        self.assertNotIn("pwl_next_dump", script)

    def test_build_script_minimize(self):
        """Test script building for minimization calculation."""
        engine_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=1000,
            relax_cell=False,
        )
        engine = LammpsEngine(
            EngineInput=engine_input,
            working_directory=self.temp_dir,
            max_evaluations=2000,
        )

        script = engine._build_script(self.structure)

        # Check minimization-specific content
        self.assertIn("min_style cg", script)
        self.assertIn("minimize 1e-06 1e-05 1000 2000", script)

    def test_build_script_minimize_with_cell_relaxation(self):
        """Test script building for minimization with cell relaxation."""
        engine_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=1000,
            relax_cell=True,
        )
        engine = LammpsEngine(
            EngineInput=engine_input,
            working_directory=self.temp_dir,
            max_evaluations=2000,
        )

        script = engine._build_script(self.structure)

        # Check cell relaxation fix
        self.assertIn("fix 1 all box/relax iso 10000.000000 vmax 0.001", script)

    def test_build_script_md_nve(self):
        """Test script building for NVE MD."""
        engine_input = CalcInputMD(
            mode="NVE", time_step=1.0, n_ionic_steps=1000, n_print=100  # 1 fs
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NVE-specific content
        self.assertIn("fix 1 all nve", script)
        self.assertIn("timestep 0.001", script)
        self.assertIn("run 1000", script)

    def test_build_script_md_nvt_nose_hoover(self):
        """Test script building for NVT MD with Nose-Hoover thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="nose-hoover",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NVT Nose-Hoover content
        self.assertIn(
            "velocity all create 300 12345 mom yes rot yes dist gaussian", script
        )
        self.assertIn("fix 1 all nvt temp 300 300 0.1", script)

    def test_build_script_md_nvt_berendsen(self):
        """Test script building for NVT MD with Berendsen thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="berendsen",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NVT Berendsen content: temp/berendsen only rescales
        # velocities, so the atoms need an integrator to move at all
        lines = script.splitlines()
        self.assertIn("fix 1 all temp/berendsen 300 300 0.1", lines)
        self.assertIn("fix 2 all nve", lines)

    def test_build_script_md_nvt_andersen(self):
        """Test script building for NVT MD with Andersen thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="andersen",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NVT Andersen content: implemented with fix langevin
        lines = script.splitlines()
        self.assertIn("fix 1 all langevin 300 300 0.1 12345", lines)
        self.assertIn("fix 2 all nve", lines)

    def test_build_script_md_nvt_temp_rescale(self):
        """Test script building for NVT MD with temp/rescale thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="temp/rescale",
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # LAMMPS syntax: 'fix ID group temp/rescale N Tstart Tstop window
        # fraction' (delta_temp field removed upstream; engine renders the
        # 0.0 window fallback), plus an integrator - temp/rescale only
        # rescales velocities.
        lines = script.splitlines()
        self.assertIn("fix 1 all temp/rescale 100 300 300 0.0 1.0", lines)
        self.assertIn("fix 2 all nve", lines)
        self.assertNotIn("units box", script)

    def test_build_script_md_nvt_temp_rescale_with_n_print_zero(self):
        # temp/rescale needs N > 0; 'thermo 0' is fine
        engine = LammpsEngine(
            EngineInput=CalcInputMD(
                mode="NVT",
                temperature=300,
                thermostat="temp/rescale",
                n_ionic_steps=10,
                n_print=0,
                seed=1,
            ),
            working_directory=self.temp_dir,
        )

        lines = engine._build_script(self.structure).splitlines()

        self.assertIn("fix 1 all temp/rescale 1 300 300 0.0 1.0", lines)
        self.assertIn("thermo 0", lines)

    def test_build_script_md_nvt_csvr(self):
        """Test script building for NVT MD with CSVR thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="temp/csvr",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NVT CSVR content. LAMMPS' syntax is 'fix ID group temp/csvr
        # Tstart Tstop Tdamp seed'; temp/csvr only rescales velocities, so
        # the atoms need an integrator too.
        lines = script.splitlines()
        self.assertIn("fix 1 all temp/csvr 300 300 0.1 12345", lines)
        self.assertIn("fix 2 all nve", lines)

    def test_build_script_md_nvt_every_thermostat_integrates(self):
        """Exactly one time integrator per NVT thermostat, so atoms move."""
        integrators = ("nve", "nvt", "npt", "nph")
        for thermostat in (
            "nose-hoover",
            "langevin",
            "andersen",
            "berendsen",
            "temp/rescale",
            "temp/csvr",
        ):
            with self.subTest(thermostat=thermostat):
                engine = LammpsEngine(
                    EngineInput=CalcInputMD(
                        mode="NVT", temperature=300, thermostat=thermostat, seed=7
                    ),
                    working_directory=self.temp_dir,
                )
                fixes = [
                    line.split()[3]
                    for line in engine._build_script(self.structure).splitlines()
                    if line.startswith("fix ")
                ]
                self.assertEqual(sum(style in integrators for style in fixes), 1, fixes)

    def test_build_script_md_nvt_langevin(self):
        """Test script building for NVT MD with Langevin thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="langevin",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NVT Langevin content
        self.assertIn("fix 1 all langevin 300 300 0.1 12345", script)
        self.assertIn("fix 2 all nve", script)

    def test_build_script_md_npt(self):
        """Test script building for NPT MD."""
        engine_input = CalcInputMD(
            mode="NPT",
            temperature=300,
            pressure=1e5,  # 1 bar
            thermostat="nose-hoover",
            thermostat_time_constant=100.0,
            pressure_damping_timescale=1000.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        script = engine._build_script(self.structure)

        # Check NPT content
        self.assertIn("fix 1 all npt temp 300 300 0.1 iso 1.0 1.0 1.0", script)

    def test_build_script_md_npt_wrong_thermostat(self):
        """Test script building for NPT MD with unsupported thermostat."""
        engine_input = CalcInputMD(
            mode="NPT",
            temperature=300,
            pressure=1e5,
            thermostat="berendsen",  # Not supported for NPT
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)

        self.assertIn(
            "NPT mode supports only 'nose-hoover' thermostat", str(context.exception)
        )

    def test_build_script_md_npt_requires_a_pressure(self):
        """NPT without a pressure raises a clear error, not a TypeError."""
        engine = LammpsEngine(
            EngineInput=CalcInputMD(mode="NPT", thermostat="nose-hoover"),
            working_directory=self.temp_dir,
        )
        self.assertIsNone(engine.EngineInput.pressure)

        with self.assertRaisesRegex(ValueError, "NPT mode requires.*pressure"):
            engine._build_script(self.structure)

    def test_build_script_md_unsupported_mode(self):
        """Test script building for unsupported MD mode."""
        engine_input = CalcInputMD(
            mode="NPH",  # Not supported
            temperature=300,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)

        self.assertIn("Unknown MD mode: NPH", str(context.exception))

    def test_build_script_unsupported_thermostat(self):
        """Test script building for unsupported thermostat."""
        engine_input = CalcInputMD(
            mode="NVT",
            temperature=300,
            thermostat="unsupported",
            thermostat_time_constant=100.0,
            time_step=1.0,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)

        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)

        self.assertIn("Unsupported thermostat: unsupported", str(context.exception))

    def test_build_script_unknown_mode(self):
        """Test script building for unknown mode."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        engine.mode = "unknown"  # Manually set invalid mode

        with (
            self.assertWarns(RuntimeWarning),
            self.assertRaises(ValueError) as context,
        ):
            engine._build_script(self.structure)

        self.assertIn("Unknown mode: unknown", str(context.exception))

    def test_write_input_file(self):
        """Test writing input file to disk."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        structure = bulk("Fe", "bcc", a=2.87)
        filepath = engine.write_input_file(structure)

        expected_path = os.path.join(self.temp_dir, "in.lmp")
        self.assertEqual(filepath, expected_path)
        self.assertTrue(os.path.exists(expected_path))

        # Check file content
        with open(filepath) as f:
            content = f.read()

        self.assertIn("units metal", content)
        self.assertEqual(content, engine._build_script(structure))
        self.assertTrue(content.endswith("\nrun 0"))

    def test_get_calculate_fn(self):
        """The default calc_fn is the LAMMPS runner, with kwargs for this structure."""
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(), working_directory=self.temp_dir
        )

        calc_fn, calc_kwargs = engine.get_calculate_fn(self.structure)

        self.assertIs(calc_fn, lammps_calculator_fn)
        self.assertEqual(
            calc_kwargs,
            {
                "working_directory": self.temp_dir,
                "lammps_input": engine._build_script(self.structure),
                "potential_elements": ["Fe"],
                "input_filename": "in.lmp",
                "command": "lmp -in in.lmp -log log.lammps",
                "lammps_log_filepath": "log.lammps",
                "units": "metal",
                "lammps_log_convergence_printout": "Total wall time:",
                "lammps_structure_filepath": "lammps.data",
                "dump_out_file_name": "dump.out",
            },
        )

    def test_get_calculate_fn_forwards_file_names(self):
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            input_script_read_data_file="struct.data",
            input_script_dump_filename="traj.dump",
        )

        _, calc_kwargs = engine.get_calculate_fn(self.structure)

        self.assertEqual(calc_kwargs["lammps_structure_filepath"], "struct.data")
        self.assertEqual(calc_kwargs["dump_out_file_name"], "traj.dump")
        script = calc_kwargs["lammps_input"]
        self.assertIn("read_data struct.data", script.splitlines())
        self.assertIn(
            "dump 1 all custom 10 traj.dump id type xsu ysu zsu fx fy fz vx vy vz "
            "c_eng",
            script.splitlines(),
        )

    def test_get_parse_fn_default(self):
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(), working_directory=self.temp_dir
        )

        parse_fn, parse_kwargs = engine.get_parse_fn()

        self.assertIs(parse_fn, parse_LammpsOutput)
        self.assertEqual(parse_kwargs, {})

    def test_get_parse_fn_custom(self):
        def custom_fn(**kwargs):
            return kwargs

        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            parse_fn=custom_fn,
            parse_fn_kwargs={"test": "value"},
        )

        parse_fn, parse_kwargs = engine.get_parse_fn()

        self.assertIs(parse_fn, custom_fn)
        self.assertEqual(parse_kwargs, {"test": "value"})

    def test_boundary_string(self):
        """Test boundary setting with string input."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        engine.input_script_boundary = "p p p"
        structure = bulk("Fe", "bcc", a=2.87)
        script = engine._build_script(structure)

        self.assertIn("boundary p p p", script)

    def test_boundary_tuple(self):
        """Test boundary setting with tuple input."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        engine.input_script_boundary = ("p", "p", "f")
        structure = bulk("Fe", "bcc", a=2.87)
        script = engine._build_script(structure)

        self.assertIn("boundary p p f", script)

    def test_build_script_minimize_max_iterations_override_warns(self):
        """Engine-level max_iterations must override EngineInput.max_iterations
        with a RuntimeWarning (engine.py:178-182).
        """
        import warnings

        engine_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=500,
            relax_cell=False,
        )
        engine = LammpsEngine(
            EngineInput=engine_input,
            working_directory=self.temp_dir,
            max_iterations=9999,
            max_evaluations=2000,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            script = engine._build_script(self.structure)

        # The override warning must have been raised.
        override_warnings = [
            w
            for w in caught
            if issubclass(w.category, RuntimeWarning)
            and "max_iterations" in str(w.message)
            and "takes precedence" in str(w.message)
        ]
        self.assertEqual(len(override_warnings), 1)
        # The script must use the engine-level override, not the EngineInput value.
        self.assertIn("minimize 1e-06 1e-05 9999 2000", script)


_NEXT_DUMP = re.compile(r'^variable pwl_next_dump equal "(?P<expression>.+)"$')
_DUMP_MODIFY_EVERY = "dump_modify 1 every v_pwl_next_dump first yes"


def _dump_steps(script: str, last_step: int) -> list[int]:
    """The steps LAMMPS dumps on, by evaluating the script's ``pwl_next_dump``.

    ``dump_modify every v_name first yes`` dumps on step 0, then on whatever
    step the variable returns, re-evaluating it on each dump step.
    """
    (match,) = [m for m in map(_NEXT_DUMP.match, script.splitlines()) if m is not None]
    expression = match["expression"]
    namespace = {"ternary": lambda condition, a, b: a if condition else b}

    def next_dump(step):
        return eval(expression, {**namespace, "step": step})  # noqa: S307

    steps = [0]
    while (step := next_dump(steps[-1])) <= last_step:
        assert step > steps[-1], "the next dump step must be in the future"
        steps.append(step)
    return steps


def _thermo_steps(every: int, last_step: int) -> list[int]:
    """LAMMPS prints thermo output every ``every`` steps and on the last step."""
    return sorted(set(range(0, last_step + 1, every)) | {last_step})


class TestDumpOnThermoSteps(unittest.TestCase):
    """MD and minimize dump on every thermo step, including the last one."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _script(self, engine_input, **kwargs):
        engine = LammpsEngine(
            EngineInput=engine_input, working_directory=self.temp_dir, **kwargs
        )
        return engine._build_script(self.structure)

    def test_md_lines(self):
        lines = self._script(
            CalcInputMD(mode="NVE", n_ionic_steps=53, n_print=7)
        ).splitlines()

        self.assertIn(
            'variable pwl_next_dump equal "ternary(step+7<53,step+7,'
            'ternary(step<53,53,step+7))"',
            lines,
        )
        self.assertIn(_DUMP_MODIFY_EVERY, lines)
        variable = next(i for i, ln in enumerate(lines) if "pwl_next_dump equal" in ln)
        dump = next(i for i, ln in enumerate(lines) if ln.startswith("dump 1 all"))
        self.assertLess(dump, variable)
        self.assertLess(variable, lines.index(_DUMP_MODIFY_EVERY))
        self.assertLess(lines.index("thermo 7"), lines.index(_DUMP_MODIFY_EVERY))
        self.assertLess(lines.index(_DUMP_MODIFY_EVERY), lines.index("run 53"))
        self.assertEqual(lines[-1], "run 53")

    def test_md_dumps_on_thermo_steps(self):
        for n_ionic_steps, n_print in [(53, 7), (40, 10), (50, 100), (1, 1), (9, 3)]:
            with self.subTest(n_ionic_steps=n_ionic_steps, n_print=n_print):
                script = self._script(
                    CalcInputMD(
                        mode="NVE", n_ionic_steps=n_ionic_steps, n_print=n_print
                    )
                )
                self.assertIn(f"thermo {n_print}", script.splitlines())
                self.assertEqual(
                    _dump_steps(script, n_ionic_steps),
                    _thermo_steps(n_print, n_ionic_steps),
                )
        self.assertEqual(
            _dump_steps(
                self._script(CalcInputMD(mode="NVE", n_ionic_steps=53, n_print=7)),
                53,
            ),
            [0, 7, 14, 21, 28, 35, 42, 49, 53],
        )

    def test_minimize_lines(self):
        lines = self._script(
            CalcInputMinimize(max_iterations=3), max_evaluations=100
        ).splitlines()

        self.assertIn(
            'variable pwl_next_dump equal "ternary(step+10<3,step+10,'
            'ternary(step<3,3,step+10))"',
            lines,
        )
        self.assertIn(_DUMP_MODIFY_EVERY, lines)
        minimize = next(i for i, ln in enumerate(lines) if ln.startswith("minimize"))
        self.assertLess(lines.index(_DUMP_MODIFY_EVERY), minimize)
        self.assertEqual(lines[-1], lines[minimize])

    def test_minimize_dumps_on_thermo_every_and_on_max_iterations(self):
        cases = [
            ({"max_iterations": 3}, {}, 3, [0, 3]),
            ({"max_iterations": 25}, {}, 25, [0, 10, 20, 25]),
            ({"max_iterations": 30}, {}, 30, [0, 10, 20, 30]),
            (
                {"max_iterations": 12},
                {"input_script_thermo_every": 5},
                12,
                [0, 5, 10, 12],
            ),
            # the dump follows the thermo interval, not input_script_dump_every,
            # so frames and energies stay aligned when the two differ
            (
                {"max_iterations": 12},
                {"input_script_thermo_every": 5, "input_script_dump_every": 3},
                12,
                [0, 5, 10, 12],
            ),
            (
                {"max_iterations": 12},
                {"input_script_dump_every": 5},
                12,
                [0, 10, 12],
            ),
        ]
        for engine_input, engine_kwargs, last, expected in cases:
            with self.subTest(**engine_input, **engine_kwargs):
                script = self._script(
                    CalcInputMinimize(**engine_input), **engine_kwargs
                )
                self.assertEqual(_dump_steps(script, last), expected)

    def test_minimize_uses_engine_max_iterations_override(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            script = self._script(
                CalcInputMinimize(max_iterations=500), max_iterations=7
            )

        self.assertIn("ternary(step<7,7,step+10)", script)
        self.assertEqual(_dump_steps(script, 7), [0, 7])

    def test_minimize_with_cell_relaxation_keeps_the_dump_lines(self):
        lines = self._script(
            CalcInputMinimize(max_iterations=40, relax_cell=True)
        ).splitlines()

        relax = next(i for i, ln in enumerate(lines) if "box/relax" in ln)
        self.assertLess(relax, lines.index(_DUMP_MODIFY_EVERY))
        self.assertEqual(_dump_steps("\n".join(lines), 40), [0, 10, 20, 30, 40])

    def test_md_n_print_zero_dumps_first_and_last_step(self):
        """'thermo 0' prints the first and last step; the dump matches it.

        A zero interval used to make the dump variable return the current
        step ("Dump 1 every variable pwl_next_dump returned a bad timestep").
        """
        for n_ionic_steps, expected in [(20, [0, 20]), (1, [0, 1]), (0, [0])]:
            with self.subTest(n_ionic_steps=n_ionic_steps):
                script = self._script(
                    CalcInputMD(mode="NVE", n_ionic_steps=n_ionic_steps, n_print=0)
                )
                self.assertIn("thermo 0", script.splitlines())
                self.assertEqual(_dump_steps(script, n_ionic_steps), expected)

    def test_minimize_thermo_every_zero_dumps_first_and_last_step(self):
        script = self._script(
            CalcInputMinimize(max_iterations=12), input_script_thermo_every=0
        )

        self.assertEqual(_dump_steps(script, 12), [0, 12])


_VELOCITY = re.compile(
    r"^velocity all create (?P<temperature>\S+) (?P<seed>\S+) "
    r"mom yes rot yes dist gaussian$"
)


def _velocity_lines(script):
    return [m for m in map(_VELOCITY.match, script.splitlines()) if m is not None]


class TestMDSeedAndInitialTemperature(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _script(self, **md):
        md.setdefault("mode", "NVT")
        md.setdefault("thermostat", "langevin")
        md.setdefault("temperature", 300)
        md.setdefault("n_ionic_steps", 10)
        md.setdefault("n_print", 5)
        engine = LammpsEngine(
            EngineInput=CalcInputMD(**md), working_directory=self.temp_dir
        )
        return engine._build_script(self.structure)

    def test_seed_none_draws_a_positive_integer_seed(self):
        self.assertIsNone(CalcInputMD().seed)
        script = self._script(seed=None)

        self.assertNotIn("None", script)
        (velocity,) = _velocity_lines(script)
        seed = velocity["seed"]
        self.assertRegex(seed, r"^[1-9][0-9]*$")
        self.assertLessEqual(int(seed), 900_000_000)
        # the thermostat uses the same seed
        self.assertIn(f"fix 1 all langevin 300 300 0.1 {seed}", script.splitlines())

    # fix langevin (thermostats "langevin" and "andersen") and temp/csvr use
    # LAMMPS' Marsaglia generator, which rejects seeds above 900000000
    # ("Invalid seed for Marsaglia random # generator"); velocity create's
    # Park-Miller generator hangs on 2**31 - 1.
    def test_seed_none_draws_a_seed_every_lammps_rng_accepts(self):
        # Seeds used to be drawn from [1, 2**31 - 1], so ~58% of seed=None
        # langevin/andersen runs failed. 2000 draws from that old range all
        # being valid has probability ~0.42**2000.
        for thermostat in ("langevin", "andersen", "temp/csvr"):
            with self.subTest(thermostat=thermostat):
                seeds = []
                for _ in range(2000):
                    script = self._script(thermostat=thermostat)
                    (velocity,) = _velocity_lines(script)
                    seeds.append(int(velocity["seed"]))
                    # the thermostat draws from the same seed
                    (thermostat_fix,) = [
                        line
                        for line in script.splitlines()
                        if line.startswith("fix 1 all ")
                    ]
                    self.assertEqual(thermostat_fix.split()[-1], velocity["seed"])
                self.assertGreaterEqual(min(seeds), 1)
                self.assertLessEqual(max(seeds), 900_000_000)

    def test_seed_none_range_bounds(self):
        """The draw is from exactly [1, 900000000], both bounds included."""
        for bound in (1, 900_000_000):
            with (
                self.subTest(bound=bound),
                patch(
                    "pyiron_workflow_lammps.engine.random.SystemRandom"
                ) as system_random,
            ):
                system_random.return_value.randint.return_value = bound
                script = self._script()

                system_random.return_value.randint.assert_called_once_with(
                    1, 900_000_000
                )
                (velocity,) = _velocity_lines(script)
                self.assertEqual(velocity["seed"], str(bound))
                self.assertIn(
                    f"fix 1 all langevin 300 300 0.1 {bound}", script.splitlines()
                )

    def test_explicit_seed_draws_nothing(self):
        with patch("pyiron_workflow_lammps.engine.random.SystemRandom") as rng:
            self._script(seed=5)
        rng.assert_not_called()

    def test_seed_none_is_not_deterministic(self):
        seeds = {_velocity_lines(self._script())[0]["seed"] for _ in range(5)}

        # five draws from [1, 900000000] all equal: probability ~2**-118
        self.assertGreater(len(seeds), 1)

    def test_explicit_seed_is_reproducible(self):
        first = self._script(seed=4321)
        second = self._script(seed=4321)

        self.assertEqual(first, second)
        (velocity,) = _velocity_lines(first)
        self.assertEqual(velocity["seed"], "4321")
        self.assertIn("fix 1 all langevin 300 300 0.1 4321", first.splitlines())

    def test_initial_temperature_zero_creates_no_velocities(self):
        for md in (
            {"mode": "NVE", "initial_temperature": 0},
            {"initial_temperature": 0.0},
            {"initial_temperature": 0, "seed": None},
        ):
            with self.subTest(**md):
                script = self._script(**md)
                self.assertNotIn("velocity", script)
                self.assertNotIn("None", script)

    def test_initial_temperature_is_used_when_given(self):
        (velocity,) = _velocity_lines(self._script(initial_temperature=500, seed=1))
        self.assertEqual(velocity["temperature"], "500")

    def test_initial_temperature_none_falls_back_to_temperature(self):
        (velocity,) = _velocity_lines(self._script(initial_temperature=None, seed=1))
        self.assertEqual(velocity["temperature"], "300")


def _fe_al():
    """Two-species B2-like cell; Fe occurs first."""
    structure = bulk("Fe", "bcc", a=2.87, cubic=True)
    structure[1].symbol = "Al"
    return structure


def _al_fe():
    """Same cell with the species order reversed; Al occurs first."""
    structure = bulk("Fe", "bcc", a=2.87, cubic=True)
    structure[0].symbol = "Al"
    return structure


def _pair_coeff(script):
    (line,) = [ln for ln in script.splitlines() if ln.startswith("pair_coeff")]
    return line


class TestLammpsEngineIsStateless(unittest.TestCase):
    """get_calculate_fn builds everything per structure and never mutates the engine."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            path_to_model="/models/Al-Fe.eam.fs",
            input_script_pair_style="eam/fs",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _assert_pristine(self):
        self.assertIsNone(self.engine.calc_fn)
        self.assertEqual(self.engine.calc_fn_kwargs, {})
        self.assertIsNone(self.engine.potential_elements)
        self.assertIsNone(self.engine.parse_fn)
        self.assertEqual(self.engine.parse_fn_kwargs, {})
        self.assertEqual(self.engine.mode, "static")

    def test_engine_unchanged_after_get_calculate_fn(self):
        before = copy.deepcopy(self.engine)

        self.engine.get_calculate_fn(_fe_al())

        self._assert_pristine()
        self.assertEqual(self.engine, before)

    def test_no_caching_between_structures(self):
        fn_a, kwargs_a = self.engine.get_calculate_fn(bulk("Fe", "bcc", a=2.87))
        fn_b, kwargs_b = self.engine.get_calculate_fn(_al_fe())
        fn_c, kwargs_c = self.engine.get_calculate_fn(_fe_al())

        self.assertIs(fn_a, fn_b)
        self.assertEqual(kwargs_a["potential_elements"], ["Fe"])
        self.assertEqual(kwargs_b["potential_elements"], ["Al", "Fe"])
        self.assertEqual(kwargs_c["potential_elements"], ["Fe", "Al"])
        self.assertEqual(
            _pair_coeff(kwargs_a["lammps_input"]),
            "pair_coeff * * /models/Al-Fe.eam.fs Fe",
        )
        self.assertEqual(
            _pair_coeff(kwargs_b["lammps_input"]),
            "pair_coeff * * /models/Al-Fe.eam.fs Al Fe",
        )
        self.assertEqual(
            _pair_coeff(kwargs_c["lammps_input"]),
            "pair_coeff * * /models/Al-Fe.eam.fs Fe Al",
        )
        self._assert_pristine()

    def test_returned_kwargs_are_independent(self):
        _, kwargs_a = self.engine.get_calculate_fn(_fe_al())
        kwargs_a["potential_elements"].append("C")
        kwargs_a["working_directory"] = "elsewhere"

        _, kwargs_b = self.engine.get_calculate_fn(_fe_al())

        self.assertEqual(kwargs_b["potential_elements"], ["Fe", "Al"])
        self.assertEqual(kwargs_b["working_directory"], self.temp_dir)

    def test_user_potential_elements_are_honoured(self):
        user_elements = ["Al", "Fe"]
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            potential_elements=user_elements,
        )

        for structure in (bulk("Fe", "bcc", a=2.87), _fe_al()):
            _, kwargs = engine.get_calculate_fn(structure)
            self.assertEqual(kwargs["potential_elements"], ["Al", "Fe"])
            self.assertTrue(_pair_coeff(kwargs["lammps_input"]).endswith(" Al Fe"))
            # A copy is handed out, so the caller cannot alias engine state.
            self.assertIsNot(kwargs["potential_elements"], user_elements)

        self.assertEqual(engine.potential_elements, ["Al", "Fe"])

    def test_with_working_directory_builds_against_new_directory(self):
        self.engine.get_calculate_fn(_fe_al())

        child = self.engine.with_working_directory("sub")
        _, kwargs = child.get_calculate_fn(_fe_al())

        self.assertEqual(
            kwargs["working_directory"], os.path.join(self.temp_dir, "sub")
        )
        self.assertEqual(self.engine.working_directory, self.temp_dir)

    def test_engine_is_pickleable_after_use(self):
        self.engine.get_calculate_fn(_fe_al())

        roundtrip = pickle.loads(pickle.dumps(self.engine))

        self.assertEqual(roundtrip, self.engine)


class TestLammpsEngineCustomFunctions(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_custom_calc_fn_gets_a_copy_of_its_kwargs(self):
        def custom_fn(structure, **kwargs):
            return structure

        custom_kwargs = {"test": "value"}
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            calc_fn=custom_fn,
            calc_fn_kwargs=custom_kwargs,
        )

        calc_fn, calc_kwargs = engine.get_calculate_fn(self.structure)

        self.assertIs(calc_fn, custom_fn)
        self.assertEqual(calc_kwargs, {"test": "value"})
        self.assertIsNot(calc_kwargs, engine.calc_fn_kwargs)
        calc_kwargs["test"] = "changed"
        self.assertEqual(engine.calc_fn_kwargs, {"test": "value"})
        self.assertIs(engine.calc_fn, custom_fn)

    def test_custom_parse_fn_is_wired_into_calc_kwargs(self):
        def custom_parser(**kwargs):
            return kwargs

        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            parse_fn=custom_parser,
            parse_fn_kwargs={"answer": 42},
        )

        calc_fn, calc_kwargs = engine.get_calculate_fn(self.structure)

        self.assertIs(calc_fn, lammps_calculator_fn)
        self.assertIs(calc_kwargs["_lammps_parser_function"], custom_parser)
        self.assertEqual(calc_kwargs["_lammps_parser_args"], {"answer": 42})
        self.assertIsNot(calc_kwargs["_lammps_parser_args"], engine.parse_fn_kwargs)
        # Every key must be a parameter of the calc_fn it is handed to.
        self.assertLessEqual(
            set(calc_kwargs), set(inspect.signature(calc_fn).parameters)
        )

    def test_default_calc_kwargs_match_calc_fn_signature(self):
        calc_fn, calc_kwargs = LammpsEngine(
            EngineInput=CalcInputStatic(), working_directory=self.temp_dir
        ).get_calculate_fn(self.structure)

        self.assertNotIn("_lammps_parser_function", calc_kwargs)
        self.assertNotIn("_lammps_parser_args", calc_kwargs)
        self.assertNotIn("structure", calc_kwargs)
        self.assertLessEqual(
            set(calc_kwargs) | {"structure"},
            set(inspect.signature(calc_fn).parameters),
        )

    def test_with_working_directory_keeps_custom_functions(self):
        def custom_calc(structure, **kwargs):
            return kwargs

        def custom_parser(**kwargs):
            return kwargs

        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            calc_fn=custom_calc,
            calc_fn_kwargs={"a": 1},
            parse_fn=custom_parser,
            parse_fn_kwargs={"b": 2},
        )

        child = engine.with_working_directory("sub")

        self.assertIsNot(child, engine)
        self.assertEqual(child.working_directory, os.path.join(self.temp_dir, "sub"))
        self.assertEqual(engine.working_directory, self.temp_dir)
        self.assertIs(child.calc_fn, custom_calc)
        self.assertEqual(child.calc_fn_kwargs, {"a": 1})
        self.assertIs(child.parse_fn, custom_parser)
        self.assertEqual(child.parse_fn_kwargs, {"b": 2})
        calc_fn, calc_kwargs = child.get_calculate_fn(self.structure)
        self.assertIs(calc_fn, custom_calc)
        self.assertEqual(calc_kwargs, {"a": 1})


class TestToggleModeWarnings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_no_warning_when_mode_agrees(self):
        inputs = (
            CalcInputStatic(),
            CalcInputMinimize(),
            CalcInputMD(mode="NVE", n_ionic_steps=10, n_print=5),
        )
        for engine_input in inputs:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                engine = LammpsEngine(
                    EngineInput=engine_input, working_directory=self.temp_dir
                )
                engine.toggle_mode()
                engine.get_calculate_fn(bulk("Fe", "bcc", a=2.87))
                engine.with_working_directory("sub")
            self.assertEqual(
                [str(w.message) for w in caught if w.category is RuntimeWarning],
                [],
                msg=type(engine_input).__name__,
            )

    def test_warns_and_keeps_mode_when_it_disagrees(self):
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(), working_directory=self.temp_dir
        )
        engine.mode = "md"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mode = engine.toggle_mode()

        self.assertEqual(mode, "md")
        self.assertEqual(engine.mode, "md")
        self.assertEqual(
            len([w for w in caught if issubclass(w.category, RuntimeWarning)]), 1
        )


class TestCalphyCompatibility(unittest.TestCase):
    """atomistics' calphy adapter rejects engines with non-default fields.

    get_calculate_fn must therefore not write anything back onto the engine.
    """

    def setUp(self):
        try:
            from pyiron_workflow_atomistics.physics.free_energy._calphy_adapter import (
                _validate_engine_only_command,
            )
        except ImportError as e:  # pragma: no cover - depends on atomistics version
            self.skipTest(f"calphy adapter not importable: {e}")
        self.validate = _validate_engine_only_command
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_minimal_engine_still_valid_after_get_calculate_fn(self):
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            command="mpirun -np 2 lmp -in in.lmp -log log.lammps",
            working_directory=self.temp_dir,
        )
        self.validate(engine)

        engine.get_calculate_fn(_fe_al())
        engine.get_calculate_fn(bulk("Fe", "bcc", a=2.87))
        self.validate(engine.with_working_directory("calphy"))

        self.validate(engine)

    def test_validation_still_rejects_overrides(self):
        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            potential_elements=["Fe"],
        )

        with self.assertRaises(ValueError):
            self.validate(engine)


if __name__ == "__main__":
    unittest.main()
