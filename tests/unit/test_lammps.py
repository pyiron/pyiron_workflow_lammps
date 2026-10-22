import inspect
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk
from pyiron_workflow_atomistics.engine import EngineOutput

from pyiron_workflow_lammps.generic import ShellOutput
from pyiron_workflow_lammps.lammps import (
    CONVERGED_STOPPING_CRITERIA,
    DEFAULT_COMMAND,
    _cells_in_reference_basis,
    _minimization_converged,
    _pressures_in_reference_frame,
    arrays_to_ase_atoms,
    get_species_map,
    get_structure_species_lists,
    lammps_calculator_fn,
    lammps_job,
    parse_LammpsOutput,
    write_LammpsInput,
    write_LammpsStructure,
)

try:
    import pyiron_workflow as pwf
except ImportError:  # pragma: no cover - pyiron_workflow is optional here
    pwf = None

RESOURCES = Path(__file__).resolve().parents[1] / "resources"
# PotEng (== TotEng, a 0-step minimisation) of the canned run in RESOURCES
RESOURCE_ENERGY = -454.367609882893
# atom types in RESOURCES/lammps.data: 1 -> C, 2 -> Fe
RESOURCE_ELEMENTS = ["C", "Fe"]
# The canned run is a 0-step 'minimize 0 0 0 0', which LAMMPS reports as
# stopped on max iterations, i.e. *not* converged.
RESOURCE_CRITERION = "max iterations"


def _canned_run(
    directory,
    stopping_criterion=RESOURCE_CRITERION,
    wall_time=True,
    log_name="log.lammps",
    data_name="lammps.data",
    dump_name="dump.out",
):
    """Copy the canned run in RESOURCES into ``directory``, editing its log.

    The log's ``Stopping criterion`` line is replaced by ``stopping_criterion``
    (removed entirely when it is None, as in a static ``run 0`` or MD log),
    and the ``Total wall time`` line is dropped unless ``wall_time``. The
    structure and dump are copied unchanged. Returns ``directory``.
    """
    os.makedirs(directory, exist_ok=True)
    lines = []
    with open(RESOURCES / "log.lammps") as f:
        for line in f:
            if "Stopping criterion =" in line:
                if stopping_criterion is not None:
                    lines.append(f"  Stopping criterion = {stopping_criterion}\n")
            elif wall_time or not line.startswith("Total wall time"):
                lines.append(line)
    with open(os.path.join(directory, log_name), "w") as f:
        f.writelines(lines)
    shutil.copy(RESOURCES / "lammps.data", os.path.join(directory, data_name))
    shutil.copy(RESOURCES / "dump.out", os.path.join(directory, dump_name))
    return str(directory)


def _fake_lammps_command(
    source, log_name="log.lammps", data_name="lammps.data", dump_name="dump.out"
):
    """A ``command`` that "runs LAMMPS" by copying a canned run from ``source``.

    ``source`` is a directory made by :func:`_canned_run`. The command runs
    after the structure and input have been written, so it also replaces the
    written structure file with the one the canned dump matches.
    """
    return " && ".join(
        f"cp {shlex.quote(os.path.join(source, name))} {shlex.quote(name)}"
        for name in (log_name, data_name, dump_name)
    )


def _write_log(directory, *criteria, wall_time=True):
    """A minimal LAMMPS log with one minimization per stopping criterion."""
    path = os.path.join(directory, "log.lammps")
    with open(path, "w") as f:
        f.write("LAMMPS (29 Aug 2024)\n")
        for criterion in criteria:
            f.write("minimize 0 1e-4 100 1000\nMinimization stats:\n")
            f.write(f"  Stopping criterion = {criterion}\n")
            f.write("  Iterations, force evaluations = 3 6\n")
        if wall_time:
            f.write("Total wall time: 0:00:01\n")
    return path


def _shell_output(return_code, stdout="", stderr=""):
    output = ShellOutput()
    output.stdout = stdout
    output.stderr = stderr
    output.return_code = return_code
    return output


def _fake_lammpsparser(**attributes):
    module = types.ModuleType("lammpsparser")
    module.write_lammps_structure = MagicMock()
    module.parse_lammps_output_files = MagicMock()
    for name, value in attributes.items():
        setattr(module, name, value)
    return module


class TestWriteLammpsStructure(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87, cubic=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_writes_data_file(self):
        result = write_LammpsStructure(
            structure=self.structure,
            working_directory=self.temp_dir,
            potential_elements=["Fe"],
            units="metal",
            file_name="test.data",
        )

        self.assertEqual(result, self.temp_dir)
        data_file = os.path.join(self.temp_dir, "test.data")
        self.assertTrue(os.path.isfile(data_file))
        with open(data_file) as f:
            content = f.read()
        self.assertRegex(content, r"\b2 atoms\b")
        self.assertEqual(get_species_map(data_file), {1: "Fe"})

    def test_delegates_to_lammpsparser(self):
        fake = _fake_lammpsparser()
        with patch.dict(sys.modules, {"lammpsparser": fake}):
            result = write_LammpsStructure(
                structure=self.structure,
                working_directory=self.temp_dir,
                potential_elements=["Fe"],
                units="metal",
                file_name="test.data",
            )

        self.assertEqual(result, self.temp_dir)
        fake.write_lammps_structure.assert_called_once()
        kwargs = fake.write_lammps_structure.call_args.kwargs
        self.assertIs(kwargs["structure"], self.structure)
        self.assertEqual(kwargs["potential_elements"], ["Fe"])
        self.assertEqual(kwargs["units"], "metal")
        self.assertEqual(kwargs["file_name"], "test.data")
        self.assertEqual(kwargs["working_directory"], self.temp_dir)


class TestWriteLammpsInput(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lammps_input = "units metal\ndimension 3\nboundary p p p"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_with_directory(self):
        result = write_LammpsInput(
            lammps_input=self.lammps_input,
            filename="test.in",
            working_directory=self.temp_dir,
        )

        expected_path = os.path.join(self.temp_dir, "test.in")
        self.assertEqual(result, expected_path)
        with open(expected_path) as f:
            self.assertEqual(f.read(), self.lammps_input)

    def test_without_directory_writes_relative_to_cwd(self):
        cwd = os.getcwd()
        os.chdir(self.temp_dir)
        try:
            result = write_LammpsInput(
                lammps_input=self.lammps_input, filename="test_current.in"
            )
            self.assertEqual(result, "test_current.in")
            with open(os.path.join(self.temp_dir, "test_current.in")) as f:
                self.assertEqual(f.read(), self.lammps_input)
        finally:
            os.chdir(cwd)

    def test_creates_directory(self):
        new_dir = os.path.join(self.temp_dir, "new", "subdir")

        result = write_LammpsInput(
            lammps_input=self.lammps_input,
            filename="test.in",
            working_directory=new_dir,
        )

        self.assertEqual(result, os.path.join(new_dir, "test.in"))
        self.assertTrue(os.path.isfile(result))


class TestGetSpeciesMap(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write(self, content):
        data_file = os.path.join(self.temp_dir, "test.data")
        with open(data_file, "w") as f:
            f.write(content)
        return data_file

    def test_with_comments(self):
        data_file = self._write(
            "LAMMPS data file\n\n4 atoms\n2 atom types\n\nMasses\n\n"
            "1 55.847 # (Fe)\n2 12.011 # (C)\n\nAtoms\n\n1 1 0.0 0.0 0.0\n"
        )

        self.assertEqual(get_species_map(data_file), {1: "Fe", 2: "C"})

    def test_without_comments(self):
        data_file = self._write(
            "LAMMPS data file\n\n2 atoms\n2 atom types\n\nMasses\n\n"
            "1 55.847\n2 12.011\n\nAtoms\n\n1 1 0.0 0.0 0.0\n"
        )

        self.assertEqual(get_species_map(data_file), {1: None, 2: None})

    def test_resource_file(self):
        self.assertEqual(
            get_species_map(str(RESOURCES / "lammps.data")), {1: "C", 2: "Fe"}
        )

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_species_map(os.path.join(self.temp_dir, "nonexistent.data"))


class TestArraysToAseAtoms(unittest.TestCase):
    def test_basic(self):
        cells = np.eye(3) * 2.0
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        atoms = arrays_to_ase_atoms(
            cells=cells,
            positions=positions,
            indices=np.array([0, 0]),
            species_lists=[["Fe", "Fe"]],
        )

        self.assertEqual(atoms.get_chemical_symbols(), ["Fe", "Fe"])
        np.testing.assert_array_almost_equal(atoms.get_positions(), positions)
        np.testing.assert_array_almost_equal(atoms.get_cell(), cells)
        self.assertTrue(atoms.get_pbc().all())

    def test_uses_last_species_list(self):
        atoms = arrays_to_ase_atoms(
            cells=np.eye(3) * 3.0,
            positions=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]]),
            indices=np.array([0, 1]),
            species_lists=[["Al", "Al"], ["Fe", "C"]],
        )

        self.assertEqual(atoms.get_chemical_symbols(), ["Fe", "C"])

    def test_no_pbc(self):
        atoms = arrays_to_ase_atoms(
            cells=np.eye(3) * 2.0,
            positions=np.array([[0.0, 0.0, 0.0]]),
            indices=np.array([0]),
            species_lists=[["Fe"]],
            pbc=False,
        )

        self.assertFalse(atoms.get_pbc().any())


class TestGetStructureSpeciesLists(unittest.TestCase):
    def test_resource_files(self):
        result = get_structure_species_lists(
            lammps_data_filepath=str(RESOURCES / "lammps.data"),
            lammps_dump_filepath=str(RESOURCES / "dump.out"),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ["C", "Fe", "C"] + ["Fe"] * 61)


class TestParseLammpsOutput(unittest.TestCase):
    """Parse the canned output in tests/resources with the real lammpsparser."""

    def _parse(self, **kwargs):
        return parse_LammpsOutput(
            working_directory=str(RESOURCES),
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
            **kwargs,
        )

    def _assert_canned_output(self, result):
        self.assertIsInstance(result, EngineOutput)
        self.assertIsInstance(result.final_energy, float)
        self.assertAlmostEqual(result.final_energy, RESOURCE_ENERGY, places=9)
        self.assertIsInstance(result.n_ionic_steps, int)
        self.assertEqual(result.n_ionic_steps, 0)
        self.assertEqual(np.shape(result.final_forces), (64, 3))
        self.assertEqual(np.shape(result.final_stress), (3, 3))
        self.assertEqual(len(result.final_structure), 64)
        self.assertEqual(result.final_structure.get_chemical_symbols().count("C"), 2)
        self.assertAlmostEqual(result.final_volume, 756.476896, places=3)

    def test_named_files(self):
        result = self._parse(
            lammps_structure_filepath="lammps.data",
            dump_out_file_name="dump.out",
            log_lammps_file_name="minimize.log",
            log_lammps_convergence_printout="Total wall time:",
        )

        self._assert_canned_output(result)
        # 'minimize 0 0 0 0' stops on max iterations: finished, not converged
        self.assertIs(result.converged, False)

    def test_default_filenames(self):
        result = self._parse()

        self._assert_canned_output(result)
        self.assertIs(result.converged, False)

    def test_unconverged(self):
        result = self._parse(log_lammps_file_name="unconverged_minimize.log")

        self._assert_canned_output(result)
        self.assertIs(result.converged, False)

    def test_non_default_file_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            _canned_run(
                tmp,
                stopping_criterion="energy tolerance",
                log_name="run.log",
                data_name="struct.data",
                dump_name="traj.dump",
            )

            result = parse_LammpsOutput(
                working_directory=tmp,
                potential_elements=RESOURCE_ELEMENTS,
                units="metal",
                lammps_structure_filepath="struct.data",
                dump_out_file_name="traj.dump",
                log_lammps_file_name="run.log",
            )

        self._assert_canned_output(result)
        self.assertIs(result.converged, True)

    def test_after_token_is_ignored(self):
        result = self._parse(after=object())

        self._assert_canned_output(result)

    def test_successful_shell_output_is_parsed(self):
        result = self._parse(after=_shell_output(0, stdout="ok", stderr="noise"))

        self._assert_canned_output(result)

    def test_failed_shell_output_raises_with_its_output(self):
        # Valid output files are present (as if left by an earlier run), but
        # the command that should have produced them failed.
        after = _shell_output(
            127, stdout="launching\n", stderr="sh: 1: lmp_nope: not found\n"
        )

        with self.assertRaises(RuntimeError) as context:
            self._parse(after=after)

        message = str(context.exception)
        self.assertIn("return code 127", message)
        self.assertIn(str(RESOURCES), message)
        self.assertIn("lmp_nope: not found", message)
        self.assertIn("launching", message)

    def test_failed_shell_output_keeps_the_tail_of_long_output(self):
        stderr = "x" * 5000 + "THE ACTUAL ERROR"
        after = _shell_output(1, stdout="s" * 5000, stderr=stderr)

        with self.assertRaises(RuntimeError) as context:
            self._parse(after=after)

        message = str(context.exception)
        self.assertIn("THE ACTUAL ERROR", message)
        self.assertLess(len(message), 5000)

    def test_failed_shell_output_raises_before_a_custom_parser(self):
        custom_parser = MagicMock()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            self._parse(after=_shell_output(2, stderr="boom"), _parser_fn=custom_parser)
        custom_parser.assert_not_called()

    def test_missing_files(self):
        with (
            tempfile.TemporaryDirectory() as empty,
            self.assertRaisesRegex(Exception, "Error parsing LAMMPS output"),
        ):
            parse_LammpsOutput(
                working_directory=empty,
                potential_elements=RESOURCE_ELEMENTS,
                units="metal",
            )

    def test_custom_parser(self):
        sentinel = object()
        custom_parser = MagicMock(return_value=sentinel)

        result = parse_LammpsOutput(
            working_directory="/does/not/matter",
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
            _parser_fn=custom_parser,
            _parser_fn_kwargs={"arg1": "value1"},
        )

        custom_parser.assert_called_once_with(arg1="value1")
        self.assertIs(result, sentinel)

    def test_custom_parser_without_kwargs(self):
        custom_parser = MagicMock(return_value="parsed")

        result = parse_LammpsOutput(
            working_directory="/does/not/matter",
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
            _parser_fn=custom_parser,
        )

        custom_parser.assert_called_once_with()
        self.assertEqual(result, "parsed")

    def test_custom_parser_failure(self):
        def failing_parser():
            raise ValueError("Parser failed")

        with self.assertRaisesRegex(Exception, "Error parsing LAMMPS output"):
            parse_LammpsOutput(
                working_directory="/does/not/matter",
                potential_elements=RESOURCE_ELEMENTS,
                units="metal",
                _parser_fn=failing_parser,
                _parser_fn_kwargs={},
            )


class TestMinimizationConverged(unittest.TestCase):
    """``converged`` follows the log's last minimizer stopping criterion."""

    NOT_CONVERGED = (
        "max iterations",
        "max force evaluations",
        "linesearch alpha is zero",
        "quadratic factors are zero",
        "trust region too small",
        "HFTN minimizer error",
        "unknown",
    )

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_converged_criteria(self):
        self.assertEqual(
            CONVERGED_STOPPING_CRITERIA,
            ("energy tolerance", "force tolerance", "forces are zero"),
        )
        for criterion in CONVERGED_STOPPING_CRITERIA:
            with self.subTest(criterion=criterion):
                self.assertIs(
                    _minimization_converged(_write_log(self.temp_dir, criterion)),
                    True,
                )

    def test_unconverged_criteria(self):
        for criterion in self.NOT_CONVERGED:
            with self.subTest(criterion=criterion):
                self.assertIs(
                    _minimization_converged(_write_log(self.temp_dir, criterion)),
                    False,
                )

    def test_log_without_minimization_counts_as_converged(self):
        # static 'run 0' and MD logs have no stopping criterion
        self.assertIs(_minimization_converged(_write_log(self.temp_dir)), True)
        empty = os.path.join(self.temp_dir, "empty.log")
        open(empty, "w").close()
        self.assertIs(_minimization_converged(empty), True)

    def test_last_minimization_decides(self):
        log = _write_log(self.temp_dir, "max iterations", "energy tolerance")
        self.assertIs(_minimization_converged(log), True)

        log = _write_log(self.temp_dir, "force tolerance", "max iterations")
        self.assertIs(_minimization_converged(log), False)

    def test_missing_log_raises(self):
        with self.assertRaises(FileNotFoundError):
            _minimization_converged(os.path.join(self.temp_dir, "missing.log"))

    def _parse(self, **canned):
        directory = _canned_run(os.path.join(self.temp_dir, "run"), **canned)
        return parse_LammpsOutput(
            working_directory=directory,
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
        )

    def test_parse_converged_needs_wall_time_and_a_converged_criterion(self):
        cases = [
            ({"stopping_criterion": "energy tolerance"}, True),
            ({"stopping_criterion": "force tolerance"}, True),
            ({"stopping_criterion": "forces are zero"}, True),
            ({"stopping_criterion": None}, True),  # e.g. a static 'run 0'
            ({"stopping_criterion": "max iterations"}, False),
            ({"stopping_criterion": "max force evaluations"}, False),
            ({"stopping_criterion": "linesearch alpha is zero"}, False),
            ({"stopping_criterion": "energy tolerance", "wall_time": False}, False),
            ({"stopping_criterion": None, "wall_time": False}, False),
        ]
        for canned, expected in cases:
            with self.subTest(**canned):
                result = self._parse(**canned)
                self.assertIs(result.converged, expected)
                self.assertAlmostEqual(result.final_energy, RESOURCE_ENERGY, places=9)


class TestParseLeavesWarningFiltersAlone(unittest.TestCase):
    """parse_LammpsOutput must not silence warnings for the whole process."""

    def test_filters_unchanged_by_parse(self):
        before = list(warnings.filters)

        parse_LammpsOutput(
            working_directory=str(RESOURCES),
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
        )

        self.assertEqual(warnings.filters, before)
        self.assertNotIn(("ignore", None, Warning, None, 0), warnings.filters)

    def test_engine_warning_still_visible_after_parse(self):
        from pyiron_workflow_atomistics.engine import CalcInputMinimize

        from pyiron_workflow_lammps.engine import LammpsEngine

        parse_LammpsOutput(
            working_directory=str(RESOURCES),
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
        )
        engine = LammpsEngine(
            EngineInput=CalcInputMinimize(), max_iterations=5, working_directory="."
        )

        # record with the filters the parse left behind (no simplefilter)
        with warnings.catch_warnings(record=True) as caught:
            engine._build_script(bulk("Fe", "bcc", a=2.87))

        self.assertTrue(
            any(
                issubclass(w.category, RuntimeWarning)
                and "takes precedence" in str(w.message)
                for w in caught
            ),
            [str(w.message) for w in caught],
        )

    def test_only_lammpsparsers_eng_unit_notice_is_filtered(self):
        """In a fresh interpreter (pytest resets filters around each test)."""
        package_parent = Path(sys.modules["pyiron_workflow_lammps"].__file__).parents[1]
        code = textwrap.dedent(f"""
            import warnings
            from pyiron_workflow_lammps.lammps import parse_LammpsOutput
            out = parse_LammpsOutput(
                working_directory={str(RESOURCES)!r},
                potential_elements={RESOURCE_ELEMENTS!r},
                units="metal",
            )
            assert ("ignore", None, Warning, None, 0) not in warnings.filters
            warnings.warn(
                "Couldn't determine the LAMMPS to pyiron unit conversion type "
                "of quantity eng.", UserWarning
            )
            warnings.warn("PWL-SENTINEL-USER", UserWarning)
            warnings.warn("PWL-SENTINEL-RUNTIME", RuntimeWarning)
            print("PARSED", out.final_energy)
            """)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(package_parent), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run(
                [sys.executable, "-W", "default", "-c", code],
                capture_output=True,
                check=False,
                text=True,
                cwd=cwd,
                env=env,
                timeout=300,
            )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("PARSED", proc.stdout)
        self.assertNotIn("unit conversion type of quantity eng", proc.stderr)
        self.assertIn("PWL-SENTINEL-USER", proc.stderr)
        self.assertIn("PWL-SENTINEL-RUNTIME", proc.stderr)


class TestParseLammpsOutputMockedParser(unittest.TestCase):
    """Default-parser branch with a stubbed ``lammpsparser.parse_lammps_output_files``.

    The structure and species are still read from a copy of the canned run
    (64 atoms), whose log says the minimization stopped on energy tolerance.
    """

    n_atoms = 64

    def setUp(self):
        n_frames = 3
        self.generic = {
            "cells": [np.eye(3) * 10.0] * n_frames,
            "positions": [np.zeros((self.n_atoms, 3))] * n_frames,
            "indices": [np.ones(self.n_atoms, dtype=int)] * n_frames,
            # MD-like: the total energy contains the kinetic energy
            "energy_pot": np.array([-100.0, -150.0, -149.5]),
            "energy_tot": np.array([-99.0, -148.0, -147.0]),
            "forces": [np.full((self.n_atoms, 3), float(i)) for i in range(n_frames)],
            "pressures": [np.eye(3) * i for i in range(n_frames)],
            "steps": np.array([0, 10, 20]),
        }
        self.parse_mock = MagicMock(
            side_effect=lambda **kwargs: {"generic": self.generic}
        )
        patcher = patch.dict(
            sys.modules,
            {
                "lammpsparser": _fake_lammpsparser(
                    parse_lammps_output_files=self.parse_mock
                )
            },
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, temp_dir)
        self.run_dir = _canned_run(
            temp_dir, stopping_criterion="energy tolerance", log_name="minimize.log"
        )
        _canned_run(
            temp_dir,
            stopping_criterion="energy tolerance",
            wall_time=False,
            log_name="unconverged_minimize.log",
        )

    def _parse(self, **kwargs):
        kwargs.setdefault("log_lammps_file_name", "minimize.log")
        return parse_LammpsOutput(
            working_directory=self.run_dir,
            potential_elements=RESOURCE_ELEMENTS,
            units="metal",
            **kwargs,
        )

    def test_builds_engine_output(self):
        result = self._parse(dump_out_file_name="dump.out")

        self.parse_mock.assert_called_once()
        kwargs = self.parse_mock.call_args.kwargs
        self.assertEqual(kwargs["working_directory"], self.run_dir)
        self.assertEqual(kwargs["potential_elements"], RESOURCE_ELEMENTS)
        self.assertEqual(kwargs["units"], "metal")
        # without an input structure the data file LAMMPS read is the
        # reference, and its (already lower-triangular) cell needs no rotation
        reference = kwargs["structure"]
        self.assertEqual(len(reference), self.n_atoms)
        np.testing.assert_allclose(kwargs["prism"].R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(kwargs["prism"].cell), np.asarray(reference.cell)
        )
        self.assertEqual(kwargs["dump_out_file_name"], "dump.out")
        self.assertEqual(kwargs["log_lammps_file_name"], "minimize.log")

        self.assertIsInstance(result, EngineOutput)
        self.assertIs(result.converged, True)
        self.assertEqual(len(result.energies), 3)
        self.assertEqual(len(result.structures), 3)
        np.testing.assert_array_equal(result.final_forces, np.full((64, 3), 2.0))
        # pressures come from the log's thermo columns, not lammpsparser's
        # "pressures" (stubbed here as diag(0, 1, 2))
        expected = _pressures_in_reference_frame(
            os.path.join(self.run_dir, "minimize.log"), "metal", kwargs["prism"]
        )
        np.testing.assert_allclose(result.final_stress, expected[-1])
        self.assertEqual(np.shape(result.stresses), np.shape(expected))

    def test_final_energy_is_potential_energy(self):
        result = self._parse()

        self.assertIsInstance(result.final_energy, float)
        self.assertEqual(result.final_energy, -149.5)
        np.testing.assert_array_equal(result.energies, [-100.0, -150.0, -149.5])

    def test_n_ionic_steps_is_last_step_as_int(self):
        result = self._parse()

        self.assertIs(type(result.n_ionic_steps), int)
        self.assertEqual(result.n_ionic_steps, 20)

    def test_step_key_fallback(self):
        self.generic["step"] = self.generic.pop("steps")

        result = self._parse()

        self.assertIs(type(result.n_ionic_steps), int)
        self.assertEqual(result.n_ionic_steps, 20)

    def test_scalar_steps(self):
        self.generic["steps"] = np.int64(7)

        result = self._parse()

        self.assertIs(type(result.n_ionic_steps), int)
        self.assertEqual(result.n_ionic_steps, 7)

    def test_missing_steps_falls_back_to_frame_count(self):
        del self.generic["steps"]

        result = self._parse()

        self.assertEqual(result.n_ionic_steps, 2)

    def test_velocities_attached_when_present(self):
        self.generic["velocities"] = [
            np.full((self.n_atoms, 3), 0.01 * i) for i in range(3)
        ]

        result = self._parse()

        self.assertTrue(np.all(result.final_structure.get_velocities() > 0))
        self.assertTrue(np.allclose(result.structures[0].get_velocities(), 0.0))

    def test_unconverged(self):
        result = self._parse(log_lammps_file_name="unconverged_minimize.log")

        self.assertIs(result.converged, False)

    def test_max_iterations_is_not_converged(self):
        _canned_run(
            self.run_dir, stopping_criterion="max iterations", log_name="minimize.log"
        )

        result = self._parse()

        self.assertIs(result.converged, False)
        self.assertEqual(result.final_energy, -149.5)

    def test_failed_command_raises_before_parsing(self):
        with self.assertRaisesRegex(RuntimeError, "return code 1"):
            self._parse(after=_shell_output(1, stderr="ERROR: Unknown pair style"))
        self.parse_mock.assert_not_called()

    def test_missing_structure_file_raises(self):
        with self.assertRaisesRegex(Exception, "Error parsing LAMMPS output"):
            parse_LammpsOutput(
                working_directory="/path/does/not/exist",
                potential_elements=RESOURCE_ELEMENTS,
                units="metal",
            )
        self.parse_mock.assert_not_called()

    def test_input_structure_is_the_reference_frame(self):
        """Given ``structure``, its cell defines the prism used for every frame."""
        from lammpsparser.structure import UnfoldingPrism

        # a primitive bcc cell is not lower triangular, so LAMMPS rotates it
        structure = bulk("Fe", "bcc", a=2.85).repeat((4, 4, 4))
        rotation = UnfoldingPrism(structure.cell).R
        self.assertFalse(np.allclose(rotation, np.eye(3)))
        stresses = np.stack([np.eye(3) * i for i in range(3)])

        with patch(
            "pyiron_workflow_lammps.lammps._pressures_in_reference_frame",
            return_value=stresses,
        ) as pressures:
            result = self._parse(structure=structure)

        kwargs = self.parse_mock.call_args.kwargs
        self.assertIs(kwargs["structure"], structure)
        np.testing.assert_allclose(kwargs["prism"].R, rotation)
        log_path, units, prism = pressures.call_args.args
        self.assertEqual(log_path, os.path.join(self.run_dir, "minimize.log"))
        self.assertEqual(units, "metal")
        self.assertIs(prism, kwargs["prism"])
        np.testing.assert_array_equal(result.final_stress, stresses[-1])

    def test_lammpsparser_pressures_are_not_used(self):
        """Its frame handling must not leak in (it would double-rotate once fixed)."""
        structure = bulk("Fe", "bcc", a=2.85).repeat((4, 4, 4))
        self.generic["pressures"] = [np.full((3, 3), np.nan)] * 3

        result = self._parse(structure=structure)

        self.assertTrue(np.all(np.isfinite(result.final_stress)))

    def test_explicit_prism_is_used(self):
        from lammpsparser.structure import UnfoldingPrism

        structure = bulk("Fe", "bcc", a=2.85).repeat((4, 4, 4))
        prism = UnfoldingPrism(structure.cell)

        self._parse(structure=structure, prism=prism)

        self.assertIs(self.parse_mock.call_args.kwargs["prism"], prism)

    def test_cells_come_back_in_the_input_basis(self):
        # LAMMPS reports the input lattice with other lattice vectors
        # (c' = c - a - b); the trajectory is given back in the input basis
        reference_cell = np.array([[8.0, 0.0, 0.0], [1.0, 9.0, 0.0], [5.0, 5.0, 10.0]])
        change = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        lammps_cell = np.linalg.inv(change) @ reference_cell
        self.generic["cells"] = [lammps_cell, lammps_cell * 1.01, lammps_cell * 1.02]
        structure = bulk("Fe", "bcc", a=2.85, cubic=True).repeat((4, 4, 4))
        structure.set_cell(reference_cell)

        result = self._parse(structure=structure)

        np.testing.assert_allclose(
            np.asarray(result.final_structure.cell), reference_cell * 1.02
        )
        for i, frame in enumerate(result.structures):
            np.testing.assert_allclose(
                np.asarray(frame.cell), reference_cell * (1 + 0.01 * i)
            )
        self.assertAlmostEqual(
            result.final_volume, abs(np.linalg.det(reference_cell * 1.02))
        )


class TestCellsInReferenceBasis(unittest.TestCase):
    """``_cells_in_reference_basis`` restores the input cell's lattice vectors."""

    def setUp(self):
        self.reference = bulk("Fe", "bcc", a=2.85).repeat((3, 3, 3))
        self.cell = np.asarray(self.reference.cell)

    def test_identity_basis_is_unchanged(self):
        cells = np.array([self.cell, self.cell * 1.01])

        result = _cells_in_reference_basis(cells, self.reference)

        np.testing.assert_array_equal(result, cells)
        self.assertEqual(result.dtype, float)

    def test_integer_basis_change_is_applied_to_every_frame(self):
        change = np.array([[1, 0, 0], [0, 1, 0], [-1, -1, 1]])
        lammps_cell = np.linalg.inv(change) @ self.cell
        strain = np.array([[1.0, 0.01, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.98]])
        cells = [lammps_cell, lammps_cell @ strain, lammps_cell * 1.03]

        result = _cells_in_reference_basis(cells, self.reference)

        self.assertEqual(result.shape, (3, 3, 3))
        np.testing.assert_allclose(result[0], self.cell, atol=1e-12)
        np.testing.assert_allclose(result[1], self.cell @ strain, atol=1e-12)
        np.testing.assert_allclose(result[2], self.cell * 1.03, atol=1e-12)
        # the same lattice: an integer, unimodular basis change of the input
        for lammps, restored in zip(cells, result, strict=True):
            self.assertAlmostEqual(
                abs(np.linalg.det(lammps)), abs(np.linalg.det(restored))
            )

    def test_non_integer_change_is_unchanged(self):
        # a rotated (or strained) cell is no integer basis change of the input
        angle = np.deg2rad(10.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        for cells in (
            np.array([self.cell @ rotation.T]),
            np.array([self.cell * 1.1, self.cell * 1.2]),
        ):
            with self.subTest(first=cells[0].tolist()):
                result = _cells_in_reference_basis(cells, self.reference)
                np.testing.assert_array_equal(result, cells)

    def test_empty_or_singular_cells_are_unchanged(self):
        empty = _cells_in_reference_basis(np.zeros((0, 3, 3)), self.reference)
        self.assertEqual(empty.shape, (0, 3, 3))
        singular = np.zeros((1, 3, 3))
        np.testing.assert_array_equal(
            _cells_in_reference_basis(singular, self.reference), singular
        )


def _canonical(name):
    """flowrep labels child nodes ``<function name>_<index>``."""
    return re.sub(r"_\d+$", "", name)


class TestLammpsJobRecipe(unittest.TestCase):
    """The workflow orders its steps by data edges (pyiron_workflow >= 0.19)."""

    def setUp(self):
        self.recipe = lammps_job.flowrep_recipe

    def test_nodes(self):
        self.assertEqual(
            sorted(_canonical(n) for n in self.recipe.nodes),
            sorted(
                [
                    "create_WorkingDirectory",
                    "write_LammpsStructure",
                    "write_LammpsInput",
                    "shell",
                    "parse_LammpsOutput",
                ]
            ),
        )
        self.assertEqual(list(self.recipe.outputs), ["lammps_output"])

    def test_data_edges_chain_every_step(self):
        edges = {
            (
                _canonical(source.node),
                source.port,
                _canonical(target.node),
                target.port,
            )
            for target, source in self.recipe.edges.items()
        }

        expected = {
            # create -> write structure
            (
                "create_WorkingDirectory",
                "working_directory",
                "write_LammpsStructure",
                "working_directory",
            ),
            # write structure -> write input
            (
                "write_LammpsStructure",
                "working_directory",
                "write_LammpsInput",
                "working_directory",
            ),
            # write input -> shell (ordering token)
            ("write_LammpsInput", "filepath", "shell", "after"),
            # shell -> parse (ordering token)
            ("shell", "output", "parse_LammpsOutput", "after"),
        }
        self.assertLessEqual(expected, edges)
        # shell and parse work in the directory the structure was written to
        self.assertIn(
            (
                "write_LammpsStructure",
                "working_directory",
                "shell",
                "working_directory",
            ),
            edges,
        )
        self.assertIn(
            (
                "write_LammpsStructure",
                "working_directory",
                "parse_LammpsOutput",
                "working_directory",
            ),
            edges,
        )

    def test_parser_inputs_reach_parse_node(self):
        input_edges = {
            (_canonical(target.node), target.port): source.port
            for target, source in self.recipe.input_edges.items()
        }

        self.assertEqual(
            input_edges[("parse_LammpsOutput", "_parser_fn")],
            "_lammps_parser_function",
        )
        self.assertEqual(
            input_edges[("parse_LammpsOutput", "_parser_fn_kwargs")],
            "_lammps_parser_args",
        )
        self.assertEqual(input_edges[("shell", "command")], "command")
        self.assertEqual(
            input_edges[("parse_LammpsOutput", "log_lammps_file_name")],
            "lammps_log_filepath",
        )
        # the parser reads the files under the names they were written with
        self.assertEqual(
            input_edges[("parse_LammpsOutput", "dump_out_file_name")],
            "dump_out_file_name",
        )
        self.assertEqual(
            input_edges[("parse_LammpsOutput", "lammps_structure_filepath")],
            "lammps_structure_filepath",
        )
        self.assertEqual(
            input_edges[("write_LammpsStructure", "file_name")],
            "lammps_structure_filepath",
        )
        self.assertEqual(
            input_edges[("write_LammpsInput", "filename")], "input_filename"
        )
        # the parser gets the input structure to rotate results back into
        self.assertEqual(input_edges[("parse_LammpsOutput", "structure")], "structure")
        self.assertEqual(
            input_edges[("write_LammpsStructure", "structure")], "structure"
        )

    def test_parse_node_takes_a_structure(self):
        parameter = inspect.signature(parse_LammpsOutput).parameters["structure"]

        self.assertIsNone(parameter.default)


class TestLammpsJob(unittest.TestCase):
    """Run the whole job with a stand-in ``command`` that copies canned output."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.working_directory = os.path.join(self.temp_dir, "job")
        self.canned = _canned_run(
            os.path.join(self.temp_dir, "canned"), stopping_criterion=None
        )
        self.structure = bulk("Fe", "bcc", a=2.87, cubic=True)
        self.kwargs = {
            "working_directory": self.working_directory,
            "structure": self.structure,
            "lammps_input": "# canned run\n",
            "units": "metal",
            "potential_elements": RESOURCE_ELEMENTS,
            "command": _fake_lammps_command(self.canned),
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _assert_ran(self, result):
        self.assertIsInstance(result, EngineOutput)
        self.assertAlmostEqual(result.final_energy, RESOURCE_ENERGY, places=9)
        self.assertIs(result.converged, True)
        with open(os.path.join(self.working_directory, "in.lmp")) as f:
            self.assertEqual(f.read(), "# canned run\n")

    def test_default_command(self):
        self.assertEqual(DEFAULT_COMMAND, "lmp -in in.lmp -log log.lammps")
        self.assertEqual(
            lammps_job.flowrep_recipe.reference.inputs_with_defaults.count("command"),
            1,
        )
        for function in (lammps_job, lammps_calculator_fn):
            with self.subTest(function=function.__name__):
                parameters = inspect.signature(function).parameters
                self.assertEqual(parameters["command"].default, DEFAULT_COMMAND)
                self.assertEqual(parameters["input_filename"].default, "in.lmp")
                self.assertEqual(
                    parameters["lammps_log_filepath"].default, "log.lammps"
                )
                self.assertEqual(
                    parameters["lammps_structure_filepath"].default, "lammps.data"
                )
                self.assertEqual(parameters["dump_out_file_name"].default, "dump.out")

    def test_direct_call(self):
        self._assert_ran(lammps_job(**self.kwargs))

    def test_custom_log_and_input_names(self):
        _canned_run(self.canned, stopping_criterion=None, log_name="minimize.log")
        self.kwargs["command"] = _fake_lammps_command(
            self.canned, log_name="minimize.log"
        )

        result = lammps_job(
            **self.kwargs,
            input_filename="input.lmp",
            lammps_log_filepath="minimize.log",
        )

        self.assertAlmostEqual(result.final_energy, RESOURCE_ENERGY, places=9)
        self.assertTrue(
            os.path.isfile(os.path.join(self.working_directory, "input.lmp"))
        )

    def test_custom_parser(self):
        seen = {}

        def custom_parser(working_directory, tag):
            seen["files"] = sorted(os.listdir(working_directory))
            return tag

        result = lammps_job(
            **self.kwargs,
            _lammps_parser_function=custom_parser,
            _lammps_parser_args={
                "working_directory": self.working_directory,
                "tag": "custom",
            },
        )

        self.assertEqual(result, "custom")
        # the parser only ran after the "LAMMPS" command had produced output
        self.assertIn("dump.out", seen["files"])
        self.assertIn("log.lammps", seen["files"])

    def test_custom_structure_and_dump_names(self):
        _canned_run(
            self.canned,
            stopping_criterion=None,
            data_name="struct.data",
            dump_name="traj.dump",
        )
        self.kwargs["command"] = _fake_lammps_command(
            self.canned, data_name="struct.data", dump_name="traj.dump"
        )

        result = lammps_job(
            **self.kwargs,
            lammps_structure_filepath="struct.data",
            dump_out_file_name="traj.dump",
        )

        self._assert_ran(result)
        files = os.listdir(self.working_directory)
        self.assertIn("struct.data", files)
        self.assertIn("traj.dump", files)
        self.assertNotIn("lammps.data", files)
        self.assertNotIn("dump.out", files)

    def test_failed_command_raises_with_stderr(self):
        self.kwargs["command"] = "echo 'ERROR: something broke' >&2; exit 3"

        with self.assertRaises(RuntimeError) as context:
            lammps_job(**self.kwargs)

        self.assertIn("return code 3", str(context.exception))
        self.assertIn("ERROR: something broke", str(context.exception))

    def test_failed_rerun_does_not_return_stale_output(self):
        self._assert_ran(lammps_job(**self.kwargs))
        self.assertTrue(
            os.path.isfile(os.path.join(self.working_directory, "log.lammps"))
        )
        # same directory, valid output of the first run still in place, but
        # the launcher of the second run does not exist
        self.kwargs["command"] = "pwl-no-such-lammps -in in.lmp -log log.lammps"
        self.kwargs["lammps_input"] = "# second run\n"

        with self.assertRaises(RuntimeError) as context:
            lammps_job(**self.kwargs)

        message = str(context.exception)
        self.assertIn("return code 127", message)
        self.assertIn("pwl-no-such-lammps", message)
        self.assertIn(self.working_directory, message)

    def test_failed_command_with_custom_parser_raises(self):
        custom_parser = MagicMock(return_value="parsed")
        self.kwargs["command"] = "exit 1"

        with self.assertRaisesRegex(RuntimeError, "return code 1"):
            lammps_job(**self.kwargs, _lammps_parser_function=custom_parser)
        custom_parser.assert_not_called()

    @unittest.skipIf(pwf is None, "pyiron_workflow is not installed")
    def test_failed_command_raises_as_workflow_node(self):
        self.kwargs["command"] = "echo 'ERROR: node broke' >&2; exit 4"

        with self.assertRaises(Exception) as context:
            pwf.run(lammps_job, **self.kwargs)

        self.assertIn("ERROR: node broke", str(context.exception))

    @unittest.skipIf(pwf is None, "pyiron_workflow is not installed")
    def test_as_workflow_node(self):
        run = pwf.run(lammps_job, **self.kwargs)

        self._assert_ran(run.outputs.lammps_output)


class TestLammpsCalculatorFn(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87, cubic=True)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("pyiron_workflow_lammps.lammps.lammps_job")
    def test_forwards_to_lammps_job(self, mock_lammps_job):
        def parser():
            return None

        result = lammps_calculator_fn(
            working_directory=self.temp_dir,
            structure=self.structure,
            lammps_input="units metal",
            units="metal",
            potential_elements=["Fe"],
            _lammps_parser_function=parser,
            _lammps_parser_args={"a": 1},
        )

        self.assertIs(result, mock_lammps_job.return_value)
        mock_lammps_job.assert_called_once_with(
            working_directory=self.temp_dir,
            structure=self.structure,
            lammps_input="units metal",
            units="metal",
            potential_elements=["Fe"],
            lammps_structure_filepath="lammps.data",
            input_filename="in.lmp",
            dump_out_file_name="dump.out",
            command=DEFAULT_COMMAND,
            lammps_log_filepath="log.lammps",
            lammps_log_convergence_printout="Total wall time:",
            _lammps_parser_function=parser,
            _lammps_parser_args={"a": 1},
        )

    @patch("pyiron_workflow_lammps.lammps.lammps_job")
    def test_forwards_file_names(self, mock_lammps_job):
        lammps_calculator_fn(
            working_directory=self.temp_dir,
            structure=self.structure,
            lammps_input="units metal",
            units="metal",
            potential_elements=["Fe"],
            input_filename="run.lmp",
            command="lmp -in run.lmp -log run.log",
            lammps_log_filepath="run.log",
            lammps_log_convergence_printout="Done",
            lammps_structure_filepath="struct.data",
            dump_out_file_name="traj.dump",
        )

        kwargs = mock_lammps_job.call_args.kwargs
        self.assertEqual(kwargs["input_filename"], "run.lmp")
        self.assertEqual(kwargs["command"], "lmp -in run.lmp -log run.log")
        self.assertEqual(kwargs["lammps_log_filepath"], "run.log")
        self.assertEqual(kwargs["lammps_log_convergence_printout"], "Done")
        self.assertEqual(kwargs["lammps_structure_filepath"], "struct.data")
        self.assertEqual(kwargs["dump_out_file_name"], "traj.dump")
        self.assertIsNone(kwargs["_lammps_parser_function"])
        self.assertIsNone(kwargs["_lammps_parser_args"])

    def test_accepts_every_default_engine_kwarg(self):
        from pyiron_workflow_atomistics.engine import CalcInputStatic

        from pyiron_workflow_lammps.engine import LammpsEngine

        engine = LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=self.temp_dir,
            input_script_read_data_file="struct.data",
            input_script_dump_filename="traj.dump",
        )
        calc_fn, calc_kwargs = engine.get_calculate_fn(self.structure)

        with patch("pyiron_workflow_lammps.lammps.lammps_job") as mock_lammps_job:
            calc_fn(structure=self.structure, **calc_kwargs)

        kwargs = mock_lammps_job.call_args.kwargs
        self.assertEqual(kwargs["lammps_structure_filepath"], "struct.data")
        self.assertEqual(kwargs["dump_out_file_name"], "traj.dump")
        self.assertIn("read_data struct.data", kwargs["lammps_input"])
        self.assertIn(" traj.dump id type ", kwargs["lammps_input"])

    def test_runs_job(self):
        working_directory = os.path.join(self.temp_dir, "calc")

        result = lammps_calculator_fn(
            working_directory=working_directory,
            structure=self.structure,
            lammps_input="# canned run\n",
            units="metal",
            potential_elements=RESOURCE_ELEMENTS,
            command=_fake_lammps_command(
                _canned_run(os.path.join(self.temp_dir, "canned"))
            ),
        )

        self.assertIsInstance(result, EngineOutput)
        self.assertAlmostEqual(result.final_energy, RESOURCE_ENERGY, places=9)


if __name__ == "__main__":
    unittest.main()


class TestPressuresInReferenceFrame(unittest.TestCase):
    """``_pressures_in_reference_frame`` on a stubbed thermo table."""

    pressure_bar = np.array(
        [[1.0e4, 2.0e3, 3.0e3], [2.0e3, 2.0e4, 4.0e3], [3.0e3, 4.0e3, 3.0e4]]
    )

    def _pressures(self, cell):
        import pandas as pd
        from lammpsparser.structure import UnfoldingPrism

        p = self.pressure_bar
        df = pd.DataFrame(
            {
                "Step": [0, 10],
                "Pxx": [0.0, p[0, 0]],
                "Pyy": [0.0, p[1, 1]],
                "Pzz": [0.0, p[2, 2]],
                "Pxy": [0.0, p[0, 1]],
                "Pxz": [0.0, p[0, 2]],
                "Pyz": [0.0, p[1, 2]],
            }
        )
        prism = UnfoldingPrism(cell)
        with patch("lammpsparser.output_raw.parse_raw_lammps_log", return_value=df):
            return _pressures_in_reference_frame("log.lammps", "metal", prism), prism

    def test_unrotated_cell_converts_bar_to_gpa(self):
        cell = bulk("Fe", "bcc", a=2.85, cubic=True).cell
        pressures, prism = self._pressures(cell)

        np.testing.assert_allclose(prism.R, np.eye(3), atol=1e-12)
        self.assertEqual(pressures.shape, (2, 3, 3))
        np.testing.assert_allclose(pressures[0], 0.0)
        np.testing.assert_allclose(pressures[1], self.pressure_bar * 1e-4)

    def test_rotated_cell_is_rotated_back(self):
        cell = bulk("Fe", "bcc", a=2.85).cell
        pressures, prism = self._pressures(cell)
        rotation = np.asarray(prism.R)

        self.assertFalse(np.allclose(rotation, np.eye(3)))
        expected = rotation @ (self.pressure_bar * 1e-4) @ rotation.T
        np.testing.assert_allclose(pressures[1], expected, atol=1e-12)
        # a rotation: symmetric with the same trace
        np.testing.assert_allclose(pressures[1], pressures[1].T, atol=1e-12)
        self.assertAlmostEqual(
            np.trace(pressures[1]), np.trace(self.pressure_bar) * 1e-4
        )
