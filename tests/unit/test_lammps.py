import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ase.build import bulk
from pyiron_workflow_atomistics.dataclass_storage import CalcInputStatic

from pyiron_workflow_lammps.engine import LammpsEngine
from pyiron_workflow_lammps.lammps import (
    arrays_to_ase_atoms,
    get_species_map,
    get_structure_species_lists,
    lammps_calculator_fn,
    lammps_job,
    parse_LammpsOutput,
    write_LammpsInput,
    write_LammpsStructure,
)


class TestWriteLammpsStructure(unittest.TestCase):
    """Test the write_LammpsStructure function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)
        self.potential_elements = ["Fe"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_write_lammps_structure_basic(self):
        """Test basic functionality of write_LammpsStructure."""
        result = write_LammpsStructure(
            structure=self.structure,
            working_directory=self.temp_dir,
            potential_elements=self.potential_elements,
            units="metal",
            file_name="test.data",
        )()
        self.assertEqual(result, self.temp_dir)


class TestWriteLammpsInput(unittest.TestCase):
    """Test the write_LammpsInput function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lammps_input = "units metal\ndimension 3\nboundary p p p"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_write_lammps_input_with_directory(self):
        """Test writing LAMMPS input to a specific directory."""
        filename = "test.in"
        result = write_LammpsInput(
            lammps_input=self.lammps_input,
            filename=filename,
            working_directory=self.temp_dir,
        )()

        expected_path = os.path.join(self.temp_dir, filename)
        self.assertEqual(result, expected_path)
        self.assertTrue(os.path.exists(expected_path))

        with open(expected_path) as f:
            content = f.read()
        self.assertEqual(content, self.lammps_input)

    def test_write_lammps_input_without_directory(self):
        """Test writing LAMMPS input to current directory."""
        filename = "test_current.in"

        # Clean up if file exists from previous test
        if os.path.exists(filename):
            os.remove(filename)

        result = write_LammpsInput(lammps_input=self.lammps_input, filename=filename)()

        self.assertEqual(result, filename)
        self.assertTrue(os.path.exists(filename))

        with open(filename) as f:
            content = f.read()
        self.assertEqual(content, self.lammps_input)

        # Clean up
        os.remove(filename)

    def test_write_lammps_input_creates_directory(self):
        """Test that write_LammpsInput creates directory if it doesn't exist."""
        new_dir = os.path.join(self.temp_dir, "new_subdir")
        filename = "test.in"

        result = write_LammpsInput(
            lammps_input=self.lammps_input, filename=filename, working_directory=new_dir
        )()

        expected_path = os.path.join(new_dir, filename)
        self.assertEqual(result, expected_path)
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.exists(expected_path))


class TestGetSpeciesMap(unittest.TestCase):
    """Test the get_species_map function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_species_map_basic(self):
        """Test basic species map extraction."""
        lammps_data_content = """LAMMPS data file

4 atoms
2 atom types

Masses

1 55.847 # (Fe)
2 12.011 # (C)

Atoms

1 1 0.0 0.0 0.0
2 1 1.435 1.435 1.435
3 2 2.87 0.0 0.0
4 2 0.0 2.87 0.0
"""
        data_file = os.path.join(self.temp_dir, "test.data")
        with open(data_file, "w") as f:
            f.write(lammps_data_content)

        species_map = get_species_map(data_file)

        expected = {1: "Fe", 2: "C"}
        self.assertEqual(species_map, expected)

    def test_get_species_map_no_comments(self):
        """Test species map extraction without element comments."""
        lammps_data_content = """LAMMPS data file

2 atoms
2 atom types

Masses

1 55.847
2 12.011

Atoms

1 1 0.0 0.0 0.0
2 2 1.435 1.435 1.435
"""
        data_file = os.path.join(self.temp_dir, "test_no_comments.data")
        with open(data_file, "w") as f:
            f.write(lammps_data_content)

        species_map = get_species_map(data_file)

        expected = {1: None, 2: None}
        self.assertEqual(species_map, expected)

    def test_get_species_map_file_not_found(self):
        """Test get_species_map with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            get_species_map("nonexistent.data")


class TestArraysToAseAtoms(unittest.TestCase):
    """Test the arrays_to_ase_atoms function."""

    def test_arrays_to_ase_atoms_basic(self):
        """Test basic conversion from arrays to ASE Atoms."""
        # Simple cubic cell
        cells = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        indices = np.array([0, 0])  # Both atoms are type 0 (Fe)
        species_lists = [["Fe", "Fe"]]  # Both atoms are Fe

        atoms = arrays_to_ase_atoms(cells, positions, indices, species_lists)

        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms.get_chemical_symbols(), ["Fe", "Fe"])
        np.testing.assert_array_almost_equal(atoms.get_positions(), positions)
        np.testing.assert_array_almost_equal(atoms.get_cell(), cells)
        self.assertTrue(atoms.get_pbc().all())

    def test_arrays_to_ase_atoms_multiple_elements(self):
        """Test conversion with multiple element types."""
        cells = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]])
        positions = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]])
        indices = np.array([0, 1])  # First atom type 0, second type 1
        species_lists = [["Fe", "C"]]  # Fe and C

        atoms = arrays_to_ase_atoms(cells, positions, indices, species_lists)

        self.assertEqual(len(atoms), 2)
        self.assertEqual(atoms.get_chemical_symbols(), ["Fe", "C"])

    def test_arrays_to_ase_atoms_no_pbc(self):
        """Test conversion with periodic boundary conditions disabled."""
        cells = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
        positions = np.array([[0.0, 0.0, 0.0]])
        indices = np.array([0])
        species_lists = [["Fe"]]

        atoms = arrays_to_ase_atoms(cells, positions, indices, species_lists, pbc=False)

        self.assertFalse(atoms.get_pbc().any())


class TestParseLammpsOutput(unittest.TestCase):
    """Test the parse_LammpsOutput function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.potential_elements = ["Fe", "C"]
        self.units = "metal"
        self.resources_dir = os.path.join(
            os.path.dirname(__file__), os.sep.join(["..", "resources"])
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_parse_lammps_output_success(self):
        """Test successful LAMMPS output parsing."""
        # Mock the structure

        result = parse_LammpsOutput(
            working_directory=self.resources_dir,
            potential_elements=self.potential_elements,
            lammps_structure_filepath="lammps.data",
            dump_out_file_name="dump.out",
            log_lammps_file_name="minimize.log",
            log_lammps_convergence_printout="Total wall time:",
            units=self.units,
        )()

        self.assertIsNotNone(result)
        self.assertTrue(result.convergence)
        self.assertEqual(result.final_energy, np.float64(-454.367609882893))

    def test_parse_lammps_output_success_default_filename(self):
        """Test successful LAMMPS output parsing."""
        # Mock the structure

        result = parse_LammpsOutput(
            working_directory=self.resources_dir,
            potential_elements=self.potential_elements,
            log_lammps_convergence_printout="Total wall time:",
            units=self.units,
        )()

        self.assertIsNotNone(result)
        self.assertTrue(result.convergence)
        self.assertEqual(result.final_energy, np.float64(-454.367609882893))

    def test_parse_lammps_output_success_unconverged(self):
        """Test successful LAMMPS output parsing."""
        # Mock the structure

        result = parse_LammpsOutput(
            working_directory=self.resources_dir,
            potential_elements=self.potential_elements,
            log_lammps_file_name="unconverged_minimize.log",
            log_lammps_convergence_printout="Total wall time:",
            units=self.units,
        )()

        self.assertIsNotNone(result)
        self.assertFalse(result.convergence)
        self.assertEqual(result.final_energy, np.float64(-454.367609882893))

    def test_parse_lammps_output_missing_files(self):
        """Test parsing with missing output files."""
        with self.assertRaises(Exception):
            parse_LammpsOutput(
                working_directory=self.temp_dir,
                potential_elements=self.potential_elements,
                units=self.units,
            )()

    def test_parse_lammps_output_custom_parser(self):
        """Test parsing with custom parser function."""
        mock_parser = MagicMock()
        mock_output = MagicMock()
        mock_parser.return_value = mock_output

        result = parse_LammpsOutput(
            working_directory=self.temp_dir,
            potential_elements=self.potential_elements,
            units=self.units,
            _parser_fn=mock_parser,
            _parser_fn_kwargs={"arg1": "value1"},
        )()

        mock_parser.assert_called_once_with(arg1="value1")
        self.assertEqual(result, mock_output)

    def test_parse_lammps_output_custom_parser_failure(self):
        """Test parsing with custom parser that fails."""

        def failing_parser():
            raise ValueError("Parser failed")

        with self.assertRaises(Exception) as context:
            parse_LammpsOutput(
                working_directory=self.temp_dir,
                potential_elements=self.potential_elements,
                units=self.units,
                _parser_fn=failing_parser,
                _parser_fn_kwargs={},
            )()

        self.assertIn("Error parsing LAMMPS output", str(context.exception))


class TestGetStructureSpeciesLists(unittest.TestCase):
    """Test the get_structure_species_lists function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.resources_dir = os.path.join(
            os.path.dirname(__file__), os.sep.join(["..", "resources"])
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_structure_species_lists(self):
        """Test species list extraction from LAMMPS dump files."""

        data_file = os.path.join(self.resources_dir, "lammps.data")
        dump_file = os.path.join(self.resources_dir, "dump.out")

        result = get_structure_species_lists(data_file, dump_file)

        expected = [
            [
                "C",
                "Fe",
                "C",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
                "Fe",
            ]
        ]
        self.assertEqual(result, expected)


class TestLammpsJob(unittest.TestCase):
    """Test the lammps_job macro node."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)
        self.lammps_input = "units metal\ndimension 3\nboundary p p p"
        self.units = "metal"
        self.potential_elements = ["Fe"]
        self.structure = bulk("Fe") * [4, 4, 4]
        self.structure.rattle(0.3)
        self.structure[0].symbol = "C"
        self.structure[2].symbol = "C"

        self.EngineInput = CalcInputStatic()

        self.Engine = LammpsEngine(EngineInput=self.EngineInput)
        self.Engine.working_directory = "EnginePrototypeStatic"
        self.Engine.command = (
            "lmp -in in.lmp -log minimize.log"
        )
        self.Engine.lammps_log_filepath = "minimize.log"
        resources_dir = os.path.join(os.path.dirname(__file__), "..", "resources")
        resources_dir = "/home/runner/work/pyiron_workflow_lammps/pyiron_workflow_lammps/pyiron_workflow_lammps/tests/unit/resources"
        self.Engine.path_to_model = os.sep.join([resources_dir, "Al-Fe.eam.fs"])
        self.potential_elements = self.Engine.get_lammps_element_order(self.structure)
        self.input_filename = "in.lmp"
        self.lammps_log_convergence_printout = "Total wall time:"
        self.calc_fn = lammps_calculator_fn
        self.calc_fn_kwargs = {
            "working_directory": self.Engine.working_directory,
            "lammps_input": self.Engine._build_script(self.structure),
            "potential_elements": self.potential_elements,
            "input_filename": self.input_filename,
            "command": self.Engine.command,
            "lammps_log_filepath": self.Engine.lammps_log_filepath,
            "units": self.Engine.input_script_units,
            "lammps_log_convergence_printout": self.lammps_log_convergence_printout,
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_lammps_job(self):
        """Test the complete LAMMPS job workflow."""
        print(os.system("which lmp"))
        print(os.system("pwd"))
        print(os.getcwd())
        print(os.listdir(os.getcwd()))
        result = lammps_job(
            working_directory=self.Engine.working_directory,
            structure=self.structure,
            lammps_input=self.Engine._build_script(self.structure),
            units=self.units,
            potential_elements=self.potential_elements,
        )()

        # Verify workflow wiring
        # self.assertEqual(result.working_dir, self.temp_dir)


class TestLammpsCalculatorFn(unittest.TestCase):
    """Test the lammps_calculator_fn function."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk("Fe", "bcc", a=2.87)
        self.lammps_input = "units metal\ndimension 3\nboundary p p p"
        self.units = "metal"
        self.potential_elements = ["Fe"]

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("pyiron_workflow_lammps.lammps.lammps_job")
    def test_lammps_calculator_fn(self, mock_lammps_job):
        """Test the lammps_calculator_fn wrapper."""
        mock_output = MagicMock()
        mock_output.__getitem__.return_value = {"lammps_output": "test_result"}
        mock_lammps_job.return_value.return_value = mock_output

        result = lammps_calculator_fn(
            working_directory=self.temp_dir,
            structure=self.structure,
            lammps_input=self.lammps_input,
            units=self.units,
            potential_elements=self.potential_elements,
        )

        mock_lammps_job.assert_called_once()
        self.assertEqual(result["lammps_output"], "test_result")


if __name__ == "__main__":
    unittest.main()
