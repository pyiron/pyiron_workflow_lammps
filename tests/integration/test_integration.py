import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock
import numpy as np
from ase import Atoms
from ase.build import bulk

from pyiron_workflow_atomistics.dataclass_storage import CalcInputStatic
from pyiron_workflow_lammps.engine import LammpsEngine
from pyiron_workflow_lammps.lammps import (
    write_LammpsStructure,
    write_LammpsInput,
    parse_LammpsOutput
)


class TestLammpsIntegration(unittest.TestCase):
    """Integration tests for LAMMPS workflow components."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk('Fe', 'bcc', a=2.87)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    @patch('pyiron_workflow_lammps.lammps.write_lammps_structure')
    @patch('pyiron_workflow_lammps.lammps.read_lammps_data')
    @patch('pyiron_workflow_lammps.lammps.parse_lammps_output_files')
    @patch('pyiron_workflow_lammps.lammps.get_structure_species_lists')
    @patch('pyiron_workflow_lammps.lammps.isLineInFile')
    def test_engine_to_lammps_integration(self, mock_is_line, mock_get_species, 
                                         mock_parse_output, mock_read_data, mock_write_structure):
        """Test integration between LammpsEngine and lammps functions."""
        # Setup engine
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        # Mock the structure writing
        mock_write_structure.return_value = None
        
        # Mock the output parsing
        mock_atoms = Atoms('Fe', positions=[[0, 0, 0]], cell=[[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]])
        mock_read_data.return_value = mock_atoms
        
        mock_output = {
            'generic': {
                'cells': [np.array([[2.87, 0, 0], [0, 2.87, 0], [0, 0, 2.87]])],
                'positions': [np.array([[0, 0, 0]])],
                'indices': [np.array([0])],
                'energy_tot': [0.0],
                'forces': [np.array([[0, 0, 0]])],
                'pressures': [np.array([0, 0, 0, 0, 0, 0])],
                'steps': [0]
            }
        }
        mock_parse_output.return_value = mock_output
        mock_get_species.return_value = [['Fe']]
        mock_is_line.node_function.return_value = True
        
        # Create dummy files
        data_file = os.path.join(self.temp_dir, "lammps.data")
        dump_file = os.path.join(self.temp_dir, "dump.out")
        log_file = os.path.join(self.temp_dir, "log.lammps")
        
        with open(data_file, 'w') as f:
            f.write("LAMMPS data file\n")
        with open(dump_file, 'w') as f:
            f.write("ITEM: TIMESTEP\n")
        with open(log_file, 'w') as f:
            f.write("Total wall time: 1.234\n")
        
        # Test the integration workflow
        # 1. Engine builds script
        script = engine._build_script(self.structure)
        self.assertIsInstance(script, str)
        self.assertIn("units metal", script)
        
        # 2. Engine writes input file
        input_file = engine.write_input_file()
        self.assertTrue(os.path.exists(input_file))
        
        # 3. Test structure writing
        result = write_LammpsStructure(
            structure=self.structure,
            working_directory=self.temp_dir,
            potential_elements=['Fe'],
            units="metal",
            file_name="test.data"
        )
        self.assertEqual(result, self.temp_dir)
        
        # 4. Test input writing
        input_path = write_LammpsInput(
            lammps_input=script,
            filename="test.in",
            working_directory=self.temp_dir
        )
        self.assertTrue(os.path.exists(input_path))
        
        # 5. Test output parsing
        output = parse_LammpsOutput(
            working_directory=self.temp_dir,
            potential_elements=['Fe'],
            units="metal"
        )
        self.assertIsNotNone(output)
        self.assertTrue(output.convergence)
        
    def test_engine_calculation_setup(self):
        """Test engine calculation function setup."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        # Test calculation function setup
        calc_fn, calc_kwargs = engine.calculate_fn(self.structure)
        
        self.assertIsNotNone(calc_fn)
        self.assertIsNotNone(calc_kwargs)
        self.assertEqual(calc_kwargs['working_directory'], self.temp_dir)
        self.assertEqual(calc_kwargs['potential_elements'], ['Fe'])
        
        # Test parse function setup
        parse_fn = engine.parse_fn(self.structure)
        self.assertIsNotNone(parse_fn)
        
    def test_engine_script_generation_modes(self):
        """Test script generation for different calculation modes."""
        # Test static mode
        static_input = CalcInputStatic()
        static_engine = LammpsEngine(EngineInput=static_input, working_directory=self.temp_dir)
        static_script = static_engine._build_script(self.structure)
        self.assertIn("minimize 0 0 0 0", static_script)
        
        # Test minimization mode
        from pyiron_workflow_atomistics.dataclass_storage import CalcInputMinimize
        min_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=1000,
            max_evaluations=2000,
            relax_cell=False
        )
        min_engine = LammpsEngine(EngineInput=min_input, working_directory=self.temp_dir)
        min_script = min_engine._build_script(self.structure)
        self.assertIn("minimize 1e-06 1e-05 1000 2000", min_script)
        
        # Test MD mode
        from pyiron_workflow_atomistics.dataclass_storage import CalcInputMD
        md_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='nose-hoover',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        md_engine = LammpsEngine(EngineInput=md_input, working_directory=self.temp_dir)
        md_script = md_engine._build_script(self.structure)
        self.assertIn("fix 1 all nvt temp 300 300 0.1", md_script)
        self.assertIn("run 1000", md_script)


if __name__ == '__main__':
    unittest.main()
