import unittest
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
from ase import Atoms, Atom
from ase.build import bulk

from pyiron_workflow_atomistics.dataclass_storage import (
    CalcInputStatic, 
    CalcInputMinimize, 
    CalcInputMD
)
from pyiron_workflow_lammps.engine import LammpsEngine


class TestLammpsEngine(unittest.TestCase):
    """Test the LammpsEngine class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.structure = bulk('Fe', 'bcc', a=2.87)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_lammps_engine_init_static(self):
        """Test LammpsEngine initialization with static calculation."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        self.assertEqual(engine.mode, 'static')
        self.assertEqual(engine.working_directory, self.temp_dir)
        self.assertEqual(engine.input_filename, "in.lmp")
        self.assertEqual(engine.command, "lmp -in in.lmp -log log.lammps")
        
    def test_lammps_engine_init_minimize(self):
        """Test LammpsEngine initialization with minimization calculation."""
        engine_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=1000,
            max_evaluations=2000,
            relax_cell=False
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        self.assertEqual(engine.mode, 'minimize')
        self.assertEqual(engine.EngineInput.energy_convergence_tolerance, 1e-6)
        self.assertEqual(engine.EngineInput.force_convergence_tolerance, 1e-5)
        
    def test_lammps_engine_init_md(self):
        """Test LammpsEngine initialization with MD calculation."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='nose-hoover',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        self.assertEqual(engine.mode, 'md')
        self.assertEqual(engine.EngineInput.mode, 'NVT')
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
        self.assertEqual(elements, ['Fe'])
        
        # Multiple elements
        multi_structure = bulk('Fe', 'bcc', a=2.87)
        multi_structure.append(Atom('C', (1.0, 1.0, 1.0)))
        elements = engine.get_lammps_element_order(multi_structure)
        self.assertEqual(elements, ['Fe', 'C'])
        
        # Preserve order
        multi_structure = bulk('C', 'diamond', a=3.57)
        multi_structure.append(Atom('Fe', (1.0, 1.0, 1.0)))
        elements = engine.get_lammps_element_order(multi_structure)
        self.assertEqual(elements, ['C', 'Fe'])
        
    def test_toggle_mode(self):
        """Test mode detection from EngineInput type."""
        engine_input = CalcInputMinimize()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        # Mode should be set during initialization
        self.assertEqual(engine.mode, 'minimize')
        
        # Test manual toggle
        engine.toggle_mode()
        self.assertEqual(engine.mode, 'minimize')
        
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
        
        # Check static-specific content
        self.assertIn("min_style cg", script)
        self.assertIn("minimize 0 0 0 0", script)
        
    def test_build_script_minimize(self):
        """Test script building for minimization calculation."""
        engine_input = CalcInputMinimize(
            energy_convergence_tolerance=1e-6,
            force_convergence_tolerance=1e-5,
            max_iterations=1000,
            max_evaluations=2000,
            relax_cell=False
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
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
            max_evaluations=2000,
            relax_cell=True
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check cell relaxation fix
        self.assertIn("fix 1 all box/relax iso 10000.000000 vmax 0.001", script)
        
    def test_build_script_md_nve(self):
        """Test script building for NVE MD."""
        engine_input = CalcInputMD(
            mode='NVE',
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100
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
            mode='NVT',
            temperature=300,
            thermostat='nose-hoover',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NVT Nose-Hoover content
        self.assertIn("velocity all create 300 12345 mom yes rot yes dist gaussian", script)
        self.assertIn("fix 1 all nvt temp 300 300 0.1", script)
        
    def test_build_script_md_nvt_berendsen(self):
        """Test script building for NVT MD with Berendsen thermostat."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='berendsen',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NVT Berendsen content
        self.assertIn("fix 1 all temp/berendsen 300 300 0.1", script)
        
    def test_build_script_md_nvt_andersen(self):
        """Test script building for NVT MD with Andersen thermostat."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='andersen',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NVT Andersen content
        self.assertIn("fix 1 all langevin 300 300 0.1 12345", script)
        self.assertIn("fix 2 all nve", script)
        
    def test_build_script_md_nvt_temp_rescale(self):
        """Test script building for NVT MD with temp/rescale thermostat."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='temp/rescale',
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345,
            delta_temp=10.0
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NVT temp/rescale content
        self.assertIn("fix 1 all temp/rescale 100 300 300 10.0 units box", script)
        
    def test_build_script_md_nvt_csvr(self):
        """Test script building for NVT MD with CSVR thermostat."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='temp/csvr',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NVT CSVR content
        self.assertIn("fix 1 all temp/csvr 300 300 0.1", script)
        
    def test_build_script_md_nvt_langevin(self):
        """Test script building for NVT MD with Langevin thermostat."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='langevin',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NVT Langevin content
        self.assertIn("fix 1 all langevin 300 300 0.1 12345", script)
        self.assertIn("fix 2 all nve", script)
        
    def test_build_script_md_npt(self):
        """Test script building for NPT MD."""
        engine_input = CalcInputMD(
            mode='NPT',
            temperature=300,
            pressure=1e5,  # 1 bar
            thermostat='nose-hoover',
            temperature_damping_timescale=0.1,
            pressure_damping_timescale=1.0,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        script = engine._build_script(self.structure)
        
        # Check NPT content
        self.assertIn("fix 1 all npt temp 300 300 0.1 iso 1.0 1.0 1.0", script)
        
    def test_build_script_md_npt_wrong_thermostat(self):
        """Test script building for NPT MD with unsupported thermostat."""
        engine_input = CalcInputMD(
            mode='NPT',
            temperature=300,
            pressure=1e5,
            thermostat='berendsen',  # Not supported for NPT
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)
        
        self.assertIn("NPT mode supports only 'nose-hoover' thermostat", str(context.exception))
        
    def test_build_script_md_unsupported_mode(self):
        """Test script building for unsupported MD mode."""
        engine_input = CalcInputMD(
            mode='NPH',  # Not supported
            temperature=300,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)
        
        self.assertIn("Unknown MD mode: NPH", str(context.exception))
        
    def test_build_script_unsupported_thermostat(self):
        """Test script building for unsupported thermostat."""
        engine_input = CalcInputMD(
            mode='NVT',
            temperature=300,
            thermostat='unsupported',
            temperature_damping_timescale=0.1,
            time_step=0.001,
            n_ionic_steps=1000,
            n_print=100,
            seed=12345
        )
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)
        
        self.assertIn("Unsupported thermostat: unsupported", str(context.exception))
        
    def test_build_script_unknown_mode(self):
        """Test script building for unknown mode."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        engine.mode = 'unknown'  # Manually set invalid mode
        
        with self.assertRaises(ValueError) as context:
            engine._build_script(self.structure)
        
        self.assertIn("Unknown mode: unknown", str(context.exception))
        
    def test_write_input_file(self):
        """Test writing input file to disk."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        structure = bulk('Fe', 'bcc', a=2.87)
        filepath = engine.write_input_file(structure)
        
        expected_path = os.path.join(self.temp_dir, "in.lmp")
        self.assertEqual(filepath, expected_path)
        self.assertTrue(os.path.exists(expected_path))
        
        # Check file content
        with open(filepath, 'r') as f:
            content = f.read()
        
        self.assertIn("units metal", content)
        self.assertIn("minimize 0 0 0 0", content)
        
    def test_get_calculate_fn(self):
        """Test calculation function setup."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        structure = bulk('Fe', 'bcc', a=2.87)
        calc_fn, calc_kwargs = engine.get_calculate_fn(structure)
        
        self.assertIsNotNone(calc_fn)
        self.assertIsNotNone(calc_kwargs)
        self.assertEqual(calc_kwargs['working_directory'], self.temp_dir)
        self.assertEqual(calc_kwargs['potential_elements'], ['Fe'])
        self.assertEqual(calc_kwargs['units'], 'metal')
        
    def test_parse_fn(self):
        """Test parse function setup."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
    
        parse_fn = engine.get_parse_fn()
        
        self.assertIsNotNone(parse_fn)
        
    def test_custom_calc_fn(self):
        """Test custom calculation function."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        custom_fn = lambda x: x
        custom_kwargs = {'test': 'value'}
        engine.calc_fn = custom_fn
        engine.calc_fn_kwargs = custom_kwargs
        
        calc_fn, calc_kwargs = engine.get_calculate_fn(self.structure)
        
        self.assertEqual(calc_fn, custom_fn)
        self.assertEqual(calc_kwargs, custom_kwargs)
        
    def test_custom_parse_fn(self):
        """Test custom parse function."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        
        custom_fn = lambda x: x
        engine.parse_fn = custom_fn
        engine.parse_fn_kwargs = {'test': 'value'}
        parse_fn, parse_kwargs = engine.get_parse_fn()
        
        self.assertEqual(parse_fn, custom_fn)
        self.assertEqual(parse_kwargs, {'test': 'value'})
        
    def test_boundary_string(self):
        """Test boundary setting with string input."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        engine.input_script_boundary = "p p p"
        structure = bulk('Fe', 'bcc', a=2.87)
        script = engine._build_script(structure)
        
        self.assertIn("boundary p p p", script)
        
    def test_boundary_tuple(self):
        """Test boundary setting with tuple input."""
        engine_input = CalcInputStatic()
        engine = LammpsEngine(EngineInput=engine_input, working_directory=self.temp_dir)
        engine.input_script_boundary = ("p", "p", "f")
        structure = bulk('Fe', 'bcc', a=2.87)
        script = engine._build_script(structure)
        
        self.assertIn("boundary p p f", script)


if __name__ == '__main__':
    unittest.main() 