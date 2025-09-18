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
