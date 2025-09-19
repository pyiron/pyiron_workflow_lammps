import unittest

import pyiron_workflow_lammps


class TestVersion(unittest.TestCase):
    def test_version(self):
        """Test that the module has a version attribute."""
        # Check if version attribute exists
        self.assertTrue(hasattr(pyiron_workflow_lammps, "__version__"))

        # Check that version is a string
        version = pyiron_workflow_lammps.__version__
        self.assertIsInstance(version, str)

        # Check that version is not empty
        self.assertGreater(len(version), 0)

        print(f"pyiron_workflow_lammps version: {version}")


class TestModuleImports(unittest.TestCase):
    def test_module_imports(self):
        """Test that all main modules can be imported."""
        # Test main module imports
        from pyiron_workflow_lammps import engine, generic, lammps

        # Test that modules have expected attributes
        self.assertTrue(hasattr(lammps, "write_LammpsStructure"))
        self.assertTrue(hasattr(lammps, "write_LammpsInput"))
        self.assertTrue(hasattr(lammps, "parse_LammpsOutput"))
        self.assertTrue(hasattr(lammps, "lammps_job"))

        self.assertTrue(hasattr(engine, "LammpsEngine"))

        self.assertTrue(hasattr(generic, "shell"))
        self.assertTrue(hasattr(generic, "create_WorkingDirectory"))
        self.assertTrue(hasattr(generic, "isLineInFile"))


if __name__ == "__main__":
    unittest.main()
