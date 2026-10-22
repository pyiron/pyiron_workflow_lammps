"""The public node names and their output labels are part of the API.

Workflows written against this package wire nodes by these names and
connect to these output labels, so renaming either breaks consumers.
"""

import unittest

from pyiron_workflow_lammps import engine, generic, lammps

NODE_OUTPUTS = {
    generic: {
        "shell": ["output"],
        "isLineInFile": ["line_found"],
        "create_WorkingDirectory": ["working_directory"],
        "delete_files_recursively": ["working_directory"],
        "compress_directory": ["compressed_file"],
        "remove_dir": ["compressed_file"],
    },
    lammps: {
        "write_LammpsStructure": ["working_directory"],
        "write_LammpsInput": ["filepath"],
        "parse_LammpsOutput": ["lammps_output"],
        "lammps_job": ["lammps_output"],
    },
}


class TestPublicApi(unittest.TestCase):
    def test_node_names_and_output_labels(self):
        for module, nodes in NODE_OUTPUTS.items():
            for name, outputs in nodes.items():
                with self.subTest(node=name):
                    function = getattr(module, name)
                    self.assertTrue(callable(function))
                    self.assertEqual(function.__name__, name)
                    self.assertEqual(list(function.flowrep_recipe.outputs), outputs)

    def test_ordering_tokens(self):
        for function in (generic.shell, lammps.parse_LammpsOutput):
            with self.subTest(node=function.__name__):
                recipe = function.flowrep_recipe
                self.assertIn("after", recipe.inputs)
                self.assertIn("after", recipe.reference.inputs_with_defaults)

    def test_plain_helpers(self):
        for name in (
            "get_structure_species_lists",
            "get_species_map",
            "arrays_to_ase_atoms",
            "lammps_calculator_fn",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(lammps, name)))

    def test_engine_class(self):
        self.assertTrue(hasattr(engine, "LammpsEngine"))

    def test_submit_to_slurm_removed(self):
        self.assertFalse(hasattr(generic, "submit_to_slurm"))


if __name__ == "__main__":
    unittest.main()
