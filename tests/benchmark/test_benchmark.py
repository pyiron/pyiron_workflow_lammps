import unittest
import time
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
    write_LammpsInput,
    get_species_map,
    arrays_to_ase_atoms,
)


class TestLammpsBenchmark(unittest.TestCase):
    """Benchmark tests for LAMMPS workflow performance."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def benchmark_script_generation(self):
        """Benchmark script generation performance."""
        # Create different sized structures
        structures = {
            "small": bulk("Fe", "bcc", a=2.87),
            "medium": bulk("Fe", "bcc", a=2.87).repeat((2, 2, 2)),
            "large": bulk("Fe", "bcc", a=2.87).repeat((4, 4, 4)),
        }

        engine_input = CalcInputStatic()

        results = {}
        for size, structure in structures.items():
            engine = LammpsEngine(
                EngineInput=engine_input, working_directory=self.temp_dir
            )

            # Time script generation
            start_time = time.time()
            script = engine._build_script(structure)
            end_time = time.time()

            results[size] = {
                "time": end_time - start_time,
                "atoms": len(structure),
                "script_length": len(script),
            }

            print(
                f"{size.capitalize()} structure ({len(structure)} atoms): "
                f"{results[size]['time']:.4f}s, {len(script)} chars"
            )

        return results

    def benchmark_input_writing(self):
        """Benchmark input file writing performance."""
        lammps_input = (
            "units metal\ndimension 3\nboundary p p p\n" * 1000
        )  # Large input

        # Test writing to different locations
        locations = [
            self.temp_dir,
            os.path.join(self.temp_dir, "subdir"),
            os.path.join(self.temp_dir, "deep", "nested", "directory"),
        ]

        results = {}
        for location in locations:
            start_time = time.time()
            result = write_LammpsInput(
                lammps_input=lammps_input,
                filename="test.in",
                working_directory=location,
            )()
            end_time = time.time()

            results[location] = {
                "time": end_time - start_time,
                "file_size": os.path.getsize(result) if os.path.exists(result) else 0,
            }

            print(
                f"Writing to {location}: {results[location]['time']:.4f}s, "
                f"{results[location]['file_size']} bytes"
            )

        return results

    def benchmark_species_mapping(self):
        """Benchmark species mapping performance."""
        # Create LAMMPS data files with different numbers of atom types
        atom_types_configs = [2, 5, 10, 20]

        results = {}
        for num_types in atom_types_configs:
            # Create LAMMPS data content
            content = "LAMMPS data file\n\n"
            content += f"{num_types * 10} atoms\n"
            content += f"{num_types} atom types\n\n"
            content += "Masses\n\n"

            for i in range(1, num_types + 1):
                content += f"{i} {55.847 + i} # (Fe{i})\n"

            content += "\nAtoms\n\n"
            for i in range(1, num_types * 10 + 1):
                atom_type = ((i - 1) % num_types) + 1
                content += f"{i} {atom_type} {i*0.1} {i*0.2} {i*0.3}\n"

            # Write to file
            data_file = os.path.join(self.temp_dir, f"test_{num_types}.data")
            with open(data_file, "w") as f:
                f.write(content)

            # Benchmark species mapping
            start_time = time.time()
            species_map = get_species_map(data_file)
            end_time = time.time()

            results[num_types] = {
                "time": end_time - start_time,
                "num_types": num_types,
                "species_map_size": len(species_map),
            }

            print(
                f"{num_types} atom types: {results[num_types]['time']:.4f}s, "
                f"{len(species_map)} species mapped"
            )

        return results

    def benchmark_array_conversion(self):
        """Benchmark array to ASE Atoms conversion performance."""
        # Create arrays of different sizes
        sizes = [10, 50, 100, 500, 1000]

        results = {}
        for size in sizes:
            # Create test arrays
            cells = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
            positions = np.random.rand(size, 3)
            indices = np.random.randint(0, 3, size)  # 3 different atom types
            base_species = ["Fe", "C", "O"]
            # Build per-atom species list for the final frame
            species_per_atom = [base_species[i] for i in indices]
            species_lists = [species_per_atom]

            # Benchmark conversion
            start_time = time.time()
            atoms = arrays_to_ase_atoms(cells, positions, indices, species_lists)
            end_time = time.time()

            results[size] = {
                "time": end_time - start_time,
                "num_atoms": len(atoms),
                "num_elements": len(set(atoms.get_chemical_symbols())),
            }

            print(
                f"{size} atoms: {results[size]['time']:.4f}s, "
                f"{len(atoms)} atoms, {len(set(atoms.get_chemical_symbols()))} elements"
            )

        return results

    def test_benchmark_performance(self):
        """Run all benchmarks and report results."""
        print("\n" + "=" * 60)
        print("LAMMPS Workflow Performance Benchmarks")
        print("=" * 60)

        # Run benchmarks
        print("\n1. Script Generation Benchmark:")
        script_results = self.benchmark_script_generation()

        print("\n2. Input Writing Benchmark:")
        writing_results = self.benchmark_input_writing()

        print("\n3. Species Mapping Benchmark:")
        mapping_results = self.benchmark_species_mapping()

        print("\n4. Array Conversion Benchmark:")
        conversion_results = self.benchmark_array_conversion()

        # Summary
        print("\n" + "=" * 60)
        print("Benchmark Summary")
        print("=" * 60)

        # Script generation summary
        avg_script_time = np.mean([r["time"] for r in script_results.values()])
        print(f"Average script generation time: {avg_script_time:.4f}s")

        # Input writing summary
        avg_writing_time = np.mean([r["time"] for r in writing_results.values()])
        print(f"Average input writing time: {avg_writing_time:.4f}s")

        # Species mapping summary
        avg_mapping_time = np.mean([r["time"] for r in mapping_results.values()])
        print(f"Average species mapping time: {avg_mapping_time:.4f}s")

        # Array conversion summary
        avg_conversion_time = np.mean([r["time"] for r in conversion_results.values()])
        print(f"Average array conversion time: {avg_conversion_time:.4f}s")

        # Performance assertions (adjust thresholds as needed)
        self.assertLess(avg_script_time, 1.0, "Script generation too slow")
        self.assertLess(avg_writing_time, 0.1, "Input writing too slow")
        self.assertLess(avg_mapping_time, 0.01, "Species mapping too slow")
        self.assertLess(avg_conversion_time, 0.1, "Array conversion too slow")

        print("\n✅ All performance benchmarks passed!")


if __name__ == "__main__":
    unittest.main()
