# pyiron_workflow_lammps Tests

This directory contains comprehensive unit tests for the `pyiron_workflow_lammps` module.

## Test Structure

```
tests/
├── unit/                    # Unit tests
│   ├── test_lammps.py      # Tests for lammps.py module
│   ├── test_engine.py      # Tests for engine.py module
│   ├── test_generic.py     # Tests for generic.py module
│   ├── test_version.py     # Tests for _version.py module
│   ├── test_tests.py       # Basic module tests
│   └── test_suite.py       # Comprehensive test suite
├── integration/            # Integration tests (future)
└── benchmark/             # Benchmark tests (future)
```

## Running Tests

### Run All Tests

To run all unit tests:

```bash
# From the project root directory
python -m pytest tests/unit/ -v

# Or run the test suite directly
python tests/unit/test_suite.py
```

### Run Individual Test Files

```bash
# Test specific modules
python -m pytest tests/unit/test_lammps.py -v
python -m pytest tests/unit/test_engine.py -v
python -m pytest tests/unit/test_generic.py -v
python -m pytest tests/unit/test_version.py -v
```

### Run with Coverage

```bash
# Install pytest-cov if not already installed
pip install pytest-cov

# Run tests with coverage
python -m pytest tests/unit/ --cov=pyiron_workflow_lammps --cov-report=html
```

## Test Coverage

The unit tests cover the following functionality:

### lammps.py Module
- `write_LammpsStructure`: Structure writing functionality
- `write_LammpsInput`: Input file writing
- `parse_LammpsOutput`: Output parsing with custom and default parsers
- `get_structure_species_lists`: Species list extraction
- `get_species_map`: Species mapping from LAMMPS data files
- `arrays_to_ase_atoms`: Array to ASE Atoms conversion
- `lammps_job`: Complete LAMMPS workflow
- `lammps_calculator_fn`: Calculator function wrapper

### engine.py Module
- `LammpsEngine`: Engine initialization and configuration
- Script building for different calculation modes:
  - Static calculations
  - Minimization (with and without cell relaxation)
  - Molecular dynamics (NVE, NVT, NPT)
  - Various thermostats (Nose-Hoover, Berendsen, Andersen, etc.)
- Element order extraction
- Input file writing
- Calculation and parsing function setup

### generic.py Module
- `Storage`: Data storage utilities
- `ShellOutput`: Shell command output handling
- `VarType`: Variable type definitions
- `FileObject`: File handling utilities
- `shell`: Shell command execution
- `isLineInFile`: File content searching
- `create_WorkingDirectory`: Directory creation
- `delete_files_recursively`: Recursive file deletion
- `compress_directory`: Directory compression
- `submit_to_slurm`: SLURM job submission
- `remove_dir`: Directory removal

### _version.py Module
- Version management functionality
- Git integration
- Command execution utilities

## Test Dependencies

The tests require the following dependencies:
- `pytest` (for running tests)
- `pytest-cov` (for coverage reports)
- `numpy` (for array operations)
- `ase` (for atomic structures)
- `unittest.mock` (for mocking)

## Writing New Tests

When adding new functionality to the module, please add corresponding unit tests:

1. Create test functions that start with `test_`
2. Use descriptive test names that explain what is being tested
3. Test both success and failure cases
4. Use mocking for external dependencies
5. Test edge cases and error conditions
6. Add docstrings to explain test purpose

Example test structure:

```python
def test_function_name_success_case(self):
    """Test successful execution of function_name."""
    # Arrange
    input_data = "test_input"
    
    # Act
    result = function_name(input_data)
    
    # Assert
    self.assertEqual(result, expected_output)

def test_function_name_failure_case(self):
    """Test function_name with invalid input."""
    # Arrange
    invalid_input = None
    
    # Act & Assert
    with self.assertRaises(ValueError):
        function_name(invalid_input)
```

## Continuous Integration

The tests are automatically run in CI/CD pipelines to ensure code quality and prevent regressions. All tests must pass before code can be merged. 