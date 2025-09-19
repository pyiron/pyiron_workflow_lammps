"""
Pytest configuration and fixtures for pyiron_workflow_atomistics tests.
"""

import tempfile

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk


@pytest.fixture
def simple_atoms():
    """Create a simple test structure."""
    return Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])


@pytest.fixture
def fcc_al_atoms():
    """Create an FCC aluminum test structure."""
    return bulk("Al", crystalstructure="fcc", a=4.0)


@pytest.fixture
def layered_atoms():
    """Create a layered structure for GB testing."""
    positions = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],  # z=0 layer
        [0, 0, 2],
        [1, 0, 2],
        [0, 1, 2],
        [1, 1, 2],  # z=2 layer
        [0, 0, 4],
        [1, 0, 4],
        [0, 1, 4],
        [1, 1, 4],  # z=4 layer
        [0, 0, 6],
        [1, 0, 6],
        [0, 1, 6],
        [1, 1, 6],  # z=6 layer
    ]
    return Atoms("H16", positions=positions, cell=[2, 2, 8])


@pytest.fixture
def mock_featuriser():
    """Create a mock featuriser for testing."""

    def _featuriser(atoms, site_index, **kwargs):
        return {"feature1": 1.0, "feature2": 2.0}

    return _featuriser


@pytest.fixture
def mock_calculator():
    """Create a mock calculator for testing."""
    from unittest.mock import Mock

    calc = Mock()
    calc.calculate.return_value = {"energy": 1.0, "forces": [[0, 0, 0]]}
    return calc


@pytest.fixture
def mock_engine():
    """Create a mock calculation engine for testing."""
    from unittest.mock import Mock

    engine = Mock()
    engine.get_calculate_fn.return_value = (Mock(), {})
    engine.calculate_fn.return_value = (Mock(), {"working_directory": "/tmp"})
    return engine


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def test_energies_volumes():
    """Create test energies and volumes for EOS testing."""
    volumes = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    energies = 0.1 * (volumes - 14.0) ** 2 + 5.0
    return energies, volumes


@pytest.fixture
def mock_engine_output():
    """Create a mock engine output for testing."""
    from unittest.mock import Mock

    output = Mock()
    output.to_dict.return_value = {
        "final_energy": 1.0,
        "final_volume": 10.0,
        "final_structure": Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]),
        "final_forces": np.zeros((2, 3)),
        "final_stress": np.zeros((3, 3)),
    }
    output.convergence = True
    return output


@pytest.fixture
def mock_engine_outputs(mock_engine_output):
    """Create multiple mock engine outputs for testing."""
    from unittest.mock import Mock

    outputs = []
    for i in range(3):
        output = Mock()
        output.to_dict.return_value = {
            "final_energy": 1.0 + i,
            "final_volume": 10.0 + i,
            "final_structure": Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]),
            "final_forces": np.zeros((2, 3)),
            "final_stress": np.zeros((3, 3)),
        }
        output.convergence = True
        outputs.append(output)
    return outputs


@pytest.fixture
def test_dataclass():
    """Create a test dataclass for testing dataclass utilities."""
    from dataclasses import dataclass

    @dataclass
    class TestClass:
        field1: str
        field2: int
        field3: float

    return TestClass("value1", 42, 3.14)


@pytest.fixture
def test_dict():
    """Create a test dictionary for testing dictionary utilities."""
    return {"param1": "value1", "param2": "value2", "param3": "value3"}


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "slow" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
