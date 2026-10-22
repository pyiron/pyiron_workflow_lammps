"""Pytest configuration for the pyiron_workflow_lammps test suite.

The suite must also run from a copy of ``tests/`` against an installed
package (the clean-install CI copies the sdist's ``tests/`` elsewhere), so
nothing here may rely on the repository root: markers are registered in
code rather than only in ``pyproject.toml``, and data files are located
relative to the test modules via ``Path(__file__)``.

``tests/unit`` and ``tests/benchmark`` are also run with
``python -m unittest discover`` by the shared pyiron CI, so they are
written as ``unittest.TestCase`` classes and do not use fixtures from here.
"""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: runs a real LAMMPS binary ('lmp' on PATH, or the "
        "LAMMPS_COMMAND environment variable); skipped when none is found",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    # The package filters lammpsparser's per-parse notice about the c_eng
    # dump column when it is imported, but pytest restores the warning
    # filters around every test and collection, so the notice would flood
    # the summary. test_lammps.py checks the package's own filter in a
    # fresh interpreter.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*unit conversion type of quantity eng.*:UserWarning",
    )


def _lammps_executable() -> str | None:
    """``$LAMMPS_COMMAND`` (e.g. ``lmp_serial`` or ``mpirun -np 2 lmp``) or ``lmp``.

    Returns None when the executable cannot be found.
    """
    command = os.environ.get("LAMMPS_COMMAND", "").strip()
    if command:
        return command if shutil.which(shlex.split(command)[0]) else None
    return "lmp" if shutil.which("lmp") else None


@pytest.fixture(scope="session")
def lammps_executable() -> str:
    """The LAMMPS launcher for integration tests; skips the test without one."""
    executable = _lammps_executable()
    if executable is None:
        pytest.skip("no LAMMPS binary: put 'lmp' on PATH or set LAMMPS_COMMAND")
    return executable


@pytest.fixture(scope="session")
def eam_potential() -> Path:
    """The vendored Al-Fe Finnis-Sinclair potential (needs LAMMPS' MANYBODY)."""
    return FIXTURES / "Al-Fe.eam.fs"
