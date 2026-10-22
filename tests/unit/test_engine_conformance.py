"""Conformance: LammpsEngine satisfies pyiron_workflow_atomistics' Engine Protocol.

Wraps the pytest-style ``EngineConformanceTests`` mixin in a
``unittest.TestCase`` so that both pytest and the shared pyiron CI (which
runs ``unittest discover``) pick it up.

The engine runs a single point on a 2-atom Fe/Al cell with ``pair_style
eam/fs`` and the vendored ``tests/fixtures/Al-Fe.eam.fs`` potential. Only
``test_run_returns_engine_output`` needs LAMMPS: it uses ``$LAMMPS_COMMAND``
(an executable, optionally with a launcher, e.g. ``mpirun -np 2 lmp``) or
else ``lmp`` on ``PATH``, and is skipped when neither is available.
"""

from __future__ import annotations

import os
import shlex
import shutil
import tempfile
import unittest
from pathlib import Path

from ase.build import bulk
from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_atomistics.testing import EngineConformanceTests

from pyiron_workflow_lammps.engine import LammpsEngine

_EAM = Path(__file__).resolve().parents[1] / "fixtures" / "Al-Fe.eam.fs"


def _lammps_executable() -> str | None:
    """The LAMMPS launcher to test with, or None if there is none."""
    command = os.environ.get("LAMMPS_COMMAND", "").strip()
    if command:
        return command if shutil.which(shlex.split(command)[0]) else None
    return "lmp" if shutil.which("lmp") else None


LAMMPS_EXECUTABLE = _lammps_executable()


class _LammpsEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        return LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=str(tmp_path),
            path_to_model=str(_EAM),
            input_script_pair_style="eam/fs",
            command=f"{LAMMPS_EXECUTABLE or 'lmp'} -in in.lmp -log log.lammps",
        )

    @staticmethod
    def test_structure_factory():
        structure = bulk("Fe", "bcc", a=2.87, cubic=True)
        structure[1].symbol = "Al"
        return structure


class TestLammpsEngineConformance(unittest.TestCase):
    """Each test delegates to a fresh mixin instance with its own tmp_path."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self._mixin = _LammpsEngineConformance()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_satisfies_engine_protocol(self) -> None:
        self._mixin.test_satisfies_engine_protocol(self.tmp_path)

    def test_with_working_directory_is_pure(self) -> None:
        self._mixin.test_with_working_directory_is_pure(self.tmp_path)

    def test_pickleable(self) -> None:
        self._mixin.test_pickleable(self.tmp_path)

    def test_get_calculate_fn_signature(self) -> None:
        self._mixin.test_get_calculate_fn_signature(self.tmp_path)

    def test_fixture_potential_present(self) -> None:
        self.assertTrue(_EAM.is_file(), f"missing {_EAM}")

    @unittest.skipIf(
        LAMMPS_EXECUTABLE is None,
        "no LAMMPS binary: put 'lmp' on PATH or set LAMMPS_COMMAND",
    )
    def test_run_returns_engine_output(self) -> None:
        self._mixin.test_run_returns_engine_output(self.tmp_path)


if __name__ == "__main__":
    unittest.main()
