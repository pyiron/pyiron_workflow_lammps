"""Conformance test: prove LammpsEngine satisfies the
pyiron_workflow_atomistics Engine Protocol contract.

Runs the 5-method EngineConformanceTests mixin from atomistics 0.0.5+
against a LammpsEngine instance configured for a single-point on
bulk Cu (FCC, a=3.6) using the vendored Al-Fe.eam.fs potential.
"""

from __future__ import annotations

from pathlib import Path

from ase.build import bulk
from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_atomistics.testing import EngineConformanceTests

from pyiron_workflow_lammps.engine import LammpsEngine

_EAM = Path(__file__).resolve().parents[1] / "fixtures" / "Al-Fe.eam.fs"


class TestLammpsEngineConformance(EngineConformanceTests):
    @staticmethod
    def engine_factory(tmp_path):
        return LammpsEngine(
            EngineInput=CalcInputStatic(),
            working_directory=str(tmp_path),
            path_to_model=str(_EAM),
            input_script_pair_style="eam/fs",
            command="lmp -in in.lmp -log log.lammps",
        )

    @staticmethod
    def test_structure_factory():
        return bulk("Fe", "bcc", a=2.85, cubic=True)
