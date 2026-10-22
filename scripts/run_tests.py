#!/usr/bin/env python3
"""Run the pyiron_workflow_lammps test suite with pytest.

Examples::

    python scripts/run_tests.py                  # everything
    python scripts/run_tests.py unit             # tests/unit only
    python scripts/run_tests.py integration -v   # real LAMMPS runs, verbose
    python scripts/run_tests.py unit --cov=pyiron_workflow_lammps

Any further arguments are passed to pytest. Integration tests need a LAMMPS
binary ('lmp' on PATH, or LAMMPS_COMMAND) and are skipped without one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent.parent / "tests"
SUITES = ("unit", "integration", "benchmark")


def main(argv: list[str]) -> int:
    paths = [str(TESTS)]
    if argv and argv[0] in SUITES:
        paths = [str(TESTS / argv[0])]
        argv = argv[1:]
    cmd = [sys.executable, "-m", "pytest", *paths, *argv]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
