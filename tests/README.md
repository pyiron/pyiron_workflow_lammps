# pyiron_workflow_lammps tests

```
tests/
├── conftest.py               # markers + LAMMPS fixtures (pytest only)
├── unit/                     # fast, no LAMMPS binary needed (except one smoke test)
│   ├── test_api.py           # public node names / output labels
│   ├── test_engine.py        # LammpsEngine: scripts, statelessness, custom fns
│   ├── test_engine_conformance.py  # atomistics EngineConformanceTests
│   ├── test_generic.py       # shell & file helpers
│   ├── test_lammps.py        # write/parse nodes, lammps_job wiring
│   ├── test_velocity_capture.py
│   └── test_version.py
├── integration/              # real LAMMPS runs (marker: integration)
│   ├── test_lammps_runs.py   # static / minimize / MD through atomistics' calculate,
│   │                         # incl. every thermostat and rotated cells vs ASE's EAM
│   └── test_readme.py        # runs the python blocks of ../README.md
├── benchmark/                # timing sanity checks
├── fixtures/Al-Fe.eam.fs     # potential used by every real LAMMPS run
└── resources/                # canned LAMMPS output (lammps.data, dump.out, logs)
```

## Running

```bash
python -m pytest tests                       # everything
python -m pytest tests/unit                  # unit tests
python -m pytest tests -m "not integration"  # skip real LAMMPS runs
python -m unittest discover tests/unit       # what the shared pyiron CI runs
```

Tests that run LAMMPS use `$LAMMPS_COMMAND` if set (an executable,
optionally with a launcher, e.g. `mpirun -np 2 lmp`), else `lmp` from `PATH`,
and are skipped when neither is found. The binary needs the `MANYBODY`
package for `pair_style eam/fs`.

## Conventions

- `tests/unit` and `tests/benchmark` are also run by `unittest discover`, so
  they use `unittest.TestCase` and no pytest fixtures or imports.
- The suite must run from a copy of `tests/` against an installed package
  (the clean-install CI does this with the sdist's `tests/`): locate data
  via `Path(__file__)`, never import from `tests`, never touch `sys.path`,
  and do not depend on files outside `tests/` (`test_readme.py` skips when
  `README.md` is not next to `tests/`).
- Markers are registered in `conftest.py` (mirrored in `pyproject.toml`).
  `conftest.py` also hides lammpsparser's per-parse `c_eng` unit notice:
  the package filters it at import, but pytest resets warning filters around
  each test (`test_lammps.py` checks the package's filter in a subprocess).
- `tests/resources` holds the output of a real 0-step `minimize 0 0 0 0`
  run, whose log says `Stopping criterion = max iterations` (not
  converged). Tests that need a converged or static log copy it with
  `_canned_run` in `test_lammps.py`, which edits only that line and the
  `Total wall time` line.
- Known source bugs are pinned by `unittest.expectedFailure` /
  `pytest.mark.xfail(strict=True)` tests with the reason in a comment; they
  turn into failures once the bug is fixed, so remove the marker then.
  There are none at the moment.
- The rotated-cell tests in `test_lammps_runs.py` compare with
  `ase.calculators.eam.EAM` on the same `tests/fixtures/Al-Fe.eam.fs`;
  `final_stress` (GPa, compression positive) is compared with
  `-atoms.get_stress(voigt=False) / ase.units.GPa`.
