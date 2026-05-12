# LammpsEngine — migrate to pyiron_workflow_atomistics 0.0.5 Engine Protocol

| Field | Value |
|---|---|
| Status | Draft |
| Date | 2026-05-12 |
| Repo | `pyiron/pyiron_workflow_lammps` |
| Upstream contract | [pyiron_workflow_atomistics Engine Protocol](https://github.com/pyiron/pyiron_workflow_atomistics/blob/main/docs/design/specs/2026-05-12-pyiron-workflow-atomistics-cleanup-design.md) |
| Upstream conformance suite | [Engine conformance suite spec](https://github.com/pyiron/pyiron_workflow_atomistics/blob/main/docs/design/specs/2026-05-12-engine-conformance-suite-design.md) |

## Problem

`pyiron_workflow_lammps` 0.0.4 imports the `Engine`/`EngineInput*` dataclasses from `pyiron_workflow_atomistics.dataclass_storage`. The cleanup PR ([atomistics #30](https://github.com/pyiron/pyiron_workflow_atomistics/pull/30)) deleted that module: the symbols moved to `pyiron_workflow_atomistics.engine` and `Engine` is now a `typing.Protocol`, not an abstract base class. As a result:

1. Installing `pyiron_workflow_lammps==0.0.4` against `pyiron_workflow_atomistics>=0.0.5` raises `ImportError: cannot import name 'dataclass_storage'` at module load.
2. Even after fixing the imports, the engine doesn't implement `with_working_directory(subdir) -> Engine` — the contract method that replaced the deleted `duplicate_engine` helper. Macros that use `subengine(engine, subdir)` from atomistics will fail at construction time with `AttributeError` against this engine.
3. The calculate function returns a `PrintableClass`-style ad-hoc object (whatever `parse_LammpsOutput` produces) rather than the canonical `EngineOutput` dataclass. Downstream code that expects `output.final_energy` / `output.converged` / `output.to_dict()` to work uniformly across engines breaks.

The job here is to migrate `LammpsEngine` to satisfy the new Protocol while keeping the LAMMPS-specific input-script generation logic intact. Numbers must not drift.

## Approach

Minimum-viable migration. Static and Minimize modes only conform; MD remains as-is functionally (the existing input-script generation already covers it) but isn't part of the conformance bar this cycle. The work fits in five surgical commits plus a release.

Match the atomistics cleanup PR's "break freely" posture: rename/move/delete symbols where it improves clarity, bump to `pyiron_workflow_lammps-0.1.0`, document the migration in a new CHANGELOG.md entry, do not add deprecation shims.

## Components

```
pyiron_workflow_lammps/
├── __init__.py
├── _version.py                       # versioneer-managed
├── engine.py                         # MODIFIED — Protocol-conformant
├── lammps.py                         # MODIFIED — returns EngineOutput
├── generic.py                        # unchanged
└── dataclass_storage.py              # DELETED — stale copy of upstream's deleted module
tests/
└── unit/
    ├── test_engine_conformance.py    # NEW — subclasses EngineConformanceTests
    └── test_numerical_regression.py  # NEW — golden energies pinned
docs/
└── design/
    ├── specs/2026-05-12-engine-protocol-migration-design.md   # this file
    └── plans/2026-05-12-engine-protocol-migration.md          # implementation plan
CHANGELOG.md                          # NEW — migration notes
```

### Contract clauses + how `LammpsEngine` satisfies each

| Protocol clause | `LammpsEngine` provision |
|---|---|
| `working_directory: str` | Existing field; unchanged. |
| `is_dataclass(engine)` | Existing `@dataclass`; unchanged. |
| `get_calculate_fn(structure) -> (callable, dict)` with `structure` not in dict | Existing implementation already conforms (`engine.py:256`). Verified. |
| `with_working_directory(subdir) -> Engine` (pure, copy via `dataclasses.replace`) | **Add new method**: `return replace(self, working_directory=os.path.join(self.working_directory, subdir))`. |
| `pickle.dumps/loads` round-trip | Holds for plain `@dataclass` with str/int/Callable fields. The `calc_fn`/`parse_fn` slots default to `None` so an unprimed engine pickles fine; once primed (after `get_calculate_fn` cache-fills) it still pickles because the callables are module-level imports. Verified via conformance suite test. |
| Calculate function returns `EngineOutput` | **Rewrite the tail of `parse_LammpsOutput` in `lammps.py`** to construct a real `pyiron_workflow_atomistics.engine.EngineOutput` dataclass from the parsed dict, instead of returning the existing `dataclass_storage.EngineOutput` object (no-args constructor + mutable attribute assignment, which is incompatible with the new strict @dataclass with required fields). |

### `with_working_directory` method

```python
from dataclasses import replace as _replace
import os as _os

def with_working_directory(self, subdir: str) -> "LammpsEngine":
    """Return a copy of this engine with working_directory composed.

    Pure — never mutates self. Re-initialises self.calc_fn / self.calc_fn_kwargs
    to None on the copy so the next get_calculate_fn() call rebuilds the script
    against the new directory.
    """
    return _replace(
        self,
        working_directory=_os.path.join(self.working_directory, subdir),
        calc_fn=None,
        calc_fn_kwargs=None,
    )
```

The `calc_fn=None, calc_fn_kwargs=None` resets matter: the existing `get_calculate_fn` caches its kwargs (including `working_directory`) on first call. Without the reset, the sub-engine would inherit the parent's stale cache.

### `EngineOutput` mapping

Today `parse_LammpsOutput` returns an object with attributes like `final_energy`, `convergence`, etc. — keys roughly correct, but the type is bespoke. Map at the end of the parser:

```python
from pyiron_workflow_atomistics.engine import EngineOutput

# Existing parse logic populates these locals
return EngineOutput(
    final_structure   = final_atoms,
    final_energy      = total_energy_eV,
    converged         = convergence_flag,
    final_forces      = final_forces_arr,
    final_stress      = final_stress_3x3,        # 3x3 tensor in eV/Å^3
    final_stress_voigt= final_stress_voigt_6,    # (6,) Voigt form
    final_volume      = final_volume_A3,
    energies          = trajectory_energies,     # list[float] for minimize/MD
    forces            = trajectory_forces,       # list[np.ndarray]
    structures        = trajectory_structures,   # list[Atoms]
    stresses          = trajectory_stresses,     # list[np.ndarray]
    n_ionic_steps     = len(trajectory_energies),
)
```

## Verification — numerical regression gate

The migration must not change LAMMPS-side numerical outputs. Before any code change lands:

1. Run every existing LAMMPS energy site on `main` at the pre-migration sha with the bundled `Al-Fe.eam.fs` (or whichever potential each test uses):
   - All existing `tests/` cases that assert a numeric value.
   - Existing notebooks that print energies (`notebooks/*.ipynb`).
   - The originals from atomistics' pre-cleanup history (`pyiron/pyiron_workflow_atomistics@39f006e:notebooks/{surface_energy,vacancy_formation_energy,bulk_solution_energy,pure_grain_boundary_study,grain_boundary_segregation}.ipynb`), since those used `LammpsEngine` upstream and act as a wider sanity bar.
2. Capture printed energies into a single pinned-golden file: `tests/unit/test_numerical_regression.py` with each value asserted via `pytest.approx(expected, abs=1e-3)` (sub-meV — rejects any real physics drift, tolerates floating-point noise).
3. Re-run after every migration commit. If a value moves, decide explicitly: legitimate fix (document in PR body and update the golden) vs. regression (investigate, do not merge).

## Commits in this PR (5 + 1 release)

| # | Commit | Touches |
|--:|--------|---------|
| 1 | `chore: swap dataclass_storage→engine import paths` | `engine.py`, `__init__.py`, `tests/`, `notebooks/`; delete local stale `dataclass_storage.py`. |
| 2 | `feat(engine): pure with_working_directory via replace` | `engine.py`. |
| 3 | `feat(engine): return EngineOutput dataclass from parser` | `lammps.py:parse_LammpsOutput`. |
| 4 | `test(engine): subclass EngineConformanceTests` | `tests/unit/test_engine_conformance.py` (new). |
| 5 | `test(numerical): pin pre-migration LAMMPS energies as golden values` | `tests/unit/test_numerical_regression.py` (new), captured before commit 1. |
| 6 | `chore: bump deps to atomistics 0.0.5, ship v0.1.0` | `pyproject.toml`, `.ci_support/{environment,lower_bound}.yml`, `CHANGELOG.md`. |

Release: tag the merge commit `pyiron_workflow_lammps-0.1.0`; `pyproject-release.yml` publishes to PyPI.

## CI footprint

`test_run_returns_engine_output` from the conformance suite calls a real LAMMPS binary. Install LAMMPS from conda-forge in `.ci_support/environment.yml`:

```yaml
dependencies:
  - lammps
```

(conda-forge ships a working serial build that's adequate for a single-point on a 4-atom Cu cell.) No GPU / MPI needed for the conformance smoke. The numerical-regression suite is heavier and may require pinning a specific LAMMPS minor version — flag at PR time if a fresh install solves to a newer LAMMPS that drifts numbers.

## Out of scope

- MD-mode contract conformance (kept functional, not certified).
- Refactoring the input-script `input_script_*` boilerplate; leave the existing 20+ fields alone.
- Replacing `parse_LammpsOutput`'s internal implementation; only its return wrapper changes.
- Migration helpers / deprecation shims — break freely; users follow the CHANGELOG.

## Risk register

1. **`Al-Fe.eam.fs` redistribution**: the upstream atomistics repo bundles it under `notebooks/`. This repo doesn't currently bundle a potential file; the numerical-regression tests will need one. Decision: vendor the same `Al-Fe.eam.fs` under `tests/fixtures/` (~2.2 MB binary in git is acceptable for a one-off; matches upstream practice).
2. **`calc_fn`/`parse_fn` cache state and pickling**: an engine that has been primed with `get_calculate_fn` then pickled has function references in its state. These are bound to module paths so they pickle fine, but a downstream consumer who calls `eng = pickle.loads(...)` and then `eng.with_working_directory(...)` against the **old** cached `working_directory` value would get wrong paths if the `calc_fn=None` reset weren't there. The reset (see method body above) makes this safe.
3. **`pip-check` dep alignment**: must mirror atomistics' pin set exactly or pip-check fails (we've seen this before on the atomistics side). Copy versions verbatim from atomistics 0.0.5's `pyproject.toml`.

## Companion repos

- [`pyiron_workflow_atomistics`](https://github.com/pyiron/pyiron_workflow_atomistics) — defines the Protocol contract and ships the conformance suite. Must release 0.0.5 before this PR opens.
- [`pyiron_workflow_vasp`](https://github.com/ligerzero-ai/pyiron_workflow_vasp) — parallel migration building a `VaspEngine` from scratch on the same contract. Independent PR.
