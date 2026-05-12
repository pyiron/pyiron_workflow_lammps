# Changelog

All notable changes to `pyiron_workflow_lammps` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: PEP 440 via `versioneer`.

## [0.1.0] — 2026-05-12

### Changed (breaking)

- `LammpsEngine` no longer inherits from
  `pyiron_workflow_atomistics.dataclass_storage.Engine`. The upstream
  cleanup PR (atomistics 0.0.5) deleted `dataclass_storage` and turned
  `Engine` into a `typing.Protocol`. The class is now a plain
  `@dataclass` that satisfies the Protocol structurally.
- All consumers that imported `Engine`, `CalcInputStatic`,
  `CalcInputMinimize`, `CalcInputMD`, or `EngineOutput` from
  `pyiron_workflow_atomistics.dataclass_storage` must switch to
  `pyiron_workflow_atomistics.engine` (e.g.
  `from pyiron_workflow_atomistics.engine import CalcInputStatic`).
- `parse_LammpsOutput` now returns the canonical
  `pyiron_workflow_atomistics.engine.EngineOutput` `@dataclass` with
  required fields `final_structure`, `final_energy`, `converged`
  populated at construction. The previous no-args
  `EngineOutput() + .final_energy = …` pattern is gone — the new
  upstream `EngineOutput` is strict.

### Added

- `LammpsEngine.with_working_directory(subdir) -> LammpsEngine` —
  pure copy via `dataclasses.replace` with the cached
  `calc_fn`/`calc_fn_kwargs` reset so sub-engines rebuild their input
  script against the new directory. Replaces the historical
  `duplicate_engine` helper from pre-0.0.5 atomistics.
- `tests/unit/test_engine_conformance.py`: subclasses the upstream
  `pyiron_workflow_atomistics.testing.EngineConformanceTests` mixin
  with a `LammpsEngine + Al-Fe.eam.fs` factory. CI verifies
  Protocol-satisfaction, `with_working_directory` purity, pickle
  round-trip, `get_calculate_fn` signature, and a single-point
  `run()` smoke against a real LAMMPS binary.
- `tests/fixtures/Al-Fe.eam.fs`: vendored copy of the EAM potential
  used by the conformance run (independent of `notebooks/`).

### Dependencies

- Bumped to track `pyiron_workflow_atomistics==0.0.5`'s pin set
  verbatim: `numpy 1.26.4`, `pandas 3.0.2`, `ase 3.28.0`,
  `pyiron-workflow 0.15.6`, `pymatgen 2026.5.4`, plus the upstream
  testing helpers shipped in the 0.0.5 release.

### Migration guide

Three-line search-and-replace covers most consumers:

```bash
sed -i 's|pyiron_workflow_atomistics\.dataclass_storage|pyiron_workflow_atomistics.engine|g' your-files
```

If you previously did `engine.copy()` + manual `working_directory`
mutation, switch to `engine.with_working_directory("subdir")`.

If you imported `Engine` to do `isinstance(eng, Engine)`, the runtime
check still works (the Protocol is `@runtime_checkable`) — just
import it from `pyiron_workflow_atomistics.engine` instead.

## [0.0.4] — pre-2026-05-12

See git history.
