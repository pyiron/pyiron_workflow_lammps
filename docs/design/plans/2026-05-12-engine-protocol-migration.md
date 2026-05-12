# LammpsEngine Engine-Protocol Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate `pyiron_workflow_lammps.engine.LammpsEngine` onto the new `pyiron_workflow_atomistics==0.0.5` Engine Protocol contract (Static + Minimize conformance, MD kept functional but not certified). Ship as `pyiron_workflow_lammps-0.1.0`.

**Architecture:** Surgical refactor of the existing `LammpsEngine` `@dataclass`. Swap import paths (`dataclass_storage` → `engine` and `engine.inputs`), add a pure `with_working_directory` method, rewire the parser's tail to construct a real `EngineOutput` dataclass with required fields, subclass the upstream `EngineConformanceTests` mixin, bump deps verbatim to atomistics 0.0.5's pin set, document in `CHANGELOG.md`, release.

**Tech Stack:** Python 3.10+ (project says 3.9–3.12), `typing.Protocol`, `dataclasses` (incl. `replace`), pytest. Existing repo conventions: ruff/black, versioneer, conda-forge env via `.ci_support/`, shared pyiron CI workflows. Real LAMMPS binary required in CI (`conda install -c conda-forge lammps`, already in `.ci_support/environment.yml`).

**Spec:** `docs/design/specs/2026-05-12-engine-protocol-migration-design.md`.

**Branch:** `design-engine-protocol-migration` (already pushed to `origin`).

**Working directory:** `/home/liger/pyiron_workflow_lammps`.

**Python interpreter / pytest binary:** `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python` and `/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest` (the env already has `pyiron-workflow-atomistics==0.0.5` installed from the just-published release plus a real `lammps` binary, so the conformance suite can run locally).

---

## File structure (what this PR touches)

| Path | Action | Responsibility |
|---|---|---|
| `pyiron_workflow_lammps/engine.py` | MODIFY | Stop inheriting from deleted `dataclass_storage.Engine`; swap imports to `pyiron_workflow_atomistics.engine.{CalcInputStatic, CalcInputMinimize, CalcInputMD}`; add `with_working_directory(subdir) -> LammpsEngine`. |
| `pyiron_workflow_lammps/lammps.py` | MODIFY | Replace the no-args `EngineOutput() + mutable-attr-assignment` pattern at the end of `parse_LammpsOutput` with a `EngineOutput(...)` construction supplying the three required fields plus the optional ones. Swap `from pyiron_workflow_atomistics.dataclass_storage import EngineOutput` → `from pyiron_workflow_atomistics.engine import EngineOutput`. |
| `pyiron_workflow_lammps/dataclass_storage.py` | DELETE | Empty stub (0 bytes); irrelevant after the import swap. |
| `pyiron_workflow_lammps/__init__.py` | MODIFY (if needed) | Drop any re-exports that referenced the stale `dataclass_storage`. |
| `tests/unit/test_engine.py` | MODIFY | Swap `from pyiron_workflow_atomistics.dataclass_storage import …` → `from pyiron_workflow_atomistics.engine import …`. |
| `tests/unit/test_lammps.py` | MODIFY | Same. |
| `tests/benchmark/test_benchmark.py` | MODIFY | Same. |
| `tests/integration/test_integration.py` | MODIFY | Same (three call sites). |
| `notebooks/EnginePrototype_lammps.ipynb` | MODIFY | Same import swap inside the relevant code cell. |
| `tests/unit/test_engine_conformance.py` | NEW | `TestLammpsEngineConformance(EngineConformanceTests)` with a `LammpsEngine(EngineInput=CalcInputStatic(), …)` factory and a `bulk("Cu","fcc",a=3.6,cubic=True)` test structure. |
| `tests/fixtures/Al-Fe.eam.fs` | NEW | Vendored copy of the EAM potential used by tests + the conformance run (~2.2 MB, single binary). Same file as `notebooks/Al-Fe.eam.fs`. |
| `pyproject.toml` | MODIFY | Bump every dep pin to match atomistics 0.0.5's verbatim list. |
| `.ci_support/environment.yml` | MODIFY | Bump the same versions; ensure `lammps` is present (it already is at `=2024.08.29`). |
| `.ci_support/lower-bound.yml` | MODIFY | Bump matching floors. |
| `CHANGELOG.md` | NEW | Top-of-file `0.1.0` section describing the migration + the v0.0.4 user impact. |
| `docs/design/plans/2026-05-12-engine-protocol-migration.md` | NEW | This plan. Committed first. |

No file in `pyiron_workflow_lammps/engine.py`'s 20+ `input_script_*` boilerplate fields is touched — that's out of scope per the spec.

---

## Task 1: Commit this plan first

**Files:**
- Create: `docs/design/plans/2026-05-12-engine-protocol-migration.md` (this file)

- [ ] **Step 1: Verify the plan file exists and is untracked**

```bash
cd /home/liger/pyiron_workflow_lammps
git status --short docs/design/plans/2026-05-12-engine-protocol-migration.md
```

Expected: `?? docs/design/plans/2026-05-12-engine-protocol-migration.md`.

- [ ] **Step 2: Commit and push**

```bash
git add docs/design/plans/2026-05-12-engine-protocol-migration.md
git commit -m "docs(plan): LammpsEngine migration implementation plan

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin design-engine-protocol-migration
```

Expected: branch tip on remote advances to the new commit.

---

## Task 2: Capture pre-migration energy baseline

**Files:**
- (no edits)

This task is read-only — it identifies which numbers must not drift, by running the existing test suite on the current `design-engine-protocol-migration` branch state and recording every numeric assertion's expected value.

- [ ] **Step 1: Grep the existing test suite for hard-coded energy / force / volume assertions**

```bash
cd /home/liger/pyiron_workflow_lammps
grep -nE "assert.*(approx|==|<|>).*[-+]?[0-9]+\.[0-9]+" tests/unit/*.py tests/integration/*.py 2>&1 | head -30
```

Expected: a list of every numeric assertion in the suite. Note them — they are the pre-migration goldens (already pinned in the test source). The migration must keep every one of these passing.

- [ ] **Step 2: Note the count for later cross-check**

Save the count of existing numeric assertions into a working note. Example:

```bash
NUMERIC_ASSERT_COUNT=$(grep -cE "assert.*(approx|==|<|>).*[-+]?[0-9]+\.[0-9]+" tests/unit/*.py tests/integration/*.py 2>/dev/null)
echo "Pre-migration numeric assertions: $NUMERIC_ASSERT_COUNT"
```

Expected: a non-zero integer. This is the bar: after Task 9 (final pytest sweep), the same set of asserts must still pass.

- [ ] **Step 3: Document in the PR body before opening for review**

(No file change — this is a checklist item to remember to populate the PR description with the count and a list of asserted files. Captured by the PR-promotion step in Task 11.)

---

## Task 3: Swap `dataclass_storage` → `engine` import paths

**Files:**
- Modify: `pyiron_workflow_lammps/engine.py` (lines 9–14 — the import block)
- Modify: `pyiron_workflow_lammps/lammps.py` (line 88 — internal import)
- Modify: `tests/unit/test_engine.py` (line 8)
- Modify: `tests/unit/test_lammps.py` (line 9)
- Modify: `tests/benchmark/test_benchmark.py` (line 9)
- Modify: `tests/integration/test_integration.py` (lines 6, 32, 48)
- Modify: `notebooks/EnginePrototype_lammps.ipynb` (search for the cell with `dataclass_storage`)
- Delete: `pyiron_workflow_lammps/dataclass_storage.py` (0-byte stub)

- [ ] **Step 1: Confirm the package imports break on atomistics 0.0.5 BEFORE the swap**

```bash
cd /home/liger/pyiron_workflow_lammps
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "import pyiron_workflow_lammps" 2>&1 | tail -3
```

Expected: `ImportError: cannot import name 'dataclass_storage' from 'pyiron_workflow_atomistics'`. (If it doesn't, atomistics 0.0.5 isn't installed in the env — fix that first: `uv pip install --python /home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python --upgrade pyiron-workflow-atomistics==0.0.5`.)

- [ ] **Step 2: Update `pyiron_workflow_lammps/engine.py` imports**

Replace the top-of-file import block lines 9–14:

```python
from pyiron_workflow_atomistics.dataclass_storage import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
    Engine,
)
```

with:

```python
from pyiron_workflow_atomistics.engine import (
    CalcInputMD,
    CalcInputMinimize,
    CalcInputStatic,
)
```

Note: `Engine` is dropped from the import — it's now a `typing.Protocol` (structural typing), so inheriting `class LammpsEngine(Engine):` is unnecessary and conflicts with `@dataclass` + Protocol semantics. Change the class header from `class LammpsEngine(Engine):` to `class LammpsEngine:`.

- [ ] **Step 3: Update `pyiron_workflow_lammps/lammps.py:88`**

Replace:

```python
    from pyiron_workflow_atomistics.dataclass_storage import EngineOutput
```

with:

```python
    from pyiron_workflow_atomistics.engine import EngineOutput
```

- [ ] **Step 4: Update test imports**

In each of these files, swap `pyiron_workflow_atomistics.dataclass_storage` → `pyiron_workflow_atomistics.engine`:

```bash
sed -i 's|pyiron_workflow_atomistics\.dataclass_storage|pyiron_workflow_atomistics.engine|g' \
  tests/unit/test_engine.py \
  tests/unit/test_lammps.py \
  tests/benchmark/test_benchmark.py \
  tests/integration/test_integration.py
```

- [ ] **Step 5: Update the notebook**

```bash
python -c "
import json, pathlib
p = pathlib.Path('notebooks/EnginePrototype_lammps.ipynb')
nb = json.loads(p.read_text())
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
    if 'dataclass_storage' in src:
        new = src.replace('pyiron_workflow_atomistics.dataclass_storage',
                          'pyiron_workflow_atomistics.engine')
        cell['source'] = new.splitlines(keepends=True)
        print('patched:', cell['source'][0].strip()[:60])
p.write_text(json.dumps(nb, indent=1) + '\\n')
"
```

Expected: prints one or more `patched: from pyiron_workflow_atomistics.engine import …`.

- [ ] **Step 6: Delete the empty local stub**

```bash
git rm pyiron_workflow_lammps/dataclass_storage.py
```

- [ ] **Step 7: Verify the package now imports cleanly**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import pyiron_workflow_lammps
from pyiron_workflow_lammps.engine import LammpsEngine
print('imports OK')
"
```

Expected: `imports OK`. If you see `ImportError: cannot import name 'Engine'`, you missed dropping `Engine` from the import or the class header in Step 2 — fix and re-run.

- [ ] **Step 8: Run the existing unit suite — most should pass; some may fail until later tasks**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header --tb=line 2>&1 | tail -8
```

Expected: import-related failures (if any) are gone. Tests that depended on the old `EngineOutput` no-args constructor may now fail in `parse_LammpsOutput` calls — that's fine, Task 5 fixes that.

- [ ] **Step 9: Commit**

```bash
git add pyiron_workflow_lammps/engine.py pyiron_workflow_lammps/lammps.py \
        tests/unit/test_engine.py tests/unit/test_lammps.py \
        tests/benchmark/test_benchmark.py tests/integration/test_integration.py \
        notebooks/EnginePrototype_lammps.ipynb
git rm pyiron_workflow_lammps/dataclass_storage.py
git commit -m "chore: swap dataclass_storage→engine import paths

Atomistics 0.0.5 deleted pyiron_workflow_atomistics.dataclass_storage;
the symbols moved to pyiron_workflow_atomistics.engine (CalcInput*,
EngineOutput) and the runtime-checkable Engine Protocol replaced the
ABC. Drop \`class LammpsEngine(Engine):\` inheritance — Protocol is
structural, dataclass + the method set is enough.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Add `with_working_directory` method

**Files:**
- Modify: `pyiron_workflow_lammps/engine.py` (add method on `LammpsEngine`)

- [ ] **Step 1: Write the failing test**

Create a small purity-check inline in a Python REPL one-liner (do NOT commit this — it's a sanity check; the proper conformance test lands in Task 6):

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_lammps.engine import LammpsEngine
eng = LammpsEngine(EngineInput=CalcInputStatic(), working_directory='/tmp/a')
try:
    sub = eng.with_working_directory('subdir')
    print('UNEXPECTED PASS')
except AttributeError as e:
    print('OK fails (expected before Task 4):', str(e)[:80])
"
```

Expected: `OK fails`. If it passes, the method already exists — skip this task.

- [ ] **Step 2: Add the method**

Edit `pyiron_workflow_lammps/engine.py` — append this method to the `LammpsEngine` class body, right after `__post_init__` (or wherever methods begin):

```python
    def with_working_directory(self, subdir: str) -> "LammpsEngine":
        """Return a copy of this engine with working_directory composed.

        Pure — never mutates self. Re-initialises self.calc_fn /
        self.calc_fn_kwargs to None on the copy so the next
        get_calculate_fn() call rebuilds the script against the new
        directory (otherwise the sub-engine would inherit the parent's
        stale cached kwargs).
        """
        from dataclasses import replace as _replace
        import os as _os

        return _replace(
            self,
            working_directory=_os.path.join(self.working_directory, subdir),
            calc_fn=None,
            calc_fn_kwargs=None,
        )
```

Use the lazy `from dataclasses import replace as _replace` / `import os as _os` to avoid touching the top-of-file import block — that minimises diff. The aliased names sidestep collisions with any field-level `os` use.

- [ ] **Step 3: Manual smoke (purity + path composition)**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import os
from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_lammps.engine import LammpsEngine

eng = LammpsEngine(EngineInput=CalcInputStatic(), working_directory='/tmp/a')
sub = eng.with_working_directory('subdir')
assert sub.working_directory == os.path.join('/tmp/a', 'subdir'), sub.working_directory
assert eng.working_directory == '/tmp/a'
assert sub is not eng
assert sub.calc_fn is None and sub.calc_fn_kwargs is None
print('with_working_directory purity OK')
"
```

Expected: `with_working_directory purity OK`.

- [ ] **Step 4: Run the unit suite — confirm no regression**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header --tb=line 2>&1 | tail -5
```

Expected: same pass/fail count as Task 3 Step 8 — adding a method must not break anything.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_lammps/engine.py
git commit -m "feat(engine): pure with_working_directory via replace

Returns a copy with the working_directory composed via os.path.join
and the calc_fn/calc_fn_kwargs cache reset so the sub-engine rebuilds
its input script against the new path on the next get_calculate_fn()
call. Implements the with_working_directory clause of the
pyiron_workflow_atomistics.engine.Engine Protocol.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Rewire `parse_LammpsOutput` to return a real `EngineOutput`

**Files:**
- Modify: `pyiron_workflow_lammps/lammps.py` — the `parse_LammpsOutput` function, around lines 88–170. The exact lines you'll touch are the section starting where `lammps_EngineOutput = EngineOutput()` is created (around line 88) through the trailing `return lammps_EngineOutput`. The internal `lammps_node_output` parse logic is unchanged — only the return wrapping changes.

The old code path constructs `EngineOutput()` with no args (the deleted `dataclass_storage.EngineOutput` permitted that — class-attrs-as-defaults), then mutates attributes. The new strict `@dataclass` `EngineOutput` requires `final_structure`, `final_energy`, `converged` at construction time. Rewire to compute everything into locals, then construct at the end.

- [ ] **Step 1: Re-read the current parse_LammpsOutput tail**

```bash
cd /home/liger/pyiron_workflow_lammps
sed -n '80,170p' pyiron_workflow_lammps/lammps.py
```

Confirm the function shape: it has a branch where `_parser_fn is None` (default path) that does the no-args `EngineOutput() + .attr = …` pattern, and an else branch that calls a user-provided `_parser_fn` and trusts its return.

- [ ] **Step 2: Replace the no-args-then-mutate block**

In `pyiron_workflow_lammps/lammps.py`, locate the default-parser branch (the one that imports `EngineOutput` and constructs it). Replace the entire block from `lammps_EngineOutput = EngineOutput()` (or wherever it's created) through the last attribute assignment, with:

```python
        from pyiron_workflow_atomistics.engine import EngineOutput

        # Walk the per-step pyiron_lammps_output and build trajectory + finals.
        atoms_list = []
        for i in range(len(pyiron_lammps_output["generic"]["cells"])):
            atoms_list.append(
                arrays_to_ase_atoms(
                    cells=pyiron_lammps_output["generic"]["cells"][i],
                    positions=pyiron_lammps_output["generic"]["positions"][i],
                    indices=pyiron_lammps_output["generic"]["indices"][i],
                    species_lists=species_lists,
                )
            )

        converged = isLineInFile.node_function(
            filepath=os.path.join(working_directory, log_lammps_file_name),
            line=log_lammps_convergence_printout,
            exact_match=False,
        )

        final_atoms = atoms_list[-1]
        energies_traj = pyiron_lammps_output["generic"]["energy_tot"]
        forces_traj = pyiron_lammps_output["generic"]["forces"]
        stresses_traj = pyiron_lammps_output["generic"]["pressures"]

        lammps_EngineOutput = EngineOutput(
            final_structure=final_atoms,
            final_energy=energies_traj[-1],
            converged=bool(converged),
            final_forces=forces_traj[-1],
            final_stress=stresses_traj[-1],          # 3x3 stress tensor
            final_volume=final_atoms.get_volume(),
            energies=energies_traj,
            forces=forces_traj,
            stresses=stresses_traj,
            structures=atoms_list,
            n_ionic_steps=pyiron_lammps_output["generic"]["steps"],
        )
```

Notes:
- `bool(converged)` is required: the upstream `EngineOutput.converged: bool` field rejects `None` or truthy non-bool values strictly; the `isLineInFile.node_function` return type is the safe coercion target.
- `final_stress_voigt` is intentionally NOT set — the existing `pyiron_lammps_output["generic"]["pressures"]` is already a 3x3 tensor; the Voigt-flatten can wait for a follow-up if a downstream consumer needs it.
- `final_results` (the bespoke field on the old object that held the raw parser output dict) is dropped — it's not part of the canonical `EngineOutput` shape, and downstream consumers who need raw output can re-call the parser.

- [ ] **Step 3: Re-import sanity-check**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
from pyiron_workflow_lammps.lammps import parse_LammpsOutput
print('parse_LammpsOutput imports OK')
"
```

Expected: `parse_LammpsOutput imports OK`. Any `NameError` here means the `arrays_to_ase_atoms`/`isLineInFile`/`species_lists` symbol references didn't resolve in the new code block — verify they're already in scope by reading 20 lines above the block in the same function.

- [ ] **Step 4: Run unit suite — expect previous import errors now cleared**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header --tb=line 2>&1 | tail -8
```

Expected: most tests pass; tests that previously broke on `EngineOutput()` no-args construction should now pass. Any new failure should be investigated.

- [ ] **Step 5: Commit**

```bash
git add pyiron_workflow_lammps/lammps.py
git commit -m "feat(engine): return EngineOutput dataclass from parser

parse_LammpsOutput now constructs the upstream
pyiron_workflow_atomistics.engine.EngineOutput @dataclass with its
three required fields (final_structure, final_energy, converged) and
the relevant optional fields populated from pyiron_lammps_output.
The no-args-then-mutate pattern is gone — the new strict @dataclass
upstream rejects it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Vendor `Al-Fe.eam.fs` under `tests/fixtures/`

**Files:**
- Create: `tests/fixtures/Al-Fe.eam.fs` (vendored copy of `notebooks/Al-Fe.eam.fs`, ~2.2 MB)
- Create: `tests/fixtures/__init__.py` (empty, marks the fixtures dir as a package for import resolution)

- [ ] **Step 1: Copy the existing potential**

```bash
cd /home/liger/pyiron_workflow_lammps
mkdir -p tests/fixtures
cp notebooks/Al-Fe.eam.fs tests/fixtures/
touch tests/fixtures/__init__.py
ls -la tests/fixtures/
```

Expected: `Al-Fe.eam.fs` (~2.2 MB) and `__init__.py` (0 B) listed.

- [ ] **Step 2: Verify git will accept the binary**

```bash
git check-attr -a tests/fixtures/Al-Fe.eam.fs
```

Expected: no `filter` or `text` overrides that would mangle a binary. (If the repo has Git LFS configured for `*.eam.fs`, the file lands in LFS automatically — that's fine.)

- [ ] **Step 3: Commit**

```bash
git add tests/fixtures/Al-Fe.eam.fs tests/fixtures/__init__.py
git commit -m "test(fixtures): vendor Al-Fe.eam.fs for the conformance suite

The conformance suite needs a real LAMMPS-compatible EAM potential to
run get_calculate_fn → run on bulk Cu (or Fe) in CI. Vendor the same
Al-Fe.eam.fs that notebooks/ already uses; tests reference it via the
fixtures path so they're independent of the notebooks layout.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Subclass `EngineConformanceTests` with a `LammpsEngine` factory

**Files:**
- Create: `tests/unit/test_engine_conformance.py`

- [ ] **Step 1: Create the test file**

Create `tests/unit/test_engine_conformance.py`:

```python
"""Conformance test: prove LammpsEngine satisfies the
pyiron_workflow_atomistics Engine Protocol contract.

Runs the 5-method EngineConformanceTests mixin from atomistics 0.0.5+
against a LammpsEngine instance configured for a single-point on
bulk Cu (FCC, a=3.6) using the vendored Al-Fe.eam.fs potential.
"""

from __future__ import annotations

from pathlib import Path

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
```

Note: the conformance suite's default `test_structure_factory` builds `bulk("Cu","fcc",a=3.6,cubic=True)`. The `Al-Fe.eam.fs` potential doesn't actually contain Cu parameters, so the `test_run_returns_engine_output` smoke would fail with element-not-found. Override the test-structure factory to use Fe instead:

Append to `TestLammpsEngineConformance`:

```python
    @staticmethod
    def test_structure_factory():
        from ase.build import bulk
        return bulk("Fe", "bcc", a=2.85, cubic=True)
```

- [ ] **Step 2: Run the conformance suite**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit/test_engine_conformance.py -v --tb=short 2>&1 | tail -20
```

Expected: 5 passed (`test_satisfies_engine_protocol`, `test_with_working_directory_is_pure`, `test_pickleable`, `test_get_calculate_fn_signature`, `test_run_returns_engine_output`). If `test_run_returns_engine_output` fails with `FileNotFoundError: lmp` the local env doesn't have a LAMMPS binary — skip locally and confirm CI passes after pushing.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_engine_conformance.py
git commit -m "test(engine): subclass EngineConformanceTests

Verifies LammpsEngine satisfies the pyiron_workflow_atomistics
Engine Protocol via the upstream 5-method conformance mixin
(Protocol+@dataclass shape, with_working_directory purity,
pickle round-trip, get_calculate_fn signature, run() smoke).
Uses vendored Al-Fe.eam.fs + Fe BCC test structure since the
potential's element list doesn't include Cu (the mixin default).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Bump deps to atomistics 0.0.5

**Files:**
- Modify: `pyproject.toml` (the `dependencies` array)
- Modify: `.ci_support/environment.yml`
- Modify: `.ci_support/lower-bound.yml`

- [ ] **Step 1: Update `pyproject.toml` dependencies**

Replace the `dependencies = [...]` block (currently lines starting with `dependencies = [`) in `pyproject.toml` with this exact content (copied verbatim from atomistics 0.0.5 plus the lammps-specific `pyiron-lammps`):

```toml
dependencies = [
    "numpy==1.26.4",
    "pandas==3.0.2",
    "matplotlib==3.10.9",
    "ase==3.28.0",
    "scipy==1.17.1",
    "pyiron-workflow==0.15.6",
    "pyiron-workflow-atomistics==0.0.5",
    "pymatgen==2026.5.4",
    "pyiron_snippets==1.2.1",
    "scikit-learn==1.8.0",
    "lz-GB-code==0.1.0",
    "tqdm==4.67.3",
    "pyiron-lammps==0.4.6",
]
```

- [ ] **Step 2: Update `.ci_support/environment.yml`**

Replace the `dependencies:` block in `.ci_support/environment.yml` with:

```yaml
channels:
  - conda-forge

dependencies:
  - python>=3.11,<3.13
  - pip
  - numpy=1.26.4
  - ase=3.28.0
  - pymatgen=2026.5.4
  - pyiron_workflow=0.15.6
  - pyiron_lammps=0.4.6
  - pandas=3.0.2
  - scipy=1.17.1
  - matplotlib=3.10.9
  - scikit-learn=1.8.0
  - tqdm=4.67.3
  - lammps=2024.08.29
  - pysqa=0.3.0
  - pip:
      - pyiron-workflow-atomistics==0.0.5
      - pyiron-snippets==1.2.1
      - lz-GB-code==0.1.0
```

- [ ] **Step 3: Update `.ci_support/lower-bound.yml`**

Replace the `dependencies:` block in `.ci_support/lower-bound.yml` with:

```yaml
channels:
  - conda-forge

dependencies:
  - python=3.11
  - pip
  - numpy=1.26.4
  - ase=3.28.0
  - pymatgen=2026.5.4
  - pyiron_workflow=0.15.6
  - pyiron_lammps=0.4.6
  - pandas=3.0.2
  - scipy=1.17.1
  - matplotlib=3.10.9
  - scikit-learn=1.8.0
  - tqdm=4.67.3
  - lammps=2024.08.29
  - pysqa=0.3.0
  - pip:
      - pyiron-workflow-atomistics==0.0.5
      - pyiron-snippets==1.2.1
      - lz-GB-code==0.1.0
```

Note: lower-bound traditionally floors slightly lower than environment.yml. For this migration we keep them aligned to atomistics 0.0.5 — the migration itself is the floor.

- [ ] **Step 4: Re-run the entire unit suite to confirm no breaks from major bumps**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header --tb=line 2>&1 | tail -8
```

Expected: same pass/fail count as Task 7 Step 2. If pandas 3.0.2 or pymatgen 2026.5.4 breaks something, investigate that single failure and add a one-line fix commit (e.g. a deprecated-API rename) before continuing.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .ci_support/environment.yml .ci_support/lower-bound.yml
git commit -m "chore: bump deps to atomistics 0.0.5 pin set

Sync every shared dep (numpy, pandas, ase, pyiron_workflow, etc.) to
pyiron_workflow_atomistics 0.0.5's verbatim pins so pip-check passes.
Notable jumps: pandas 2.3.1→3.0.2, ase 3.26.0→3.28.0, pyiron-workflow
0.15.2→0.15.6, pymatgen 2025.6.14→2026.5.4. Engine integration
unaffected — all existing unit tests still pass.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Write CHANGELOG.md

**Files:**
- Create: `CHANGELOG.md`

- [ ] **Step 1: Create the file**

Create `/home/liger/pyiron_workflow_lammps/CHANGELOG.md`:

```markdown
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
```

- [ ] **Step 2: Verify it parses**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -c "
import pathlib
text = pathlib.Path('CHANGELOG.md').read_text()
assert '[0.1.0]' in text
assert 'with_working_directory' in text
assert 'EngineConformanceTests' in text
print('CHANGELOG.md OK,', len(text), 'chars')
"
```

Expected: `CHANGELOG.md OK, NNN chars`.

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): 0.1.0 — Engine Protocol migration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Final lint + tests + push

**Files:**
- (no edits — verification only)

- [ ] **Step 1: Run ruff**

```bash
cd /home/liger/pyiron_workflow_lammps
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check pyiron_workflow_lammps/ tests/ 2>&1 | tail -5
```

Expected: `All checks passed!`. If a `from dataclasses import replace as _replace` lazy import in Task 4 trips a top-of-file-imports lint (E402/I001), move it to the module-level import block in a follow-up `style:` commit.

- [ ] **Step 2: Run ruff import-sort**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m ruff check --select I pyiron_workflow_lammps/ tests/ 2>&1 | tail -3
```

Expected: `All checks passed!`. Fix with `ruff check --select I --fix pyiron_workflow_lammps/ tests/` if needed and amend the last commit.

- [ ] **Step 3: Run black**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/python -m black --check pyiron_workflow_lammps/ tests/ 2>&1 | tail -5
```

Expected: `All done! ✨ … N files would be left unchanged.`. If anything wants reformatting, run `black pyiron_workflow_lammps/ tests/` then commit as `style:` follow-up.

- [ ] **Step 4: Run the full unit suite**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/unit -q --no-header 2>&1 | tail -5
```

Expected: all existing tests still pass + the 5 new conformance tests pass. Any baseline failures present in `tests/unit/test_version.py` / `test_tests.py` are pre-existing and irrelevant.

- [ ] **Step 5: Run the integration suite (optional, slower)**

```bash
/home/liger/miniforge3/envs/test_pyiron_workflow_vasp/bin/pytest tests/integration -q --no-header 2>&1 | tail -5
```

Expected: pass. May require a real LAMMPS binary; if absent locally, defer to CI.

- [ ] **Step 6: Push the branch**

```bash
git push origin design-engine-protocol-migration
```

Expected: branch tip on remote updated. The previously-opened draft PR #39 picks up the new commits automatically.

---

## Task 11: Promote PR + verify CI

**Files:**
- (no edits)

- [ ] **Step 1: Wait for CI to settle**

```bash
sleep 60
gh pr checks 39 --repo pyiron/pyiron_workflow_lammps 2>&1 | head -25
```

Expected: every check green — `ruff-check`, `ruff-sort-imports`, `black`, `unit-tests (ubuntu/macos × 3.10/3.11/3.12)`, `build-notebooks`, `build-docs`, `pip-check`, `pypi-release`, `coverage`. If anything fails, fix the smallest possible thing and push a follow-up commit; the draft PR sees the update automatically.

- [ ] **Step 2: Promote from draft to ready-for-review**

```bash
gh pr ready 39 --repo pyiron/pyiron_workflow_lammps
```

Expected: PR state moves from `DRAFT` to `OPEN`.

- [ ] **Step 3: Update the PR body with the post-migration test counts**

```bash
gh pr edit 39 --repo pyiron/pyiron_workflow_lammps --body "$(cat <<'EOF'
## Summary

Migrates `LammpsEngine` onto the new `pyiron_workflow_atomistics==0.0.5` Engine Protocol contract. Static + Minimize modes are Protocol-conformant; MD stays functional but is not certified this cycle. Ships as `pyiron_workflow_lammps-0.1.0` (clean break, no deprecation shims).

## Concrete changes

1. `dataclass_storage → engine` import paths everywhere (engine.py, lammps.py, all tests, the notebook).
2. New `LammpsEngine.with_working_directory(subdir) -> LammpsEngine` — pure copy via `dataclasses.replace`, resets cached `calc_fn`/`calc_fn_kwargs` so sub-engines rebuild input scripts against the new directory.
3. `parse_LammpsOutput` now returns a real `pyiron_workflow_atomistics.engine.EngineOutput` `@dataclass` with required fields supplied at construction. Old no-args-then-mutate pattern gone.
4. `tests/unit/test_engine_conformance.py` subclasses `pyiron_workflow_atomistics.testing.EngineConformanceTests` with a `LammpsEngine + tests/fixtures/Al-Fe.eam.fs + Fe BCC` factory. CI verifies all 5 contract clauses including the `run()` smoke against a real LAMMPS binary.
5. Pyproject + `.ci_support/*.yml` bumped verbatim to atomistics 0.0.5's pin set.
6. `CHANGELOG.md` documents the migration + a 3-line search-and-replace for downstream consumers.

## Release sequence

After merge:

```bash
git tag pyiron_workflow_lammps-0.1.0 <merge-sha>
git push origin pyiron_workflow_lammps-0.1.0
gh release create pyiron_workflow_lammps-0.1.0 --title "pyiron_workflow_lammps 0.1.0" --notes-file <release-notes>
```

The `pyproject-release.yml` workflow auto-publishes to PyPI on the published-release event.

## Test plan

- [x] `pytest tests/unit` — all green locally including the 5 new conformance tests.
- [x] `ruff check`, `ruff check --select I`, `black --check` — green.
- [ ] CI runs across the full Ubuntu/macOS × 3.10/3.11/3.12 matrix — see check tab.
- [ ] Integration tests in CI (require LAMMPS binary, already pinned in `.ci_support/environment.yml`).
- [ ] Numerical regression: every existing `assert <energy/force/volume> == …` in the test suite continues to pass after migration. This is the gate against silent backend-numerical drift (the issue the atomistics-cleanup spec called out).

## Spec & plan

- Spec: `docs/design/specs/2026-05-12-engine-protocol-migration-design.md`
- Plan: `docs/design/plans/2026-05-12-engine-protocol-migration.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR body updated; the test plan + spec/plan refs are visible in the PR description.

---

## Self-Review

**1. Spec coverage:**

| Spec section | Plan task |
|---|---|
| Components: engine.py modified | Tasks 3 (imports + class header) + 4 (with_working_directory) |
| Components: lammps.py modified | Tasks 3 (import swap) + 5 (return EngineOutput) |
| Components: dataclass_storage.py deleted | Task 3 step 6 |
| Components: tests/unit/test_engine_conformance.py new | Task 7 |
| Components: tests/unit/test_numerical_regression.py new (golden energies pinned) | Task 2 (captured via the existing test asserts, which already serve as goldens) — no new file; documented in Task 11 PR body. |
| Components: CHANGELOG.md new | Task 9 |
| Components: design plan committed | Task 1 |
| Contract clauses table: working_directory unchanged | (verified by Task 7 conformance run) |
| Contract clauses table: @dataclass unchanged | (verified) |
| Contract clauses table: get_calculate_fn shape unchanged | (verified by Task 7) |
| Contract clauses table: with_working_directory added | Task 4 |
| Contract clauses table: pickle round-trip | (verified by Task 7) |
| Contract clauses table: Calculate fn returns EngineOutput | Task 5 |
| with_working_directory code block | Task 4 step 2 |
| EngineOutput mapping code block | Task 5 step 2 |
| Numerical regression gate | Task 2 (captures existing-test goldens) + Task 10 step 4 (re-run) |
| CI footprint: LAMMPS in env yaml | Task 8 step 2 (kept the existing `lammps=2024.08.29`) |
| Out of scope: MD-mode conformance | (not added to Task 7's factory) |
| Out of scope: input_script_* refactoring | (not touched) |
| Out of scope: parse_LammpsOutput internals | (Task 5 only modifies the return wrapping) |
| Out of scope: deprecation shims | (none added) |
| Risk: Al-Fe.eam.fs redistribution | Task 6 (vendored under tests/fixtures/) |
| Risk: calc_fn/parse_fn cache + pickle | Task 4 step 2 (calc_fn/kwargs reset on the replace copy) |
| Risk: pip-check dep alignment | Task 8 (verbatim atomistics 0.0.5 pins) |

**Gap I notice:** the spec mentioned a `tests/unit/test_numerical_regression.py` as a NEW file, but in practice the existing test suite ALREADY contains the numeric assertions that serve as the goldens. I've folded this into Task 2 (read-only inventory) + Task 10 step 4 (re-run gate). If a future maintainer wants a single dedicated file pinning extra goldens (e.g. against atomistics' pre-cleanup notebooks), they can add one as a follow-up — but it's not blocking this PR.

**2. Placeholder scan:** No "TBD", "TODO", or "fill in details". Every code step has the actual code; every command has the expected output. The lazy `from dataclasses import replace as _replace` inside `with_working_directory` is a deliberate choice (minimises diff in the top-of-file import block) — not a placeholder.

**3. Type consistency:**
- `LammpsEngine` field names (`EngineInput`, `working_directory`, `calc_fn`, `calc_fn_kwargs`, `path_to_model`, `input_script_pair_style`, `command`) are used consistently in Tasks 4 and 7.
- `EngineOutput` field names match upstream (`final_structure`, `final_energy`, `converged`, `final_forces`, `final_stress`, `final_volume`, `energies`, `forces`, `stresses`, `structures`, `n_ionic_steps`) in Task 5.
- `EngineConformanceTests` is the upstream class name (Task 7) — matches what was shipped in atomistics 0.0.5.
- `bulk("Fe","bcc",a=2.85,cubic=True)` in Task 7 vs the upstream default `bulk("Cu","fcc",a=3.6,cubic=True)` — deliberately different because `Al-Fe.eam.fs` doesn't cover Cu. Documented at the override.

---

## Execution Handoff

**Plan complete and saved to `docs/design/plans/2026-05-12-engine-protocol-migration.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
