# pyiron_workflow_lammps

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)

A LAMMPS engine for [`pyiron_workflow`](https://github.com/pyiron/pyiron_workflow) /
[`pyiron_workflow_atomistics`](https://github.com/pyiron/pyiron_workflow_atomistics).

It wraps a LAMMPS executable behind the atomistics `Engine` protocol so that
static, minimisation and MD calculations can be driven from a `pyiron_workflow`
graph the same way as any other ASE-style calculator — write structure, write
input, shell out, parse, return an `EngineOutput`.

## What it provides

- **`LammpsEngine`** (`pyiron_workflow_lammps.engine`) — a `@dataclass` that
  satisfies the `pyiron_workflow_atomistics.engine.Engine` protocol. The
  calculation mode (`static` / `minimize` / `md`) is inferred from which
  `CalcInput*` dataclass is passed:
  - `CalcInputStatic` → single-point energy
  - `CalcInputMinimize` → `min_style cg` minimisation, optional cell relax
    via `fix box/relax`
  - `CalcInputMD` → MD with `NVE`, `NVT` (nose-hoover, berendsen, langevin,
    andersen, temp/rescale, temp/csvr) or `NPT` (nose-hoover only)
- **`lammps_job`** (`pyiron_workflow_lammps.lammps`) — `pwf.as_macro_node`
  composing `create_WorkingDirectory → write_LammpsStructure →
  write_LammpsInput → shell → parse_LammpsOutput`. Used internally by
  `LammpsEngine.get_calculate_fn`, but also runnable standalone.
- **`parse_LammpsOutput`** — reads `log.lammps` + `dump.out` via
  `pyiron_lammps` and returns the canonical
  `pyiron_workflow_atomistics.engine.EngineOutput` with trajectories,
  per-step energies/forces/stresses, final structure, and a
  `converged` flag derived from the LAMMPS log.
- **`generic.py`** helpers — `shell`, `create_WorkingDirectory`,
  `compress_directory`, `delete_files_recursively`, `submit_to_slurm`,
  `isLineInFile`. Most are wrapped as `pwf` function nodes.

## Installation

Linux or macOS, Python 3.11–3.12. Windows users should use WSL2
(`lammps=2024.08.29` from `conda-forge` has no Windows build).

### Via conda-forge (recommended; bundles a LAMMPS binary)

```bash
mamba env create -f .ci_support/environment.yml
mamba activate <env>
pip install --no-build-isolation -e .
```

The `--no-build-isolation` flag is required because `versioneer` imports the
package at build time and needs runtime dependencies present in the build env.

### From PyPI (you supply your own LAMMPS)

```bash
pip install pyiron_workflow_lammps
```

You then need a LAMMPS executable on `PATH` (or referenced via
`Engine.command`) that has the pair styles you intend to use compiled in
(e.g. `MANYBODY` for `eam/fs`, `ML-PACE` / `ML-GRACE` for ACE / GRACE).

## Quick start

```python
from ase.build import bulk
from pyiron_workflow import Workflow
from pyiron_workflow_atomistics.engine import CalcInputMinimize, calculate
from pyiron_workflow_lammps.engine import LammpsEngine

structure = bulk("Fe", cubic=True) * [5, 5, 5]

calc_input = CalcInputMinimize(
    energy_convergence_tolerance=1e-5,
    force_convergence_tolerance=1e-5,
    max_iterations=10_000,
)

engine = LammpsEngine(EngineInput=calc_input)
engine.working_directory = "Fe_minimize"
engine.command = "lmp -in in.lmp -log minimize.log"
engine.lammps_log_filepath = "minimize.log"
engine.input_script_pair_style = "eam/fs"
engine.path_to_model = "/path/to/Fe.eam.fs"

wf = Workflow("Fe_min", delete_existing_savefiles=True)
wf.calc = calculate(structure=structure, engine=engine)
wf.run()

out = wf.calc.outputs.engine_output.value
print(out.final_energy, out.converged, out.n_ionic_steps)
```

A more complete walkthrough — minimise → NVT(langevin) → NPT(nose-hoover) —
lives in [`notebooks/example.ipynb`](notebooks/example.ipynb), committed
with the executed cell outputs (energy-trace plots inline) so the results
are visible directly on GitHub. The bundled `notebooks/Al-Fe.eam.fs`
potential lets it run end-to-end against any LAMMPS build with the
`MANYBODY` package.

## Engine knobs worth knowing

`LammpsEngine` exposes the LAMMPS input as field defaults you override on the
instance — no string templating required:

| Field | Default | Purpose |
|---|---|---|
| `command` | `"lmp -in in.lmp -log log.lammps"` | shell call to LAMMPS |
| `working_directory` | `os.getcwd()` | where input/output go |
| `input_script_units` | `"metal"` | LAMMPS `units` |
| `input_script_pair_style` | `"grace"` | LAMMPS `pair_style` |
| `path_to_model` | `"/path/to/model"` | filled into `pair_coeff * *` |
| `input_script_min_style` | `"cg"` | minimiser |
| `input_script_relax_type` | `"iso"` | `box/relax` mode (iso/aniso/tri) |
| `input_script_relax_pressure` | `1e4` | bar, target for cell relax |
| `max_evaluations` | `10000` | LAMMPS `minimize` 4th arg |
| `max_iterations` | `None` | overrides `EngineInput.max_iterations` if set |

`LammpsEngine.with_working_directory(subdir)` returns a pure copy with the
working directory composed and the cached `calc_fn`/`calc_fn_kwargs` reset —
use it for sub-engines (replaces the pre-0.0.5 `duplicate_engine` helper).

## Tests

```bash
python -m pytest tests/unit/ -v                    # unit tests, no LAMMPS needed
python -m pytest tests/unit/ --cov=pyiron_workflow_lammps
```

`tests/unit/test_engine_conformance.py` runs a single-point smoke against a
real LAMMPS binary using `tests/fixtures/Al-Fe.eam.fs`; it skips when no
`lmp` is on `PATH`. See `tests/README.md` for details.

## Compatibility

Tracks `pyiron_workflow_atomistics==0.0.6` and `pyiron-workflow==0.15.6`.
Pin set is duplicated in `pyproject.toml`, `.ci_support/environment.yml`
and `.binder/environment.yml`. See [`CHANGELOG.md`](CHANGELOG.md) for the
0.0.4 → 0.1.0 migration notes (engine protocol change, dropped Windows).

## License

BSD 3-Clause — see [`LICENSE`](LICENSE).
