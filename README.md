# pyiron_workflow_lammps

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)

A LAMMPS engine for [`pyiron_workflow`](https://github.com/pyiron/pyiron_workflow) /
[`pyiron_workflow_atomistics`](https://github.com/pyiron/pyiron_workflow_atomistics).

It wraps a LAMMPS executable behind the atomistics `Engine` protocol, so
static, minimisation and MD calculations are run through the same
`calculate(structure=..., engine=...)` node as any other engine: write the
structure, write the input, run LAMMPS, parse, return an `EngineOutput`.

The nodes are [`flowrep`](https://github.com/pyiron/flowrep) functions
(`@fr.atomic` / `@fr.workflow`), which `pyiron_workflow` >= 0.19 executes.
They stay plain Python callables: call one directly to get its return value,
or run it as a node with `pyiron_workflow.node(fn).run(**inputs)` /
`pyiron_workflow.run(fn, ...)` to get a `Run` record whose results are under
`.outputs`.

## What it provides

- **`LammpsEngine`** (`pyiron_workflow_lammps.engine`): a `@dataclass` that
  satisfies the `pyiron_workflow_atomistics.engine.Engine` protocol. The
  calculation mode (`static` / `minimize` / `md`) is inferred from the
  `CalcInput*` dataclass passed as `EngineInput`:
  - `CalcInputStatic`: single-point energy (LAMMPS `run 0`)
  - `CalcInputMinimize`: `min_style cg` minimisation; with `relax_cell=True`
    also the cell, via `fix box/relax` towards `input_script_relax_pressure`
    (default 0, i.e. a zero-pressure relaxation)
  - `CalcInputMD`: MD with `NVE`, `NVT` (nose-hoover, berendsen, langevin,
    andersen, temp/rescale, temp/csvr) or `NPT` (nose-hoover only; requires
    `pressure`, in Pa, else `ValueError`). Times in `CalcInputMD` are
    femtoseconds and are converted to the LAMMPS `units` time unit. See
    "Thermostats" below.

  The engine is stateless: `get_calculate_fn(structure)` builds the input
  script for each structure and never mutates the engine, so one engine (and
  its `with_working_directory(...)` copies) can be reused across structures
  and threads.
- **`lammps_job`** (`pyiron_workflow_lammps.lammps`): a `flowrep` workflow
  composing `create_WorkingDirectory -> write_LammpsStructure ->
  write_LammpsInput -> shell -> parse_LammpsOutput`. `LammpsEngine` uses it
  internally; it can also be run on its own.
- **`parse_LammpsOutput`**: reads `log.lammps` and `dump.out` with
  [`lammpsparser`](https://github.com/pyiron/lammpsparser) and returns the
  `pyiron_workflow_atomistics.engine.EngineOutput` (conventions below).
- **`generic.py`** helpers as `flowrep` nodes: `shell`,
  `create_WorkingDirectory`, `isLineInFile`, `delete_files_recursively`,
  `compress_directory`, `remove_dir`.

## Installation

Linux or macOS, Python 3.12 to 3.14. Windows users should use WSL2
(conda-forge has no Windows build of `lammps`).

### From PyPI (you supply LAMMPS)

```bash
pip install pyiron_workflow_lammps
```

You need a LAMMPS executable, by default `lmp` on `PATH` (see the
`command` field), built with the pair styles you use: e.g. `MANYBODY` for
`eam/fs`, `ML-PACE` for ACE / GRACE. conda-forge's `lammps` package
provides one:

```bash
conda install -c conda-forge lammps
```

### Development install

```bash
conda env create -n pwl -f .ci_support/environment.yml   # includes lammps
conda activate pwl
pip install -e .
```

`.ci_support/environment.yml` carries the exact versions CI tests with (the
`==` pins in `pyproject.toml`); the published package declares ranges
around them.

## Quick start

`LammpsEngine`'s default `input_script_pair_style` is `"grace"`; for any other
potential set `input_script_pair_style` and `path_to_model` (the file is
written into `pair_coeff * * <path_to_model> <elements>`).

```python
from ase.build import bulk
from pyiron_workflow_atomistics.engine import CalcInputMinimize, calculate

from pyiron_workflow_lammps.engine import LammpsEngine

structure = bulk("Fe", cubic=True) * [3, 3, 3]
structure.rattle(0.05, seed=0)

engine = LammpsEngine(
    EngineInput=CalcInputMinimize(
        energy_convergence_tolerance=1e-5,
        force_convergence_tolerance=1e-5,
    ),
    working_directory="Fe_minimize",
    command="lmp -in in.lmp -log minimize.log",
    lammps_log_filepath="minimize.log",
    input_script_pair_style="eam/fs",
    path_to_model="/path/to/Fe.eam.fs",
)

out = calculate(structure=structure, engine=engine)
print(out.final_energy, out.converged, out.n_ionic_steps)
```

The same node run through `pyiron_workflow`, in a sub-directory:

```python
import pyiron_workflow as pwf

run = pwf.node(calculate).run(
    structure=structure, engine=engine.with_working_directory("as_node")
)
print(run.status, run.outputs.engine_output.final_energy)
```

A longer walkthrough (static, minimisation, a minimisation capped by
`max_iterations`, NVT with a Langevin thermostat, NPT with Nose-Hoover) is in
[`notebooks/example.ipynb`](notebooks/example.ipynb), committed with its
executed outputs. It uses the bundled `notebooks/Al-Fe.eam.fs` potential, so
run it from `notebooks/`; it needs a LAMMPS build with the `MANYBODY` package
and `matplotlib` for the plots (`.ci_support/environment-notebooks.yml`).

## Output conventions

`parse_LammpsOutput` fills the `EngineOutput` as follows:

| Field | Content |
|---|---|
| `final_energy`, `energies` | Potential energy in eV (lammpsparser `energy_pot`), i.e. the energy consistent with the forces. For MD this excludes the kinetic energy. |
| `final_forces`, `forces` | Forces in eV/Å, in the frame of the input structure. |
| `final_stress`, `stresses` | The LAMMPS pressure tensor from the thermo columns `pxx`…`pzz` of the log, converted by lammpsparser's unit converter and rotated into the input structure's frame: 3x3, in **GPa, compression positive**. This is not ASE's convention (eV/Å³, tension positive): ASE's `atoms.get_stress(voigt=False)` corresponds to `-final_stress * ase.units.GPa`. `final_stress_voigt` is not set. |
| `final_structure`, `structures` | `ase.Atoms` per dump frame, with velocities attached when dumped, in the input structure's frame and with its lattice vectors (see "Rotated cells"). |
| `final_volume` | Volume of the final cell in Å³. |
| `n_ionic_steps` | The last LAMMPS step number, as an `int`. |
| `converged` | `True` when the log contains `lammps_log_convergence_printout` (default `"Total wall time:"`, i.e. LAMMPS finished) **and**, for a minimisation, the last `Stopping criterion =` in the log is `energy tolerance`, `force tolerance` or `forces are zero`. A minimisation stopped by `max iterations`, `max force evaluations`, a failed line search etc. gives `False` (its last frame is still returned). Static (`run 0`) and MD runs have no stopping criterion and only need the first condition. |

The dump is written on the thermo output steps, including step 0 and the
last step of the run: every `n_print` steps for MD, every
`input_script_thermo_every` steps for a minimisation. So
`len(structures) == len(energies)`, and `final_structure` / `final_forces`
come from the same step as `final_energy`, also when a minimisation stops
between two thermo intervals. `n_print=0` (LAMMPS `thermo 0`) gives the first
and the last step only.

### Rotated cells

LAMMPS needs a lower-triangular cell with bounded tilt factors, so cells that
are not (primitive bcc/fcc cells and their supercells, sheared or otherwise
general triclinic cells) are rotated on the way in, and LAMMPS may pick
different, shorter lattice vectors for them. `lammps_job` passes the input
structure to `parse_LammpsOutput`, which rotates positions, forces and
pressure tensors back and restores the input cell's lattice vectors, so
`final_structure.cell` is the input cell (for a fixed-cell run) and forces and
stresses can be compared with any other calculator on the input structure.
`parse_LammpsOutput` called on its own without `structure` reports LAMMPS'
own frame.

### Minimisation tolerances

`CalcInputMinimize`'s tolerances are passed to LAMMPS `minimize` unchanged:

- `energy_convergence_tolerance` is LAMMPS' `etol`, which is **relative and
  dimensionless** (the energy change between iterations divided by the
  energy magnitude), not an absolute energy in eV;
- `force_convergence_tolerance` is LAMMPS' `ftol`, a bound on the 2-norm of
  the global force vector in eV/Å, not a per-atom maximum force.

`converged=True` therefore means LAMMPS met its own tolerances as defined
above. Set `energy_convergence_tolerance=0.0` to stop on the force criterion
alone.

### Thermostats

| `thermostat` | LAMMPS fixes |
|---|---|
| `nose-hoover` | `fix nvt` |
| `langevin` | `fix langevin` + `fix nve` |
| `andersen` | `fix langevin` + `fix nve`: **implemented as a Langevin thermostat**, not with Andersen's stochastic collisions; identical to `langevin` |
| `berendsen` | `fix temp/berendsen` + `fix nve` |
| `temp/rescale` | `fix temp/rescale <n_print> T T 0.0 1.0` (full rescale every `n_print` steps) + `fix nve` |
| `temp/csvr` | `fix temp/csvr` (with the seed) + `fix nve` |

`temp/rescale` and `temp/csvr` work with the engine although
`CalcInputMD.thermostat`'s type hint lists only the first four.

### Run behaviour

- **A failed LAMMPS run raises.** If the command exits with a non-zero return
  code, `parse_LammpsOutput` raises a `RuntimeError` carrying the tail of
  LAMMPS' stdout and stderr, instead of parsing whatever files an earlier run
  left in the working directory.
- **Static** calculations run `run 0` (one energy/force evaluation), not a
  zero-step minimisation, so `converged` is `True` for them.
- **MD seeds:** `CalcInputMD(seed=None)` draws a fresh random seed in
  [1, 900000000] for every input script (non-deterministic, as `CalcInputMD`
  defines it; 900000000 is the largest seed the Marsaglia generator behind
  `fix langevin` accepts); pass an int for reproducible velocities. The
  same seed is used for `velocity create` and the stochastic thermostats. `initial_temperature=0` is respected: no
  `velocity ... create` line is written (atoms start at rest) instead of
  falling back to `temperature`.
- **File names:** non-default `input_script_read_data_file` and
  `input_script_dump_filename` are used consistently for writing the
  structure, the input script and parsing.
- **Working directories** are created with `os.makedirs(..., exist_ok=True)`,
  so nodes running concurrently can create the same or nested directories.

## Engine fields worth knowing

`LammpsEngine` exposes the LAMMPS input as dataclass fields, set as keyword
arguments or attributes:

| Field | Default | Purpose |
|---|---|---|
| `command` | `"lmp -in in.lmp -log log.lammps"` | shell command that runs LAMMPS |
| `lammps_log_filepath` | `"log.lammps"` | log file to parse; must match `-log` in `command` |
| `working_directory` | `os.getcwd()` | where input and output files go |
| `input_script_units` | `"metal"` | LAMMPS `units` |
| `input_script_pair_style` | `"grace"` | LAMMPS `pair_style` |
| `path_to_model` | `"/path/to/model"` | potential file used in `pair_coeff * *` |
| `potential_elements` | `None` | element order for `pair_coeff`; `None` means the structure's elements in order of first occurrence |
| `input_script_min_style` | `"cg"` | minimiser |
| `input_script_relax_type` | `"iso"` | `box/relax` mode (iso/aniso/tri) |
| `input_script_relax_pressure` | `0.0` | `fix box/relax` target pressure for `relax_cell=True`, in the LAMMPS pressure unit (bar for `metal`) |
| `input_script_read_data_file` | `"lammps.data"` | structure file written and read with `read_data` |
| `input_script_dump_filename` | `"dump.out"` | trajectory dump file |
| `input_script_thermo_every` | `10` | thermo and dump interval of a minimisation (MD uses `n_print`) |
| `input_script_dump_every` | `10` | interval on the initial `dump` line only; the dump then follows the thermo steps, so it does not change the frames |
| `max_evaluations` | `10000` | 4th argument of LAMMPS `minimize` |
| `max_iterations` | `None` | overrides `EngineInput.max_iterations` if set |
| `calc_fn`, `calc_fn_kwargs` | `None` | custom calculation function (returned by `get_calculate_fn` with a copy of the kwargs) |
| `parse_fn`, `parse_fn_kwargs` | `None` | custom output parser, passed to `parse_LammpsOutput` |

`LammpsEngine.with_working_directory(subdir)` returns a copy with the working
directory joined with `subdir`; a custom `calc_fn` or `parse_fn` carries over.

### Cell relaxation target pressure

`CalcInputMinimize(relax_cell=True)` adds
`fix 1 all box/relax <input_script_relax_type> <input_script_relax_pressure>`.
The default `input_script_relax_pressure=0.0` relaxes to zero pressure. The
value is in the LAMMPS pressure unit (bar under `units metal`, compression
positive); for example, to relax under 1 GPa of hydrostatic pressure:

```text
LammpsEngine(EngineInput=CalcInputMinimize(relax_cell=True), input_script_relax_pressure=1e4, ...)
```

0.1.0 defaulted to `1e4` bar (1 GPa); pass `input_script_relax_pressure=1e4`
to reproduce its cell relaxations.

## Tests

```bash
pip install -e ".[test]"
python -m pytest tests
```

Tests that need LAMMPS use `lmp` from `PATH` (or the `LAMMPS_COMMAND`
environment variable) and are skipped without it. See `tests/README.md`.

## Compatibility

Requires `pyiron_workflow` >= 0.19 (the `flowrep` node API),
`pyiron_workflow_atomistics` >= 0.2.2 and `lammpsparser`. The exact versions
CI runs against are the `==` pins in `pyproject.toml`, mirrored in
`.ci_support/environment.yml`; the release workflow publishes them as ranges
with floors from `.ci_support/lower-bound.yml` and caps from
`.ci_support/upper-bound.yml`. See [`CHANGELOG.md`](CHANGELOG.md) for the
breaking changes in 0.2.0.

## License

BSD 3-Clause, see [`LICENSE`](LICENSE).
