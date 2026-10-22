# Changelog

All notable changes to `pyiron_workflow_lammps` are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning: PEP 440 via `versioneer`.

## [0.2.0] — unreleased

Ports the package to `pyiron_workflow` >= 0.19, whose node API is
[`flowrep`](https://github.com/pyiron/flowrep), and releases the fixes merged
since 0.1.0.

### Changed (breaking)

- **`pyiron_workflow` >= 0.19 / `flowrep`.** The nodes in `generic.py` and
  `lammps.py` are `@fr.atomic` functions instead of
  `Workflow.wrap.as_function_node` / `pwf.as_function_node`, and `lammps_job`
  is an `@fr.workflow` instead of an `as_macro_node`. Node names are
  unchanged (`shell`, `isLineInFile`, `create_WorkingDirectory`,
  `delete_files_recursively`, `compress_directory`, `remove_dir`,
  `write_LammpsStructure`, `write_LammpsInput`, `parse_LammpsOutput`,
  `lammps_job`). The decorated functions are plain callables: call them
  directly for the return value, or run them as nodes with
  `pyiron_workflow.node(fn).run(**inputs)` / `pyiron_workflow.run(...)` and
  read `run.outputs.<label>`. `.node_function`, `Workflow(...)` attribute
  assignment (`wf.calc = calculate(...)`) and `.outputs.<x>.value` no longer
  exist. Execution order inside `lammps_job` comes from data edges: `shell`
  and `parse_LammpsOutput` take an `after` ordering token.
- **`final_energy` / `energies` are the potential energy** (lammpsparser
  `energy_pot`), the energy consistent with the forces, instead of `etotal`,
  which includes the kinetic energy in MD. Static and minimisation results
  are unchanged.
- **Cell relaxations target zero pressure.** `LammpsEngine.input_script_relax_pressure`
  defaults to `0.0` instead of `1e4`, which is 10 000 bar = 1 GPa of
  hydrostatic compression under `units metal`, so `CalcInputMinimize(relax_cell=True)`
  now relaxes to zero pressure like ASE. Pass `input_script_relax_pressure=1e4`
  for the 0.1.0 behaviour.
- **`n_ionic_steps` is an `int`**: the last LAMMPS step number, instead of the
  array of thermo steps.
- **`LammpsEngine` is stateless.** `get_calculate_fn(structure)` builds the
  input script per structure and no longer caches it on the engine or
  mutates `calc_fn`, `calc_fn_kwargs` or `potential_elements`, so one engine
  and its copies can be reused across structures and threads (and still pass
  `pyiron_workflow_atomistics`' calphy default-field validation after a run).
  A user-set `potential_elements` is honoured instead of being overwritten.
  A custom `calc_fn` is returned with a copy of `calc_fn_kwargs`.
- **Default `command`** of `lammps_job` / `lammps_calculator_fn` is the
  engine's `"lmp -in in.lmp -log log.lammps"` instead of a site-specific
  `mpirun` path.
- **Python 3.11 to 3.14** (`requires-python = ">=3.11, <3.15"`); 3.10 is no
  longer listed.
- **Parser: `lammpsparser`** replaces `pyiron_lammps` (#61; `pyiron-lammps`
  0.5.4 cannot be installed together with `pyiron_workflow_atomistics`
  >= 0.2.2).
- **`converged` means converged.** It still requires
  `lammps_log_convergence_printout` (`"Total wall time:"`) in the log, and
  now also, for a minimisation, that the last `Stopping criterion =` line is
  `energy tolerance`, `force tolerance` or `forces are zero`
  (`lammps.CONVERGED_STOPPING_CRITERIA`). A minimisation stopped by
  `max iterations` / `max force evaluations` / a failed line search reports
  `converged=False`; before it was `True` whenever LAMMPS exited normally.
  Consumers that filter on `converged` see the difference: e.g.
  `pyiron_workflow_atomistics`'
  `extract_outputs_from_EngineOutputs(only_converged=True)` and workflows
  built on it now drop (or raise on)
  minimisations capped by `max_iterations` / `max_evaluations` that they
  used to accept. `converged` means LAMMPS met *its own* tolerances:
  `CalcInputMinimize.energy_convergence_tolerance` is passed unchanged as
  LAMMPS' `etol`, which is relative (dimensionless: the energy change between
  iterations divided by the energy magnitude), not an absolute energy in eV,
  and `force_convergence_tolerance` is LAMMPS' `ftol`, a bound on the 2-norm
  of the global force vector (eV/Å), not a per-atom maximum force.
- **Static calculations run `run 0`** instead of `min_style ...` +
  `minimize 0 0 0 0` (which LAMMPS reports as stopped on "max iterations").
  The energy is unchanged.
- **Trajectory frames are the thermo steps.** MD dumps every `n_print` steps
  (it dumped every `input_script_dump_every` = 10 while thermo followed
  `n_print`), a minimisation every `input_script_thermo_every` steps (its
  thermo interval; `input_script_dump_every` no longer sets the interval),
  and both also dump the last step of the run, so
  `len(structures) == len(energies)` and `final_structure` / `final_forces`
  belong to the same step as `final_energy` (an unconverged minimisation used
  to lose its last frame). `n_print=0` (LAMMPS `thermo 0`: first and last
  step only) gives exactly those two frames.
- **Results are in the input structure's frame.** LAMMPS needs a
  lower-triangular cell with bounded tilt factors, so it rotates cells that
  are not (primitive bcc/fcc cells and their supercells, sheared cells, ...)
  and may describe them with other lattice vectors. `lammps_job` now passes
  its `structure` to `parse_LammpsOutput` (new `structure` input), which
  rotates positions, forces and pressure tensors back and restores the input
  cell's lattice vectors (when LAMMPS' cell is an integer basis change of
  it). Before, such cells returned forces, `final_stress` and cells in
  LAMMPS' rotated frame (forces off by up to ~1.4 eV/Å per component for a
  primitive bcc supercell; lammpsparser <= 0.0.10 never rotates pressures
  back); results now agree with ASE's EAM calculator to ~1e-7. Cells LAMMPS
  does not rotate (orthogonal, or already lower triangular) are unchanged.
  Called without `structure`, `parse_LammpsOutput` uses the frame of the
  data file LAMMPS read, as before.
- **`CalcInputMD(mode="NPT")` requires `pressure`.** Without it the script
  build raises `ValueError("NPT mode requires CalcInputMD.pressure (in
  Pa)")` instead of a `TypeError` from `None / 1e5`.
- **A failed LAMMPS command raises.** `parse_LammpsOutput` raises a
  `RuntimeError` with the tail of the command's stdout/stderr when the
  preceding `shell` step (its `after` input) exited non-zero, instead of
  parsing files an earlier run left in the working directory.
- **`CalcInputMD(seed=None)` is non-deterministic**: a random seed in
  [1, 900000000] is drawn per input script (it used to be a fixed 12345);
  900000000 is the largest seed the Marsaglia generator behind `fix langevin`
  accepts. `initial_temperature=0`
  is respected (no `velocity ... create` line) instead of falling back to
  `temperature`.

### Fixed

- NVT thermostats `berendsen`, `temp/rescale` and `temp/csvr` integrate the
  equations of motion: they only rescale velocities, and without a
  `fix nve` next to them LAMMPS warned "No fixes with time integration,
  atoms won't move" and the atoms stayed put. `temp/rescale` also had an
  invalid argument list (`... <window> units box` instead of
  `... <window> <fraction>`, so every run failed; it now rescales fully,
  fraction 1.0, every `n_print` steps) and `temp/csvr` lacked its seed
  argument (every run failed with "Incorrect number of arguments").
- A custom `parse_fn` / `parse_fn_kwargs` set on `LammpsEngine` is now used
  (passed to `parse_LammpsOutput` as `_lammps_parser_function` /
  `_lammps_parser_args`); before it was only returned by `get_parse_fn()`.
- `with_working_directory` keeps a custom `calc_fn` instead of resetting it.
- `toggle_mode` no longer warns when the existing mode agrees with
  `EngineInput`.
- `shell` passes the command to the shell as one string.
- Non-default `input_script_read_data_file` / `input_script_dump_filename`
  are forwarded to the run (`lammps_calculator_fn` gained
  `lammps_structure_filepath` and `dump_out_file_name`); before, the script
  used the custom names while the run and parser looked for `lammps.data` /
  `dump.out`.
- `create_WorkingDirectory` uses `os.makedirs(..., exist_ok=True)`; sibling
  nodes creating the same or nested directories concurrently crashed with
  `FileExistsError` under `pyiron_workflow` 0.20.
- No process-wide `warnings.filterwarnings("ignore")` in
  `parse_LammpsOutput` any more; only lammpsparser's per-parse notice about
  the `c_eng` unit conversion is filtered.
- MD time units (#60): `CalcInputMD.time_step`, `thermostat_time_constant`
  and `pressure_damping_timescale` are femtoseconds and are now converted to
  the LAMMPS time unit (ps for `units metal`); before, `time_step=2.0` became
  a 2 ps step. `seed=None` is no longer written literally into the input.
- `dump_modify` format string has the right number of fields (#60); the old
  one crashed LAMMPS on the first MD dump frame.
- Per-atom velocities are attached to the trajectory frames in
  `EngineOutput.structures`, converted to ASE units (#59).

### Notes

- The `andersen` thermostat is implemented with LAMMPS `fix langevin`
  (plus `fix nve`), i.e. it is a Langevin thermostat, not Andersen's
  stochastic-collision method; it behaves exactly like `langevin`.
- `temp/rescale` and `temp/csvr` are accepted by the engine although
  `CalcInputMD.thermostat`'s type hint lists only `nose-hoover`, `langevin`,
  `berendsen` and `andersen`.

### Removed (breaking)

- `submit_to_slurm` (it relied on `PickleStorage`, which `pyiron_workflow`
  0.19 removed) and the `[cluster]` extra (`pysqa`, `h5py`).
- Dependencies that were never imported: `pandas`, `matplotlib`, `scipy`,
  `scikit-learn`, `tqdm`, `lz-GB-code`.

### Dependencies

- Runtime dependencies are the directly imported packages only: `ase`,
  `flowrep`, `lammpsparser`, `numpy`, `pyiron_snippets`, `pyiron_workflow`,
  `pyiron_workflow_atomistics`, `pymatgen`. Published ranges:
  `pyiron_workflow >=0.19.0,<0.21.0`, `flowrep >=0.6.0,<0.7.0`,
  `pyiron_workflow_atomistics >=0.2.2,<0.4.0`,
  `lammpsparser >=0.0.7,<0.1.0`, `pymatgen >=2024.8.8,<2027.0.0`,
  `numpy >=1.26.4,<2.6.0`, `ase >=3.24.0,<3.30.0`,
  `pyiron_snippets >=1.3.0,<2.0.0`.
- CI pins: `pyiron_workflow` 0.20.3, `flowrep` 0.6.7,
  `pyiron_workflow_atomistics` 0.3.0, `lammpsparser` 0.0.10, `numpy` 2.4.6,
  `ase` 3.29.0, `pymatgen` 2026.9.24, `pyiron_snippets` 1.5.0; conda-forge
  `lammps` 2025.07.22.

### Packaging

- Built with `python -m build` in isolation; nothing is imported at build
  time (versioneer supplies the version via `setup.py`). The wheel ships a
  static `_version.py`, and the sdist contains the complete test suite.
- Releases publish every 0.x dependency as a `>=floor,<next-minor` range
  (`.ci_support/upper-bound.yml`) instead of an exact `==` pin.
- CI: the unit tests also run against the floors
  (`.ci_support/lower-bound.yml`, python 3.11); a check fails when a CI pin
  is outside the range the release would publish; the LAMMPS integration
  tests run against the built wheel on python 3.11 to 3.14.
- The docs landing page includes the top-level `README.md` instead of a
  copy.

## [0.1.0] — 2026-05-13

### Removed (breaking)

- **Windows support dropped.** `conda-forge` ships no Windows build of
  `lammps=2024.08.29` (the pinned version), so the CI matrix can never
  solve. The `Operating System :: OS Independent` classifier is replaced
  with explicit `POSIX :: Linux` + `MacOS` entries. Users on Windows
  should run via WSL2.

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
