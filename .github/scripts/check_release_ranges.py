"""Check that the release rewrite keeps every CI pin inside its published range.

``pyproject.toml`` pins each runtime dependency exactly (``==``); the release
workflow (``release.yml`` -> ``pyproject-release.yml``) rewrites those pins to
``>=floor,<cap`` ranges from ``.ci_support/lower-bound.yml`` and
``.ci_support/upper-bound.yml``. Dependabot bumps the pins but never the
bound files, so a bump past a cap would publish a range that excludes the
only version CI tested, without any error. This script runs the same rewrite
(same script, same inputs as ``release.yml``) on a temporary copy and fails if
any pin is not contained in its rewritten specifier.

Run from the repository root with pyyaml, toml (the versions release.yml
installs) and packaging available::

    python .github/scripts/check_release_ranges.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path.cwd()
REWRITE = ROOT / ".github/actions/.support/update_pyproject_dependencies.py"


def release_inputs() -> dict[str, str]:
    """The rewrite inputs release.yml passes, with pyproject-release.yml defaults."""
    release = yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text())
    reusable = yaml.safe_load(
        (ROOT / ".github/workflows/pyproject-release.yml").read_text()
    )
    # PyYAML reads the bare `on:` key as the boolean True
    declared = (reusable.get("on") or reusable[True])["workflow_call"]["inputs"]
    inputs = {k: str(v["default"]) for k, v in declared.items() if "default" in v}
    (job,) = release["jobs"].values()
    inputs.update({k: str(v) for k, v in job.get("with", {}).items()})
    return inputs


def main() -> int:
    inputs = release_inputs()
    name_map = inputs["pypi-to-conda-name-map-file"]
    if "$" in name_map:
        sys.exit(f"release.yml must pass an explicit name map file, got {name_map}")
    input_toml = ROOT / inputs["input-toml"]
    with tempfile.TemporaryDirectory() as tmp:
        output_toml = Path(tmp) / "pyproject.toml"
        cmd = [
            sys.executable,
            str(REWRITE),
            f"--input_toml={input_toml}",
            f"--lower_bound_yaml={inputs['lower-bound-yaml']}",
            f"--upper_bound_yaml={inputs['upper-bound-yaml']}",
            f"--semantic_upper_bound={inputs['semantic-upper-bound']}",
            f"--always_pin_unstable={inputs['always-pin-unstable']}",
            f"--output_toml={output_toml}",
            f"--pypi_to_conda_name_map_file={name_map}",
        ]
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        released = tomllib.loads(output_toml.read_text())["project"]["dependencies"]
    pinned = tomllib.loads(input_toml.read_text())["project"]["dependencies"]

    ranges = {canonicalize_name(r.name): r for r in map(Requirement, released)}
    errors = []
    for dep in pinned:
        req = Requirement(dep)
        (spec,) = req.specifier  # the rewrite itself requires NAME==VERSION
        pin = spec.version
        rewritten = ranges[canonicalize_name(req.name)]
        ok = rewritten.specifier.contains(pin, prereleases=True)
        print(f"{'ok' if ok else 'FAIL':4}  {req.name}=={pin}  ->  {rewritten}")
        if not ok:
            errors.append(
                f"{req.name}=={pin} (the CI pin) is outside the published range "
                f"{rewritten.specifier}: move the bound in "
                f"{inputs['upper-bound-yaml']} / {inputs['lower-bound-yaml']}"
            )
    for e in errors:
        print("error:", e, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
