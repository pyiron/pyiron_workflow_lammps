"""Run the ``python`` code blocks of the top-level README.md.

The blocks are executed in order, in one namespace, inside a temporary
working directory, so later blocks may use names defined by earlier ones.
Any string literal naming an ``*.eam.fs`` file (e.g. the placeholder
``"/path/to/Fe.eam.fs"``) is replaced by the vendored
``tests/fixtures/Al-Fe.eam.fs``; everything else runs as written, using
``lmp`` from ``PATH``.

Skipped when README.md is not next to ``tests/`` (e.g. a copy of the test
suite run against an installed package) or when ``lmp`` is not on ``PATH``.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

README = Path(__file__).resolve().parents[2] / "README.md"
_FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
_EAM_LITERAL = re.compile(r"""(["'])[^"'\n]*\.eam\.fs\1""")


def _python_blocks() -> list[str]:
    if not README.is_file():
        return []
    return _FENCE.findall(README.read_text(encoding="utf-8"))


def test_readme_python_blocks(tmp_path, monkeypatch, eam_potential):
    blocks = _python_blocks()
    if not blocks:
        pytest.skip(f"no python code blocks in {README} (or no README)")
    if shutil.which("lmp") is None:
        pytest.skip("the README examples call 'lmp', which is not on PATH")

    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__readme__"}
    for i, block in enumerate(blocks):
        source = _EAM_LITERAL.sub(lambda _: repr(str(eam_potential)), block)
        code = compile(source, f"README.md python block {i}", "exec")
        exec(code, namespace)
