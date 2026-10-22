"""Version handling.

In a source tree (plain checkout or editable install) versioneer queries
git, so ``__version__`` may legitimately be ``0+untagged...`` (no tags
fetched) or ``0+unknown`` (no git at all). Only an installed build from a
tagged sdist/wheel must carry a real version, which then also has to match
the installed distribution metadata.
"""

import importlib.metadata
import re
import unittest
from pathlib import Path

import pyiron_workflow_lammps
import pyiron_workflow_lammps._version as version_module

_PACKAGE_DIR = Path(pyiron_workflow_lammps.__file__).resolve().parent
FROM_SOURCE_TREE = (_PACKAGE_DIR.parent / "pyproject.toml").is_file()

# Built distributions carry versioneer's short static _version.py, which only
# defines get_versions(); the git-querying template is only in the source tree.
HAS_VERSIONEER_TEMPLATE = hasattr(version_module, "get_config")


class TestVersion(unittest.TestCase):
    def test_version_is_a_pep440ish_string(self):
        version = pyiron_workflow_lammps.__version__

        self.assertIsInstance(version, str)
        # "1.2.3", "0.1.0.post0.dev16", "0+untagged.5.gabc1234", "0+unknown"
        self.assertRegex(version, r"^\d+(\.\d+)*([.+-]\S*)?$")

    def test_get_versions(self):
        versions = version_module.get_versions()

        for key in ("version", "full-revisionid", "dirty", "error", "date"):
            self.assertIn(key, versions)
        self.assertEqual(versions["version"], pyiron_workflow_lammps.__version__)


@unittest.skipIf(FROM_SOURCE_TREE, "imported from a source tree")
class TestInstalledVersion(unittest.TestCase):
    """An installed (non-editable) build must know its version."""

    def test_version_is_resolved(self):
        version = pyiron_workflow_lammps.__version__

        self.assertNotEqual(version, "0+unknown")
        self.assertIsNone(re.match(r"^0\+untagged", version), version)

    def test_version_matches_metadata(self):
        self.assertEqual(
            pyiron_workflow_lammps.__version__,
            importlib.metadata.version("pyiron_workflow_lammps"),
        )

    def test_version_file_is_static(self):
        self.assertFalse(
            HAS_VERSIONEER_TEMPLATE,
            "installed _version.py is the versioneer template; "
            "is versionfile_build set in [tool.versioneer]?",
        )


@unittest.skipUnless(
    HAS_VERSIONEER_TEMPLATE, "built _version.py has no versioneer internals"
)
class TestVersioneerTemplate(unittest.TestCase):
    def test_config_matches_tag_scheme(self):
        config = version_module.get_config()

        self.assertEqual(config.VCS, "git")
        self.assertEqual(config.style, "pep440-pre")
        self.assertEqual(config.tag_prefix, "pyiron_workflow_lammps-")
        self.assertEqual(
            config.versionfile_source, "pyiron_workflow_lammps/_version.py"
        )

    def test_render_of_tagged_and_untagged_pieces(self):
        pieces = {
            "long": "a" * 40,
            "short": "aaaaaaa",
            "error": None,
            "dirty": False,
            "closest-tag": "0.2.0",
            "distance": 0,
            "date": None,
            "branch": "main",
        }
        self.assertEqual(
            version_module.render(pieces, "pep440-pre")["version"], "0.2.0"
        )

        pieces.update({"closest-tag": None, "distance": 5})
        untagged = version_module.render(pieces, "pep440-pre")["version"]
        self.assertTrue(untagged.startswith("0"), untagged)


if __name__ == "__main__":
    unittest.main()
