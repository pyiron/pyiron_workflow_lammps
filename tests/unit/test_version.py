import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the module path to sys.path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyiron_workflow_lammps'))

from pyiron_workflow_lammps._version import (
    get_keywords,
    get_config,
    NotThisMethod,
    run_command,
    register_vcs_handler,
    HANDLERS,
    LONG_VERSION_PY
)


class TestVersion(unittest.TestCase):
    """Test the version module functionality."""
    
    def test_get_keywords(self):
        """Test get_keywords function."""
        keywords = get_keywords()
        
        self.assertIsInstance(keywords, dict)
        self.assertIn('refnames', keywords)
        self.assertIn('full', keywords)
        self.assertIn('date', keywords)
        
        # These should be git format strings
        self.assertEqual(keywords['refnames'], "$Format:%d$")
        self.assertEqual(keywords['full'], "$Format:%H$")
        self.assertEqual(keywords['date'], "$Format:%ci$")
        
    def test_get_config(self):
        """Test get_config function."""
        config = get_config()
        
        self.assertEqual(config.VCS, "git")
        self.assertEqual(config.style, "pep440-pre")
        self.assertEqual(config.tag_prefix, "pyiron_workflow_lammps-")
        self.assertEqual(config.parentdir_prefix, "pyiron_workflow_lammps")
        self.assertEqual(config.versionfile_source, "pyiron_workflow_lammps/_version.py")
        self.assertFalse(config.verbose)
        
    def test_not_this_method_exception(self):
        """Test NotThisMethod exception."""
        with self.assertRaises(NotThisMethod):
            raise NotThisMethod("Test exception")
            
    def test_register_vcs_handler(self):
        """Test register_vcs_handler decorator."""
        @register_vcs_handler("test_vcs", "test_method")
        def test_handler():
            return "test_result"
            
        self.assertIn("test_vcs", HANDLERS)
        self.assertIn("test_method", HANDLERS["test_vcs"])
        self.assertEqual(HANDLERS["test_vcs"]["test_method"], test_handler)
        
    @patch('subprocess.Popen')
    def test_run_command_success(self, mock_popen):
        """Test run_command with successful execution."""
        mock_process = MagicMock()
        # return BYTES so .decode() in run_command works
        mock_process.communicate.return_value = (b"test output", b"test error")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process

        # your function expects (commands, args)
        output, return_code = run_command(["echo"], ["test"])

        self.assertEqual(output, "test output")
        self.assertEqual(return_code, 0)
        mock_popen.assert_called_once()
        
    @patch('subprocess.Popen')
    def test_run_command_failure(self, mock_popen):
        """Test run_command with failed execution."""
        mock_process = MagicMock()
        # return BYTES so .decode() in run_command works
        mock_process.communicate.return_value = (b"", b"error message")
        mock_process.returncode = 1
        mock_popen.return_value = mock_process
        
        output, return_code = run_command(["invalid_command"], [])
        
        self.assertIsNone(output)
        self.assertEqual(return_code, 1)
        
    @patch('subprocess.Popen')
    def test_run_command_with_cwd(self, mock_popen):
        """Test run_command with custom working directory."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        run_command(["echo"], ["test"], cwd="/test/directory")
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        self.assertEqual(call_args[1]['cwd'], "/test/directory")
        
    @patch('subprocess.Popen')
    def test_run_command_with_env(self, mock_popen):
        """Test run_command with custom environment."""
        mock_process = MagicMock()
        mock_process.communicate.return_value = (b"output", b"")
        mock_process.returncode = 0
        mock_popen.return_value = mock_process
        
        env_vars = {"TEST_VAR": "test_value"}
        run_command(["echo"], ["test"], env=env_vars)
        
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        self.assertIn("TEST_VAR", call_args[1]['env'])
        self.assertEqual(call_args[1]['env']["TEST_VAR"], "test_value")
        
    def test_long_version_py_initialization(self):
        """Test LONG_VERSION_PY initialization."""
        self.assertIsInstance(LONG_VERSION_PY, dict)
        
    def test_handlers_initialization(self):
        """Test HANDLERS initialization."""
        self.assertIsInstance(HANDLERS, dict)


if __name__ == '__main__':
    unittest.main() 