import unittest
import tempfile
import os
import shutil
import subprocess
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from pyiron_workflow_lammps.generic import (
    Storage,
    ShellOutput,
    VarType,
    FileObject,
    shell,
    isLineInFile,
    create_WorkingDirectory,
    delete_files_recursively,
    compress_directory,
    submit_to_slurm,
    remove_dir
)


class TestStorage(unittest.TestCase):
    """Test the Storage class."""
    
    def test_convert_to_dict(self):
        """Test _convert_to_dict method."""
        class TestClass:
            def __init__(self):
                self.public_attr = "value"
                self._private_attr = "private"
                self.another_attr = 42
                
        instance = TestClass()
        result = Storage._convert_to_dict(instance)
        
        expected = {
            'public_attr': 'value',
            'another_attr': 42
        }
        self.assertEqual(result, expected)
        self.assertNotIn('_private_attr', result)


class TestShellOutput(unittest.TestCase):
    """Test the ShellOutput class."""
    
    def test_shell_output_attributes(self):
        """Test ShellOutput attributes."""
        output = ShellOutput()
        output.stdout = "test output"
        output.stderr = "test error"
        output.return_code = 0
        output.dump = FileObject("test.dump")
        output.log = FileObject("test.log")
        
        self.assertEqual(output.stdout, "test output")
        self.assertEqual(output.stderr, "test error")
        self.assertEqual(output.return_code, 0)
        self.assertIsInstance(output.dump, FileObject)
        self.assertIsInstance(output.log, FileObject)


class TestVarType(unittest.TestCase):
    """Test the VarType class."""
    
    def test_vartype_init(self):
        """Test VarType initialization."""
        var = VarType(
            value=42,
            dat_type=int,
            label="test_label",
            store=1,
            generic=True,
            doc="test documentation"
        )
        
        self.assertEqual(var.value, 42)
        self.assertEqual(var.type, int)
        self.assertEqual(var.label, "test_label")
        self.assertEqual(var.store, 1)
        self.assertTrue(var.generic)
        self.assertEqual(var.doc, "test documentation")
        
    def test_vartype_defaults(self):
        """Test VarType with default values."""
        var = VarType()
        
        self.assertIsNone(var.value)
        self.assertIsNone(var.type)
        self.assertIsNone(var.label)
        self.assertEqual(var.store, 0)
        self.assertIsNone(var.generic)
        self.assertIsNone(var.doc)


class TestFileObject(unittest.TestCase):
    """Test the FileObject class."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_fileobject_init_with_path(self):
        """Test FileObject initialization with path only."""
        file_obj = FileObject("test.txt")
        
        self.assertEqual(file_obj.path, "test.txt")
        self.assertEqual(file_obj.name, "test.txt")
        self.assertFalse(file_obj.is_file)
        
    def test_fileobject_init_with_directory(self):
        """Test FileObject initialization with directory."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
            
        file_obj = FileObject("test.txt", directory=self.temp_dir)
        
        self.assertEqual(file_obj.path, test_file)
        self.assertEqual(file_obj.name, "test.txt")
        self.assertTrue(file_obj.is_file)
        
    def test_fileobject_repr(self):
        """Test FileObject string representation."""
        file_obj = FileObject("test.txt")
        repr_str = repr(file_obj)
        
        self.assertIn("FileObject", repr_str)
        self.assertIn("test.txt", repr_str)
        self.assertIn("False", repr_str)  # is_file should be False


class TestShell(unittest.TestCase):
    """Test the shell function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    @patch('subprocess.run')
    def test_shell_basic(self, mock_run):
        """Test basic shell command execution."""
        mock_process = MagicMock()
        mock_process.stdout = "test output"
        mock_process.stderr = "test error"
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        result = shell("echo test", working_directory=self.temp_dir).run()
        
        self.assertIsInstance(result, ShellOutput)
        self.assertEqual(result.stdout, "test output")
        self.assertEqual(result.stderr, "test error")
        self.assertEqual(result.return_code, 0)
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], ["echo test"])
        self.assertEqual(call_args[1]['cwd'], self.temp_dir)
        self.assertTrue(call_args[1]['shell'])
        
    @patch('subprocess.run')
    def test_shell_with_environment(self, mock_run):
        """Test shell command with environment variables."""
        mock_process = MagicMock()
        mock_process.stdout = "output"
        mock_process.stderr = "error"
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        env_vars = {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}
        
        result = shell("echo test", working_directory=self.temp_dir, environment=env_vars).run()
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertIn("TEST_VAR", call_args[1]['env'])
        self.assertEqual(call_args[1]['env']["TEST_VAR"], "test_value")
        
    @patch('subprocess.run')
    def test_shell_with_arguments(self, mock_run):
        """Test shell command with additional arguments."""
        mock_process = MagicMock()
        mock_process.stdout = "output"
        mock_process.stderr = "error"
        mock_process.returncode = 0
        mock_run.return_value = mock_process
        
        arguments = ["arg1", "arg2", "arg3"]
        
        result = shell("test_command", working_directory=self.temp_dir, arguments=arguments).run()
        
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], ["test_command", "arg1", "arg2", "arg3"])
        
    @patch('subprocess.run')
    def test_shell_error_return_code(self, mock_run):
        """Test shell command with error return code."""
        mock_process = MagicMock()
        mock_process.stdout = "output"
        mock_process.stderr = "error message"
        mock_process.returncode = 1
        mock_run.return_value = mock_process
        
        result = shell("failing_command", working_directory=self.temp_dir).run()
        
        self.assertEqual(result.return_code, 1)
        self.assertEqual(result.stderr, "error message")


class TestIsLineInFile(unittest.TestCase):
    """Test the isLineInFile function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_is_line_in_file_exact_match_true(self):
        """Test exact line matching when line exists."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("line 1\n")
            f.write("target line\n")
            f.write("line 3\n")
            
        result = isLineInFile.node_function(test_file, "target line", exact_match=True)
        
        self.assertTrue(result)
        
    def test_is_line_in_file_exact_match_false(self):
        """Test exact line matching when line doesn't exist."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("line 1\n")
            f.write("different line\n")
            f.write("line 3\n")
            
        result = isLineInFile.node_function(test_file, "target line", exact_match=True)
        
        self.assertFalse(result)
        
    def test_is_line_in_file_partial_match_true(self):
        """Test partial line matching when substring exists."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("line 1\n")
            f.write("this contains target substring\n")
            f.write("line 3\n")
            
        result = isLineInFile.node_function(test_file, "target", exact_match=False)
        
        self.assertTrue(result)
        
    def test_is_line_in_file_partial_match_false(self):
        """Test partial line matching when substring doesn't exist."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("line 1\n")
            f.write("different content\n")
            f.write("line 3\n")
            
        result = isLineInFile.node_function(test_file, "target", exact_match=False)
        
        self.assertFalse(result)
        
    def test_is_line_in_file_file_not_found(self):
        """Test behavior when file doesn't exist."""
        non_existent_file = os.path.join(self.temp_dir, "nonexistent.txt")
        
        result = isLineInFile.node_function(non_existent_file, "test", exact_match=True)
        
        self.assertFalse(result)
        
    def test_is_line_in_file_empty_file(self):
        """Test behavior with empty file."""
        test_file = os.path.join(self.temp_dir, "empty.txt")
        with open(test_file, 'w') as f:
            pass  # Empty file
            
        result = isLineInFile.node_function(test_file, "test", exact_match=True)
        
        self.assertFalse(result)


class TestCreateWorkingDirectory(unittest.TestCase):
    """Test the create_WorkingDirectory function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_create_working_directory_new(self):
        """Test creating a new working directory."""
        new_dir = os.path.join(self.temp_dir, "new_working_dir")
        
        result = create_WorkingDirectory(new_dir).run()
        
        self.assertEqual(result, new_dir)
        self.assertTrue(os.path.exists(new_dir))
        self.assertTrue(os.path.isdir(new_dir))
        
    def test_create_working_directory_exists(self):
        """Test behavior when directory already exists."""
        existing_dir = os.path.join(self.temp_dir, "existing_dir")
        os.makedirs(existing_dir)
        
        result = create_WorkingDirectory(existing_dir).run()
        
        self.assertEqual(result, existing_dir)
        self.assertTrue(os.path.exists(existing_dir))
        
    def test_create_working_directory_exists_quiet(self):
        """Test behavior when directory exists with quiet=True."""
        existing_dir = os.path.join(self.temp_dir, "existing_dir")
        os.makedirs(existing_dir)
        
        result = create_WorkingDirectory(existing_dir, quiet=True).run()
        
        self.assertEqual(result, existing_dir)
        self.assertTrue(os.path.exists(existing_dir))


class TestDeleteFilesRecursively(unittest.TestCase):
    """Test the delete_files_recursively function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_delete_files_recursively_success(self):
        """Test successful file deletion."""
        # Create test files
        test_file1 = os.path.join(self.temp_dir, "file1.txt")
        test_file2 = os.path.join(self.temp_dir, "file2.txt")
        subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(subdir)
        test_file3 = os.path.join(subdir, "file3.txt")
        
        with open(test_file1, 'w') as f:
            f.write("content1")
        with open(test_file2, 'w') as f:
            f.write("content2")
        with open(test_file3, 'w') as f:
            f.write("content3")
            
        files_to_delete = ["file1.txt", "file3.txt"]
        
        result = delete_files_recursively(self.temp_dir, files_to_delete).run()
        
        self.assertEqual(result, self.temp_dir)
        self.assertFalse(os.path.exists(test_file1))
        self.assertTrue(os.path.exists(test_file2))  # Should not be deleted
        self.assertFalse(os.path.exists(test_file3))
        
    def test_delete_files_recursively_invalid_directory(self):
        """Test behavior with invalid directory."""
        invalid_dir = os.path.join(self.temp_dir, "nonexistent")
        
        result = delete_files_recursively(invalid_dir, ["test.txt"]).run()
        
        self.assertEqual(result, invalid_dir)
        
    def test_delete_files_recursively_no_matching_files(self):
        """Test behavior when no files match the deletion criteria."""
        test_file = os.path.join(self.temp_dir, "keep_this.txt")
        with open(test_file, 'w') as f:
            f.write("content")
            
        result = delete_files_recursively(self.temp_dir, ["delete_this.txt"]).run() 
        
        self.assertEqual(result, self.temp_dir)
        self.assertTrue(os.path.exists(test_file))  # Should still exist


class TestCompressDirectory(unittest.TestCase):
    """Test the compress_directory function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_compress_directory_basic(self):
        """Test basic directory compression."""
        # Create test files
        test_file1 = os.path.join(self.temp_dir, "file1.txt")
        test_file2 = os.path.join(self.temp_dir, "file2.txt")
        
        with open(test_file1, 'w') as f:
            f.write("content1")
        with open(test_file2, 'w') as f:
            f.write("content2")
            
        result = compress_directory(self.temp_dir, actually_compress=True).run()
        
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith(".tar.gz"))
        self.assertTrue(os.path.exists(result))
        
    def test_compress_directory_inside_dir(self):
        """Test compression with inside_dir=True."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("content")
            
        result = compress_directory(self.temp_dir, inside_dir=True, actually_compress=True).run()
        
        expected_path = os.path.join(self.temp_dir, os.path.basename(self.temp_dir) + ".tar.gz")
        self.assertEqual(result, expected_path)
        
    def test_compress_directory_outside_dir(self):
        """Test compression with inside_dir=False."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("content")
            
        result = compress_directory(self.temp_dir, inside_dir=False, actually_compress=True).run()
        
        expected_path = os.path.join(os.path.dirname(self.temp_dir), os.path.basename(self.temp_dir) + ".tar.gz")
        self.assertEqual(result, expected_path)
        
    def test_compress_directory_exclude_files(self):
        """Test compression with file exclusion."""
        test_file1 = os.path.join(self.temp_dir, "keep.txt")
        test_file2 = os.path.join(self.temp_dir, "exclude.txt")
        
        with open(test_file1, 'w') as f:
            f.write("keep this")
        with open(test_file2, 'w') as f:
            f.write("exclude this")
            
        result = compress_directory(
            self.temp_dir, 
            exclude_files=["exclude.txt"],
            actually_compress=True
        ).run()
        
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        
    def test_compress_directory_exclude_patterns(self):
        """Test compression with pattern exclusion."""
        test_file1 = os.path.join(self.temp_dir, "keep.txt")
        test_file2 = os.path.join(self.temp_dir, "exclude.log")
        
        with open(test_file1, 'w') as f:
            f.write("keep this")
        with open(test_file2, 'w') as f:
            f.write("exclude this")
            
        result = compress_directory(
            self.temp_dir, 
            exclude_file_patterns=["*.log"],
            actually_compress=True
        ).run()
        
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
        
    def test_compress_directory_no_compression(self):
        """Test compression with actually_compress=False."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("content")
            
        result = compress_directory(self.temp_dir, actually_compress=False).run()
        
        self.assertIsNone(result)


class TestSubmitToSlurm(unittest.TestCase):
    """Test the submit_to_slurm function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    @patch('subprocess.run')
    def test_submit_to_slurm_basic(self, mock_run):
        """Test basic SLURM submission."""
        mock_node = MagicMock()
        mock_node.graph_root = mock_node
        mock_node.full_label = "test_node"
        mock_node.lexical_delimiter = "."
        mock_node.as_path.return_value = Path(self.temp_dir)
        
        mock_submission = MagicMock()
        mock_run.return_value = mock_submission
        
        result = submit_to_slurm(mock_node)
        
        self.assertEqual(result, mock_submission)
        mock_run.assert_called_once()
        
    def test_submit_to_slurm_not_root_node(self):
        """Test SLURM submission with non-root node."""
        mock_node = MagicMock()
        mock_root = MagicMock()
        mock_node.graph_root = mock_root
        mock_node.full_label = "child_node"
        
        with self.assertRaises(ValueError) as context:
            submit_to_slurm(mock_node)
        
        self.assertIn("Can only submit parent-most nodes", str(context.exception))


class TestRemoveDir(unittest.TestCase):
    """Test the remove_dir function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
    def test_remove_dir_actually_remove(self):
        """Test directory removal with actually_remove=True."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("content")
            
        result = remove_dir(self.temp_dir, actually_remove=True).run()
        
        self.assertEqual(result, self.temp_dir)
        self.assertFalse(os.path.exists(self.temp_dir))
        
    def test_remove_dir_no_remove(self):
        """Test directory removal with actually_remove=False."""
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("content")
            
        result = remove_dir(self.temp_dir, actually_remove=False).run()
        
        self.assertEqual(result, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))  # Should still exist


if __name__ == '__main__':
    unittest.main() 