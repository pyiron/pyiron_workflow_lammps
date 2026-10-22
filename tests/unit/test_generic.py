import os
import shutil
import tarfile
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from pyiron_workflow_lammps.generic import (
    FileObject,
    ShellOutput,
    Storage,
    VarType,
    compress_directory,
    create_WorkingDirectory,
    delete_files_recursively,
    isLineInFile,
    remove_dir,
    shell,
)

try:
    import pyiron_workflow as pwf
except ImportError:  # pragma: no cover - pyiron_workflow is optional here
    pwf = None


def _completed(stdout="output", stderr="", returncode=0):
    proc = MagicMock()
    proc.stdout = stdout
    proc.stderr = stderr
    proc.returncode = returncode
    return proc


class TestStorage(unittest.TestCase):
    def test_convert_to_dict(self):
        class TestClass:
            def __init__(self):
                self.public_attr = "value"
                self._private_attr = "private"
                self.another_attr = 42

        result = Storage._convert_to_dict(TestClass())

        self.assertEqual(result, {"public_attr": "value", "another_attr": 42})


class TestShellOutput(unittest.TestCase):
    def test_shell_output_attributes(self):
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
    def test_vartype_init(self):
        var = VarType(
            value=42,
            dat_type=int,
            label="test_label",
            store=1,
            generic=True,
            doc="test documentation",
        )

        self.assertEqual(var.value, 42)
        self.assertEqual(var.type, int)
        self.assertEqual(var.label, "test_label")
        self.assertEqual(var.store, 1)
        self.assertTrue(var.generic)
        self.assertEqual(var.doc, "test documentation")

    def test_vartype_defaults(self):
        var = VarType()

        self.assertIsNone(var.value)
        self.assertIsNone(var.type)
        self.assertIsNone(var.label)
        self.assertEqual(var.store, 0)
        self.assertIsNone(var.generic)
        self.assertIsNone(var.doc)


class TestFileObject(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_fileobject_init_with_path(self):
        missing = os.path.join(self.temp_dir, "test.txt")
        file_obj = FileObject(missing)

        self.assertEqual(file_obj.path, missing)
        self.assertEqual(file_obj.name, "test.txt")
        self.assertFalse(file_obj.is_file)

    def test_fileobject_init_with_directory(self):
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test content")

        file_obj = FileObject("test.txt", directory=self.temp_dir)

        self.assertEqual(file_obj.path, test_file)
        self.assertEqual(file_obj.name, "test.txt")
        self.assertTrue(file_obj.is_file)

    def test_fileobject_repr(self):
        repr_str = repr(FileObject(os.path.join(self.temp_dir, "test.txt")))

        self.assertIn("FileObject", repr_str)
        self.assertIn("test.txt", repr_str)
        self.assertIn("False", repr_str)


class TestShell(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("pyiron_workflow_lammps.generic.subprocess.run")
    def test_shell_basic(self, mock_run):
        mock_run.return_value = _completed("test output", "test error", 0)

        result = shell(command="echo test", working_directory=self.temp_dir)

        self.assertIsInstance(result, ShellOutput)
        self.assertEqual(result.stdout, "test output")
        self.assertEqual(result.stderr, "test error")
        self.assertEqual(result.return_code, 0)
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        # shell=True needs a single command string
        self.assertEqual(args[0], "echo test")
        self.assertEqual(kwargs["cwd"], self.temp_dir)
        self.assertTrue(kwargs["shell"])

    @patch("pyiron_workflow_lammps.generic.subprocess.run")
    def test_shell_with_environment(self, mock_run):
        mock_run.return_value = _completed()

        shell(
            command="echo test",
            working_directory=self.temp_dir,
            environment={"TEST_VAR": "test_value", "NUMBER": 3},
        )

        env = mock_run.call_args.kwargs["env"]
        self.assertEqual(env["TEST_VAR"], "test_value")
        self.assertEqual(env["NUMBER"], "3")
        # The parent environment is inherited, not replaced.
        self.assertIn("PATH", env)

    @patch("pyiron_workflow_lammps.generic.subprocess.run")
    def test_shell_with_arguments(self, mock_run):
        mock_run.return_value = _completed()

        shell(
            command="test_command",
            working_directory=self.temp_dir,
            arguments=["arg1", "arg2", 3],
        )

        self.assertEqual(mock_run.call_args.args[0], "test_command arg1 arg2 3")

    @patch("pyiron_workflow_lammps.generic.subprocess.run")
    def test_shell_error_return_code(self, mock_run):
        mock_run.return_value = _completed("output", "error message", 1)

        result = shell(command="failing_command", working_directory=self.temp_dir)

        self.assertEqual(result.return_code, 1)
        self.assertEqual(result.stderr, "error message")

    def test_shell_runs_in_working_directory_without_chdir(self):
        cwd_before = os.getcwd()

        result = shell(
            command="pwd && echo $MY_VAR",
            working_directory=self.temp_dir,
            environment={"MY_VAR": "hello"},
            after="ordering token, ignored",
        )

        self.assertEqual(result.return_code, 0)
        lines = result.stdout.splitlines()
        self.assertEqual(os.path.realpath(lines[0]), os.path.realpath(self.temp_dir))
        self.assertEqual(lines[1], "hello")
        self.assertEqual(os.getcwd(), cwd_before)

    def test_shell_real_arguments_are_all_passed(self):
        result = shell(command="echo", arguments=["a", "b", "c"])

        self.assertEqual(result.stdout.strip(), "a b c")

    def test_shell_real_failure(self):
        result = shell(command="exit 3", working_directory=self.temp_dir)

        self.assertEqual(result.return_code, 3)


class TestIsLineInFile(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.txt")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _write(self, *lines):
        with open(self.test_file, "w") as f:
            f.writelines(line + "\n" for line in lines)

    def test_exact_match_true(self):
        self._write("line 1", "target line", "line 3")

        self.assertTrue(
            isLineInFile(filepath=self.test_file, line="target line", exact_match=True)
        )

    def test_exact_match_false(self):
        self._write("line 1", "target line plus", "line 3")

        self.assertFalse(
            isLineInFile(filepath=self.test_file, line="target line", exact_match=True)
        )

    def test_partial_match_true(self):
        self._write("line 1", "this contains target substring", "line 3")

        self.assertTrue(
            isLineInFile(filepath=self.test_file, line="target", exact_match=False)
        )

    def test_partial_match_false(self):
        self._write("line 1", "different content", "line 3")

        self.assertFalse(
            isLineInFile(filepath=self.test_file, line="target", exact_match=False)
        )

    def test_default_is_exact_match(self):
        self._write("this contains target substring")

        self.assertFalse(isLineInFile(filepath=self.test_file, line="target"))

    def test_file_not_found(self):
        missing = os.path.join(self.temp_dir, "nonexistent.txt")

        self.assertFalse(isLineInFile(filepath=missing, line="test"))

    def test_empty_file(self):
        self._write()

        self.assertFalse(isLineInFile(filepath=self.test_file, line="test"))


class TestCreateWorkingDirectory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_new(self):
        new_dir = os.path.join(self.temp_dir, "a", "b")

        result = create_WorkingDirectory(working_directory=new_dir)

        self.assertEqual(result, new_dir)
        self.assertTrue(os.path.isdir(new_dir))

    def test_existing(self):
        existing_dir = os.path.join(self.temp_dir, "existing_dir")
        os.makedirs(existing_dir)
        marker = os.path.join(existing_dir, "keep.txt")
        with open(marker, "w") as f:
            f.write("x")

        for quiet in (False, True):
            result = create_WorkingDirectory(
                working_directory=existing_dir, quiet=quiet
            )
            self.assertEqual(result, existing_dir)
        self.assertTrue(os.path.exists(marker))

    def test_concurrent_nested_creation(self):
        """Sibling nodes create a directory and its subdirectories at once.

        A check-then-create raced here: makedirs of ``base/sub`` creates
        ``base`` between another thread's existence check and its makedirs.
        """
        n_threads, n_rounds = 16, 30
        for round_ in range(n_rounds):
            base = os.path.join(self.temp_dir, f"round{round_}", "calc")
            # shared (base, base/sub, ...) and per-thread (.../t<i>) targets
            targets = [
                os.path.join(base, *(["sub"] * (i % 4)), *([f"t{i}"] * (i % 2)))
                for i in range(n_threads)
            ]
            barrier = threading.Barrier(n_threads)

            def create(target, barrier=barrier):
                barrier.wait()
                return create_WorkingDirectory(working_directory=target, quiet=True)

            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                results = list(pool.map(create, targets))

            self.assertEqual(results, targets)
            for target in targets:
                self.assertTrue(os.path.isdir(target), target)

    def test_path_that_is_a_file_raises(self):
        path = os.path.join(self.temp_dir, "a_file")
        with open(path, "w") as f:
            f.write("x")

        with self.assertRaises(FileExistsError):
            create_WorkingDirectory(working_directory=path)

    def test_logs_whether_it_created_the_directory(self):
        new_dir = os.path.join(self.temp_dir, "logged")
        with patch("pyiron_workflow_lammps.generic.logger") as logger:
            create_WorkingDirectory(working_directory=new_dir)
            create_WorkingDirectory(working_directory=new_dir)
            create_WorkingDirectory(working_directory=new_dir, quiet=True)

        self.assertEqual(logger.info.call_count, 2)
        self.assertIn("made directory", logger.info.call_args_list[0].args[0])
        self.assertIn("already exists", logger.info.call_args_list[1].args[0])
        logger.warning.assert_called_once()
        self.assertIn("already exists", logger.warning.call_args.args[0])


class TestDeleteFilesRecursively(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def _touch(self, *parts):
        path = os.path.join(self.temp_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("content")
        return path

    def test_success(self):
        file1 = self._touch("file1.txt")
        file2 = self._touch("file2.txt")
        file3 = self._touch("subdir", "file1.txt")

        result = delete_files_recursively(
            working_directory=self.temp_dir, files_to_be_deleted=["file1.txt"]
        )

        self.assertEqual(result, self.temp_dir)
        self.assertFalse(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))
        self.assertFalse(os.path.exists(file3))

    def test_invalid_directory(self):
        invalid_dir = os.path.join(self.temp_dir, "nonexistent")

        result = delete_files_recursively(
            working_directory=invalid_dir, files_to_be_deleted=["test.txt"]
        )

        self.assertEqual(result, invalid_dir)

    def test_no_matching_files(self):
        keep = self._touch("keep_this.txt")

        delete_files_recursively(
            working_directory=self.temp_dir, files_to_be_deleted=["delete_this.txt"]
        )

        self.assertTrue(os.path.exists(keep))

    def test_remove_raises_is_logged_not_propagated(self):
        doomed = self._touch("doomed.txt")

        with patch(
            "pyiron_workflow_lammps.generic.os.remove",
            side_effect=PermissionError("nope"),
        ) as mock_remove:
            result = delete_files_recursively(
                working_directory=self.temp_dir, files_to_be_deleted=["doomed.txt"]
            )

        self.assertEqual(result, self.temp_dir)
        mock_remove.assert_called_once()
        self.assertTrue(os.path.exists(doomed))


class TestCompressDirectory(unittest.TestCase):
    def setUp(self):
        self.parent = tempfile.mkdtemp()
        self.temp_dir = os.path.join(self.parent, "calc")
        os.makedirs(self.temp_dir)
        for name in ("keep.txt", "exclude.txt", "exclude.log"):
            with open(os.path.join(self.temp_dir, name), "w") as f:
                f.write(name)

    def tearDown(self):
        shutil.rmtree(self.parent)

    def _members(self, tarball):
        with tarfile.open(tarball) as tar:
            return sorted(os.path.basename(n) for n in tar.getnames())

    def test_inside_dir(self):
        result = compress_directory(directory_path=self.temp_dir, inside_dir=True)

        self.assertEqual(result, os.path.join(self.temp_dir, "calc.tar.gz"))
        self.assertEqual(
            self._members(result), ["exclude.log", "exclude.txt", "keep.txt"]
        )

    def test_outside_dir(self):
        result = compress_directory(directory_path=self.temp_dir, inside_dir=False)

        self.assertEqual(result, os.path.join(self.parent, "calc.tar.gz"))
        self.assertTrue(os.path.exists(result))

    def test_exclusions(self):
        result = compress_directory(
            directory_path=self.temp_dir,
            exclude_files=["exclude.txt"],
            exclude_file_patterns=["*.log"],
        )

        self.assertEqual(self._members(result), ["keep.txt"])

    def test_no_compression(self):
        result = compress_directory(
            directory_path=self.temp_dir, actually_compress=False
        )

        self.assertIsNone(result)
        self.assertFalse(os.path.exists(os.path.join(self.temp_dir, "calc.tar.gz")))


class TestRemoveDir(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        with open(os.path.join(self.temp_dir, "test.txt"), "w") as f:
            f.write("content")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_actually_remove(self):
        result = remove_dir(directory_path=self.temp_dir, actually_remove=True)

        self.assertEqual(result, self.temp_dir)
        self.assertFalse(os.path.exists(self.temp_dir))

    def test_default_keeps_directory(self):
        result = remove_dir(directory_path=self.temp_dir)

        self.assertEqual(result, self.temp_dir)
        self.assertTrue(os.path.exists(self.temp_dir))


@unittest.skipIf(pwf is None, "pyiron_workflow is not installed")
class TestGenericAsNodes(unittest.TestCase):
    """The decorated functions also run as pyiron_workflow nodes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_shell_node(self):
        run = pwf.node(shell).run(command="echo node", working_directory=self.temp_dir)

        self.assertEqual(run.outputs.output.stdout.strip(), "node")
        self.assertEqual(run.outputs.output.return_code, 0)

    def test_create_working_directory_node(self):
        new_dir = os.path.join(self.temp_dir, "made_by_node")

        run = pwf.node(create_WorkingDirectory).run(working_directory=new_dir)

        self.assertEqual(run.outputs.working_directory, new_dir)
        self.assertTrue(os.path.isdir(new_dir))

    def test_is_line_in_file_node(self):
        path = os.path.join(self.temp_dir, "log.lammps")
        with open(path, "w") as f:
            f.write("Total wall time: 0:00:01\n")

        run = pwf.run(
            isLineInFile, filepath=path, line="Total wall time:", exact_match=False
        )

        self.assertTrue(run.outputs.line_found)


if __name__ == "__main__":
    unittest.main()
