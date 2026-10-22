# for development and testing only
# provide functionality, data types etc. that will be later moved to the workflow code
from __future__ import annotations

import fnmatch
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import flowrep as fr
from pyiron_snippets.logger import logger


class Storage:
    def _convert_to_dict(instance):
        # Get the attributes of the instance
        attributes = vars(instance)

        # Convert attributes to a dictionary
        result_dict = {
            key: value for key, value in attributes.items() if "_" not in key[0]
        }

        return result_dict


class ShellOutput(Storage):
    stdout: str
    stderr: str
    return_code: int
    dump: FileObject  # TODO: should be done in a specific lammps object
    log: FileObject


class VarType:
    def __init__(
        self,
        value=None,
        dat_type=None,
        label: str = None,
        store: int = 0,
        generic: bool = None,
        doc: str = None,
    ):
        self.value = value
        self.type = dat_type
        self.label = label
        self.store = store
        self.generic = generic
        self.doc = doc


class FileObject:
    def __init__(self, path=".", directory=None):
        if directory is None:
            self._path = Path(path)
        else:
            self._path = Path(directory) / Path(path)

    def __repr__(self):
        return f"FileObject: {self._path} {self.is_file}"

    @property
    def path(self):
        # Note conversion to string (needed to satisfy glob which is used e.g. in dump parser)
        return str(self._path)

    @property
    def is_file(self):
        return self._path.is_file()

    @property
    def name(self):
        return self._path.name


@fr.atomic("output")
def shell(
    command: str,
    working_directory: str | None = None,
    environment: dict[str, str] | None = None,
    arguments: list[str] | None = None,
    after: object = None,
) -> ShellOutput:
    """
    Run a shell command in the specified working directory.

    Args:
        command (str): The command to execute.
        working_directory (str | None, optional): The working directory. Defaults to None.
        environment (Optional[dict[str, str]], optional): Environment variables to set. Defaults to None.
        arguments (Optional[list[str]], optional): Command line arguments appended
            to ``command``. Defaults to None.
        after: An ordering token. It is never read, but accepting it lets a
            caller create a data edge that forces this node to run after
            another. pyiron_workflow 0.19 has no execution signals, so
            ordering must come from data flow.

    Returns:
        ShellOutput: Object containing stdout, stderr, and return code.

    Note:
        Never calls ``os.chdir``: ``subprocess.run``'s ``cwd`` scopes the child
        process, and pyiron_workflow >= 0.19 runs sibling nodes on separate
        threads, so changing the process cwd would race.
    """
    logger.info(f"shell is in {working_directory}")
    if environment is None:
        environment = {}
    if arguments is None:
        arguments = []
    environ = dict(os.environ)
    environ.update({k: str(v) for k, v in environment.items()})
    # With shell=True the command must be one string: a list silently drops
    # every element after the first on POSIX.
    full_command = " ".join([command, *map(str, arguments)]) if arguments else command
    proc = subprocess.run(
        full_command,
        check=False,
        capture_output=True,
        cwd=working_directory,
        encoding="utf8",
        env=environ,
        shell=True,
    )
    output = ShellOutput()
    output.stdout = proc.stdout
    output.stderr = proc.stderr
    output.return_code = proc.returncode
    return output


@fr.atomic("line_found")
def isLineInFile(filepath: str, line: str, exact_match: bool = True) -> bool:
    """
    Check if a specific line exists in a file.

    Args:
        filepath (str): Path to the file to search in.
        line (str): The line to search for.
        exact_match (bool, optional): If True, the line must match exactly. If False,
                                     the line can be a substring of any line in the file.
                                     Defaults to True.

    Returns:
        bool: True if the line is found, False otherwise.
    """
    line_found = False  # Initialize the result as False
    try:
        with open(filepath) as file:
            for file_line in file:
                if exact_match and line == file_line.strip():
                    line_found = True
                    break  # Exit loop if the line is found
                elif not exact_match and line in file_line:
                    line_found = True
                    break  # Exit loop if a partial match is found
    except FileNotFoundError:
        logger.error(f"File '{filepath}' not found.")
    return line_found


@fr.atomic("working_directory")
def create_WorkingDirectory(working_directory: str, quiet: bool = False) -> str:
    """
    Create a working directory if it doesn't exist.

    Args:
        working_directory (str): Path to the directory to create.
        quiet (bool, optional): If True, suppress warnings. Defaults to False.

    Returns:
        str: Path to the created or existing directory.
    """
    # Check if working_directory exists
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
        logger.info(f"made directory '{working_directory}'")
    elif not quiet:
        logger.warning(
            f"Directory '{working_directory}' already exists. Existing files may be overwritten."
        )
    else:
        logger.info(
            f"Directory '{working_directory}' already exists. Existing files will not be overwritten."
        )

    return working_directory


@fr.atomic("working_directory")
def delete_files_recursively(working_directory: str, files_to_be_deleted: list[str]):
    """
    Recursively delete specific files in a directory and its subdirectories.

    Args:
        working_directory (str): The directory to search for files.
        files_to_be_deleted (list[str]): List of filenames to delete.
    """
    if not os.path.isdir(working_directory):
        logger.error(f"Error: {working_directory} is not a valid directory.")
    else:
        for root, _, files in os.walk(working_directory):
            for file in files:
                if file in files_to_be_deleted:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted: {file_path}")
                    except Exception as e:
                        logger.error(f"Error deleting {file_path}: {e}")
    return working_directory


@fr.atomic("compressed_file")
def compress_directory(
    directory_path: str,
    exclude_files=[],
    exclude_file_patterns=[],
    print_message=True,
    inside_dir=True,
    actually_compress=True,
):
    """
    Compresses a directory and its contents into a tarball with gzip compression.

    Parameters:
        directory_path (str): The path of the directory to compress.
        exclude_files (list, optional): A list of filenames to exclude from the compression. Defaults to an empty list.
        exclude_file_patterns (list, optional): A list of file patterns (glob patterns) to match against filenames and exclude from the compression. Defaults to an empty list.
        print_message (bool, optional): Determines whether to print a message indicating the compression. Defaults to True.
        inside_dir (bool, optional): Determines whether the output tarball should be placed inside the source directory or in the same directory as the source directory. Defaults to True.

    Usage:
        # Compress a directory and place the resulting tarball inside the directory
        compress_directory("/path/to/source_directory")

        # Compress a directory and place the resulting tarball in the same directory as the source directory
        compress_directory("/path/to/source_directory", inside_dir=False)

        # Compress a directory and exclude specific files from the compression
        compress_directory("/path/to/source_directory", exclude_files=["file1.txt", "file2.jpg"])

        # Compress a directory and exclude files matching specific file patterns from the compression
        compress_directory("/path/to/source_directory", exclude_file_patterns=["*.txt", "*.log"], inside_dir=False)

    Note:
        - The function creates a tarball with gzip compression of the directory and its contents.
        - The resulting tarball will be placed either inside the source directory (if inside_dir is True) or in the same directory as the source directory (if inside_dir is False).
        - Files specified in the `exclude_files` list and those matching the `exclude_file_patterns` will be excluded from the compression.
        - The `print_message` parameter controls whether a message indicating the compression is printed. By default, it is set to True.
    """
    if actually_compress:
        if inside_dir:
            output_file = os.path.join(
                directory_path, os.path.basename(directory_path) + ".tar.gz"
            )
        else:
            output_file = os.path.join(
                os.path.dirname(directory_path),
                os.path.basename(directory_path) + ".tar.gz",
            )
        with tarfile.open(output_file, "w:gz") as tar:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Exclude the output tarball from being added
                    if file_path == output_file:
                        continue
                    if any(
                        fnmatch.fnmatch(file, pattern)
                        for pattern in exclude_file_patterns
                    ):
                        continue
                    if file in exclude_files:
                        continue
                    arcname = os.path.join(
                        os.path.basename(directory_path),
                        os.path.relpath(file_path, directory_path),
                    )
                    tar.add(file_path, arcname=arcname)
                    # tar.add(file_path, arcname=os.path.relpath(file_path, directory_path))
                    # print(f"{file} added")
        if print_message:
            logger.info(f"compress_directory: compressed directory at {directory_path}")
    else:
        output_file = None
        logger.info("compress_directory: no compression")
    return output_file


@fr.atomic("compressed_file")
def remove_dir(directory_path, actually_remove=False):
    if actually_remove:
        shutil.rmtree(directory_path, ignore_errors=True)
    return directory_path
