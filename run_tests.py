#!/usr/bin/env python3
"""
Test runner script for pyiron_workflow_lammps.

This script provides an easy way to run all unit tests for the module.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_pytest_tests(verbose=False, coverage=False, specific_file=None):
    """Run tests using pytest."""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=pyiron_workflow_lammps", "--cov-report=html"])

    if specific_file:
        cmd.append(specific_file)
    else:
        cmd.append("tests/unit/")

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=Path(__file__).parent)
    return result.returncode


def run_unittest_suite():
    """Run tests using the unittest test suite."""
    test_suite_path = Path(__file__).parent / "tests" / "unit" / "test_suite.py"

    if not test_suite_path.exists():
        print(f"Error: Test suite not found at {test_suite_path}")
        return 1

    print(f"Running test suite: {test_suite_path}")
    result = subprocess.run([sys.executable, str(test_suite_path)], check=False)
    return result.returncode


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run unit tests for pyiron_workflow_lammps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py -v                 # Run with verbose output
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --unittest         # Use unittest instead of pytest
  python run_tests.py tests/unit/test_lammps.py  # Run specific test file
        """,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Run tests with verbose output"
    )

    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )

    parser.add_argument(
        "--unittest",
        action="store_true",
        help="Use unittest test suite instead of pytest",
    )

    parser.add_argument(
        "test_file", nargs="?", help="Specific test file to run (optional)"
    )

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: Please run this script from the project root directory")
        print("Current directory:", os.getcwd())
        return 1

    # Check if tests directory exists
    if not Path("tests/unit").exists():
        print("Error: Tests directory not found")
        return 1

    print("=" * 60)
    print("pyiron_workflow_lammps Test Runner")
    print("=" * 60)

    try:
        if args.unittest:
            return_code = run_unittest_suite()
        else:
            return_code = run_pytest_tests(
                verbose=args.verbose,
                coverage=args.coverage,
                specific_file=args.test_file,
            )

        print("=" * 60)
        if return_code == 0:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed!")
        print("=" * 60)

        return return_code

    except KeyboardInterrupt:
        print("\n\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
