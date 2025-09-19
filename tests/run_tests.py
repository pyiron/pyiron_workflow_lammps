#!/usr/bin/env python
"""
Test runner script for pyiron_workflow_atomistics.

This script provides a convenient way to run tests with different configurations.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, parallel=False):
    """
    Run tests with specified configuration.

    Parameters
    ----------
    test_type : str
        Type of tests to run ('unit', 'integration', 'all')
    verbose : bool
        Run tests in verbose mode
    coverage : bool
        Generate coverage report
    parallel : bool
        Run tests in parallel
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test path based on type
    if test_type == "unit":
        cmd.append("tests/unit/")
    elif test_type == "integration":
        cmd.append("tests/integration/")
    else:  # all
        cmd.append("tests/")

    # Add options
    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(
            [
                "--cov=pyiron_workflow_atomistics",
                "--cov-report=html",
                "--cov-report=term",
            ]
        )

    if parallel:
        cmd.extend(["-n", "auto"])

    # Add markers to exclude slow tests by default
    if test_type == "unit":
        cmd.extend(["-m", "not slow"])

    print(f"Running command: {' '.join(cmd)}")

    # Run tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run pyiron_workflow_atomistics tests")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument("--slow", action="store_true", help="Include slow tests")

    args = parser.parse_args()

    # Check if we're in the right directory
    if not Path("pyiron_workflow_atomistics").exists():
        print("Error: Please run this script from the project root directory")
        sys.exit(1)

    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
    )

    if args.slow and args.type == "unit":
        print("\nRunning slow tests...")
        slow_cmd = ["python", "-m", "pytest", "tests/unit/", "-m", "slow"]
        if args.verbose:
            slow_cmd.append("-v")
        subprocess.run(slow_cmd)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
