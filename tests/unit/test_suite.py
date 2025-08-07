import unittest
import sys
import os

# Add the module path to sys.path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pyiron_workflow_lammps'))

# Import all test modules
from test_lammps import *
from test_engine import *
from test_generic import *
from test_version import *
from test_tests import *


def create_test_suite():
    """Create a comprehensive test suite for all unit tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases from each module
    test_suite.addTest(unittest.makeSuite(TestWriteLammpsStructure))
    test_suite.addTest(unittest.makeSuite(TestWriteLammpsInput))
    test_suite.addTest(unittest.makeSuite(TestGetSpeciesMap))
    test_suite.addTest(unittest.makeSuite(TestArraysToAseAtoms))
    test_suite.addTest(unittest.makeSuite(TestParseLammpsOutput))
    test_suite.addTest(unittest.makeSuite(TestGetStructureSpeciesLists))
    test_suite.addTest(unittest.makeSuite(TestLammpsJob))
    test_suite.addTest(unittest.makeSuite(TestLammpsCalculatorFn))
    
    test_suite.addTest(unittest.makeSuite(TestLammpsEngine))
    
    test_suite.addTest(unittest.makeSuite(TestStorage))
    test_suite.addTest(unittest.makeSuite(TestShellOutput))
    test_suite.addTest(unittest.makeSuite(TestVarType))
    test_suite.addTest(unittest.makeSuite(TestFileObject))
    test_suite.addTest(unittest.makeSuite(TestShell))
    test_suite.addTest(unittest.makeSuite(TestIsLineInFile))
    test_suite.addTest(unittest.makeSuite(TestCreateWorkingDirectory))
    test_suite.addTest(unittest.makeSuite(TestDeleteFilesRecursively))
    test_suite.addTest(unittest.makeSuite(TestCompressDirectory))
    test_suite.addTest(unittest.makeSuite(TestSubmitToSlurm))
    test_suite.addTest(unittest.makeSuite(TestRemoveDir))
    
    test_suite.addTest(unittest.makeSuite(TestVersion))
    
    test_suite.addTest(unittest.makeSuite(TestModuleImports))
    
    return test_suite


def run_tests():
    """Run all unit tests and return the result."""
    # Create test suite
    suite = create_test_suite()
    
    # Create test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run tests
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"{'='*60}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 