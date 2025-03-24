#!/usr/bin/env python3
"""
Run all unit tests for Sequential VSA-OGM.

This script runs all unit tests for the Sequential VSA-OGM system.
"""

import unittest
import sys
import os

# Add parent directory to path to import src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests():
    """Run all unit tests."""
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)))
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Return exit code based on test result
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
