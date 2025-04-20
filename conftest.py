import sys
import os

# Add parent directory to Python path so tests can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests')))

# NOTE: This file is kept for backwards compatibility 
# The actual test configuration is in tests/conftest.py 