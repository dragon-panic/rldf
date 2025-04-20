import pytest
import logging
import sys
import os

# Add parent directory to path so tests can import project modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment import GridWorld
from agent import Agent

# Configure logging for tests
def pytest_addoption(parser):
    parser.addoption(
        "--rldf-log-level",
        action="store",
        default="WARNING",
        help="Set the logging level for tests (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.addoption(
        "--show-visualization",
        action="store_true",
        help="Enable actual pygame visualization during tests (not recommended for CI)"
    )
    parser.addoption(
        "--show-plots",
        action="store_true",
        help="Show matplotlib plots during tests (not recommended for CI)"
    )

def pytest_configure(config):
    """Configure the test environment"""
    # Make sure we can find our mock modules
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Configure headless mode (no visualization windows) by default
    if not config.getoption("--show-visualization"):
        # Set up mock pygame for all tests - now using the tests directory version
        try:
            from tests.mock_pygame import pygame
            sys.modules['pygame'] = pygame
        except ImportError:
            pass
    
    # Configure non-interactive matplotlib by default
    if not config.getoption("--show-plots"):
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pass

@pytest.fixture(scope="session", autouse=True)
def configure_logging(request):
    """Set up logging based on command line options."""
    log_level = request.config.getoption("--rldf-log-level").upper()
    level = getattr(logging, log_level, logging.WARNING)
    
    # Configure root logger
    logging.basicConfig(
        format='%(levelname)s - %(message)s',
        level=level
    )
    
    # Configure specific loggers used in tests
    for logger_name in ["farming_test", "mock_visualize", "mock_pygame"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    yield
    # Cleanup after tests if needed

@pytest.fixture
def env():
    """Create a small test environment."""
    return GridWorld(width=20, height=20, water_probability=0.1)

@pytest.fixture
def agent(env):
    """Create an agent in the test environment."""
    return Agent(env, start_row=5, start_col=5)

@pytest.fixture
def grid_world():
    """Create a GridWorld instance for testing."""
    return GridWorld(width=20, height=20, water_probability=0.1)

@pytest.fixture
def grid():
    """Create a GridWorld instance for resource testing."""
    return GridWorld(width=20, height=20, water_probability=0.1) 