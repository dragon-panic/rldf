import logging

# Set up logging
logger = logging.getLogger("mock_visualize")

class MockVisualizer:
    """Mock version of GameVisualizer that doesn't create actual pygame windows.
    
    This class provides the same interface as GameVisualizer but doesn't 
    initialize pygame or create any windows, making it suitable for tests.
    """
    
    def __init__(self, grid_world, cell_size=30, info_width=300):
        """
        Initialize the mock visualizer.
        
        Args:
            grid_world: GridWorld instance
            cell_size: Size of each cell in pixels (ignored in mock)
            info_width: Width of the info panel in pixels (ignored in mock)
        """
        self.grid_world = grid_world
        self.cell_size = cell_size
        self.info_width = info_width
        self.agent = None
        logger.debug("MockVisualizer initialized")
    
    def set_agent(self, agent):
        """Set the agent to be visualized."""
        self.agent = agent
        logger.debug(f"Agent set at position {agent.get_position()}")
    
    def update(self):
        """Update the display (mock implementation)."""
        if self.agent:
            status = self.agent.get_status()
            logger.debug(f"Update display - Agent at {status['position']}, " 
                        f"Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
    
    def run_simulation(self, frames_per_step=30):
        """Mock simulation running."""
        logger.debug("Mock simulation run requested (not actually running)")
        return 