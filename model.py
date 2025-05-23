import numpy as np
import torch
import torch.nn as nn
import logging
from environment import GridWorld

# Get the logger that's configured in train.py/main.py
logger = logging.getLogger(__name__)


class ObservationEncoder:
    """
    Encodes the local environment around the agent for neural network input.
    Extracts a 7x7 grid around the agent and encodes different grid properties as channels.
    """
    
    def __init__(self, environment):
        """
        Initialize the observation encoder.
        
        Args:
            environment: A GridWorld instance
        """
        self.environment = environment
        self.observation_size = 7  # 7x7 grid around the agent
        
        # Define the channels for encoding
        self.num_channels = 7
        # Channel meanings:
        # 0: Cell type (0=empty, 1=water, 2=soil, 3=plant)
        # 1: Water level (0-10)
        # 2: Soil fertility (0-10)
        # 3: Plant state (0=none, 1=seed, 2=growing, 3=mature)
        # 4: Agent's hunger level (0-100)
        # 5: Agent's thirst level (0-100)
        # 6: Agent's energy level (0-100)
    
    def get_observation(self, agent):
        """
        Get an encoded observation around the agent.
        
        Args:
            agent: The agent to center the observation around
            
        Returns:
            torch.Tensor: A tensor of shape (num_channels, observation_size, observation_size)
        """
        # Initialize the observation tensor
        observation = np.zeros((self.num_channels, self.observation_size, self.observation_size))
        
        # Get the agent's position
        agent_row, agent_col = agent.row, agent.col
        
        # Define the radius around the agent
        radius = self.observation_size // 2
        
        # Fill the observation tensor with data from the environment
        for i in range(self.observation_size):
            for j in range(self.observation_size):
                # Calculate the corresponding position in the environment
                env_row = agent_row + (i - radius)
                env_col = agent_col + (j - radius)
                
                # Check if the position is within bounds
                if (0 <= env_row < self.environment.height and 
                    0 <= env_col < self.environment.width):
                    # Fill in the channels with data
                    observation[0, i, j] = self.environment.grid[env_row, env_col]
                    observation[1, i, j] = self.environment.water_level[env_row, env_col] / 10.0  # Normalize to [0,1]
                    observation[2, i, j] = self.environment.soil_fertility[env_row, env_col] / 10.0  # Normalize to [0,1]
                    observation[3, i, j] = self.environment.plant_state[env_row, env_col] / 3.0  # Normalize to [0,1]
                else:
                    # For out-of-bounds cells, mark as invalid with -1
                    observation[0, i, j] = -1
                    observation[1, i, j] = -1
                    observation[2, i, j] = -1
                    observation[3, i, j] = -1
        
        # Add the agent's status to the observation
        observation[4, :, :] = agent.hunger / 100.0  # Normalize to [0,1]
        observation[5, :, :] = agent.thirst / 100.0  # Normalize to [0,1]
        observation[6, :, :] = agent.energy / 100.0  # Normalize to [0,1]
        
        # Convert to torch tensor
        return torch.FloatTensor(observation)


class RLModelBase(nn.Module):
    """
    Base class for all RL model architectures.
    Defines the common interface that all RL models should implement.
    """
    
    def __init__(self):
        super(RLModelBase, self).__init__()
    
    def forward(self, x):
        """
        Forward pass of the network. Must be implemented by subclasses.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, grid_size, grid_size)
            
        Returns:
            Implementation-dependent
        """
        raise NotImplementedError("Subclasses must implement forward()")


def test_observation_encoder():
    """
    Test the observation encoder with a dummy environment and agent.
    """
    # Create a dummy environment
    env = GridWorld(width=20, height=20)
    
    # Create a dummy agent
    class DummyAgent:
        def __init__(self):
            self.row = 10
            self.col = 10
            self.hunger = 50.0
            self.thirst = 50.0
            self.energy = 50.0
    
    agent = DummyAgent()
    
    # Create the observation encoder
    encoder = ObservationEncoder(env)
    
    # Get a sample observation
    observation = encoder.get_observation(agent)
    
    # Check the shape
    expected_shape = (7, 7, 7)  # (channels, height, width)
    assert observation.shape == expected_shape, f"Expected shape {expected_shape}, got {observation.shape}"
    
    logger.info("Observation encoder test passed!")


if __name__ == "__main__":
    # Set up basic logging configuration when this file is run directly
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    test_observation_encoder() 