import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from environment import GridWorld


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


class AgentCNN(nn.Module):
    """
    A simple convolutional neural network for agent decision-making.
    Input: 7x7 grid with multiple channels
    Output: Action probabilities
    """
    
    def __init__(self, num_channels=7, grid_size=7, num_actions=9):
        """
        Initialize the CNN model.
        
        Args:
            num_channels: Number of input channels
            grid_size: Size of the grid (e.g., 7 for a 7x7 grid)
            num_actions: Number of possible actions
        """
        super(AgentCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate the size of the flattened features
        # After 2 conv layers with padding, the size remains the same
        flattened_size = 32 * grid_size * grid_size
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.fc2 = nn.Linear(256, num_actions)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, grid_size, grid_size)
            
        Returns:
            torch.Tensor: Action probabilities
        """
        # Apply convolution layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        
        # Output layer (action probabilities)
        action_probs = F.softmax(self.fc2(x), dim=1)
        
        return action_probs


def test_model_forward_pass():
    """
    Test the model with a dummy observation to ensure forward pass works.
    """
    # Create a dummy environment and agent
    env = GridWorld(width=20, height=20)
    
    # Create a dummy agent (position doesn't matter for this test)
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
    
    # Create the model
    model = AgentCNN()
    
    # Add batch dimension for the model
    observation = observation.unsqueeze(0)  # Shape: (1, num_channels, grid_size, grid_size)
    
    # Forward pass
    action_probs = model(observation)
    
    print(f"Observation shape: {observation.shape}")
    print(f"Action probabilities shape: {action_probs.shape}")
    print(f"Action probabilities: {action_probs}")
    
    # The output should be a vector of 9 probabilities (one for each action)
    assert action_probs.shape == (1, 9), f"Expected shape (1, 9), got {action_probs.shape}"
    assert torch.isclose(action_probs.sum(), torch.tensor(1.0)), f"Sum of probabilities should be 1, got {action_probs.sum().item()}"
    
    print("Forward pass test passed!")


if __name__ == "__main__":
    test_model_forward_pass() 