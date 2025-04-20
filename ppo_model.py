import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from model import ObservationEncoder

# Get the logger that's configured in train.py/main.py
logger = logging.getLogger(__name__)

class PPOModel(nn.Module):
    """
    CNN model for PPO algorithm with both policy and value function heads.
    Input: 7x7 grid with multiple channels
    Output: Action probabilities and value function
    
    This model has a dual-head architecture:
    1. Policy head: produces a probability distribution over actions
    2. Value head: estimates the value of the current state
    """
    
    def __init__(self, num_channels=7, grid_size=7, num_actions=9):
        """
        Initialize the CNN model for PPO.
        
        Args:
            num_channels: Number of input channels
            grid_size: Size of the grid (e.g., 7 for a 7x7 grid)
            num_actions: Number of possible actions
        """
        super(PPOModel, self).__init__()
        
        # Define the convolutional layers (shared network)
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        # Calculate the size of the flattened features
        flattened_size = 32 * grid_size * grid_size
        
        # Define the shared fully connected layer
        self.fc_shared = nn.Linear(flattened_size, 256)
        
        # Define the policy head
        self.policy_head = nn.Linear(256, num_actions)
        
        # Define the value head
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, grid_size, grid_size)
            
        Returns:
            tuple: (action_probs, state_values, action_logits)
                action_probs: Probabilities for each action
                state_values: Estimated value of the state
                action_logits: Raw (non-softmaxed) action scores for more stable training
        """
        # Shared layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Shared dense layer
        x = torch.relu(self.fc_shared(x))
        
        # Policy head (action probabilities)
        action_logits = self.policy_head(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        # Value head (state value)
        state_values = self.value_head(x)
        
        return action_probs, state_values, action_logits

    def get_policy(self, x):
        """
        Get only the policy output (action probabilities).
        Useful for inference when value isn't needed.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, grid_size, grid_size)
            
        Returns:
            torch.Tensor: Action probabilities
        """
        action_probs, _, _ = self.forward(x)
        return action_probs
    
    def get_value(self, x):
        """
        Get only the value function output.
        
        Args:
            x: Input tensor of shape (batch_size, num_channels, grid_size, grid_size)
            
        Returns:
            torch.Tensor: State value estimate
        """
        _, state_values, _ = self.forward(x)
        return state_values


def test_ppo_model():
    """
    Test the PPO model with a dummy input.
    """
    # Create a dummy input tensor (batch_size=1, channels=7, grid_size=7x7)
    dummy_input = torch.randn(1, 7, 7, 7)
    
    # Create the model
    model = PPOModel()
    
    # Forward pass
    action_probs, state_values, action_logits = model(dummy_input)
    
    # Check the action probabilities
    assert action_probs.shape == (1, 9), f"Expected shape (1, 9), got {action_probs.shape}"
    assert torch.isclose(action_probs.sum(), torch.tensor(1.0)), f"Sum of probabilities should be 1, got {action_probs.sum().item()}"
    
    # Check the state values
    assert state_values.shape == (1, 1), f"Expected shape (1, 1), got {state_values.shape}"
    
    # Check the action logits
    assert action_logits.shape == (1, 9), f"Expected shape (1, 9), got {action_logits.shape}"
    
    # Test convenience methods
    policy_only = model.get_policy(dummy_input)
    assert torch.allclose(policy_only, action_probs), "get_policy should return the same as the first element of forward()"
    
    value_only = model.get_value(dummy_input)
    assert torch.allclose(value_only, state_values), "get_value should return the same as the second element of forward()"
    
    logger.info("PPO model test passed!")


if __name__ == "__main__":
    # Set up basic logging configuration when this file is run directly
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    test_ppo_model() 