import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from model import ObservationEncoder

# Get the logger that's configured in train.py/main.py
logger = logging.getLogger(__name__)

class REINFORCEModel(nn.Module):
    """
    A convolutional neural network for the REINFORCE algorithm.
    Input: 7x7 grid with multiple channels
    Output: Action probabilities
    
    This model has a single output head that produces a probability distribution over actions.
    """
    
    def __init__(self, num_channels=7, grid_size=7, num_actions=9):
        """
        Initialize the CNN model for REINFORCE.
        
        Args:
            num_channels: Number of input channels
            grid_size: Size of the grid (e.g., 7 for a 7x7 grid)
            num_actions: Number of possible actions
        """
        super(REINFORCEModel, self).__init__()
        
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


def test_reinforce_model():
    """
    Test the REINFORCE model with a dummy input.
    """
    # Create a dummy input tensor (batch_size=1, channels=7, grid_size=7x7)
    dummy_input = torch.randn(1, 7, 7, 7)
    
    # Create the model
    model = REINFORCEModel()
    
    # Forward pass
    action_probs = model(dummy_input)
    
    # Check the output shape and properties
    assert action_probs.shape == (1, 9), f"Expected shape (1, 9), got {action_probs.shape}"
    assert torch.isclose(action_probs.sum(), torch.tensor(1.0)), f"Sum of probabilities should be 1, got {action_probs.sum().item()}"
    
    logger.info("REINFORCE model test passed!")


if __name__ == "__main__":
    # Set up basic logging configuration when this file is run directly
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    test_reinforce_model() 