import pytest
import torch
import numpy as np
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder, AgentCNN


def test_observation_encoder(env, agent):
    """Test that the observation encoder creates properly shaped tensors."""
    encoder = ObservationEncoder(env)
    observation = encoder.get_observation(agent)
    
    # Check shape (channels, height, width)
    assert observation.shape == (7, 7, 7)
    
    # Check tensor properties
    assert isinstance(observation, torch.Tensor)
    assert observation.dtype == torch.float32
    
    # Check that agent's status is encoded in the right channels
    hunger_channel = observation[4, :, :]
    thirst_channel = observation[5, :, :]
    energy_channel = observation[6, :, :]
    
    # All values should be the same in these channels
    assert torch.all(hunger_channel == agent.hunger / 100.0)
    assert torch.all(thirst_channel == agent.thirst / 100.0)
    assert torch.all(energy_channel == agent.energy / 100.0)


def test_agent_cnn():
    """Test that the AgentCNN model runs and produces valid outputs."""
    # Create a model with default parameters
    model = AgentCNN()
    
    # Create a batch of dummy observations (batch_size=2, channels=7, height=7, width=7)
    dummy_obs = torch.rand(2, 7, 7, 7)
    
    # Forward pass
    output = model(dummy_obs)
    
    # Check output shape (batch_size, num_actions)
    assert output.shape == (2, 9)
    
    # Check that outputs are valid probabilities
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)
    
    # Check that probabilities sum to 1 for each batch item
    assert torch.allclose(output.sum(dim=1), torch.tensor([1.0, 1.0]))


def test_model_integration(env, agent):
    """Test the integration between the observation encoder and model."""
    encoder = ObservationEncoder(env)
    model = AgentCNN()
    
    # Get observation from the environment
    observation = encoder.get_observation(agent)
    
    # Add batch dimension
    observation = observation.unsqueeze(0)
    
    # Forward pass
    output = model(observation)
    
    # Check output shape and properties
    assert output.shape == (1, 9)
    assert torch.all(output >= 0)
    assert torch.all(output <= 1)
    assert torch.isclose(output.sum(), torch.tensor(1.0))


if __name__ == "__main__":
    # Create test instances
    env = GridWorld(width=20, height=20)
    agent = Agent(env, start_row=5, start_col=5)
    
    # Run tests directly
    test_observation_encoder(env, agent)
    test_agent_cnn()
    test_model_integration(env, agent)
    
    print("All tests passed!") 