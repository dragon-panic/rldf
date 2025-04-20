import pytest
import torch
import numpy as np
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder
from reinforce_model import REINFORCEModel
from model_based_agent import ModelBasedAgent
import os


def find_valid_start_position(env, preferred_row=None, preferred_col=None):
    """
    Find a valid starting position (non-water cell) for an agent.
    
    Args:
        env: GridWorld environment
        preferred_row: Preferred row coordinate (will try to find nearest valid cell if this is water)
        preferred_col: Preferred column coordinate (will try to find nearest valid cell if this is water)
        
    Returns:
        tuple: (row, col) of valid starting position
    """
    # Use center of grid if no preference given
    if preferred_row is None:
        preferred_row = env.height // 2
    if preferred_col is None:
        preferred_col = env.width // 2
    
    # Ensure coordinates are in bounds
    preferred_row = min(max(0, preferred_row), env.height - 1)
    preferred_col = min(max(0, preferred_col), env.width - 1)
    
    # If preferred position is not water, return it
    if env.grid[preferred_row, preferred_col] != GridWorld.WATER:
        return preferred_row, preferred_col
    
    # Find the nearest non-water cell
    search_radius = 1
    
    # Expand search radius until we find a non-water cell
    while search_radius < max(env.height, env.width):
        # Check cells in a square around the preferred position
        for r_offset in range(-search_radius, search_radius + 1):
            for c_offset in range(-search_radius, search_radius + 1):
                # Only check the perimeter of the square
                if abs(r_offset) == search_radius or abs(c_offset) == search_radius:
                    r = preferred_row + r_offset
                    c = preferred_col + c_offset
                    
                    # Ensure position is within bounds
                    if (0 <= r < env.height and 
                        0 <= c < env.width and
                        env.grid[r, c] != GridWorld.WATER):
                        
                        return r, c
        
        # Increase search radius if needed
        search_radius += 1
    
    # If still no valid position found, do a full grid search
    for r in range(env.height):
        for c in range(env.width):
            if env.grid[r, c] != GridWorld.WATER:
                return r, c
    
    # This should never happen unless the entire grid is water
    raise ValueError("Could not find a valid starting position - grid is all water")


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
    """Test that the REINFORCEModel runs and produces valid outputs."""
    # Create a model with default parameters
    model = REINFORCEModel()
    
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
    model = REINFORCEModel()
    
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


def test_model_based_agent_status_updates():
    """Test that model-based agent's hunger and thirst increase when idle."""
    print("\nTesting model-based agent status updates:")
    
    # Create a simple environment
    env = GridWorld(width=5, height=5, water_probability=0.1)
    
    # Find a valid starting position
    start_row, start_col = find_valid_start_position(env)
    
    # Create the model-based agent
    agent = ModelBasedAgent(env, start_row=start_row, start_col=start_col)
    agent.hunger = 0.0
    agent.thirst = 0.0
    agent.energy = 100.0
    
    # Initial values
    initial_hunger = agent.hunger
    initial_thirst = agent.thirst
    
    print(f"  Initial values - Hunger: {initial_hunger}, Thirst: {initial_thirst}")
    
    # Call update_status multiple times without taking any actions
    num_updates = 10
    for i in range(num_updates):
        agent.update_status()
        if i % 3 == 0:  # Print every few updates
            print(f"  After {i+1} updates - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}")
    
    # Verify hunger and thirst increased
    assert agent.hunger > initial_hunger, f"Hunger should increase when idle (was {initial_hunger}, now {agent.hunger})"
    assert agent.thirst > initial_thirst, f"Thirst should increase when idle (was {initial_thirst}, now {agent.thirst})"
    
    print(f"  Final values - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}")
    print("  Status update test passed!")
    
    # Now test with a full AI step
    agent.hunger = 40.0
    agent.thirst = 40.0
    initial_hunger = agent.hunger
    initial_thirst = agent.thirst
    
    print("\n  Testing AI step:")
    print(f"  Before step - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}")
    
    result = agent.step_ai()
    
    print(f"  After step - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}")
    print(f"  AI action: {agent.current_task}, Success: {result['success']}")
    
    # Verify hunger and thirst also increase with AI step
    assert agent.hunger > initial_hunger, f"Hunger should increase after AI step (was {initial_hunger}, now {agent.hunger})"
    
    # Check thirst changes based on agent actions
    if agent.current_task == "Drinking" and result['success']:
        # If the agent was drinking successfully, thirst should decrease
        assert agent.thirst < initial_thirst, f"Thirst should decrease after successful drinking (was {initial_thirst}, now {agent.thirst})"
    else:
        # For all other actions, thirst should increase
        assert agent.thirst > initial_thirst, f"Thirst should increase after AI step (was {initial_thirst}, now {agent.thirst})"
    
    print("  AI step test passed!")


if __name__ == "__main__":
    # Create test instances
    env = GridWorld(width=20, height=20)
    
    # Find a valid starting position
    start_row, start_col = find_valid_start_position(env)
    
    # Create agent
    agent = Agent(env, start_row=start_row, start_col=start_col)
    
    # Run tests directly
    test_observation_encoder(env, agent)
    test_agent_cnn()
    test_model_integration(env, agent)
    test_model_based_agent_status_updates()
    
    print("All tests passed!") 