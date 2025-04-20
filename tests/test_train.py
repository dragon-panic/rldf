import pytest
import torch
import numpy as np
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder
from reinforce_model import REINFORCEModel
from train import RLAgent, calculate_reward, train_reinforce


def test_rl_agent(env):
    """Test that the RLAgent properly extends the base Agent class."""
    # Create a model and encoder
    encoder = ObservationEncoder(env)
    model = REINFORCEModel()
    
    # Create an RL agent
    rl_agent = RLAgent(env, model, encoder, start_row=5, start_col=5)
    
    # Check inheritance
    assert isinstance(rl_agent, Agent)
    
    # Check additional attributes
    assert hasattr(rl_agent, 'model')
    assert hasattr(rl_agent, 'encoder')
    assert hasattr(rl_agent, 'log_probs')
    assert hasattr(rl_agent, 'rewards')
    assert hasattr(rl_agent, 'is_alive')


def test_decide_action(env):
    """Test that the RL agent can decide actions."""
    # Create a model and encoder
    encoder = ObservationEncoder(env)
    model = REINFORCEModel()
    
    # Create an RL agent
    rl_agent = RLAgent(env, model, encoder, start_row=5, start_col=5)
    
    # Test decide_action
    action = rl_agent.decide_action()
    
    # Check that the action is valid
    assert 0 <= action < 9


def test_reward_function(env):
    """Test that the reward function returns reasonable values."""
    # Create a model and encoder
    encoder = ObservationEncoder(env)
    model = REINFORCEModel()
    
    # Create an RL agent
    rl_agent = RLAgent(env, model, encoder, start_row=5, start_col=5)
    
    # Test successful and unsuccessful actions
    success_reward = calculate_reward(rl_agent, Agent.PLANT_SEED, True)
    failure_reward = calculate_reward(rl_agent, Agent.PLANT_SEED, False)
    
    # Check that success gives a higher reward than failure
    assert success_reward > failure_reward
    
    # Test rewards for different actions
    plant_reward = calculate_reward(rl_agent, Agent.PLANT_SEED, True)
    harvest_reward = calculate_reward(rl_agent, Agent.HARVEST, True)
    
    # Harvesting should give more reward than planting
    assert harvest_reward > plant_reward
    
    # Test rewards with different vital states
    rl_agent.hunger = 90  # Critical hunger
    hungry_reward = calculate_reward(rl_agent, Agent.MOVE_NORTH, True)
    
    rl_agent.hunger = 20  # Low hunger
    full_reward = calculate_reward(rl_agent, Agent.MOVE_NORTH, True)
    
    # Being hungry should give a lower reward
    assert hungry_reward < full_reward


def test_short_training_run(env):
    """Test that the training function runs without errors."""
    # Run a very short training session (2 episodes, 10 steps each)
    model, history = train_reinforce(env, num_episodes=2, max_steps=10)
    
    # Check that the history contains the expected keys
    assert 'rewards' in history
    assert 'lengths' in history
    assert 'moving_avg' in history
    
    # Check that the history has the right length
    assert len(history['rewards']) == 2
    assert len(history['lengths']) == 2
    assert len(history['moving_avg']) == 2
    
    # Check that the model is of the right type
    assert isinstance(model, REINFORCEModel)


if __name__ == "__main__":
    # Create test instances
    env = GridWorld(width=20, height=20)
    
    # Run tests directly
    test_rl_agent(env)
    test_decide_action(env)
    test_reward_function(env)
    test_short_training_run(env)
    
    print("All tests passed!") 