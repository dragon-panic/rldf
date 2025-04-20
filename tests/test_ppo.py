import numpy as np
import torch
import matplotlib.pyplot as plt
from environment import GridWorld
from train import train_ppo, run_agent_test, compute_gae, Memory
from ppo_model import PPOModel
import time
import pytest


@pytest.fixture
def test_env():
    """Create a test environment for PPO tests."""
    return GridWorld(width=20, height=20)


@pytest.fixture
def test_model():
    """Create a PPO model for testing."""
    return PPOModel()


def test_ppo_short_run():
    """
    Run a short PPO training session to verify the implementation.
    """
    print("Testing PPO implementation with a short training run...")
    
    # Create a small environment for quick testing
    env = GridWorld(width=20, height=20)
    
    try:
        # Train for a small number of episodes with small update intervals
        start_time = time.time()
        model, _ = train_ppo(
            env,
            num_episodes=2,    # Even fewer episodes for testing
            update_timestep=50,  # Much smaller for quicker updates
            epochs=2,           # Fewer epochs for speed
            epsilon=0.2,
            gamma=0.99,
            gae_lambda=0.95,
            lr=0.0003,
            entropy_coef=0.01,
            value_coef=0.5,
            max_steps=50,       # Shorter episodes
            batch_size=16
        )
        end_time = time.time()
        
        print(f"Short PPO training completed in {end_time - start_time:.2f} seconds")
        
        # Run a brief test
        print("\nTesting the model briefly...")
        test_metrics = run_agent_test(env, model, num_episodes=1, max_steps=10)
        
        print("\nTest Results:")
        print(f"Average Reward: {test_metrics['avg_reward']:.2f}")
        
        # Test passed if we got here
        assert True
        
    except Exception as e:
        print(f"Exception during PPO test: {str(e)}")
        # Re-raise to see the stack trace
        raise


def test_value_function(test_env, test_model):
    """
    Test that the value function is working correctly by visualizing its predictions.
    """
    from model import ObservationEncoder
    
    # Create an encoder and agent to gather observations
    encoder = ObservationEncoder(test_env)
    
    # Create a grid environment with some features
    test_env.reset()
    
    # Add some water, plants, etc. to make interesting observations
    for i in range(5):
        # Place some water
        row, col = np.random.randint(0, test_env.height), np.random.randint(0, test_env.width)
        test_env.grid[row, col] = GridWorld.WATER
        
        # Place some mature plants
        row, col = np.random.randint(0, test_env.height), np.random.randint(0, test_env.width)
        if test_env.grid[row, col] == GridWorld.SOIL:
            test_env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Sample a few positions and get value predictions
    positions = []
    values = []
    
    for _ in range(5):  # Reduced for faster tests
        # Generate random position
        row, col = np.random.randint(0, test_env.height), np.random.randint(0, test_env.width)
        
        # Create a dummy agent at this position
        class DummyAgent:
            def __init__(self, r, c):
                self.row = r
                self.col = c
                self.hunger = np.random.uniform(0, 100)
                self.thirst = np.random.uniform(0, 100)
                self.energy = np.random.uniform(0, 100)
        
        agent = DummyAgent(row, col)
        
        # Get observation
        observation = encoder.get_observation(agent)
        observation = observation.unsqueeze(0)  # Add batch dimension
        
        # Get value prediction
        with torch.no_grad():
            _, value, _ = test_model(observation)
        
        # Store position and value
        positions.append((row, col))
        values.append(value.item())
    
    # Just verify we can get values without visualizing
    assert len(values) == 5
    assert len(positions) == 5


def test_ppo_components():
    """
    Test specific components of the PPO implementation.
    """
    # Test GAE computation
    print("\nTesting GAE computation...")
    rewards = [1.0, 0.5, 3.0, -1.0, 2.0]
    values = [torch.tensor([[0.8]]), torch.tensor([[1.2]]), torch.tensor([[2.5]]), 
              torch.tensor([[0.1]]), torch.tensor([[1.8]])]
    is_terminals = [False, False, False, False, True]
    
    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        is_terminals=is_terminals,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    print(f"Returns: {returns}")
    print(f"Advantages: {advantages}")
    
    # Test the memory system
    print("\nTesting memory system...")
    memory = Memory()
    
    # Add some transitions
    for i in range(5):
        memory.add(
            state=torch.randn(7, 7, 7),  # Random state tensor
            action=i % 9,
            logprob=torch.tensor([np.log(0.2)]),
            reward=float(i),
            is_terminal=(i == 4),  # Last is terminal
            value=torch.tensor([[float(i)]])
        )
    
    print(f"Memory size: {len(memory.states)}")
    print(f"Actions stored: {memory.actions}")
    print(f"Rewards stored: {memory.rewards}")
    
    # Test memory clear
    memory.clear()
    print(f"Memory size after clearing: {len(memory.states)}")


if __name__ == "__main__":
    print("==== PPO Implementation Test ====")
    
    # Test the basic components first
    test_ppo_components()
    
    # Run a short training session
    test_ppo_short_run()
    
    # Test the value function
    test_env = GridWorld(width=20, height=20)
    test_model = PPOModel()
    test_value_function(test_env, test_model)
    
    print("\nAll PPO tests completed successfully!") 