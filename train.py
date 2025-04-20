import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder, AgentCNN
import matplotlib.pyplot as plt
from collections import deque


class RLAgent(Agent):
    """
    An agent that uses a neural network to make decisions.
    Extends the basic Agent class with RL capabilities.
    """
    
    def __init__(self, environment, model, encoder, start_row=0, start_col=0):
        """
        Initialize the RL agent.
        
        Args:
            environment: A GridWorld instance
            model: The neural network model for decision making
            encoder: An ObservationEncoder instance
            start_row: Initial row position
            start_col: Initial column position
        """
        super().__init__(environment, start_row, start_col)
        self.model = model
        self.encoder = encoder
        
        # For tracking episode data
        self.log_probs = []
        self.rewards = []
        self.episode_length = 0
        self.is_alive = True
    
    def decide_action(self):
        """
        Decide the next action based on the neural network.
        
        Returns:
            int: The action to take
        """
        # Get the current observation
        observation = self.encoder.get_observation(self)
        
        # Add batch dimension and send to device
        observation = observation.unsqueeze(0)
        
        # Get action probabilities from the model
        with torch.no_grad():
            action_probs = self.model(observation)
        
        # Sample an action from the probability distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        
        return action.item()
    
    def decide_action_training(self):
        """
        Decide action during training and store log probability for training.
        
        Returns:
            int: The action to take
        """
        # Get the current observation
        observation = self.encoder.get_observation(self)
        
        # Add batch dimension and send to device
        observation = observation.unsqueeze(0)
        
        # Get action probabilities from the model
        action_probs = self.model(observation)
        
        # Sample an action from the probability distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        
        # Store log probability for training
        log_prob = action_distribution.log_prob(action)
        self.log_probs.append(log_prob)
        
        return action.item()
    
    def reset_episode(self):
        """Reset the agent for a new episode."""
        # Reset position and status
        self.row = np.random.randint(0, self.environment.height)
        self.col = np.random.randint(0, self.environment.width)
        
        # Make sure we're not starting on water
        while self.environment.grid[self.row, self.col] == GridWorld.WATER:
            self.row = np.random.randint(0, self.environment.height)
            self.col = np.random.randint(0, self.environment.width)
        
        # Reset status attributes
        self.energy = 100.0
        self.health = 100.0
        self.hunger = 0.0
        self.thirst = 0.0
        self.seeds = 10
        
        # Reset episode tracking
        self.log_probs = []
        self.rewards = []
        self.episode_length = 0
        self.is_alive = True


def calculate_reward(agent, action, success, prev_state=None):
    """
    Calculate the reward for an action.
    
    Args:
        agent: The agent that performed the action
        action: The action that was performed
        success: Whether the action was successful
        prev_state: Previous agent state for comparing changes
        
    Returns:
        float: The reward value
    """
    reward = 0.0
    
    # Small penalty per action to encourage efficiency
    reward -= 0.1
    
    # Penalties for letting vitals drop low
    if agent.hunger > 80:
        reward -= 0.5
    if agent.thirst > 80:
        reward -= 0.5
    if agent.energy < 20:
        reward -= 0.3
    
    # If agent health is getting low, larger penalty
    if agent.health < 50:
        reward -= 0.5
    
    # Additional penalties for critical conditions
    if agent.hunger > 95 or agent.thirst > 95 or agent.energy < 5:
        reward -= 2.0
    
    # If the action was not successful, small penalty
    if not success:
        reward -= 0.2
        return reward
    
    # Rewards for successful farming actions
    if action == Agent.PLANT_SEED:
        reward += 1.0  # Planting is good
    
    elif action == Agent.TEND_PLANT:
        reward += 0.5  # Tending plants is good
    
    elif action == Agent.HARVEST:
        reward += 3.0  # Harvesting is very good - primary goal
    
    elif action == Agent.EAT:
        # Reward scales with hunger level
        hunger_factor = min(1.0, agent.hunger / 60.0)
        reward += 2.0 * hunger_factor
    
    elif action == Agent.DRINK:
        # Reward scales with thirst level
        thirst_factor = min(1.0, agent.thirst / 60.0)
        reward += 2.0 * thirst_factor
    
    # Small reward for staying alive
    reward += 0.05
    
    return reward


def train_reinforce(env, num_episodes=1000, gamma=0.99, lr=0.001, max_steps=1000):
    """
    Train the agent using the REINFORCE algorithm.
    
    Args:
        env: The environment to train in
        num_episodes: Number of episodes to train for
        gamma: Discount factor
        lr: Learning rate
        max_steps: Maximum steps per episode
        
    Returns:
        tuple: Trained model and training history
    """
    # Create the model, encoder, and agent
    encoder = ObservationEncoder(env)
    model = AgentCNN()
    agent = RLAgent(env, model, encoder)
    
    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    moving_avg_rewards = []
    
    # For tracking progress
    reward_window = deque(maxlen=100)
    
    # Training loop
    for episode in range(num_episodes):
        # Reset environment and agent
        env.reset()
        agent.reset_episode()
        
        episode_reward = 0
        step = 0
        
        # Run episode
        while step < max_steps and agent.is_alive:
            # Decide action
            action = agent.decide_action_training()
            
            # Get the previous state for reward calculation
            prev_state = agent.get_status()
            
            # Take the action
            success = agent.step(action)
            
            # Calculate reward
            reward = calculate_reward(agent, action, success, prev_state)
            agent.rewards.append(reward)
            episode_reward += reward
            
            # Check if agent is alive (health > 0)
            if agent.health <= 0:
                agent.is_alive = False
                # Extra penalty for dying
                agent.rewards[-1] -= 10.0
                episode_reward -= 10.0
            
            step += 1
        
        # Update episode tracking
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        reward_window.append(episode_reward)
        moving_avg = np.mean(list(reward_window))
        moving_avg_rewards.append(moving_avg)
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(agent.rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Convert to tensor and normalize
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = []
        for log_prob, G in zip(agent.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update the model
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, "
                  f"Avg Reward: {moving_avg:.2f}, "
                  f"Episode Length: {step}")
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.plot(moving_avg_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['Rewards', 'Moving Average'])
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Length')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    
    return model, {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'moving_avg': moving_avg_rewards
    }


def test_agent(env, model, num_episodes=10, max_steps=1000):
    """
    Test the trained agent.
    
    Args:
        env: The environment to test in
        model: The trained neural network model
        num_episodes: Number of episodes to test for
        max_steps: Maximum steps per episode
        
    Returns:
        dict: Test metrics
    """
    encoder = ObservationEncoder(env)
    agent = RLAgent(env, model, encoder)
    
    test_rewards = []
    test_lengths = []
    seeds_planted = []
    plants_harvested = []
    
    for episode in range(num_episodes):
        env.reset()
        agent.reset_episode()
        
        episode_reward = 0
        step = 0
        episode_seeds_planted = 0
        episode_plants_harvested = 0
        
        while step < max_steps and agent.is_alive:
            # Decide action
            action = agent.decide_action()
            
            # Take the action
            success = agent.step(action)
            
            # Track farming statistics
            if success:
                if action == Agent.PLANT_SEED:
                    episode_seeds_planted += 1
                elif action == Agent.HARVEST:
                    episode_plants_harvested += 1
            
            # Calculate reward for tracking (not used in decision making here)
            reward = calculate_reward(agent, action, success)
            episode_reward += reward
            
            # Check if agent is alive
            if agent.health <= 0:
                agent.is_alive = False
            
            step += 1
        
        # Update tracking
        test_rewards.append(episode_reward)
        test_lengths.append(step)
        seeds_planted.append(episode_seeds_planted)
        plants_harvested.append(episode_plants_harvested)
        
        print(f"Test Episode {episode + 1}, "
              f"Reward: {episode_reward:.2f}, "
              f"Length: {step}, "
              f"Seeds Planted: {episode_seeds_planted}, "
              f"Plants Harvested: {episode_plants_harvested}")
    
    # Return test metrics
    return {
        'rewards': test_rewards,
        'lengths': test_lengths,
        'seeds_planted': seeds_planted,
        'plants_harvested': plants_harvested,
        'avg_reward': np.mean(test_rewards),
        'avg_length': np.mean(test_lengths),
        'avg_seeds_planted': np.mean(seeds_planted),
        'avg_plants_harvested': np.mean(plants_harvested)
    }


if __name__ == "__main__":
    # Create the environment
    env = GridWorld(width=50, height=50)
    
    # Train the agent
    print("Starting training...")
    model, training_history = train_reinforce(env, num_episodes=500, max_steps=500)
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_agent.pth')
    
    # Test the trained agent
    print("\nTesting trained agent...")
    test_metrics = test_agent(env, model, num_episodes=5)
    
    print("\nTest Results:")
    print(f"Average Reward: {test_metrics['avg_reward']:.2f}")
    print(f"Average Episode Length: {test_metrics['avg_length']:.2f}")
    print(f"Average Seeds Planted: {test_metrics['avg_seeds_planted']:.2f}")
    print(f"Average Plants Harvested: {test_metrics['avg_plants_harvested']:.2f}") 