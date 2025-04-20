import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder, AgentCNN
import matplotlib.pyplot as plt
from collections import deque
import time
import os
import logging

# Set up logging with a root logger configuration so all modules inherit the settings
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to set log level for all loggers
def set_log_level(log_level):
    """Set the log level for all loggers in the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Set root logger level - this affects all loggers
    logging.getLogger().setLevel(numeric_level)
    
    # Also set our module logger level explicitly
    logger.setLevel(numeric_level)


class PPOAgentCNN(nn.Module):
    """
    CNN model for PPO algorithm with both policy and value function heads.
    Input: 7x7 grid with multiple channels
    Output: Action probabilities and value function
    """
    
    def __init__(self, num_channels=7, grid_size=7, num_actions=9):
        """
        Initialize the CNN model for PPO.
        
        Args:
            num_channels: Number of input channels
            grid_size: Size of the grid (e.g., 7 for a 7x7 grid)
            num_actions: Number of possible actions
        """
        super(PPOAgentCNN, self).__init__()
        
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
            tuple: (action_probs, state_values)
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


class RLAgent(Agent):
    """
    An agent that uses a neural network to make decisions.
    Enhanced for PPO algorithm.
    """
    
    def __init__(self, environment, model, encoder, start_row=0, start_col=0):
        """
        Initialize the RL agent for PPO.
        
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
        self.states = []
        self.actions = []
        self.action_probs = []
        self.action_logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []
        self.episode_length = 0
        self.is_alive = True
    
    def decide_action(self):
        """
        Decide the next action based on the neural network (for inference).
        
        Returns:
            int: The action to take
        """
        # Get the current observation
        observation = self.encoder.get_observation(self)
        
        # Add batch dimension
        observation = observation.unsqueeze(0)
        
        # Get action probabilities and state value from the model
        with torch.no_grad():
            # Check if the model is PPO (returns 3 values) or not (returns 1 value)
            model_output = self.model(observation)
            if isinstance(model_output, tuple) and len(model_output) == 3:
                action_probs, _, _ = model_output
            else:
                action_probs = model_output
        
        # Sample an action from the probability distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        
        return action.item()
    
    def decide_action_training(self):
        """
        Decide action during training and store data for PPO training.
        
        Returns:
            int: The action to take
        """
        # Get the current observation
        observation = self.encoder.get_observation(self)
        
        # Store state
        self.states.append(observation)
        
        # Add batch dimension
        observation = observation.unsqueeze(0)
        
        # Get action probabilities and state value from the model
        model_output = self.model(observation)
        
        # Check if the model is PPO (returns 3 values) or not (returns 1 value)
        if isinstance(model_output, tuple) and len(model_output) == 3:
            action_probs, state_value, action_logits = model_output
        else:
            action_probs = model_output
            # Create dummy values for state_value when using non-PPO model
            state_value = torch.zeros(1, 1)
        
        # Sample an action from the probability distribution
        action_distribution = torch.distributions.Categorical(action_probs)
        action = action_distribution.sample()
        
        # Store action, action probability, and state value for training
        self.actions.append(action.item())
        self.action_probs.append(action_probs[0, action.item()].item())
        self.action_logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(state_value)
        
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
        self.states = []
        self.actions = []
        self.action_probs = []
        self.action_logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []
        self.episode_length = 0
        self.is_alive = True
    
    def clear_memory(self):
        """Clear memory after update."""
        self.states = []
        self.actions = []
        self.action_probs = []
        self.action_logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []
    
    # For backward compatibility with train_reinforce
    @property
    def log_probs(self):
        return self.action_logprobs


class Memory:
    """PPO memory for storing and batching experiences."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
    
    def add(self, state, action, logprob, reward, is_terminal, value):
        """Add transition to memory."""
        self.states.append(state)
        self.actions.append(action)
        self.old_logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        self.values.append(value)
    
    def clear(self):
        """Clear memory."""
        self.states = []
        self.actions = []
        self.old_logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []


def calculate_reward(agent, action, success, prev_state=None, emergence=False):
    """
    Calculate the reward for an action.
    
    Args:
        agent: The agent that performed the action
        action: The action that was performed
        success: Whether the action was successful
        prev_state: Previous agent state for comparing changes
        emergence: Whether to use simplified rewards for emergent behavior
        
    Returns:
        float: The reward value
    """
    if emergence:
        return calculate_emergence_reward(agent, action, success, prev_state)
    else:
        return calculate_structured_reward(agent, action, success, prev_state)


def calculate_structured_reward(agent, action, success, prev_state=None):
    """
    Calculate a detailed, structured reward for specific actions.
    Used for the standard, hand-crafted approach.
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
        # Verify we actually planted a seed (which consumes a seed)
        initial_seeds = prev_state['seeds'] if prev_state else agent.seeds + 1
        if agent.seeds < initial_seeds:
            reward += 1.0  # Planting is good
        else:
            # Attempted to plant but failed (no seeds, wrong terrain, etc.)
            reward -= 0.5
    
    elif action == Agent.TEND_PLANT:
        # We already verified success is True, which means the plant was tended
        # in the tend_plant method, but let's add an extra check
        if success:
            reward += 0.5  # Tending plants is good
        else:
            # Attempted to tend but failed (no plant, mature plant, etc.)
            reward -= 0.3
    
    elif action == Agent.HARVEST:
        # Check if the action was actually legitimate (harvested a mature plant)
        # We can tell this is true if:
        # 1. success is True (already checked)
        # 2. We got more seeds as a result (this is the result of a proper harvest)
        initial_seeds = prev_state['seeds'] if prev_state else 0
        if agent.seeds > initial_seeds:
            reward += 3.0  # Harvesting is very good - but only if it was a real harvest
        else:
            # Attempting to harvest with nothing to harvest should actually be penalized
            reward -= 0.5
    
    elif action == Agent.EAT:
        # Reward scales with hunger level
        # But only if hunger actually decreased
        initial_hunger = prev_state['hunger'] if prev_state else 100.0
        if agent.hunger < initial_hunger:
            hunger_factor = min(1.0, initial_hunger / 60.0)  # Use initial hunger to scale reward
            reward += 2.0 * hunger_factor
        else:
            # Attempting to eat with nothing to eat should be penalized
            reward -= 0.5
    
    elif action == Agent.DRINK:
        # Reward scales with thirst level
        # But only if thirst actually decreased
        initial_thirst = prev_state['thirst'] if prev_state else 100.0
        if agent.thirst < initial_thirst:
            thirst_factor = min(1.0, initial_thirst / 60.0)  # Use initial thirst to scale reward
            reward += 2.0 * thirst_factor
        else:
            # Attempting to drink with no water nearby should be penalized
            reward -= 0.5
    
    # Small reward for staying alive
    reward += 0.05
    
    return reward


def calculate_emergence_reward(agent, action, success, prev_state=None):
    """
    Calculate a simple reward based primarily on survival and state improvement.
    Used for the emergent behavior approach.
    """
    reward = 0.0
    
    # Base survival reward: small bonus for staying alive
    reward += 0.05
    
    # Only use prev_state if it's available
    if prev_state:
        # Reward improvements in vital stats
        # Hunger reduction
        if agent.hunger < prev_state['hunger']:
            # Greater reward for larger hunger reduction
            hunger_change = prev_state['hunger'] - agent.hunger
            reward += 0.5 * min(hunger_change / 20.0, 1.0)  # Cap at 1.0 for 20+ point reduction
        
        # Thirst reduction
        if agent.thirst < prev_state['thirst']:
            # Greater reward for larger thirst reduction
            thirst_change = prev_state['thirst'] - agent.thirst
            reward += 0.5 * min(thirst_change / 20.0, 1.0)  # Cap at 1.0 for 20+ point reduction
            
        # Seed acquisition (relevant for farming)
        if agent.seeds > prev_state['seeds']:
            # Small reward for gaining seeds
            seed_change = agent.seeds - prev_state['seeds']
            reward += 0.2 * seed_change
    
    # Penalties for critical conditions (to incentivize avoiding dangerous states)
    if agent.hunger > 80:
        # Penalty scales with hunger severity
        hunger_penalty = 0.01 * (agent.hunger - 80) / 20.0
        reward -= hunger_penalty
        
    if agent.thirst > 80:
        # Penalty scales with thirst severity
        thirst_penalty = 0.01 * (agent.thirst - 80) / 20.0
        reward -= thirst_penalty
        
    if agent.health < 50:
        # Penalty scales with health loss severity
        health_penalty = 0.01 * (50 - agent.health) / 50.0
        reward -= health_penalty
    
    # The previous version had a terminal penalty of -10.0 for dying
    # We'll keep that in the training loop where terminal states are handled
    
    return reward


def train_ppo(env, num_episodes=2000, update_timestep=2000, epochs=10, epsilon=0.2, 
              gamma=0.99, gae_lambda=0.95, lr=0.0003, entropy_coef=0.05, value_coef=0.5,
              max_steps=1000, max_grad_norm=0.5, batch_size=64, emergence=False, log_level='info'):
    """
    Train the agent using the PPO algorithm.
    
    Args:
        env: The environment to train in
        num_episodes: Number of episodes to train for
        update_timestep: Number of timesteps between updates
        epochs: Number of epochs when optimizing the surrogate
        epsilon: Clipping parameter
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        lr: Learning rate
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        max_steps: Maximum steps per episode
        max_grad_norm: Maximum norm for gradient clipping
        batch_size: Mini batch size for updates
        emergence: Whether to use simplified rewards for emergent behavior
        log_level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        tuple: Trained model and training history
    """
    # Set logging level for all loggers
    set_log_level(log_level)
    
    # Create the model, encoder, and agent
    encoder = ObservationEncoder(env)
    model = PPOAgentCNN()
    agent = RLAgent(env, model, encoder)
    
    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Tracking metrics
    episode_rewards = []
    episode_lengths = []
    moving_avg_rewards = []
    
    # For tracking progress
    reward_window = deque(maxlen=100)
    memory = Memory()
    time_step = 0
    total_steps = 0
    
    # Training loop
    for episode in range(num_episodes):
        start_time = time.time()
        
        # Reset environment and agent
        env.reset()
        agent.reset_episode()
        
        episode_reward = 0
        step = 0
        
        # Run episode
        while step < max_steps and agent.is_alive:
            time_step += 1
            total_steps += 1
            
            # Decide action
            action = agent.decide_action_training()
            
            # Get the previous state for reward calculation
            prev_state = agent.get_status()
            
            # Take the action
            success = agent.step(action)
            
            # Calculate reward
            reward = calculate_reward(agent, action, success, prev_state, emergence)
            
            # Check if agent is alive (health > 0)
            is_terminal = False
            if agent.health <= 0:
                agent.is_alive = False
                # Extra penalty for dying
                reward -= 10.0
                is_terminal = True
            
            # Add this timestep to memory
            memory.add(
                agent.states[-1],
                agent.actions[-1],
                agent.action_logprobs[-1],
                reward,
                is_terminal,
                agent.state_values[-1]
            )
            
            agent.rewards.append(reward)
            agent.is_terminals.append(is_terminal)
            episode_reward += reward
            step += 1
            
            # If enough timesteps have passed, perform PPO update
            if time_step % update_timestep == 0:
                # Compute returns and advantages using GAE
                returns, advantages = compute_gae(memory.rewards, memory.values, 
                                                  memory.is_terminals, gamma, gae_lambda)
                
                # Update the policy and value function
                update_ppo(model, optimizer, memory, returns, advantages, epochs, epsilon, 
                           batch_size, entropy_coef, value_coef, max_grad_norm)
                
                # Clear memory
                memory.clear()
                
                logger.debug(f"PPO update at timestep {time_step}")
        
        # Update episode tracking
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)
        reward_window.append(episode_reward)
        moving_avg = np.mean(list(reward_window))
        moving_avg_rewards.append(moving_avg)
        
        # Print progress
        end_time = time.time()
        
        # Log detailed info for each episode at debug level
        logger.debug(f"Episode {episode + 1}/{num_episodes}, " 
                    f"Reward: {episode_reward:.2f}, "
                    f"Avg Reward: {moving_avg:.2f}, "
                    f"Episode Length: {step}, "
                    f"Total Steps: {total_steps}, "
                    f"Time: {end_time - start_time:.2f}s")
        
        # Log summary info every 10 episodes at info level
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {moving_avg:.2f}, "
                      f"Episode Length: {step}, "
                      f"Total Steps: {total_steps}, "
                      f"Time: {end_time - start_time:.2f}s")
    
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
    plt.savefig('models/ppo_training_progress.png')
    
    return model, {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'moving_avg': moving_avg_rewards
    }


def compute_gae(rewards, values, is_terminals, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE) and returns.
    
    Args:
        rewards: List of rewards
        values: List of state values
        is_terminals: List of terminal flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        tuple: (returns, advantages)
    """
    # Convert to tensors
    rewards = torch.tensor(rewards, dtype=torch.float32)
    values = torch.cat(values).squeeze()
    is_terminals = torch.tensor(is_terminals, dtype=torch.bool)
    
    # Initialize return and advantage lists
    returns = []
    advantages = []
    gae = 0
    
    # Loop through rewards in reverse order
    for i in reversed(range(len(rewards))):
        # If terminal state or last state, next value is 0
        # Otherwise, next value is value of next state
        if i == len(rewards) - 1 or is_terminals[i]:
            next_value = 0
        else:
            next_value = values[i + 1]
        
        # Calculate delta (TD error)
        delta = rewards[i] + gamma * next_value * (1 - is_terminals[i].float()) - values[i]
        
        # Calculate GAE
        gae = delta + gamma * gae_lambda * (1 - is_terminals[i].float()) * gae
        
        # Add to lists (prepend)
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])
    
    # Convert to tensors
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return returns, advantages


def update_ppo(model, optimizer, memory, returns, advantages, epochs, epsilon, 
               batch_size, entropy_coef, value_coef, max_grad_norm):
    """
    Update the policy and value function using PPO.
    
    Args:
        model: The neural network model
        optimizer: The optimizer
        memory: The replay memory
        returns: The computed returns
        advantages: The computed advantages
        epochs: Number of epochs to optimize
        epsilon: Clipping parameter
        batch_size: Mini batch size
        entropy_coef: Entropy coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum norm for gradient clipping
    """
    # Convert memory to tensors
    old_states = torch.stack(memory.states)
    old_actions = torch.tensor(memory.actions, dtype=torch.long)
    old_logprobs = torch.cat(memory.old_logprobs).detach()  # Detach to prevent backprop through old logprobs
    
    # Convert returns and advantages to tensors and detach
    returns = returns.detach()
    advantages = advantages.detach()
    
    # Number of samples
    n_samples = len(old_states)
    
    # Optimize policy for K epochs
    for epoch in range(epochs):
        # Generate random indices
        indices = torch.randperm(n_samples)
        
        # Create mini-batches
        for start_idx in range(0, n_samples, batch_size):
            # Get mini-batch indices
            idxs = indices[start_idx:min(start_idx + batch_size, n_samples)]
            
            # Get mini-batch of old data
            batch_states = old_states[idxs]
            batch_actions = old_actions[idxs]
            batch_logprobs = old_logprobs[idxs]
            batch_returns = returns[idxs]
            batch_advantages = advantages[idxs]
            
            # Forward pass - create a new computation graph
            action_probs, state_values, _ = model(batch_states)
            
            # Get distribution
            dist = torch.distributions.Categorical(action_probs)
            
            # Calculate entropy
            entropy = dist.entropy().mean()
            
            # Calculate log probabilities of actions
            new_logprobs = dist.log_prob(batch_actions)
            
            # Calculate probability ratio
            ratio = torch.exp(new_logprobs - batch_logprobs)
            
            # Calculate surrogate losses
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * batch_advantages
            
            # Calculate policy loss
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = 0.5 * (batch_returns - state_values.squeeze()).pow(2).mean()
            
            # Calculate total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update parameters
            optimizer.step()


def run_agent_test(env, model, num_episodes=10, max_steps=1000, args=None, log_level='info'):
    """
    Test the trained agent.
    
    Args:
        env: The environment to test in
        model: The trained neural network model
        num_episodes: Number of episodes to test for
        max_steps: Maximum steps per episode
        args: Command line arguments for configuration
        log_level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        dict: Test metrics
    """
    # Set logging level for all loggers
    set_log_level(log_level)
    
    encoder = ObservationEncoder(env)
    agent = RLAgent(env, model, encoder)
    
    # Default emergence flag to False if args is None
    emergence = args.emergence if args else False
    
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
            
            # Log detailed step information at debug level
            logger.debug(f"Test Episode {episode + 1}, Step {step + 1}, "
                        f"Action: {action}, Success: {success}, "
                        f"Health: {agent.health:.1f}, Hunger: {agent.hunger:.1f}, "
                        f"Thirst: {agent.thirst:.1f}, Energy: {agent.energy:.1f}")
            
            # Calculate reward for tracking (not used in decision making here)
            reward = calculate_reward(agent, action, success, None, emergence)
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
        
        logger.info(f"Test Episode {episode + 1}, "
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


def train_reinforce(env, num_episodes=1000, gamma=0.99, lr=0.001, max_steps=1000, 
                   emergence=False, log_level='info'):
    """
    Train the agent using the REINFORCE algorithm.
    
    Args:
        env: The environment to train in
        num_episodes: Number of episodes to train for
        gamma: Discount factor
        lr: Learning rate
        max_steps: Maximum steps per episode
        emergence: Whether to use simplified rewards for emergent behavior
        log_level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        tuple: Trained model and training history
    """
    # Set logging level for all loggers
    set_log_level(log_level)
    
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
        start_time = time.time()
        
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
            reward = calculate_reward(agent, action, success, prev_state, emergence)
            agent.rewards.append(reward)
            episode_reward += reward
            
            # Log detailed step information at debug level
            logger.debug(f"Episode {episode + 1}, Step {step + 1}, "
                        f"Action: {action}, Success: {success}, Reward: {reward:.2f}")
            
            # Check if agent is alive (health > 0)
            if agent.health <= 0:
                agent.is_alive = False
                # Extra penalty for dying
                agent.rewards[-1] -= 10.0
                episode_reward -= 10.0
                logger.debug(f"Agent died at step {step + 1}")
            
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
        
        end_time = time.time()
        
        # Log detailed info at debug level
        logger.debug(f"Episode {episode + 1}/{num_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Avg Reward: {moving_avg:.2f}, "
                    f"Episode Length: {step}, "
                    f"Time: {end_time - start_time:.2f}s")
        
        # Print progress at info level
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{num_episodes}, "
                      f"Avg Reward: {moving_avg:.2f}, "
                      f"Episode Length: {step}, "
                      f"Time: {end_time - start_time:.2f}s")
    
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
    plt.savefig('models/training_progress.png')
    
    return model, {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'moving_avg': moving_avg_rewards
    }

