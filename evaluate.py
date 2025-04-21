import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
import logging
import argparse
from collections import defaultdict
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder
from rule_based_agent import RuleBasedAgent
from model_based_agent import ModelBasedAgent
from rl_models import REINFORCEModel, PPOModel
from train import RLAgent, set_log_level

# Set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_agent(env, agent, num_episodes=50, max_steps=1000, log_level='info'):
    """
    Evaluate an agent on the given environment.
    
    Args:
        env: The environment to evaluate in
        agent: The agent to evaluate
        num_episodes: Number of episodes to evaluate for
        max_steps: Maximum steps per episode
        log_level: Logging level ('debug', 'info', 'warning', 'error', 'critical')
        
    Returns:
        dict: Evaluation metrics
    """
    # Set logging level for all loggers
    set_log_level(log_level)
    
    logger.info("Starting agent evaluation...")
    
    # Tracking metrics
    metrics = {
        'rewards': [],
        'lengths': [],
        'seeds_planted': [],
        'plants_tended': [],
        'plants_harvested': [],
        'food_eaten': [],
        'water_drunk': [],
        'survival_times': [],
        'caused_of_death': defaultdict(int)
    }
    
    for episode in range(num_episodes):
        logger.info(f"Starting evaluation episode {episode+1}/{num_episodes}")
        # Reset environment and agent
        env.reset()
        logger.info("Environment reset")
        
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
            logger.info("Agent reset via reset_episode")
        else:
            # For rule-based agents that don't have reset_episode
            agent.row = np.random.randint(0, env.height)
            agent.col = np.random.randint(0, env.width)
            while env.grid[agent.row, agent.col] == GridWorld.WATER:
                agent.row = np.random.randint(0, env.height)
                agent.col = np.random.randint(0, env.width)
            agent.energy = 100.0
            agent.health = 100.0
            agent.hunger = 0.0
            agent.thirst = 0.0
            agent.seeds = 10
            if hasattr(agent, 'is_alive'):
                agent.is_alive = True
            logger.info("Agent reset manually")
        
        episode_reward = 0.0
        episode_stats = {
            'seeds_planted': 0,
            'plants_tended': 0,
            'plants_harvested': 0,
            'food_eaten': 0,
            'water_drunk': 0,
        }
        
        step = 0
        is_alive = True
        cause_of_death = "Survived"
        
        try:
            logger.info(f"Starting episode steps loop (max_steps={max_steps})")
            while step < max_steps and is_alive:
                # Get action from agent
                if hasattr(agent, 'decide_action'):
                    # RL agent
                    action = agent.decide_action()
                    success = agent.step(action)
                    logger.debug(f"Step {step+1}: RL agent action {action}, success={success}")
                elif hasattr(agent, 'step_ai'):
                    # Rule-based agent
                    logger.debug(f"Step {step+1}: About to call step_ai")
                    result = agent.step_ai()
                    success = result.get('success', False)
                    action = result.get('action', -1)
                    is_alive = result.get('alive', True)
                    if not is_alive:
                        cause_of_death = result.get('cause_of_death', "Unknown")
                    logger.debug(f"Step {step+1}: Rule-based agent action {action}, success={success}, alive={is_alive}")
                else:
                    # Basic agent
                    actions = [
                        Agent.MOVE_NORTH, Agent.MOVE_SOUTH, 
                        Agent.MOVE_EAST, Agent.MOVE_WEST,
                        Agent.EAT, Agent.DRINK, 
                        Agent.PLANT_SEED, Agent.TEND_PLANT, Agent.HARVEST
                    ]
                    action = np.random.choice(actions)
                    success = agent.step(action)
                    logger.debug(f"Step {step+1}: Basic agent action {action}, success={success}")
                
                # Track specific actions if successful
                if success:
                    if action == Agent.PLANT_SEED:
                        episode_stats['seeds_planted'] += 1
                    elif action == Agent.TEND_PLANT:
                        episode_stats['plants_tended'] += 1
                    elif action == Agent.HARVEST:
                        episode_stats['plants_harvested'] += 1
                    elif action == Agent.EAT:
                        episode_stats['food_eaten'] += 1
                    elif action == Agent.DRINK:
                        episode_stats['water_drunk'] += 1
                
                # Check if agent died (for agents with is_alive property)
                if hasattr(agent, 'is_alive'):
                    is_alive = agent.is_alive
                    if not is_alive:
                        if agent.hunger >= 100:
                            cause_of_death = "Starvation"
                        elif agent.thirst >= 100:
                            cause_of_death = "Dehydration"
                        elif agent.energy <= 0:
                            cause_of_death = "Exhaustion"
                        else:
                            cause_of_death = "Unknown"
                
                # Calculate theoretical reward (for tracking, not used in decision making)
                if hasattr(agent, 'rewards') and len(agent.rewards) > 0:
                    episode_reward += agent.rewards[-1]
                else:
                    # Simple reward approximation for non-RL agents
                    reward = 0.1  # Base survival reward
                    if success:
                        if action == Agent.HARVEST:
                            reward += 1.0
                        elif action == Agent.PLANT_SEED:
                            reward += 0.5
                        elif action == Agent.TEND_PLANT:
                            reward += 0.3
                        elif action == Agent.EAT:
                            reward += 0.5
                        elif action == Agent.DRINK:
                            reward += 0.5
                    episode_reward += reward
                
                step += 1
                
                if step % 20 == 0:
                    logger.info(f"  Completed {step} steps in episode {episode+1}")
        except Exception as e:
            logger.error(f"Error during episode execution: {e}")
            import traceback
            logger.error(traceback.format_exc())
            break
        
        # Record metrics for this episode
        metrics['rewards'].append(episode_reward)
        metrics['lengths'].append(step)
        metrics['seeds_planted'].append(episode_stats['seeds_planted'])
        metrics['plants_tended'].append(episode_stats['plants_tended'])
        metrics['plants_harvested'].append(episode_stats['plants_harvested'])
        metrics['food_eaten'].append(episode_stats['food_eaten'])
        metrics['water_drunk'].append(episode_stats['water_drunk'])
        metrics['survival_times'].append(step)
        metrics['caused_of_death'][cause_of_death] += 1
        
        # Log progress
        logger.info(f"Evaluated episode {episode + 1}/{num_episodes}, " 
                  f"Steps: {step}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Harvested: {episode_stats['plants_harvested']}, "
                  f"Cause of death: {cause_of_death}")
    
    # Calculate aggregate metrics
    try:
        logger.info("Calculating aggregate metrics...")
        for key in ['rewards', 'lengths', 'seeds_planted', 'plants_tended', 
                    'plants_harvested', 'food_eaten', 'water_drunk', 'survival_times']:
            metrics[f'avg_{key}'] = np.mean(metrics[key])
            metrics[f'std_{key}'] = np.std(metrics[key])
        
        # Calculate survival rate
        survival_count = metrics['caused_of_death'].get('Survived', 0)
        metrics['survival_rate'] = survival_count / num_episodes
        logger.info("Finished evaluation.")
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return metrics

def compare_agents(env_config, agents_dict, num_episodes=50, max_steps=1000, log_level='info'):
    """
    Compare multiple agents on the same environment configuration.
    
    Args:
        env_config: Dictionary of environment parameters
        agents_dict: Dictionary mapping agent names to agent instances
        num_episodes: Number of episodes to evaluate each agent
        max_steps: Maximum steps per episode
        log_level: Logging level
        
    Returns:
        dict: Comparison metrics for all agents
    """
    set_log_level(log_level)
    comparison = {}
    
    for agent_name, agent in agents_dict.items():
        logger.info(f"Evaluating {agent_name}...")
        env = GridWorld(**env_config)
        metrics = evaluate_agent(env, agent, num_episodes, max_steps, log_level)
        comparison[agent_name] = metrics
        
        logger.info(f"Results for {agent_name}:")
        logger.info(f"  Average Survival Time: {metrics['avg_survival_times']:.1f} steps")
        logger.info(f"  Survival Rate: {metrics['survival_rate']*100:.1f}%")
        logger.info(f"  Average Plants Harvested: {metrics['avg_plants_harvested']:.2f}")
        logger.info(f"  Average Seeds Planted: {metrics['avg_seeds_planted']:.2f}")
    
    return comparison

def plot_comparison(comparison, metrics_to_plot=None, title='Agent Comparison'):
    """
    Plot comparison of different agents.
    
    Args:
        comparison: Dictionary from compare_agents
        metrics_to_plot: List of metric names to plot (default: survival_times, plants_harvested)
        title: Plot title
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['survival_times', 'plants_harvested', 'seeds_planted', 'food_eaten']
    
    agent_names = list(comparison.keys())
    num_metrics = len(metrics_to_plot)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 6))
    if num_metrics == 1:
        axes = [axes]  # Make sure axes is a list
    
    for i, metric in enumerate(metrics_to_plot):
        metric_values = [comparison[agent][f'avg_{metric}'] for agent in agent_names]
        metric_stds = [comparison[agent][f'std_{metric}'] for agent in agent_names]
        
        # Create bar chart with error bars
        axes[i].bar(agent_names, metric_values, yerr=metric_stds, capsize=10)
        axes[i].set_title(f'Average {metric.replace("_", " ").title()}')
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for j, v in enumerate(metric_values):
            axes[i].text(j, v + metric_stds[j] + 0.1, f'{v:.1f}', 
                      ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/agent_comparison.png', dpi=300)
    plt.show()

def plot_training_progress(history, title='Training Progress'):
    """
    Plot the training progress.
    
    Args:
        history: Dictionary containing training history
        title: Plot title
    """
    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(history['rewards'], 'b-', alpha=0.3, label='Episode Reward')
    if 'moving_avg' in history:
        ax1.plot(history['moving_avg'], 'r-', linewidth=2, label='Moving Average')
    ax1.set_title('Rewards over Episodes')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot episode lengths
    ax2.plot(history['lengths'], 'g-', alpha=0.3, label='Episode Length')
    
    # Calculate moving average for lengths if not provided
    if 'moving_avg_lengths' not in history:
        window_size = min(100, len(history['lengths']))
        moving_avg_lengths = []
        for i in range(len(history['lengths'])):
            start_idx = max(0, i - window_size + 1)
            moving_avg_lengths.append(np.mean(history['lengths'][start_idx:i+1]))
        
        ax2.plot(moving_avg_lengths, 'm-', linewidth=2, label='Moving Average')
    else:
        ax2.plot(history['moving_avg_lengths'], 'm-', linewidth=2, label='Moving Average')
    
    ax2.set_title('Episode Lengths over Episodes')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_progress.png', dpi=300)
    plt.show()

def create_agent(agent_type, env, model_path=None):
    """
    Create an agent of the specified type.
    
    Args:
        agent_type: Type of agent ('rule_based', 'ppo', 'reinforce')
        env: Environment to place the agent in
        model_path: Path to the model file for RL agents
        
    Returns:
        Agent instance
    """
    if agent_type == 'rule_based':
        # Create rule-based agent
        return RuleBasedAgent(env)
    
    elif agent_type == 'ppo':
        # Create PPO agent with ModelBasedAgent which loads the model internally
        return ModelBasedAgent(env, model_path=model_path)
    
    elif agent_type == 'reinforce':
        # Create REINFORCE agent with ModelBasedAgent
        return ModelBasedAgent(env, model_path=model_path)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate and compare agents')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'compare', 'visualize'],
                      default='evaluate', help='Evaluation mode')
    parser.add_argument('--agent-type', type=str, choices=['rule_based', 'ppo', 'reinforce'],
                      default='ppo', help='Type of agent to evaluate')
    parser.add_argument('--model-path', type=str, default='models/ppo_trained_agent.pth',
                      help='Path to the trained model file')
    parser.add_argument('--num-episodes', type=int, default=50,
                      help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=1000,
                      help='Maximum steps per episode')
    parser.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'],
                      default='info', help='Logging level')
    args = parser.parse_args()
    
    # Set the logging level
    set_log_level(args.log_level)
    
    logger.info(f"Starting with arguments: {args}")
    
    try:
        if args.mode == 'evaluate':
            # Create environment
            logger.info("Creating environment...")
            env = GridWorld(width=100, height=100, water_probability=0.1)
            logger.info("Environment created successfully.")
            
            # Create agent
            logger.info(f"Creating agent of type {args.agent_type} with model {args.model_path}...")
            agent = create_agent(args.agent_type, env, args.model_path)
            logger.info("Agent created successfully.")
            
            # Evaluate agent
            logger.info("Starting evaluation...")
            metrics = evaluate_agent(env, agent, args.num_episodes, args.max_steps, args.log_level)
            
            # Print results
            logger.info("\nEvaluation Results:")
            logger.info(f"Average Survival Time: {metrics['avg_survival_times']:.1f} steps")
            logger.info(f"Survival Rate: {metrics['survival_rate']*100:.1f}%")
            logger.info(f"Average Reward: {metrics['avg_rewards']:.2f}")
            logger.info(f"Average Plants Harvested: {metrics['avg_plants_harvested']:.2f}")
            logger.info(f"Average Seeds Planted: {metrics['avg_seeds_planted']:.2f}")
            logger.info(f"Average Plants Tended: {metrics['avg_plants_tended']:.2f}")
            logger.info(f"Average Food Eaten: {metrics['avg_food_eaten']:.2f}")
            logger.info(f"Average Water Drunk: {metrics['avg_water_drunk']:.2f}")
            logger.info("\nCauses of Death:")
            for cause, count in metrics['caused_of_death'].items():
                logger.info(f"  {cause}: {count} ({count/args.num_episodes*100:.1f}%)")
        
        elif args.mode == 'compare':
            # Environment configuration
            env_config = {
                'width': 100,
                'height': 100,
                'water_probability': 0.1
            }
            
            # Create environments and agents
            env1 = GridWorld(**env_config)
            env2 = GridWorld(**env_config)
            env3 = GridWorld(**env_config)
            
            rule_based_agent = RuleBasedAgent(env1)
            
            # Create PPO agent
            ppo_model_path = 'models/ppo_trained_agent.pth'
            ppo_agent = ModelBasedAgent(env2, model_path=ppo_model_path)
            
            # Create REINFORCE agent 
            reinforce_model_path = 'models/reinforce_trained_agent.pth'
            reinforce_agent = ModelBasedAgent(env3, model_path=reinforce_model_path)
            
            # Prepare agents dictionary
            agents_dict = {
                'Rule-Based': rule_based_agent,
                'PPO': ppo_agent,
                'REINFORCE': reinforce_agent
            }
            
            # Compare agents
            comparison = compare_agents(env_config, agents_dict, args.num_episodes, args.max_steps, args.log_level)
            
            # Plot comparison
            plot_comparison(comparison)
        
        elif args.mode == 'visualize':
            # For visualizing the agent behavior, we'll use the existing main.py script
            import subprocess
            import sys
            
            cmd = [
                sys.executable, "main.py",
                "--mode", "ai",
                "--agent-type", 'model_based' if args.agent_type in ['ppo', 'reinforce'] else 'rule_based',
                "--model-path", args.model_path,
                "--log-level", args.log_level
            ]
            
            logger.info(f"Launching visualization of {args.agent_type} agent...")
            subprocess.run(cmd)
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 