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
from model import ObservationEncoder, AgentCNN
from rule_based_agent import RuleBasedAgent
from train import (
    train_reinforce, 
    train_ppo, 
    RLAgent, 
    PPOAgentCNN, 
    set_log_level, 
    run_agent_test
)
from evaluate import evaluate_agent, compare_agents, plot_comparison, create_agent, plot_training_progress

# Set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class RLFramework:
    """
    A unified framework for training, evaluating, and comparing RL agents.
    """
    
    def __init__(self, args):
        """Initialize the framework with command line arguments."""
        self.args = args
        set_log_level(args.log_level)
        
        # Ensure models directory exists
        os.makedirs('models', exist_ok=True)
        
        # Create environment based on args
        self.env_config = {
            'width': args.grid_size,
            'height': args.grid_size,
            'water_probability': args.water_prob
        }
        
        # Initialize training histories
        self.training_histories = {}
    
    def train_agent(self, algorithm, model_path=None):
        """
        Train an agent using the specified algorithm.
        
        Args:
            algorithm: 'ppo' or 'reinforce'
            model_path: Path to save the trained model
            
        Returns:
            tuple: (trained model, training history)
        """
        # Create environment for training
        env = GridWorld(**self.env_config)
        
        # Number of episodes based on algorithm
        if algorithm == 'ppo':
            num_episodes = self.args.ppo_episodes
            if num_episodes <= 0:
                logger.info(f"Skipping {algorithm} training (episodes set to {num_episodes})")
                return None, None
            logger.info(f"Training PPO agent for {num_episodes} episodes...")
            
            # Train using PPO
            model, history = train_ppo(
                env=env,
                num_episodes=num_episodes,
                update_timestep=self.args.update_timestep,
                epochs=self.args.ppo_epochs,
                epsilon=self.args.ppo_epsilon,
                gamma=self.args.gamma,
                gae_lambda=self.args.gae_lambda,
                lr=self.args.learning_rate,
                entropy_coef=self.args.entropy_coef,
                value_coef=self.args.value_coef,
                max_steps=self.args.max_steps,
                max_grad_norm=self.args.max_grad_norm,
                batch_size=self.args.batch_size,
                emergence=self.args.emergence,
                log_level=self.args.log_level
            )
            
        elif algorithm == 'reinforce':
            num_episodes = self.args.reinforce_episodes
            if num_episodes <= 0:
                logger.info(f"Skipping {algorithm} training (episodes set to {num_episodes})")
                return None, None
            logger.info(f"Training REINFORCE agent for {num_episodes} episodes...")
            
            # Train using REINFORCE
            model, history = train_reinforce(
                env=env,
                num_episodes=num_episodes,
                gamma=self.args.gamma,
                lr=self.args.learning_rate,
                max_steps=self.args.max_steps,
                emergence=self.args.emergence,
                log_level=self.args.log_level
            )
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Save the model if training was performed
        if model is not None and model_path is not None:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Plot training progress
            plot_training_progress(history, title=f'{algorithm.upper()} Training Progress')
        
        return model, history
    
    def evaluate_single_agent(self, agent_type, model_path=None):
        """
        Evaluate a single agent.
        
        Args:
            agent_type: 'rule_based', 'reinforce', or 'ppo'
            model_path: Path to the model file (for RL agents)
            
        Returns:
            dict: Evaluation metrics
        """
        # Create environment
        env = GridWorld(**self.env_config)
        
        # Create agent
        agent = create_agent(agent_type, env, model_path)
        
        # Evaluate agent
        logger.info(f"Evaluating {agent_type} agent...")
        metrics = evaluate_agent(
            env=env,
            agent=agent,
            num_episodes=self.args.eval_episodes,
            max_steps=self.args.max_steps,
            log_level=self.args.log_level
        )
        
        # Print summary
        logger.info(f"\n{agent_type.upper()} Agent Evaluation:")
        logger.info(f"Average Reward: {metrics['avg_rewards']:.2f}")
        logger.info(f"Average Episode Length: {metrics['avg_lengths']:.1f}")
        logger.info(f"Average Seeds Planted: {metrics['avg_seeds_planted']:.2f}")
        logger.info(f"Average Plants Harvested: {metrics['avg_plants_harvested']:.2f}")
        
        return metrics
    
    def compare_agents(self, agent_types, model_paths=None):
        """
        Compare multiple agents.
        
        Args:
            agent_types: List of agent types to compare
            model_paths: Dictionary mapping agent types to model paths
            
        Returns:
            dict: Comparison results
        """
        if model_paths is None:
            model_paths = {}
            
        # Create agents dictionary
        agents_dict = {}
        for agent_type in agent_types:
            # Create environment for each agent
            env = GridWorld(**self.env_config)
            
            # Create agent
            model_path = model_paths.get(agent_type)
            agent = create_agent(agent_type, env, model_path)
            
            # Add to dictionary with formatted name
            if agent_type == 'rule_based':
                name = 'Rule-Based'
            else:
                name = agent_type.upper()
            
            agents_dict[name] = agent
        
        # Compare agents
        logger.info(f"Comparing agents: {', '.join(agent_types)}...")
        comparison_results = compare_agents(
            env_config=self.env_config,
            agents_dict=agents_dict,
            num_episodes=self.args.eval_episodes,
            max_steps=self.args.max_steps,
            log_level=self.args.log_level
        )
        
        # Plot comparison
        metrics_to_plot = [
            'rewards', 'lengths', 'plants_harvested',
            'seeds_planted', 'survival_times', 'food_eaten', 'water_drunk'
        ]
        plot_comparison(comparison_results, metrics_to_plot, title='Agent Comparison')
        
        return comparison_results
    
    def visualize_agent(self, agent_type, model_path=None):
        """
        Visualize an agent's behavior.
        
        Args:
            agent_type: 'rule_based', 'reinforce', or 'ppo'
            model_path: Path to the model file (for RL agents)
        """
        # Create environment
        env = GridWorld(**self.env_config)
        
        if agent_type in ['reinforce', 'ppo']:
            # For RL agents, we use run_agent_test
            if model_path and os.path.exists(model_path):
                if agent_type == 'reinforce':
                    model = AgentCNN(num_channels=7, grid_size=7, num_actions=9)
                else:  # ppo
                    model = PPOAgentCNN(num_channels=7, grid_size=7, num_actions=9)
                
                model.load_state_dict(torch.load(model_path))
                logger.info(f"Visualizing {agent_type.upper()} agent...")
                run_agent_test(
                    env=env,
                    model=model,
                    num_episodes=1,
                    max_steps=self.args.max_steps,
                    log_level=self.args.log_level
                )
            else:
                logger.warning(f"Cannot visualize {agent_type} agent: no model found at {model_path}")
        else:
            # For rule-based agent, we need to use the compare_agents.py script
            logger.info("For Rule-Based agent visualization, use the compare_agents.py script")
    
    def print_comparison_summary(self, comparison_results):
        """
        Print a detailed summary of the agent comparison.
        
        Args:
            comparison_results: Dictionary of agent metrics from compare_agents
        """
        logger.info("\n===== AGENT COMPARISON SUMMARY =====")
        
        # Print average metrics for each agent
        for agent_name, metrics in comparison_results.items():
            logger.info(f"\n{agent_name} Agent:")
            logger.info(f"  Average reward: {metrics['avg_rewards']:.2f}")
            logger.info(f"  Average episode length: {metrics['avg_lengths']:.2f}")
            logger.info(f"  Average plants harvested: {metrics['avg_plants_harvested']:.2f}")
            logger.info(f"  Average seeds planted: {metrics['avg_seeds_planted']:.2f}")
            logger.info(f"  Average survival time: {metrics['avg_survival_times']:.2f}")
            logger.info(f"  Average food eaten: {metrics['avg_food_eaten']:.2f}")
            logger.info(f"  Average water drunk: {metrics['avg_water_drunk']:.2f}")
            
            # Print causes of death
            logger.info("  Causes of death:")
            total_episodes = len(metrics.get('survival_times', []))
            if total_episodes == 0:
                total_episodes = self.args.eval_episodes  # Fallback to command line argument
                
            for cause, count in metrics['caused_of_death'].items():
                logger.info(f"    {cause}: {count}/{total_episodes} ({count/total_episodes*100:.1f}%)")
        
        # Determine the best agent based on different metrics
        metric_leaders = {}
        metric_labels = {
            'avg_rewards': 'Highest reward',
            'avg_lengths': 'Longest episodes',
            'avg_plants_harvested': 'Most plants harvested', 
            'avg_seeds_planted': 'Most seeds planted',
            'avg_survival_times': 'Longest survival time',
            'avg_food_eaten': 'Most food eaten',
            'avg_water_drunk': 'Most water drunk'
        }
        
        for metric, label in metric_labels.items():
            best_agent = max(comparison_results.items(), key=lambda x: x[1][metric])[0]
            metric_leaders[label] = best_agent
        
        logger.info("\nBest performing agents by metric:")
        for label, agent in metric_leaders.items():
            logger.info(f"  {label}: {agent}")
    
    def run(self):
        """Run the framework based on command line arguments."""
        # Dictionary of model paths
        model_paths = {
            'reinforce': self.args.reinforce_model,
            'ppo': self.args.ppo_model
        }
        
        # Train agents if requested
        if self.args.train:
            if self.args.reinforce_episodes > 0:
                reinforce_model, reinforce_history = self.train_agent('reinforce', model_paths['reinforce'])
                self.training_histories['reinforce'] = reinforce_history
            
            if self.args.ppo_episodes > 0:
                ppo_model, ppo_history = self.train_agent('ppo', model_paths['ppo'])
                self.training_histories['ppo'] = ppo_history
        
        # Determine which agents to compare
        agents_to_compare = []
        
        # Always include rule-based
        agents_to_compare.append('rule_based')
        
        # Add REINFORCE if either it was trained or a model exists
        if (self.args.train and self.args.reinforce_episodes > 0) or (
            not self.args.train and os.path.exists(model_paths['reinforce'])):
            agents_to_compare.append('reinforce')
        
        # Add PPO if either it was trained or a model exists
        if (self.args.train and self.args.ppo_episodes > 0) or (
            not self.args.train and os.path.exists(model_paths['ppo'])):
            agents_to_compare.append('ppo')
        
        # Compare agents
        comparison_results = self.compare_agents(agents_to_compare, model_paths)
        
        # Print comparison summary
        self.print_comparison_summary(comparison_results)
        
        # Visualize agents if requested
        if self.args.visualize:
            for agent_type in agents_to_compare:
                if agent_type != 'rule_based':  # Rule-based requires separate script
                    self.visualize_agent(agent_type, model_paths.get(agent_type))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train, evaluate, and compare RL agents')
    
    # Mode selection
    parser.add_argument('--train', action='store_true', help='Train new agents')
    parser.add_argument('--evaluate', action='store_true', help='Only evaluate without comparison (faster)')
    parser.add_argument('--visualize', action='store_true', help='Visualize agent behavior')
    
    # Environment parameters
    parser.add_argument('--grid-size', type=int, default=50, help='Size of the grid world')
    parser.add_argument('--water-prob', type=float, default=0.1, help='Probability of water cells')
    
    # General training parameters
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum steps per episode')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--emergence', action='store_true', help='Use simplified rewards for emergent behavior')
    
    # Algorithm-specific training parameters
    parser.add_argument('--reinforce-episodes', type=int, default=1000, help='Number of episodes for REINFORCE training')
    parser.add_argument('--ppo-episodes', type=int, default=2000, help='Number of episodes for PPO training')
    
    # PPO-specific parameters
    parser.add_argument('--update-timestep', type=int, default=2000, help='Number of timesteps between PPO updates')
    parser.add_argument('--ppo-epochs', type=int, default=10, help='Number of PPO epochs')
    parser.add_argument('--ppo-epsilon', type=float, default=0.2, help='PPO clipping parameter')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.05, help='Entropy coefficient for PPO')
    parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient for PPO')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Maximum norm for gradient clipping')
    parser.add_argument('--batch-size', type=int, default=64, help='Mini batch size for PPO updates')
    
    # Evaluation parameters
    parser.add_argument('--eval-episodes', type=int, default=50, help='Number of episodes for evaluation')
    
    # Model paths
    parser.add_argument('--reinforce-model', type=str, default='models/reinforce_trained_agent.pth', help='Path to REINFORCE model')
    parser.add_argument('--ppo-model', type=str, default='models/ppo_trained_agent.pth', help='Path to PPO model')
    
    # Logging
    parser.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Logging level')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create and run the framework
    framework = RLFramework(args)
    framework.run()


if __name__ == "__main__":
    main() 