import argparse
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
from environment import GridWorld
from train import train_ppo, train_reinforce, run_agent_test, set_log_level
from evaluate import evaluate_agent, compare_agents, plot_comparison, create_agent, plot_training_progress

# Set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_models_directory():
    """Ensure that the models directory exists."""
    os.makedirs('models', exist_ok=True)

def train_and_evaluate_agent(args):
    """
    Train and evaluate the agent with the specified algorithm and parameters.
    
    Args:
        args: Command line arguments
    """
    # Set logging level for all loggers
    set_log_level(args.log_level)
    
    # Ensure models directory exists
    ensure_models_directory()
    
    # Create environment
    env = GridWorld(width=100, height=100, water_probability=0.1)
    
    # Train the agent
    logger.info(f"Training agent using {args.algorithm} algorithm...")
    
    if args.algorithm == 'ppo':
        # Train using PPO
        model, history = train_ppo(
            env=env,
            num_episodes=args.num_episodes,
            update_timestep=args.update_timestep,
            epochs=args.ppo_epochs,
            epsilon=args.ppo_epsilon,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            lr=args.learning_rate,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_steps=args.max_steps,
            max_grad_norm=args.max_grad_norm,
            batch_size=args.batch_size,
            emergence=args.emergence,
            log_level=args.log_level
        )
        
        # Save the model
        model_path = os.path.join('models', 'ppo_trained_agent.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
    elif args.algorithm == 'reinforce':
        # Train using REINFORCE
        model, history = train_reinforce(
            env=env,
            num_episodes=args.num_episodes,
            gamma=args.gamma,
            lr=args.learning_rate,
            max_steps=args.max_steps,
            emergence=args.emergence,
            log_level=args.log_level
        )
        
        # Save the model
        model_path = os.path.join('models', 'reinforce_trained_agent.pth')
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")
        
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    # Plot and save training progress
    plot_training_progress(history, title=f'{args.algorithm.upper()} Training Progress')
    
    # Test the trained agent
    logger.info("Testing the trained agent...")
    test_metrics = run_agent_test(
        env=env,
        model=model,
        num_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        args=args,
        log_level=args.log_level
    )
    
    logger.info("\nTest Results:")
    logger.info(f"Average Reward: {test_metrics['avg_reward']:.2f}")
    logger.info(f"Average Episode Length: {test_metrics['avg_length']:.1f}")
    logger.info(f"Average Seeds Planted: {test_metrics['avg_seeds_planted']:.2f}")
    logger.info(f"Average Plants Harvested: {test_metrics['avg_plants_harvested']:.2f}")
    
    # Compare with rule-based agent if requested
    if args.compare:
        logger.info("\nComparing with rule-based agent...")
        
        # Environment configuration
        env_config = {
            'width': 100,
            'height': 100,
            'water_probability': 0.1
        }
        
        # Create environments
        env1 = GridWorld(**env_config)
        env2 = GridWorld(**env_config)
        
        # Create rule-based agent
        rule_based_agent = create_agent('rule_based', env1)
        
        # Create RL agent (with the trained model)
        rl_agent = create_agent(
            'ppo' if args.algorithm == 'ppo' else 'reinforce',
            env2,
            model_path=model_path
        )
        
        # Prepare agents dictionary
        agents_dict = {
            'Rule-Based': rule_based_agent,
            args.algorithm.upper(): rl_agent
        }
        
        # Compare agents
        comparison = compare_agents(
            env_config=env_config,
            agents_dict=agents_dict,
            num_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            log_level=args.log_level
        )
        
        # Plot comparison
        plot_comparison(comparison)

def main():
    """Main function to run the training and evaluation pipeline."""
    parser = argparse.ArgumentParser(description='Train and evaluate RL farming agent')
    
    # Basic parameters
    parser.add_argument('--algorithm', type=str, choices=['ppo', 'reinforce'],
                      default='ppo', help='RL algorithm to use')
    parser.add_argument('--num-episodes', type=int, default=2000,
                      help='Number of episodes to train')
    parser.add_argument('--eval-episodes', type=int, default=50,
                      help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=1000,
                      help='Maximum steps per episode')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor')
    parser.add_argument('--learning-rate', type=float, default=0.0003,
                      help='Learning rate')
    parser.add_argument('--compare', action='store_true',
                      help='Compare with rule-based agent after training')
    parser.add_argument('--emergence', action='store_true',
                      help='Use simplified rewards for emergent behavior')
    parser.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'],
                      default='info', help='Logging level')
    
    # PPO-specific parameters
    parser.add_argument('--update-timestep', type=int, default=2000,
                      help='Number of timesteps between PPO updates')
    parser.add_argument('--ppo-epochs', type=int, default=10,
                      help='Number of PPO epochs')
    parser.add_argument('--ppo-epsilon', type=float, default=0.2,
                      help='PPO clipping parameter')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                      help='GAE lambda parameter')
    parser.add_argument('--entropy-coef', type=float, default=0.05,
                      help='Entropy coefficient for PPO')
    parser.add_argument('--value-coef', type=float, default=0.5,
                      help='Value loss coefficient for PPO')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                      help='Maximum norm for gradient clipping')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Mini batch size for PPO updates')
    
    args = parser.parse_args()
    
    # Train and evaluate
    train_and_evaluate_agent(args)

if __name__ == "__main__":
    main() 