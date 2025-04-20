import pygame
import sys
import numpy as np
import argparse
import os
import torch
import time
import logging
from collections import defaultdict
from environment import GridWorld
from agent import Agent
from model import ObservationEncoder, AgentCNN
from rule_based_agent import RuleBasedAgent
from model_based_agent import ModelBasedAgent
from train import PPOAgentCNN, RLAgent, set_log_level
from visualize import GameVisualizer

# Set up logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationVisualizer(GameVisualizer):
    """Enhanced visualization for evaluation with metrics display."""
    
    def __init__(self, grid_world, cell_size=30, info_width=400):
        super().__init__(grid_world, cell_size, info_width)
        self.metrics = defaultdict(list)
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_step = 0
        self.episode_metrics = defaultdict(int)
        self.agent_type = "Unknown"
        self.is_evaluating = False
        self.metrics_font = None
        self.title_font = None
        
    def initialize_pygame(self):
        """Initialize pygame with additional fonts for metrics display."""
        super().initialize_pygame()
        self.metrics_font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 16, bold=True)
    
    def draw_metrics_panel(self):
        """Draw the metrics panel with current statistics."""
        if not self.metrics_font or not self.title_font:
            return
            
        # Draw metrics background
        metrics_bg = pygame.Rect(
            self.grid_width_px, 0,
            self.info_width, self.grid_height_px
        )
        pygame.draw.rect(self.screen, (240, 240, 240), metrics_bg)
        
        # Draw title
        title = f"{self.agent_type} Agent Evaluation"
        title_surface = self.title_font.render(title, True, (0, 0, 0))
        self.screen.blit(title_surface, (self.grid_width_px + 10, 10))
        
        # Draw episode progress
        progress_text = f"Episode: {self.current_episode}/{self.total_episodes}"
        progress_surface = self.metrics_font.render(progress_text, True, (0, 0, 0))
        self.screen.blit(progress_surface, (self.grid_width_px + 10, 40))
        
        # Draw step counter
        step_text = f"Step: {self.episode_step}"
        step_surface = self.metrics_font.render(step_text, True, (0, 0, 0))
        self.screen.blit(step_surface, (self.grid_width_px + 10, 60))
        
        # Draw current episode metrics
        y_pos = 90
        metrics_title = self.title_font.render("Current Episode:", True, (0, 0, 100))
        self.screen.blit(metrics_title, (self.grid_width_px + 10, y_pos))
        y_pos += 25
        
        metrics_to_display = [
            ("Seeds Planted", self.episode_metrics['seeds_planted']),
            ("Plants Tended", self.episode_metrics['plants_tended']),
            ("Plants Harvested", self.episode_metrics['plants_harvested']),
            ("Food Eaten", self.episode_metrics['food_eaten']),
            ("Water Drunk", self.episode_metrics['water_drunk'])
        ]
        
        for label, value in metrics_to_display:
            metric_text = f"{label}: {value}"
            metric_surface = self.metrics_font.render(metric_text, True, (0, 0, 0))
            self.screen.blit(metric_surface, (self.grid_width_px + 20, y_pos))
            y_pos += 20
        
        # Draw agent status with colored indicators
        y_pos += 10
        status_title = self.title_font.render("Agent Status:", True, (0, 0, 100))
        self.screen.blit(status_title, (self.grid_width_px + 10, y_pos))
        y_pos += 25
        
        if hasattr(self.agent, 'hunger'):
            # Determine color based on hunger level (green = good, red = bad)
            hunger_color = self.get_status_color(self.agent.hunger, reverse=True)
            hunger_text = f"Hunger: {self.agent.hunger:.1f}"
            hunger_surface = self.metrics_font.render(hunger_text, True, hunger_color)
            self.screen.blit(hunger_surface, (self.grid_width_px + 20, y_pos))
            y_pos += 20
        
        if hasattr(self.agent, 'thirst'):
            # Determine color based on thirst level (green = good, red = bad)
            thirst_color = self.get_status_color(self.agent.thirst, reverse=True)
            thirst_text = f"Thirst: {self.agent.thirst:.1f}"
            thirst_surface = self.metrics_font.render(thirst_text, True, thirst_color)
            self.screen.blit(thirst_surface, (self.grid_width_px + 20, y_pos))
            y_pos += 20
        
        if hasattr(self.agent, 'energy'):
            # Determine color based on energy level (green = good, red = bad)
            energy_color = self.get_status_color(self.agent.energy)
            energy_text = f"Energy: {self.agent.energy:.1f}"
            energy_surface = self.metrics_font.render(energy_text, True, energy_color)
            self.screen.blit(energy_surface, (self.grid_width_px + 20, y_pos))
            y_pos += 20
        
        if hasattr(self.agent, 'health'):
            # Determine color based on health level (green = good, red = bad)
            health_color = self.get_status_color(self.agent.health)
            health_text = f"Health: {self.agent.health:.1f}"
            health_surface = self.metrics_font.render(health_text, True, health_color)
            self.screen.blit(health_surface, (self.grid_width_px + 20, y_pos))
            y_pos += 20
        
        if hasattr(self.agent, 'seeds'):
            seeds_text = f"Seeds: {self.agent.seeds}"
            seeds_surface = self.metrics_font.render(seeds_text, True, (0, 0, 0))
            self.screen.blit(seeds_surface, (self.grid_width_px + 20, y_pos))
            y_pos += 20
        
        # If we have aggregated metrics from multiple episodes, show averages
        if self.current_episode > 1 and len(self.metrics) > 0:
            y_pos += 20
            avg_title = self.title_font.render("Averages Across Episodes:", True, (0, 0, 100))
            self.screen.blit(avg_title, (self.grid_width_px + 10, y_pos))
            y_pos += 25
            
            # Calculate and display averages
            for metric_name in ['seeds_planted', 'plants_harvested', 'food_eaten', 'water_drunk', 'survival_steps']:
                if metric_name in self.metrics and len(self.metrics[metric_name]) > 0:
                    avg_value = np.mean(self.metrics[metric_name])
                    metric_label = metric_name.replace('_', ' ').title()
                    avg_text = f"Avg {metric_label}: {avg_value:.2f}"
                    avg_surface = self.metrics_font.render(avg_text, True, (0, 0, 0))
                    self.screen.blit(avg_surface, (self.grid_width_px + 20, y_pos))
                    y_pos += 20
    
    def get_status_color(self, value, reverse=False):
        """
        Get a color indicating the status (green = good, red = bad).
        
        Args:
            value: The value to determine color for (0-100)
            reverse: If True, high values are bad (hunger/thirst)
                    If False, high values are good (health/energy)
        
        Returns:
            tuple: RGB color values
        """
        if reverse:
            # For hunger/thirst (high is bad)
            if value < 30:
                return (0, 150, 0)  # Green - good
            elif value < 60:
                return (150, 150, 0)  # Yellow - caution
            elif value < 80:
                return (200, 100, 0)  # Orange - warning
            else:
                return (200, 0, 0)  # Red - danger
        else:
            # For health/energy (high is good)
            if value > 70:
                return (0, 150, 0)  # Green - good
            elif value > 40:
                return (150, 150, 0)  # Yellow - caution
            elif value > 20:
                return (200, 100, 0)  # Orange - warning
            else:
                return (200, 0, 0)  # Red - danger
    
    def update(self):
        """Update the visualization."""
        super().update()
        self.draw_metrics_panel()
    
    def run_evaluation(self, agent, num_episodes=10, max_steps=1000, delay_ms=50):
        """
        Run evaluation of the agent with visualization.
        
        Args:
            agent: The agent to evaluate
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            delay_ms: Delay between steps in milliseconds
        
        Returns:
            dict: Evaluation metrics
        """
        self.agent = agent
        self.total_episodes = num_episodes
        self.agent_type = agent.__class__.__name__
        
        # Initialize pygame if not already done
        if not pygame.get_init():
            self.initialize_pygame()
        
        # Reset metrics
        self.metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            # Reset environment and agent
            self.grid_world.reset()
            
            if hasattr(agent, 'reset_episode'):
                agent.reset_episode()
            else:
                # For rule-based agents that don't have reset_episode
                agent.row = np.random.randint(0, self.grid_world.height)
                agent.col = np.random.randint(0, self.grid_world.width)
                while self.grid_world.grid[agent.row, agent.col] == GridWorld.WATER:
                    agent.row = np.random.randint(0, self.grid_world.height)
                    agent.col = np.random.randint(0, self.grid_world.width)
                agent.energy = 100.0
                agent.health = 100.0
                agent.hunger = 0.0
                agent.thirst = 0.0
                agent.seeds = 10
                if hasattr(agent, 'is_alive'):
                    agent.is_alive = True
            
            # Reset episode metrics
            self.current_episode = episode + 1
            self.episode_step = 0
            self.episode_metrics = defaultdict(int)
            
            # Run episode
            is_alive = True
            for step in range(max_steps):
                # Update episode step counter
                self.episode_step = step + 1
                
                # Get action from agent
                if hasattr(agent, 'decide_action'):
                    # RL agent
                    action = agent.decide_action()
                    success = agent.step(action)
                elif hasattr(agent, 'step_ai'):
                    # Rule-based agent
                    result = agent.step_ai()
                    success = result.get('success', False)
                    action = result.get('action', -1)
                    is_alive = result.get('alive', True)
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
                
                # Track specific actions if successful
                if success:
                    if action == Agent.PLANT_SEED:
                        self.episode_metrics['seeds_planted'] += 1
                    elif action == Agent.TEND_PLANT:
                        self.episode_metrics['plants_tended'] += 1
                    elif action == Agent.HARVEST:
                        self.episode_metrics['plants_harvested'] += 1
                    elif action == Agent.EAT:
                        self.episode_metrics['food_eaten'] += 1
                    elif action == Agent.DRINK:
                        self.episode_metrics['water_drunk'] += 1
                
                # Check if agent died (for agents with is_alive property)
                if hasattr(agent, 'is_alive'):
                    is_alive = agent.is_alive
                
                # Update visualization
                self.update()
                pygame.display.flip()
                
                # Process events (allow quitting)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return self.metrics
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return self.metrics
                
                # Add delay to make visualization visible
                pygame.time.delay(delay_ms)
                
                # Break if agent died
                if not is_alive:
                    break
            
            # Record episode metrics
            self.metrics['seeds_planted'].append(self.episode_metrics['seeds_planted'])
            self.metrics['plants_tended'].append(self.episode_metrics['plants_tended'])
            self.metrics['plants_harvested'].append(self.episode_metrics['plants_harvested'])
            self.metrics['food_eaten'].append(self.episode_metrics['food_eaten'])
            self.metrics['water_drunk'].append(self.episode_metrics['water_drunk'])
            self.metrics['survival_steps'].append(self.episode_step)
            
            # Log episode results
            logger.info(f"Episode {episode + 1}/{num_episodes}: "
                      f"Steps: {self.episode_step}, "
                      f"Seeds Planted: {self.episode_metrics['seeds_planted']}, "
                      f"Plants Harvested: {self.episode_metrics['plants_harvested']}")
            
            # Pause briefly between episodes
            pygame.time.delay(500)
        
        # Calculate aggregate metrics
        results = {}
        for key in self.metrics:
            results[f'avg_{key}'] = np.mean(self.metrics[key])
            results[f'std_{key}'] = np.std(self.metrics[key])
        
        return results

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
    
    elif agent_type in ['ppo', 'reinforce']:
        # Create model-based agent with the specified model
        return ModelBasedAgent(env, model_path=model_path)
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def main():
    """Main function to visualize agent evaluation."""
    parser = argparse.ArgumentParser(description='Visualize agent evaluation')
    parser.add_argument('--agent-type', type=str, choices=['rule_based', 'ppo', 'reinforce'],
                      default='ppo', help='Type of agent to evaluate')
    parser.add_argument('--model-path', type=str, default='models/ppo_trained_agent.pth',
                      help='Path to the trained model file')
    parser.add_argument('--width', type=int, default=30,
                      help='Width of the environment grid')
    parser.add_argument('--height', type=int, default=20,
                      help='Height of the environment grid')
    parser.add_argument('--cell-size', type=int, default=25,
                      help='Size of grid cells in pixels')
    parser.add_argument('--num-episodes', type=int, default=5,
                      help='Number of episodes to evaluate')
    parser.add_argument('--max-steps', type=int, default=500,
                      help='Maximum steps per episode')
    parser.add_argument('--delay', type=int, default=50,
                      help='Delay between steps in milliseconds')
    parser.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'],
                      default='info', help='Logging level')
    args = parser.parse_args()
    
    # Set the logging level
    set_log_level(args.log_level)
    
    # Create environment
    env = GridWorld(width=args.width, height=args.height, water_probability=0.15)
    
    # Create agent
    agent = create_agent(args.agent_type, env, args.model_path)
    
    # Create visualizer
    visualizer = EvaluationVisualizer(env, cell_size=args.cell_size)
    
    # Run evaluation
    logger.info(f"Visualizing evaluation of {args.agent_type} agent...")
    metrics = visualizer.run_evaluation(
        agent=agent,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        delay_ms=args.delay
    )
    
    # Print results
    logger.info("\nEvaluation Results:")
    if hasattr(metrics, 'get'):
        logger.info(f"Average Survival Time: {metrics.get('avg_survival_steps', 0):.1f} steps")
        logger.info(f"Average Plants Harvested: {metrics.get('avg_plants_harvested', 0):.2f}")
        logger.info(f"Average Seeds Planted: {metrics.get('avg_seeds_planted', 0):.2f}")
    
    pygame.quit()

if __name__ == "__main__":
    main() 