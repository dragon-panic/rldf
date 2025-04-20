import numpy as np
from environment import GridWorld
from rule_based_agent import RuleBasedAgent
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(format='%(levelname)s - %(message)s')
logger = logging.getLogger("farming_test")
logger.setLevel(logging.INFO)  # Default level is INFO

# Add TRACE level (even more detailed than DEBUG)
TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, message, args, **kwargs)

logging.Logger.trace = trace

# Make matplotlib non-interactive in test environment
if 'pytest' in sys.modules:
    matplotlib.use('Agg')

def test_improved_farming(steps=500, visualize=True, log_level=logging.INFO):
    """
    Test the agent's improved farming capabilities with adjusted parameters.
    
    Args:
        steps: Number of steps to run
        visualize: Whether to show the pygame visualization
        log_level: Logging level to use (default: INFO)
    """
    # Set logging level
    logger.setLevel(log_level)
    
    # Create a farming-focused environment
    env = GridWorld(width=20, height=20, water_probability=0.05)
    
    # Place water sources in specific locations
    for row in range(5, 8):
        for col in range(5, 8):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Make soil around water very fertile
    for row in range(4, 9):
        for col in range(4, 9):
            if env.grid[row, col] == GridWorld.SOIL:
                env.soil_fertility[row, col] = 10.0
    
    # Add a few mature plants to start with
    for row, col in [(10, 10), (11, 11), (12, 12)]:
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Create agent
    agent = RuleBasedAgent(env, start_row=15, start_col=15)
    
    # Set initial agent state
    agent.hunger = 40.0  # Moderate hunger
    agent.thirst = 40.0  # Moderate thirst
    agent.seeds = 3      # Start with a few seeds
    
    # Visualization setup
    if visualize:
        # Check if we're running as a test
        if 'pytest' in sys.modules:
            # Use mock visualizer for tests
            from mock_visualize import MockVisualizer
            visualizer = MockVisualizer(env, cell_size=25, info_width=300)
            visualizer.set_agent(agent)
            pygame_module = __import__('mock_pygame').pygame
        else:
            # Use real visualizer for interactive use
            from visualize import GameVisualizer
            import pygame as pygame_module
            visualizer = GameVisualizer(env, cell_size=25, info_width=300)
            visualizer.set_agent(agent)
            pygame_module.init()
            pygame_module.display.set_caption("Improved Farming Test")
            clock = pygame_module.Clock()
    
    # Tracking metrics
    metrics = {
        'hunger': np.zeros(steps),
        'thirst': np.zeros(steps),
        'health': np.zeros(steps),
        'energy': np.zeros(steps),
        'seeds': np.zeros(steps),
        'plants_harvested': 0,
        'seeds_planted': 0,
        'plants_tended': 0,
        'plants_eaten': 0,
        'water_drunk': 0,
        'task_distribution': {},
        'plant_growth_stages': np.zeros((steps, 3)),  # Track seeds, growing, mature plants
    }
    
    # Run simulation
    logger.info("Running improved farming test...")
    running = True
    paused = False
    
    for step in range(steps):
        if visualize and not 'pytest' in sys.modules:
            # Only process pygame events when not in test mode
            # Process events
            for event in pygame_module.event.get():
                if event.type == pygame_module.QUIT:
                    running = False
                elif event.type == pygame_module.KEYDOWN:
                    if event.key == pygame_module.K_ESCAPE:
                        running = False
                    elif event.key == pygame_module.K_SPACE:
                        paused = not paused
            
            if not running:
                break
            
            if paused:
                visualizer.update()
                if 'clock' in locals():
                    clock.tick(60)
                continue
        
        # Count plants in different growth stages before step
        seed_count = 0
        growing_count = 0
        mature_count = 0
        for r in range(env.height):
            for c in range(env.width):
                if env.grid[r, c] == GridWorld.PLANT:
                    if env.plant_state[r, c] == GridWorld.PLANT_SEED:
                        seed_count += 1
                    elif env.plant_state[r, c] == GridWorld.PLANT_GROWING:
                        growing_count += 1
                    elif env.plant_state[r, c] == GridWorld.PLANT_MATURE:
                        mature_count += 1
        
        metrics['plant_growth_stages'][step] = [seed_count, growing_count, mature_count]
        
        # Record pre-step state
        pre_action_history_len = len(agent.action_history)
        
        # Step the agent
        result = agent.step_ai()
        status = result['status']
        
        # Record metrics
        metrics['hunger'][step] = status['hunger']
        metrics['thirst'][step] = status['thirst']
        metrics['health'][step] = status['health']
        metrics['energy'][step] = status['energy']
        metrics['seeds'][step] = status['seeds']
        
        # Track agent's current task
        if agent.current_task not in metrics['task_distribution']:
            metrics['task_distribution'][agent.current_task] = 0
        metrics['task_distribution'][agent.current_task] += 1
        
        # Track actions taken
        if len(agent.action_history) > pre_action_history_len:
            last_action = agent.action_history[-1]
            action_type = last_action[0] if isinstance(last_action, tuple) else None
            
            if action_type == 'eat':
                metrics['plants_eaten'] += 1
            elif action_type == 'drink':
                metrics['water_drunk'] += 1
            elif action_type == 'plant_seed':
                metrics['seeds_planted'] += 1
            elif action_type == 'tend_plant':
                metrics['plants_tended'] += 1
            elif action_type == 'harvest':
                metrics['plants_harvested'] += 1
        
        # Step the environment more frequently with the new growth rates
        if step % 3 == 0:  # Was every 5 steps, now every 3
            env.step()
        
        # Log status periodically
        if step % 50 == 0:
            # Only show detailed status at DEBUG level or below
            logger.debug(f"Step {step}:")
            logger.debug(f"  Position: {status['position']}")
            logger.debug(f"  Task: {agent.current_task}")
            logger.debug(f"  Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
            logger.debug(f"  Hunger: {status['hunger']:.1f}, Thirst: {status['thirst']:.1f}")
            logger.debug(f"  Seeds: {status['seeds']}")
            logger.debug(f"  Plants - Seed: {seed_count}, Growing: {growing_count}, Mature: {mature_count}")
            logger.debug(f"  Plants harvested: {metrics['plants_harvested']}")
            logger.debug(f"  Seeds planted: {metrics['seeds_planted']}")
            logger.debug(f"  Plants tended: {metrics['plants_tended']}")
            
            # Show just a simple progress indicator at INFO level
            logger.info(f"Step {step}/{steps} - Health: {status['health']:.1f}, Seeds: {status['seeds']}")
        
        # Even more detailed tracing for each step
        logger.trace(f"Step {step} - Position: {status['position']}, Task: {agent.current_task}")
        
        # Update visualization
        if visualize:
            visualizer.update()
            if not 'pytest' in sys.modules and 'clock' in locals():
                clock.tick(60)
        
        # Check if agent died
        if not result['alive']:
            logger.warning(f"Agent died at step {step}! Cause: {result.get('cause_of_death', 'Unknown')}")
            break
    
    if visualize and not 'pytest' in sys.modules:
        pygame_module.quit()
    
    # Print final results
    logger.info("\nImproved Farming Test Results:")
    logger.info(f"Steps completed: {step+1} of {steps}")
    logger.info(f"Final status:")
    logger.info(f"  Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
    logger.info(f"  Hunger: {status['hunger']:.1f}, Thirst: {status['thirst']:.1f}")
    logger.info(f"  Seeds: {status['seeds']}")
    logger.info("Action counts:")
    logger.info(f"  Plants eaten: {metrics['plants_eaten']}")
    logger.info(f"  Water drunk: {metrics['water_drunk']}")
    logger.info(f"  Seeds planted: {metrics['seeds_planted']}")
    logger.info(f"  Plants tended: {metrics['plants_tended']}")
    logger.info(f"  Plants harvested: {metrics['plants_harvested']}")
    
    # Calculate seed multiplication factor
    initial_seeds = 3
    seed_multiplier = status['seeds'] / initial_seeds if initial_seeds > 0 else 0
    logger.info(f"Seed multiplication factor: {seed_multiplier:.1f}x")
    
    # Task distribution at DEBUG level
    logger.debug("\nTask distribution:")
    for task, count in metrics['task_distribution'].items():
        percentage = (count / (step+1)) * 100
        logger.debug(f"  {task}: {count} steps ({percentage:.1f}%)")
    
    # Plot results if not in test mode or specifically requested
    if not 'pytest' in sys.modules or (visualize and '--show-plots' in sys.argv):
        plot_farming_results(metrics, step+1)
    
    # Only return metrics when run directly, not when run as a test
    if __name__ == "__main__":
        return metrics

def plot_farming_results(metrics, steps):
    """Plot the results of the improved farming test."""
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot 1: Agent status
    axs[0].plot(range(steps), metrics['health'][:steps], 'g-', label='Health')
    axs[0].plot(range(steps), metrics['hunger'][:steps], 'r-', label='Hunger')
    axs[0].plot(range(steps), metrics['thirst'][:steps], 'b-', label='Thirst')
    axs[0].plot(range(steps), metrics['energy'][:steps], 'y-', label='Energy')
    axs[0].set_title('Agent Status Over Time')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: Seeds count
    axs[1].plot(range(steps), metrics['seeds'][:steps], 'm-', label='Seeds')
    axs[1].set_title('Seed Count Over Time')
    axs[1].set_ylabel('Count')
    axs[1].grid(True)
    
    # Plot 3: Plant growth stages
    axs[2].stackplot(range(steps), 
                    metrics['plant_growth_stages'][:steps, 0],  # Seeds
                    metrics['plant_growth_stages'][:steps, 1],  # Growing
                    metrics['plant_growth_stages'][:steps, 2],  # Mature
                    labels=['Seeds', 'Growing', 'Mature'],
                    colors=['#FFD700', '#90EE90', '#228B22'])
    axs[2].set_title('Plant Growth Stages Over Time')
    axs[2].set_xlabel('Steps')
    axs[2].set_ylabel('Count')
    axs[2].grid(True)
    axs[2].legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('models/improved_farming_results.png')
    
    # Only show if not in test mode
    if not 'pytest' in sys.modules:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    # Run with default INFO level
    test_improved_farming(steps=300) 
    
    # Uncomment one of these for more verbose output
    # test_improved_farming(steps=300, log_level=logging.DEBUG)
    # test_improved_farming(steps=300, log_level=TRACE_LEVEL) 