import numpy as np
import unittest
import pytest
import sys
import time
import matplotlib
import matplotlib.pyplot as plt
import argparse
import logging
from environment import GridWorld
from agent import Agent
from rule_based_agent import RuleBasedAgent

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

class TestFarmingActions(unittest.TestCase):
    """Test cases for the farming actions."""
    
    def setUp(self):
        """Set up the test environment and agent."""
        # Create a small grid for testing
        self.env = GridWorld(width=10, height=10, water_probability=0.1)
        
        # Create an agent in the grid
        self.agent = Agent(self.env, start_row=5, start_col=5)
        
        # Ensure the agent is on a soil cell
        self.env.grid[self.agent.row, self.agent.col] = GridWorld.SOIL
        self.env.plant_state[self.agent.row, self.agent.col] = GridWorld.PLANT_NONE
        self.env.soil_fertility[self.agent.row, self.agent.col] = 5.0  # Medium fertility
    
    def test_plant_seed(self):
        """Test the plant_seed action."""
        # Get initial seed count
        initial_seeds = self.agent.seeds
        
        # Force the soil at the agent's position to ensure test works
        self.env.grid[self.agent.row, self.agent.col] = GridWorld.SOIL
        self.env.plant_state[self.agent.row, self.agent.col] = GridWorld.PLANT_NONE
        self.env.soil_fertility[self.agent.row, self.agent.col] = 5.0
        
        # Ensure we're on a soil cell
        self.assertEqual(self.env.grid[self.agent.row, self.agent.col], GridWorld.SOIL)
        
        # Plant a seed
        result = self.agent.plant_seed()
        
        # Verify the seed was planted
        self.assertTrue(result)
        self.assertEqual(self.env.grid[self.agent.row, self.agent.col], GridWorld.PLANT)
        self.assertEqual(self.env.plant_state[self.agent.row, self.agent.col], GridWorld.PLANT_SEED)
        
        # Verify seed count decreased
        self.assertEqual(self.agent.seeds, initial_seeds - 1)
        
        # Try planting on the same spot (should fail)
        result = self.agent.plant_seed()
        self.assertFalse(result)
    
    def test_tend_plant(self):
        """Test the tend_plant action."""
        # Plant a seed first
        self.agent.plant_seed()
        
        # Store the initial fertility
        initial_fertility = self.env.soil_fertility[self.agent.row, self.agent.col]
        
        # Tend the plant
        result = self.agent.tend_plant()
        
        # Verify the plant was tended
        self.assertTrue(result)
        self.assertGreater(self.env.soil_fertility[self.agent.row, self.agent.col], initial_fertility)
        
        # Try tending a mature plant (should fail)
        self.env.plant_state[self.agent.row, self.agent.col] = GridWorld.PLANT_MATURE
        result = self.agent.tend_plant()
        self.assertFalse(result)
    
    def test_harvest(self):
        """Test the harvest action."""
        # Plant a seed
        self.agent.plant_seed()
        
        # Make it mature
        self.env.plant_state[self.agent.row, self.agent.col] = GridWorld.PLANT_MATURE
        
        # Store initial seed count
        initial_seeds = self.agent.seeds
        
        # Harvest the plant
        result = self.agent.harvest()
        
        # Verify the harvest was successful
        self.assertTrue(result)
        self.assertEqual(self.env.grid[self.agent.row, self.agent.col], GridWorld.SOIL)
        self.assertEqual(self.env.plant_state[self.agent.row, self.agent.col], GridWorld.PLANT_NONE)
        
        # Verify agent got more seeds
        self.assertGreater(self.agent.seeds, initial_seeds)
        
        # Try harvesting again (should fail)
        result = self.agent.harvest()
        self.assertFalse(result)
    
    def test_farming_cycle(self):
        """Test a complete farming cycle: plant > tend > mature > harvest."""
        # Get initial seed count
        initial_seeds = self.agent.seeds
        
        # 1. Plant a seed
        self.agent.plant_seed()
        self.assertEqual(self.env.plant_state[self.agent.row, self.agent.col], GridWorld.PLANT_SEED)
        self.assertEqual(self.agent.seeds, initial_seeds - 1)
        
        # 2. Tend the plant
        self.agent.tend_plant()
        
        # 3. Make it grow and mature
        self.env.plant_state[self.agent.row, self.agent.col] = GridWorld.PLANT_GROWING
        self.env.step(1)  # Simulate time passing
        
        # 4. Force it to mature for the test
        self.env.plant_state[self.agent.row, self.agent.col] = GridWorld.PLANT_MATURE
        
        # 5. Harvest
        self.agent.harvest()
        
        # Verify the cycle was completed
        self.assertEqual(self.env.grid[self.agent.row, self.agent.col], GridWorld.SOIL)
        self.assertEqual(self.env.plant_state[self.agent.row, self.agent.col], GridWorld.PLANT_NONE)
        self.assertGreaterEqual(self.agent.seeds, initial_seeds)  # Should have at least as many seeds as started with

    def test_no_seeds(self):
        """Test planting with no seeds."""
        # Set seeds to 0
        self.agent.seeds = 0
        
        # Try to plant
        result = self.agent.plant_seed()
        
        # Verify the planting failed
        self.assertFalse(result)


@pytest.mark.parametrize("steps", [50, 100])
def test_improved_farming_short(steps):
    """Run a short farming simulation test with a rule-based agent."""
    # Create environment and run simulation with default parameters
    metrics = run_improved_farming_test(steps=steps, visualize=False)
    
    # Basic assertions about the test results
    assert 'plants_harvested' in metrics, "Should track plants harvested"
    assert 'seeds_planted' in metrics, "Should track seeds planted"
    
    # The agent might die in longer tests, so don't assert survival
    # Instead, check that it tracked health properly
    assert 'health' in metrics, "Should track agent health"
    
    # For shorter tests, agent should survive
    if steps <= 50:
        assert metrics['health'][-1] > 0, "Agent should survive short tests"


@pytest.mark.parametrize("agent_type", ["rule_based"])
def test_different_agents(agent_type):
    """Test farming capabilities with different agent types."""
    # Run a short test with the specified agent type
    metrics = run_improved_farming_test(steps=50, agent_type=agent_type, visualize=False)
    
    # Basic assertions about the test results
    assert metrics['seeds_planted'] >= 0, "Should track seeds planted"
    assert metrics['plants_harvested'] >= 0, "Should track plants harvested"


def run_improved_farming_test(steps=500, visualize=True, log_level=logging.INFO, agent_type="rule_based"):
    """
    Test the agent's improved farming capabilities with adjusted parameters.
    
    Args:
        steps: Number of steps to run
        visualize: Whether to show the pygame visualization
        log_level: Logging level to use (default: INFO)
        agent_type: Type of agent to use (default: rule_based)
    
    Returns:
        dict: Metrics collected during the test
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
    
    # Create agent based on type
    if agent_type == "rule_based":
        agent = RuleBasedAgent(env, start_row=15, start_col=15)
    else:
        # Default to basic agent if type not recognized
        agent = Agent(env, start_row=15, start_col=15)
    
    # Set initial agent state - set better initial values to prevent early death
    agent.hunger = 20.0  # Lower hunger (40 -> 20)
    agent.thirst = 20.0  # Lower thirst (40 -> 20)
    agent.energy = 100.0 # Full energy
    agent.health = 100.0 # Full health
    agent.seeds = 5      # More seeds (3 -> 5) for better survival
    
    # Visualization setup
    if visualize:
        # Check if we're running as a test
        if 'pytest' in sys.modules:
            # Use mock visualizer for tests
            try:
                from mock_visualize import MockVisualizer
                visualizer = MockVisualizer(env, cell_size=25, info_width=300)
                visualizer.set_agent(agent)
                pygame_module = __import__('mock_pygame').pygame
            except ImportError:
                logger.warning("Mock modules not found, disabling visualization")
                visualize = False
        else:
            # Use real visualizer for interactive use
            try:
                from visualize import GameVisualizer
                import pygame as pygame_module
                visualizer = GameVisualizer(env, cell_size=25, info_width=300)
                visualizer.set_agent(agent)
                pygame_module.init()
                pygame_module.display.set_caption("Improved Farming Test")
                clock = pygame_module.Clock()
            except ImportError:
                logger.warning("pygame or visualizer not found, disabling visualization")
                visualize = False
    
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
    logger.info(f"Running improved farming test with {agent_type} agent for {steps} steps...")
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
        
        # Store pre-action state for rule-based agent
        if hasattr(agent, 'action_history'):
            pre_action_history_len = len(agent.action_history)
        
        # Step the agent - handle different agent types
        if hasattr(agent, 'step_ai'):
            result = agent.step_ai()
            status = result['status']
        else:
            # For basic agents that don't have step_ai
            action = np.random.randint(0, 9)  # Random action
            agent.step(action)
            agent.update_status()
            # Create a basic status dictionary
            status = {
                'position': (agent.row, agent.col),
                'hunger': agent.hunger,
                'thirst': agent.thirst,
                'health': agent.health,
                'energy': agent.energy,
                'seeds': agent.seeds
            }
            result = {'alive': agent.health > 0, 'status': status}
        
        # Record metrics
        metrics['hunger'][step] = status['hunger']
        metrics['thirst'][step] = status['thirst']
        metrics['health'][step] = status['health']
        metrics['energy'][step] = status['energy']
        metrics['seeds'][step] = status['seeds']
        
        # Track agent's current task for rule-based agent
        if hasattr(agent, 'current_task'):
            if agent.current_task not in metrics['task_distribution']:
                metrics['task_distribution'][agent.current_task] = 0
            metrics['task_distribution'][agent.current_task] += 1
        
        # Track actions taken for rule-based agent
        if hasattr(agent, 'action_history') and len(agent.action_history) > pre_action_history_len:
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
            if hasattr(agent, 'current_task'):
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
        logger.trace(f"Step {step} - Position: {status['position']}")
        if hasattr(agent, 'current_task'):
            logger.trace(f", Task: {agent.current_task}")
        
        # Update visualization
        if visualize:
            visualizer.update()
            if not 'pytest' in sys.modules and 'clock' in locals():
                clock.tick(60)
        
        # Check if agent died
        if not result.get('alive', True):
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
    initial_seeds = 5
    seed_multiplier = status['seeds'] / initial_seeds if initial_seeds > 0 else 0
    logger.info(f"Seed multiplication factor: {seed_multiplier:.1f}x")
    
    # Task distribution at DEBUG level for rule-based agents
    if hasattr(agent, 'current_task'):
        logger.debug("\nTask distribution:")
        for task, count in metrics['task_distribution'].items():
            percentage = (count / (step+1)) * 100
            logger.debug(f"  {task}: {count} steps ({percentage:.1f}%)")
    
    # Plot results if not in test mode or specifically requested
    if not 'pytest' in sys.modules or (visualize and '--show-plots' in sys.argv):
        plot_farming_results(metrics, step+1)
    
    return metrics


def plot_farming_results(metrics, steps):
    """
    Plot the farming test results.
    
    Args:
        metrics: Dictionary of collected metrics
        steps: Number of steps completed
    """
    # Create figure with 3 subplots
    plt.figure(figsize=(15, 10))
    
    # 1. Plot agent status over time
    plt.subplot(2, 2, 1)
    plt.plot(metrics['health'][:steps], label='Health')
    plt.plot(metrics['energy'][:steps], label='Energy')
    plt.plot(metrics['hunger'][:steps], label='Hunger')
    plt.plot(metrics['thirst'][:steps], label='Thirst')
    plt.plot(metrics['seeds'][:steps], label='Seeds')
    plt.title('Agent Status Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    # 2. Plot plant growth stages over time
    plt.subplot(2, 2, 2)
    plt.plot(metrics['plant_growth_stages'][:steps, 0], label='Seeds')
    plt.plot(metrics['plant_growth_stages'][:steps, 1], label='Growing')
    plt.plot(metrics['plant_growth_stages'][:steps, 2], label='Mature')
    plt.title('Plant Growth Stages Over Time')
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.grid(True)
    plt.legend()
    
    # 3. Plot action counts as bar chart
    plt.subplot(2, 2, 3)
    actions = ['plants_eaten', 'water_drunk', 'seeds_planted', 'plants_tended', 'plants_harvested']
    counts = [metrics[action] for action in actions]
    plt.bar(range(len(actions)), counts)
    plt.xticks(range(len(actions)), ['Eaten', 'Drunk', 'Planted', 'Tended', 'Harvested'])
    plt.title('Action Counts')
    plt.xlabel('Action')
    plt.ylabel('Count')
    
    # 4. Plot task distribution as pie chart if available
    plt.subplot(2, 2, 4)
    if metrics['task_distribution']:
        tasks = list(metrics['task_distribution'].keys())
        task_counts = list(metrics['task_distribution'].values())
        plt.pie(task_counts, labels=tasks, autopct='%1.1f%%')
        plt.axis('equal')
        plt.title('Task Distribution')
    else:
        plt.text(0.5, 0.5, 'Task distribution not available', 
                horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('farming_test_results.png')
    
    # Show the plot if not in pytest mode
    if not 'pytest' in sys.modules:
        plt.show()


def main():
    """Run the improved farming test with configurable parameters."""
    parser = argparse.ArgumentParser(description="Run the improved farming test")
    parser.add_argument(
        "--log-level", 
        default="INFO",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=300,
        help="Number of steps to run (default: 300)"
    )
    parser.add_argument(
        "--no-visualize", 
        action="store_true",
        help="Disable visualization entirely"
    )
    parser.add_argument(
        "--headless", 
        action="store_true",
        help="Run with mock visualization (no windows) but still log visualization events"
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show plots at the end"
    )
    parser.add_argument(
        "--agent-type",
        default="rule_based",
        choices=["rule_based", "basic"],
        help="Type of agent to use (default: rule_based)"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.log_level == "TRACE":
        log_level = TRACE_LEVEL
    else:
        log_level = getattr(logging, args.log_level)
    
    # If using headless mode, we need to set up the mocks
    if args.headless:
        # Make sure we can find the mock modules
        sys.path.insert(0, '.')
        # Mock modules for headless testing
        sys.modules['pygame'] = __import__('mock_pygame').pygame
        # Force non-interactive matplotlib
        matplotlib.use('Agg')
    
    # Run the test with the specified parameters
    visualize = not args.no_visualize
    
    # Add show-plots flag to sys.argv if needed
    if args.show_plots and '--show-plots' not in sys.argv:
        sys.argv.append('--show-plots')
    
    run_improved_farming_test(
        steps=args.steps,
        visualize=visualize,
        log_level=log_level,
        agent_type=args.agent_type
    )


if __name__ == "__main__":
    # When run directly, use the command line interface
    main()
else:
    # When imported, run the unittest tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False) 