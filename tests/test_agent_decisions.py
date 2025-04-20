import numpy as np
from environment import GridWorld
from rule_based_agent import RuleBasedAgent
from agent import Agent
import sys
import time

def test_agent_decisions(steps=100, scenario="basic"):
    """
    Test the agent's decision-making capabilities in various scenarios.
    
    Args:
        steps: Number of steps to run
        scenario: The scenario to test ("basic", "hunger", "thirst", "farming")
    """
    # Create environment based on scenario
    if scenario == "basic":
        env = create_basic_environment()
    elif scenario == "hunger":
        env = create_hunger_scenario()
    elif scenario == "thirst":
        env = create_thirst_scenario()
    elif scenario == "farming":
        env = create_farming_scenario()
    else:
        print(f"Unknown scenario: {scenario}")
        return
    
    # Create agent
    agent = RuleBasedAgent(env, start_row=10, start_col=10)
    
    # Set initial state based on scenario
    if scenario == "hunger":
        agent.hunger = 75.0
        agent.thirst = 30.0
    elif scenario == "thirst":
        agent.hunger = 30.0
        agent.thirst = 75.0
    elif scenario == "farming":
        agent.hunger = 40.0
        agent.thirst = 40.0
        agent.seeds = 3
    
    # Create visualizer - use mock visualizer when running as a test
    if 'pytest' in sys.modules:
        # Use mock visualizer for tests
        from mock_visualize import MockVisualizer
        visualizer = MockVisualizer(env, cell_size=25, info_width=300)
        visualizer.set_agent(agent)
        from tests.mock_pygame import pygame as pygame_module
    else:
        # Use real visualizer for interactive use
        from visualize import GameVisualizer
        import pygame as pygame_module
        visualizer = GameVisualizer(env, cell_size=25, info_width=300)
        visualizer.set_agent(agent)
        pygame_module.init()
        pygame_module.display.set_caption(f"Agent Decision Test - {scenario.capitalize()}")
        clock = pygame_module.Clock()
    
    # Track metrics
    metrics = {
        'decisions': [],
        'plants_harvested': 0,
        'seeds_planted': 0,
        'plants_tended': 0,
        'water_drunk': 0,
        'plants_eaten': 0,
        'died': False,
        'cause_of_death': None
    }
    
    running = True
    paused = False
    step_count = 0
    delay = 25  # Frames between steps
    timer = 0
    
    while running and step_count < steps:
        # Process events - only when not in test mode
        if not 'pytest' in sys.modules:
            for event in pygame_module.event.get():
                if event.type == pygame_module.QUIT:
                    running = False
                elif event.type == pygame_module.KEYDOWN:
                    if event.key == pygame_module.K_ESCAPE:
                        running = False
                    elif event.key == pygame_module.K_SPACE:
                        paused = not paused
                        print(f"Simulation {'paused' if paused else 'resumed'}")
        
        # Run agent step
        if not paused:
            timer += 1
            if timer >= delay:
                timer = 0
                
                # Perform AI step
                if agent.is_alive:
                    # Record pre-step state
                    pre_seeds = agent.seeds
                    pre_action_history = len(agent.action_history)
                    
                    # Decide the next action (without actually taking it)
                    action = agent.decide_action()
                    
                    # Save the decision
                    metrics['decisions'].append({
                        'step': step_count,
                        'hunger': agent.hunger,
                        'thirst': agent.thirst,
                        'energy': agent.energy,
                        'health': agent.health,
                        'task': agent.current_task,
                        'action': action_to_string(action)
                    })
                    
                    # Take the step
                    result = agent.step_ai()
                    
                    # Track actions
                    if len(agent.action_history) > pre_action_history:
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
                    
                    # Check if agent died
                    if not result['alive'] and not metrics['died']:
                        metrics['died'] = True
                        metrics['cause_of_death'] = result.get('cause_of_death', 'Unknown cause')
                        print(f"Agent died! Cause: {metrics['cause_of_death']}")
                
                step_count += 1
                
                # Print step info every 10 steps
                if step_count % 10 == 0:
                    print(f"Step {step_count}/{steps}, Task: {agent.current_task}, "
                          f"Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}, "
                          f"Health: {agent.health:.1f}")
        
        # Update visualization
        visualizer.update()
        
        # Handle clock ticks for real pygame
        if not 'pytest' in sys.modules and 'clock' in locals():
            clock.tick(60)
    
    # Clean up pygame if not in test mode
    if not 'pytest' in sys.modules:
        pygame_module.quit()
    
    # Print final results
    print("\nTest Results:")
    print(f"Steps completed: {step_count}")
    print(f"Final status - Health: {agent.health:.1f}, Energy: {agent.energy:.1f}, "
          f"Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}")
    print(f"Seeds: {agent.seeds}")
    print(f"Plants eaten: {metrics['plants_eaten']}")
    print(f"Water drunk: {metrics['water_drunk']}")
    print(f"Seeds planted: {metrics['seeds_planted']}")
    print(f"Plants tended: {metrics['plants_tended']}")
    print(f"Plants harvested: {metrics['plants_harvested']}")
    if metrics['died']:
        print(f"Agent died. Cause: {metrics['cause_of_death']}")
    
    # Print task distribution
    task_counts = {}
    for d in metrics['decisions']:
        task = d['task']
        if task not in task_counts:
            task_counts[task] = 0
        task_counts[task] += 1
    
    print("\nTask distribution:")
    for task, count in task_counts.items():
        percentage = (count / len(metrics['decisions'])) * 100
        print(f"  {task}: {count} steps ({percentage:.1f}%)")
    
    # Add assertions for pytest (in test mode)
    if 'pytest' in sys.modules:
        # Basic assertions to check if the agent performed actions
        assert step_count > 0, "No steps were executed"
        assert len(metrics['decisions']) > 0, "No decisions were recorded"
        assert agent.is_alive, "Agent should be alive at the end of the test"
        
        # Check if the agent performed some meaningful actions
        # For basic scenario, we expect at least some farming actions
        if scenario == "basic" or scenario == "farming":
            assert metrics['seeds_planted'] + metrics['plants_tended'] + metrics['plants_harvested'] > 0, \
                "Agent should perform some farming actions in basic/farming scenario"
    
    # Only return metrics when not running as a test
    if not 'pytest' in sys.modules:
        return metrics

def create_basic_environment():
    """Create a basic test environment with mixed resources."""
    env = GridWorld(width=30, height=20, water_probability=0.0)
    
    # Add water sources
    for row in range(5, 8):
        for col in range(5, 8):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Add some mature plants
    for position in [(10, 5), (11, 6), (12, 7), (15, 15), (16, 16)]:
        row, col = position
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Add some growing plants
    for position in [(15, 5), (16, 6), (17, 7)]:
        row, col = position
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_GROWING
    
    # Add some seeds
    for position in [(5, 15), (6, 16), (7, 17)]:
        row, col = position
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_SEED
    
    # Make soil fertile in some areas
    for row in range(2, 5):
        for col in range(12, 15):
            if env.grid[row, col] == GridWorld.SOIL:
                env.soil_fertility[row, col] = 8.0
    
    return env

def create_hunger_scenario():
    """Create an environment to test hunger prioritization."""
    env = GridWorld(width=30, height=20, water_probability=0.0)
    
    # Add water sources far away
    for row in range(1, 3):
        for col in range(25, 28):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Add mature plants relatively close
    for position in [(13, 12), (14, 13), (15, 14)]:
        row, col = position
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    return env

def create_thirst_scenario():
    """Create an environment to test thirst prioritization."""
    env = GridWorld(width=30, height=20, water_probability=0.0)
    
    # Add water sources relatively close
    for row in range(12, 14):
        for col in range(14, 16):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Add mature plants far away
    for position in [(5, 25), (6, 26), (7, 27)]:
        row, col = position
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    return env

def create_farming_scenario():
    """Create an environment to test farming behaviors."""
    env = GridWorld(width=30, height=20, water_probability=0.0)
    
    # Add water sources
    for row in range(5, 8):
        for col in range(5, 8):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Add some mature plants
    for position in [(15, 15), (16, 16)]:
        row, col = position
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Make soil fertile in some areas around water
    for row in range(4, 9):
        for col in range(4, 9):
            if env.grid[row, col] == GridWorld.SOIL:
                env.soil_fertility[row, col] = 8.0
    
    return env

def action_to_string(action):
    """Convert an action code to a string description."""
    if action is None:
        return "None"
    
    action_names = {
        Agent.MOVE_NORTH: "Move North",
        Agent.MOVE_SOUTH: "Move South",
        Agent.MOVE_EAST: "Move East",
        Agent.MOVE_WEST: "Move West",
        Agent.EAT: "Eat",
        Agent.DRINK: "Drink",
        Agent.PLANT_SEED: "Plant Seed",
        Agent.TEND_PLANT: "Tend Plant",
        Agent.HARVEST: "Harvest"
    }
    
    return action_names.get(action, f"Unknown ({action})")

if __name__ == "__main__":
    import sys
    
    # Default parameters
    scenario = "basic"
    steps = 100
    
    # Process command line arguments
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    if len(sys.argv) > 2:
        steps = int(sys.argv[2])
    
    # Run the test
    test_agent_decisions(steps=steps, scenario=scenario) 