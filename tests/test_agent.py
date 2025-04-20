import numpy as np
from environment import GridWorld
from agent import Agent
from simple_visualize import visualize_grid

def test_agent_initialization():
    """Test agent initialization and attribute setting."""
    # Create an environment
    env = GridWorld(width=20, height=15, water_probability=0.2)
    
    # Create an agent
    agent = Agent(env, start_row=5, start_col=5)
    
    # Check initial values
    assert agent.row == 5
    assert agent.col == 5
    assert agent.energy == 100.0
    assert agent.health == 100.0
    assert agent.hunger == 0.0
    assert agent.thirst == 0.0
    
    # Test get_position and get_status
    position = agent.get_position()
    assert position == (5, 5)
    
    status = agent.get_status()
    assert status['position'] == (5, 5)
    assert status['energy'] == 100.0
    assert status['health'] == 100.0
    assert status['hunger'] == 0.0
    assert status['thirst'] == 0.0
    
    print("Agent initialization test passed!")
    # For main function only
    if __name__ == "__main__":
        return env, agent

def test_basic_movement(env, agent):
    """Test individual movement actions in a controlled environment."""
    print("\nTesting basic movement:")
    
    # Create a small empty environment without water for controlled testing
    test_env = GridWorld(width=5, height=5, water_probability=0)
    test_agent = Agent(test_env, start_row=2, start_col=2)
    
    # Test each direction
    directions = [
        (Agent.MOVE_NORTH, "North", -1, 0),
        (Agent.MOVE_EAST, "East", 0, 1),
        (Agent.MOVE_SOUTH, "South", 1, 0),
        (Agent.MOVE_WEST, "West", 0, -1)
    ]
    
    for direction, name, dr, dc in directions:
        # Get current position
        row, col = test_agent.row, test_agent.col
        
        # Try to move
        result = test_agent.step(direction)
        
        # Check result
        expected_row = row + dr
        expected_col = col + dc
        print(f"  Move {name}: from ({row}, {col}) to ({test_agent.row}, {test_agent.col})")
        
        assert result['success'] == True
        assert test_agent.row == expected_row
        assert test_agent.col == expected_col
    
    print("  Basic movement tests passed!")
    # For main function only
    if __name__ == "__main__":
        return agent

def test_water_blocking():
    """Test that water blocks movement."""
    print("\nTesting water blocking movement:")
    
    # Create a small environment with water in the middle
    test_env = GridWorld(width=3, height=3, water_probability=0)
    # Set water in the center
    test_env.set_cell(1, 1, GridWorld.WATER)
    
    # Place agent in top-left
    test_agent = Agent(test_env, start_row=0, start_col=0)
    
    # Try to move to the water cell diagonally (should fail)
    # First move right
    result = test_agent.step(Agent.MOVE_EAST)
    assert result['success'] == True
    assert test_agent.row == 0
    assert test_agent.col == 1
    
    # Then try to move down to water (should fail)
    result = test_agent.step(Agent.MOVE_SOUTH)
    assert result['success'] == False
    assert test_agent.row == 0
    assert test_agent.col == 1
    
    print("  Water blocking test passed!")

def test_action_effects(agent):
    """Test that actions have appropriate effects on agent status."""
    print("\nTesting action effects on agent status:")
    
    # Reset agent status
    agent.energy = 100.0
    agent.hunger = 0.0
    agent.thirst = 0.0
    
    # Get initial values
    initial_energy = agent.energy
    initial_hunger = agent.hunger
    initial_thirst = agent.thirst
    
    # Take several move actions
    moves = 5
    for i in range(moves):
        # Try to move in a valid direction
        for direction in [Agent.MOVE_NORTH, Agent.MOVE_EAST, Agent.MOVE_SOUTH, Agent.MOVE_WEST]:
            if agent.move(direction):
                break
    
    # Check that actions had appropriate effects
    print(f"  Initial energy: {initial_energy}, Current: {agent.energy}")
    print(f"  Initial hunger: {initial_hunger}, Current: {agent.hunger}")
    print(f"  Initial thirst: {initial_thirst}, Current: {agent.thirst}")
    
    # Energy should decrease, hunger and thirst should increase
    assert agent.hunger > initial_hunger or agent.thirst > initial_thirst
    
    print("  Action effects test passed!")
    # For main function only
    if __name__ == "__main__":
        return agent

def test_eat_action(env, agent):
    """Test agent's eat action."""
    print("\nTesting eat action:")
    
    # Find a soil cell for planting
    soil_position = None
    for row in range(env.height):
        for col in range(env.width):
            if env.grid[row, col] == GridWorld.SOIL:
                soil_position = (row, col)
                break
        if soil_position:
            break
    
    if soil_position:
        row, col = soil_position
        
        # Plant a mature plant
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
        
        # Move agent to the plant
        agent.row = row
        agent.col = col
        
        # Test hunger before eating
        agent.hunger = 50.0  # Set a baseline hunger level
        initial_hunger = agent.hunger
        
        # Test eat action
        result = agent.step(Agent.EAT)
        assert result['success'] == True
        assert agent.hunger < initial_hunger
        assert env.grid[row, col] == GridWorld.SOIL
        
        print(f"  Agent ate plant at ({row}, {col})")
        print(f"  Hunger before: {initial_hunger}, after: {agent.hunger}")
        print(f"  Cell reset to soil: {env.grid[row, col] == GridWorld.SOIL}")
        
        # Try to eat again (should fail since there's no more plant)
        result = agent.step(Agent.EAT)
        assert result['success'] == False
        
        print("  Eat action test passed!")
    else:
        print("  No suitable soil found for eat test")

def test_drink_action(env, agent):
    """Test agent's drink action."""
    print("\nTesting drink action:")
    
    # Find a water cell and adjacent land
    water_adjacent_to_land = None
    for row in range(env.height):
        for col in range(env.width):
            if env.grid[row, col] == GridWorld.WATER:
                # Check if there's adjacent land
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    r, c = row + dr, col + dc
                    if (0 <= r < env.height and 
                        0 <= c < env.width and 
                        env.grid[r, c] != GridWorld.WATER):
                        water_adjacent_to_land = (r, c)  # Land position
                        break
                if water_adjacent_to_land:
                    break
        if water_adjacent_to_land:
            break
    
    if water_adjacent_to_land:
        land_row, land_col = water_adjacent_to_land
        
        # Move agent to the land cell next to water
        agent.row = land_row
        agent.col = land_col
        
        # Test thirst before drinking
        agent.thirst = 50.0
        initial_thirst = agent.thirst
        
        # Test drink action
        result = agent.step(Agent.DRINK)
        assert result['success'] == True
        assert agent.thirst < initial_thirst
        
        print(f"  Agent drank water near ({land_row}, {land_col})")
        print(f"  Thirst before: {initial_thirst}, after: {agent.thirst}")
        
        print("  Drink action test passed!")
    else:
        print("  No suitable water/land interface found for drink test")

def test_status_updates(agent):
    """Test agent status updates with hunger and thirst."""
    print("\nTesting status updates:")
    
    # Set high hunger and thirst
    agent.health = 100.0
    agent.hunger = 90.0
    agent.thirst = 90.0
    
    # Update status a few times
    for i in range(5):
        agent.update_status()
        print(f"  After update {i+1}: Health = {agent.health:.1f}, Energy = {agent.energy:.1f}")
    
    # Verify health decreased due to hunger and thirst
    assert agent.health < 100.0
    
    # Reset hunger and thirst to low values
    agent.health = 80.0
    agent.hunger = 10.0
    agent.thirst = 10.0
    
    # Update status a few times
    for i in range(5):
        agent.update_status()
        print(f"  After update {i+1}: Health = {agent.health:.1f}, Energy = {agent.energy:.1f}")
    
    # Verify health increased and energy regenerated
    assert agent.health > 80.0
    assert agent.energy > 0.0
    
    print("  Status update test passed!")

def test_idle_status_updates():
    """Test that agent hunger and thirst increase even when idle (no actions)."""
    print("\nTesting idle status updates:")
    
    # Create a simple environment
    env = GridWorld(width=5, height=5, water_probability=0.1)
    
    # Create agent with zero hunger and thirst
    agent = Agent(env, start_row=2, start_col=2)
    agent.hunger = 0.0
    agent.thirst = 0.0
    agent.energy = 100.0
    
    # Initial values
    initial_hunger = agent.hunger
    initial_thirst = agent.thirst
    initial_energy = agent.energy
    
    print(f"  Initial values - Hunger: {initial_hunger}, Thirst: {initial_thirst}, Energy: {initial_energy}")
    
    # Call update_status multiple times without taking any actions
    num_updates = 10
    for i in range(num_updates):
        agent.update_status()
        if i % 3 == 0:  # Print every few updates
            print(f"  After {i+1} updates - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}, Energy: {agent.energy:.1f}")
    
    # Verify hunger and thirst increased
    assert agent.hunger > initial_hunger, f"Hunger should increase when idle (was {initial_hunger}, now {agent.hunger})"
    assert agent.thirst > initial_thirst, f"Thirst should increase when idle (was {initial_thirst}, now {agent.thirst})"
    
    # Now test with high hunger and thirst to ensure energy decreases
    print("\n  Testing with high hunger and thirst:")
    
    # Reset agent but with high hunger and thirst
    agent = Agent(env, start_row=2, start_col=2)
    agent.hunger = 60.0
    agent.thirst = 60.0
    agent.energy = 100.0
    
    initial_energy = agent.energy
    print(f"  Initial values - Hunger: {agent.hunger}, Thirst: {agent.thirst}, Energy: {initial_energy}")
    
    # Call update_status multiple times
    for i in range(num_updates):
        agent.update_status()
        if i % 3 == 0:  # Print every few updates
            print(f"  After {i+1} updates - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}, Energy: {agent.energy:.1f}")
    
    # Now energy should definitely decrease due to high hunger/thirst
    assert agent.energy < initial_energy, f"Energy should decrease when idle with high hunger/thirst (was {initial_energy}, now {agent.energy})"
    
    print(f"  Final values - Hunger: {agent.hunger:.1f}, Thirst: {agent.thirst:.1f}, Energy: {agent.energy:.1f}")
    print("  Idle status update test passed!")

def test_agent_visualization(env, agent):
    """Test agent visualization in the environment."""
    print("\nTesting agent visualization:")
    
    # Visualize the environment with the agent
    visualize_grid(env, max_display_size=15, show_details='type', agent=agent)
    
    # Move the agent and visualize again
    agent.step(Agent.MOVE_EAST)
    visualize_grid(env, max_display_size=15, show_details='type', agent=agent)
    
    # Move near water (if possible)
    water_adjacent_to_land = None
    for row in range(env.height):
        for col in range(env.width):
            if env.grid[row, col] == GridWorld.WATER:
                # Check if there's adjacent land
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    r, c = row + dr, col + dc
                    if (0 <= r < env.height and 
                        0 <= c < env.width and 
                        env.grid[r, c] != GridWorld.WATER):
                        water_adjacent_to_land = (r, c)  # Land position
                        break
                if water_adjacent_to_land:
                    break
        if water_adjacent_to_land:
            break
    
    if water_adjacent_to_land:
        land_row, land_col = water_adjacent_to_land
        
        # Move agent to the land cell next to water
        agent.row = land_row
        agent.col = land_col
        
        # Visualize again
        print("\nAgent next to water:")
        visualize_grid(env, max_display_size=15, show_details='type', agent=agent)
    
    print("  Visualization test passed!")

def find_valid_start_position(env, preferred_row=None, preferred_col=None):
    """
    Find a valid starting position (non-water cell) for an agent.
    
    Args:
        env: GridWorld environment
        preferred_row: Preferred row coordinate (will try to find nearest valid cell if this is water)
        preferred_col: Preferred column coordinate (will try to find nearest valid cell if this is water)
        
    Returns:
        tuple: (row, col) of valid starting position
    """
    # Use center of grid if no preference given
    if preferred_row is None:
        preferred_row = env.height // 2
    if preferred_col is None:
        preferred_col = env.width // 2
    
    # Ensure coordinates are in bounds
    preferred_row = min(max(0, preferred_row), env.height - 1)
    preferred_col = min(max(0, preferred_col), env.width - 1)
    
    # If preferred position is not water, return it
    if env.grid[preferred_row, preferred_col] != GridWorld.WATER:
        return preferred_row, preferred_col
    
    # Find the nearest non-water cell
    found_valid_position = False
    search_radius = 1
    
    # Expand search radius until we find a non-water cell
    while not found_valid_position and search_radius < max(env.height, env.width):
        # Check cells in a square around the preferred position
        for r_offset in range(-search_radius, search_radius + 1):
            for c_offset in range(-search_radius, search_radius + 1):
                # Only check the perimeter of the square
                if abs(r_offset) == search_radius or abs(c_offset) == search_radius:
                    r = preferred_row + r_offset
                    c = preferred_col + c_offset
                    
                    # Ensure position is within bounds
                    if (0 <= r < env.height and 
                        0 <= c < env.width and
                        env.grid[r, c] != GridWorld.WATER):
                        
                        return r, c
        
        # Increase search radius if needed
        search_radius += 1
    
    # If still no valid position found, do a full grid search
    for r in range(env.height):
        for c in range(env.width):
            if env.grid[r, c] != GridWorld.WATER:
                return r, c
    
    # This should never happen unless the entire grid is water
    raise ValueError("Could not find a valid starting position - grid is all water")

def run_scenario():
    """Run a small scenario with the agent interacting with the environment."""
    print("\nRunning agent scenario:")
    
    # Create a smaller environment for the scenario
    env = GridWorld(width=15, height=10, water_probability=0.1)
    
    # Add some plants
    for _ in range(10):
        row = np.random.randint(0, 10)
        col = np.random.randint(0, 15)
        if env.grid[row, col] == GridWorld.SOIL:
            env.set_cell(row, col, GridWorld.PLANT)
            # Set some plants to be mature
            if np.random.random() < 0.5:
                env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Find a valid starting position
    start_row, start_col = find_valid_start_position(env, preferred_row=5, preferred_col=7)
    
    # Create an agent in the middle
    agent = Agent(env, start_row=start_row, start_col=start_col)
    
    # Simulate some hunger and thirst
    agent.hunger = 40.0
    agent.thirst = 60.0
    
    # Visualize initial state
    print("\nInitial state:")
    visualize_grid(env, agent=agent)
    
    # Agent tries to find food or water
    print("\nAgent starts looking for resources:")
    
    # Run simulation for several steps
    for i in range(10):
        # Choose a random action
        action = np.random.choice([
            Agent.MOVE_NORTH, Agent.MOVE_SOUTH, 
            Agent.MOVE_EAST, Agent.MOVE_WEST,
            Agent.EAT, Agent.DRINK
        ])
        
        # Let agent take a step
        result = agent.step(action)
        
        # Display results every few steps
        if i % 3 == 0:
            action_name = {
                Agent.MOVE_NORTH: "Move North",
                Agent.MOVE_SOUTH: "Move South",
                Agent.MOVE_EAST: "Move East",
                Agent.MOVE_WEST: "Move West",
                Agent.EAT: "Eat",
                Agent.DRINK: "Drink"
            }.get(action, "Unknown")
            
            print(f"\nStep {i+1} - Action: {action_name}, Success: {result['success']}")
            visualize_grid(env, agent=agent)
    
    print("\nScenario completed!")

if __name__ == "__main__":
    print("Running agent tests...")
    
    env, agent = test_agent_initialization()
    
    # Test movement in a controlled environment
    test_basic_movement(env, agent)
    test_water_blocking()
    test_action_effects(agent)
    
    # Test agent actions
    test_eat_action(env, agent)
    test_drink_action(env, agent)
    test_status_updates(agent)
    test_idle_status_updates()
    test_agent_visualization(env, agent)
    
    print("\nRunning agent scenario...")
    run_scenario()
    
    print("\nAll tests completed!") 