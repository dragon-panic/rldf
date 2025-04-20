import numpy as np
from environment import GridWorld
from simple_visualize import visualize_grid

def test_water_proximity():
    """Test water proximity calculation."""
    # Create a small grid with some water
    grid = GridWorld(width=10, height=10, water_probability=0)
    
    # Set specific water sources for testing
    grid.set_cell(2, 2, GridWorld.WATER)  # Water source
    
    # Calculate water proximity for different cells
    water_source = grid.calculate_water_proximity(2, 2)  # The water cell itself
    adjacent = grid.calculate_water_proximity(3, 2)  # Adjacent to water (1 step away)
    further = grid.calculate_water_proximity(5, 5)  # Further from water
    
    # Print results
    print(f"Water proximity test:")
    print(f"  At water source (2,2): {water_source}")
    print(f"  Adjacent to water (3,2): {adjacent}")
    print(f"  Further from water (5,5): {further}")
    
    # Verify decreasing values with distance
    assert water_source == 10.0  # Water source has max water level
    assert adjacent > 0.0  # Adjacent cell has some water
    assert adjacent > further  # Water level decreases with distance
    
    print("  Water proximity test passed!")
    # For the next test
    return grid if __name__ == "__main__" else None

def test_water_levels_update(grid=None):
    """Test updating water levels based on proximity."""
    # Create grid if not provided (for standalone test)
    if grid is None:
        grid = GridWorld(width=10, height=10, water_probability=0)
        grid.set_cell(2, 2, GridWorld.WATER)  # Water source
    else:
        # Ensure the cell at (2,2) is water no matter what
        grid.set_cell(2, 2, GridWorld.WATER)
    
    # Update water levels
    grid.update_water_levels()
    
    # Check some values
    water_at_source = grid.water_level[2, 2]
    water_adjacent = grid.water_level[3, 2]
    water_far = grid.water_level[8, 8]
    
    print(f"\nWater levels update test:")
    print(f"  Water level at source (2,2): {water_at_source}")
    print(f"  Water level adjacent (3,2): {water_adjacent}")
    print(f"  Water level far away (8,8): {water_far}")
    
    # Verify
    assert water_at_source == 10.0, f"Water level at source should be 10.0, got {water_at_source}"
    assert water_adjacent > water_far, f"Water level at adjacent cell ({water_adjacent}) should be greater than far cell ({water_far})"
    print("  Water levels update test passed!")

def test_fertility_regeneration():
    """Test soil fertility regeneration."""
    # Create grid
    grid = GridWorld(width=5, height=5, water_probability=0.1)
    
    # Set specific fertility values for testing
    row, col = 2, 2
    grid.soil_fertility[row, col] = 5.0
    
    # Save initial fertility
    initial_fertility = grid.soil_fertility[row, col]
    
    # Update fertility multiple times
    for i in range(5):
        grid.update_fertility(fertility_regen_rate=0.5)
        print(f"  After update {i+1}, fertility at ({row},{col}): {grid.soil_fertility[row, col]}")
    
    # Check if fertility increased
    assert grid.soil_fertility[row, col] > initial_fertility
    print("  Fertility regeneration test passed!")
    # For main function
    return grid if __name__ == "__main__" else None

def test_plant_growth():
    """Test plant growth based on water and fertility."""
    # Create a custom grid for testing
    grid = GridWorld(width=10, height=10, water_probability=0)
    
    # Add water source adjacent to a plant
    grid.set_cell(1, 1, GridWorld.WATER)
    
    # Plant in different locations with varying water proximity
    grid.set_cell(2, 2, GridWorld.PLANT)  # Near water
    grid.set_cell(5, 5, GridWorld.PLANT)  # Medium distance
    grid.set_cell(8, 8, GridWorld.PLANT)  # Far from water
    
    # Make soil very fertile at the locations (but with varying levels)
    grid.soil_fertility[2, 2] = 10.0  # Maximum fertility
    grid.soil_fertility[5, 5] = 5.0   # Medium fertility
    grid.soil_fertility[8, 8] = 1.0   # Low fertility
    
    # Set specific plant states
    grid.plant_state[2, 2] = GridWorld.PLANT_SEED
    grid.plant_state[5, 5] = GridWorld.PLANT_SEED
    grid.plant_state[8, 8] = GridWorld.PLANT_SEED
    
    # Display initial grid
    print("\nInitial grid (plant states):")
    visualize_grid(grid, show_details='plant_state')
    
    # Update water levels
    grid.update_water_levels()
    
    # Display water levels
    print("\nWater levels:")
    visualize_grid(grid, show_details='water')
    
    # Run multiple updates to test plant growth
    for i in range(10):
        # Step the environment
        grid.step()
        
        # Print plant states after each step
        print(f"\nPlant states after step {i+1}:")
        print(f"  Near water (2,2): {grid.plant_state[2, 2]}")
        print(f"  Medium distance (5,5): {grid.plant_state[5, 5]}")
        print(f"  Far from water (8,8): {grid.plant_state[8, 8]}")
    
    # Display final grid
    print("\nFinal grid (plant states):")
    visualize_grid(grid, show_details='plant_state')
    
    # Plants with better conditions should grow better
    assert grid.plant_state[2, 2] >= grid.plant_state[8, 8], "Plant near water should grow at least as well as plant far from water"
    print("Plant growth test completed!")
    # For main function
    return grid if __name__ == "__main__" else None

def integration_test():
    """Full integration test simulating multiple time steps."""
    # Create a grid
    grid = GridWorld(width=20, height=10, water_probability=0.15)
    
    # Add some plants
    for _ in range(10):
        row = np.random.randint(0, 10)
        col = np.random.randint(0, 20)
        if grid.grid[row, col] == GridWorld.SOIL:
            grid.set_cell(row, col, GridWorld.PLANT)
    
    # Show initial state
    print("\nIntegration Test: Initial state")
    visualize_grid(grid, show_details='type')
    
    # Run simulation for several steps
    for i in range(10):
        grid.step()
        
        if i % 3 == 0:  # Show every few steps to save space
            print(f"\nAfter {i+1} steps:")
            visualize_grid(grid, show_details='type')
            print("\nWater levels:")
            visualize_grid(grid, show_details='water')
            print("\nFertility levels:")
            visualize_grid(grid, show_details='fertility')
            print("\nPlant states:")
            visualize_grid(grid, show_details='plant_state')
    
    # Count plants in different states
    seeds = np.sum(grid.plant_state == GridWorld.PLANT_SEED)
    growing = np.sum(grid.plant_state == GridWorld.PLANT_GROWING)
    mature = np.sum(grid.plant_state == GridWorld.PLANT_MATURE)
    
    print(f"\nFinal plant counts:")
    print(f"  Seeds: {seeds}")
    print(f"  Growing: {growing}")
    print(f"  Mature: {mature}")
    
    return grid if __name__ == "__main__" else None

if __name__ == "__main__":
    print("Running resource and growth mechanics tests...")
    
    grid = test_water_proximity()
    test_water_levels_update(grid)
    
    grid = test_fertility_regeneration()
    
    grid = test_plant_growth()
    
    print("\nRunning full integration test...")
    integration_test()
    
    print("\nAll tests completed!") 