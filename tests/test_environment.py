import numpy as np
from environment import GridWorld
from simple_visualize import visualize_grid

def test_grid_creation():
    """Test the creation of a grid world with various parameters."""
    # Test default grid creation (100x100)
    grid_world = GridWorld()
    assert grid_world.width == 100
    assert grid_world.height == 100
    assert grid_world.grid.shape == (100, 100)
    
    # Test custom-sized grid
    grid_world = GridWorld(width=50, height=30)
    assert grid_world.width == 50
    assert grid_world.height == 30
    assert grid_world.grid.shape == (30, 50)
    
    # Check that grid is initialized with water and soil
    grid = grid_world.get_grid_state()
    assert np.any(grid == GridWorld.WATER), "No water cells found in grid"
    assert np.any(grid == GridWorld.SOIL), "No soil cells found in grid"
    
    print("Grid creation tests passed!")
    # Don't return anything for pytest compatibility

def test_cell_operations():
    """Test getting and setting cell values."""
    # Create a grid world for testing
    grid_world = GridWorld(width=50, height=30)
    
    # Test get_cell
    row, col = 5, 10
    cell_value = grid_world.get_cell(row, col)
    assert cell_value is not None, "Failed to get cell value"
    
    # Test set_cell
    grid_world.set_cell(row, col, GridWorld.PLANT)
    assert grid_world.get_cell(row, col) == GridWorld.PLANT, "Failed to set cell value"
    
    # Test out-of-bounds access
    assert grid_world.get_cell(-1, -1) is None, "Should return None for out-of-bounds coordinates"
    assert not grid_world.set_cell(-1, -1, GridWorld.WATER), "Should return False for out-of-bounds coordinates"
    
    print("Cell operation tests passed!")

def test_reset():
    """Test resetting the grid."""
    # Create a grid world for testing
    grid_world = GridWorld(width=50, height=30)
    
    # Place some plants
    for i in range(10):
        grid_world.set_cell(i, i, GridWorld.PLANT)
    
    # Check that plants exist
    assert np.any(grid_world.grid == GridWorld.PLANT), "No plants found after placing them"
    
    # Reset the grid
    grid_world.reset()
    
    # Check that plants are gone
    assert not np.any(grid_world.grid == GridWorld.PLANT), "Plants still exist after reset"
    
    print("Reset test passed!")

if __name__ == "__main__":
    print("Running GridWorld tests...")
    
    test_grid_creation()
    test_cell_operations()
    test_reset()
    
    print("\nAll tests passed! Now showing a sample visualization:")
    
    # Create a smaller grid for easier visualization
    test_grid = GridWorld(width=20, height=10, water_probability=0.2)
    
    # Add some plants for demonstration
    for i in range(5):
        test_grid.set_cell(np.random.randint(0, 10), np.random.randint(0, 20), GridWorld.PLANT)
    
    # Visualize the grid
    visualize_grid(test_grid)
    
    print("\nTesting sampling of larger grid:")
    large_grid = GridWorld(width=100, height=100, water_probability=0.15)
    visualize_grid(large_grid) 