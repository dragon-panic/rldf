import numpy as np
from environment import GridWorld

def visualize_grid(grid_world, max_display_size=20, show_details='type', agent=None):
    """
    Print a text representation of the grid.
    
    Args:
        grid_world: A GridWorld instance
        max_display_size: Maximum number of rows/columns to display
        show_details: What to display ('type', 'water', 'fertility', 'plant_state')
        agent: Optional Agent instance to display on the grid
    """
    grid = grid_world.get_grid_state()
    height, width = grid.shape
    
    # If grid is too large, sample it
    if height > max_display_size or width > max_display_size:
        # Define step sizes to sample the grid
        row_step = max(1, height // max_display_size)
        col_step = max(1, width // max_display_size)
        
        sampled_grid = grid[::row_step, ::col_step]
        display_height, display_width = sampled_grid.shape
    else:
        sampled_grid = grid
        display_height, display_width = height, width
    
    # Print top border
    print("+" + "-" * (display_width * 2 + 1) + "+")
    
    # Print grid
    for row in range(display_height):
        print("| ", end="")
        for col in range(display_width):
            original_row = row * row_step if height > max_display_size else row
            original_col = col * col_step if width > max_display_size else col
            
            # Check if there's an agent at this position
            has_agent = False
            if agent is not None:
                agent_row, agent_col = agent.get_position()
                has_agent = (original_row == agent_row and original_col == agent_col)
            
            # If there's an agent, show the agent character
            if has_agent:
                print("A ", end="")
                continue
                
            cell_type = sampled_grid[row, col]
            
            # Determine what character to show based on the requested details
            if show_details == 'type':
                char = GridWorld.CELL_CHARS.get(cell_type, "?")
                
            elif show_details == 'water':
                # Show water level (0-9 for levels 0-9, W for 10)
                water_level = grid_world.water_level[original_row, original_col]
                if water_level >= 10:
                    char = 'W'
                else:
                    char = str(int(water_level))
                    
            elif show_details == 'fertility':
                # Show fertility level (0-9 for levels 0-9, F for 10)
                fertility = grid_world.soil_fertility[original_row, original_col]
                if fertility >= 10:
                    char = 'F'
                else:
                    char = str(int(fertility))
                    
            elif show_details == 'plant_state':
                if cell_type == GridWorld.PLANT:
                    plant_state = grid_world.plant_state[original_row, original_col]
                    char = GridWorld.PLANT_CHARS.get(plant_state, "?")
                else:
                    char = GridWorld.CELL_CHARS.get(cell_type, "?")
            else:
                char = GridWorld.CELL_CHARS.get(cell_type, "?")
                
            print(char + " ", end="")
        print("|")
    
    # Print bottom border
    print("+" + "-" * (display_width * 2 + 1) + "+")
    
    # Print legend
    print(f"\nLegend ({show_details}):")
    
    if show_details == 'type':
        for cell_type, char in GridWorld.CELL_CHARS.items():
            cell_name = {
                GridWorld.EMPTY: "Empty",
                GridWorld.WATER: "Water",
                GridWorld.SOIL: "Soil",
                GridWorld.PLANT: "Plant"
            }.get(cell_type, "Unknown")
            print(f"  {char} - {cell_name}")
        if agent is not None:
            print(f"  A - Agent")
            
    elif show_details == 'water':
        for i in range(10):
            print(f"  {i} - Water level {i}")
        print("  W - Water level 10 (max)")
        
    elif show_details == 'fertility':
        for i in range(10):
            print(f"  {i} - Fertility level {i}")
        print("  F - Fertility level 10 (max)")
        
    elif show_details == 'plant_state':
        for plant_state, char in GridWorld.PLANT_CHARS.items():
            state_name = {
                GridWorld.PLANT_NONE: "None",
                GridWorld.PLANT_SEED: "Seed",
                GridWorld.PLANT_GROWING: "Growing",
                GridWorld.PLANT_MATURE: "Mature"
            }.get(plant_state, "Unknown")
            print(f"  {char} - Plant {state_name}")
        print("  ~ - Water")
        print("  . - Soil")
        if agent is not None:
            print(f"  A - Agent")
    
    if height > max_display_size or width > max_display_size:
        print(f"\nNote: Showing sampled view of {display_height}x{display_width} from original {height}x{width} grid")
    
    # If agent is provided, show agent status
    if agent is not None:
        status = agent.get_status()
        print("\nAgent Status:")
        print(f"  Position: ({status['position'][0]}, {status['position'][1]})")
        print(f"  Health: {status['health']:.1f}/100")
        print(f"  Energy: {status['energy']:.1f}/100")
        print(f"  Hunger: {status['hunger']:.1f}/100")
        print(f"  Thirst: {status['thirst']:.1f}/100") 