import numpy as np

class GridWorld:
    """A simple 2D grid environment for a farming simulation."""
    
    # Cell types
    EMPTY = 0
    WATER = 1
    SOIL = 2
    PLANT = 3
    
    # Plant growth states
    PLANT_NONE = 0
    PLANT_SEED = 1
    PLANT_GROWING = 2
    PLANT_MATURE = 3
    
    # Characters for visualization
    CELL_CHARS = {
        EMPTY: ' ',
        WATER: '~',
        SOIL: '.',
        PLANT: '*'
    }
    
    # Growth state characters for visualization
    PLANT_CHARS = {
        PLANT_NONE: ' ',
        PLANT_SEED: '.',
        PLANT_GROWING: ',',
        PLANT_MATURE: '*'
    }
    
    def __init__(self, width=100, height=100, water_probability=0.1):
        """
        Initialize the grid world.
        
        Args:
            width: Width of the grid
            height: Height of the grid
            water_probability: Probability of a cell being water during initialization
        """
        self.width = width
        self.height = height
        
        # Main grid - cell types
        self.grid = np.zeros((height, width), dtype=int)
        
        # Cell properties
        self.water_level = np.zeros((height, width), dtype=float)
        self.soil_fertility = np.zeros((height, width), dtype=float)
        self.plant_state = np.zeros((height, width), dtype=int)
        
        self.initialize_grid(water_probability)
    
    def initialize_grid(self, water_probability):
        """Randomly initialize the grid with water sources."""
        # Initialize some cells as water based on probability
        water_cells = np.random.random((self.height, self.width)) < water_probability
        self.grid[water_cells] = self.WATER
        
        # Set water levels to maximum for water cells
        self.water_level[water_cells] = 10.0
        
        # Initialize remaining cells as soil
        soil_cells = self.grid == self.EMPTY
        self.grid[soil_cells] = self.SOIL
        
        # Set random initial soil fertility (3-7 range)
        self.soil_fertility[soil_cells] = np.random.uniform(3.0, 7.0, size=np.sum(soil_cells))
    
    def get_cell(self, row, col):
        """Get the type of a cell at the specified coordinates."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.grid[row, col]
        return None
    
    def set_cell(self, row, col, cell_type):
        """Set the type of a cell at the specified coordinates."""
        if 0 <= row < self.height and 0 <= col < self.width:
            self.grid[row, col] = cell_type
            
            # Reset properties based on new cell type
            if cell_type == self.WATER:
                self.water_level[row, col] = 10.0
                self.soil_fertility[row, col] = 0.0
                self.plant_state[row, col] = self.PLANT_NONE
            elif cell_type == self.SOIL:
                self.water_level[row, col] = 0.0
                # Fertility set to medium if not already set
                if self.soil_fertility[row, col] == 0.0:
                    self.soil_fertility[row, col] = 5.0
                self.plant_state[row, col] = self.PLANT_NONE
            elif cell_type == self.PLANT:
                # Plants start as seeds
                self.plant_state[row, col] = self.PLANT_SEED
            
            return True
        return False
    
    def get_cell_properties(self, row, col):
        """Get all properties of a cell at the specified coordinates."""
        if 0 <= row < self.height and 0 <= col < self.width:
            return {
                'type': self.grid[row, col],
                'water_level': self.water_level[row, col],
                'soil_fertility': self.soil_fertility[row, col],
                'plant_state': self.plant_state[row, col]
            }
        return None
    
    def get_grid_state(self):
        """Return a copy of the current grid state."""
        return self.grid.copy()
    
    def reset(self, water_probability=0.1):
        """Reset the grid to initial state."""
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.water_level = np.zeros((self.height, self.width), dtype=float)
        self.soil_fertility = np.zeros((self.height, self.width), dtype=float)
        self.plant_state = np.zeros((self.height, self.width), dtype=int)
        self.initialize_grid(water_probability)
    
    def calculate_water_proximity(self, row, col, radius=3):
        """Calculate water proximity score for a cell within given radius."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return 0.0
        
        # If the cell is water, it has maximum water level
        if self.grid[row, col] == self.WATER:
            return 10.0
        
        # Check surrounding cells for water sources
        water_score = 0.0
        max_distance = 2 * radius  # Maximum possible Manhattan distance within radius
        
        # Search in surrounding area
        for r in range(max(0, row - radius), min(self.height, row + radius + 1)):
            for c in range(max(0, col - radius), min(self.width, col + radius + 1)):
                # Skip the current cell
                if r == row and c == col:
                    continue
                    
                if self.grid[r, c] == self.WATER:
                    # Calculate Manhattan distance
                    distance = abs(r - row) + abs(c - col)
                    
                    # Only consider cells within specified radius
                    if distance <= radius:
                        # Convert distance to a score (closer = higher score)
                        # A cell at distance 1 gets a higher score than one at distance 2, etc.
                        water_score += (radius - distance + 1) / radius
        
        # Scale the result to be between 0 and 10
        return min(10.0, water_score * 3.0)
    
    def update_water_levels(self):
        """Update water levels based on water sources and proximity."""
        # Water cells always have maximum water level
        self.water_level[self.grid == self.WATER] = 10.0
        
        # For soil and plant cells, calculate water level based on proximity to water
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row, col] in [self.SOIL, self.PLANT]:
                    self.water_level[row, col] = self.calculate_water_proximity(row, col)
    
    def update_fertility(self, fertility_regen_rate=0.1):
        """Regenerate soil fertility over time."""
        # Fertility only applies to soil and plant cells
        valid_cells = np.logical_or(self.grid == self.SOIL, self.grid == self.PLANT)
        
        # Increase fertility up to maximum of 10
        self.soil_fertility[valid_cells] = np.minimum(
            10.0, 
            self.soil_fertility[valid_cells] + fertility_regen_rate
        )
    
    def update_plant_growth(self):
        """Update plant growth states based on water and fertility."""
        for row in range(self.height):
            for col in range(self.width):
                if self.grid[row, col] == self.PLANT:
                    water = self.water_level[row, col]
                    fertility = self.soil_fertility[row, col]
                    
                    # Plants need both water and fertile soil to grow
                    # Increased growth probability by multiplier
                    growth_chance = (water * fertility) / 100.0 * 3.0  # Multiplied by 3 for faster growth
                    growth_chance = min(0.7, growth_chance)  # Cap at 70% chance per update
                    
                    # Apply growth based on current state
                    if self.plant_state[row, col] == self.PLANT_SEED:
                        if np.random.random() < growth_chance:
                            self.plant_state[row, col] = self.PLANT_GROWING
                            # Growth consumes some fertility
                            self.soil_fertility[row, col] = max(0, fertility - 1.0)
                    
                    elif self.plant_state[row, col] == self.PLANT_GROWING:
                        if np.random.random() < growth_chance:
                            self.plant_state[row, col] = self.PLANT_MATURE
                            # Growth consumes more fertility
                            self.soil_fertility[row, col] = max(0, fertility - 2.0)
    
    def step(self, num_steps=1):
        """Step the environment forward by a number of time steps."""
        for _ in range(num_steps):
            # Update water proximity first
            self.update_water_levels()
            
            # Update plant growth
            self.update_plant_growth()
            
            # Regenerate fertility
            self.update_fertility()
            
        return {
            'grid': self.get_grid_state(),
            'water_levels': self.water_level.copy(),
            'soil_fertility': self.soil_fertility.copy(),
            'plant_states': self.plant_state.copy()
        } 