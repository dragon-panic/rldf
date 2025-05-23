import numpy as np
from environment import GridWorld
import logging

# Get the existing logger from train.py or create one if it doesn't exist
logger = logging.getLogger(__name__)

class Agent:
    """A simple agent that can interact with the grid environment."""
    
    # Action types
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    EAT = 4
    DRINK = 5
    PLANT_SEED = 6
    TEND_PLANT = 7
    HARVEST = 8
    
    # Direction vectors for movement
    DIRECTION_VECTORS = {
        MOVE_NORTH: (-1, 0),
        MOVE_SOUTH: (1, 0),
        MOVE_EAST: (0, 1),
        MOVE_WEST: (0, -1)
    }
    
    def __init__(self, environment, start_row=0, start_col=0):
        """
        Initialize an agent.
        
        Args:
            environment: A GridWorld instance
            start_row: Initial row position
            start_col: Initial column position
        """
        self.environment = environment
        
        # Position - first ensure we're in bounds
        start_row = min(max(0, start_row), environment.height - 1)
        start_col = min(max(0, start_col), environment.width - 1)
        
        # Ensure agent is not placed on water
        if environment.grid[start_row, start_col] == GridWorld.WATER:
            # Find the nearest non-water cell
            found_valid_position = False
            search_radius = 1
            
            # Expand search radius until we find a non-water cell
            while not found_valid_position and search_radius < max(environment.height, environment.width):
                # Check cells in a square around the starting position
                for r_offset in range(-search_radius, search_radius + 1):
                    for c_offset in range(-search_radius, search_radius + 1):
                        # Only check the perimeter of the square
                        if abs(r_offset) == search_radius or abs(c_offset) == search_radius:
                            r = start_row + r_offset
                            c = start_col + c_offset
                            
                            # Ensure position is within bounds
                            if (0 <= r < environment.height and 
                                0 <= c < environment.width and
                                environment.grid[r, c] != GridWorld.WATER):
                                
                                start_row, start_col = r, c
                                found_valid_position = True
                                break
                    
                    if found_valid_position:
                        break
                
                # Increase search radius if needed
                if not found_valid_position:
                    search_radius += 1
            
            # If still no valid position found, do a full grid search
            if not found_valid_position:
                for r in range(environment.height):
                    for c in range(environment.width):
                        if environment.grid[r, c] != GridWorld.WATER:
                            start_row, start_col = r, c
                            found_valid_position = True
                            break
                    if found_valid_position:
                        break
                        
            logger.info(f"Agent was moved from water to position ({start_row}, {start_col})")
        
        # Set final position
        self.row = start_row
        self.col = start_col
        
        # Status attributes
        self.energy = 100.0  # 0-100
        self.health = 100.0  # 0-100
        self.hunger = 0.0    # 0-100
        self.thirst = 0.0    # 0-100
        self.seeds = 10      # Start with 10 seeds
        
        # History
        self.action_history = []
    
    def get_position(self):
        """Get the current position of the agent."""
        return (self.row, self.col)
    
    def get_status(self):
        """Get the current status of the agent."""
        return {
            'position': self.get_position(),
            'energy': self.energy,
            'health': self.health,
            'hunger': self.hunger,
            'thirst': self.thirst,
            'seeds': self.seeds
        }
    
    def move(self, direction):
        """
        Move the agent in the specified direction.
        
        Args:
            direction: One of MOVE_NORTH, MOVE_SOUTH, MOVE_EAST, MOVE_WEST
        
        Returns:
            bool: Whether the move was successful
        """
        if direction not in self.DIRECTION_VECTORS:
            logger.warning(f"Invalid direction: {direction}")
            return False
        
        # Get the movement vector
        delta_row, delta_col = self.DIRECTION_VECTORS[direction]
        
        # Calculate new position
        new_row = self.row + delta_row
        new_col = self.col + delta_col
        
        # Debug movements
        logger.debug(f"Moving: dir={direction}, delta=({delta_row},{delta_col}), from=({self.row},{self.col}), to=({new_row},{new_col})")
        
        # Check if the move is valid (within bounds and not into water)
        if (0 <= new_row < self.environment.height and 
            0 <= new_col < self.environment.width and
            self.environment.grid[new_row, new_col] != GridWorld.WATER):
            
            # Update position
            self.row = new_row
            self.col = new_col
            
            # Consume energy for movement
            self.energy = max(0, self.energy - 1.0)
            
            # Increase hunger and thirst (reduced rates)
            self.hunger = min(100, self.hunger + 0.25)
            self.thirst = min(100, self.thirst + 0.5)
            
            # Log action
            self.action_history.append(('move', direction))
            return True
        else:
            # Debug why move failed
            if new_row < 0 or new_row >= self.environment.height or new_col < 0 or new_col >= self.environment.width:
                logger.debug(f"Move failed: Out of bounds")
            elif self.environment.grid[new_row, new_col] == GridWorld.WATER:
                logger.debug(f"Move failed: Destination is water")
            return False
    
    def eat(self):
        """
        Eat food from the current cell if available.
        
        Returns:
            bool: Whether the agent was able to eat
        """
        # Check if the current cell has a mature plant
        if (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
            self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_MATURE):
            
            # Reduce hunger
            food_value = 30.0  # A mature plant provides 30 food points
            self.hunger = max(0, self.hunger - food_value)
            
            # Reset the cell to soil after eating
            self.environment.grid[self.row, self.col] = GridWorld.SOIL
            self.environment.plant_state[self.row, self.col] = GridWorld.PLANT_NONE
            
            # Log action
            self.action_history.append(('eat',))
            return True
        
        return False
    
    def drink(self):
        """
        Drink water from the current cell if available.
        
        Returns:
            bool: Whether the agent was able to drink
        """
        # Check if the current cell is next to water
        water_nearby = False
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = self.row + dr, self.col + dc
                if (0 <= r < self.environment.height and 
                    0 <= c < self.environment.width and
                    self.environment.grid[r, c] == GridWorld.WATER):
                    water_nearby = True
                    break
            if water_nearby:
                break
        
        if water_nearby:
            # Reduce thirst
            water_value = 30.0  # Drinking reduces thirst by 30 points
            self.thirst = max(0, self.thirst - water_value)
            
            # Log action
            self.action_history.append(('drink',))
            return True
        
        return False
    
    def plant_seed(self):
        """
        Plant a seed at the current cell if it's suitable soil.
        
        Returns:
            bool: Whether the seed was planted successfully
        """
        # Check if the agent has seeds
        if self.seeds <= 0:
            return False
            
        # Check if the current cell is suitable for planting
        if (self.environment.grid[self.row, self.col] == GridWorld.SOIL and
            self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_NONE and
            self.environment.soil_fertility[self.row, self.col] >= 3.0):
            
            # Plant the seed
            self.environment.grid[self.row, self.col] = GridWorld.PLANT
            self.environment.plant_state[self.row, self.col] = GridWorld.PLANT_SEED
            
            # Consume a seed
            self.seeds -= 1
            
            # Consume energy for planting
            self.energy = max(0, self.energy - 5.0)
            
            # Increase hunger and thirst slightly (reduced rates)
            self.hunger = min(100, self.hunger + 0.5)
            self.thirst = min(100, self.thirst + 0.5)
            
            # Log action
            self.action_history.append(('plant_seed',))
            return True
            
        return False
    
    def tend_plant(self):
        """
        Tend to a plant at the current cell to improve its growth chance.
        
        Returns:
            bool: Whether the plant was tended successfully
        """
        # Check if the current cell has a plant that can be tended
        if (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
            (self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_SEED or
             self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_GROWING)):
             
            # Improve soil fertility slightly
            current_fertility = self.environment.soil_fertility[self.row, self.col]
            self.environment.soil_fertility[self.row, self.col] = min(10.0, current_fertility + 1.0)
            
            # Consume energy for tending
            self.energy = max(0, self.energy - 3.0)
            
            # Increase hunger and thirst slightly (reduced rates)
            self.hunger = min(100, self.hunger + 0.35)
            self.thirst = min(100, self.thirst + 0.35)
            
            # Log action
            self.action_history.append(('tend_plant',))
            return True
            
        return False
    
    def harvest(self):
        """
        Harvest a mature plant from the current cell, obtaining seeds and food.
        
        Returns:
            bool: Whether the plant was harvested successfully
        """
        # Check if the current cell has a mature plant
        if (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
            self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_MATURE):
            
            # Get seeds from the mature plant
            new_seeds = np.random.randint(2, 5)  # 2-4 seeds from each harvest
            self.seeds += new_seeds
            
            # Reset the cell to soil after harvesting
            self.environment.grid[self.row, self.col] = GridWorld.SOIL
            self.environment.plant_state[self.row, self.col] = GridWorld.PLANT_NONE
            
            # Reduce fertility after harvesting
            self.environment.soil_fertility[self.row, self.col] = max(
                0, self.environment.soil_fertility[self.row, self.col] - 2.0
            )
            
            # Consume energy for harvesting
            self.energy = max(0, self.energy - 4.0)
            
            # Increase hunger and thirst slightly (reduced rates)
            self.hunger = min(100, self.hunger + 0.5)
            self.thirst = min(100, self.thirst + 0.5)
            
            # Log action
            self.action_history.append(('harvest', new_seeds))
            return True
            
        return False
    
    def update_status(self):
        """Update the agent's status based on hunger and thirst."""
        # Increase hunger and thirst over time (even when idle)
        self.hunger = min(100, self.hunger + 0.5)  # Passive hunger increase
        self.thirst = min(100, self.thirst + 0.8)  # Passive thirst increase (faster than hunger)
        
        # Energy decreases slightly over time
        self.energy = max(0, self.energy - 0.2)  # Passive energy decrease
        
        # Hunger and thirst affect health
        if self.hunger > 80 or self.thirst > 80:
            self.health = max(0, self.health - 2.0)
        elif self.hunger > 50 or self.thirst > 50:
            self.health = max(0, self.health - 1.0)
        else:
            # Slowly recover health if not hungry or thirsty
            self.health = min(100, self.health + 0.5)
        
        # Energy regeneration is affected by hunger and thirst
        if self.hunger < 30 and self.thirst < 30:
            self.energy = min(100, self.energy + 1.0)
        
        # Check if the agent has died (health = 0)
        if self.health <= 0:
            self.health = 0  # Ensure health is exactly 0 when dead
            return False  # Agent is dead
        
        return True  # Agent is alive
    
    def step(self, action):
        """
        Execute an action and update the agent's state.
        
        Args:
            action: One of the agent's action types
            
        Returns:
            dict: Result of the action including success status
        """
        success = False
        
        if action in [self.MOVE_NORTH, self.MOVE_SOUTH, self.MOVE_EAST, self.MOVE_WEST]:
            success = self.move(action)
        elif action == self.EAT:
            success = self.eat()
        elif action == self.DRINK:
            success = self.drink()
        elif action == self.PLANT_SEED:
            success = self.plant_seed()
        elif action == self.TEND_PLANT:
            success = self.tend_plant()
        elif action == self.HARVEST:
            success = self.harvest()
        
        # Update agent's status
        alive = self.update_status()
        
        return {
            'success': success,
            'alive': alive,
            'status': self.get_status()
        } 