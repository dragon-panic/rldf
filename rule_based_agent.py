import numpy as np
from environment import GridWorld
from agent import Agent

class RuleBasedAgent(Agent):
    """
    A rule-based agent that extends the basic Agent with decision-making logic
    for survival and farming.
    """
    
    # Decision thresholds
    CRITICAL_HUNGER = 80.0
    CRITICAL_THIRST = 80.0
    HIGH_HUNGER = 60.0
    HIGH_THIRST = 60.0
    MEDIUM_HUNGER = 40.0
    MEDIUM_THIRST = 40.0
    LOW_ENERGY = 30.0
    
    # Soil fertility thresholds
    GOOD_FERTILITY = 7.0
    MIN_FERTILITY = 3.0
    
    def __init__(self, environment, start_row=0, start_col=0):
        """Initialize the rule-based agent."""
        super().__init__(environment, start_row, start_col)
        self.target_position = None  # Used for pathfinding
        self.current_task = "explore"  # Current task the agent is performing
        self.is_alive = True  # Track whether the agent is alive
        self.memory = {
            "water_positions": set(),  # Remember water source positions
            "fertile_soil": set(),     # Remember fertile soil positions
            "plants": {                # Remember plant positions and states
                "seeds": set(),
                "growing": set(),
                "mature": set()
            },
            "last_explored": None      # Last position explored
        }
    
    def decide_action(self):
        """
        Decide the next action based on the agent's current state and environment.
        
        Returns:
            int: The action to take
        """
        # First priority: Survival (critical needs)
        if self.hunger > self.CRITICAL_HUNGER:
            # Critical hunger - find and eat food immediately
            action = self.handle_critical_hunger()
            if action is not None:
                return action
        
        if self.thirst > self.CRITICAL_THIRST:
            # Critical thirst - find and drink water immediately
            action = self.handle_critical_thirst()
            if action is not None:
                return action
        
        # Second priority: Moderate needs
        if self.hunger > self.HIGH_HUNGER:
            # High hunger - seek food
            action = self.handle_high_hunger()
            if action is not None:
                return action
        
        if self.thirst > self.HIGH_THIRST:
            # High thirst - seek water
            action = self.handle_high_thirst()
            if action is not None:
                return action
        
        # Third priority: Farming activities if energy is sufficient
        if self.energy > self.LOW_ENERGY:
            # First check if we can harvest a mature plant we're standing on
            if self.check_current_cell_for_harvest():
                self.current_task = "harvesting"
                return self.HARVEST
            
            # Check if we can tend a plant we're standing on
            if self.check_current_cell_for_tending():
                self.current_task = "tending"
                return self.TEND_PLANT
            
            # Check if we should plant a seed
            if self.seeds > 0 and self.check_current_cell_for_planting():
                self.current_task = "planting"
                return self.PLANT_SEED
            
            # Look for farming tasks in the vicinity
            action = self.find_farming_task()
            if action is not None:
                return action
        
        # Fourth priority: Exploration and resource discovery
        self.current_task = "exploring"
        return self.explore()
    
    def handle_critical_hunger(self):
        """Handle critical hunger by finding and eating food immediately."""
        # Check if we're already on a mature plant
        if self.can_eat_here():
            return self.EAT
        
        # Look for a mature plant in memory
        mature_plants = list(self.memory["plants"]["mature"])
        if mature_plants:
            # Move toward the nearest mature plant
            return self.move_toward_position(mature_plants[0])
        
        # Look for mature plants in the visible area
        mature_position = self.scan_for_mature_plants()
        if mature_position:
            self.memory["plants"]["mature"].add(mature_position)
            return self.move_toward_position(mature_position)
        
        # If no mature plants found, move randomly in search of food
        return self.random_move()
    
    def handle_critical_thirst(self):
        """Handle critical thirst by finding and drinking water immediately."""
        # Check if we can drink where we are
        if self.can_drink_here():
            return self.DRINK
        
        # Look for known water sources
        water_positions = list(self.memory["water_positions"])
        if water_positions:
            # Move toward the nearest water source
            return self.move_toward_position(water_positions[0])
        
        # Scan for water sources in the visible area
        water_position = self.scan_for_water()
        if water_position:
            self.memory["water_positions"].add(water_position)
            return self.move_toward_position(water_position)
        
        # If no water found, move randomly in search of water
        return self.random_move()
    
    def handle_high_hunger(self):
        """Handle high hunger by seeking food."""
        # Same as critical hunger but with lower urgency
        return self.handle_critical_hunger()
    
    def handle_high_thirst(self):
        """Handle high thirst by seeking water."""
        # Same as critical thirst but with lower urgency
        return self.handle_critical_thirst()
    
    def can_eat_here(self):
        """Check if the agent can eat at the current position."""
        return (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
                self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_MATURE)
    
    def can_drink_here(self):
        """Check if the agent can drink at the current position."""
        # Check if any adjacent cell is water
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r, c = self.row + dr, self.col + dc
                if (0 <= r < self.environment.height and 
                    0 <= c < self.environment.width and
                    self.environment.grid[r, c] == GridWorld.WATER):
                    return True
        return False
    
    def check_current_cell_for_harvest(self):
        """Check if the current cell has a mature plant that can be harvested."""
        return (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
                self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_MATURE)
    
    def check_current_cell_for_tending(self):
        """Check if the current cell has a plant that can be tended."""
        return (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
                (self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_SEED or
                 self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_GROWING))
    
    def check_current_cell_for_planting(self):
        """Check if the current cell is suitable for planting."""
        return (self.environment.grid[self.row, self.col] == GridWorld.SOIL and
                self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_NONE and
                self.environment.soil_fertility[self.row, self.col] >= self.MIN_FERTILITY)
    
    def find_farming_task(self):
        """Look for farming tasks in the vicinity."""
        # Check if there are growing plants that need tending
        growing_plants = list(self.memory["plants"]["growing"])
        if growing_plants:
            # Move toward the nearest growing plant
            return self.move_toward_position(growing_plants[0])
        
        # Check if there are seeds that need tending
        seeds = list(self.memory["plants"]["seeds"])
        if seeds:
            # Move toward the nearest seed
            return self.move_toward_position(seeds[0])
        
        # If we have seeds, look for good soil to plant in
        if self.seeds > 0:
            fertile_soils = list(self.memory["fertile_soil"])
            if fertile_soils:
                # Move toward the nearest fertile soil
                return self.move_toward_position(fertile_soils[0])
            
            # Scan for fertile soil in the visible area
            fertile_position = self.scan_for_fertile_soil()
            if fertile_position:
                self.memory["fertile_soil"].add(fertile_position)
                return self.move_toward_position(fertile_position)
        
        # If no farming tasks found, return None
        return None
    
    def explore(self):
        """Explore the environment to discover resources."""
        # Choose a random direction to explore
        return self.random_move()
    
    def random_move(self):
        """Choose a random direction to move."""
        directions = [self.MOVE_NORTH, self.MOVE_SOUTH, self.MOVE_EAST, self.MOVE_WEST]
        np.random.shuffle(directions)
        
        # Try each direction until a valid move is found
        for direction in directions:
            delta_row, delta_col = self.DIRECTION_VECTORS[direction]
            new_row = self.row + delta_row
            new_col = self.col + delta_col
            
            # Check if the move is valid
            if (0 <= new_row < self.environment.height and 
                0 <= new_col < self.environment.width and
                self.environment.grid[new_row, new_col] != GridWorld.WATER):
                # Update memory with information about the new position
                self.update_memory(new_row, new_col)
                return direction
        
        # If no valid move is found, stay in place
        return None
    
    def move_toward_position(self, position):
        """
        Move toward a target position using a simple pathfinding approach.
        
        Args:
            position: Tuple (row, col) of the target position
            
        Returns:
            int: Direction to move
        """
        target_row, target_col = position
        
        # Calculate the direction to move
        row_diff = target_row - self.row
        col_diff = target_col - self.col
        
        # Try vertical and horizontal directions in order of priority
        directions = []
        
        # Prioritize the larger difference
        if abs(row_diff) > abs(col_diff):
            # Vertical priority
            if row_diff > 0:
                directions.append(self.MOVE_SOUTH)
            elif row_diff < 0:
                directions.append(self.MOVE_NORTH)
                
            # Then try horizontal
            if col_diff > 0:
                directions.append(self.MOVE_EAST)
            elif col_diff < 0:
                directions.append(self.MOVE_WEST)
        else:
            # Horizontal priority
            if col_diff > 0:
                directions.append(self.MOVE_EAST)
            elif col_diff < 0:
                directions.append(self.MOVE_WEST)
                
            # Then try vertical
            if row_diff > 0:
                directions.append(self.MOVE_SOUTH)
            elif row_diff < 0:
                directions.append(self.MOVE_NORTH)
        
        # Try each direction until a valid move is found
        for direction in directions:
            delta_row, delta_col = self.DIRECTION_VECTORS[direction]
            new_row = self.row + delta_row
            new_col = self.col + delta_col
            
            # Check if the move is valid
            if (0 <= new_row < self.environment.height and 
                0 <= new_col < self.environment.width and
                self.environment.grid[new_row, new_col] != GridWorld.WATER):
                return direction
        
        # If no valid move is found along direct path, use random move
        return self.random_move()
    
    def scan_for_water(self):
        """
        Scan the local area for water sources.
        
        Returns:
            tuple: (row, col) of the nearest water source, or None if none found
        """
        # Define the scan radius
        scan_radius = 5
        
        # Scan the area around the agent
        for dist in range(1, scan_radius + 1):
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    # Only check the perimeter
                    if abs(dr) == dist or abs(dc) == dist:
                        r, c = self.row + dr, self.col + dc
                        if (0 <= r < self.environment.height and 
                            0 <= c < self.environment.width and
                            self.environment.grid[r, c] == GridWorld.WATER):
                            return (r, c)
        
        return None
    
    def scan_for_mature_plants(self):
        """
        Scan the local area for mature plants.
        
        Returns:
            tuple: (row, col) of the nearest mature plant, or None if none found
        """
        # Define the scan radius
        scan_radius = 5
        
        # Scan the area around the agent
        for dist in range(1, scan_radius + 1):
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    # Only check the perimeter
                    if abs(dr) == dist or abs(dc) == dist:
                        r, c = self.row + dr, self.col + dc
                        if (0 <= r < self.environment.height and 
                            0 <= c < self.environment.width and
                            self.environment.grid[r, c] == GridWorld.PLANT and
                            self.environment.plant_state[r, c] == GridWorld.PLANT_MATURE):
                            return (r, c)
        
        return None
    
    def scan_for_fertile_soil(self):
        """
        Scan the local area for fertile soil.
        
        Returns:
            tuple: (row, col) of the nearest fertile soil, or None if none found
        """
        # Define the scan radius
        scan_radius = 5
        
        # Scan the area around the agent
        for dist in range(1, scan_radius + 1):
            for dr in range(-dist, dist + 1):
                for dc in range(-dist, dist + 1):
                    # Only check the perimeter
                    if abs(dr) == dist or abs(dc) == dist:
                        r, c = self.row + dr, self.col + dc
                        if (0 <= r < self.environment.height and 
                            0 <= c < self.environment.width and
                            self.environment.grid[r, c] == GridWorld.SOIL and
                            self.environment.plant_state[r, c] == GridWorld.PLANT_NONE and
                            self.environment.soil_fertility[r, c] >= self.MIN_FERTILITY):
                            return (r, c)
        
        return None
    
    def update_memory(self, row, col):
        """Update the agent's memory with information about a position."""
        # Skip if out of bounds
        if not (0 <= row < self.environment.height and 0 <= col < self.environment.width):
            return
        
        # Update memory based on cell type
        cell_type = self.environment.grid[row, col]
        
        if cell_type == GridWorld.WATER:
            self.memory["water_positions"].add((row, col))
        
        elif cell_type == GridWorld.SOIL:
            # Check soil fertility
            fertility = self.environment.soil_fertility[row, col]
            if fertility >= self.MIN_FERTILITY:
                self.memory["fertile_soil"].add((row, col))
        
        elif cell_type == GridWorld.PLANT:
            # Check plant state
            plant_state = self.environment.plant_state[row, col]
            if plant_state == GridWorld.PLANT_SEED:
                self.memory["plants"]["seeds"].add((row, col))
                # Remove from other sets if present
                self.memory["plants"]["growing"].discard((row, col))
                self.memory["plants"]["mature"].discard((row, col))
                self.memory["fertile_soil"].discard((row, col))
            
            elif plant_state == GridWorld.PLANT_GROWING:
                self.memory["plants"]["growing"].add((row, col))
                # Remove from other sets if present
                self.memory["plants"]["seeds"].discard((row, col))
                self.memory["plants"]["mature"].discard((row, col))
                self.memory["fertile_soil"].discard((row, col))
            
            elif plant_state == GridWorld.PLANT_MATURE:
                self.memory["plants"]["mature"].add((row, col))
                # Remove from other sets if present
                self.memory["plants"]["seeds"].discard((row, col))
                self.memory["plants"]["growing"].discard((row, col))
                self.memory["fertile_soil"].discard((row, col))
    
    def scan_surroundings(self):
        """Scan and update memory with the current surroundings."""
        # Scan in a radius around the agent
        scan_radius = 3
        
        for dr in range(-scan_radius, scan_radius + 1):
            for dc in range(-scan_radius, scan_radius + 1):
                r, c = self.row + dr, self.col + dc
                self.update_memory(r, c)
    
    def step_ai(self):
        """
        Execute one AI step for the agent.
        
        Returns:
            dict: Result of the action
        """
        # If already dead, return without doing anything
        if not self.is_alive:
            return {
                'success': False,
                'alive': False,
                'status': self.get_status(),
                'cause_of_death': 'Health depleted'
            }
        
        # Display the surrounding environment
        print(f"DEBUG - Agent Environment:")
        self.environment.print_surrounding_area(self.row, self.col, 3)  # 3 cells in each direction = 7x7 grid
        print(f"DEBUG - Agent status: Health={self.health:.1f}, Hunger={self.hunger:.1f}, Thirst={self.thirst:.1f}, Energy={self.energy:.1f}, Seeds={self.seeds}")
        
        # Scan surroundings to update memory
        self.scan_surroundings()
        
        # Decide the next action
        action = self.decide_action()
        
        # Execute the action if valid
        if action is not None:
            result = self.step(action)
            # Check if the agent died after this action
            if not result['alive']:
                self.is_alive = False
                result['cause_of_death'] = 'Health depleted'
            return result
        else:
            # Just update status if no action is taken
            alive = self.update_status()
            # Check if agent died from status update
            if not alive:
                self.is_alive = False
                return {
                    'success': False,
                    'alive': False,
                    'status': self.get_status(),
                    'cause_of_death': 'Health depleted'
                }
            return {
                'success': False,
                'alive': True,
                'status': self.get_status()
            }


def test_rule_based_agent():
    """Test the rule-based agent in a simple environment."""
    # Create environment
    env = GridWorld(width=20, height=15, water_probability=0.1)
    
    # Add some plants and water to make it interesting
    for _ in range(10):
        row = np.random.randint(0, env.height)
        col = np.random.randint(0, env.width)
        if env.grid[row, col] == GridWorld.SOIL:
            env.set_cell(row, col, GridWorld.PLANT)
            # Set some plants to be mature
            if np.random.random() < 0.5:
                env.plant_state[row, col] = GridWorld.PLANT_MATURE
            else:
                env.plant_state[row, col] = GridWorld.PLANT_GROWING
    
    # Create agent
    agent = RuleBasedAgent(env, start_row=env.height // 2, start_col=env.width // 2)
    
    # Set initial agent state for test
    agent.hunger = 50.0  # Medium hunger
    agent.thirst = 50.0  # Medium thirst
    
    # Run simulation for a number of steps
    print("Starting rule-based agent simulation...")
    for i in range(100):
        result = agent.step_ai()
        
        # Print status periodically
        if i % 10 == 0:
            status = result['status']
            print(f"Step {i}:")
            print(f"  Position: {status['position']}")
            print(f"  Task: {agent.current_task}")
            print(f"  Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
            print(f"  Hunger: {status['hunger']:.1f}, Thirst: {status['thirst']:.1f}")
            print(f"  Seeds: {status['seeds']}")
            print(f"  Last action: {agent.action_history[-1] if agent.action_history else 'None'}")
            print()
        
        # Step the environment (simulate time passing)
        if i % 5 == 0:
            env.step()
        
        # Break if agent dies
        if not result['alive']:
            print("Agent died!")
            break
    
    print("Simulation completed")
    
    # Print summary
    print("\nFinal Status:")
    print(f"Alive: {result['alive']}")
    status = result['status']
    print(f"Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
    print(f"Hunger: {status['hunger']:.1f}, Thirst: {status['thirst']:.1f}")
    print(f"Seeds: {status['seeds']}")
    print(f"Total actions taken: {len(agent.action_history)}")
    
    # Print action statistics
    action_counts = {}
    for action in agent.action_history:
        action_type = action[0]
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    print("\nAction Statistics:")
    for action_type, count in action_counts.items():
        print(f"{action_type}: {count}")


if __name__ == "__main__":
    test_rule_based_agent() 