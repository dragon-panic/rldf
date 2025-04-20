import numpy as np
import unittest
from environment import GridWorld
from agent import Agent

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

if __name__ == '__main__':
    unittest.main() 