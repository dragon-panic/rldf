import numpy as np
import time
from environment import GridWorld
from agent import Agent

def demonstrate_farming_cycle():
    """
    Demonstrate a complete farming cycle from planting to harvesting.
    This simulation shows how a farmer agent can:
    1. Plant seeds in suitable soil
    2. Tend to plants to improve growth chances
    3. Harvest mature plants to obtain food and more seeds
    """
    # Create a small grid for demonstration
    env = GridWorld(width=5, height=5, water_probability=0.1)
    
    # Create an agent in the grid
    agent = Agent(env, start_row=2, start_col=2)
    
    # Ensure the agent is on a soil cell with good fertility
    env.grid[2, 2] = GridWorld.SOIL
    env.plant_state[2, 2] = GridWorld.PLANT_NONE
    env.soil_fertility[2, 2] = 8.0  # High fertility for demonstration
    
    # Print initial state
    print("\n===== FARMING SIMULATION DEMONSTRATION =====")
    print("\nInitial state:")
    print(f"Agent position: {agent.get_position()}")
    print(f"Agent seeds: {agent.seeds}")
    print(f"Cell type: {env.get_cell_properties(2, 2)['type']}")
    print(f"Soil fertility: {env.get_cell_properties(2, 2)['soil_fertility']}")
    print(f"Plant state: {env.get_cell_properties(2, 2)['plant_state']}")
    
    # STEP 1: Plant a seed
    print("\n----- STEP 1: Planting a Seed -----")
    result = agent.plant_seed()
    print(f"Planting successful: {result}")
    print(f"Agent seeds remaining: {agent.seeds}")
    print(f"Cell type now: {env.get_cell_properties(2, 2)['type']}")
    print(f"Plant state: {env.get_cell_properties(2, 2)['plant_state']}")
    
    # STEP 2: Tend the plant
    print("\n----- STEP 2: Tending the Plant -----")
    # Note the fertility before tending
    fertility_before = env.get_cell_properties(2, 2)['soil_fertility']
    result = agent.tend_plant()
    fertility_after = env.get_cell_properties(2, 2)['soil_fertility']
    
    print(f"Tending successful: {result}")
    print(f"Soil fertility before: {fertility_before}")
    print(f"Soil fertility after: {fertility_after}")
    print(f"Plant state: {env.get_cell_properties(2, 2)['plant_state']}")
    
    # STEP 3: Simulate growth to growing state
    print("\n----- STEP 3: Plant Growing -----")
    print("Simulating time passing (plant grows from seed to growing state)...")
    env.plant_state[2, 2] = GridWorld.PLANT_GROWING
    env.step(2)  # Simulate time passing
    
    print(f"Plant state now: {env.get_cell_properties(2, 2)['plant_state']}")
    
    # STEP 4: Tend the growing plant again
    print("\n----- STEP 4: Tending the Growing Plant -----")
    fertility_before = env.get_cell_properties(2, 2)['soil_fertility']
    result = agent.tend_plant()
    fertility_after = env.get_cell_properties(2, 2)['soil_fertility']
    
    print(f"Tending successful: {result}")
    print(f"Soil fertility before: {fertility_before}")
    print(f"Soil fertility after: {fertility_after}")
    
    # STEP 5: Simulate growth to mature state
    print("\n----- STEP 5: Plant Maturing -----")
    print("Simulating time passing (plant grows from growing to mature state)...")
    env.plant_state[2, 2] = GridWorld.PLANT_MATURE
    env.step(3)  # Simulate time passing
    
    print(f"Plant state now: {env.get_cell_properties(2, 2)['plant_state']}")
    
    # STEP 6: Harvest the mature plant
    print("\n----- STEP 6: Harvesting the Mature Plant -----")
    seeds_before = agent.seeds
    result = agent.harvest()
    seeds_after = agent.seeds
    
    print(f"Harvesting successful: {result}")
    print(f"Agent seeds before: {seeds_before}")
    print(f"Agent seeds after: {seeds_after}")
    print(f"Seeds gained: {seeds_after - seeds_before}")
    print(f"Cell type now: {env.get_cell_properties(2, 2)['type']}")
    print(f"Plant state: {env.get_cell_properties(2, 2)['plant_state']}")
    
    # STEP 7: Plant another seed in the same spot
    print("\n----- STEP 7: Starting a New Farming Cycle -----")
    result = agent.plant_seed()
    print(f"Planting successful: {result}")
    print(f"Agent seeds remaining: {agent.seeds}")
    print(f"Cell type now: {env.get_cell_properties(2, 2)['type']}")
    print(f"Plant state: {env.get_cell_properties(2, 2)['plant_state']}")
    
    print("\n===== FARMING CYCLE COMPLETE =====")
    print("The agent has successfully completed a full farming cycle:")
    print("1. Planted a seed")
    print("2. Tended the young plant")
    print("3. Waited for it to grow")
    print("4. Tended the growing plant")
    print("5. Waited for it to mature")
    print("6. Harvested the mature plant (gaining seeds)")
    print("7. Started the cycle again with a new seed")

if __name__ == "__main__":
    demonstrate_farming_cycle() 