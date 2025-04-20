from environment import GridWorld
from agent import Agent

# Create a small test environment
env = GridWorld(width=10, height=10, water_probability=0.1)

# Create an agent at a specific position
agent = Agent(env, start_row=5, start_col=5)

# Ensure the position is soil
env.grid[5, 5] = GridWorld.SOIL
env.plant_state[5, 5] = GridWorld.PLANT_NONE
env.soil_fertility[5, 5] = 5.0

# Print initial state
print(f"Initial grid cell: {env.grid[5, 5]}")
print(f"Expected SOIL value: {GridWorld.SOIL}")
print(f"Initial plant state: {env.plant_state[5, 5]}")
print(f"Initial soil fertility: {env.soil_fertility[5, 5]}")
print(f"Initial agent seeds: {agent.seeds}")

# Plant a seed
result = agent.plant_seed()

# Print result
print(f"\nPlant seed result: {result}")
print(f"Final grid cell: {env.grid[5, 5]}")
print(f"Expected PLANT value: {GridWorld.PLANT}")
print(f"Final plant state: {env.plant_state[5, 5]}")
print(f"Expected PLANT_SEED value: {GridWorld.PLANT_SEED}")
print(f"Final agent seeds: {agent.seeds}")

# Test equality
print(f"\nGrid cell == PLANT?: {env.grid[5, 5] == GridWorld.PLANT}")
print(f"Grid cell == SOIL?: {env.grid[5, 5] == GridWorld.SOIL}")
print(f"Types: grid cell type={type(env.grid[5, 5])}, PLANT type={type(GridWorld.PLANT)}") 