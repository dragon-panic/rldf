# RL Farming Simulation

A reinforcement learning project for simulating farming agents in a resource-based environment, inspired by Dwarf Fortress-style simulation games.

## Project Overview

This project aims to develop AI-controlled agents that learn to survive and thrive in a simulated environment through reinforcement learning rather than hard-coded behaviors. The final goal is to have agents with different specializations (farmers, fighters, organizers) working together in a complex ecosystem.

We're starting with the farmer agent type in a simplified environment to establish core mechanics.

## Phase 1: Basic Grid Environment ✓

- Created a 2D grid world with basic cells (water, soil, plant)
- Implemented basic grid manipulation methods
- Added simple visualization

## Phase 2: Resources and Growth Mechanics ✓

- Added cell properties:
  - Water level (0-10)
  - Soil fertility (0-10)
  - Plant growth states (None, Seed, Growing, Mature)
- Implemented resource dynamics:
  - Water proximity calculation
  - Soil fertility regeneration
  - Plant growth based on water and fertility
- Enhanced visualization to display different resource properties

## Phase 3: Agent Implementation ✓

- Created Agent class with:
  - Position tracking
  - Health, energy, hunger and thirst status
  - Basic actions (move, eat, drink)
  - Status updates based on hunger/thirst
- Added agent tests and basic scenarios
- Implemented agent visualizations

## Phase 4: Enhanced Visualization ✓

- Added pygame-based graphical interface
- Color coding for different cell types and states
- Interactive agent control with keyboard
- Real-time status display
- Recent action history

## Phase 5: Farming Actions ✓

- Added farming-specific actions:
  - Plant seeds in fertile soil
  - Tend to plants to improve growth
  - Harvest mature plants to gain seeds
- Implemented seed inventory for agents
- Created complete farming cycle simulation
- Updated visualization to show farming actions and seed count
- Added comprehensive tests for farming mechanics

## Phase 6: Rule-Based Agent Logic ✓

- Implemented a rule-based AI agent with:
  - Priority-based decision making (survival before farming)
  - Memory of environment features (water, fertile soil, plants)
  - Basic pathfinding to resources
  - Farming strategy (planting, tending, harvesting)
- Added visualization for AI-controlled agent
- Created testing framework to evaluate agent performance
- Implemented metrics to measure survival and farming success
- Demonstrated that agent can effectively farm and survive

## Phase 7: Model-Based Agent Implementation ✓

- Added neural network-based agent that uses a trained model
- Implemented agent selection feature (rule-based or model-based)
- Created a comparison tool to run both agents side by side
- Enhanced visualization to display agent type and current task

## Phase 8: Preliminary RL Training ✓

- Implemented basic REINFORCE algorithm for policy gradient learning
- Created training loop with episode-based updates
- Added reward function balancing survival and farming priorities
- Implemented model saving and loading functionality
- Added visualization of training progress

## Phase 9: Complete RL Implementation ✓

- Enhanced the RL implementation with PPO algorithm:
  - Added value function estimation
  - Implemented policy clipping
  - Added entropy bonus for exploration
  - Created proper batching of experiences
- Implemented Generalized Advantage Estimation (GAE)
- Added comprehensive testing framework for RL components
- Improved training efficiency with proper hyperparameter settings
- Created visualizations for value function assessment

## Project Structure

- `environment.py`: Defines the GridWorld class for the 2D environment
- `agent.py`: Implements the Agent class with actions and status tracking
- `rule_based_agent.py`: Extends Agent with rule-based decision-making logic
- `simple_visualize.py`: Basic text visualization of the grid
- `visualize.py`: Pygame-based graphical visualization
- `main.py`: Unified visualization system with multiple modes (manual, AI, hybrid)
- `model.py`: Neural network architecture for agent decision making
- `model_based_agent.py`: Neural network-based agent implementation
- `compare_agents.py`: Tool to run both agent types side by side for comparison
- `train.py`: Implementation of RL algorithms (REINFORCE and PPO)
- `test_ppo.py`: Test script to verify PPO implementation components
- `test_environment.py`: Tests to verify grid functionality
- `test_resources.py`: Tests for resource dynamics and growth mechanics
- `test_agent.py`: Tests for agent functionality
- `test_farming.py`: Tests for farming actions
- `test_farming_cycle.py`: Demonstrates a complete farming cycle
- `test_rule_based_agent.py`: Evaluates the rule-based agent's performance
- `test_visualize.py`: Demo for graphical visualization

## Usage

Run the test scripts to verify the environment setup:

```bash
python test_environment.py      # Test basic grid functionality
python test_resources.py        # Test resource dynamics
python test_agent.py            # Test agent functionality
python test_farming.py          # Test farming actions
python test_farming_cycle.py    # See a complete farming cycle demonstration
python test_rule_based_agent.py # Test rule-based agent performance
python test_visualize.py        # Run graphical visualization demo
python test_ppo.py              # Test PPO implementation components
```

Use the main.py script for an interactive simulation with different modes:

```bash
python main.py                                # Run in hybrid mode (default)
python main.py --mode manual                  # Run with manual agent control
python main.py --mode ai                      # Run with AI-controlled agent
python main.py --agent-type rule_based        # Use rule-based agent (default)
python main.py --agent-type model_based       # Use neural network agent
python main.py --mode ai --agent-type model_based  # Run neural network agent in AI mode
python main.py --death                        # Run with near-death AI scenario
python main.py --cell-size 30                 # Change the display cell size
```

To compare both agent types side by side:

```bash
python compare_agents.py                      # Run both agents side by side
python compare_agents.py --death              # Compare agents in death scenario
python compare_agents.py --cell-size 20       # Adjust display size for comparison
```

To train an agent using reinforcement learning:

```bash
python train.py                               # Train using default PPO settings
python -c "from train import train_reinforce; from environment import GridWorld; \
           model, _ = train_reinforce(GridWorld(30, 30), num_episodes=200)"  # Train with REINFORCE
```

## Visualization Controls

The main.py script supports the following controls based on the mode:

### Manual Mode Controls
- **Arrow Keys**: Move the agent in different directions
- **E**: Eat (if on a mature plant)
- **D**: Drink (if adjacent to water)
- **P**: Plant seed (if on fertile soil)
- **T**: Tend plant (if on a seed or growing plant)
- **H**: Harvest (if on a mature plant)
- **Space**: Pause/Resume simulation
- **ESC**: Exit the visualization

### AI Mode Controls
- **Space**: Pause/Resume simulation
- **Tab**: Step simulation manually when paused
- **ESC**: Exit the visualization

### Hybrid Mode Controls
- All of the controls from both Manual and AI modes
- **A**: Toggle AI control (on/off)

## Requirements

See `requirements.txt` for dependencies:
- torch (>=2.0.0)
- numpy (>=1.21.0)
- pygame (>=2.1.0)
- matplotlib (>=3.5.0)
- gymnasium (>=0.28.0)

## Development Plan

1. Basic Grid Environment ✓
2. Resources and Growth Mechanics ✓
3. Agent Implementation ✓
4. Enhanced Visualization ✓
5. Farming Actions ✓
6. Rule-Based Agent Logic ✓
7. Model-Based Agent Implementation ✓
8. Preliminary RL Training ✓
9. Complete RL Implementation ✓
10. Advanced Environment Dynamics (next)
11. Multi-Agent Integration 