# RL Farming Simulation

A reinforcement learning project for simulating farming agents in a resource-based environment, inspired by Dwarf Fortress-style simulation games. The project uses the Proximal Policy Optimization (PPO) algorithm to train agents that can learn complex survival and farming behaviors.

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

## Phase 10: Training and Evaluation ✓

- Created comprehensive evaluation framework:
  - Performance metrics for agent behaviors
  - Survival statistics and farming productivity tracking
  - Comparative analysis between agent types
- Implemented visualization tools for performance monitoring:
  - Training progress visualization
  - Agent behavior visualization during evaluation
  - Comparative metrics visualization
- Added complete evaluation pipeline:
  - End-to-end training and evaluation workflow
  - Batch evaluation across multiple episodes and scenarios
  - Detailed performance reports and visualizations
- Created comprehensive documentation in the `docs/` folder

## Training Approaches

The project supports two different approaches to training agents:

1. **Structured Rewards (Default)**: This approach uses hand-crafted rewards for specific actions (planting, harvesting, eating, etc.) and carefully balances them to produce effective behavior.

2. **Emergence Mode**: This approach uses a simplified reward function that primarily rewards survival and state improvement, without specifying which actions the agent should take. The agent must discover effective strategies on its own. Enable with `--emergence` flag.

The emergence approach tends to need longer training time but may discover novel strategies and behaviors that weren't explicitly programmed. It's particularly interesting for observing how complex behaviors like farming cycles can emerge from simple survival-focused rewards.

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
- `evaluate.py`: Comprehensive agent evaluation framework
- `run_training_and_evaluation.py`: End-to-end training and evaluation pipeline
- `visualize_evaluation.py`: Graphical visualization of agent performance
- `docs/`: Comprehensive documentation for using the system

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
# Run in AI mode with neural network agent (default)
python main.py

# Specify a custom model file to use
python main.py --model-path my_custom_model.pth

# Run with manual agent control
python main.py --mode manual

# Run in hybrid mode (manual + AI)
python main.py --mode hybrid

# Use rule-based agent
python main.py --agent-type rule_based

# Run hybrid mode with rule-based agent
python main.py --mode hybrid --agent-type rule_based

# Run with near-death AI scenario
python main.py --death

# Change the display cell size
python main.py --cell-size 30
```

To compare both agent types side by side:

```bash
python compare_agents.py                      # Run both agents side by side
python compare_agents.py --death              # Compare agents in death scenario
python compare_agents.py --cell-size 20       # Adjust display size for comparison
```

To train an agent using reinforcement learning:

```bash
# Complete training and evaluation pipeline
python run_training_and_evaluation.py --algorithm ppo

# Train with PPO and compare with rule-based agent
python run_training_and_evaluation.py --algorithm ppo --compare

# Train with REINFORCE algorithm
python run_training_and_evaluation.py --algorithm reinforce

# Customize training parameters
python run_training_and_evaluation.py --num-episodes 1000 --max-steps 800 --learning-rate 0.0002

# Use emergence mode with simplified rewards
python run_training_and_evaluation.py --emergence

# Get help on all available parameters
python run_training_and_evaluation.py --help
```

To evaluate agent performance:

```bash
# Evaluate a trained model
python evaluate.py --agent-type ppo

# Compare different agent types
python evaluate.py --mode compare

# Visualize agent behavior
python evaluate.py --mode visualize

# Detailed visualization of agent performance
python visualize_evaluation.py --agent-type ppo --num-episodes 3
```

## Visualization Controls

The main.py script supports the following controls based on the mode:

### AI Mode Controls (Default)
- **Space**: Pause/Resume simulation
- **Tab**: Step simulation manually when paused
- **ESC**: Exit the visualization

### Manual Mode Controls
- **Arrow Keys**: Move the agent in different directions
- **E**: Eat (if on a mature plant)
- **D**: Drink (if adjacent to water)
- **P**: Plant seed (if on fertile soil)
- **T**: Tend plant (if on a seed or growing plant)
- **H**: Harvest (if on a mature plant)
- **Space**: Pause/Resume simulation
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
10. Training and Evaluation ✓
11. Advanced Environment Dynamics (next)
12. Multi-Agent Integration 