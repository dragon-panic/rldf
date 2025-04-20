# RL Agents for Farm Environment

This project implements reinforcement learning agents for a simple grid-based farming environment. Agents need to learn how to survive by managing resources like hunger, thirst, and energy while collecting resources and growing plants.

## Types of Agents

The project implements the following agent types:

1. **Rule-Based Agent**: A baseline agent that uses hardcoded rules for decision-making
2. **REINFORCE Agent**: A basic policy gradient RL algorithm
3. **PPO Agent**: Proximal Policy Optimization agent, a more advanced RL algorithm

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Quick Start

### Train and Compare All Agents

```bash
python rl_framework.py --train --visualize
```

This command will:
- Train a REINFORCE agent (1000 episodes by default)
- Train a PPO agent (2000 episodes by default)
- Compare both trained agents with the rule-based agent
- Visualize the behavior of each agent

### Customize Training Parameters

```bash
python rl_framework.py --train --reinforce-episodes 5000 --ppo-episodes 3000 --grid-size 50
```

### Compare Using Existing Models

```bash
python rl_framework.py --reinforce-model models/reinforce_trained_agent.pth --ppo-model models/ppo_trained_agent.pth
```

### Only Visualize Agents

```bash
python rl_framework.py --visualize
```

## Command Line Options

### For rl_framework.py:

```
# Mode selection
--train                Train new agents
--evaluate             Only evaluate without comparison (faster)
--visualize            Visualize agent behavior

# Environment parameters
--grid-size N          Size of the grid world (default: 50)
--water-prob P         Probability of water cells (default: 0.1)

# General training parameters
--max-steps N          Maximum steps per episode (default: 1000)
--gamma G              Discount factor (default: 0.99)
--learning-rate LR     Learning rate (default: 0.0003)
--emergence            Use simplified rewards for emergent behavior

# Algorithm-specific training parameters
--reinforce-episodes N Number of episodes for REINFORCE training (default: 1000)
--ppo-episodes N       Number of episodes for PPO training (default: 2000)

# PPO-specific parameters
--update-timestep N    Number of timesteps between PPO updates (default: 2000)
--ppo-epochs N         Number of PPO epochs (default: 10)
--ppo-epsilon N        PPO clipping parameter (default: 0.2)
--gae-lambda N         GAE lambda parameter (default: 0.95)
--entropy-coef N       Entropy coefficient for PPO (default: 0.05)
--value-coef N         Value loss coefficient for PPO (default: 0.5)
--max-grad-norm N      Maximum norm for gradient clipping (default: 0.5)
--batch-size N         Mini-batch size for PPO updates (default: 64)

# Evaluation parameters
--eval-episodes N      Number of episodes for evaluation (default: 50)

# Model paths
--reinforce-model PATH Path to REINFORCE model (default: models/reinforce_trained_agent.pth)
--ppo-model PATH       Path to PPO model (default: models/ppo_trained_agent.pth)

# Logging
--log-level LEVEL      Logging level (default: info)
```

## Expected Results

After training the agents for sufficient episodes:

- **Rule-Based Agent**: Should achieve modest survival times by following simple rules
- **REINFORCE Agent**: Should learn to survive longer by finding water and planting/harvesting food
- **PPO Agent**: Should achieve the best performance, efficiently managing resources

## Model Architecture

### REINFORCE Agent

Uses a simple CNN architecture:
- Input: Grid state representation (channels for different cell types)
- 3 convolutional layers with ReLU activations
- Flatten layer + 2 fully connected layers
- Output: Action probabilities (softmax)

### PPO Agent

Uses a dual-head CNN architecture:
- Shared feature extractor with 3 convolutional layers
- Policy head: 2 fully connected layers outputting action probabilities
- Value head: 2 fully connected layers outputting state value estimate

## File Descriptions

- `environment.py`: Implements the grid-based farming environment
- `agent.py`: Base agent class and rule-based agent implementation
- `rule_based_agent.py`: Implementation of the rule-based agent
- `model.py`: Neural network models for RL agents
- `train.py`: Implementation of training algorithms and utility functions
- `evaluate.py`: Functions for evaluating and comparing agents
- `rl_framework.py`: Unified framework for training, evaluation, and comparison
- `visualize.py`: Environment rendering and visualization tools
- `visualize_evaluation.py`: Additional visualization tools for evaluation results
- `simple_visualize.py`: Simplified visualization utilities

## Customization

### Reward Engineering

Modify the reward functions in `environment.py` to encourage different behaviors:

- Default rewards are defined in the `step()` method
- For emergent behavior, use the `--emergence` flag which uses simpler rewards

### Environment Customization

Adjust environment parameters in `environment.py` or via command line:

- Grid size (affects difficulty and exploration)
- Water probability (affects resource availability)
- Maximum steps (affects episode length)

### Neural Network Architecture

Modify the model architecture in `model.py`:

- Change layer sizes, types, or activation functions
- Adjust input/output processing
- Add recurrent layers for memory 