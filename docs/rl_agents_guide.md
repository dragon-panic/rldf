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

### Train a REINFORCE Agent

```bash
python train.py --episodes 5000 --agent-type reinforce --grid-size 50
```

### Train a PPO Agent

```bash
python train_ppo.py --episodes 3000 --grid-size 50
```

### Evaluate an Agent

```bash
python evaluate.py --agent-type reinforce --model-path models/reinforce_model.pt --episodes 100
```

### Compare Agents

```bash
python rl_framework.py --eval-episodes 50 --grid-size 50
```

## Command Line Options

### For training (train.py):

```
--episodes N            Number of episodes for training (default: 1000)
--max-steps N           Maximum steps per episode (default: 1000)
--grid-size N           Size of the grid world (default: 30)
--water-prob P          Probability of water cells (default: 0.1)
--learning-rate LR      Learning rate (default: 0.001)
--gamma G               Discount factor (default: 0.99)
--agent-type TYPE       Type of agent (default: reinforce)
--model-path PATH       Path to save the model (default: models/reinforce_model.pt)
--log-level LEVEL       Logging level (default: info)
--render                Render the environment during training
--cell-size N           Cell size for rendering (default: 10)
--emergence             Use simplified rewards for emergent behavior
```

### For PPO training (train_ppo.py):

```
--episodes N            Number of episodes for training (default: 1000)
--max-steps N           Maximum steps per episode (default: 1000)
--grid-size N           Size of the grid world (default: 30)
--water-prob P          Probability of water cells (default: 0.1)
--learning-rate LR      Learning rate (default: 0.0003)
--gamma G               Discount factor (default: 0.99)
--update-timestep N     Timesteps between PPO updates (default: 2000)
--ppo-epochs N          Number of PPO epochs (default: 4)
--batch-size N          Mini-batch size (default: 64)
--model-path PATH       Path to save the model (default: models/ppo_model.pt)
--render                Render the environment during training
--cell-size N           Cell size for rendering (default: 10)
--emergence             Use simplified rewards for emergent behavior
```

### For evaluation (evaluate.py):

```
--agent-type TYPE       Type of agent (rule_based, reinforce, ppo) (default: rule_based)
--model-path PATH       Path to the model (default: models/reinforce_model.pt)
--episodes N            Number of episodes for evaluation (default: 10)
--grid-size N           Size of the grid world (default: 30)
--water-prob P          Probability of water cells (default: 0.1)
--render                Render the environment during evaluation
--cell-size N           Cell size for rendering (default: 20)
--max-steps N           Maximum steps per episode (default: 1000)
```

### For comparison (rl_framework.py):

```
--eval-episodes N        Number of episodes for evaluation (default: 50)
--grid-size N            Size of the grid world (default: 50)
--water-prob P           Probability of water cells (default: 0.1)
--reinforce-model PATH   Path to REINFORCE model (default: models/reinforce_trained_agent.pth)
--ppo-model PATH         Path to PPO model (default: models/ppo_trained_agent.pth)
--visualize              Visualize agent behavior
--log-level LEVEL        Logging level (default: info)
--max-steps N            Maximum steps per episode (default: 1000)
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
- `utils.py`: Utility functions
- `visualization.py`: Environment rendering utilities

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

Modify the model architecture in `rl_agent.py` or `ppo_agent.py`:

- Change layer sizes, types, or activation functions
- Adjust input/output processing
- Add recurrent layers for memory 