# Unified RL Agent Framework

This project provides a modular framework for training, evaluating, and comparing different agent types in a grid-based farming environment:

1. **Rule-Based Agent**: Follows hard-coded rules for decision making
2. **REINFORCE Agent**: Uses policy gradient RL algorithm
3. **PPO Agent**: Uses Proximal Policy Optimization, a more advanced RL algorithm

## Features

The framework unifies several key capabilities:

- **Training**: Train REINFORCE and PPO agents with customizable parameters
- **Evaluation**: Measure agent performance across multiple metrics
- **Comparison**: Compare rule-based and RL agents side-by-side
- **Visualization**: View agent behavior in the environment

## Getting Started

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

The main framework script `rl_framework.py` provides a unified interface:

```bash
# Train both REINFORCE and PPO agents, then compare with rule-based
python rl_framework.py --train

# Train only PPO and compare with rule-based
python rl_framework.py --train --reinforce-episodes 0

# Evaluate pre-trained models against rule-based
python rl_framework.py

# Visualize agent behavior
python rl_framework.py --visualize
```

## Command Line Options

### Mode Selection
```
--train         Train new agents
--evaluate      Only evaluate single agents without comparison (faster)
--visualize     Visualize agent behavior
```

### Environment Parameters
```
--grid-size N   Size of the grid world (default: 50)
--water-prob P  Probability of water cells (default: 0.1)
```

### Training Parameters
```
--reinforce-episodes N   Number of episodes for REINFORCE training (default: 1000)
--ppo-episodes N         Number of episodes for PPO training (default: 2000)
--max-steps N            Maximum steps per episode (default: 1000)
--gamma G                Discount factor (default: 0.99)
--learning-rate LR       Learning rate (default: 0.0003)
--emergence              Use simplified rewards for emergent behavior
```

### PPO-Specific Parameters
```
--update-timestep N      Timesteps between PPO updates (default: 2000)
--ppo-epochs N           Number of PPO epochs (default: 10)
--ppo-epsilon E          PPO clipping parameter (default: 0.2)
--gae-lambda L           GAE lambda parameter (default: 0.95)
--entropy-coef C         Entropy coefficient (default: 0.05)
--value-coef C           Value loss coefficient (default: 0.5)
--max-grad-norm G        Max gradient norm (default: 0.5)
--batch-size N           Mini-batch size (default: 64)
```

### Evaluation and Model Parameters
```
--eval-episodes N        Number of episodes for evaluation (default: 50)
--reinforce-model PATH   Path to REINFORCE model
--ppo-model PATH         Path to PPO model
--log-level LEVEL        Logging level (default: info)
```

## Examples

### Train for More Episodes

```bash
python rl_framework.py --train --reinforce-episodes 2000 --ppo-episodes 4000
```

### Custom Environment

```bash
python rl_framework.py --train --grid-size 100 --water-prob 0.2
```

### Fine-tune PPO Training

```bash
python rl_framework.py --train --ppo-episodes 1000 --update-timestep 1000 --ppo-epochs 15 --entropy-coef 0.1
```

### Compare on a Small Grid with Visualization

```bash
python rl_framework.py --grid-size 30 --eval-episodes 20 --visualize
```

## Evaluation Metrics

The framework compares agents on multiple metrics:

- **Reward**: Theoretical accumulated reward
- **Episode Length**: How many steps the agent survived
- **Plants Harvested**: Number of mature plants harvested
- **Seeds Planted**: Number of seeds planted
- **Survival Time**: How long the agent survived
- **Food Eaten**: Number of times the agent ate
- **Water Drunk**: Number of times the agent drank

A detailed summary shows which agent performs best in each category.

## Architecture Overview

The framework uses a modular design:

1. **Environment** (`GridWorld`): A 2D grid with water, soil, and plants
2. **Agents**:
   - `RuleBasedAgent`: Uses predefined rules to make decisions
   - `RLAgent`: Uses neural networks to make decisions
3. **Models**:
   - `AgentCNN`: Basic CNN for REINFORCE
   - `PPOAgentCNN`: Dual-head CNN for PPO
4. **Training Algorithms**:
   - REINFORCE: Simple policy gradient
   - PPO: Advanced policy optimization

## Customizing Agent Behavior

- Rule-based agent: Edit thresholds in `rule_based_agent.py`
- RL agents: Adjust reward functions in `train.py`
- Environment: Modify parameters in `environment.py`

## Agent Visualization

To visualize agent behavior in the environment:

```bash
python rl_framework.py --visualize
```

You can specify which agents to visualize by using the appropriate model paths:

```bash
python rl_framework.py --visualize --reinforce-model models/my_reinforce_model.pth --ppo-model models/my_ppo_model.pth
```

## Extending the Framework

To add new agent types:
1. Create a new agent class in a separate file
2. Add the agent type to the `create_agent` function in `evaluate.py`
3. Update the `RLFramework` class to handle the new agent type 