# Phase 10: Training and Evaluation Guide

This guide explains the training and evaluation system for the Reinforcement Learning Farming Simulation. This represents Phase 10 of our development plan, focused on training models, evaluating their performance, and visualizing the results.

## Components Overview

We have implemented several scripts to handle different aspects of training and evaluation:

1. **train.py**: Core training algorithms (PPO and REINFORCE) that have been extended with evaluation metrics tracking.
2. **evaluate.py**: Dedicated evaluation functions to thoroughly assess agent performance.
3. **run_training_and_evaluation.py**: End-to-end pipeline for training and evaluating agents.
4. **visualize_evaluation.py**: Graphical visualization of agent performance during evaluation.
5. **test_evaluate.py**: Simple testing script for basic evaluation.

## Training Models

To train a model, use the `run_training_and_evaluation.py` script. This script provides a complete pipeline for training and initial evaluation.

### Example Commands:

```bash
# Train a PPO agent for 2000 episodes (default)
python run_training_and_evaluation.py --algorithm ppo

# Train a REINFORCE agent with 1000 episodes
python run_training_and_evaluation.py --algorithm reinforce --num-episodes 1000

# Train with custom parameters
python run_training_and_evaluation.py --algorithm ppo --num-episodes 500 --learning-rate 0.0001 --gamma 0.98

# Train and compare with rule-based agent
python run_training_and_evaluation.py --algorithm ppo --compare
```

### Key Training Parameters:

- `--algorithm`: Choose between 'ppo' (default) or 'reinforce'
- `--num-episodes`: Number of training episodes (default: 2000)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--learning-rate`: Learning rate for optimization (default: 0.0003)
- `--gamma`: Discount factor for future rewards (default: 0.99)
- `--compare`: If set, compares trained agent against rule-based agent after training
- `--emergence`: If set, uses simplified rewards for emergent behavior
- `--log-level`: Set logging verbosity ('debug', 'info', 'warning', 'error', 'critical')

### PPO-Specific Parameters:

- `--update-timestep`: Number of timesteps between PPO updates (default: 2000) 
- `--ppo-epochs`: Number of PPO optimization epochs (default: 10)
- `--ppo-epsilon`: PPO clipping parameter (default: 0.2)
- `--gae-lambda`: GAE lambda parameter (default: 0.95)
- `--entropy-coef`: Entropy coefficient (default: 0.05)
- `--value-coef`: Value loss coefficient (default: 0.5)
- `--max-grad-norm`: Maximum norm for gradient clipping (default: 0.5)
- `--batch-size`: Mini batch size for PPO updates (default: 64)

## Evaluating Models

To evaluate a previously trained model, use the `evaluate.py` script.

### Example Commands:

```bash
# Evaluate a PPO model
python evaluate.py --agent-type ppo --model-path models/ppo_trained_agent.pth

# Evaluate with custom parameters
python evaluate.py --agent-type ppo --num-episodes 100 --max-steps 500

# Compare different agent types
python evaluate.py --mode compare

# Visualize agent behavior through main.py
python evaluate.py --mode visualize --agent-type ppo
```

### Key Evaluation Parameters:

- `--mode`: Evaluation mode ('evaluate', 'compare', 'visualize')
- `--agent-type`: Type of agent to evaluate ('rule_based', 'ppo', 'reinforce')
- `--model-path`: Path to the trained model file
- `--num-episodes`: Number of episodes to evaluate
- `--max-steps`: Maximum steps per episode
- `--log-level`: Set logging verbosity

## Visualizing Evaluation

For a graphical visualization of agent performance during evaluation, use the `visualize_evaluation.py` script.

### Example Commands:

```bash
# Visualize PPO agent
python visualize_evaluation.py --agent-type ppo

# Visualize with custom parameters 
python visualize_evaluation.py --agent-type rule_based --width 30 --height 20 --num-episodes 3
```

### Key Visualization Parameters:

- `--agent-type`: Type of agent to evaluate ('rule_based', 'ppo', 'reinforce')
- `--model-path`: Path to the trained model file
- `--width`: Width of the environment grid (default: 30)
- `--height`: Height of the environment grid (default: 20)
- `--cell-size`: Size of grid cells in pixels (default: 25)
- `--num-episodes`: Number of episodes to evaluate (default: 5)
- `--max-steps`: Maximum steps per episode (default: 500)
- `--delay`: Delay between steps in milliseconds (default: 50)

## Evaluation Metrics

The evaluation system tracks several key metrics:

1. **Survival Time**: How long the agent survives in an episode
2. **Survival Rate**: Percentage of episodes where the agent survives the full duration
3. **Seeds Planted**: Number of seeds planted per episode
4. **Plants Tended**: Number of plants tended per episode
5. **Plants Harvested**: Number of plants harvested per episode
6. **Food Eaten**: Number of food items consumed per episode
7. **Water Drunk**: Number of times the agent drank water
8. **Causes of Death**: Analysis of what caused agent death (starvation, dehydration, exhaustion)

## Comparing Agents

When comparing different agent types (rule-based vs. RL agents), the system evaluates all agents in identical environments to ensure a fair comparison. Comparison results are visualized with bar charts showing differences in key metrics.

## Conclusion

This training and evaluation system allows for:

1. Training RL agents with different algorithms and hyperparameters
2. Thorough evaluation of agent performance across multiple metrics
3. Visual comparison of different agent types
4. Graphical visualization of agent behavior during evaluation

The system provides insights into agent effectiveness and can guide further improvements to the RL algorithms and reward structures.

## Future Work

Potential enhancements for this system include:

1. Hyperparameter optimization to find optimal training settings
2. More sophisticated metrics for evaluating agent behavior
3. Enhanced visualization of agent decision-making processes
4. Integration with TensorBoard for more extensive training monitoring
5. Parallelization of training and evaluation for faster experimentation 