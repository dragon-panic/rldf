# Visualization Guide for RL Digital Farming

This document explains the visualization charts and graphs produced during training and evaluation of the reinforcement learning agents in the digital farming environment.

## Overview

The project generates several PNG files that visualize different aspects of agent performance:

1. **Training Progress**
   - `models/training_progress.png` - REINFORCE algorithm training metrics
   - `models/ppo_training_progress.png` - PPO algorithm training metrics

2. **Evaluation Comparisons**
   - `models/agent_comparison.png` - Comparative performance of different agent types
   - `models/rule_based_agent_performance.png` - Rule-based agent behavior metrics
   - `models/improved_farming_results.png` - Detailed farming behavior analysis

## Training Visualization Files

### `ppo_training_progress.png`

This visualization is generated during PPO (Proximal Policy Optimization) training and displays:

- **Left plot**: Episode rewards over time
  - Blue line: Raw episode rewards
  - Red line: Moving average rewards (smoothed over 100 episodes)
  - Upward trend indicates the agent is learning effectively

- **Right plot**: Episode lengths over time
  - Longer episodes generally indicate the agent is surviving longer
  - Plateaus may indicate the agent has reached maximum survival capability

### `training_progress.png`

This visualization is generated during REINFORCE algorithm training and contains similar plots to the PPO visualization:

- **Top plot**: Episode rewards with both raw and moving average values
- **Bottom plot**: Episode lengths showing how long the agent survived each episode

## Evaluation Visualization Files

### `agent_comparison.png`

This bar chart compares the performance of different agent types (Rule-Based, PPO, REINFORCE) across multiple metrics:

- **Survival Times**: Average number of steps the agent survived
- **Plants Harvested**: Average number of plants successfully harvested
- **Seeds Planted**: Average number of seeds planted
- **Food Eaten**: Average amount of food consumed

The chart includes error bars showing the standard deviation, giving insight into consistency across evaluation episodes.

### `rule_based_agent_performance.png`

This visualization shows the performance of the rule-based agent across multiple trials:

- **Health**: Agent's health level over time (green)
- **Hunger**: Agent's hunger level over time (red)
- **Thirst**: Agent's thirst level over time (blue)
- **Energy**: Agent's energy level over time (yellow)
- **Seeds**: Number of seeds the agent possesses over time (magenta)

The solid line represents the average across trials, while the shaded area shows the min-max range.

### `improved_farming_results.png`

This three-panel visualization focuses specifically on farming capabilities:

- **Top panel**: Agent status metrics (health, hunger, thirst, energy)
- **Middle panel**: Seed count over time
- **Bottom panel**: Stacked area chart showing plant growth stages (seeds, growing, mature)

## How to Use These Visualizations

### For Model Training Assessment

1. **Check Learning Progress**:
   - Open `models/ppo_training_progress.png` or `models/training_progress.png`
   - Look for increasing trends in the rewards plot
   - A flat or decreasing rewards curve suggests the model is not learning effectively

2. **Assess Training Stability**:
   - Examine the smoothness of the moving average line
   - Wide fluctuations indicate unstable learning
   - Smooth upward trends suggest stable improvement

3. **Identify Training Completion**:
   - Training can be considered complete when the rewards plateau
   - If the moving average flattens for an extended period, additional training may not yield improvements

### For Model Comparison

1. **Compare Agent Types**:
   - Open `models/agent_comparison.png`
   - Identify which agent type performs best on each metric
   - Consider trade-offs (e.g., one agent might harvest more but survive less time)

2. **Analyze Behavioral Differences**:
   - Compare `rule_based_agent_performance.png` with training results
   - Look for differences in how RL agents manage resources versus rule-based approaches

### For Detailed Behavior Analysis

1. **Farming Capability Assessment**:
   - Open `models/improved_farming_results.png`
   - Check if seed count increases over time (middle panel)
   - Analyze if the agent maintains a healthy balance of plants at different growth stages (bottom panel)

2. **Survival Strategy Analysis**:
   - Study how vitals (health, hunger, thirst) change over time
   - Effective agents should maintain relatively stable vital statistics 

## Generating New Visualizations

These visualizations are automatically generated during:

1. **Training**: When running `train.py` or the training functions from `run_training_and_evaluation.py`
2. **Evaluation**: When running evaluation scripts or comparison tests
3. **Testing**: Some visualizations are generated during specific tests like `test_improved_farming.py`

To generate fresh visualizations:

```bash
# For training visualizations
python run_training_and_evaluation.py --algorithm ppo --num-episodes 1000

# For agent comparison
python evaluate.py --mode compare --num-episodes 50

# For farming-specific analysis
python test_improved_farming.py
```

## Interpreting Key Indicators

1. **Seed Multiplication Factor**:
   - A value >1.0 indicates the agent can sustainably farm (creating more seeds than it started with)
   - Higher values suggest more efficient farming strategies

2. **Survival Rate**:
   - The percentage of evaluation episodes where the agent survived the full duration
   - Look for this information in the text output of evaluation scripts

3. **Plant Growth Balance**:
   - In the stacked area chart of `improved_farming_results.png`
   - Effective agents maintain a balance between growth stages
   - Too many seeds suggests poor tending
   - Too few mature plants suggests poor harvesting timing 