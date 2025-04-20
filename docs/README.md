# Reinforcement Learning Digital Farming Documentation

Welcome to the documentation for the Reinforcement Learning Digital Farming (RLDF) project. This documentation provides guides on understanding the project's training, evaluation, and visualization components.

## Documentation Index

- [Training and Evaluation Guide](training_evaluation_guide.md) - Detailed instructions for training and evaluating models
- [Visualization Guide](visualization_guide.md) - Explains all visualization outputs and how to interpret them
- [Model Comparison Guide](model_comparison_guide.md) - Detailed guide for comparing PPO vs REINFORCE models

## Project Overview

The RLDF project implements reinforcement learning agents that learn to survive and farm in a simulated environment. The project includes:

1. **Environment**: A grid-based world with water, soil, and plants
2. **Agents**: 
   - Rule-based agent with hard-coded behaviors
   - Reinforcement Learning agents (PPO and REINFORCE)
3. **Training**: Scripts to train agents using different algorithms
4. **Evaluation**: Tools to evaluate and compare agent performance
5. **Visualization**: Graphs and charts to analyze training and behavior

## Getting Started

To begin using the visualization tools:

1. Run training to generate model training charts:
   ```
   python run_training_and_evaluation.py --algorithm ppo --num-episodes 1000
   ```

2. Evaluate and compare models:
   ```
   python evaluate.py --mode compare
   ```

3. Test farming capabilities:
   ```
   python test_improved_farming.py
   ```

4. View the generated visualizations in the `models/` directory

## Notes on Using the Visualizations

- All visualization PNG files are saved in the `models/` folder
- Compare different training runs by renaming output files before starting new training
- For more customized visualizations, modify the plotting functions in the respective scripts 