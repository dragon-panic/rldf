# Policy Gradient Methods Tutorial

This tutorial provides an introduction to Policy Gradient methods for reinforcement learning, with a focus on the REINFORCE and PPO algorithms implemented in this project.

## Introduction to Policy Gradient Methods

Policy gradient methods are a class of reinforcement learning algorithms that optimize the policy directly by following the gradient of expected return with respect to the policy parameters. Unlike value-based methods (like Q-learning), policy gradient methods:

- Learn a policy function directly, without needing a value function
- Can learn stochastic policies, which may be beneficial in certain environments
- Can be applied to continuous action spaces naturally
- Often have better convergence properties than value-based methods

## The Policy Gradient Theorem

The core of policy gradient methods is the Policy Gradient Theorem, which gives us the gradient of the expected return with respect to policy parameters θ:

∇θJ(θ) = E[∇θ log π(a|s;θ) · Q(s,a)]

Where:
- J(θ) is the expected return
- π(a|s;θ) is the policy (probability of taking action a in state s)
- Q(s,a) is the action-value function

## REINFORCE Algorithm

REINFORCE is the simplest policy gradient algorithm. The key steps are:

1. **Sample trajectories** from the current policy
2. **Calculate returns** for each timestep in the trajectories
3. **Update the policy** by maximizing the log probability of actions taken, weighted by the returns

### Algorithm (Pseudo-code)

```
Initialize policy parameters θ
for each episode do
    Generate episode S0, A0, R1, S1, A1, ..., ST-1, AT-1, RT using policy π(a|s;θ)
    for t = 0 to T-1 do
        G ← return from step t
        θ ← θ + α ∇θ log π(At|St;θ) · G
    end for
end for
```

### Implementation Highlights

Our REINFORCE implementation in `rl_agent.py` uses:

- A CNN to process the grid state and output action probabilities
- PyTorch's automatic differentiation for computing gradients
- A baseline (critic) to reduce variance in updates

Key code snippet for the update:

```python
log_probs = torch.log(saved_actions)
policy_loss = -(log_probs * rewards).sum()
optimizer.zero_grad()
policy_loss.backward()
optimizer.step()
```

## PPO (Proximal Policy Optimization)

PPO improves upon REINFORCE by:

1. Using multiple epochs of minibatch updates
2. Clipping the policy update to prevent too large changes
3. Adding a value function term to the loss function

### PPO Objective Function

PPO uses a "surrogate" objective function:

L(θ) = E[min(rt(θ)At, clip(rt(θ), 1-ε, 1+ε)At)]

Where:
- rt(θ) is the ratio of probabilities π(a|s;θ) / π(a|s;θold)
- At is the advantage estimate
- ε is a hyperparameter (typically 0.1 or 0.2)

### Algorithm (Pseudo-code)

```
Initialize policy parameters θ and value function parameters φ
for each iteration do
    Collect trajectory data using current policy π(a|s;θ)
    Compute advantages At
    for K epochs do
        for each minibatch do
            Update θ by maximizing PPO objective
            Update φ to minimize value function error
        end for
    end for
end for
```

### Implementation Highlights

Our PPO implementation in `ppo_agent.py` includes:

- A CNN architecture with shared layers and separate policy and value heads
- Memory buffer for storing trajectories
- Clipped surrogate objective for stable updates
- Value function loss and entropy bonus

Key code snippet for the PPO update:

```python
# Compute ratio
ratio = (new_probs / old_probs)
            
# Compute surrogate losses
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
# Final losses
actor_loss = -torch.min(surr1, surr2).mean()
critic_loss = 0.5 * (returns - values).pow(2).mean()
entropy_loss = -self.entropy_coef * entropies.mean()
            
loss = actor_loss + critic_loss - entropy_loss
```

## Practical Tips for Training Policy Gradient Agents

1. **Reward Scaling**: Normalize rewards to stabilize training
2. **Learning Rate Scheduling**: Start with a higher learning rate and decrease it over time
3. **Entropy Regularization**: Add an entropy bonus to encourage exploration
4. **Hyperparameter Tuning**: Key parameters to tune include:
   - Learning rate
   - Discount factor (gamma)
   - Number of epochs (for PPO)
   - Clipping parameter (for PPO)
   - Entropy coefficient

## Comparing REINFORCE and PPO

| Aspect | REINFORCE | PPO |
|--------|-----------|-----|
| Stability | Less stable | More stable due to clipping |
| Sample Efficiency | Less efficient | More efficient |
| Implementation Complexity | Simpler | More complex |
| Performance | Good | Better |
| Hyperparameter Sensitivity | More sensitive | Less sensitive |

## Further Reading

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement learning: An introduction. MIT press.
2. Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning.
3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
4. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438. 