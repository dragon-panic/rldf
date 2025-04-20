import numpy as np
import matplotlib.pyplot as plt
from environment import GridWorld
from rule_based_agent import RuleBasedAgent

def run_survival_test(steps=300, water_probability=0.1, num_trials=5):
    """
    Run a survival test for the rule-based agent.
    
    Args:
        steps: Number of steps to run the simulation
        water_probability: Probability of water in the environment
        num_trials: Number of trial runs
    
    Returns:
        dict: Results of the test
    """
    # Results storage
    results = {
        'health': np.zeros((num_trials, steps)),
        'hunger': np.zeros((num_trials, steps)),
        'thirst': np.zeros((num_trials, steps)),
        'energy': np.zeros((num_trials, steps)),
        'seeds': np.zeros((num_trials, steps)),
        'survival_time': np.zeros(num_trials),
        'actions': [{} for _ in range(num_trials)]
    }
    
    # Run trials
    for trial in range(num_trials):
        print(f"Running trial {trial + 1} of {num_trials}...")
        
        # Create environment
        env = GridWorld(width=30, height=20, water_probability=water_probability)
        
        # Add some plants to make it interesting
        for _ in range(15):
            row = np.random.randint(0, env.height)
            col = np.random.randint(0, env.width)
            if env.grid[row, col] == GridWorld.SOIL:
                env.set_cell(row, col, GridWorld.PLANT)
                # Set some plants to be mature
                if np.random.random() < 0.5:
                    env.plant_state[row, col] = GridWorld.PLANT_MATURE
                else:
                    env.plant_state[row, col] = GridWorld.PLANT_GROWING
        
        # Create agent
        agent = RuleBasedAgent(env, start_row=env.height // 2, start_col=env.width // 2)
        
        # Set initial agent state
        agent.hunger = 50.0
        agent.thirst = 50.0
        agent.seeds = 3
        
        # Run simulation
        survived = True
        for step in range(steps):
            # Step the agent
            result = agent.step_ai()
            
            # Record status
            status = result['status']
            results['health'][trial, step] = status['health']
            results['hunger'][trial, step] = status['hunger']
            results['thirst'][trial, step] = status['thirst']
            results['energy'][trial, step] = status['energy']
            results['seeds'][trial, step] = status['seeds']
            
            # Record action history
            if agent.action_history and agent.action_history[-1][0] in results['actions'][trial]:
                results['actions'][trial][agent.action_history[-1][0]] += 1
            elif agent.action_history:
                results['actions'][trial][agent.action_history[-1][0]] = 1
            
            # Step the environment periodically
            if step % 5 == 0:
                env.step()
            
            # Check if agent died
            if not result['alive']:
                survived = False
                results['survival_time'][trial] = step
                break
        
        # If agent survived the whole time
        if survived:
            results['survival_time'][trial] = steps
    
    return results

def plot_results(results, steps):
    """Plot the results of the survival test."""
    num_trials = results['health'].shape[0]
    
    # Calculate averages across trials
    avg_health = np.mean(results['health'], axis=0)
    avg_hunger = np.mean(results['hunger'], axis=0)
    avg_thirst = np.mean(results['thirst'], axis=0)
    avg_energy = np.mean(results['energy'], axis=0)
    avg_seeds = np.mean(results['seeds'], axis=0)
    
    # Create the figure
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    
    # Plot health
    axes[0].plot(range(steps), avg_health, 'g-')
    axes[0].fill_between(range(steps), 
                         np.min(results['health'], axis=0),
                         np.max(results['health'], axis=0), 
                         alpha=0.2, color='g')
    axes[0].set_ylabel('Health')
    axes[0].set_title(f'Agent Performance (Average of {num_trials} Trials)')
    axes[0].grid(True)
    
    # Plot hunger
    axes[1].plot(range(steps), avg_hunger, 'r-')
    axes[1].fill_between(range(steps), 
                         np.min(results['hunger'], axis=0),
                         np.max(results['hunger'], axis=0), 
                         alpha=0.2, color='r')
    axes[1].set_ylabel('Hunger')
    axes[1].grid(True)
    
    # Plot thirst
    axes[2].plot(range(steps), avg_thirst, 'b-')
    axes[2].fill_between(range(steps), 
                         np.min(results['thirst'], axis=0),
                         np.max(results['thirst'], axis=0), 
                         alpha=0.2, color='b')
    axes[2].set_ylabel('Thirst')
    axes[2].grid(True)
    
    # Plot energy
    axes[3].plot(range(steps), avg_energy, 'y-')
    axes[3].fill_between(range(steps), 
                         np.min(results['energy'], axis=0),
                         np.max(results['energy'], axis=0), 
                         alpha=0.2, color='y')
    axes[3].set_ylabel('Energy')
    axes[3].grid(True)
    
    # Plot seeds
    axes[4].plot(range(steps), avg_seeds, 'm-')
    axes[4].fill_between(range(steps), 
                         np.min(results['seeds'], axis=0),
                         np.max(results['seeds'], axis=0), 
                         alpha=0.2, color='m')
    axes[4].set_ylabel('Seeds')
    axes[4].set_xlabel('Steps')
    axes[4].grid(True)
    
    plt.tight_layout()
    plt.savefig('rule_based_agent_performance.png')
    plt.show()
    
    # Print survival statistics
    avg_survival = np.mean(results['survival_time'])
    min_survival = np.min(results['survival_time'])
    max_survival = np.max(results['survival_time'])
    
    print(f"Survival Statistics:")
    print(f"  Average survival time: {avg_survival:.1f} steps")
    print(f"  Minimum survival time: {min_survival:.1f} steps")
    print(f"  Maximum survival time: {max_survival:.1f} steps")
    
    # Print action statistics
    print("\nAction Statistics (average per trial):")
    all_actions = set()
    for trial in range(num_trials):
        all_actions.update(results['actions'][trial].keys())
    
    for action in sorted(all_actions):
        counts = [results['actions'][trial].get(action, 0) for trial in range(num_trials)]
        avg_count = np.mean(counts)
        print(f"  {action}: {avg_count:.1f}")

def test_farming_capability(steps=500):
    """Test the agent's farming capabilities in a controlled environment."""
    # Create a specific environment for farming test
    env = GridWorld(width=15, height=15, water_probability=0.05)
    
    # Place water sources in specific locations
    for row in range(3, 6):
        for col in range(3, 6):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Make soil around water very fertile
    for row in range(2, 7):
        for col in range(2, 7):
            if env.grid[row, col] == GridWorld.SOIL:
                env.soil_fertility[row, col] = 10.0
    
    # Add a few mature plants
    for row, col in [(8, 8), (8, 9), (9, 8)]:
        env.set_cell(row, col, GridWorld.PLANT)
        env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Create agent in a corner
    agent = RuleBasedAgent(env, start_row=12, start_col=12)
    
    # Set initial agent state
    agent.hunger = 70.0  # Start hungry to encourage foraging
    agent.thirst = 70.0  # Start thirsty to encourage seeking water
    agent.seeds = 1      # Start with just one seed
    
    # Run simulation
    print("Running farming capability test...")
    
    # Track farming metrics
    metrics = {
        'plants_harvested': 0,
        'seeds_planted': 0,
        'plants_tended': 0,
        'final_seed_count': 0
    }
    
    # Run for specified steps
    for step in range(steps):
        # Step the agent
        result = agent.step_ai()
        
        # Track farming actions
        if agent.action_history:
            last_action = agent.action_history[-1]
            if last_action[0] == 'harvest':
                metrics['plants_harvested'] += 1
            elif last_action[0] == 'plant_seed':
                metrics['seeds_planted'] += 1
            elif last_action[0] == 'tend_plant':
                metrics['plants_tended'] += 1
        
        # Step the environment periodically to allow plants to grow
        if step % 5 == 0:
            env.step()
        
        # Print status periodically
        if step % 100 == 0:
            status = result['status']
            print(f"Step {step}:")
            print(f"  Position: {status['position']}")
            print(f"  Task: {agent.current_task}")
            print(f"  Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
            print(f"  Hunger: {status['hunger']:.1f}, Thirst: {status['thirst']:.1f}")
            print(f"  Seeds: {status['seeds']}")
            print(f"  Plants harvested: {metrics['plants_harvested']}")
            print(f"  Seeds planted: {metrics['seeds_planted']}")
            print(f"  Plants tended: {metrics['plants_tended']}")
            print()
        
        # Break if agent dies
        if not result['alive']:
            print("Agent died!")
            break
    
    # Final metrics
    metrics['final_seed_count'] = agent.seeds
    
    # Print final results
    print("\nFarming Capability Test Results:")
    print(f"  Plants harvested: {metrics['plants_harvested']}")
    print(f"  Seeds planted: {metrics['seeds_planted']}")
    print(f"  Plants tended: {metrics['plants_tended']}")
    print(f"  Final seed count: {metrics['final_seed_count']}")
    print(f"  Seed multiplication factor: {metrics['final_seed_count'] / 1.0:.1f}x")
    
    # Evaluate success using assertions instead of return value
    assert metrics['plants_harvested'] > 0 or metrics['seeds_planted'] > 0 or metrics['final_seed_count'] > 1, "Agent should demonstrate some farming capabilities"
    
    if __name__ == "__main__":
        if metrics['plants_harvested'] > 0 and metrics['seeds_planted'] > 0 and metrics['final_seed_count'] > 1:
            print("\nTest Result: PASS - Agent demonstrated farming capabilities")
            return True
        else:
            print("\nTest Result: FAIL - Agent did not demonstrate sufficient farming capabilities")
            return False

def test_agent_death(steps=100):
    """Test the agent's death mechanism by putting it in a hostile environment."""
    # Create a specific environment for death test
    env = GridWorld(width=15, height=15, water_probability=0.05)
    
    # Create agent in a corner
    agent = RuleBasedAgent(env, start_row=12, start_col=12)
    
    # Set initial agent state to near-critical
    agent.hunger = 90.0   # Very hungry
    agent.thirst = 90.0   # Very thirsty
    agent.health = 10.0   # Low health
    agent.seeds = 1       # Just one seed (not that it will matter)
    
    # Run simulation
    print("Running agent death test...")
    
    # Track health progression
    health_history = []
    
    # Run for specified steps or until agent dies
    for step in range(steps):
        # Step the agent
        result = agent.step_ai()
        
        # Record health
        health_history.append(result['status']['health'])
        
        # Print status periodically
        if step % 10 == 0 or not result['alive']:
            status = result['status']
            print(f"Step {step}:")
            print(f"  Position: {status['position']}")
            print(f"  Task: {agent.current_task}")
            print(f"  Health: {status['health']:.1f}, Energy: {status['energy']:.1f}")
            print(f"  Hunger: {status['hunger']:.1f}, Thirst: {status['thirst']:.1f}")
            print(f"  Alive: {result['alive']}")
            if not result['alive']:
                print(f"  Cause of death: {result.get('cause_of_death', 'Unknown')}")
            print()
        
        # Step the environment periodically
        if step % 5 == 0:
            env.step()
        
        # Break if agent dies
        if not result['alive']:
            print("Agent died!")
            break
    
    # Print final results
    print("\nAgent Death Test Results:")
    print(f"  Initial health: 10.0")
    print(f"  Final health: {health_history[-1]:.1f}")
    print(f"  Survived for {len(health_history)} steps")
    print(f"  Agent alive at end: {agent.is_alive}")
    
    # Assert that health decreases over time
    assert health_history[0] > health_history[-1], "Agent health should decrease in a hostile environment"
    
    # For main function only
    if __name__ == "__main__":
        return health_history

if __name__ == "__main__":
    # Run the survival test
    print("====== RUNNING SURVIVAL TEST ======")
    steps = 300
    results = run_survival_test(steps=steps, num_trials=3)
    plot_results(results, steps)
    
    # Run the farming capability test
    print("\n====== RUNNING FARMING CAPABILITY TEST ======")
    test_farming_capability(steps=300)
    
    # Run the agent death test
    print("\n====== RUNNING AGENT DEATH TEST ======")
    test_agent_death(steps=50) 