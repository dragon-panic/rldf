import logging
import numpy as np
from environment import GridWorld
from rule_based_agent import RuleBasedAgent
from model_based_agent import ModelBasedAgent

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_rule_based_agent():
    """Test evaluation with a rule-based agent"""
    logger.info("Creating environment...")
    env = GridWorld(width=20, height=15, water_probability=0.1)
    
    logger.info("Creating rule-based agent...")
    agent = RuleBasedAgent(env)
    
    logger.info("Running rule-based agent for 100 steps...")
    step = 0
    survived = True
    
    while step < 100 and survived:
        result = agent.step_ai()
        success = result.get('success', False)
        action = result.get('action', -1)
        survived = result.get('alive', True)
        
        if step % 10 == 0:
            logger.info(f"Step {step}: Action {action}, Success: {success}, Alive: {survived}")
            logger.info(f"  Status - Health: {agent.health:.1f}, Hunger: {agent.hunger:.1f}, "
                      f"Thirst: {agent.thirst:.1f}, Energy: {agent.energy:.1f}, Seeds: {agent.seeds}")
        
        step += 1
    
    logger.info(f"Test completed after {step} steps")
    logger.info(f"Agent survived: {survived}")
    if not survived:
        if agent.hunger >= 100:
            cause = "Starvation"
        elif agent.thirst >= 100:
            cause = "Dehydration"
        elif agent.energy <= 0:
            cause = "Exhaustion"
        else:
            cause = "Unknown"
        logger.info(f"Cause of death: {cause}")

def test_model_based_agent():
    """Test evaluation with a model-based agent"""
    logger.info("Creating environment...")
    env = GridWorld(width=20, height=15, water_probability=0.1)
    
    logger.info("Creating model-based agent...")
    agent = ModelBasedAgent(env, model_path="models/ppo_trained_agent.pth")
    
    logger.info("Running model-based agent for 100 steps...")
    step = 0
    survived = True
    
    while step < 100 and survived:
        result = agent.step_ai()
        success = result.get('success', False)
        action = result.get('action', -1)
        survived = result.get('alive', True)
        
        if step % 10 == 0:
            logger.info(f"Step {step}: Action {action}, Success: {success}, Alive: {survived}")
            logger.info(f"  Status - Health: {agent.health:.1f}, Hunger: {agent.hunger:.1f}, "
                      f"Thirst: {agent.thirst:.1f}, Energy: {agent.energy:.1f}, Seeds: {agent.seeds}")
        
        step += 1
    
    logger.info(f"Test completed after {step} steps")
    logger.info(f"Agent survived: {survived}")
    if not survived:
        if agent.hunger >= 100:
            cause = "Starvation"
        elif agent.thirst >= 100:
            cause = "Dehydration"
        elif agent.energy <= 0:
            cause = "Exhaustion"
        else:
            cause = "Unknown"
        logger.info(f"Cause of death: {cause}")

def collect_simple_metrics(agent_type='rule_based', num_episodes=5, max_steps=100):
    """Collect simple metrics on agent performance"""
    metrics = {
        'survival_steps': [],
        'plants_harvested': [],
        'seeds_planted': [],
        'survived': []
    }
    
    for episode in range(num_episodes):
        logger.info(f"Episode {episode+1}/{num_episodes}")
        
        # Create environment and agent
        env = GridWorld(width=20, height=15, water_probability=0.1)
        
        if agent_type == 'rule_based':
            agent = RuleBasedAgent(env)
        else:  # model_based
            agent = ModelBasedAgent(env, model_path="models/ppo_trained_agent.pth")
        
        # Run episode
        step = 0
        survived = True
        episode_stats = {'plants_harvested': 0, 'seeds_planted': 0}
        
        while step < max_steps and survived:
            result = agent.step_ai()
            success = result.get('success', False)
            action = result.get('action', -1)
            survived = result.get('alive', True)
            
            # Track specific actions
            if success:
                if action == 6:  # PLANT_SEED
                    episode_stats['seeds_planted'] += 1
                elif action == 8:  # HARVEST
                    episode_stats['plants_harvested'] += 1
            
            step += 1
        
        # Record metrics
        metrics['survival_steps'].append(step)
        metrics['plants_harvested'].append(episode_stats['plants_harvested'])
        metrics['seeds_planted'].append(episode_stats['seeds_planted'])
        metrics['survived'].append(1 if survived else 0)
        
        logger.info(f"  Steps: {step}, Harvested: {episode_stats['plants_harvested']}, "
                  f"Planted: {episode_stats['seeds_planted']}, Survived: {survived}")
    
    # Calculate averages
    logger.info("\nResults:")
    logger.info(f"Average Survival Time: {np.mean(metrics['survival_steps']):.1f} steps")
    logger.info(f"Average Plants Harvested: {np.mean(metrics['plants_harvested']):.2f}")
    logger.info(f"Average Seeds Planted: {np.mean(metrics['seeds_planted']):.2f}")
    logger.info(f"Survival Rate: {np.mean(metrics['survived'])*100:.1f}%")
    
    return metrics

if __name__ == "__main__":
    logger.info("Starting test_evaluate.py...")
    
    # Uncomment these to run individual tests
    # test_rule_based_agent()
    # test_model_based_agent()
    
    # Collect metrics
    logger.info("\nTesting Rule-Based Agent:")
    rule_based_metrics = collect_simple_metrics('rule_based', num_episodes=3, max_steps=100)
    
    logger.info("\nTesting Model-Based Agent:")
    model_based_metrics = collect_simple_metrics('model_based', num_episodes=3, max_steps=100)
    
    # Compare
    logger.info("\nComparison:")
    logger.info(f"Rule-Based Survival Rate: {np.mean(rule_based_metrics['survived'])*100:.1f}%")
    logger.info(f"Model-Based Survival Rate: {np.mean(model_based_metrics['survived'])*100:.1f}%")
    
    logger.info(f"Rule-Based Avg Harvested: {np.mean(rule_based_metrics['plants_harvested']):.2f}")
    logger.info(f"Model-Based Avg Harvested: {np.mean(model_based_metrics['plants_harvested']):.2f}")
    
    logger.info("Test complete.") 