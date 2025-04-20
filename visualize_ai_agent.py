import pygame
import numpy as np
from environment import GridWorld
from rule_based_agent import RuleBasedAgent
from visualize import GameVisualizer

def visualize_rule_based_agent():
    """Create a visualization of the rule-based agent in action."""
    # Create environment
    env = GridWorld(width=30, height=20, water_probability=0.15)
    
    # Add some plants to make it interesting
    for _ in range(20):
        row = np.random.randint(0, env.height)
        col = np.random.randint(0, env.width)
        if env.grid[row, col] == GridWorld.SOIL:
            env.set_cell(row, col, GridWorld.PLANT)
            # Set some plants to be mature
            plant_state = np.random.choice([
                GridWorld.PLANT_SEED,
                GridWorld.PLANT_GROWING,
                GridWorld.PLANT_MATURE
            ], p=[0.2, 0.3, 0.5])  # 20% seed, 30% growing, 50% mature
            env.plant_state[row, col] = plant_state
    
    # Create rule-based agent
    agent = RuleBasedAgent(env, start_row=env.height // 2, start_col=env.width // 2)
    
    # Set initial agent state 
    agent.hunger = 40.0
    agent.thirst = 40.0
    agent.seeds = 5
    
    # Create visualizer
    visualizer = GameVisualizer(env, cell_size=25, info_width=300)
    visualizer.set_agent(agent)
    
    # Run the AI-controlled visualization
    run_ai_visualization(env, agent, visualizer)

def visualize_agent_death():
    """Create a visualization that shows the agent's death process."""
    # Create environment with minimal resources
    env = GridWorld(width=30, height=20, water_probability=0.05)
    
    # Deliberately position water far from the agent
    for row in range(2, 5):
        for col in range(25, 28):
            env.set_cell(row, col, GridWorld.WATER)
    
    # Create rule-based agent
    agent = RuleBasedAgent(env, start_row=10, start_col=10)
    
    # Set initial agent state to near-critical
    agent.hunger = 85.0  # Very hungry
    agent.thirst = 85.0  # Very thirsty
    agent.health = 20.0  # Low health
    agent.seeds = 1      # Just one seed
    
    # Create visualizer
    visualizer = GameVisualizer(env, cell_size=25, info_width=300)
    visualizer.set_agent(agent)
    
    # Run the visualization
    run_ai_visualization(env, agent, visualizer)

def run_ai_visualization(env, agent, visualizer):
    """
    Run the visualization with the AI-controlled agent.
    
    Args:
        env: The GridWorld environment
        agent: The RuleBasedAgent
        visualizer: The GameVisualizer
    """
    # Initialize pygame
    pygame.init()
    pygame.display.set_caption("Rule-Based Agent Simulation")
    clock = pygame.time.Clock()
    
    # Create a font for additional info
    font = pygame.font.SysFont('Arial', 14)
    death_font = pygame.font.SysFont('Arial', 24, bold=True)
    
    # Display control instructions
    print("Rule-Based Agent Simulation")
    print("---------------------------")
    print("Controls:")
    print("  Space: Pause/Resume simulation")
    print("  A: Toggle AI control (on/off)")
    print("  Tab: Step simulation manually when paused")
    print("  Arrow Keys: Manual control when AI is off")
    print("  E, D, P, T, H: Manual actions when AI is off")
    print("  Esc: Quit")
    
    # Simulation state
    running = True
    paused = False
    ai_control = True
    step_count = 0
    ai_delay = 15  # Frames between AI steps
    ai_timer = 0
    env_step_interval = 10  # Steps between environment updates
    
    # For tracking agent death
    death_reported = False
    death_message = ""
    
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    print(f"Simulation {'paused' if paused else 'resumed'}")
                elif event.key == pygame.K_a:
                    ai_control = not ai_control
                    print(f"AI control {'enabled' if ai_control else 'disabled'}")
                elif event.key == pygame.K_TAB and paused:
                    # Manual step when paused
                    if ai_control:
                        result = agent.step_ai()
                        if not result['alive'] and not death_reported:
                            death_reported = True
                            death_message = result.get('cause_of_death', 'Unknown cause')
                            print(f"Agent died! Cause: {death_message}")
                    step_count += 1
                    print(f"Manual step {step_count}, Task: {agent.current_task}")
                
                # Manual control when AI is off
                if not ai_control and agent.is_alive:
                    if event.key == pygame.K_UP:
                        agent.step(agent.MOVE_NORTH)
                    elif event.key == pygame.K_DOWN:
                        agent.step(agent.MOVE_SOUTH)
                    elif event.key == pygame.K_RIGHT:
                        agent.step(agent.MOVE_EAST)
                    elif event.key == pygame.K_LEFT:
                        agent.step(agent.MOVE_WEST)
                    elif event.key == pygame.K_e:
                        agent.step(agent.EAT)
                    elif event.key == pygame.K_d:
                        agent.step(agent.DRINK)
                    elif event.key == pygame.K_p:
                        agent.step(agent.PLANT_SEED)
                    elif event.key == pygame.K_t:
                        agent.step(agent.TEND_PLANT)
                    elif event.key == pygame.K_h:
                        agent.step(agent.HARVEST)
        
        # Run AI actions
        if not paused and ai_control and agent.is_alive:
            ai_timer += 1
            if ai_timer >= ai_delay:
                ai_timer = 0
                result = agent.step_ai()
                step_count += 1
                
                # Check if agent died
                if not result['alive'] and not death_reported:
                    death_reported = True
                    death_message = result.get('cause_of_death', 'Unknown cause')
                    print(f"Agent died! Cause: {death_message}")
                
                # Step the environment periodically
                if step_count % env_step_interval == 0:
                    env.step()
        
        # Update visualization
        visualizer.update()
        
        # Display additional information
        status_text = f"Step: {step_count}  Task: {agent.current_task}  AI: {'On' if ai_control else 'Off'}"
        if not agent.is_alive:
            status_text += "  Status: DEAD"
        
        info_surface = font.render(status_text, True, (0, 0, 0))
        visualizer.screen.blit(info_surface, (10, 10))
        
        # Display death message if agent is dead
        if not agent.is_alive:
            death_surface = death_font.render(f"AGENT DIED - {death_message}", True, (255, 0, 0))
            # Center the text on screen
            text_rect = death_surface.get_rect(center=(visualizer.screen.get_width()//2, visualizer.screen.get_height()//2))
            visualizer.screen.blit(death_surface, text_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    # Choose which visualization to run:
    # visualize_rule_based_agent()  # Regular agent
    visualize_agent_death()  # Agent set up to die quickly 