import pygame
import numpy as np
import argparse
import logging
from environment import GridWorld
from agent import Agent
from rule_based_agent import RuleBasedAgent
from model_based_agent import ModelBasedAgent
from visualize import GameVisualizer

# Set up logging to be consistent with train.py
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to set log level for all loggers (same as in train.py)
def set_log_level(log_level):
    """Set the log level for all loggers in the application."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Set root logger level - this affects all loggers
    logging.getLogger().setLevel(numeric_level)
    
    # Also set our module logger level explicitly
    logger.setLevel(numeric_level)

class EnhancedVisualizer(GameVisualizer):
    """Enhanced visualization that supports both manual and AI control modes"""
    
    def __init__(self, grid_world, cell_size=30, info_width=300):
        super().__init__(grid_world, cell_size, info_width)
        self.ai_control = True
        self.death_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.death_reported = False
        self.death_message = ""
        self.restart_prompt = False
        
    def draw_status_info(self, step_count):
        """Draw additional status information"""
        font = pygame.font.SysFont('Arial', 14)
        
        # Determine agent type
        agent_type = "Manual"
        if "RuleBasedAgent" in self.agent.__class__.__name__:
            agent_type = "Rule-Based"
        elif "ModelBasedAgent" in self.agent.__class__.__name__:
            agent_type = "Neural Network"
        
        if hasattr(self.agent, 'current_task'):
            status_text = f"Step: {step_count}  Agent: {agent_type}  Task: {self.agent.current_task}  AI: {'On' if self.ai_control else 'Off'}"
        else:
            status_text = f"Step: {step_count}  Agent: {agent_type}  AI: {'On' if self.ai_control else 'Off'}"
            
        if hasattr(self.agent, 'is_alive') and not self.agent.is_alive:
            status_text += "  Status: DEAD"
        
        info_surface = font.render(status_text, True, (0, 0, 0))
        self.screen.blit(info_surface, (10, 10))
        
        # Display death message if agent is dead
        if hasattr(self.agent, 'is_alive') and not self.agent.is_alive:
            death_surface = self.death_font.render(f"AGENT DIED - {self.death_message}", True, (255, 0, 0))
            # Center the text on screen
            text_rect = death_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2))
            self.screen.blit(death_surface, text_rect)
            
            # Add restart prompt
            if self.restart_prompt:
                restart_font = pygame.font.SysFont('Arial', 18)
                restart_surface = restart_font.render("Press any key to restart", True, (0, 0, 255))
                restart_rect = restart_surface.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 + 40))
                self.screen.blit(restart_surface, restart_rect)
    
    def run_simulation(self, mode='manual', frames_per_step=30):
        """
        Run an interactive simulation.
        
        Args:
            mode: 'manual', 'ai', or 'hybrid'
            frames_per_step: Number of frames to wait between automated steps
        """
        if not self.agent:
            logger.error("Error: No agent provided for simulation")
            return
            
        running = True
        paused = False
        frame_count = 0
        step_count = 0
        ai_delay = 8  # Frames between AI steps
        ai_timer = 0
        env_step_interval = 5  # Steps between environment updates
        
        self.ai_control = (mode == 'ai')
        
        # Determine agent type
        agent_type = "Manual"
        if "RuleBasedAgent" in self.agent.__class__.__name__:
            agent_type = "Rule-Based AI"
        elif "ModelBasedAgent" in self.agent.__class__.__name__:
            agent_type = "Neural Network AI"
        
        # Display appropriate control instructions based on mode
        logger.info(f"Running in {mode.upper()} mode with {agent_type} agent")
        logger.info("Simulation Controls:")
        
        if mode in ['manual', 'hybrid']:
            logger.info("  Arrow Keys: Move agent")
            logger.info("  E: Eat plant")
            logger.info("  D: Drink water")
            logger.info("  P: Plant seed")
            logger.info("  T: Tend plant")
            logger.info("  H: Harvest plant")
        
        logger.info("  Space: Pause/Resume simulation")
        
        if mode == 'hybrid':
            logger.info("  A: Toggle AI control (on/off)")
            logger.info("  Tab: Step simulation manually when paused")
        
        logger.info("  Esc: Quit")
        
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
                        logger.info(f"Simulation {'paused' if paused else 'resumed'}")
                    
                    # Restart game if agent is dead and any key is pressed
                    if hasattr(self.agent, 'is_alive') and not self.agent.is_alive and self.restart_prompt:
                        # Reinitialize the environment and agent with the same settings
                        if mode == 'manual':
                            self.grid_world, self.agent = setup_manual_environment()
                        else:
                            if "RuleBasedAgent" in self.agent.__class__.__name__:
                                agent_type = 'rule_based'
                            elif "ModelBasedAgent" in self.agent.__class__.__name__:
                                agent_type = 'model_based'
                            
                            # Get the model path if it's a model-based agent
                            model_path = None
                            if hasattr(self.agent, 'model_path'):
                                model_path = self.agent.model_path
                            else:
                                model_path = 'models/ppo_trained_agent.pth'
                            
                            # Reset the environment and agent
                            self.grid_world, self.agent = setup_ai_environment(
                                agent_type=agent_type,
                                model_path=model_path,
                                death_scenario=False  # Start with standard scenario
                            )
                        
                        self.death_reported = False
                        self.death_message = ""
                        self.restart_prompt = False
                        step_count = 0
                        ai_timer = 0
                        frame_count = 0
                        logger.info("Simulation restarted with a new agent")
                        continue
                    
                    # Handle AI control toggle in hybrid mode
                    if mode == 'hybrid' and event.key == pygame.K_a:
                        self.ai_control = not self.ai_control
                        logger.info(f"AI control {'enabled' if self.ai_control else 'disabled'}")
                    
                    # Manual stepping when paused in AI or hybrid mode
                    if (mode in ['ai', 'hybrid']) and event.key == pygame.K_TAB and paused:
                        if self.ai_control and hasattr(self.agent, 'step_ai'):
                            result = self.agent.step_ai()
                            if not result.get('alive', True) and not self.death_reported:
                                self.death_reported = True
                                self.death_message = result.get('cause_of_death', 'Unknown cause')
                                logger.warning(f"Agent died! Cause: {self.death_message}")
                                self.restart_prompt = True
                        step_count += 1
                        if hasattr(self.agent, 'current_task'):
                            logger.debug(f"Manual step {step_count}, Task: {self.agent.current_task}")
                        else:
                            logger.debug(f"Manual step {step_count}")
                    
                    # Manual control in manual mode or hybrid mode with AI off
                    if (mode == 'manual' or (mode == 'hybrid' and not self.ai_control)) and \
                       (not hasattr(self.agent, 'is_alive') or self.agent.is_alive):
                        if event.key == pygame.K_UP:
                            self.agent.step(Agent.MOVE_NORTH)
                        elif event.key == pygame.K_DOWN:
                            self.agent.step(Agent.MOVE_SOUTH)
                        elif event.key == pygame.K_RIGHT:
                            self.agent.step(Agent.MOVE_EAST)
                        elif event.key == pygame.K_LEFT:
                            self.agent.step(Agent.MOVE_WEST)
                        elif event.key == pygame.K_e:
                            self.agent.step(Agent.EAT)
                        elif event.key == pygame.K_d:
                            self.agent.step(Agent.DRINK)
                        elif event.key == pygame.K_p:
                            self.agent.step(Agent.PLANT_SEED)
                        elif event.key == pygame.K_t:
                            self.agent.step(Agent.TEND_PLANT)
                        elif event.key == pygame.K_h:
                            self.agent.step(Agent.HARVEST)
                        
                        # Check if agent died after a manual action
                        if hasattr(self.agent, 'is_alive') and not self.agent.is_alive and not self.death_reported:
                            self.death_reported = True
                            self.death_message = "Died from starvation, thirst, or other causes"
                            logger.warning(f"Agent died! Cause: {self.death_message}")
                            self.restart_prompt = True
            
            # Run AI actions
            agent_alive = not hasattr(self.agent, 'is_alive') or self.agent.is_alive
            if mode in ['ai', 'hybrid'] and not paused and self.ai_control and agent_alive:
                ai_timer += 1
                if ai_timer >= ai_delay:
                    ai_timer = 0
                    if hasattr(self.agent, 'step_ai'):
                        result = self.agent.step_ai()
                        step_count += 1
                        
                        # Check if agent died
                        if not result.get('alive', True) and not self.death_reported:
                            self.death_reported = True
                            self.death_message = result.get('cause_of_death', 'Unknown cause')
                            logger.warning(f"Agent died! Cause: {self.death_message}")
                            self.restart_prompt = True
                    else:
                        # Basic status update for non-AI agents
                        self.agent.update_status()
                    
                    # Step the environment periodically
                    if step_count % env_step_interval == 0:
                        self.grid_world.step()
            elif mode == 'manual' and not paused:
                frame_count += 1
                if frame_count >= frames_per_step:
                    frame_count = 0
                    self.agent.update_status()
                    step_count += 1
            
            # Always update status for idle agents, even when paused
            # This is to ensure hunger and thirst increase over time
            if not paused and agent_alive:
                idle_update_frequency = 60  # Update status every 60 frames (roughly once per second)
                if frame_count % idle_update_frequency == 0:
                    # Ensure agent is still alive
                    if hasattr(self.agent, 'is_alive') and self.agent.is_alive:
                        pass  # Status is already updated by step_ai or manual update
                    elif not hasattr(self.agent, 'is_alive'):
                        # For basic agents without is_alive attribute
                        self.agent.update_status()
            
            # Update the display
            self.update()
            self.draw_status_info(step_count)
            pygame.display.flip()
            
            # Control simulation speed
            pygame.time.Clock().tick(60)
        
        pygame.quit()

def find_valid_start_position(env, preferred_row=None, preferred_col=None):
    """
    Find a valid starting position (non-water cell) for an agent.
    
    Args:
        env: GridWorld environment
        preferred_row: Preferred row coordinate (will try to find nearest valid cell if this is water)
        preferred_col: Preferred column coordinate (will try to find nearest valid cell if this is water)
        
    Returns:
        tuple: (row, col) of valid starting position
    """
    # Use center of grid if no preference given
    if preferred_row is None:
        preferred_row = env.height // 2
    if preferred_col is None:
        preferred_col = env.width // 2
    
    # Ensure coordinates are in bounds
    preferred_row = min(max(0, preferred_row), env.height - 1)
    preferred_col = min(max(0, preferred_col), env.width - 1)
    
    # If preferred position is not water, return it
    if env.grid[preferred_row, preferred_col] != GridWorld.WATER:
        return preferred_row, preferred_col
    
    # Find the nearest non-water cell
    found_valid_position = False
    search_radius = 1
    
    # Expand search radius until we find a non-water cell
    while not found_valid_position and search_radius < max(env.height, env.width):
        # Check cells in a square around the preferred position
        for r_offset in range(-search_radius, search_radius + 1):
            for c_offset in range(-search_radius, search_radius + 1):
                # Only check the perimeter of the square
                if abs(r_offset) == search_radius or abs(c_offset) == search_radius:
                    r = preferred_row + r_offset
                    c = preferred_col + c_offset
                    
                    # Ensure position is within bounds
                    if (0 <= r < env.height and 
                        0 <= c < env.width and
                        env.grid[r, c] != GridWorld.WATER):
                        
                        return r, c
        
        # Increase search radius if needed
        search_radius += 1
    
    # If still no valid position found, do a full grid search
    for r in range(env.height):
        for c in range(env.width):
            if env.grid[r, c] != GridWorld.WATER:
                return r, c
    
    # This should never happen unless the entire grid is water
    raise ValueError("Could not find a valid starting position - grid is all water")

def setup_manual_environment():
    """Set up a demo environment for manual control"""
    env = GridWorld(width=20, height=15, water_probability=0.2)
    
    # Add some plants
    for _ in range(15):
        row = np.random.randint(0, env.height)
        col = np.random.randint(0, env.width)
        if env.grid[row, col] == GridWorld.SOIL:
            env.set_cell(row, col, GridWorld.PLANT)
            # Set some plants to be mature
            if np.random.random() < 0.5:
                env.plant_state[row, col] = GridWorld.PLANT_MATURE
    
    # Find a valid starting position
    start_row, start_col = find_valid_start_position(env)
    
    # Create agent
    agent = Agent(env, start_row=start_row, start_col=start_col)
    
    # Set initial agent state for demo
    agent.hunger = 30.0
    agent.thirst = 40.0
    
    return env, agent

def setup_ai_environment(agent_type='rule_based', death_scenario=False, model_path='models/ppo_trained_agent.pth'):
    """Set up an environment for AI agent control"""
    if death_scenario:
        # Create environment with minimal resources
        env = GridWorld(width=30, height=20, water_probability=0.05)
        
        # Deliberately position water far from the agent
        for row in range(2, 5):
            for col in range(25, 28):
                env.set_cell(row, col, GridWorld.WATER)
        
        # Find a valid starting position
        start_row, start_col = find_valid_start_position(env, preferred_row=10, preferred_col=10)
        
        # Create AI agent based on selected type
        if agent_type == 'model_based':
            agent = ModelBasedAgent(env, model_path=model_path, start_row=start_row, start_col=start_col)
        else:  # Default to rule-based
            agent = RuleBasedAgent(env, start_row=start_row, start_col=start_col)
        
        # Set initial agent state to near-critical
        agent.hunger = 85.0  # Very hungry
        agent.thirst = 85.0  # Very thirsty
        agent.health = 20.0  # Low health
        agent.seeds = 1      # Just one seed
    else:
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
        
        # Find a valid starting position
        start_row, start_col = find_valid_start_position(env)
        
        # Create AI agent based on selected type
        if agent_type == 'model_based':
            agent = ModelBasedAgent(env, model_path=model_path, start_row=start_row, start_col=start_col)
        else:  # Default to rule-based
            agent = RuleBasedAgent(env, start_row=start_row, start_col=start_col)
        
        # Set initial agent state 
        agent.hunger = 40.0
        agent.thirst = 40.0
        agent.seeds = 5
    
    return env, agent

def main():
    """Main function to run the simulation"""
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Run the RL Farming Simulation')
    parser.add_argument('--mode', type=str, choices=['manual', 'ai', 'hybrid'], 
                        default='ai', help='Simulation mode')
    parser.add_argument('--agent-type', type=str, choices=['rule_based', 'model_based'],
                        default='model_based', help='Type of AI agent to use')
    parser.add_argument('--death', action='store_true', 
                        help='Run death scenario (AI agent only)')
    parser.add_argument('--cell-size', type=int, default=25,
                        help='Size of grid cells in pixels')
    parser.add_argument('--model-path', type=str, default='models/ppo_trained_agent.pth',
                        help='Path to the trained model file (for model-based agent, can be models/ppo_trained_agent.pth or models/reinforce_trained_agent.pth)')
    parser.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Logging level for output verbosity')
    args = parser.parse_args()
    
    # Set the logging level for all loggers
    set_log_level(args.log_level)
    
    # Print information about the current mode and agent type
    logger.info(f"Running simulation in {args.mode.upper()} mode with {args.agent_type.upper().replace('_', ' ')} agent")
    if args.mode == 'manual':
        logger.info("Note: Default mode has been changed to 'ai' with 'model_based' agent.")
        logger.info("To use other modes, specify with --mode and --agent-type arguments.")
    
    # Setup environment and agent based on mode
    if args.mode == 'manual':
        env, agent = setup_manual_environment()
    else:  # 'ai' or 'hybrid'
        env, agent = setup_ai_environment(agent_type=args.agent_type, death_scenario=args.death, model_path=args.model_path)
    
    # Create visualizer
    visualizer = EnhancedVisualizer(env, cell_size=args.cell_size)
    visualizer.set_agent(agent)
    
    # Run simulation
    visualizer.run_simulation(mode=args.mode)

if __name__ == "__main__":
    main() 