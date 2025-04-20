import pygame
import sys
import numpy as np
from environment import GridWorld
from agent import Agent

class GameVisualizer:
    """Pygame-based visualization for the GridWorld environment and Agent."""
    
    # Colors
    EMPTY_COLOR = (200, 200, 200)  # Light gray
    WATER_COLOR = (64, 164, 223)   # Blue
    SOIL_COLOR = (139, 69, 19)     # Brown
    PLANT_COLORS = {
        GridWorld.PLANT_NONE: (0, 0, 0),        # Black (not used)
        GridWorld.PLANT_SEED: (210, 180, 140),  # Light brown
        GridWorld.PLANT_GROWING: (144, 238, 144),  # Light green
        GridWorld.PLANT_MATURE: (0, 128, 0)     # Green
    }
    AGENT_COLOR = (255, 0, 0)      # Red
    GRID_LINE_COLOR = (50, 50, 50)  # Dark gray
    TEXT_COLOR = (0, 0, 0)         # Black
    
    def __init__(self, grid_world, cell_size=30, info_width=300):
        """
        Initialize the visualizer.
        
        Args:
            grid_world: GridWorld instance
            cell_size: Size of each cell in pixels
            info_width: Width of the info panel in pixels
        """
        self.grid_world = grid_world
        self.cell_size = cell_size
        self.info_width = info_width
        
        # Calculate display dimensions
        self.width = grid_world.width * cell_size
        self.height = grid_world.height * cell_size
        self.display_width = self.width + info_width
        
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("GridWorld Simulation")
        self.screen = pygame.display.set_mode((self.display_width, self.height))
        self.font = pygame.font.SysFont('Arial', 14)
        self.clock = pygame.time.Clock()
        
        # Set up agent (to be provided later)
        self.agent = None
    
    def set_agent(self, agent):
        """Set the agent to be visualized."""
        self.agent = agent
    
    def draw_grid(self):
        """Draw the grid with cell types and properties."""
        # Fill the background
        self.screen.fill((255, 255, 255))
        
        # Draw each cell
        for row in range(self.grid_world.height):
            for col in range(self.grid_world.width):
                rect = pygame.Rect(
                    col * self.cell_size, 
                    row * self.cell_size,
                    self.cell_size, 
                    self.cell_size
                )
                
                # Get cell type
                cell_type = self.grid_world.grid[row, col]
                
                # Color based on cell type
                if cell_type == GridWorld.EMPTY:
                    color = self.EMPTY_COLOR
                elif cell_type == GridWorld.WATER:
                    color = self.WATER_COLOR
                    # Variation based on water level
                    water_level = self.grid_world.water_level[row, col]
                    # Adjust blue channel based on water level (0-10)
                    blue_adjust = max(0, min(255, int(223 + (water_level * 3))))
                    color = (64, 164, blue_adjust)
                elif cell_type == GridWorld.SOIL:
                    color = self.SOIL_COLOR
                    # Variation based on fertility
                    fertility = self.grid_world.soil_fertility[row, col]
                    # Darker brown for more fertile soil
                    darkness = max(69, 139 - int(fertility * 5))
                    color = (139, darkness, 19)
                elif cell_type == GridWorld.PLANT:
                    plant_state = self.grid_world.plant_state[row, col]
                    color = self.PLANT_COLORS.get(plant_state, (0, 255, 0))
                
                # Draw cell
                pygame.draw.rect(self.screen, color, rect)
                
                # Optional: Add water level or fertility text
                if cell_type == GridWorld.WATER:
                    water_text = self.font.render(
                        f"{int(self.grid_world.water_level[row, col])}", 
                        True, 
                        (255, 255, 255)
                    )
                    text_rect = water_text.get_rect(
                        center=(col * self.cell_size + self.cell_size // 2,
                               row * self.cell_size + self.cell_size // 2)
                    )
                    self.screen.blit(water_text, text_rect)
                elif cell_type == GridWorld.SOIL:
                    fertility_text = self.font.render(
                        f"{int(self.grid_world.soil_fertility[row, col])}", 
                        True, 
                        (255, 255, 255)
                    )
                    text_rect = fertility_text.get_rect(
                        center=(col * self.cell_size + self.cell_size // 2,
                               row * self.cell_size + self.cell_size // 2)
                    )
                    self.screen.blit(fertility_text, text_rect)
        
        # Draw grid lines
        for row in range(self.grid_world.height + 1):
            pygame.draw.line(
                self.screen,
                self.GRID_LINE_COLOR,
                (0, row * self.cell_size),
                (self.width, row * self.cell_size),
                1
            )
        
        for col in range(self.grid_world.width + 1):
            pygame.draw.line(
                self.screen,
                self.GRID_LINE_COLOR,
                (col * self.cell_size, 0),
                (col * self.cell_size, self.height),
                1
            )
    
    def draw_agent(self):
        """Draw the agent on the grid."""
        if self.agent:
            row, col = self.agent.get_position()
            
            # Agent is drawn as a circle in the cell
            center_x = col * self.cell_size + self.cell_size // 2
            center_y = row * self.cell_size + self.cell_size // 2
            radius = self.cell_size // 3
            
            # Draw agent
            pygame.draw.circle(self.screen, self.AGENT_COLOR, (center_x, center_y), radius)
            
            # Draw agent's "face" in direction of last move (if any)
            if self.agent.action_history:
                last_action = self.agent.action_history[-1]
                if last_action[0] == 'move':
                    direction = last_action[1]
                    # Calculate position of the "eye" based on direction
                    eye_offset = 5
                    if direction == Agent.MOVE_NORTH:
                        eye_x, eye_y = center_x, center_y - eye_offset
                    elif direction == Agent.MOVE_SOUTH:
                        eye_x, eye_y = center_x, center_y + eye_offset
                    elif direction == Agent.MOVE_EAST:
                        eye_x, eye_y = center_x + eye_offset, center_y
                    elif direction == Agent.MOVE_WEST:
                        eye_x, eye_y = center_x - eye_offset, center_y
                    else:
                        eye_x, eye_y = center_x, center_y
                    
                    # Draw eye
                    pygame.draw.circle(self.screen, (255, 255, 255), (eye_x, eye_y), 3)
    
    def draw_info_panel(self):
        """Draw information panel on the right side."""
        if not self.agent:
            return
            
        # Draw background for info panel
        info_rect = pygame.Rect(self.width, 0, self.info_width, self.height)
        pygame.draw.rect(self.screen, (240, 240, 240), info_rect)
        
        # Get agent status
        status = self.agent.get_status()
        
        # Line height for text
        line_height = 25
        
        # Draw the heading
        title = self.font.render("Agent Status", True, self.TEXT_COLOR)
        self.screen.blit(title, (self.width + 10, 10))
        
        # Position
        pos_text = self.font.render(
            f"Position: ({status['position'][0]}, {status['position'][1]})", 
            True, 
            self.TEXT_COLOR
        )
        self.screen.blit(pos_text, (self.width + 10, 40))
        
        # Draw status bars
        statuses = [
            ("Health", status['health'], 100, (0, 255, 0)),
            ("Energy", status['energy'], 100, (255, 255, 0)),
            ("Hunger", status['hunger'], 100, (255, 165, 0)),
            ("Thirst", status['thirst'], 100, (64, 164, 223))
        ]
        
        for i, (name, value, max_value, color) in enumerate(statuses):
            y_pos = 70 + i * 60
            
            # Draw label
            label = self.font.render(f"{name}: {value:.1f}/{max_value}", True, self.TEXT_COLOR)
            self.screen.blit(label, (self.width + 10, y_pos))
            
            # Draw bar background
            bar_width = self.info_width - 20
            bar_height = 20
            bg_rect = pygame.Rect(self.width + 10, y_pos + 25, bar_width, bar_height)
            pygame.draw.rect(self.screen, (200, 200, 200), bg_rect)
            
            # Draw actual bar
            fill_width = int((value / max_value) * bar_width)
            fill_rect = pygame.Rect(self.width + 10, y_pos + 25, fill_width, bar_height)
            pygame.draw.rect(self.screen, color, fill_rect)
        
        # Draw seeds count
        seeds_y = 310
        seeds_text = self.font.render(f"Seeds: {status['seeds']}", True, self.TEXT_COLOR)
        self.screen.blit(seeds_text, (self.width + 10, seeds_y))
        
        # Draw action history
        history_y = 350
        history_title = self.font.render("Recent Actions:", True, self.TEXT_COLOR)
        self.screen.blit(history_title, (self.width + 10, history_y))
        
        # Show last 5 actions
        for i, action in enumerate(self.agent.action_history[-5:]):
            if action[0] == 'move':
                direction_names = {
                    Agent.MOVE_NORTH: "North",
                    Agent.MOVE_SOUTH: "South",
                    Agent.MOVE_EAST: "East",
                    Agent.MOVE_WEST: "West"
                }
                action_text = f"Move {direction_names.get(action[1], '?')}"
            elif action[0] == 'eat':
                action_text = "Eat plant"
            elif action[0] == 'drink':
                action_text = "Drink water"
            elif action[0] == 'plant_seed':
                action_text = "Plant seed"
            elif action[0] == 'tend_plant':
                action_text = "Tend plant"
            elif action[0] == 'harvest':
                action_text = f"Harvest plant (+{action[1]} seeds)"
            else:
                action_text = str(action)
                
            action_label = self.font.render(action_text, True, self.TEXT_COLOR)
            self.screen.blit(action_label, (self.width + 20, history_y + 25 + i * 20))
    
    def update(self):
        """Update the display."""
        self.draw_grid()
        self.draw_info_panel()
        self.draw_agent()
        pygame.display.flip()
    
    def run_simulation(self, frames_per_step=30):
        """
        Run an interactive simulation.
        
        Args:
            frames_per_step: Number of frames to wait between automated steps
        """
        if not self.agent:
            print("Error: No agent provided for simulation")
            return
            
        running = True
        paused = False
        frame_count = 0
        
        # Display control instructions
        print("Simulation Controls:")
        print("  Arrow Keys: Move agent")
        print("  E: Eat plant")
        print("  D: Drink water")
        print("  P: Plant seed")
        print("  T: Tend plant")
        print("  H: Harvest plant")
        print("  Space: Pause/Resume simulation")
        print("  Esc: Quit")
        
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
                    # Manual control with arrow keys
                    elif event.key == pygame.K_UP:
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
            
            if not paused:
                frame_count += 1
                if frame_count >= frames_per_step:
                    frame_count = 0
                    # For automated stepping, you can add agent AI logic here
                    # For now, we'll just update status without moving
                    self.agent.update_status()
            
            # Update the display
            self.update()
            
            # Control simulation speed
            self.clock.tick(60)
        
        pygame.quit()

def main():
    """Demo function to show visualization."""
    # Create environment and agent
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
    
    # Create agent
    agent = Agent(env, start_row=env.height // 2, start_col=env.width // 2)
    
    # Set initial agent state for demo
    agent.hunger = 30.0
    agent.thirst = 40.0
    
    # Create and run visualizer
    visualizer = GameVisualizer(env)
    visualizer.set_agent(agent)
    visualizer.run_simulation()

if __name__ == "__main__":
    main() 