import torch
import numpy as np
from agent import Agent
from model import ObservationEncoder, AgentCNN

class ModelBasedAgent(Agent):
    """
    A model-based agent that uses a neural network to make decisions.
    """
    
    def __init__(self, environment, model_path='trained_agent.pth', start_row=0, start_col=0):
        """
        Initialize the model-based agent.
        
        Args:
            environment: The environment the agent will operate in.
            model_path: Path to the trained model weights.
            start_row: Initial row position.
            start_col: Initial column position.
        """
        super().__init__(environment, start_row, start_col)
        
        # Set up the model
        self.encoder = ObservationEncoder(environment)
        self.model = AgentCNN()
        
        # Load trained weights
        try:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set to evaluation mode
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using untrained model (random decisions)")
        
        # Track current task
        self.current_task = "Initializing"
        self.is_alive = True
        
    def step_ai(self):
        """
        Take an AI-controlled step using the model.
        
        Returns:
            dict: A result dictionary containing information about the action.
        """
        # Check if agent is alive
        if not self.is_alive:
            return {"alive": False, "cause_of_death": "Agent is already dead"}
        
        # Update status before making a decision
        is_alive = self.update_status()
        if not is_alive:
            self.is_alive = False
            cause = ""
            if self.hunger >= 100:
                cause = "Starvation"
            elif self.thirst >= 100:
                cause = "Dehydration"
            else:
                cause = "Health depleted"
            return {"alive": False, "cause_of_death": cause}
        
        # Get observation
        observation = self.encoder.get_observation(self)
        observation = observation.unsqueeze(0)  # Add batch dimension
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = self.model(observation).squeeze(0)
        
        # Get the highest probability action
        action_idx = torch.argmax(action_probs).item()
        
        # Map the action index to actual action
        action_map = {
            0: Agent.MOVE_NORTH,
            1: Agent.MOVE_SOUTH,
            2: Agent.MOVE_EAST,
            3: Agent.MOVE_WEST,
            4: Agent.EAT,
            5: Agent.DRINK,
            6: Agent.PLANT_SEED,
            7: Agent.TEND_PLANT,
            8: Agent.HARVEST
        }
        
        action = action_map[action_idx]
        
        # Set current task based on action
        self._set_current_task(action)
        
        # Execute the action
        result = self.step(action)
        
        # For debugging: print action and success
        print(f"ModelAgent action: {self.current_task}, Success: {result['success']}")
        
        return {"alive": result['alive'], "action": action, "success": result['success']}
    
    def _set_current_task(self, action):
        """Update the current task based on the selected action."""
        if action == Agent.MOVE_NORTH:
            self.current_task = "Moving North"
        elif action == Agent.MOVE_SOUTH:
            self.current_task = "Moving South"
        elif action == Agent.MOVE_EAST:
            self.current_task = "Moving East"
        elif action == Agent.MOVE_WEST:
            self.current_task = "Moving West"
        elif action == Agent.EAT:
            self.current_task = "Eating"
        elif action == Agent.DRINK:
            self.current_task = "Drinking"
        elif action == Agent.PLANT_SEED:
            self.current_task = "Planting Seed"
        elif action == Agent.TEND_PLANT:
            self.current_task = "Tending Plant"
        elif action == Agent.HARVEST:
            self.current_task = "Harvesting" 