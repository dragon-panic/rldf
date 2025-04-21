import torch
import random
import numpy as np
from agent import Agent
from model import ObservationEncoder
from reinforce_model import REINFORCEModel
import torch.nn.functional as F
from environment import GridWorld
import logging

# Get the logger that's configured in train.py/main.py
logger = logging.getLogger(__name__)

try:
    from ppo_model import PPOModel
    PPO_AVAILABLE = True
except ImportError:
    logger.warning("Warning: PPOModel not available. Will only use standard REINFORCEModel.")
    PPO_AVAILABLE = False

class ModelBasedAgent(Agent):
    """
    A model-based agent that uses a neural network to make decisions.
    """
    
    def __init__(self, environment, model_path='models/ppo_trained_agent.pth', start_row=0, start_col=0):
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
        
        # Determine model type based on filename
        if 'ppo' in model_path and PPO_AVAILABLE:
            self.model = PPOModel()
            self.is_ppo_model = True
            logger.info("Using PPO model architecture")
        else:
            self.model = REINFORCEModel()
            self.is_ppo_model = False
            if 'ppo' in model_path and not PPO_AVAILABLE:
                logger.warning("Warning: PPO model requested but PPOModel not available. Using standard REINFORCEModel.")
            else:
                logger.info("Using standard REINFORCEModel architecture")
        
        # Load trained weights
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            import os
            
            # Check if the file exists at the specified path
            if os.path.exists(model_path):
                logger.info(f"File exists and has size: {os.path.getsize(model_path)} bytes")
            else:
                # If not, try looking for it in the models directory
                model_filename = os.path.basename(model_path)
                models_dir_path = os.path.join('models', model_filename)
                
                if os.path.exists(models_dir_path):
                    logger.info(f"File found in models directory instead: {models_dir_path}")
                    model_path = models_dir_path
                    logger.info(f"Using model from: {model_path}")
                else:
                    logger.warning(f"File does not exist at specified path: {model_path}")
                    logger.warning(f"File also not found in models directory: {models_dir_path}")
                    logger.warning(f"Current working directory: {os.getcwd()}")
                    
                    # List available model files in both current and models directory
                    logger.info("Available model files in current directory:")
                    for file in os.listdir():
                        if file.endswith('.pth'):
                            logger.info(f"  - {file} ({os.path.getsize(file)} bytes)")
                    
                    if os.path.exists('models'):
                        logger.info("Available model files in models directory:")
                        for file in os.listdir('models'):
                            if file.endswith('.pth'):
                                logger.info(f"  - models/{file} ({os.path.getsize(os.path.join('models', file))} bytes)")
                    else:
                        logger.warning("The 'models' directory does not exist yet.")
            
            model_state = torch.load(model_path)
            self.model.load_state_dict(model_state)
            self.model.eval()  # Set to evaluation mode
            
            # Debug info about model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Successfully loaded model from {model_path}")
            logger.info(f"Model has {total_params} parameters")
            
            # Print a parameter sample to verify non-zero weights
            for name, param in self.model.named_parameters():
                logger.debug(f"Parameter {name}: shape={param.shape}, sample={param.data.flatten()[:3]}")
                break  # Just print the first parameter
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Using untrained model (random decisions)")
        
        # Track current task
        self.current_task = "Initializing"
        self.is_alive = True
        self.model_path = model_path  # Store model path for potential restarting
        
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
        
        # Get observation and action probabilities
        observation = self.encoder.get_observation(self)
        action_probs = self.get_action_probs_from_observation(observation)
        
        # Print agent's current status for debugging
        logger.debug(f"Agent Environment:")
        self.environment.print_surrounding_area(self.row, self.col, 3)  # 3 cells in each direction = 7x7 grid
        logger.debug(f"Agent status: Health={self.health:.1f}, Hunger={self.hunger:.1f}, Thirst={self.thirst:.1f}, Energy={self.energy:.1f}, Seeds={self.seeds}")
        
        # Print action probabilities for debugging
        action_names = ["Move North", "Move South", "Move East", "West", 
                        "Eat", "Drink", "Plant Seed", "Tend Plant", "Harvest"]
        
        logger.debug("Action probabilities:")
        for i, (name, prob) in enumerate(zip(action_names, action_probs.tolist())):
            logger.debug(f"  {i}: {name} = {prob:.3f}")
            
        # Use the shared action selection logic
        action_result = self._select_and_take_action(action_probs)
        
        # Return result
        return {
            "alive": self.is_alive, 
            "action": action_result["action"], 
            "success": action_result["success"],
            "attempted_actions": action_result["attempted_actions"]
        }
        
    def _select_and_take_action(self, action_probs):
        """
        Select an action based on probabilities and execute it.
        This is shared logic used by both step_ai and decide_action.
        
        Args:
            action_probs: Action probabilities from the model
            
        Returns:
            dict: Result containing the action, success status, and attempted actions
        """
        # Convert to numpy array for easier sorting
        if isinstance(action_probs, torch.Tensor):
            probs_array = action_probs.numpy()
        else:
            probs_array = action_probs
            
        # Sort actions by probability (descending)
        action_indices = np.argsort(-probs_array)
        
        # Map indices to action constants
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
        
        # Action names for logging
        action_names = ["Move North", "Move South", "Move East", "Move West", 
                        "Eat", "Drink", "Plant Seed", "Tend Plant", "Harvest"]
                        
        # Try actions in order of probability until one succeeds
        success = False
        chosen_action = None
        attempted_actions = []
        
        for action_idx in action_indices:
            action = action_map[action_idx.item()]
            action_name = action_names[action_idx.item()]
            
            # Remember this attempt
            attempted_actions.append(action_name)
            
            # Set current task based on action
            self._set_current_task(action)
            
            # Try to execute the action
            logger.debug(f"Trying action: {action_name}")
            result = self.step(action)
            
            # Check if action was successful
            if result['success']:
                success = True
                chosen_action = action
                logger.debug(f"Action {action_name} succeeded")
                break
            else:
                logger.debug(f"Action {action_name} failed, trying next action")
        
        # If all actions failed, just use the highest probability action for logging
        if not success:
            logger.warning("All actions failed to execute successfully")
            chosen_action = action_map[action_indices[0].item()]
            
        # Return result
        return {
            "action": chosen_action,
            "success": success,
            "attempted_actions": attempted_actions
        }
    
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

    def get_action_probs_from_observation(self, observation):
        """
        Get action probabilities directly from an observation.
        
        Args:
            observation: The current observation tensor
            
        Returns:
            Action probabilities as a tensor.
        """
        # Add batch dimension if needed
        x = observation.unsqueeze(0) if len(observation.shape) == 3 else observation
        
        # Forward pass through the model
        with torch.no_grad():
            if self.is_ppo_model:
                model_output = self.model(x)
                # Check if the model is returning 3 values (PPO with action_probs, state_value, action_logits)
                if isinstance(model_output, tuple) and len(model_output) == 3:
                    action_probs, _, _ = model_output
                else:
                    action_probs, _ = model_output  # Backward compatibility
                action_probs = action_probs.squeeze(0)
                logger.debug(f"PPO Value prediction: {model_output[1].item():.3f}")
            else:
                action_probs = self.model(x).squeeze(0)
                
        return action_probs
            
    def get_action_probs(self, state):
        """
        Get action probabilities from the model.
        
        Args:
            state: The current state observation.
            
        Returns:
            Action probabilities as a numpy array.
        """
        # Convert state to tensor format expected by model
        observation = self.encoder.get_observation(state) if not isinstance(state, torch.Tensor) else state
        x = observation.unsqueeze(0) if isinstance(observation, torch.Tensor) else torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension
        
        logger.debug(f"Input tensor shape: {x.shape}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        logger.debug(f"Input tensor range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        
        # Get action probabilities directly
        action_probs = self.get_action_probs_from_observation(x if isinstance(observation, torch.Tensor) else observation)
        
        # Check for NaN values
        if torch.isnan(action_probs).any():
            logger.warning("WARNING: NaN values detected in action probabilities!")
            # Replace NaNs with uniform distribution
            action_probs = torch.ones_like(action_probs) / action_probs.size(0)
        
        # Check for zero probabilities
        if (action_probs.sum() < 1e-10):
            logger.warning("WARNING: All action probabilities are near zero!")
            action_probs = torch.ones_like(action_probs) / action_probs.size(0)
        
        # Normalize probabilities
        action_probs = F.softmax(action_probs, dim=0)
        
        logger.debug(f"Output action probs: {action_probs.numpy()}")
        logger.debug(f"Sum of probabilities: {action_probs.sum().item():.6f}")
        
        return action_probs.numpy()
        
    def decide_action(self):
        """
        Decide the next action based on the neural network.
        This is used for evaluation and testing.
        
        Returns:
            int: The action to take
        """
        # Get the current observation
        observation = self.encoder.get_observation(self)
        
        # Get action probabilities
        action_probs = self.get_action_probs(observation)
        
        # Print current situation for debugging
        logger.debug(f"Current position: ({self.row}, {self.col})")
        logger.debug(f"Current task: {self.current_task}")
        logger.debug(f"Agent status - Health: {self.health:.1f}, Hunger: {self.hunger:.1f}, Thirst: {self.thirst:.1f}")
        
        try:
            # Use a simplified version of the shared logic that only validates rather than executing
            # This is because decide_action() should only return the action, not perform it
            
            # Sort actions by probability (descending)
            action_indices = np.argsort(-action_probs)
            
            # Check each action in order of probability
            for action_idx in action_indices:
                action = action_idx
                action_name = self.action_to_string(action)
                
                # Check if this action might succeed
                if self._is_action_likely_valid(action):
                    logger.debug(f"Selected action: {action} ({action_name}) - likely valid")
                    return action
            
            # If no action looks valid, take the highest probability one anyway
            action = action_indices[0]
            logger.debug(f"Selected action: {action} ({self.action_to_string(action)}) - highest probability")
            return action
            
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            # Fallback to random action if selection fails
            return np.random.randint(0, len(action_probs))
            
    def _is_action_likely_valid(self, action):
        """
        Check if an action is likely to be valid without actually performing it.
        
        Args:
            action: The action to check
            
        Returns:
            bool: Whether the action is likely to be valid
        """
        # Movement actions - check if destination is valid
        if action in [self.MOVE_NORTH, self.MOVE_SOUTH, self.MOVE_EAST, self.MOVE_WEST]:
            # Get movement vector
            delta_row, delta_col = self.DIRECTION_VECTORS.get(action, (0, 0))
            new_row = self.row + delta_row
            new_col = self.col + delta_col
            
            # Check if the move is valid (within bounds and not into water)
            if (0 <= new_row < self.environment.height and 
                0 <= new_col < self.environment.width and
                self.environment.grid[new_row, new_col] != GridWorld.WATER):
                return True
            return False
            
        # Eat action - check if there's a mature plant on the current cell
        elif action == self.EAT:
            return (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
                   self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_MATURE)
                   
        # Drink action - check if there's water nearby
        elif action == self.DRINK:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = self.row + dr, self.col + dc
                    if (0 <= r < self.environment.height and 
                        0 <= c < self.environment.width and
                        self.environment.grid[r, c] == GridWorld.WATER):
                        return True
            return False
            
        # Plant seed action - check if current cell is valid for planting
        elif action == self.PLANT_SEED:
            return (self.seeds > 0 and
                   self.environment.grid[self.row, self.col] == GridWorld.SOIL and
                   self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_NONE and
                   self.environment.soil_fertility[self.row, self.col] >= 3.0)
                   
        # Tend plant action - check if there's a plant to tend
        elif action == self.TEND_PLANT:
            return (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
                   (self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_SEED or
                    self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_GROWING))
                    
        # Harvest action - check if there's a mature plant
        elif action == self.HARVEST:
            return (self.environment.grid[self.row, self.col] == GridWorld.PLANT and
                   self.environment.plant_state[self.row, self.col] == GridWorld.PLANT_MATURE)
        
        # Default to True for unknown actions
        return True

    def action_to_string(self, action):
        """Convert action index to descriptive string."""
        action_names = {
            self.MOVE_NORTH: "Move North",
            self.MOVE_SOUTH: "Move South", 
            self.MOVE_EAST: "Move East",
            self.MOVE_WEST: "Move West",
            self.EAT: "Eat",
            self.DRINK: "Drink",
            self.PLANT_SEED: "Plant Seed",
            self.TEND_PLANT: "Tend Plant",
            self.HARVEST: "Harvest"
        }
        return action_names.get(action, f"Unknown action: {action}") 