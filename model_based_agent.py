import torch
import random
import numpy as np
from agent import Agent
from model import ObservationEncoder, AgentCNN
import torch.nn.functional as F
from environment import GridWorld

try:
    from train import PPOAgentCNN
    PPO_AVAILABLE = True
except ImportError:
    print("Warning: PPOAgentCNN not available. Will only use standard AgentCNN models.")
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
            self.model = PPOAgentCNN()
            self.is_ppo_model = True
            print("Using PPO model architecture")
        else:
            self.model = AgentCNN()
            self.is_ppo_model = False
            if 'ppo' in model_path and not PPO_AVAILABLE:
                print("Warning: PPO model requested but PPOAgentCNN not available. Using standard AgentCNN.")
            else:
                print("Using standard AgentCNN architecture")
        
        # Load trained weights
        try:
            print(f"Attempting to load model from: {model_path}")
            import os
            
            # Check if the file exists at the specified path
            if os.path.exists(model_path):
                print(f"File exists and has size: {os.path.getsize(model_path)} bytes")
            else:
                # If not, try looking for it in the models directory
                model_filename = os.path.basename(model_path)
                models_dir_path = os.path.join('models', model_filename)
                
                if os.path.exists(models_dir_path):
                    print(f"File found in models directory instead: {models_dir_path}")
                    model_path = models_dir_path
                    print(f"Using model from: {model_path}")
                else:
                    print(f"File does not exist at specified path: {model_path}")
                    print(f"File also not found in models directory: {models_dir_path}")
                    print(f"Current working directory: {os.getcwd()}")
                    
                    # List available model files in both current and models directory
                    print("Available model files in current directory:")
                    for file in os.listdir():
                        if file.endswith('.pth'):
                            print(f"  - {file} ({os.path.getsize(file)} bytes)")
                    
                    if os.path.exists('models'):
                        print("Available model files in models directory:")
                        for file in os.listdir('models'):
                            if file.endswith('.pth'):
                                print(f"  - models/{file} ({os.path.getsize(os.path.join('models', file))} bytes)")
                    else:
                        print("The 'models' directory does not exist yet.")
            
            model_state = torch.load(model_path)
            self.model.load_state_dict(model_state)
            self.model.eval()  # Set to evaluation mode
            
            # Debug info about model parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Successfully loaded model from {model_path}")
            print(f"Model has {total_params} parameters")
            
            # Print a parameter sample to verify non-zero weights
            for name, param in self.model.named_parameters():
                print(f"Parameter {name}: shape={param.shape}, sample={param.data.flatten()[:3]}")
                break  # Just print the first parameter
                
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
        
        # Print agent's current status for debugging
        print(f"DEBUG - Agent Environment:")
        self.environment.print_surrounding_area(self.row, self.col, 3)  # 3 cells in each direction = 7x7 grid
        print(f"DEBUG - Agent status: Health={self.health:.1f}, Hunger={self.hunger:.1f}, Thirst={self.thirst:.1f}, Energy={self.energy:.1f}, Seeds={self.seeds}")
        
        # Get action probabilities - handle different model types
        with torch.no_grad():
            if self.is_ppo_model:
                action_probs, state_value, _ = self.model(observation)
                action_probs = action_probs.squeeze(0)
                print(f"DEBUG - PPO Value prediction: {state_value.item():.3f}")
            else:
                action_probs = self.model(observation).squeeze(0)
        
        # Print action probabilities for debugging
        action_names = ["Move North", "Move South", "Move East", "Move West", 
                        "Eat", "Drink", "Plant Seed", "Tend Plant", "Harvest"]
        
        print("DEBUG - Action probabilities:")
        for i, (name, prob) in enumerate(zip(action_names, action_probs.tolist())):
            print(f"  {i}: {name} = {prob:.3f}")
        
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
        
        # Print chosen action
        print(f"DEBUG - Chosen action: {self.current_task}")
        
        # Execute the action
        result = self.step(action)
        
        # Print action result
        print(f"DEBUG - Action result: success={result['success']}, alive={result['alive']}\n")
        
        # Return result
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
        
        print(f"Input tensor shape: {x.shape}, mean: {x.mean().item():.4f}, std: {x.std().item():.4f}")
        print(f"Input tensor range: [{x.min().item():.4f}, {x.max().item():.4f}]")
        
        with torch.no_grad():
            if self.is_ppo_model:
                action_probs, _ = self.model(x)
                action_probs = action_probs.squeeze(0)
            else:
                action_probs = self.model(x).squeeze(0)
        
        # Check for NaN values
        if torch.isnan(action_probs).any():
            print("WARNING: NaN values detected in action probabilities!")
            # Replace NaNs with uniform distribution
            action_probs = torch.ones_like(action_probs) / action_probs.size(0)
        
        # Check for zero probabilities
        if (action_probs.sum() < 1e-10):
            print("WARNING: All action probabilities are near zero!")
            action_probs = torch.ones_like(action_probs) / action_probs.size(0)
        
        # Normalize probabilities
        action_probs = F.softmax(action_probs, dim=0)
        
        print(f"Output action probs: {action_probs.numpy()}")
        print(f"Sum of probabilities: {action_probs.sum().item():.6f}")
        
        return action_probs.numpy()
        
    def decide_action(self):
        """
        Decide the next action based on the neural network.
        
        Returns:
            int: The action to take
        """
        # Get the current observation
        observation = self.encoder.get_observation(self)
        
        # Get action probabilities
        action_probs = self.get_action_probs(observation)
        
        # Print current situation for debugging
        print(f"Current position: ({self.row}, {self.col})")
        print(f"Current task: {self.current_task}")
        print(f"Agent status - Health: {self.health:.1f}, Hunger: {self.hunger:.1f}, Thirst: {self.thirst:.1f}")
        
        # Sample an action from the probability distribution
        try:
            action = np.random.choice(len(action_probs), p=action_probs)
            print(f"Selected action: {action} ({self.action_to_string(action)})")
            return action
        except Exception as e:
            print(f"Error sampling action: {e}")
            # Fallback to random action if sampling fails
            return np.random.randint(0, len(action_probs))
            
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