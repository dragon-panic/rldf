"""
Interface file for RL models.
This file provides a unified interface for accessing both REINFORCE and PPO models.
"""

from model import ObservationEncoder, RLModelBase
from reinforce_model import REINFORCEModel
from ppo_model import PPOModel

# Export all the relevant classes for easy importing
__all__ = ['ObservationEncoder', 'RLModelBase', 'REINFORCEModel', 'PPOModel']


def create_model(model_type, **kwargs):
    """
    Factory function to create a model of the specified type.
    
    Args:
        model_type: 'reinforce' or 'ppo'
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        An instance of the specified model type
    """
    if model_type.lower() == 'reinforce':
        return REINFORCEModel(**kwargs)
    elif model_type.lower() == 'ppo':
        return PPOModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'reinforce' or 'ppo'.")


def load_model(model_type, model_path, **kwargs):
    """
    Factory function to create and load a model from a saved state.
    
    Args:
        model_type: 'reinforce' or 'ppo'
        model_path: Path to the saved model state
        **kwargs: Additional arguments to pass to the model constructor
        
    Returns:
        An instance of the specified model type with loaded weights
    """
    import torch
    
    # Create the model
    model = create_model(model_type, **kwargs)
    
    # Load the state
    model.load_state_dict(torch.load(model_path))
    
    # Set to evaluation mode
    model.eval()
    
    return model 