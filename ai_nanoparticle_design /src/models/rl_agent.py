import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from ..environment.nanoparticle_env import NanoparticleEnvironment
from .bayesian_optimizer import NanoparticleBayesianOptimizer

class OptimizationCallback(BaseCallback):
    """Custom callback for logging optimization progress."""
    
    def __init__(self, verbose=0):
        super(OptimizationCallback, self).__init__(verbose)
        self.training_history = []
    
    def _on_step(self) -> bool:
        """
        Log information for each optimization step.
        """
        info = self.locals['infos'][0]
        self.training_history.append({
            'step': self.num_timesteps,
            'reward': self.locals['rewards'][0],
            'distance_to_target': info['distance_to_target']
        })
        return True

class NanoparticleRLAgent:
    """
    Reinforcement learning agent for nanoparticle design optimization.
    Combines SAC (Soft Actor-Critic) with Bayesian optimization for efficient exploration.
    """
    
    def __init__(
        self,
        target_properties: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]] = None,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        buffer_size: int = 1000000
    ):
        """
        Initialize the RL agent.
        
        Args:
            target_properties: Dictionary of target nanoparticle properties
            bounds: Dictionary of parameter bounds (optional)
            learning_rate: Learning rate for the SAC algorithm
            batch_size: Batch size for training
            buffer_size: Size of the replay buffer
        """
        # Initialize the environment
        self.env = NanoparticleEnvironment(
            target_properties=target_properties,
            bounds=bounds
        )
        
        # Initialize the SAC agent
        self.agent = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=buffer_size,
            verbose=1
        )
        
        # Initialize Bayesian optimizer
        self.bayesian_optimizer = NanoparticleBayesianOptimizer(
            parameter_bounds=self.env.bounds,
            objective_function=self._objective_function
        )
        
        self.callback = OptimizationCallback()
        
    def _objective_function(self, **kwargs) -> float:
        """
        Objective function for Bayesian optimization.
        Maps parameter values to a reward score.
        """
        # Set environment state to the suggested parameters
        obs = self.env.reset()
        action = np.array(list(kwargs.values()))
        
        # Take action and get reward
        _, reward, _, _ = self.env.step(action)
        
        return reward
    
    def train(
        self,
        total_timesteps: int,
        bayesian_iterations: int = 10
    ) -> Dict[str, List]:
        """
        Train the agent using both RL and Bayesian optimization.
        
        Args:
            total_timesteps: Number of timesteps for RL training
            bayesian_iterations: Number of Bayesian optimization iterations
            
        Returns:
            Dictionary containing training history
        """
        # First, run Bayesian optimization to find promising regions
        print("Running Bayesian optimization...")
        optimal_params = self.bayesian_optimizer.optimize(bayesian_iterations)
        
        # Use the optimal parameters from Bayesian optimization to guide initial exploration
        self.env.reset()
        self.env.current_state = np.array(list(optimal_params.values()))
        
        # Train the RL agent
        print("Training RL agent...")
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=self.callback
        )
        
        return {
            'rl_history': self.callback.training_history,
            'bayesian_history': self.bayesian_optimizer.get_optimization_history()
        }
    
    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict the optimal action for a given state.
        
        Args:
            state: Current state observation
            
        Returns:
            Tuple of (action, action_info)
        """
        action, _ = self.agent.predict(state, deterministic=True)
        
        # Get uncertainty estimates from Bayesian optimizer
        parameters = {
            name: value for name, value in zip(self.env.bounds.keys(), action)
        }
        mean, std = self.bayesian_optimizer.predict_uncertainty(parameters)
        
        action_info = {
            'mean_prediction': mean,
            'uncertainty': std
        }
        
        return action, action_info
    
    def save(self, path: str):
        """Save the agent's model and optimization history."""
        self.agent.save(f"{path}_rl")
        torch.save({
            'bayesian_history': self.bayesian_optimizer.get_optimization_history(),
            'rl_history': self.callback.training_history
        }, f"{path}_history")
    
    def load(self, path: str):
        """Load a saved agent model and optimization history."""
        self.agent = SAC.load(f"{path}_rl")
        checkpoint = torch.load(f"{path}_history")
        self.callback.training_history = checkpoint['rl_history']
        self.bayesian_optimizer.optimization_history = checkpoint['bayesian_history'] 