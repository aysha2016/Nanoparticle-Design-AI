import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

class NanoparticleVisualizer:
    """Utility class for visualizing nanoparticle optimization results."""
    
    @staticmethod
    def plot_optimization_history(
        history: Dict[str, List],
        save_path: str = None
    ):
        """
        Plot the optimization history including rewards and distances.
        
        Args:
            history: Dictionary containing RL and Bayesian optimization history
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot RL training history
        rl_data = pd.DataFrame(history['rl_history'])
        ax1.plot(rl_data['step'], rl_data['reward'], label='RL Reward')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reinforcement Learning Training Progress')
        ax1.legend()
        
        # Plot Bayesian optimization history
        bo_data = pd.DataFrame(history['bayesian_history'])
        ax2.plot(bo_data['iteration'], bo_data['objective_value'], label='BO Objective')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Bayesian Optimization Progress')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_parameter_surface(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        param1_name: str,
        param2_name: str,
        save_path: str = None
    ):
        """
        Plot the acquisition function surface for two parameters.
        
        Args:
            X, Y, Z: Meshgrid arrays for surface plotting
            param1_name: Name of first parameter
            param2_name: Name of second parameter
            save_path: Optional path to save the plot
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surface = ax.plot_surface(
            X, Y, Z,
            cmap='viridis',
            alpha=0.8
        )
        
        ax.set_xlabel(param1_name)
        ax.set_ylabel(param2_name)
        ax.set_zlabel('Acquisition Value')
        ax.set_title('Bayesian Optimization Acquisition Surface')
        
        plt.colorbar(surface)
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_property_distributions(
        predictions: List[Dict[str, float]],
        target_properties: Dict[str, float],
        save_path: str = None
    ):
        """
        Plot distributions of predicted nanoparticle properties.
        
        Args:
            predictions: List of dictionaries containing predicted properties
            target_properties: Dictionary of target property values
            save_path: Optional path to save the plot
        """
        properties = list(target_properties.keys())
        n_properties = len(properties)
        
        fig, axes = plt.subplots(1, n_properties, figsize=(5*n_properties, 4))
        
        for i, prop in enumerate(properties):
            values = [p[prop] for p in predictions]
            
            sns.histplot(values, ax=axes[i], kde=True)
            axes[i].axvline(
                target_properties[prop],
                color='r',
                linestyle='--',
                label='Target'
            )
            
            axes[i].set_title(f'{prop} Distribution')
            axes[i].set_xlabel(prop)
            axes[i].set_ylabel('Count')
            axes[i].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_uncertainty_map(
        parameter_values: Dict[str, List[float]],
        uncertainties: List[float],
        param1: str,
        param2: str,
        save_path: str = None
    ):
        """
        Plot uncertainty map for two parameters.
        
        Args:
            parameter_values: Dictionary of parameter values
            uncertainties: List of uncertainty values
            param1, param2: Names of parameters to plot
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            parameter_values[param1],
            parameter_values[param2],
            c=uncertainties,
            cmap='viridis',
            alpha=0.6
        )
        
        plt.colorbar(scatter, label='Prediction Uncertainty')
        plt.xlabel(param1)
        plt.ylabel(param2)
        plt.title('Prediction Uncertainty Map')
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_convergence_analysis(
        history: Dict[str, List],
        window_size: int = 50,
        save_path: str = None
    ):
        """
        Plot convergence analysis of the optimization process.
        
        Args:
            history: Dictionary containing optimization history
            window_size: Window size for moving average
            save_path: Optional path to save the plot
        """
        rl_data = pd.DataFrame(history['rl_history'])
        
        # Calculate moving averages
        reward_ma = rl_data['reward'].rolling(window=window_size).mean()
        distance_ma = rl_data['distance_to_target'].rolling(window=window_size).mean()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot reward convergence
        ax1.plot(rl_data['step'], reward_ma, label='Moving Average')
        ax1.plot(rl_data['step'], rl_data['reward'], alpha=0.2, label='Raw')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Reward')
        ax1.set_title('Reward Convergence Analysis')
        ax1.legend()
        
        # Plot distance convergence
        ax2.plot(rl_data['step'], distance_ma, label='Moving Average')
        ax2.plot(rl_data['step'], rl_data['distance_to_target'], alpha=0.2, label='Raw')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Distance to Target')
        ax2.set_title('Distance Convergence Analysis')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show() 