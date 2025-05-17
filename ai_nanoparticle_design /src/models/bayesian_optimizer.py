import numpy as np
from typing import Dict, List, Tuple, Callable
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import gpytorch
import torch

class NanoparticleBayesianOptimizer:
    """
    Bayesian optimization for nanoparticle design using Gaussian Process regression.
    """
    
    def __init__(
        self,
        parameter_bounds: Dict[str, Tuple[float, float]],
        objective_function: Callable,
        initial_points: int = 5,
        exploration_weight: float = 0.1
    ):
        """
        Initialize the Bayesian optimizer for nanoparticle design.
        
        Args:
            parameter_bounds: Dictionary of parameter names and their bounds
            objective_function: Function to optimize (should return a scalar value)
            initial_points: Number of random points to evaluate before optimization
            exploration_weight: Trade-off between exploration and exploitation (0 to 1)
        """
        self.parameter_bounds = parameter_bounds
        self.objective_function = objective_function
        self.initial_points = initial_points
        
        # Convert bounds dictionary to format required by BayesianOptimization
        self.pbounds = {
            name: bounds for name, bounds in parameter_bounds.items()
        }
        
        # Initialize Bayesian optimization
        self.optimizer = BayesianOptimization(
            f=self.objective_function,
            pbounds=self.pbounds,
            random_state=42
        )
        
        # Set up the acquisition function
        self.utility = UtilityFunction(
            kind="ucb",
            kappa=exploration_weight,
            xi=0.0
        )
        
        self.optimization_history = []
    
    def optimize(self, n_iterations: int) -> Dict[str, float]:
        """
        Run the optimization process.
        
        Args:
            n_iterations: Number of optimization iterations
            
        Returns:
            Dictionary containing optimal parameters
        """
        # Initial random exploration
        self.optimizer.maximize(
            init_points=self.initial_points,
            n_iter=0
        )
        
        # Main optimization loop
        for i in range(n_iterations):
            # Suggest next point to evaluate
            next_point = self.optimizer.suggest(self.utility)
            
            # Evaluate objective function
            target = self.objective_function(**next_point)
            
            # Register the result
            self.optimizer.register(
                params=next_point,
                target=target
            )
            
            # Store optimization history
            self.optimization_history.append({
                'iteration': i,
                'parameters': next_point,
                'objective_value': target
            })
        
        return self.optimizer.max['params']
    
    def get_acquisition_surface(
        self,
        param1: str,
        param2: str,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate the acquisition function surface for visualization.
        
        Args:
            param1: Name of first parameter to visualize
            param2: Name of second parameter to visualize
            n_points: Number of points to evaluate in each dimension
            
        Returns:
            Tuple of (X, Y, Z) arrays for surface plotting
        """
        bounds1 = self.parameter_bounds[param1]
        bounds2 = self.parameter_bounds[param2]
        
        x = np.linspace(bounds1[0], bounds1[1], n_points)
        y = np.linspace(bounds2[0], bounds2[1], n_points)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i in range(n_points):
            for j in range(n_points):
                point = {
                    param1: X[i, j],
                    param2: Y[i, j]
                }
                # Fill in other parameters with their mean values
                for param, bounds in self.parameter_bounds.items():
                    if param not in [param1, param2]:
                        point[param] = np.mean(bounds)
                
                Z[i, j] = self.utility._ucb(
                    self.optimizer._gp,
                    np.array([list(point.values())]),
                    self.optimizer._space.target.max()
                )
        
        return X, Y, Z
    
    def get_optimization_history(self) -> List[Dict]:
        """Return the history of optimization iterations."""
        return self.optimization_history
    
    def predict_uncertainty(self, parameters: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict the mean and standard deviation at a given point using the GP model.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Tuple of (mean, std) predictions
        """
        x = np.array([list(parameters.values())])
        mean, std = self.optimizer._gp.predict(x, return_std=True)
        return float(mean[0]), float(std[0]) 