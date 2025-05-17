import gym
import numpy as np
from gym import spaces
from typing import Dict, Tuple, Any

class NanoparticleEnvironment(gym.Env):
    """
    Custom Environment for nanoparticle design optimization that follows gym interface.
    This environment simulates the process of designing nanoparticles with specific properties.
    """
    
    def __init__(self, target_properties: Dict[str, float], bounds: Dict[str, Tuple[float, float]] = None):
        super(NanoparticleEnvironment, self).__init__()
        
        self.target_properties = target_properties
        
        # Default bounds if none provided
        self.bounds = bounds or {
            'size': (10, 500),  # nm
            'zeta_potential': (-50, 50),  # mV
            'release_rate': (0.1, 2.0)  # hr^-1
        }
        
        # Define action space (continuous values for each parameter)
        self.action_space = spaces.Box(
            low=np.array([b[0] for b in self.bounds.values()]),
            high=np.array([b[1] for b in self.bounds.values()]),
            dtype=np.float32
        )
        
        # Define observation space (includes current state and target properties)
        self.observation_space = spaces.Box(
            low=np.array([b[0] for b in self.bounds.values()] * 2),  # Current + Target
            high=np.array([b[1] for b in self.bounds.values()] * 2),
            dtype=np.float32
        )
        
        self.current_state = None
        self.steps = 0
        self.max_steps = 100
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.steps = 0
        self.current_state = np.array([
            self.bounds['size'][0],
            self.bounds['zeta_potential'][0],
            self.bounds['release_rate'][0]
        ])
        
        # Observation includes both current state and target properties
        observation = np.concatenate([
            self.current_state,
            [
                self.target_properties['size'],
                self.target_properties['zeta_potential'],
                self.target_properties['release_rate']
            ]
        ])
        
        return observation
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment."""
        self.steps += 1
        
        # Update current state based on action
        self.current_state = np.clip(
            action,
            [b[0] for b in self.bounds.values()],
            [b[1] for b in self.bounds.values()]
        )
        
        # Calculate reward based on distance to target properties
        reward = self._calculate_reward()
        
        # Check if episode should end
        done = self.steps >= self.max_steps
        
        # Prepare observation
        observation = np.concatenate([
            self.current_state,
            [
                self.target_properties['size'],
                self.target_properties['zeta_potential'],
                self.target_properties['release_rate']
            ]
        ])
        
        info = {
            'distance_to_target': self._calculate_distance_to_target(),
            'steps': self.steps
        }
        
        return observation, reward, done, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on how close the current state is to target properties."""
        distance = self._calculate_distance_to_target()
        
        # Convert distance to reward (negative distance with scaling)
        reward = -distance * 0.1
        
        # Bonus reward for being very close to target
        if distance < 0.1:
            reward += 10.0
            
        return reward
    
    def _calculate_distance_to_target(self) -> float:
        """Calculate normalized distance between current state and target properties."""
        # Normalize each dimension
        size_diff = (self.current_state[0] - self.target_properties['size']) / (self.bounds['size'][1] - self.bounds['size'][0])
        zeta_diff = (self.current_state[1] - self.target_properties['zeta_potential']) / (self.bounds['zeta_potential'][1] - self.bounds['zeta_potential'][0])
        release_diff = (self.current_state[2] - self.target_properties['release_rate']) / (self.bounds['release_rate'][1] - self.bounds['release_rate'][0])
        
        # Calculate Euclidean distance
        distance = np.sqrt(size_diff**2 + zeta_diff**2 + release_diff**2)
        
        return distance
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            print(f"\nCurrent State:")
            print(f"Size: {self.current_state[0]:.2f} nm")
            print(f"Zeta Potential: {self.current_state[1]:.2f} mV")
            print(f"Release Rate: {self.current_state[2]:.2f} hr^-1")
            print(f"\nTarget Properties:")
            print(f"Size: {self.target_properties['size']:.2f} nm")
            print(f"Zeta Potential: {self.target_properties['zeta_potential']:.2f} mV")
            print(f"Release Rate: {self.target_properties['release_rate']:.2f} hr^-1") 