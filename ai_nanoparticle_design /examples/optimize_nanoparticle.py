import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.rl_agent import NanoparticleRLAgent
from src.utils.visualization import NanoparticleVisualizer

def main():
    # Define target properties for the nanoparticle
    target_properties = {
        'size': 100.0,  # nm
        'zeta_potential': -30.0,  # mV
        'release_rate': 0.5  # hr^-1
    }
    
    # Define parameter bounds (optional)
    bounds = {
        'size': (10.0, 500.0),
        'zeta_potential': (-50.0, 50.0),
        'release_rate': (0.1, 2.0)
    }
    
    # Initialize the RL agent
    agent = NanoparticleRLAgent(
        target_properties=target_properties,
        bounds=bounds,
        learning_rate=3e-4,
        batch_size=256
    )
    
    # Train the agent
    print("Starting optimization process...")
    history = agent.train(
        total_timesteps=10000,
        bayesian_iterations=20
    )
    
    # Visualize results
    visualizer = NanoparticleVisualizer()
    
    # Plot optimization history
    visualizer.plot_optimization_history(
        history,
        save_path='optimization_history.png'
    )
    
    # Generate predictions for analysis
    predictions = []
    for _ in range(100):
        state = agent.env.reset()
        action, action_info = agent.predict(state)
        
        prediction = {
            'size': action[0],
            'zeta_potential': action[1],
            'release_rate': action[2]
        }
        predictions.append(prediction)
    
    # Plot property distributions
    visualizer.plot_property_distributions(
        predictions,
        target_properties,
        save_path='property_distributions.png'
    )
    
    # Plot convergence analysis
    visualizer.plot_convergence_analysis(
        history,
        window_size=50,
        save_path='convergence_analysis.png'
    )
    
    # Save the trained agent
    agent.save('nanoparticle_agent')
    print("Optimization complete. Results saved to current directory.")

if __name__ == "__main__":
    main() 