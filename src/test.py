"""
Testing script for trained RL agents.

This script loads trained models and evaluates them on test episodes,
collecting statistics about performance.
"""

import argparse
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import csv
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent))

from environments import EnvironmentWrapper
from algorithms import A2C, SAC, PPO
from utils.plotting import save_statistics_plot


def load_config(algorithm: str, environment: str) -> dict:
    """
    Load configuration for the specified algorithm and environment.
    
    Args:
        algorithm: Algorithm name (a2c, sac, ppo)
        environment: Environment name
        
    Returns:
        Configuration dictionary for the environment
    """
    config_path = Path(__file__).parent.parent / "configs" / f"{algorithm}_config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if environment not in configs:
        raise ValueError(
            f"Environment {environment} not found in {algorithm} config. "
            f"Available: {list(configs.keys())}"
        )
    
    return configs[environment]


def get_algorithm_class(algorithm: str):
    """
    Get the algorithm class based on the algorithm name.
    
    Args:
        algorithm: Algorithm name (a2c, sac, ppo)
        
    Returns:
        Algorithm class
    """
    algorithms = {
        'a2c': A2C,
        'sac': SAC,
        'ppo': PPO
    }
    
    if algorithm not in algorithms:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(algorithms.keys())}"
        )
    
    return algorithms[algorithm]


def compute_statistics(values: list) -> Dict[str, float]:
    """
    Compute statistics from a list of values.
    
    Args:
        values: List of numerical values
        
    Returns:
        Dictionary with mean, std, min, max
    """
    values_array = np.array(values)
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array))
    }


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(
        description="Test trained RL agents on Gymnasium environments"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        required=True,
        choices=['a2c', 'sac', 'ppo'],
        help='Algorithm used for training'
    )
    parser.add_argument(
        '--environment',
        type=str,
        required=True,
        choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1'],
        help='Gymnasium environment to test on'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to the trained model (default: models/{algorithm}_{environment}.pth)'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of test episodes (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render the environment during testing'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save test results and plots to results/ directory'
    )
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Determine model path
    if args.model_path is None:
        model_path = Path(__file__).parent.parent / "models" / f"{args.algorithm}_{args.environment}.pth"
    else:
        model_path = Path(args.model_path)
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    
    # Load configuration
    config = load_config(args.algorithm, args.environment)
    
    # Initialize environment
    render_mode = 'human' if args.render else None
    print(f"Initializing environment: {args.environment}")
    env = EnvironmentWrapper(args.environment, render_mode=render_mode)
    
    # Get state and action dimensions
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    is_continuous = not env.is_discrete_action
    
    # Initialize algorithm
    print(f"Initializing {args.algorithm.upper()} algorithm...")
    AlgorithmClass = get_algorithm_class(args.algorithm)
    
    # For PPO, pass continuous flag
    if args.algorithm == 'ppo':
        agent = AlgorithmClass(state_dim, action_dim, config, continuous=is_continuous)
    else:
        agent = AlgorithmClass(state_dim, action_dim, config)
    
    # Load trained model
    try:
        agent.load(str(model_path))
        print("Model loaded successfully!")
    except NotImplementedError:
        print(f"Error: load() method not implemented for {args.algorithm}")
        env.close()
        sys.exit(1)
    
    # Run test episodes
    print(f"\nRunning {args.num_episodes} test episodes...")
    episode_returns = []
    episode_durations = []
    
    try:
        test_stats = agent.test(env, num_episodes=args.num_episodes)
        
        # Display statistics
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        for key, value in test_stats.items():
            print(f"{key}: {value}")
        print("="*50)
        
        # Save results if requested
        if args.save_results:
            results_dir = Path(__file__).parent.parent / "results"
            results_dir.mkdir(exist_ok=True)
            
            # Save statistics to text file
            results_file = results_dir / f"{args.algorithm}_{args.environment}_test_results.txt"
            with open(results_file, 'w') as f:
                f.write(f"Test Results for {args.algorithm.upper()} on {args.environment}\n")
                f.write("="*50 + "\n")
                for key, value in test_stats.items():
                    if key not in ['episode_rewards', 'episode_durations']:
                        f.write(f"{key}: {value}\n")
            
            print(f"\nResults saved to: {results_file}")
            
            # Save detailed episode data to CSV
            csv_file = results_dir / f"{args.algorithm}_{args.environment}_test_episodes.csv"
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward', 'Duration'])
                for i, (reward, duration) in enumerate(zip(
                    test_stats.get('episode_rewards', []),
                    test_stats.get('episode_durations', [])
                ), 1):
                    writer.writerow([i, reward, duration])
            
            print(f"Episode data saved to: {csv_file}")
            
            # Generate and save plots if duration data is available
            if 'episode_durations' in test_stats:
                plot_path = results_dir / f"{args.algorithm}_{args.environment}_test_plot.png"
                save_statistics_plot(
                    test_stats['episode_durations'],
                    title=f"{args.algorithm.upper()} on {args.environment} - Test Episode Durations",
                    save_path=str(plot_path)
                )
                print(f"Plot saved to: {plot_path}")
        
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print(f"Please implement the test() method in {args.algorithm}.py")
        env.close()
        sys.exit(1)
    
    # Close environment
    env.close()
    print("\nTesting session finished!")


if __name__ == "__main__":
    main()
