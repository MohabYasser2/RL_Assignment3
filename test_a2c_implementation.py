"""
Test script for A2C implementation.
This script tests the A2C algorithm on CartPole-v1 environment.
"""

import gymnasium as gym
import yaml
from src.algorithms.a2c import A2C

def test_a2c():
    """Test A2C implementation on CartPole."""
    
    # Load configuration
    with open('configs/a2c_config.yaml', 'r') as f:
        configs = yaml.safe_load(f)
    
    env_name = 'CartPole-v1'
    config = configs[env_name]
    
    # Create environment
    env = gym.make(env_name)
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Configuration: {config}")
    print("\n" + "="*50 + "\n")
    
    # Initialize A2C agent
    agent = A2C(state_dim, action_dim, config)
    
    print("Testing basic functionality...")
    
    # Test action selection
    state, _ = env.reset()
    action = agent.select_action(state, deterministic=False)
    print(f"✓ Stochastic action selection works: {action}")
    
    action = agent.select_action(state, deterministic=True)
    print(f"✓ Deterministic action selection works: {action}")
    
    # Test short training run
    print("\n" + "="*50)
    print("Running short training (10 episodes)...\n")
    
    results = agent.train(env, num_episodes=10)
    
    print("\n" + "="*50)
    print("Training Results:")
    print(f"  Mean Reward: {results['mean_reward']:.2f}")
    print(f"  Std Reward: {results['std_reward']:.2f}")
    print(f"  Total Episodes: {len(results['episode_rewards'])}")
    
    # Test save/load
    print("\n" + "="*50)
    print("Testing save/load functionality...")
    
    model_path = "models/a2c_test.pth"
    agent.save(model_path)
    
    # Create new agent and load
    agent2 = A2C(state_dim, action_dim, config)
    agent2.load(model_path)
    print("✓ Save/Load works correctly")
    
    # Test the loaded model
    print("\n" + "="*50)
    print("Testing loaded model (5 episodes)...\n")
    
    test_results = agent2.test(env, num_episodes=5)
    print(f"Test Mean Reward: {test_results['mean_reward']:.2f}")
    print(f"Test Std Reward: {test_results['std_reward']:.2f}")
    
    env.close()
    
    print("\n" + "="*50)
    print("✓ All tests passed successfully!")
    print("="*50)

if __name__ == "__main__":
    test_a2c()
