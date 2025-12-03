"""
Quick training script to create a minimal trained model for testing.
Trains PPO on CartPole for just 100 episodes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import yaml
from environments import EnvironmentWrapper
from algorithms.ppo import PPO

def main():
    print("Quick PPO Training (100 episodes for testing)")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "ppo_config.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    config = configs['CartPole-v1']
    
    # Override to fewer episodes
    config['episodes'] = 100
    
    # Create environment
    env = EnvironmentWrapper('CartPole-v1')
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    is_continuous = not env.is_discrete_action
    
    # Create agent
    agent = PPO(state_dim, action_dim, config, continuous=is_continuous)
    
    # Train
    print("\nTraining for 100 episodes (this will take 2-3 minutes)...")
    agent.train(env, num_episodes=100, logger=None)
    
    # Save
    model_path = Path(__file__).parent / "models" / "ppo_CartPole-v1.pth"
    agent.save(str(model_path))
    
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {model_path}")
    print(f"{'='*60}")
    print("\nYou can now:")
    print("  1. Test: python src/test.py --algorithm ppo --environment CartPole-v1")
    print("  2. Record: python src/record.py --algorithm ppo --environment CartPole-v1 --episodes 3")
    
    env.close()

if __name__ == "__main__":
    main()
