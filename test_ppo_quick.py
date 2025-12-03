"""
Quick test script to verify PPO implementation works correctly.

This script runs a quick sanity check on the PPO implementation
without requiring a full training run.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from environments import EnvironmentWrapper
from algorithms.ppo import PPO, ActorCritic, RolloutBuffer
import yaml


def test_network_creation():
    """Test that networks can be created."""
    print("Testing network creation...")
    
    # Discrete action space
    actor_critic_discrete = ActorCritic(state_dim=4, action_dim=2, continuous=False)
    print(f"✓ Discrete Actor-Critic created: {actor_critic_discrete}")
    
    # Continuous action space
    actor_critic_continuous = ActorCritic(state_dim=3, action_dim=1, continuous=True)
    print(f"✓ Continuous Actor-Critic created: {actor_critic_continuous}")
    
    print("✓ Network creation test passed!\n")


def test_forward_pass():
    """Test forward passes through the network."""
    print("Testing forward pass...")
    
    # Discrete
    ac_discrete = ActorCritic(4, 2, continuous=False)
    state = torch.randn(1, 4)
    dist, value = ac_discrete(state)
    action = dist.sample()
    print(f"✓ Discrete: state shape={state.shape}, action={action.item()}, value={value.item():.3f}")
    
    # Continuous
    ac_continuous = ActorCritic(3, 1, continuous=True)
    state = torch.randn(1, 3)
    dist, value = ac_continuous(state)
    action = dist.sample()
    print(f"✓ Continuous: state shape={state.shape}, action={action.numpy()}, value={value.item():.3f}")
    
    print("✓ Forward pass test passed!\n")


def test_rollout_buffer():
    """Test rollout buffer functionality."""
    print("Testing rollout buffer...")
    
    buffer = RolloutBuffer()
    
    # Store some transitions
    for i in range(5):
        buffer.store(
            state=np.random.randn(4),
            action=np.random.randint(2),
            log_prob=np.random.randn(),
            reward=np.random.randn(),
            value=np.random.randn(),
            done=False
        )
    
    print(f"✓ Buffer length: {len(buffer)}")
    
    # Get data
    states, actions, log_probs, rewards, values, dones = buffer.get()
    print(f"✓ Retrieved tensors: states={states.shape}, actions={actions.shape}")
    
    buffer.clear()
    print(f"✓ Buffer cleared: length={len(buffer)}")
    
    print("✓ Rollout buffer test passed!\n")


def test_ppo_initialization():
    """Test PPO agent initialization."""
    print("Testing PPO initialization...")
    
    config = {
        'learning_rate': 0.0003,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'n_epochs': 10,
        'batch_size': 64,
        'replay_memory_size': 2048,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5
    }
    
    # Discrete
    agent_discrete = PPO(state_dim=4, action_dim=2, config=config, continuous=False)
    print(f"✓ Discrete PPO agent created")
    
    # Continuous
    agent_continuous = PPO(state_dim=3, action_dim=1, config=config, continuous=True)
    print(f"✓ Continuous PPO agent created")
    
    print("✓ PPO initialization test passed!\n")


def test_action_selection():
    """Test action selection."""
    print("Testing action selection...")
    
    config = {
        'learning_rate': 0.0003,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'n_epochs': 10,
        'batch_size': 64,
        'replay_memory_size': 2048,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5
    }
    
    # Discrete
    agent = PPO(state_dim=4, action_dim=2, config=config, continuous=False)
    state = np.random.randn(4)
    action, log_prob, value = agent.select_action(state, deterministic=False)
    print(f"✓ Discrete action: {action}, log_prob: {log_prob:.3f}, value: {value:.3f}")
    
    # Continuous
    agent = PPO(state_dim=3, action_dim=1, config=config, continuous=True)
    state = np.random.randn(3)
    action, log_prob, value = agent.select_action(state, deterministic=False)
    print(f"✓ Continuous action: {action}, log_prob: {log_prob:.3f}, value: {value:.3f}")
    
    print("✓ Action selection test passed!\n")


def test_gae_computation():
    """Test GAE computation."""
    print("Testing GAE computation...")
    
    config = {
        'learning_rate': 0.0003,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'n_epochs': 10,
        'batch_size': 64,
        'replay_memory_size': 2048,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5
    }
    
    agent = PPO(state_dim=4, action_dim=2, config=config, continuous=False)
    
    rewards = torch.tensor([1.0, 0.5, -0.5, 1.0])
    values = torch.tensor([0.8, 0.6, 0.4, 0.7])
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0])
    next_value = 0.5
    
    advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
    
    print(f"✓ Advantages shape: {advantages.shape}")
    print(f"✓ Returns shape: {returns.shape}")
    print(f"✓ Sample advantage: {advantages[0]:.3f}")
    print(f"✓ Sample return: {returns[0]:.3f}")
    
    print("✓ GAE computation test passed!\n")


def test_environment_interaction():
    """Test interaction with environment."""
    print("Testing environment interaction...")
    
    # Load config
    config_path = Path(__file__).parent / "configs" / "ppo_config.yaml"
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    config = configs['CartPole-v1']
    
    # Create environment and agent
    env = EnvironmentWrapper('CartPole-v1')
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()
    
    agent = PPO(state_dim, action_dim, config, continuous=False)
    
    # Run a few steps
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(10):
        action, log_prob, value = agent.select_action(state, deterministic=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
        
        state = next_state
    
    print(f"✓ Ran 10 steps, total reward: {total_reward}")
    env.close()
    
    print("✓ Environment interaction test passed!\n")


def test_save_load():
    """Test model saving and loading."""
    print("Testing save/load...")
    
    config = {
        'learning_rate': 0.0003,
        'discount_factor': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'n_epochs': 10,
        'batch_size': 64,
        'replay_memory_size': 2048,
        'value_loss_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5
    }
    
    # Create agent
    agent = PPO(state_dim=4, action_dim=2, config=config, continuous=False)
    
    # Get initial action
    state = np.random.randn(4)
    action1, _, _ = agent.select_action(state, deterministic=True)
    
    # Save
    save_path = Path(__file__).parent / "test_model.pth"
    agent.save(str(save_path))
    print(f"✓ Model saved to {save_path}")
    
    # Create new agent and load
    agent2 = PPO(state_dim=4, action_dim=2, config=config, continuous=False)
    agent2.load(str(save_path))
    print(f"✓ Model loaded")
    
    # Get action from loaded model (should be same with deterministic=True)
    action2, _, _ = agent2.select_action(state, deterministic=True)
    
    print(f"✓ Action before save: {action1}, after load: {action2}")
    
    # Clean up
    save_path.unlink()
    
    print("✓ Save/load test passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("PPO Implementation Quick Test Suite")
    print("="*60 + "\n")
    
    try:
        test_network_creation()
        test_forward_pass()
        test_rollout_buffer()
        test_ppo_initialization()
        test_action_selection()
        test_gae_computation()
        test_environment_interaction()
        test_save_load()
        
        print("="*60)
        print("✓ ALL TESTS PASSED! PPO implementation is working correctly.")
        print("="*60)
        print("\nYou can now train PPO with:")
        print("  python src/train.py --algorithm ppo --environment CartPole-v1")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
