# A2C Implementation Guide

## Overview

This document explains the step-by-step implementation of the A2C (Advantage Actor-Critic) algorithm based on the assignment requirements.

## Algorithm Components

### 1. Actor-Critic Network Architecture

The `ActorCritic` class implements a neural network with shared feature extraction:

```
Input State → Shared Layers → Actor Head (Policy)
                            → Critic Head (Value Function)
```

**Key Features:**

- **Shared Layers**: Two hidden layers (128 units each) with ReLU activation
- **Actor Head**:
  - Discrete actions: Outputs action logits
  - Continuous actions: Outputs mean and log_std
- **Critic Head**: Outputs state value V(s)

**Why share layers?**

- Reduces parameters
- Shared representations benefit both policy and value learning
- Standard practice in A2C/A3C algorithms

### 2. Action Selection

**Stochastic Mode (Training):**

- Discrete: Sample from Categorical distribution
- Continuous: Sample from Normal distribution

**Deterministic Mode (Testing):**

- Discrete: Choose argmax of action logits
- Continuous: Use mean of distribution

### 3. Advantage Calculation

A2C uses **n-step returns** and **Generalized Advantage Estimation (GAE)**:

```
TD Error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
Advantage: A_t = δ_t + γ * A_{t+1}
Return: R_t = A_t + V(s_t)
```

**Key Parameters:**

- `gamma` (γ): Discount factor (default: 0.99)
- `n_steps`: Number of steps for bootstrapping (default: 5)

### 4. Loss Function

The total loss combines three components:

```
Total Loss = Policy Loss + β₁ * Value Loss + β₂ * Entropy Loss
```

**Components:**

1. **Policy Loss** (Actor):

   ```
   L_policy = -mean(log π(a|s) * A(s,a))
   ```

   - Maximizes expected return
   - Weighted by advantage (reduces variance)

2. **Value Loss** (Critic):

   ```
   L_value = MSE(V(s), R)
   ```

   - Mean squared error between predicted value and return
   - Coefficient: `value_loss_coef` (default: 0.5)

3. **Entropy Loss**:
   ```
   L_entropy = -mean(H[π(·|s)])
   ```
   - Encourages exploration
   - Coefficient: `entropy_coef` (default: 0.01)

### 5. Training Loop

**Algorithm Flow:**

```
For each episode:
    Reset environment
    While not done:
        1. Select action using current policy
        2. Execute action in environment
        3. Store transition (s, a, r, s', done)

        If batch_size == n_steps OR done:
            4. Compute returns and advantages
            5. Compute loss
            6. Backpropagate and update networks
            7. Clip gradients (max_grad_norm = 0.5)
            8. Reset batch
```

**Key Features:**

- **N-step updates**: Updates every `n_steps` or at episode end
- **Gradient clipping**: Prevents exploding gradients
- **Advantage normalization**: Stabilizes training

### 6. Hyperparameters

From `configs/a2c_config.yaml`:

| Parameter       | CartPole | Acrobot | MountainCar | Pendulum |
| --------------- | -------- | ------- | ----------- | -------- |
| Learning Rate   | 0.0007   | 0.0005  | 0.0005      | 0.0003   |
| Discount (γ)    | 0.99     | 0.99    | 0.99        | 0.99     |
| N-steps         | 5        | 5       | 10          | 5        |
| Entropy Coef    | 0.01     | 0.01    | 0.05        | 0.01     |
| Value Loss Coef | 0.5      | 0.5     | 0.5         | 0.5      |
| Max Grad Norm   | 0.5      | 0.5     | 0.5         | 0.5      |
| Episodes        | 1000     | 2000    | 3000        | 1500     |

## Implementation Steps Summary

### Step 1: Network Architecture ✓

- Created `ActorCritic` class with shared layers
- Implemented forward pass for both discrete and continuous actions
- Added `get_action_and_value()` method for sampling

### Step 2: Action Selection ✓

- Implemented `select_action()` for both modes
- Handles discrete and continuous action spaces

### Step 3: Advantage Computation ✓

- Implemented `compute_returns_and_advantages()`
- Uses n-step returns with GAE
- Handles episode termination correctly

### Step 4: Loss Computation ✓

- Implemented `compute_loss()` method
- Combines policy, value, and entropy losses
- Normalizes advantages for stability

### Step 5: Training Loop ✓

- Implemented complete `train()` method
- Collects n-step trajectories
- Updates policy and value function
- Logs progress and metrics

### Step 6: Testing ✓

- Implemented `test()` method
- Uses deterministic action selection
- Computes comprehensive statistics

### Step 7: Save/Load ✓

- Implemented model persistence
- Saves both networks and optimizer state

## Usage Example

```python
import gymnasium as gym
import yaml
from src.algorithms.a2c import A2C

# Load configuration
with open('configs/a2c_config.yaml', 'r') as f:
    configs = yaml.safe_load(f)

# Create environment
env = gym.make('CartPole-v1')

# Initialize agent
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
config = configs['CartPole-v1']

agent = A2C(state_dim, action_dim, config)

# Train
results = agent.train(env, num_episodes=1000)

# Test
test_results = agent.test(env, num_episodes=100)

# Save model
agent.save('models/a2c_cartpole.pth')
```

## Testing the Implementation

Run the test script:

```bash
python test_a2c_implementation.py
```

This will:

1. Test action selection (stochastic and deterministic)
2. Run a short training session
3. Test save/load functionality
4. Verify the loaded model works

## Key Differences from Other Algorithms

**A2C vs A3C:**

- A2C: Synchronous updates (wait for all workers)
- A3C: Asynchronous updates (workers update independently)

**A2C vs PPO:**

- A2C: Direct policy gradient with advantage
- PPO: Clipped surrogate objective with trust region

**A2C vs SAC:**

- A2C: On-policy, works for discrete and continuous
- SAC: Off-policy, continuous actions only, uses replay buffer

## Common Issues and Solutions

1. **Training instability**: Adjust learning rate or increase `max_grad_norm`
2. **Slow learning**: Increase `n_steps` or adjust entropy coefficient
3. **Poor exploration**: Increase `entropy_coef`
4. **Value function not learning**: Adjust `value_loss_coef`

## References

- Original A3C Paper: "Asynchronous Methods for Deep Reinforcement Learning" (Mnih et al., 2016)
- GAE Paper: "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2016)
