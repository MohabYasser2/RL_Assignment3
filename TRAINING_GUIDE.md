# PPO Training Guide

This guide will help you train and evaluate the PPO agent on different environments.

## Quick Start (CartPole-v1)

CartPole is the easiest environment and the best place to start:

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run Quick Test (Optional but Recommended)

```bash
python test_ppo_quick.py
```

This will verify that the PPO implementation is working correctly. You should see all tests pass.

### Step 3: Train PPO on CartPole

```bash
python src/train.py --algorithm ppo --environment CartPole-v1 --no-wandb
```

**What to expect:**
- Training should complete in 5-10 minutes on CPU
- The agent should reach rewards ~200+ within 200-300 episodes
- Progress bar will show current reward and average reward
- Model will be saved to `models/ppo_CartPole-v1.pth`

**Output example:**
```
Training PPO: 100%|████████████| 500/500 [05:23<00:00, reward=245.32, avg_reward=198.45]
Training completed!
Mean reward (last 100 episodes): 198.45
```

### Step 4: Test the Trained Agent

```bash
python src/test.py --algorithm ppo --environment CartPole-v1 --save-results
```

**What happens:**
- Runs 100 test episodes
- Displays statistics (mean, std, min, max)
- Saves results to `results/ppo_CartPole-v1_test_results.txt`
- Saves episode data to CSV
- Generates performance plots

### Step 5: Record Videos

```bash
python src/record.py --algorithm ppo --environment CartPole-v1 --episodes 5
```

Videos will be saved to `videos/` directory.

## Training on Other Environments

### Acrobot-v1 (Moderate Difficulty)

```bash
# Train
python src/train.py --algorithm ppo --environment Acrobot-v1 --no-wandb

# Test
python src/test.py --algorithm ppo --environment Acrobot-v1 --save-results
```

**Expected performance:**
- Training time: 15-20 minutes
- Solved at: Average reward ≥ -100
- Episodes needed: ~800-1000

### MountainCar-v0 (Challenging - Sparse Rewards)

```bash
# Train
python src/train.py --algorithm ppo --environment MountainCar-v0 --no-wandb

# Test
python src/test.py --algorithm ppo --environment MountainCar-v0 --save-results
```

**Expected performance:**
- Training time: 20-30 minutes
- Solved at: Average reward ≥ -110
- Episodes needed: ~1500-2000
- Note: This environment is harder due to sparse rewards

### Pendulum-v1 (Continuous Control)

```bash
# Train
python src/train.py --algorithm ppo --environment Pendulum-v1 --no-wandb

# Test
python src/test.py --algorithm ppo --environment Pendulum-v1 --save-results
```

**Expected performance:**
- Training time: 10-15 minutes
- Solved at: Average reward ≥ -200
- Episodes needed: ~400-500
- Note: First continuous action space environment

## Using Weights & Biases

To enable experiment tracking with W&B:

### 1. Login to W&B

```bash
wandb login
```

### 2. Train with W&B enabled (remove --no-wandb flag)

```bash
python src/train.py --algorithm ppo --environment CartPole-v1
```

### 3. View your experiments

Go to https://wandb.ai and view your runs with detailed metrics:
- Episode rewards
- Episode lengths
- Actor loss
- Critic loss
- Entropy
- Average reward

## Understanding the Training Output

During training, you'll see a progress bar like this:

```
Training PPO: 45%|████▌     | 225/500 [02:45<03:22, reward=145.32, avg_reward=142.18, length=145]
```

**What each metric means:**
- **reward**: Reward from the most recent episode
- **avg_reward**: Average reward over last 100 episodes (this is what matters!)
- **length**: Duration of the most recent episode

**When is the agent "solved"?**
- CartPole: avg_reward ≥ 195
- Acrobot: avg_reward ≥ -100
- MountainCar: avg_reward ≥ -110
- Pendulum: avg_reward ≥ -200

## Troubleshooting

### Training is slow
- Use GPU if available (automatic detection)
- Reduce trajectory length in config
- Train for fewer episodes initially

### Agent not learning
- Check that hyperparameters match your environment in config file
- Try increasing number of training episodes
- Monitor entropy (should be > 0 for exploration)

### Out of memory
- Reduce batch_size in config file
- Reduce trajectory_length (replay_memory_size)

### Import errors
- Make sure you're in the project root directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`

## Configuration Tuning

Edit `configs/ppo_config.yaml` to adjust hyperparameters:

**Key parameters:**
- `learning_rate`: Step size for optimization (default: 3e-4)
- `clip_range`: PPO clipping parameter (default: 0.2)
- `entropy_coef`: Exploration bonus (higher = more exploration)
- `n_epochs`: Training epochs per update (default: 10)
- `batch_size`: Minibatch size (default: 64)

## Next Steps

1. **Train on all environments** to compare PPO performance
2. **Implement A2C and SAC** using PPO as a template
3. **Compare algorithms** on the same environment
4. **Tune hyperparameters** for better performance
5. **Try other Gymnasium environments** (LunarLander, BipedalWalker, etc.)

## Tips for Best Results

1. **Start with CartPole** - It's fast and you'll see results quickly
2. **Monitor avg_reward** - This is more important than individual episode rewards
3. **Be patient** - Some environments need 1000+ episodes
4. **Save your models** - Models are automatically saved after training
5. **Use W&B** - Great for comparing different runs

## Getting Help

If you encounter issues:
1. Run the quick test: `python test_ppo_quick.py`
2. Check that all files are in the correct locations
3. Verify dependencies are installed
4. Check the error messages carefully

Happy training! 🚀
