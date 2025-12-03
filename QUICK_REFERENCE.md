# PPO Quick Reference

## Installation
```bash
pip install -r requirements.txt
```

## Verify Installation
```bash
python test_ppo_quick.py
```

## Quick Training (100 Episodes)
```bash
# Fast training for testing (2-3 minutes)
python train_quick.py
```

## Training Commands

### CartPole-v1 (Easiest - Start Here!)
```bash
python src/train.py --algorithm ppo --environment CartPole-v1 --no-wandb
```

### Acrobot-v1
```bash
python src/train.py --algorithm ppo --environment Acrobot-v1 --no-wandb
```

### MountainCar-v0
```bash
python src/train.py --algorithm ppo --environment MountainCar-v0 --no-wandb
```

### Pendulum-v1 (Continuous Actions)
```bash
python src/train.py --algorithm ppo --environment Pendulum-v1 --no-wandb
```

### With Weights & Biases
```bash
# First login
wandb login

# Then train (remove --no-wandb flag)
python src/train.py --algorithm ppo --environment CartPole-v1
```

## Testing Commands

### Test Trained Model
```bash
python src/test.py --algorithm ppo --environment CartPole-v1 --save-results
```

### Test with Custom Model Path
```bash
python src/test.py --algorithm ppo --environment CartPole-v1 --model-path models/my_model.pth --save-results
```

### Test with Rendering
```bash
python src/test.py --algorithm ppo --environment CartPole-v1 --render
```

## Recording Commands

### Record 5 Episodes
```bash
python src/record.py --algorithm ppo --environment CartPole-v1 --episodes 5
```

### Record with Custom Model
```bash
python src/record.py --algorithm ppo --environment CartPole-v1 --model-path models/my_model.pth --episodes 10
```

## File Locations

### Models
```
models/ppo_CartPole-v1.pth
models/ppo_Acrobot-v1.pth
models/ppo_MountainCar-v0.pth
models/ppo_Pendulum-v1.pth
```

### Test Results
```
results/ppo_CartPole-v1_test_results.txt      # Statistics
results/ppo_CartPole-v1_test_episodes.csv     # Episode data
results/ppo_CartPole-v1_test_plot.png         # Plots
```

### Videos
```
videos/ppo_CartPole-v1-episode-0.mp4
videos/ppo_CartPole-v1-episode-1.mp4
...
```

## Configuration Files

### Edit Hyperparameters
```
configs/ppo_config.yaml
```

Key parameters:
- `learning_rate`: 0.0003
- `clip_range`: 0.2
- `entropy_coef`: 0.01
- `n_epochs`: 10
- `batch_size`: 64

## Common Workflows

### Quick Test Run
```bash
# 1. Verify installation
python test_ppo_quick.py

# 2. Train for 50 episodes (quick test)
python src/train.py --algorithm ppo --environment CartPole-v1 --no-wandb
# Then manually stop with Ctrl+C after ~50 episodes

# 3. Test the model
python src/test.py --algorithm ppo --environment CartPole-v1
```

### Full Training Run
```bash
# 1. Train to completion
python src/train.py --algorithm ppo --environment CartPole-v1 --no-wandb

# 2. Test thoroughly
python src/test.py --algorithm ppo --environment CartPole-v1 --save-results --num-episodes 100

# 3. Record videos
python src/record.py --algorithm ppo --environment CartPole-v1 --episodes 5
```

### With Experiment Tracking
```bash
# 1. Login to W&B
wandb login

# 2. Train with tracking
python src/train.py --algorithm ppo --environment CartPole-v1

# 3. View results at wandb.ai
```

## Troubleshooting

### "Import torch could not be resolved"
```bash
pip install torch
```

### "Import gymnasium could not be resolved"
```bash
pip install gymnasium gymnasium[box2d]
```

### Training too slow
- Use GPU if available (automatic)
- Reduce episodes in config file
- Reduce trajectory_length in config

### Agent not learning
- Train for more episodes
- Check hyperparameters in config
- Try different environment

### Out of memory
- Reduce batch_size in config
- Reduce replay_memory_size in config
- Use CPU instead of GPU

## Performance Benchmarks

| Environment | Target Avg Reward | Expected Episodes | Time (CPU) |
|-------------|-------------------|-------------------|------------|
| CartPole-v1 | ≥ 195 | 200-300 | 5-10 min |
| Acrobot-v1 | ≥ -100 | 800-1000 | 15-20 min |
| MountainCar-v0 | ≥ -110 | 1500-2000 | 20-30 min |
| Pendulum-v1 | ≥ -200 | 400-500 | 10-15 min |

## Tips

1. **Always start with CartPole** - fastest to train and verify
2. **Use --no-wandb** for quick tests
3. **Monitor avg_reward** not individual episode rewards
4. **Save results with --save-results** to track progress
5. **Use tqdm progress bars** to monitor training

## Getting Help

- Read `README.md` for full documentation
- Read `TRAINING_GUIDE.md` for detailed guide
- Check `PPO_IMPLEMENTATION_SUMMARY.md` for technical details
- Run `python test_ppo_quick.py` to verify setup
