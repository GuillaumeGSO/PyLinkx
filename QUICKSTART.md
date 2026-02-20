# Quick Start Guide - PyLinkx RL Training

A step-by-step guide to get started with training RL agents on PyLinkx.

## 1. Setup Environment

```bash
# Navigate to project root
cd /path/to/PyLinkx

# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Verify Installation

Test that the Gymnasium environment is working:

```bash
python3 src/train.py --mode test
```

**Expected Output:**

- Environment created successfully
- 10 random steps executed
- "✓ Environment working correctly!" message

## 3. Train Your First Agent

Train a basic PPO agent:

```bash
python3 src/train.py --mode train --timesteps 50000
```

**Customization:**

- `--timesteps N` - Train for N steps (default: 100000)
- `--model PATH` - Save path (default: models/ppo_pylinkx.zip)

**Training will:**

- Create 4 parallel training environments
- Train for ~50,000 steps (~2-5 min on modern GPU)
- Save the best model to `models/ppo_pylinkx.zip`
- Print training progress and metrics

## 4. Evaluate the Trained Agent

Evaluate the trained agent:

```bash
python3 src/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 10
```

**Options:**

- `--eval-episodes N` - Run N evaluation episodes
- `--render` - Visualize games in terminal

**Output shows:**

- Mean reward (higher is better)
- Episode length statistics
- Individual episode results

## 5. Experiment with Different Settings

### Longer Training

```bash
python3 src/train.py --mode train --timesteps 500000
```

### Different Model Paths

```bash
python3 src/train.py --mode train --model models/my_agent.zip --timesteps 100000
python3 src/train.py --mode evaluate --model models/my_agent.zip
```

### View Games

```bash
python3 src/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 5 --render
```

## 6. Understanding Results

### Training Output

```
Using cuda device
| rollout/  |
| ep_len_mean    |      100 |
| ep_rew_mean    |    -0.45 |

| fps                 |     500 |
| iterations          |      12 |
| time_elapsed        |    10.5 |
```

- `ep_rew_mean`: Average reward per episode (higher is better, max 2.0)
- `ep_len_mean`: Average episode length (duration)
- `fps`: Training speed (frames per second)

### Evaluation Output

```
Episode  1: Reward =   -0.50, Length = 50
Episode  2: Reward =    1.00, Length = 120
Episode  3: Reward =    1.00, Length = 98
...
Statistics:
   Mean Reward:     0.40 ± 0.45
   Mean Length:     85.2 ± 23.4
   Max Reward:      1.00
   Min Reward:     -0.50
```
TODO add differentiated winning rewards
- `Reward = 2.00`: Agent won game by path
- `Reward = 1.00`: Agent won game by surface
- `Reward = -0.50`: Agent lost game
- `Mean Reward > 0`: Agent is learning

## 7. Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"

**Fix:** Install dependencies

```bash
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"

**Fix:** Reduce training batch or use CPU

```bash
python train.py --mode train --timesteps 10000
```

### Issue: Training is very slow

**Fix:** It's normal! Progress speeds up after few iterations. Or reduce timesteps:

```bash
python train.py --mode train --timesteps 10000
```

### Issue: Agents not improving

**Fix:** Train longer! RL requires many steps:

```bash
python train.py --mode train --timesteps 500000
```

## 8. Next Steps

### Experiment with Hyperparameters

Modify `train.py` to adjust:

- `learning_rate` - How quickly agent learns
- `gamma` - Reward discount factor (0.99 = value future rewards)
- `n_epochs` - Training epochs per batch
- `batch_size` - Batch size for training

### Try Different Algorithms

Modify `train.py` to use DQN instead of PPO:

```python
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, ...)
```

### Add Dense Rewards

Modify `game_env.py` `_calculate_reward()` to add score-based rewards:

```python
score_reward = (new_score - old_score) * 0.01
return terminal_reward + score_reward
```

### Visualize Training

Install TensorBoard and track metrics:

```bash
pip install tensorboard

# Modify train.py to use TensorboardCallback
# Then view:
tensorboard --logdir ./tb_logs/
```

## 8. Directory Structure

After training, your structure will look like:

```
PyLinkx/
├── models/
│   ├── ppo_pylinkx.zip          # Trained model
│   └── best_model.zip            # Best model during training (if using EvalCallback)
├── src/
│   ├── game_env.py
│   ├── train.py
│   ├── game.py
│   └── ...
├── tests/
│   ├── test_rl_env.py
│   └── ...
└── README.md
```

## Learning Resources

- **PyLinkx Specific**: See [src/Reinforcement learning/MIGRATION_SUMMARY.md](src/Reinforcement%20learning/MIGRATION_SUMMARY.md)
- **Gymnasium Docs**: https://gymnasium.farama.org/
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **RL Fundamentals**: https://spinningup.openai.com/

## Troubleshooting

**For detailed diagnostics:**

```bash
# Check environment works with more verbose output
python -c "
import sys
sys.path.insert(0, 'src')
from game_env import PyLinkxEnv
env = PyLinkxEnv()
obs, info = env.reset()
print(f'Observation shape: {obs.shape}')
print(f'Action space: {env.action_space}')
for i in range(5):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    print(f'Step {i+1}: reward={reward}, done={done or truncated}')
env.close()
print('✓ Environment working correctly!')
"
```

---

**Happy training! 🚀**

For issues or questions, check the comprehensive test suite in `tests/test_rl_env.py` for usage examples.
