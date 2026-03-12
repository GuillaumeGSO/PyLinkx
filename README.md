# PyLinkx - Pygame Game with Gymnasium RL Integration

PyLinkx is a two-player block placement game on a 9×9 grid with a full Gymnasium RL environment for training reinforcement learning agents. Players place tetris-like pieces and win by connecting opposite borders (path win) or holding the largest contiguous area when all pieces are used (score win).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Play Interactively

```bash
python src/main.py
```

**Goal:** Connect two opposite borders of the 9×9 grid with your pieces (path win), or hold the largest contiguous area when all pieces are used (score win). Players alternate turns.

**Controls:**

| Key | Action |
|-----|--------|
| `Tab` | Cycle to next piece in queue |
| `←` / `→` | Move piece left / right |
| `↑` | Rotate piece 90° clockwise |
| `Enter` | Flip piece horizontally |
| `↓` | Drop piece (place it, ends your turn) |
| `R` | Restart (game over screen) |
| `Esc` | Quit |

### RL Training

```bash
# Verify environment works
python src/train.py --mode test

# Train a PPO agent
python src/train.py --mode train --timesteps 100000 --envs 4 --maxsteps 100

# Evaluate a trained model
python src/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 10 --render
```

**Training options:**
- `--timesteps`: Number of training steps (default: 100000)
- `--envs`: Number of parallel environments (default: 4)
- `--maxsteps`: Max steps per episode (default: 100)
- `--model`: Path to save/load model (default: `models/ppo_pylinkx.zip`)
- `--eval-episodes`: Number of evaluation episodes (default: 100)
- `--render`: Show game visualization during evaluation

### Tests

```bash
pytest                          # All tests
pytest tests/test_rl_env.py -v  # RL environment only
pytest --cov=src                # With coverage
```

## Project Structure

```
.
├── src/
│   ├── main.py              # Interactive game entry point
│   ├── game.py              # Core game logic and RL interface
│   ├── player.py            # Player state and piece queue
│   ├── piece.py             # Tetris piece definitions (7 shapes)
│   ├── game_renderer.py     # Pygame rendering
│   ├── game_env.py          # Gymnasium environment wrapper
│   └── train.py             # PPO training, evaluation, and test scripts
├── tests/
│   ├── test_game_*.py       # Game logic tests
│   ├── test_player_*.py     # Player scoring tests
│   └── test_rl_env.py       # Gymnasium environment tests
├── models/                  # Saved model checkpoints
├── requirements.txt
└── README.md
```

## RL Environment

### Action Space

`Discrete(6)` — six actions:

| Value | Action | Effect |
|-------|--------|--------|
| 0 | Cycle piece | Switch to next piece in queue |
| 1 | Move left | Move current piece left |
| 2 | Move right | Move current piece right |
| 3 | Rotate | Rotate 90° clockwise |
| 4 | Flip | Flip piece horizontally |
| 5 | Drop | Place piece at ghost position (ends turn) |

### Observation Space

`Dict` with two components:

- **`grid`**: `Box(9, 9, 1)` — game grid normalized to `[0.0, 0.5, 1.0]` (empty / player 1 / player 2)
- **`scalars`**: `Box(27,)` — 27 normalized scalar features:
  - Player value, piece x position, player 1 score, ghost y (−1 if no valid drop)
  - Piece type id, remaining pieces ratio, ghost presence flag, player 2 score
  - Game over flag, last action validity, remaining turn steps ratio
  - Current piece shape padded to 4×4 (16 values)

### Reward Structure

| Event | Reward |
|-------|--------|
| Path win | +10.0 |
| Score win | +7.5 |
| Loss | −7.5 |
| Drop (place piece) | +0.1 + 0.01 × score_delta |
| Forced drop (procrastination penalty) | −0.05 |
| Invalid action | −1.0 |
| Per step | −0.001 |

### Training Configuration (PPO)

| Hyperparameter | Value |
|---------------|-------|
| Policy | MultiInputPolicy |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| batch_size | 128 |
| n_epochs | 10 |
| gamma | 0.999 |
| ent_coef | 0.02 |
| Reward normalization | VecNormalize |

## Dependencies

- **pygame** — game rendering and UI
- **gymnasium** — RL environment standard
- **numpy** — numerical computing
- **stable-baselines3** — PPO implementation
- **pytest** — testing framework
