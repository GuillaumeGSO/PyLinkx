# PyLinkx - Pygame Game with Gymnasium RL Integration

PyLinkx is a two-player block placement game with full Gymnasium RL integration, enabling reinforcement learning agent training.

## Features

- **Interactive Pygame UI**: Play the game manually with graphical interface
- **Gymnasium Environment**: Fully compatible RL environment for agent training
- **RL Training Suite**: Pre-configured training scripts with Stable-Baselines3
- **Comprehensive Testing**: Unit tests for game logic and environment

## Setup

1. Make sure you have Python 3.12+ and pip installed.
2. (Recommended) Use a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run the Game

### Play Interactively

```bash
python src/main.py
```

### RL Training

#### Quick Test

Test that the environment works:

```bash
python src/train.py --mode test
```

#### Train an Agent

Train a PPO agent from scratch:

```bash
python src/train.py --mode train --timesteps 100000
```

Options:

- `--timesteps`: Number of training steps (default: 100000)
- `--model`: Path to save model (default: models/ppo_pylinkx.zip)

#### Evaluate a Trained Agent

Evaluate an existing model:

```bash
python src/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 10
```

Options:

- `--model`: Path to trained model file
- `--eval-episodes`: Number of evaluation episodes (default: 100)
- `--render`: Show game visualization during evaluation

## Project Structure

```
.
├── src/
│   ├── main.py              # Interactive game entry point
│   ├── game.py              # Core game logic with RL methods
│   ├── player.py            # Player and scoring logic
│   ├── piece.py             # Tetris piece definitions
│   ├── game_renderer.py     # Pygame rendering
│   ├── game_env.py          # Gymnasium environment wrapper
│   └── train.py             # RL training script
├── tests/
│   ├── test_game_*.py       # Game logic tests
│   └── test_rl_env.py       # Environment tests
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## RL Environment (Gymnasium)

### Action Space

Discrete(4) - Four possible actions:

- **0**: Move piece left
- **1**: Move piece right
- **2**: Rotate piece
- **3**: Drop piece (finalize placement)

### Observation Space

Box(9, 9, 1) - Game grid as numpy array:

- 9x9 grid with values [0=empty, 1=player1, 2=player2]
- Expandable with additional state features

### Reward Structure

Sparse rewards (can be extended):

- **+1.0**: Win game
- **-0.5**: Lose/game over
- **0.0**: During gameplay

## Key Files for RL Integration

- **[src/game_env.py](src/game_env.py)** - PyLinkxEnv class implementing gymnasium.Env
- **[src/game.py](src/game.py)** - Refactored game logic with RL methods:
  - `get_observation()` - Extract game state
  - `execute_action(action)` - Apply player action
  - `get_reward()` - Calculate reward
- **[src/train.py](src/train.py)** - Training, evaluation, and demo scripts

## Migration to Gymnasium

The codebase has been refactored to support RL training:

1. **Game Logic** - Decoupled from UI for programmatic control
2. **Environment Wrapper** - Gymnasium-compatible wrapper for RL agents
3. **State Representation** - Grid and score tracking for observations
4. **Reward System** - Defined rewards for agent feedback
5. **Training Scripts** - Integration with Stable-Baselines3

### Testing Refactoring

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rl_env.py -v
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_rl_env.py -v
```

## Dependencies

- **pygame** - Game rendering and UI
- **gymnasium** - RL environment standard
- **numpy** - Numerical computing
- **stable-baselines3** - RL algorithms (PPO, DQN, etc.)
- **pytest** - Testing framework

## Development

### Game Logic Tests

- `tests/test_game_*.py` - Validate core game mechanics
- `tests/test_player_*.py` - Player scoring and actions

### RL Environment Tests

- `tests/test_rl_env.py` - Gymnasium environment validation

## Future Enhancements

- [ ] Multi-agent self-play training (PettingZoo)
- [ ] Dense reward shaping (score-based rewards)
- [ ] PPO vs DQN algorithm comparison
- [ ] Advanced opponents (heuristic or RL-based)
- [ ] Hyperparameter tuning utilities
- [ ] Training visualization and monitoring
