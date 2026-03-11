# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyLinkx is a two-player block placement game (on a 9x9 grid) with a Gymnasium RL environment for training reinforcement learning agents. Players place tetris-like pieces and win either by connecting opposite borders of the grid with their pieces (path win) or by having the largest contiguous area when all pieces are used (score win).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

```bash
# Play interactively
python src/main.py

# Run all tests
pytest

# Run a single test file
pytest tests/test_rl_env.py -v

# Run tests with coverage
pytest --cov=src

# Test the RL environment setup
python src/train.py --mode test

# Train a PPO agent
python src/train.py --mode train --timesteps 100000 --envs 4 --maxsteps 100

# Evaluate a trained model
python src/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 10 --render
```

## Architecture

The codebase is split into pure game logic and the RL wrapper:

**Core game layer** (`src/`):
- `game.py` — `Game` class: 9x9 grid state, piece placement rules (`is_valid_move`, `is_fully_supported`), turn management, and win detection. Also exposes RL interface methods: `get_observation()`, `execute_action(action)`.
- `player.py` — `Player` class: holds each player's piece queue (2× each shape, shuffled), score (largest contiguous area via flood-fill), and win condition check (`check_if_winner` — BFS for border-to-border path).
- `piece.py` — `Piece` class + `TETRIS_SHAPES` dict: 7 shapes (L, S, c, T, I, u, b), each piece supports `rotate()` and `flip()`.
- `game_renderer.py` — Pygame rendering, decoupled from game logic.
- `main.py` — Interactive entry point using Pygame event loop.

**RL layer**:
- `game_env.py` — `PyLinkxEnv(gym.Env)`: wraps `Game` into a Gymnasium environment. Observation space is `Dict{"grid": Box(9,9,1), "scalars": Box(27,)}`. Action space is `Discrete(6)` (cycle piece, move left/right, rotate, flip, drop). Reward: +2000 path win, +1500 score win, +10 per drop, −50 invalid action, −0.1 per step.
- `train.py` — Training script using Stable-Baselines3 PPO with `MultiInputPolicy`. Supports `--mode test|train|evaluate`.

**Key design constraint**: `pytest.ini` sets `pythonpath = src`, so all imports within `src/` use bare module names (e.g., `from game import Game`, not `from src.game import Game`). Tests must follow this same pattern.

## Win Conditions

1. **Path win** (priority): A player's pieces form a connected path (8-directional) from left edge to right edge, OR from top edge to bottom edge.
2. **Score win** (fallback): When all players exhaust their pieces, the player with the largest single contiguous group of their cells wins.

## Validation After Changes

After any refactoring, run these three commands and verify none produce errors (non-zero exit or exception tracebacks):

```bash
# 1. Unit tests
pytest

# 2. RL environment sanity check
python src/train.py --mode test

# 3. Quick train + evaluate cycle
python src/train.py --mode train --timesteps 10000
python src/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 5
```

Expected: test mode prints "✓ Environment working correctly!", train completes and saves the model, evaluate prints episode stats without exceptions.

## RL Action Space (Actions enum in game.py)

| Value | Name | Effect |
|-------|------|--------|
| 0 | ACTION_CYCLE_PIECE | Switch to next piece in queue |
| 1 | ACTION_MOVE_LEFT | Move current piece left |
| 2 | ACTION_MOVE_RIGHT | Move current piece right |
| 3 | ACTION_ROTATE | Rotate 90° clockwise |
| 4 | ACTION_FLIP | Flip horizontally |
| 5 | ACTION_DROP | Place piece at ghost position (ends turn) |