# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyLinkx is a two-player block placement game (on a 9x9 grid) with a Gymnasium RL environment for training reinforcement learning agents. Players place tetris-like pieces and win either by connecting opposite borders of the grid with their pieces (path win) or by having the largest contiguous area when all pieces are used (score win).

## Setup

```bash
uv sync                          # install runtime deps (play the game)
uv sync --group test             # adds pytest + sb3/torch (run tests)
uv sync --group train            # adds sb3/torch (training only, no pytest)
uv sync --group export           # adds sb3/torch + onnxscript (ONNX export)
uv sync --group web              # adds pygbag (web build)
uv sync --group build            # adds pyinstaller (build executables)
```

Switching groups locally is clean — `uv sync --group <name>` installs exactly that group and removes anything outside it. No need to wipe the venv manually.

`uv.lock` is not committed — CI resolves deps fresh on each run, matching the spirit of per-group `requirements-*.txt` files.

## Running Commands

**Prefix Python commands with `uv run`** — this ensures the project venv is used without manual activation, no `source .venv/bin/activate` needed:

```bash
uv run <command>
```

`source .venv/bin/activate` is only needed if you want to run bare commands (`python`, `pytest`) without the `uv run` prefix in your terminal session.

## Commands

```bash
# Play interactively
uv run python src/main.py

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_rl_env.py -v

# Run tests with coverage
uv run pytest --cov=src

# Test the RL environment setup
uv run python src/training/train.py --mode test

# Train a PPO agent (P2 = drop-first fallback)
uv run python src/training/train.py --mode train --timesteps 100000 --envs 4 --maxsteps 100

# Train with opponent model (iterative self-play)
uv run python src/training/train.py --mode train --timesteps 100000 --opponent-model models/ppo_pylinkx.zip

# Evaluate a trained model
uv run python src/training/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 10 --render

# Run the self-play training pipeline
uv run python src/pipeline/pipeline.py --baseline-model models/base_line_model.zip

# Run pipeline with custom pool pruning (keep baseline + last 3 RL loops, default)
uv run python src/pipeline/pipeline.py --baseline-model models/base_line_model.zip --pool-lookback 3

# Evaluate all loop models in a round-robin matrix
uv run python src/pipeline/evaluate_matrix.py --from-manifest src/pipeline/manifest.json --episodes 200

# Export game models from models/ to src/models/ as ONNX (run after updating game models)
# Requires: uv sync --group export
uv run python scripts/export_onnx.py

# Test web build locally (starts dev server at http://localhost:8000)
uv run pygbag src/main.py

# Web test with Playwright (headless browser screenshots + console capture)
# Prerequisites (one-time): npm install && npx playwright install chromium
# Start pygbag first, then run a scenario:
uv run pygbag --port 8000 src/main.py &
node scripts/web_test.js --scenario vs-hard
# Available scenarios: vs-hard, vs-medium, vs-easy, 2p, menu
# Manual key sequence with waits and screenshots:
node scripts/web_test.js --keys "ArrowDown,Enter,wait:3000,screenshot,ArrowDown,ArrowDown,Enter"
# Screenshots saved to ./screenshots/ (gitignored)

# Build standalone executable (all platforms — no torch required)
# PyLinkx.spec is gitignored; see .github/workflows/deploy.yml for the full pyinstaller flags.
# Local quick build (generates a fresh spec):
uv run pyinstaller --onefile --name PyLinkx --paths src --add-data "src/models:models" --add-data "src/assets:assets" --hidden-import game.game --hidden-import game.game_renderer --hidden-import game.menu_renderer --hidden-import game.player --hidden-import game.piece --collect-all onnxruntime --exclude-module torch --exclude-module torchvision --exclude-module torchaudio --exclude-module stable_baselines3 --exclude-module sb3_contrib src/main.py
```

## Versioning

**`src/version.py`** — single source of truth for the project version string:

```python
__version__ = "0.3.2"
```

- Imported by `src/game/menu_renderer.py` to display the version on the in-game menu.
- Follow **semver**: bump **minor** (0.x.0) for new features, bump **patch** (0.0.x) for bug fixes and small improvements.
- **Bump the version whenever a new branch is created**, before any other work, using the appropriate increment for the planned change.

## Architecture

The codebase is split into pure game logic and the RL wrapper:

**Core game layer** (`src/game/`):
- `game.py` — `Game` class: 9x9 grid state, piece placement rules (`is_valid_move`, `is_fully_supported`), turn management, and win detection. Also exposes `execute_action(action)` as a programmatic dispatcher used by both `main.py` and `game_env.py`.
- `player.py` — `Player` class: holds each player's piece queue (2× each shape, shuffled), score (largest contiguous area via flood-fill), and win condition check (`check_if_winner` — BFS for border-to-border path).
- `piece.py` — `Piece` class + `TETRIS_SHAPES` dict: 7 shapes (L, S, c, T, I, u, b), each piece supports `rotate()` and `flip()`.
- `game_renderer.py` — Pygame rendering, decoupled from game logic.
- `main.py` (`src/`) — Interactive entry point using Pygame event loop.

**Inference layer** (`src/inference/`):
- `onnx_policy.py` — `OnnxPolicy`: ONNX runtime wrapper, drop-in replacement for `MaskablePPO.predict()`. No PyTorch needed.
- `wasm_onnx_policy.py` — `WasmOnnxPolicy` / `WasmModelLoader`: browser-side ONNX inference via onnxruntime-web CDN. Used in the pygbag WASM build where native onnxruntime is unavailable.
- `observation.py` — `build_observation` / `compute_action_mask` / `compute_path_progress`. Shared by training, gameplay, and WASM inference.
- `tactical.py` — 1-ply lookahead safety net. `find_tactical_move(game, player_idx)` returns a winning placement (if any legal placement wins this turn), else a placement that neutralizes every pre-existing opponent one-move threat, else `None`. Called in `main.py` before the model loop so all three difficulty models short-circuit on obvious tactics.

**Developer scripts** (`scripts/`):
- `export_onnx.py` — Converts `models/{easy,medium,hard}_model.zip` → `src/models/*.onnx`. Run after updating game models. Requires export deps (`uv sync --group export`).

**RL layer** (`src/training/`):
- `game_env.py` — `PyLinkxEnv(gym.Env)`: wraps `Game` into a Gymnasium environment. Agent plays as P1 only; P2 is controlled internally by a frozen opponent model (`opponent_model_path`) or a drop-first fallback. Observation space is `Dict{"grid": Box(9,9,1), "scalars": Box(258,)}` — scalars include piece position, scores, path progress, current piece shape, and full piece inventory (canonical shape × count) for both players. Action space is `Discrete(6)` (cycle piece, move left/right, rotate, flip, drop). Reward (P1 perspective): +100 path win, +20 score win, −100/−20 P2 wins, +1.0 per drop + path progress bonus, −0.1 invalid, −0.05 cycle, −0.001 other.
- `train.py` — Training script using sb3-contrib MaskablePPO with custom `PyLinkxFeaturesExtractor` (CNN 3×Conv2d→128-dim for grid + MLP 258→128-dim for scalars, total features_dim=256). Supports `--mode test|train|evaluate` and `--opponent-model` for iterative self-play.

**Pipeline** (`src/pipeline/`):
- `pipeline.py` — Automated self-play training pipeline. Runs iterative loops, versions models under `src/pipeline/models/loop_N/`, evaluates vs baseline, and selects Easy/Medium/Hard difficulty models. State tracked in `src/pipeline/manifest.json`. Opponent pool is pruned to baseline + last `--pool-lookback` (default 3) RL loops, with linear sampling weights so recent opponents are favoured.
- `evaluate_matrix.py` — Round-robin evaluation across all loop models. Prints win-rate tables and optionally saves results to JSON.
- `models/` — Pipeline working directory. Loop model checkpoints live here (`loop_N/best_model.zip`). Not committed alongside game models.
- `manifest.json` — Tracks training history, per-loop metrics, and the selected difficulty triplet.

**Model directories**:
- `src/models/` — Game-ready ONNX models bundled with the itch.io web build: `easy_model.onnx`, `medium_model.onnx`, `hard_model.onnx`. Loaded by `main.py` via `__file__`-relative paths. Generate with `scripts/export_onnx.py`.
- `models/` (project root) — Training working directory: `base_line_model.zip`, `ppo_pylinkx.zip` (output of `train.py`), `easy_model.zip`, `medium_model.zip`, `hard_model.zip` (source models for ONNX export). Not bundled in builds.
- `src/pipeline/models/` — Pipeline working directory. Loop checkpoints produced during training runs.

**Key design constraint**: `pytest.ini` sets `pythonpath = .` (project root), so all imports use the full `src.` prefix (e.g., `from src.game.game import Game`). Intra-package imports within `src/game/` use relative form (e.g., `from .piece import Piece`). Scripts run directly (`main.py`, `train.py`, `pipeline.py`, `evaluate_matrix.py`) insert the project root into `sys.path` at startup so the same import style works when executed as scripts.

## Win Conditions

1. **Path win** (priority): A player's pieces form a connected path (8-directional) from left edge to right edge, OR from top edge to bottom edge.
2. **Score win** (fallback): When all players exhaust their pieces, the player with the largest single contiguous group of their cells wins.

## Documentation

When making changes that affect user-facing behavior — game rules, controls, win conditions, RL action/observation space, CLI commands, or setup steps — update `README.md` to reflect the new behavior.

## Validation After Changes

After any refactoring, run these three commands and verify none produce errors (non-zero exit or exception tracebacks):

```bash
# 1. Unit tests
uv run pytest

# 2. RL environment sanity check
uv run python src/training/train.py --mode test

# 3. Quick train + evaluate cycle
uv run python src/training/train.py --mode train --timesteps 10000
uv run python src/training/train.py --mode evaluate --model models/ppo_pylinkx.zip --eval-episodes 5
```

Expected: test mode prints "✓ Environment working correctly!", train completes and saves the model, evaluate prints episode stats without exceptions.

When changing the AI integration in `main.py` or the tactical layer, also do an interactive smoke test (`uv run python src/main.py`, pick a difficulty) and confirm the AI still takes obvious one-move wins and blocks obvious one-move threats.

### Fixing failing tests

All tests should pass. If any fail after a change:

1. Run `pytest -v` to identify which tests fail and read the full error message.
2. Check whether the test expectation is stale (wrong value, wrong shape, wrong signature) vs. a real regression in game logic.
3. **Stale test** — update the test to match the current implementation. Common patterns:
   - Observation is `dict` with keys `"grid"` and `"scalars"` — use `obs["grid"].shape`, not `obs.shape`; scalars shape is `(258,)`
   - Imports inside `tests/` must use full `src.` paths: `from src.game.game import Game`, `from src.game.piece import Piece`, `from src.game.player import Player`, `from src.training.game_env import PyLinkxEnv` — mismatched import paths cause `isinstance` to silently return `False`
   - Reward values: ±100/20 for wins/losses, +1.0 per drop, −0.1 invalid, −0.05 cycle, −0.001 other
   - `_calculate_reward(player_idx, action_valid, action, terminated)` requires all 4 arguments; `action` is an `Actions` int, not a string
4. **Real regression** — fix the source code, then re-run validation.

## Code Style Principles

- **KISS**: Keep solutions simple. Prefer the simplest approach that works.
- **Single Responsibility**: Each function/class does one thing. Game logic stays in `game.py`; rendering stays in `game_renderer.py`; RL wrapping stays in `game_env.py`. Specifically: action masks, observation building, and reward logic belong in `game_env.py` — not `game.py`. `game.py` must not import `numpy` or reference Gymnasium concepts.
- **No over-engineering**: Don't add abstractions, helpers, or configurability for hypothetical future needs. Three similar lines beat a premature abstraction.
- **SOLID**:
  - **S**ingle Responsibility — one reason to change per class/function (already covered above)
  - **O**pen/Closed — open for extension, closed for modification; add new piece types or win conditions without rewriting existing methods
  - **L**iskov Substitution — subtypes must be substitutable for their base type; a custom `Piece` subclass must work anywhere `Piece` is used
  - **I**nterface Segregation — don't force callers to depend on methods they don't use; keep `game.py`, `player.py`, and `game_env.py` interfaces focused
  - **D**ependency Inversion — depend on abstractions not concretions; `game_env.py` depends on `Game`'s public interface, not its internals
- **No speculative features**: Only implement what is explicitly requested. No extra error handling, fallbacks, or validation beyond what's needed at system boundaries.

## RL Action Space (Actions enum in game.py)

| Value | Name | Effect |
|-------|------|--------|
| 0 | ACTION_CYCLE_PIECE | Switch to next piece in queue |
| 1 | ACTION_MOVE_LEFT | Move current piece left |
| 2 | ACTION_MOVE_RIGHT | Move current piece right |
| 3 | ACTION_ROTATE | Rotate 90° clockwise |
| 4 | ACTION_FLIP | Flip horizontally |
| 5 | ACTION_DROP | Place piece at ghost position (ends turn) |