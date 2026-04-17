# Gymnasium RL Environment for PyLinkx
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from src.game.game import Game, Actions
from src.game.game_renderer import GameRenderer
from src.game.piece import TETRIS_SHAPES
from src.training.observation import (
    PIECE_MAP, compute_action_mask, compute_path_progress, build_observation,
    _build_piece_inventory, _CANONICAL_SHAPES,
)


class PyLinkxEnv(gym.Env):
    """
    Gymnasium environment wrapper for the PyLinkx game.

    Single-agent mode: Agent plays as Player 1. Player 2 is controlled by
    a frozen opponent model or a drop-first fallback policy.
    """

    metadata = {"render_modes": ["debug"], "render_fps": 8}
    PIECE_MAP = PIECE_MAP

    def __init__(self, render_mode=None, max_steps=500, max_steps_by_turn=100,
                 opponent_model_path=None, opponent_model_paths=None, opponent_weights=None):
        """
        Initialize the PyLinkx Gymnasium environment.

        Args:
            render_mode: Rendering mode (None or "debug")
            max_steps: Maximum steps per episode to prevent infinite loops
            max_steps_by_turn: Maximum steps allowed per turn before a drop is forced
            opponent_model_path: Path to a single MaskablePPO model for P2 (backwards-compatible).
            opponent_model_paths: List of model paths to sample from (weighted pool).
            opponent_weights: Sampling weights for opponent_model_paths (linear if None).
                If None or file missing, P2 uses a drop-first fallback policy.
        """
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._max_steps_by_turn = max_steps_by_turn
        self.step_count = 0
        self.valid_action = True
        self.game = Game()

        # Opponent model for Player 2
        self._opponent_model = None
        if opponent_model_paths:
            import random
            from sb3_contrib import MaskablePPO
            weights = opponent_weights or list(range(1, len(opponent_model_paths) + 1))
            sampled = random.choices(opponent_model_paths, weights=weights, k=1)[0]
            if os.path.exists(sampled):
                self._opponent_model = MaskablePPO.load(sampled)
        elif opponent_model_path and os.path.exists(opponent_model_path):
            from sb3_contrib import MaskablePPO
            self._opponent_model = MaskablePPO.load(opponent_model_path)

        # Action space: 6 discrete actions (0-5)
        self.action_space = spaces.Discrete(len(Actions))

        # Observation space: grid (9x9, 1 channel) + 258 scalar features
        # Grid: 9x9 cells with values normalized to [0.0, 0.5, 1.0]
        # Scalars [0:34]:  player value, piece x, scores, can_drop flag, piece id,
        #                  remaining ratio, game over flag, action validity,
        #                  remaining turn ratio, padded piece shape (4x4=16),
        #                  path progress: p1(h, v, best, area), p2(h, v, best, area)
        # Scalars [34:146]: P1 piece inventory — 7 types × 16 cells (canonical shape × count/2)
        # Scalars [146:258]: P2 piece inventory — same encoding
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(9, 9, 1), dtype=np.float32),
                "scalars": spaces.Box(low=-1.0, high=1.0, shape=(258,), dtype=np.float32),
            }
        )
        self._path_progress = [[0.0, 0.0], [0.0, 0.0]]  # [player_idx][h, v] BFS progress

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Reset the environment to initial state.

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        self.game.reset()
        self.step_count = 0
        self.steps_for_current_turn = 0
        self.max_steps_by_turn = self._max_steps_by_turn
        self._path_progress = [[0.0, 0.0], [0.0, 0.0]]
        self.valid_action = True
        self._score_delta = 0.0
        self._p1_drops = 0
        self._p2_drops = 0

        # Initialize first piece
        self.game.start_turn()

        # If P1 got auto-passed (edge case), play P2's turn
        if self.game.status != Game.GAMEOVER and self.game.players.index(self.game.current_player) != 0:
            self._play_opponent_turn()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        """
        Execute one step of the environment with the given action.
        Returns:
            observation, reward, terminated, truncated, info
        """
        
        if self.step_count >= self.max_steps:
            # Episode truncated due to max steps
            observation = self._get_observation()
            return observation, 0.0, False, True, self._get_info()

        self.step_count += 1
        self.steps_for_current_turn += 1

        forced_drop = self.steps_for_current_turn >= self.max_steps_by_turn
        if forced_drop:
            action = Actions.ACTION_DROP  # Force drop to end turn
            if self.render_mode == "debug":
                print("Max steps for turn reached. Forcing drop action.")

        # Capture pre-action state for reward shaping
        acting_player_idx = self.game.players.index(self.game.current_player)
        old_score = self.game.players[acting_player_idx].score
        old_progress = list(self._path_progress[acting_player_idx])
        # Execute the action (update() is called inside execute_action)
        self.valid_action = self.game.execute_action(action)
        self._score_delta = self.game.players[acting_player_idx].score - old_score
        if not self.valid_action and self.render_mode == "debug":
            print(f"Invalid action {Actions(action).name}")

        # If forced drop failed (no valid ghost position), auto-pass the player
        if forced_drop and not self.valid_action:
            if self.render_mode == "debug":
                print("Forced drop failed. Auto-passing player.")
            self.game.force_pass()
            self.valid_action = True

        # Reset turn counter on successful drop; update path progress cache
        progress_delta = (0.0, 0.0)
        if action == Actions.ACTION_DROP and self.valid_action:
            self._p1_drops += 1
            self.steps_for_current_turn = 0
            new_progress = self._compute_path_progress(acting_player_idx)
            self._path_progress[acting_player_idx] = list(new_progress)
            progress_delta = (new_progress[0] - old_progress[0], new_progress[1] - old_progress[1])

        # Check if game is over (invalid action no longer terminates episode)
        terminated = self.game.status == Game.GAMEOVER

        # Calculate reward using acting player (current_player may have changed after DROP)
        reward = self._calculate_reward(
            acting_player_idx, self.valid_action, action, terminated, self._score_delta, forced_drop, progress_delta
        )

        # After P1's action, if it's now P2's turn, play P2's turn internally
        if not terminated and self.game.players.index(self.game.current_player) != 0:
            old_p2_progress = list(self._path_progress[1])
            self._play_opponent_turn()
            # Penalize P1 for opponent path progress gained during P2's turn
            new_p2_progress = self._path_progress[1]
            p2_progress_delta = max(new_p2_progress[0] - old_p2_progress[0],
                                    new_p2_progress[1] - old_p2_progress[1])
            reward -= p2_progress_delta * 5.0
            # Re-check termination (P2 might have won during their turn)
            terminated = self.game.status == Game.GAMEOVER
            if terminated and self.game.winner and self.game.winner != self.game.players[0]:
                reward = -100.0 if self.game.win_type == "path" else -20.0

        # Get next observation
        observation = self._get_observation()
        info = self._get_info(self.valid_action)

        return observation, reward, terminated, False, info

    def render(self, renderer=None, action: Actions | None = None):
        """Render the current game state."""
        if self.render_mode == "debug":
            if renderer:
                renderer.draw()
                pygame.display.flip()

            #mask = self.valid_action_mask()
            #valid_actions = [Actions(i).name for i, v in enumerate(mask) if v]
            #print(f"Player: {self.game.current_player.name} Step: {self.step_count} Action: {Actions(action).name if action is not None else '-'} Valid: {valid_actions}")

    def valid_action_mask(self) -> np.ndarray:
        """Returns a binary mask (1=valid, 0=invalid) for MaskablePPO."""
        return compute_action_mask(self.game)

    def _flip_observation_perspective(self, obs: dict) -> dict:
        """Flip observation so an opponent model (trained as P1) sees P2's perspective as its own."""
        grid = obs["grid"].copy()
        # Swap P1 cells (0.5) and P2 cells (1.0); empty (0.0) stays
        p1_mask = grid == 0.5
        p2_mask = grid == 1.0
        grid[p1_mask] = 1.0
        grid[p2_mask] = 0.5

        scalars = obs["scalars"].copy()
        scalars[0] = 0.0                        # current_player → P1 perspective
        scalars[2], scalars[6] = scalars[6], scalars[2]  # swap P1/P2 scores
        p1_progress = scalars[26:30].copy()      # swap P1/P2 path progress
        scalars[26:30] = scalars[30:34]
        scalars[30:34] = p1_progress
        p1_inv = scalars[34:146].copy()          # swap P1/P2 piece inventories
        scalars[34:146] = scalars[146:258]
        scalars[146:258] = p1_inv

        return {"grid": grid, "scalars": scalars}

    def _get_opponent_action(self) -> int:
        """Get P2's next action using opponent model or drop-first fallback."""
        mask = self.valid_action_mask()
        if self._opponent_model is not None:
            obs = self._get_observation()
            obs = self._flip_observation_perspective(obs)
            action, _ = self._opponent_model.predict(obs, deterministic=False, action_masks=mask)
            return int(action)
        # Drop-first: drop immediately when possible, otherwise random valid action
        if mask[Actions.ACTION_DROP]:
            return Actions.ACTION_DROP
        valid_indices = np.where(mask == 1)[0]
        return int(np.random.choice(valid_indices))

    def _play_opponent_turn(self):
        """Play P2's turn(s) internally until it's P1's turn or game ends."""
        opponent_steps = 0
        while (self.game.status != Game.GAMEOVER
               and self.game.players.index(self.game.current_player) != 0
               and self.step_count < self.max_steps):

            opponent_steps += 1
            self.step_count += 1

            if opponent_steps >= self._max_steps_by_turn:
                success = self.game.execute_action(Actions.ACTION_DROP)
                if not success:
                    self.game.force_pass()
                break

            action = self._get_opponent_action()
            self.game.execute_action(action)

            if action == Actions.ACTION_DROP:
                self._p2_drops += 1
                # Update P2's path progress cache
                p2_idx = 1
                self._path_progress[p2_idx] = list(self._compute_path_progress(p2_idx))
                opponent_steps = 0  # Reset for next piece if P2 has another turn

    def _get_observation(self) -> dict:
        return build_observation(
            self.game, self.max_steps_by_turn, self.steps_for_current_turn,
            self.valid_action, self._path_progress,
        )

    def _get_info(self, action_valid=None) -> dict:
        """Get additional information about the environment state."""
        return {
            "current_player_idx": self.game.players.index(self.game.current_player),
            "scores": [p.score for p in self.game.players],
            "game_over": self.game.status == Game.GAMEOVER,
            "winner_idx": (
                self.game.players.index(self.game.winner) if self.game.winner else None
            ),
            "win_type": self.game.win_type,  # 'path' or 'score' or None
            "step_count": self.step_count,
            "action_valid": action_valid,
            "p1_drops": self._p1_drops,
            "p2_drops": self._p2_drops,
        }

    def _compute_path_progress(self, player_idx: int) -> tuple[float, float]:
        return compute_path_progress(self.game, player_idx)

    def _calculate_reward(
        self, player_idx: int, action_valid: bool, action: int, terminated: bool,
        score_delta: float = 0.0, forced_drop: bool = False, progress_delta: tuple[float, float] = (0.0, 0.0)
    ) -> float:
        """
        Calculate reward for the current action.

        Path-finding wins are more valuable as they require strategic placement.
        """
        if not action_valid:
            return -0.1
        if terminated:
            if self.game.winner and self.game.players.index(self.game.winner) == player_idx:
                return 100.0 if self.game.win_type == "path" else 20.0
            else:
                return -20.0
        # In play rewards/penalties
        if action == Actions.ACTION_DROP:
            base = 1.0 + (progress_delta[0] + progress_delta[1]) * 10.0
            return base - 0.5 if forced_drop else base
        if action == Actions.ACTION_CYCLE_PIECE:
            return -0.05
        return -0.001

    def close(self):
        """Clean up resources."""
        pass
