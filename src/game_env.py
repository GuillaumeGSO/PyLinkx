# Gymnasium RL Environment for PyLinkx
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game import Game, Actions
from game_renderer import GameRenderer


class PyLinkxEnv(gym.Env):
    """
    Gymnasium environment wrapper for the PyLinkx game.

    Single-agent mode: Agent plays against itself or a fixed opponent.
    Supports both training and evaluation.
    """

    metadata = {"render_modes": ["debug"], "render_fps": 8}
    PIECE_MAP = {"L": 0, "S": 1, "c": 2, "T": 3, "I": 4, "u": 5, "b": 6}

    def __init__(self, render_mode=None, max_steps=500, max_steps_by_turn=100):
        """
        Initialize the PyLinkx Gymnasium environment.

        Args:
            render_mode: Rendering mode (None or "debug")
            max_steps: Maximum steps per episode to prevent infinite loops
            max_steps_by_turn: Maximum steps allowed per turn before a drop is forced
        """
        self.render_mode = render_mode
        self.max_steps = max_steps
        self._max_steps_by_turn = max_steps_by_turn
        self.step_count = 0
        self.valid_action = True
        self.game = Game()

        # Action space: 6 discrete actions (0-5)
        self.action_space = spaces.Discrete(len(Actions))

        # Observation space: grid (9x9, 1 channel) + 34 scalar features
        # Grid: 9x9 cells with values normalized to [0.0, 0.5, 1.0]
        # Scalars: player value, piece x, scores, can_drop flag, piece id,
        #          remaining ratio, game over flag, action validity,
        #          remaining turn ratio, padded piece shape (4x4=16),
        #          edge touch flags: p1(left/right/top/bottom), p2(left/right/top/bottom)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(9, 9, 1), dtype=np.float32),
                "scalars": spaces.Box(low=-1.0, high=1.0, shape=(34,), dtype=np.float32),
            }
        )
        self.last_scores = [0, 0]  # Track score changes for dense rewards

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
        self.last_scores = [0, 0]
        self.valid_action = True
        self._score_delta = 0.0

        # Initialize first piece
        self.game.start_turn()

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
        # Execute the action (update() is called inside execute_action)
        self.valid_action = self.game.execute_action(action)
        self._score_delta = self.game.players[acting_player_idx].score - old_score
        if not self.valid_action and self.render_mode == "debug":
            print(f"Invalid action {Actions(action).name}")

        # If forced drop failed (no valid ghost position), auto-pass the player
        if forced_drop and not self.valid_action:
            if self.render_mode == "debug":
                print("Forced drop failed. Auto-passing player.")
            self.game.current_player.give_up()
            remaining = self.game.get_players_in_play()
            if not remaining:
                self.game.winner = self.game.check_for_winner()
            elif self.game.one_extra_turn_remaining:
                self.game._declare_score_winner()
            else:
                self.game.one_extra_turn_remaining = True
                self.game.current_player = self.game.get_next_player()
                self.game.start_turn()
            self.valid_action = True

        # Reset turn counter on successful drop
        if action == Actions.ACTION_DROP and self.valid_action:
            self.steps_for_current_turn = 0

        # Check if game is over (invalid action no longer terminates episode)
        terminated = self.game.status == Game.GAMEOVER

        # Calculate reward using acting player (current_player may have changed after DROP)
        reward = self._calculate_reward(
            acting_player_idx, self.valid_action, action, terminated, self._score_delta, forced_drop
        )

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

            print(f"Player: {self.game.current_player.name} Step: {self.step_count} Action: {Actions(action).name if action is not None else '-'}")

    def valid_action_mask(self) -> np.ndarray:
        """Returns a binary mask (1=valid, 0=invalid) for MaskablePPO."""
        return self.game.get_valid_actions()

    def _get_padded_shape(self, shape: list[list[int]]) -> np.ndarray:
        """Pads any piece shape into a fixed 4x4 array."""
        padded = np.zeros((4, 4), dtype=np.float32)
        rows = len(shape)
        cols = len(shape[0])
        # Place the shape in the top-left of the 4x4 grid
        padded[:rows, :cols] = np.array(shape)
        return padded.flatten()  # Returns 16 scalars

    def _get_edge_flags(self) -> np.ndarray:
        """
        Returns 8 binary float32 values: one per player per edge direction.
        [p1_left, p1_right, p1_top, p1_bottom, p2_left, p2_right, p2_top, p2_bottom]
        A flag is 1.0 if any of the player's cells touches that grid edge, else 0.0.
        """
        grid = np.array(self.game.grid, dtype=np.int8)
        g = self.game.GRID_SIZE - 1
        flags = []
        for player_val in (1, 2):
            mask = grid == player_val
            flags.append(float(np.any(mask[:, 0])))   # touches left  (col 0)
            flags.append(float(np.any(mask[:, g])))   # touches right (col g)
            flags.append(float(np.any(mask[0, :])))   # touches top   (row 0)
            flags.append(float(np.any(mask[g, :])))   # touches bottom (row g)
        return np.array(flags, dtype=np.float32)

    def _get_observation(self) -> dict:
        """
        Captures the grid for pathfinding (border connection)
        and scalars for the current game state.
        """
        # 1. Grid (9, 9, 1) - Normalized to [0.0, 0.5, 1.0]
        # The CNN will learn to detect 'chains' of 1s or 2s across the grid.
        grid_array = np.array(self.game.grid, dtype=np.float32) / 2.0
        grid_array = np.expand_dims(grid_array, axis=-1)

        # 2. Contextual Scalars
        current_piece = self.game.current_piece
        nb_players = len(self.game.players)
        max_pieces = 2 * len(self.PIECE_MAP)
        current_piece_id = float(self.PIECE_MAP[current_piece.shape_name]) / len(self.PIECE_MAP)
        remaining_ratio = float(len(self.game.current_player.pieces)) / max_pieces
        grid_cells = float(self.game.GRID_SIZE * self.game.GRID_SIZE)
        player_scores = [float(p.score) / grid_cells for p in self.game.players]

        other_scalars = np.array(
            [
                float(self.game.current_player.value - 1) / (nb_players - 1),  # Normalized to [0, 1]
                float(current_piece.x) / self.game.GRID_SIZE,  # Normalized x position
                player_scores[0],  # Player 1 score normalized
                float(1.0 if self.game.ghost_grid_y else 0.0),  # Can drop flag (1.0 if valid drop position exists)
                current_piece_id,  # Normalized piece type id
                remaining_ratio,  # Fraction of pieces remaining
                player_scores[1],  # Player 2 score normalized
                float(self.game.status == Game.GAMEOVER),  # Game over flag
                float(self.valid_action),  # Last action validity
            ],
            dtype=np.float32,
        )
        # 3. Padded piece shape (4x4 = 16 values)
        shape_vals = self._get_padded_shape(current_piece.shape)

        remaining_actions_ratio = (self.max_steps_by_turn - self.steps_for_current_turn) / self.max_steps_by_turn

        # Concatenate into a single (34,) array: 9 scalars + 1 ratio + 16 shape + 8 edge flags
        scalars = np.concatenate([
            other_scalars,
            [remaining_actions_ratio],
            shape_vals,
            self._get_edge_flags(),   # idx 27-34: p1/p2 edge touch flags
        ])

        return {"grid": grid_array, "scalars": scalars}

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
        }

    def _calculate_reward(
        self, player_idx: int, action_valid: bool, action: int, terminated: bool, score_delta: float = 0.0, forced_drop: bool = False
    ) -> float:
        """
        Calculate reward for the current action.

        Path-finding wins are more valuable as they require strategic placement.
        """
        if not action_valid:
            return -0.1  # Small penalty for invalid action — keeps it proportional to DROP reward
        if terminated:
            if self.game.winner and self.game.players.index(self.game.winner) == player_idx:
                return 50.0 if self.game.win_type == "path" else 37.5
            else:
                return -37.5  # Loss penalty
        # In play rewards/penalties
        if action == Actions.ACTION_DROP:
            base = 1.0 + 0.1 * score_delta  # Encourage placement + reward area growth
            return base - 0.5 if forced_drop else base  # Penalize procrastination
        if action == Actions.ACTION_CYCLE_PIECE:
            return -0.05  # Discourage idle cycling without committing to a placement
        return -0.001

    def close(self):
        """Clean up resources."""
        pass
