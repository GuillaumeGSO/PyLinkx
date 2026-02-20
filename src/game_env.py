# Gymnasium RL Environment for PyLinkx
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game
from player import Player
from piece import Piece, TETRIS_SHAPES
import random


class PyLinkxEnv(gym.Env):
    """
    Gymnasium environment wrapper for the PyLinkx game.

    Single-agent mode: Agent plays against itself or a fixed opponent.
    Supports both training and evaluation.
    """

    metadata = {"render_modes": ["debug"], "render_fps": 30}

    # Action space: 0=move_left, 1=move_right, 2=rotate, 3=drop, 4=pass
    ACTION_MOVE_LEFT = 0
    ACTION_MOVE_RIGHT = 1
    ACTION_ROTATE = 2
    ACTION_DROP = 3
    ACTION_PASS = 4

    def __init__(self, render_mode=None, max_steps=500):
        """
        Initialize the PyLinkx Gymnasium environment.

        Args:
            render_mode: Rendering mode (None or "debug")
            max_steps: Maximum steps per episode to prevent infinite loops
        """
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.step_count = 0
        self.game = Game()

        # Action space: 5 discrete actions (0-4)
        self.action_space = spaces.Discrete(5)

        # Observation space: grid (9x9) + 4 scalar features
        # Grid: 9x9 cells with values [0, 1, 2] (0=empty, 1=player1, 2=player2)
        # Scalars: [current_player_idx, player1_score, player2_score, is_game_over]
        self.observation_space = spaces.Box(
            low=0,
            high=2,
            shape=(9, 9, 1),  # Flattened as (9, 9, 1) for CNN compatibility
            dtype=np.int8,
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
        self.last_scores = [0, 0]

        # Initialize first piece
        self._initialize_next_piece()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int):
        """
        Execute one step of the environment with the given action.

        Args:
            action: Action index (0-4)
                0 = move_left
                1 = move_right
                2 = rotate
                3 = drop (finalize placement)
                4 = pass (give up)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.step_count >= self.max_steps:
            # Episode truncated due to max steps
            observation = self._get_observation()
            return observation, 0.0, False, True, self._get_info()

        self.step_count += 1

        # Execute the action
        action_valid = self.game.execute_action(action)

        # Check if game is over
        terminated = self.game.status == Game.GAMEOVER

        # Calculate reward
        player_idx = self.game.players.index(self.game.current_player)
        reward = self._calculate_reward(player_idx, terminated)

        # Get next observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        """Render the current game state."""
        if self.render_mode == "debug":
            print(self.game)
            print(f"Step: {self.step_count}")
            print(f"Grid:\n{np.array(self.game.grid)}")
            print(f"Scores: {[p.score for p in self.game.players]}")

    def _initialize_next_piece(self):
        """Initialize the next piece for the current player."""
        next_piece = self.game.current_player.next_piece()
        if next_piece:
            self.game.set_current_piece(next_piece)
        else:
            # Player is out of pieces
            self.game.current_player.give_up()

    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation as a numpy array.
        Returns grid as (9, 9, 1) array for CNN compatibility.
        """
        grid_array = np.array(self.game.grid, dtype=np.int8)
        grid_array = np.expand_dims(grid_array, axis=-1)  # Add channel dimension
        return grid_array

    def _get_info(self) -> dict:
        """Get additional information about the environment state."""
        return {
            "current_player_idx": self.game.players.index(self.game.current_player),
            "scores": [p.score for p in self.game.players],
            "game_over": self.game.status == Game.GAMEOVER,
            "winner_idx": (
                self.game.players.index(self.game.winner) if self.game.winner else None
            ),
            "step_count": self.step_count,
        }

    def _calculate_reward(self, player_idx: int, terminated: bool) -> float:
        """
        Calculate reward for the current action.

        Sparse reward: +1 for win, -0.5 for loss, 0 otherwise
        Can be extended with dense rewards based on score changes.
        """
        if terminated:
            if (
                self.game.winner
                and self.game.players.index(self.game.winner) == player_idx
            ):
                return 1.0
            else:
                return -0.5

        return 0.0  # No reward during gameplay (sparse)

    def close(self):
        """Clean up resources."""
        pass

    def seed(self, seed=None):
        """Set the random seed."""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
