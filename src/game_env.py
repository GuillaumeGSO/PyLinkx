# Gymnasium RL Environment for PyLinkx
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import Game
import random


class PyLinkxEnv(gym.Env):
    """
    Gymnasium environment wrapper for the PyLinkx game.

    Single-agent mode: Agent plays against itself or a fixed opponent.
    Supports both training and evaluation.
    """

    metadata = {"render_modes": ["debug"], "render_fps": 5}

    ACTION_CYCLE_PIECE = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2
    ACTION_ROTATE = 3
    ACTION_FLIP = 4
    ACTION_DROP = 5
    ACTION_PASS = 6

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

        # Action space: 7 discrete actions (0-6)
        self.action_space = spaces.Discrete(7)

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
            action: Action index (0-6)
                0 = select next piece (cycle through available pieces)
                1 = move_left
                2 = move_right
                3 = rotate
                4 = flip (horizontal)
                5 = drop (finalize placement)
                6 = pass (give up)

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

        # If action was successful and game is not over, initialize next piece
        if action_valid and not terminated:
            self._initialize_next_piece()

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
            # pygame.init()
            # screen = pygame.display.set_mode(
            # (GameRenderer.SCREEN_WIDTH, GameRenderer.SCREEN_HEIGHT)
            # )
            # pygame.display.set_caption("PyLinkx RL Environment - Debug Render")
            # render = GameRenderer(screen, self.game)
            # render.draw()
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
            "win_type": self.game.win_type,  # 'path' or 'score' or None
            "step_count": self.step_count,
        }

    def _calculate_reward(self, player_idx: int, terminated: bool) -> float:
        """
        Calculate reward for the current action.

        Reward structure:
        - Path-finding win: +2.0 (higher reward for strategic victory)
        - Score-based win: +1.0 (when all pieces used or players passed)
        - Loss/Game Over: -0.5
        - During gameplay: 0.0

        Path-finding wins are more valuable as they require strategic placement.
        """
        if terminated:
            if (
                self.game.winner
                and self.game.players.index(self.game.winner) == player_idx
            ):
                # Player won
                if self.game.win_type == "path":
                    return 2.0  # Higher reward for path-finding victory
                else:
                    return 1.0  # Standard reward for score-based victory
            else:
                # Player lost
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
