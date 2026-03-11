# Gymnasium RL Environment for PyLinkx
from enum import IntEnum
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game import Game
import random

from game_renderer import GameRenderer


class Actions(IntEnum):
    ACTION_CYCLE_PIECE = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2
    ACTION_ROTATE = 3
    ACTION_FLIP = 4
    ACTION_DROP = 5
    ACTION_PASS = 6


class PyLinkxEnv(gym.Env):
    """
    Gymnasium environment wrapper for the PyLinkx game.

    Single-agent mode: Agent plays against itself or a fixed opponent.
    Supports both training and evaluation.
    """

    metadata = {"render_fps": 12}

    def __init__(self, render_mode=None, max_steps=200):
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
        self.renderer = None

        pygame.init()
        screen = pygame.display.set_mode(
            (GameRenderer.SCREEN_WIDTH, GameRenderer.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("PyLinkx RL Environment - Debug Render")
        self.renderer = GameRenderer(screen, self.game)
        self.clock = pygame.time.Clock()

        # Action space: 7 discrete actions (0-6)
        self.action_space = spaces.Discrete(len(Actions))

        # Observation space: grid (9x9) + 4 scalar features
        # Grid: 9x9 cells with values [0, 1, 2] (0=empty, 1=player1, 2=player2)
        # Scalars: [current_player_idx, player1_score, player2_score, is_game_over] -> to be updated
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=2, shape=(9, 9, 1), dtype=np.int8),
                "scalars": spaces.Box(low=-1, high=1, shape=(23,), dtype=np.float32),
            }
        )
        self.last_scores = [0, 0]  # Track score changes for dense rewards

    def valid_action_mask(self) -> np.ndarray:
        # Call the logic already defined in your Game class
        # Ensure it returns a numpy array of type np.int8
        mask = self.game.get_valid_actions(self.action_space.n) 
        return np.array(mask, dtype=np.int8)

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
        self.max_steps_by_turn = 20 # Max steps allowed for a single turn before forcing a pass
        self.last_scores = [0, 0]

        # Initialize first piece
        self._initialize_next_piece()

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

        if self.steps_for_current_turn >= self.max_steps_by_turn:
            action = Actions.ACTION_PASS  # Force pass to end turn
            if self.render_mode == "debug":
                print(f"Max steps for turn reached. Forcing pass action.")
        if action == Actions.ACTION_PASS:
            self.steps_for_current_turn = 0  # Reset turn step count on drop
        # Execute the action
        self.game.execute_action(action)
        self.game.update()
        
        # Check if game is over
        terminated = self.game.status == Game.GAMEOVER

        # Calculate reward
        player_idx = self.game.players.index(self.game.current_player)
        reward = self._calculate_reward(
            player_idx, action, terminated
        )

        # Get next observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def render(self):
        """Render the current game state."""
        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"]) if self.clock else None
            self.renderer.draw()
            pygame.display.flip()

    # SHOULD NOT BE HERE; move to game logic
    def _initialize_next_piece(self):
        """Initialize the next piece for the current player."""
        next_piece = self.game.current_player.next_piece()
        if next_piece:
            self.game.set_current_piece(next_piece)
        else:
            # Player is out of pieces
            self.game.current_player.give_up()

    def _get_padded_shape(self, shape: list[list[int]]) -> np.ndarray:
        """Pads any piece shape into a fixed 4x4 array."""
        padded = np.zeros((4, 4), dtype=np.float32)
        rows = len(shape)
        cols = len(shape[0])
        # Place the shape in the top-left of the 4x4 grid
        padded[:rows, :cols] = np.array(shape)
        return padded.flatten()  # Returns 16 scalars

    def _get_observation(self) -> dict:
        """
        Captures the grid for pathfinding (border connection)
        and scalars for the current game state.
        """
        # 1. Grid (9, 9, 1) - Crucial for the "Border Connection" win condition
        # The CNN will learn to detect 'chains' of 1s or 2s across the grid.
        grid_array = np.array(self.game.grid, dtype=np.int8)
        grid_array = np.expand_dims(grid_array, axis=-1)

        # 2. Contextual Scalars
        current_piece = self.game.current_piece
        piece_map = {"L": 0, "S": 1, "c": 2, "T": 3, "I": 4, "u": 5, "b": 6}
        max_pieces = 2 * len(piece_map)  # move this logic to game
        current_piece_id = float(piece_map[current_piece.shape_name]) / len(
            piece_map
        )  # Normalised
        remaining_ratio = float(len(self.game.current_player.pieces)) / max_pieces

        other_scalars = np.array(
            [
                float(
                    self.game.current_player.value
                ),  # Allow player to know its value in the grid
                float(current_piece.x) / self.game.GRID_SIZE,  # Normalize x position
                float(current_piece.y) / self.game.GRID_SIZE,  # Normalize y position
                current_piece_id,  # Categorical encoding of piece type
                remaining_ratio,  # Percentage of pieces left
                float(self.game.status == Game.GAMEOVER),  # Game state flag,
            ],
            dtype=np.float32,
        )
        # 2. Get your 16-element shape array
        shape_vals = self._get_padded_shape(current_piece.shape)

        remaining_actions_ratio = (self.max_steps_by_turn - self.steps_for_current_turn) / self.max_steps_by_turn

        # 3. Use concatenate to merge them into a single (25,) array
        scalars = np.concatenate([
            np.array(other_scalars, dtype=np.float32), 
            [remaining_actions_ratio],
            shape_vals
        ])

        return {"grid": grid_array, "scalars": scalars}

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

    def _calculate_reward(
        self, player_idx: int, action: int, terminated: bool
    ) -> float:
        """
        Calculate reward for the current action.

        Path-finding wins are more valuable as they require strategic placement.
        """
        if action == Actions.ACTION_PASS:
                return -100.0  # Large negative reward for giving up
        
        if terminated:
            if (
                self.game.winner
                and self.game.players.index(self.game.winner) == player_idx
            ):
                # Player won
                if self.game.win_type == "path":
                    return 2000.0  # Higher reward for path-finding victory
                else:
                    return 1500.0  # Standard reward for score-based victory
        
        if action == Actions.ACTION_DROP:
            return 5.0  # Small reward for successfully dropping a piece
        return -0.1

    def close(self):
        """Clean up resources."""
        pass

    def seed(self, seed=None):
        """Set the random seed."""
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
