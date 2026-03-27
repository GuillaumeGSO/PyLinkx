# Gymnasium RL Environment for PyLinkx
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game import Game, Actions
from game_renderer import GameRenderer


class PyLinkxEnv(gym.Env):
    """
    Gymnasium environment wrapper for the PyLinkx game.

    Single-agent mode: Agent plays as Player 1. Player 2 is controlled by
    a frozen opponent model or a drop-first fallback policy.
    """

    metadata = {"render_modes": ["debug"], "render_fps": 8}
    PIECE_MAP = {"L": 0, "S": 1, "c": 2, "T": 3, "I": 4, "u": 5, "b": 6}

    def __init__(self, render_mode=None, max_steps=500, max_steps_by_turn=100,
                 opponent_model_path=None):
        """
        Initialize the PyLinkx Gymnasium environment.

        Args:
            render_mode: Rendering mode (None or "debug")
            max_steps: Maximum steps per episode to prevent infinite loops
            max_steps_by_turn: Maximum steps allowed per turn before a drop is forced
            opponent_model_path: Path to a saved MaskablePPO model for P2.
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
        if opponent_model_path and os.path.exists(opponent_model_path):
            from sb3_contrib import MaskablePPO
            self._opponent_model = MaskablePPO.load(opponent_model_path)

        # Action space: 6 discrete actions (0-5)
        self.action_space = spaces.Discrete(len(Actions))

        # Observation space: grid (9x9, 1 channel) + 34 scalar features
        # Grid: 9x9 cells with values normalized to [0.0, 0.5, 1.0]
        # Scalars: player value, piece x, scores, can_drop flag, piece id,
        #          remaining ratio, game over flag, action validity,
        #          remaining turn ratio, padded piece shape (4x4=16),
        #          path progress: p1(h, v, best, area), p2(h, v, best, area)
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0.0, high=1.0, shape=(9, 9, 1), dtype=np.float32),
                "scalars": spaces.Box(low=-1.0, high=1.0, shape=(34,), dtype=np.float32),
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
            self._play_opponent_turn()
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
        piece = self.game.current_piece if hasattr(self.game, "current_piece") else None
        return np.array([
            1,  # CYCLE — always valid
            int(piece is not None and self.game.can_move_piece(piece, dx=-1)),
            int(piece is not None and self.game.can_move_piece(piece, dx=1)),
            int(piece is not None and self.game.can_rotate(piece)),
            int(piece is not None and self.game.can_flip(piece)),
            int(self.game.can_drop()),
        ], dtype=np.int8)

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

    def _get_padded_shape(self, shape: list[list[int]]) -> np.ndarray:
        """Pads any piece shape into a fixed 4x4 array."""
        padded = np.zeros((4, 4), dtype=np.float32)
        rows = len(shape)
        cols = len(shape[0])
        # Place the shape in the top-left of the 4x4 grid
        padded[:rows, :cols] = np.array(shape)
        return padded.flatten()  # Returns 16 scalars

    def _get_path_progress_scalars(self) -> np.ndarray:
        """
        Returns 8 continuous float32 values representing BFS path progress per player.
        [p1_h, p1_v, p1_best, p1_area, p2_h, p2_v, p2_best, p2_area]
        h/v: fraction of grid crossed in horizontal/vertical direction (0–1).
        best: max(h, v). area: largest contiguous group / 81.
        """
        grid_cells = float(self.game.GRID_SIZE * self.game.GRID_SIZE)
        result = []
        for i, player in enumerate(self.game.players):
            h, v = self._path_progress[i]
            result.extend([h, v, max(h, v), float(player.score) / grid_cells])
        return np.array(result, dtype=np.float32)

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

        # Concatenate into a single (34,) array: 9 scalars + 1 ratio + 16 shape + 8 path progress
        scalars = np.concatenate([
            other_scalars,
            [remaining_actions_ratio],
            shape_vals,
            self._get_path_progress_scalars(),  # idx 26-33: p1/p2 BFS path progress
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
            "p2_drops": self._p2_drops,
        }

    def _compute_path_progress(self, player_idx: int) -> tuple[float, float]:
        """
        BFS path progress for a player. Returns (h_progress, v_progress).

        Horizontal (symmetric): best of left→right or right→left frontier, normalized to [0, 1].
        Vertical (bottom-only): how far up from the bottom the connected component reaches.
        """
        player_val = self.game.players[player_idx].value
        grid = self.game.grid
        G = self.game.GRID_SIZE
        g = G - 1

        # Horizontal: seed from left edge, track max col reached
        h_from_left = -1
        start = [(r, 0) for r in range(G) if grid[r][0] == player_val]
        if start:
            visited = set(start)
            stack = list(start)
            while stack:
                r, c = stack.pop()
                if c > h_from_left:
                    h_from_left = c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in visited and grid[nr][nc] == player_val:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

        # Horizontal: seed from right edge, track min col reached
        h_from_right = G
        start = [(r, g) for r in range(G) if grid[r][g] == player_val]
        if start:
            min_col = g
            visited = set(start)
            stack = list(start)
            while stack:
                r, c = stack.pop()
                if c < min_col:
                    min_col = c
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in visited and grid[nr][nc] == player_val:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
            h_from_right = min_col

        h_progress = max(
            h_from_left / g if h_from_left >= 0 else 0.0,
            (g - h_from_right) / g if h_from_right < G else 0.0,
        )

        # Vertical: seed from bottom edge only, track min row reached
        v_progress = 0.0
        start = [(g, c) for c in range(G) if grid[g][c] == player_val]
        if start:
            min_row = g
            visited = set(start)
            stack = list(start)
            while stack:
                r, c = stack.pop()
                if r < min_row:
                    min_row = r
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in visited and grid[nr][nc] == player_val:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
            v_progress = (g - min_row) / g

        return (h_progress, v_progress)

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
