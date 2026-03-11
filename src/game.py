# Game logic for PyLinkx
import random

import numpy as np
from pygame import mask
from player import Player
from piece import Piece


class Game:
    PLAYING = "playing"
    GAMEOVER = "gameover"
    GRID_SIZE = 9

    def __init__(self):
        # Initialize game state here
        self.players = []
        self.current_piece: Piece
        self.current_player: Player
        self.reset()

    def reset(self):
        # Reset the game state
        self.grid = [[0 for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.status = Game.PLAYING
        self.players = [
            Player("Player 1", 1, (255, 215, 0)),  # Yellow
            Player("Player 2", 2, (220, 20, 60)),  # Red
            # Player("Player 3", 3, (0, 128, 0)),  # Green
        ]
        self.current_player = self.players[0]
        self.winner = None
        self.win_type = None  # 'path' or 'score'
        self.ghost_grid_y = None

    def __repr__(self) -> str:
        for row in self.grid:
            print(row)
        return f"Game State: ${self.status}"

    def set_current_piece(self, piece: Piece | None):
        if piece is None:
            return
        self.current_piece = piece
        self.current_piece.x = random.randint(0, self.GRID_SIZE - piece.width())
        self.ghost_grid_y = self.calculate_ghost_position(self.current_piece)

    def get_players_in_play(self):
        return [player for player in self.players if not player.has_gave_up]

    def play_drop_piece(self, piece: Piece, player: Player):
        self.ghost_grid_y = self.calculate_ghost_position(piece)
        if self.ghost_grid_y is not None:
            self.place_piece_on_grid(piece, piece.x, self.ghost_grid_y, player)
            self.current_player.drop_piece(piece)
            if not self.current_player.has_pieces():
                self.current_player.give_up()
            self.winner = self.check_for_winner()
            if self.winner or len(self.get_players_in_play()) == 0:
                self.status = Game.GAMEOVER
            return True
        return False
    
    def can_move_piece(self, piece: Piece, dx: int) -> bool:
        new_x = piece.x + dx
        if new_x < 0 or new_x + piece.width() > self.GRID_SIZE:
            return False
        return True

    def move_piece_left(self, piece: Piece) -> bool:
        can_move = self.can_move_piece(piece, dx=-1)
        if can_move:
            piece.move_left()
        return can_move

    def move_piece_right(self, piece: Piece) -> bool:
        can_move = self.can_move_piece(piece, dx=1)
        if can_move:
            piece.move_right()
        return can_move

    def rotate_piece(self, piece: Piece) -> bool:
        if piece.shape_name in ["u"]:
            return False
        piece.rotate()
        # Ensure the piece doesn't go out of bounds after rotation
        if piece.x + piece.width() > self.GRID_SIZE:
            piece.x = self.GRID_SIZE - piece.width()
        return True
    
    def flip_piece(self, piece: Piece) -> bool:
        prev_piece_shape = piece.shape
        piece.flip()
        if piece.shape == prev_piece_shape:
            return False
        return True

    def give_up_and_check(self, player: Player):
        player.give_up()
        if self.get_players_in_play() == []:
            self.status = Game.GAMEOVER

    def calculate_ghost_position(self, piece: Piece)-> int | None:
        ghost_grid_y = None
        if self.is_valid_move(piece, piece.x, 0):
            for y_test in range(self.GRID_SIZE - piece.height() + 1):
                if self.is_valid_move(piece, piece.x, y_test):
                    ghost_grid_y = y_test
                else:
                    break

        if ghost_grid_y is not None and not self.is_fully_supported(
            piece, piece.x, ghost_grid_y
        ):
            ghost_grid_y = None
        return ghost_grid_y

    def update(self):
        # print("Updating game state...")
        # print(self)
        self.ghost_grid_y = self.calculate_ghost_position(self.current_piece)
        self.update_scores()
        self.winner = self.check_for_winner()

    def check_for_winner(self):
        # First, check for path-finding win (higher reward)
        for player in self.players:
            if player.check_if_winner(self.grid):
                self.status = Game.GAMEOVER
                self.winner = player
                self.win_type = "path"
                return player

        # Second, check for score-based win when all players are out
        remaining_players = self.get_players_in_play()
        if not remaining_players:
            # All players have given up or run out of pieces
            # Winner is the one with highest score
            self.update_scores()  # Ensure scores are up to date
            max_score = max(player.score for player in self.players)
            winners = [p for p in self.players if p.score == max_score]

            # If there's a tie, first player wins (could be randomized)
            self.winner = winners[0]
            self.win_type = "score"
            self.status = Game.GAMEOVER
            return self.winner

        return None

    def get_next_player(self) -> Player:  # type: ignore
        remaining_players = self.get_players_in_play()
        if not remaining_players:
            self.status = Game.GAMEOVER
            return self.current_player

        for p in range(len(self.players)):
            if self.players[p] == self.current_player:
                next_index = (p + 1) % len(self.players)
                while self.players[next_index].has_gave_up:
                    next_index = (next_index + 1) % len(self.players)
                return self.players[next_index]

    def update_scores(self):
        for player in self.players:
            player.score = player.calculate_score(self.grid)

    def is_valid_move(self, piece: Piece, grid_x, grid_y):
        shape_cells = set()

        # 1. Bounds & overlap check
        for r, row in enumerate(piece.shape):
            for c, value in enumerate(row):
                if value == 1:
                    tx, ty = grid_x + c, grid_y + r

                    if not (0 <= tx < self.GRID_SIZE and 0 <= ty < self.GRID_SIZE):
                        return False
                    if self.grid[ty][tx] > 0:
                        return False

                    shape_cells.add((tx, ty))

        return True

    def is_fully_supported(self, piece: Piece, grid_x, grid_y):
        shape_height = piece.height()
        shape_width = piece.width()
        grid_height = len(self.grid)

        for c in range(shape_width):
            # 1. Find the lowest block in this specific column of the shape
            lowest_r = -1
            for r in reversed(range(shape_height)):
                if piece.shape[r][c] == 1:
                    lowest_r = r
                    break

            # 2. If this column of the shape is empty (no blocks), skip to next column
            if lowest_r == -1:
                continue

            # 3. Calculate the position in the grid directly below this block
            tx = grid_x + c
            ty = grid_y + lowest_r
            below_y = ty + 1

            # 4. Check Support:
            # If it's touching the floor, it's supported
            if below_y >= grid_height:
                continue  # This column is supported by the floor

            # If there is a block in the grid below it, it's supported
            if self.grid[below_y][tx] > 0:
                continue  # This column is supported by another block

            # 5. If we reach here, this specific column has air underneath it
            return False

        # If we checked all columns and none returned False, the whole piece is supported
        return True

    def place_piece_on_grid(self, piece, grid_x, grid_y, player: Player):
        for row_idx, row in enumerate(piece.shape):
            for col_idx, value in enumerate(row):
                if value == 1:
                    self.grid[grid_y + row_idx][grid_x + col_idx] = player.value

    # ===== RL/Programmatic Interface Methods =====

    def get_observation(self) -> dict:
        """
        Returns the current game state as an observation dictionary.
        Suitable for RL agents to receive state information.
        """
        return {
            "grid": [row[:] for row in self.grid],  # Copy of grid
            "current_player_idx": self.players.index(self.current_player),
            "scores": [player.score for player in self.players],
            "current_piece": (
                self.current_piece if hasattr(self, "current_piece") else None
            ),
            "is_game_over": self.status == Game.GAMEOVER,
            "winner_idx": self.players.index(self.winner) if self.winner else None,
            "win_type": self.win_type,  # 'path', 'score', or None
        }
    
    def get_valid_actions(self, num_actions: int) -> np.ndarray:
        from game_env import Actions
        mask = np.ones(num_actions, dtype=np.int8)
        for action in range(num_actions):
            # We only check 'can' we do it, we don't 'do' it
            can_do = False
            if action == Actions.ACTION_CYCLE_PIECE:
                can_do = True # Usually always valid
            elif action == Actions.ACTION_MOVE_LEFT:
                can_do = self.can_move_piece(self.current_piece, dx=-1)
            elif action == Actions.ACTION_MOVE_RIGHT:
                can_do = self.can_move_piece(self.current_piece, dx=1)
            elif action == Actions.ACTION_ROTATE:
                can_do = self.can_rotate(self.current_piece)
            elif action == Actions.ACTION_FLIP:
                can_do = self.can_flip(self.current_piece)
            elif action == Actions.ACTION_DROP:               
                can_do = self.ghost_grid_y is not None
            elif action == Actions.ACTION_PASS:
                can_do = False
            
            if not can_do:
                mask[action] = 0
            
        return mask
    
    def can_rotate(self, piece: Piece) -> bool:
        return piece.shape_name not in ["u"]
    
    def can_flip(self, piece: Piece) -> bool:
        return piece.shape_name in ["L", "S", "c"]
        
    def execute_action(self, action: int) -> bool:
        """
        Executes an action on the current piece or player state.
        Returns True if action was valid and executed, False otherwise.
        """
        from game_env import Actions

        if action == Actions.ACTION_PASS:  # pass/give_up
            self.give_up_and_check(self.current_player)
            self.current_player = self.get_next_player()
            self.set_current_piece(self.current_player.next_piece())
            return True

        if not hasattr(self, "current_piece"):
            return False

        if action == Actions.ACTION_CYCLE_PIECE:  # select next piece
            self.set_current_piece(self.current_player.next_piece())
            return True
        elif action == Actions.ACTION_MOVE_LEFT:
            return self.move_piece_left(self.current_piece)
        elif action == Actions.ACTION_MOVE_RIGHT:
            return self.move_piece_right(self.current_piece)
        elif action == Actions.ACTION_ROTATE:  # rotate
            return self.rotate_piece(self.current_piece)
        elif action == Actions.ACTION_FLIP:  # flip horizontally
            return self.flip_piece(self.current_piece)
        elif action == Actions.ACTION_DROP:  # drop
            success = self.play_drop_piece(self.current_piece, self.current_player)
            if success:
                self.current_player = self.get_next_player()
                self.set_current_piece(self.current_player.next_piece())
                return success
        return False

    def can_move_piece(self, piece: Piece, dx: int) -> bool:
        new_x = piece.x + dx
        return self.is_valid_move(piece, new_x, piece.y)
    
    def reset_piece_position(self):
        """Reset the current piece to starting position (x=0, y=0)."""
        if hasattr(self, "current_piece"):
            self.current_piece.x = 0
            self.current_piece.y = 0
