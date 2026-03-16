# Game logic for PyLinkx
import random
from enum import IntEnum
from player import Player
from piece import Piece, rotate_shape, flip_shape


class Actions(IntEnum):
    ACTION_CYCLE_PIECE = 0
    ACTION_MOVE_LEFT = 1
    ACTION_MOVE_RIGHT = 2
    ACTION_ROTATE = 3
    ACTION_FLIP = 4
    ACTION_DROP = 5


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
        self.one_extra_turn_remaining = False

    def __repr__(self) -> str:
        rows = "\n".join(str(row) for row in self.grid)
        return f"Game State: {self.status}\n{rows}"

    def set_current_piece(self, piece: Piece | None):
        if piece is None:
            return
        self.current_piece = piece
        self.current_piece.x = random.randint(0, self.GRID_SIZE - piece.width())
        self.ghost_grid_y = self.calculate_ghost_position(self.current_piece)

    def get_players_in_play(self):
        return [player for player in self.players if not player.has_given_up]

    def play_drop_piece(self, piece: Piece, player: Player):
        self.ghost_grid_y = self.calculate_ghost_position(piece)
        if self.ghost_grid_y is not None:
            self.place_piece_on_grid(piece, piece.x, self.ghost_grid_y, player)
            self.current_player.drop_piece(piece)

            gave_up = False
            if not self.current_player.has_pieces():
                self.current_player.give_up()
                gave_up = True

            if self.one_extra_turn_remaining:
                # Bonus turn just used — end game (path win has priority)
                self.winner = self.check_for_winner()
                if not self.winner:
                    self._declare_score_winner()
                self.status = Game.GAMEOVER
            else:
                self.winner = self.check_for_winner()
                if self.winner or not self.get_players_in_play():
                    self.status = Game.GAMEOVER
                elif gave_up and self.get_players_in_play():
                    # Player just exhausted; give opponent one extra turn
                    self.one_extra_turn_remaining = True

            return True
        return False

    def can_move_piece(self, piece: Piece, dx: int) -> bool:
        new_x = piece.x + dx
        return 0 <= new_x and new_x + piece.width() <= self.GRID_SIZE

    def can_rotate(self, piece: Piece) -> bool:
        return piece.shape_name != "u"

    def can_flip(self, piece: Piece) -> bool:
        return flip_shape(piece.shape) != piece.shape

    def can_drop(self) -> bool:
        return self.ghost_grid_y is not None

    def move_piece_left(self, piece: Piece) -> bool:
        if self.can_move_piece(piece, dx=-1):
            piece.move_left()
            return True
        return False

    def move_piece_right(self, piece: Piece) -> bool:
        if self.can_move_piece(piece, dx=1):
            piece.move_right()
            return True
        return False

    def rotate_piece(self, piece: Piece) -> bool:
        if not self.can_rotate(piece):
            return False
        piece.rotate()
        # Ensure the piece doesn't go out of bounds after rotation
        if piece.x + piece.width() > self.GRID_SIZE:
            piece.x = self.GRID_SIZE - piece.width()
        return True

    def flip_piece(self, piece: Piece) -> bool:
        if not self.can_flip(piece):
            return False
        piece.flip()
        return True

    def give_up_and_check(self, player: Player):
        player.give_up()
        if self.get_players_in_play() == []:
            self.status = Game.GAMEOVER

    def calculate_ghost_position(self, piece: Piece):
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
        self.ghost_grid_y = self.calculate_ghost_position(self.current_piece)
        self.update_scores()
        self.winner = self.check_for_winner()

    def _declare_score_winner(self):
        self.update_scores()
        max_score = max(p.score for p in self.players)
        self.winner = next(p for p in self.players if p.score == max_score)
        self.win_type = "score"
        self.status = Game.GAMEOVER

    def check_for_winner(self):
        if self.status == Game.GAMEOVER:
            return self.winner
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
                while self.players[next_index].has_given_up:
                    next_index = (next_index + 1) % len(self.players)
                return self.players[next_index]

    def update_scores(self):
        for player in self.players:
            player.score = player.calculate_score(self.grid)

    def is_valid_move(self, piece: Piece, grid_x, grid_y):
        for r, row in enumerate(piece.shape):
            for c, value in enumerate(row):
                if value == 1:
                    tx, ty = grid_x + c, grid_y + r
                    if not (0 <= tx < self.GRID_SIZE and 0 <= ty < self.GRID_SIZE):
                        return False
                    if self.grid[ty][tx] > 0:
                        return False
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

    def player_has_valid_moves(self, player: Player) -> bool:
        for piece in player.pieces:
            orig_shape, orig_x = piece.shape, piece.x
            seen, shape = set(), orig_shape
            for _ in range(4):
                for candidate in [shape, flip_shape(shape)]:
                    key = tuple(tuple(r) for r in candidate)
                    if key not in seen:
                        seen.add(key)
                        piece.shape = candidate
                        for x in range(self.GRID_SIZE - len(candidate[0]) + 1):
                            piece.x = x
                            if self.calculate_ghost_position(piece) is not None:
                                piece.shape, piece.x = orig_shape, orig_x
                                return True
                shape = rotate_shape(shape)
            piece.shape, piece.x = orig_shape, orig_x
        return False

    def place_piece_on_grid(self, piece, grid_x, grid_y, player: Player):
        for row_idx, row in enumerate(piece.shape):
            for col_idx, value in enumerate(row):
                if value == 1:
                    self.grid[grid_y + row_idx][grid_x + col_idx] = player.value

    def execute_action(self, action: int) -> bool:
        """
        Executes an action on the current piece or player state.
        Returns success flag. Always calls update() before returning.
        """
        if not hasattr(self, "current_piece"):
            return False

        success = True

        if action == Actions.ACTION_CYCLE_PIECE:
            self.set_current_piece(self.current_player.next_piece())
        elif action == Actions.ACTION_MOVE_LEFT:
            success = self.move_piece_left(self.current_piece)
        elif action == Actions.ACTION_MOVE_RIGHT:
            success = self.move_piece_right(self.current_piece)
        elif action == Actions.ACTION_ROTATE:
            success = self.rotate_piece(self.current_piece)
        elif action == Actions.ACTION_FLIP:
            success = self.flip_piece(self.current_piece)
        elif action == Actions.ACTION_DROP:
            success = self.play_drop_piece(self.current_piece, self.current_player)
            if success:
                self.current_player = self.get_next_player()
                self.start_turn()
        else:
            return False

        self.update()
        return success

    def start_turn(self):
        """Set up the current player's next piece to begin their turn.

        Auto-passes a player who is blocked (has pieces but no valid placement),
        granting the opponent one extra turn. Recursion is bounded by GAMEOVER.
        """
        if self.status == Game.GAMEOVER:
            return

        next_piece = self.current_player.next_piece()
        if next_piece:
            self.set_current_piece(next_piece)
            if not self.player_has_valid_moves(self.current_player):
                self.current_player.give_up()
                remaining = self.get_players_in_play()
                if not remaining:
                    self.winner = self.check_for_winner()
                elif self.one_extra_turn_remaining:
                    self._declare_score_winner()
                else:
                    self.one_extra_turn_remaining = True
                    self.current_player = self.get_next_player()
                    self.start_turn()
        else:
            self.current_player.give_up()
