# Game logic for PyLinkx

from player import Player
from piece import Piece


class Game:
    PLAYING = "playing"
    GAMEOVER = "gameover"

    def __init__(self):
        # Initialize game state here
        self.reset()
        self.players = [
            Player("Player 1", (255, 215, 0)),  # Yellow
            Player("Player 2", (220, 20, 60)),  # Red
            # Player("Player 3", (0, 128, 0)),  # Green
        ]
        # Add more game state as needed
        self.current_piece: Piece

    def reset(self):
        # Reset the game state
        self.grid = [[0 for _ in range(9)] for _ in range(9)]
        self.status = Game.PLAYING
        self.winner = None

    def __repr__(self) -> str:
        # Simple text representation of the game state
        for row in self.grid:
            print(row)
        return f"Game State: ${self.status}"

    def play_drop_piece(self, piece: Piece, grid_x, turn):
        ghost_grid_y = self.calculate_ghost_position(piece, grid_x)
        if ghost_grid_y is not None:
            self.place_piece_on_grid(piece, grid_x, ghost_grid_y, turn)
            self.players[turn].drop_piece(piece)
            self.winner = self.check_for_winner()
            if self.winner:
                self.status = Game.GAMEOVER

    def move_piece_left(self, piece: Piece):
        if piece.x > 0:
            piece.move_left()

    def move_piece_right(self, piece: Piece):
        if piece.x < 9 - piece.width():
            piece.move_right()
            
    def rotate_piece(self, piece: Piece):
        piece.rotate()
        # Ensure the piece doesn't go out of bounds after rotation
        if piece.x + piece.width() > 9:
            piece.x = 9 - piece.width()

    def calculate_ghost_position(self, piece: Piece, grid_x):
        ghost_grid_y = None
        if self.is_valid_move(piece, grid_x, 0):
            for y_test in range(9 - piece.height() + 1):
                if self.is_valid_move(piece, grid_x, y_test):
                    ghost_grid_y = y_test
                else:
                    break

        if ghost_grid_y is not None and not self.is_fully_supported(
            piece, grid_x, ghost_grid_y
        ):
            ghost_grid_y = None
        return ghost_grid_y

    def update(self):
        print("Updating game state...")
        print(self)
        self.check_for_draw()

    def check_for_winner(self):
        for player in self.players:
            if player.check_for_winner(self.grid, self.players.index(player) + 1):
                self.status = Game.GAMEOVER
                return player

    def check_for_draw(self):
        # print("Checking for draw...")
        pass

    def is_valid_move(self, piece: Piece, grid_x, grid_y):
        shape_cells = set()

        # 1. Bounds & overlap check
        for r, row in enumerate(piece.shape):
            for c, value in enumerate(row):
                if value == 1:
                    tx, ty = grid_x + c, grid_y + r

                    if not (0 <= tx < 9 and 0 <= ty < 9):
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

    def place_piece_on_grid(self, piece, grid_x, grid_y, turn):
        for row_idx, row in enumerate(piece.shape):
            for col_idx, value in enumerate(row):
                if value == 1:
                    # Store the player's ID (turn + 1) to remember who placed it
                    self.grid[grid_y + row_idx][grid_x + col_idx] = turn + 1
