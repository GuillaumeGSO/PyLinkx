# Game logic for PyLinkx

from src.player import Player


class Game:
    def __init__(self):
        # Initialize game state here
        self.reset()
        self.players = [
            Player("Player 1", (255, 215, 0)),  # Yellow
            Player("Player 2", (220, 20, 60)),  # Red
        ]
        # Add more game state as needed

    def __repr__(self) -> str:
        # Simple text representation of the game state
        for row in self.grid:
            print(row)
        return ""

    def update(self):
        # Update game state each frame
        pass

    def reset(self):
        # Reset the game state
        self.grid = [[0 for _ in range(9)] for _ in range(9)]

    def is_valid_move(self, shape, grid_x, grid_y):
        shape_cells = set()
        shape_height = len(shape)
        shape_width = len(shape[0])

        # 1. Bounds & overlap check
        for r, row in enumerate(shape):
            for c, value in enumerate(row):
                if value == 1:
                    tx, ty = grid_x + c, grid_y + r

                    if not (0 <= tx < 9 and 0 <= ty < 9):
                        return False
                    if self.grid[ty][tx] > 0:
                        return False

                    shape_cells.add((tx, ty))

        # 2. Support check: bottom-most block per column
        for c in range(shape_width):
            # find lowest block in this shape column
            for r in reversed(range(shape_height)):
                if shape[r][c] == 1:
                    tx = grid_x + c
                    ty = grid_y + r
                    below_y = ty + 1

                    # bottom of grid = supported
                    if below_y >= 9:
                        break

                    # supported by grid
                    if self.grid[below_y][tx] > 0:
                        break

                    # otherwise unsupported
                    return False

        return True

    def place_piece(self, shape, grid_x, grid_y, turn):
        for row_idx, row in enumerate(shape):
            for col_idx, value in enumerate(row):
                if value == 1:
                    # Store the player's ID (turn + 1) to remember who placed it
                    self.grid[grid_y + row_idx][grid_x + col_idx] = turn + 1
