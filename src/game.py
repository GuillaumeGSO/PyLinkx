# Game logic for PyLinkx

from player import Player
from piece import Piece


class Game:
    def __init__(self):
        # Initialize game state here
        self.reset()
        self.players = [
            Player("Player 1", (255, 215, 0)),  # Yellow
            Player("Player 2", (220, 20, 60)),  # Red
            # Player("Player 3", (0, 128, 0)),  # Green
        ]
        # Add more game state as needed

    def __repr__(self) -> str:
        # Simple text representation of the game state
        for row in self.grid:
            print(row)
        return ""

    def update(self):
        print(self)
        # Update game state each frame
        self.check_for_winner(1)
        self.check_for_draw()

    def reset(self):
        # Reset the game state
        self.grid = [[0 for _ in range(9)] for _ in range(9)]
        self.running = True

    def check_for_winner(self, turn):
        pass
        grid = self.grid
        rows = len(grid)
        cols = len(grid[0])

        start_nodes = [(r, 0) for r in range(rows) if grid[r][0] == turn]
        if not start_nodes:
            return False

        stack = start_nodes
        visited = set(start_nodes)

        while stack:
            r, c = stack.pop()
            if c == cols - 1:
                return True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue 
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if grid[nr][nc] == turn and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
        return False

    
    def check_for_draw(self):
        print("Checking for draw...")
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
