from inspect import stack
import random
from piece import Piece, TETRIS_SHAPES


class Player:
    def __init__(self, name, color):
        self.name = name
        self.score = 0
        self.color = color
        self.pieces = [Piece(shape, self) for shape in TETRIS_SHAPES]
        self.pieces.extend([Piece(shape, self) for shape in TETRIS_SHAPES])
        random.shuffle(self.pieces)
        self.piece_index = 0
        # Add more player attributes as needed

    def drop_piece(self, piece: Piece):
        self.pieces.remove(piece)

    def next_piece(self):
        self.piece_index = self.piece_index % len(self.pieces)
        piece = self.pieces[self.piece_index]

        self.piece_index = (self.piece_index + 1) % len(self.pieces)
        print("New piece selected:")
        print(piece)

        return piece

    def has_pieces(self):
        return len(self.pieces) > 0

    def check_for_winner(self, grid, turn):
        rows = len(grid)
        cols = len(grid[0])

        # --- 1. CHECK HORIZONTAL WIN (Left to Right) ---
        stack = [(r, 0) for r in range(rows) if grid[r][0] == turn]
        visited = set(stack)

        while stack:
            r, c = stack.pop()
            if c == cols - 1:
                return True  # Reached the right edge

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        if grid[nr][nc] == turn:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

        # --- 2. CHECK VERTICAL WIN (Top to Bottom) ---
        stack = [(0, c) for c in range(cols) if grid[0][c] == turn]
        visited = set(stack)

        while stack:
            r, c = stack.pop()
            if r == rows - 1:
                return True  # Reached the bottom edge

            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                        if grid[nr][nc] == turn:
                            visited.add((nr, nc))
                            stack.append((nr, nc))

        return False

    def calculate_score(self, grid, turn):
        rows, cols = len(grid), len(grid[0])
        max_area = 0

        # We use a copy to avoid destroying the original grid data
        temp_grid = [row[:] for row in grid]

        for r in range(rows):
            for c in range(cols):
                # When we hit a 1, we've found a new zone
                if temp_grid[r][c] == turn:
                    current_area = 0
                    stack = [(r, c)]
                    temp_grid[r][c] = 0  # Mark as visited
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        current_area += 1
                        
                        # Explore 4-directional neighbors
                        for dr in [-1, 0, 1]:
                            for dc in [-1, 0, 1]:
                                nr, nc = curr_r + dr, curr_c + dc
                                if (0 <= nr < rows and 0 <= nc < cols and 
                                    temp_grid[nr][nc] == turn):
                                    temp_grid[nr][nc] = 0
                                    stack.append((nr, nc))
                    
                    # Update max if this new zone is strictly larger
                    if current_area > max_area:
                        max_area = current_area
                    
        return max_area