from piece import Piece, TETRIS_SHAPES


class Player:
    def __init__(self, name, color):
        self.name = name
        self.score = None
        self.color = color
        self.pieces = [Piece(shape, self) for shape in TETRIS_SHAPES]
        self.pieces.extend([Piece(shape, self) for shape in TETRIS_SHAPES])
        self.index = 0
        # Add more player attributes as needed

    def drop_piece(self, piece: Piece):
        self.pieces.remove(piece)

    def next_piece(self):
        # Sécurité : si des éléments ont été supprimés, on ajuste l'index
        self.index = self.index % len(self.pieces)
        piece = self.pieces[self.index]

        # On avance l'index pour le prochain appel
        self.index = (self.index + 1) % len(self.pieces)
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
