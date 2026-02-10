from piece import Piece, TETRIS_SHAPES


class Player:
    def __init__(self, name, color):
        self.name = name
        self.score = None
        self.color = color
        self.pieces = [Piece(shape, self) for shape in TETRIS_SHAPES]
        self.pieces.extend([Piece(shape, self) for shape in TETRIS_SHAPES])
        self.index= 0
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
    
    def check_for_winner(self, grid, val):
        rows = len(grid)
        cols = len(grid[0])

        start_nodes = [(r, 0) for r in range(rows) if grid[r][0] == val]
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
                        if grid[nr][nc] == val and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            stack.append((nr, nc))
        return False
