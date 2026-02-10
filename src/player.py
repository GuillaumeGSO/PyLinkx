from piece import Piece, TETRIS_SHAPES


class Player:
    def __init__(self, name, color):
        self.name = name
        self.score = 0
        self.color = color
        self.pieces = [Piece(shape, self) for shape in TETRIS_SHAPES]
        self.pieces.extend([Piece(shape, self) for shape in TETRIS_SHAPES])
        self.index= 0
        # Add more player attributes as needed

    def drop_piece(self, piece: Piece):
        print(f"Dropping piece {piece.shape_name} for player {self.name}")
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
