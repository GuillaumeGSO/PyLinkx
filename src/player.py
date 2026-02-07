# Player class for PyLinkx

from piece import Piece, TETRIS_SHAPES

class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.pieces = [Piece(shape, self) for shape in TETRIS_SHAPES]
        # Add more player attributes as needed
