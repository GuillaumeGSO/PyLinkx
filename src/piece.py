# Piece (Pawn) class and Tetris shapes for PyLinkx



TETRIS_SHAPES = {
    'S': [[0, 1, 1], [1, 1, 0]],
    'L': [[0, 0, 1], [1, 1, 1]],
    'c': [[0, 1], [ 1, 1]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'I': [[1, 1, 1, 1]],
    'u': [[1]],
    'b': [[1, 1]]
}

class Piece:
    def __init__(self, shape_name, owner):
        self.shape_name = shape_name
        self.shape = TETRIS_SHAPES[shape_name]
        self.owner = owner  # Player instance, should use type Player
        self.color = owner.color
        # self.played = False
