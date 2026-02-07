# Piece (Pawn) class and Tetris shapes for PyLinkx

TETRIS_SHAPES = {
    'I': [[1, 1, 1, 1]],
    'O': [[1, 1], [1, 1]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'Z': [[1, 1, 0], [0, 1, 1]],
    'J': [[1, 0, 0], [1, 1, 1]],
    'L': [[0, 0, 1], [1, 1, 1]],
}

class Piece:
    def __init__(self, shape_name, owner):
        self.shape_name = shape_name
        self.shape = TETRIS_SHAPES[shape_name]
        self.owner = owner  # Player instance
        # You can add color or other attributes here
