# Piece (Pawn) class and Tetris shapes for PyLinkx

TETRIS_SHAPES: dict[str, list[list[int]]] = {
    'L': [[0, 0, 1], [1, 1, 1]],
    'S': [[0, 1, 1], [1, 1, 0]],
    'c': [[0, 1], [ 1, 1]],
    'T': [[0, 1, 0], [1, 1, 1]],
    'I': [[1, 1, 1, 1]],
    'u': [[1]],
    'b': [[1, 1]]
}
def flip_shape(shape):
    return [list(row[::-1]) for row in shape]

def rotate_shape(shape):
    return [list(row) for row in zip(*shape[::-1])]

class Piece:
    def __init__(self, shape_name, owner):
        self.shape_name = shape_name
        self.shape = TETRIS_SHAPES[shape_name]
        #self.owner = owner  # Player instance, should use type Player
        self.color = owner.color
        # self.played = False
        
    def rotate(self):
        self.shape = rotate_shape(self.shape)
    
    def flip(self):
        self.shape = flip_shape(self.shape)
        return self.shape

    def width(self):
        return len(self.shape[0])
    
    def height(self):
        return len(self.shape)