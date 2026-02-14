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
        self.x = 0
        self.y = 0
        # self.played = False
    
    def __repr__(self):
        for row in self.shape:
            print("   " * self.x + str(row))
        return f"{self.x}"
    
    def rotate(self):
        self.shape = rotate_shape(self.shape)
        print(self)

    def flip(self):
        self.shape = flip_shape(self.shape)
        print(self)
    
    def move_left(self):
        self.x -= 1
        print(self)
    
    def move_right(self):
        self.x += 1
        print(self)
        
    def width(self):
        return len(self.shape[0])
    
    def height(self):
        return len(self.shape)