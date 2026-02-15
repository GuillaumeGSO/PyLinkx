from src.game import Game
from src.piece import Piece


def test_L_shape_floating_left_is_invalid():
    game = Game()
    piece = Piece("c", game.players[0])
    piece.shape = [[1, 1], [0, 1]]
    assert not game.is_valid_move(piece, 0, 0)


def test_L_shape_supported_is_valid():
    game = Game()
    piece = Piece("c", game.players[0])
    piece.shape = [[1, 0], [1, 1]]
    assert game.is_valid_move(piece, 0, 0)


def test_T_shape_is_valid_on_empty_grid():
    game = Game()
    piece = Piece("T", game.players[0])
    piece.shape = [[0, 1, 0], [1, 1, 1]]

    assert game.is_valid_move(piece, 0, 0)


def test_L_shape_allowed_only_on_rightmost_column():
    game = Game()
    game.grid[8] = [1, 1, 1, 1, 1, 1, 1, 1, 0]
    piece = Piece("c", game.players[0])
    piece.shape = [[1, 1], [0, 1]]
    for x in range(0, 7):
        assert not game.is_valid_move(piece, x, 7)

    assert game.is_valid_move(piece, 7, 7)


def test_overlap_is_invalid():
    game = Game()
    game.grid[7][7] = 1
    piece = Piece("u", game.players[0])
    piece.shape = [[1]]
    assert not game.is_valid_move(piece, 7, 7)


def test_out_of_bounds_is_invalid():
    game = Game()
    piece = Piece("b", game.players[0])
    piece.shape = [[1, 1]]
    assert not game.is_valid_move(piece, 8, 0)  # out of right bounds
    assert not game.is_valid_move(piece, 0, 9)  # out of bottom bounds
