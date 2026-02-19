from src.piece import Piece
from src.player import Player


def test_next_piece_loop_normal():
    player = Player("one", 1, None)
    piece0 = Piece("c", player)
    piece1 = Piece("u", player)
    piece2 = Piece("L", player)
    player.piece_index = 0
    player.pieces = [piece0, piece1, piece2]
    result = player.next_piece()
    assert result == piece1
    assert player.piece_index == 1
    result = player.next_piece()
    assert result == piece2
    assert player.piece_index == 2
    result = player.next_piece()
    assert result == piece0
    assert player.piece_index == 0
    result = player.next_piece()
    assert result == piece1
    assert player.piece_index == 1
    


