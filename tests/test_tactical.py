from src.game.game import Game
from src.game.piece import Piece
from src.inference.tactical import (
    Placement,
    enumerate_placements,
    execute_placement,
    find_tactical_move,
)


def _fresh_game_with_p1_piece(shape_name: str) -> Game:
    """Game with P1 having exactly one piece of the given shape, turn started."""
    game = Game()
    game.players[0].pieces = [Piece(shape_name, game.players[0])]
    game.players[0].piece_index = -1
    game.start_turn()
    return game


def test_finds_horizontal_win():
    game = _fresh_game_with_p1_piece("u")
    # P1 has row 8 cols 0..7; needs (8, 8) to complete left-to-right path.
    game.grid[8] = [1, 1, 1, 1, 1, 1, 1, 1, 0]

    move = find_tactical_move(game, player_idx=0)

    assert move is not None
    assert move.x == 8
    assert move.ghost_y == 8
    assert move.shape == ((1,),)


def test_finds_vertical_win():
    game = _fresh_game_with_p1_piece("u")
    # P1 stack in col 4, rows 1..8, supported by floor. Needs (0, 4) for top-to-bottom.
    for r in range(1, 9):
        game.grid[r][4] = 1

    move = find_tactical_move(game, player_idx=0)

    assert move is not None
    assert move.x == 4
    assert move.ghost_y == 0


def test_blocks_opponent_vertical_threat():
    game = _fresh_game_with_p1_piece("u")
    game.players[1].pieces = [Piece("u", game.players[1])]
    game.players[1].piece_index = -1
    # P2 stack col 4 rows 1..8 — P2 wins by dropping 'u' at (4, 0).
    for r in range(1, 9):
        game.grid[r][4] = 2

    move = find_tactical_move(game, player_idx=0)

    assert move is not None
    # The only blocking placement is at (4, 0) — occupying the cell P2 needs.
    assert (move.x, move.ghost_y) == (4, 0)


def test_no_tactical_move_on_empty_board():
    game = Game()
    game.start_turn()

    assert find_tactical_move(game, player_idx=0) is None


def test_prefers_win_over_block():
    game = _fresh_game_with_p1_piece("u")
    game.players[1].pieces = [Piece("u", game.players[1])]
    game.players[1].piece_index = -1
    # P1 stack col 2 rows 1..8 — P1 wins at (2, 0).
    # P2 stack col 6 rows 1..8 — P2 threatens to win at (6, 0).
    # P1's win and P2's threat are in different columns, so a non-winning
    # block does not exist; the tactical layer must still return the win.
    for r in range(1, 9):
        game.grid[r][2] = 1
        game.grid[r][6] = 2

    move = find_tactical_move(game, player_idx=0)

    assert move is not None
    assert (move.x, move.ghost_y) == (2, 0)
    # Confirm the returned placement is genuinely a win for P1.
    for r, row in enumerate(move.shape):
        for c, v in enumerate(row):
            if v == 1:
                game.grid[move.ghost_y + r][move.x + c] = game.players[0].value
    assert game.players[0].check_if_winner(game.grid)


def test_execute_placement_applies_drop():
    game = _fresh_game_with_p1_piece("u")
    target = Placement(game.current_piece, ((1,),), x=3, ghost_y=8)

    ok = execute_placement(game, target)

    assert ok is True
    assert game.grid[8][3] == 1
    # Turn advanced to P2.
    assert game.players.index(game.current_player) == 1


def test_enumerate_placements_deduplicates_single_cell_piece():
    game = _fresh_game_with_p1_piece("u")

    placements = enumerate_placements(game, game.players[0])

    # 'u' has one distinct orientation and lands on the floor at every column.
    xs = sorted(p.x for p in placements)
    assert xs == list(range(game.GRID_SIZE))
    assert all(p.ghost_y == 8 for p in placements)
    assert all(p.shape == ((1,),) for p in placements)
