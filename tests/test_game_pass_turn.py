"""Tests for auto-pass + one extra turn rule."""
from src.game.game import Game
from src.game.piece import Piece


def _fill_grid(game, leave_open=None):
    """Fill entire grid with a neutral value (3), optionally leaving one cell open."""
    for r in range(game.GRID_SIZE):
        game.grid[r] = [3] * game.GRID_SIZE
    if leave_open:
        r, c = leave_open
        game.grid[r][c] = 0


def test_no_false_skip_player_with_valid_moves():
    """A player with valid pieces on an empty board is not auto-skipped."""
    game = Game()
    player = game.players[0]
    assert game.player_has_valid_moves(player) is True
    game.start_turn()
    assert not player.has_given_up


def test_blocked_auto_skip():
    """A player whose pieces cannot be placed is given up and opponent's turn starts."""
    game = Game()
    _fill_grid(game)
    player_a = game.players[0]
    player_b = game.players[1]
    # Give player_b a cell to play in, but player_a nothing fits
    game.grid[8][0] = 0
    player_b.pieces = [Piece("u", player_b)]

    game.current_player = player_a
    game.start_turn()

    assert player_a.has_given_up
    assert game.current_player == player_b


def test_blocked_one_extra_turn_score_win():
    """After A is blocked, B gets one extra DROP then GAMEOVER by score."""
    game = Game()
    _fill_grid(game)
    player_a = game.players[0]
    player_b = game.players[1]

    # One open cell at the bottom-left; only a 'u' piece fits
    game.grid[8][0] = 0
    player_b.pieces = [Piece("u", player_b)]

    game.current_player = player_a
    game.start_turn()

    # A should be skipped, one_extra_turn_remaining set, B is current
    assert player_a.has_given_up
    assert game.one_extra_turn_remaining
    assert game.current_player == player_b

    # B drops their bonus piece
    piece_b = player_b.pieces[0]
    piece_b.x = 0
    game.play_drop_piece(piece_b, player_b)

    assert game.status == Game.GAMEOVER
    assert game.win_type == "score"


def test_both_blocked_immediate_gameover():
    """When no player can place anything, game ends immediately as score win."""
    game = Game()
    _fill_grid(game)
    # All cells occupied — no piece fits for anyone
    game.current_player = game.players[0]
    game.start_turn()

    assert game.status == Game.GAMEOVER
    assert game.win_type == "score"


def test_exhausted_one_extra_turn_score_win():
    """When A places their last piece, B gets exactly one bonus DROP, then score win."""
    game = Game()
    player_a = game.players[0]
    player_b = game.players[1]

    # Give A exactly one piece and room to place it at bottom row
    u_piece = Piece("u", player_a)
    player_a.pieces = [u_piece]
    player_a.piece_index = 0

    u_piece.x = 0
    game.play_drop_piece(u_piece, player_a)

    # A exhausted → one_extra_turn_remaining should be set (no path win)
    assert game.one_extra_turn_remaining
    assert player_a.has_given_up

    # B now makes their bonus drop (use 'u' piece — always placeable on an empty grid)
    game.current_player = player_b
    u_b = Piece("u", player_b)
    player_b.pieces.insert(0, u_b)
    u_b.x = 1
    game.play_drop_piece(u_b, player_b)

    assert game.status == Game.GAMEOVER
    assert game.win_type == "score"


def test_bonus_turn_path_win():
    """If B wins by path during their bonus turn, win_type is 'path' not 'score'."""
    game = Game()
    player_a = game.players[0]
    player_b = game.players[1]

    # Pre-fill player_b cells to be one piece away from a horizontal path win:
    # fill columns 1-8 of row 8 with player_b's value
    for col in range(1, 9):
        game.grid[8][col] = player_b.value

    # Give A exactly one piece
    u_a = Piece("u", player_a)
    player_a.pieces = [u_a]
    player_a.piece_index = 0
    u_a.x = 5
    game.play_drop_piece(u_a, player_a)

    assert game.one_extra_turn_remaining

    # Give B a 'u' piece that will complete the path at column 0
    u_b = Piece("u", player_b)
    player_b.pieces = [u_b] + player_b.pieces
    player_b.piece_index = 0
    u_b.x = 0
    game.current_player = player_b
    game.play_drop_piece(u_b, player_b)

    assert game.status == Game.GAMEOVER
    assert game.win_type == "path"
    assert game.winner == player_b
