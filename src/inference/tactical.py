"""1-ply tactical safety net for the AI opponent.

Before trusting the policy's next action, enumerate every legal final placement
for the current player. If any wins immediately, play it. Otherwise, if the
opponent has a placement that wins next turn, play a placement that removes
every such threat.
"""
from typing import NamedTuple

try:
    from src.game.game import Actions
    from src.game.piece import rotate_shape, flip_shape
except ImportError:
    from game.game import Actions
    from game.piece import rotate_shape, flip_shape


class Placement(NamedTuple):
    piece_obj: object  # Piece reference (identity used for cycling)
    shape: tuple       # canonical tuple-of-tuples
    x: int
    ghost_y: int


def _shape_key(shape) -> tuple:
    return tuple(tuple(r) for r in shape)


def enumerate_placements(game, player) -> list[Placement]:
    """All legal final placements for `player` on the current board."""
    placements: list[Placement] = []
    G = game.GRID_SIZE
    for piece in player.pieces:
        orig_shape, orig_x = piece.shape, piece.x
        seen: set[tuple] = set()
        shape = orig_shape
        for _ in range(4):
            for candidate in (shape, flip_shape(shape)):
                key = _shape_key(candidate)
                if key in seen:
                    continue
                seen.add(key)
                piece.shape = candidate
                max_x = G - len(candidate[0])
                for x in range(max_x + 1):
                    piece.x = x
                    ghost_y = game.calculate_ghost_position(piece)
                    if ghost_y is not None:
                        placements.append(Placement(piece, key, x, ghost_y))
            shape = rotate_shape(shape)
        piece.shape, piece.x = orig_shape, orig_x
    return placements


def _paint(grid, placement: Placement, value: int) -> list[tuple[int, int]]:
    """Write `value` into grid cells occupied by placement; return cell list for undo."""
    cells: list[tuple[int, int]] = []
    for r, row in enumerate(placement.shape):
        for c, v in enumerate(row):
            if v == 1:
                y = placement.ghost_y + r
                x = placement.x + c
                cells.append((y, x))
                grid[y][x] = value
    return cells


def _unpaint(grid, cells: list[tuple[int, int]]) -> None:
    for y, x in cells:
        grid[y][x] = 0


def find_winning_placement(game, player, placements: list[Placement]) -> Placement | None:
    """First placement in `placements` that makes `player` win immediately."""
    for p in placements:
        cells = _paint(game.grid, p, player.value)
        won = player.check_if_winner(game.grid)
        _unpaint(game.grid, cells)
        if won:
            return p
    return None


def find_threats(game, opponent) -> list[Placement]:
    """Opponent placements that win immediately on the current board."""
    threats: list[Placement] = []
    for p in enumerate_placements(game, opponent):
        cells = _paint(game.grid, p, opponent.value)
        won = opponent.check_if_winner(game.grid)
        _unpaint(game.grid, cells)
        if won:
            threats.append(p)
    return threats


def _threat_still_wins(game, threat: Placement, opponent) -> bool:
    """After my move, is `threat` still a legal winning placement for opponent?

    Support may have changed (ghost_y shifts), or cells may be occupied.
    Recompute ghost_y; if none, threat is dead. Otherwise paint and check win.
    """
    # Temporarily install threat.shape on opponent's piece so calculate_ghost_position works.
    piece = threat.piece_obj
    saved_shape, saved_x = piece.shape, piece.x
    piece.shape = [list(r) for r in threat.shape]
    piece.x = threat.x
    new_ghost = game.calculate_ghost_position(piece)
    piece.shape, piece.x = saved_shape, saved_x
    if new_ghost is None:
        return False
    # Paint at the new ghost position and check win.
    shifted = Placement(piece, threat.shape, threat.x, new_ghost)
    cells = _paint(game.grid, shifted, opponent.value)
    won = opponent.check_if_winner(game.grid)
    _unpaint(game.grid, cells)
    return won


def find_blocking_placement(game, my_placements: list[Placement],
                            threats: list[Placement],
                            me, opponent) -> Placement | None:
    """First placement that neutralizes every pre-existing opponent threat."""
    for mine in my_placements:
        my_cells = _paint(game.grid, mine, me.value)
        blocks_all = all(not _threat_still_wins(game, t, opponent) for t in threats)
        _unpaint(game.grid, my_cells)
        if blocks_all:
            return mine
    return None


def find_tactical_move(game, player_idx: int = 0) -> Placement | None:
    """Return a winning placement, else a blocking placement, else None."""
    me = game.players[player_idx]
    opponent = game.players[1 - player_idx]
    my_placements = enumerate_placements(game, me)
    if not my_placements:
        return None

    win = find_winning_placement(game, me, my_placements)
    if win is not None:
        return win

    threats = find_threats(game, opponent)
    if not threats:
        return None

    return find_blocking_placement(game, my_placements, threats, me, opponent)


def execute_placement(game, placement: Placement) -> bool:
    """Drive Game.execute_action to apply `placement` for the current player.

    Cycles to the target piece, installs the target shape and column, then drops.
    Returns True on successful drop.
    """
    player = game.current_player
    target = placement.piece_obj

    # Cycle until current piece is the target. Bounded by pieces count.
    for _ in range(len(player.pieces) + 1):
        if player.pieces[player.piece_index] is target:
            break
        game.execute_action(Actions.ACTION_CYCLE_PIECE)
    else:
        return False

    piece = game.current_piece
    piece.shape = [list(r) for r in placement.shape]
    piece.x = placement.x
    return game.execute_action(Actions.ACTION_DROP)
