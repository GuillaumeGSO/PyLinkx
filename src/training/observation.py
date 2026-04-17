"""Observation building and action masking for PyLinkx.
Extracted from game_env.py so it can be imported without gymnasium
(needed for WASM/browser inference).
"""
import numpy as np

try:
    from src.game.game import Game, Actions
    from src.game.piece import TETRIS_SHAPES
except ImportError:
    from game.game import Game, Actions
    from game.piece import TETRIS_SHAPES

PIECE_MAP = {"L": 0, "S": 1, "c": 2, "T": 3, "I": 4, "u": 5, "b": 6}

# Pre-computed canonical 4x4 padded shapes for each piece type, flattened (16 values each)
_CANONICAL_SHAPES: dict[str, np.ndarray] = {}
for _name in PIECE_MAP:
    _padded = np.zeros((4, 4), dtype=np.float32)
    _s = TETRIS_SHAPES[_name]
    _padded[:len(_s), :len(_s[0])] = _s
    _CANONICAL_SHAPES[_name] = _padded.flatten()


def _build_piece_inventory(player) -> np.ndarray:
    """Build 7x16=112 inventory for a player. Cell = canonical_shape x (count/2)."""
    counts: dict[str, int] = {}
    for p in player.pieces:
        counts[p.shape_name] = counts.get(p.shape_name, 0) + 1
    inv = np.zeros(7 * 16, dtype=np.float32)
    for name, idx in PIECE_MAP.items():
        count = counts.get(name, 0)
        if count > 0:
            inv[idx * 16:(idx + 1) * 16] = _CANONICAL_SHAPES[name] * (count / 2.0)
    return inv


def compute_action_mask(game) -> np.ndarray:
    """Returns a binary mask (1=valid, 0=invalid) for all 6 actions."""
    piece = game.current_piece if hasattr(game, "current_piece") else None
    return np.array([
        1,
        int(piece is not None and game.can_move_piece(piece, dx=-1)),
        int(piece is not None and game.can_move_piece(piece, dx=1)),
        int(piece is not None and game.can_rotate(piece)),
        int(piece is not None and game.can_flip(piece)),
        int(game.can_drop()),
    ], dtype=np.int8)


def compute_path_progress(game, player_idx: int) -> tuple[float, float]:
    """BFS path progress for a player. Returns (h_progress, v_progress)."""
    player_val = game.players[player_idx].value
    grid = game.grid
    G = game.GRID_SIZE
    g = G - 1

    h_from_left = -1
    start = [(r, 0) for r in range(G) if grid[r][0] == player_val]
    if start:
        visited = set(start)
        stack = list(start)
        while stack:
            r, c = stack.pop()
            if c > h_from_left:
                h_from_left = c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in visited and grid[nr][nc] == player_val:
                        visited.add((nr, nc))
                        stack.append((nr, nc))

    h_from_right = G
    start = [(r, g) for r in range(G) if grid[r][g] == player_val]
    if start:
        min_col = g
        visited = set(start)
        stack = list(start)
        while stack:
            r, c = stack.pop()
            if c < min_col:
                min_col = c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in visited and grid[nr][nc] == player_val:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
        h_from_right = min_col

    h_progress = max(
        h_from_left / g if h_from_left >= 0 else 0.0,
        (g - h_from_right) / g if h_from_right < G else 0.0,
    )

    v_progress = 0.0
    start = [(g, c) for c in range(G) if grid[g][c] == player_val]
    if start:
        min_row = g
        visited = set(start)
        stack = list(start)
        while stack:
            r, c = stack.pop()
            if r < min_row:
                min_row = r
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < G and 0 <= nc < G and (nr, nc) not in visited and grid[nr][nc] == player_val:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
        v_progress = (g - min_row) / g

    return (h_progress, v_progress)


def build_observation(game, max_steps_by_turn: int, steps_for_current_turn: int,
                      last_action_valid: bool, path_progress=None) -> dict:
    """
    Build the observation dict from game state.

    path_progress: [[h0, v0], [h1, v1]] cached BFS values per player.
    If None, BFS is computed fresh for both players.
    """
    if path_progress is None:
        path_progress = [
            list(compute_path_progress(game, 0)),
            list(compute_path_progress(game, 1)),
        ]

    grid_array = np.array(game.grid, dtype=np.float32) / 2.0
    grid_array = np.expand_dims(grid_array, axis=-1)

    current_piece = game.current_piece
    nb_players = len(game.players)
    max_pieces = 2 * len(PIECE_MAP)
    current_piece_id = float(PIECE_MAP[current_piece.shape_name]) / len(PIECE_MAP)
    remaining_ratio = float(len(game.current_player.pieces)) / max_pieces
    grid_cells = float(game.GRID_SIZE * game.GRID_SIZE)
    player_scores = [float(p.score) / grid_cells for p in game.players]

    other_scalars = np.array([
        float(game.current_player.value - 1) / (nb_players - 1),
        float(current_piece.x) / game.GRID_SIZE,
        player_scores[0],
        float(1.0 if game.ghost_grid_y else 0.0),
        current_piece_id,
        remaining_ratio,
        player_scores[1],
        float(game.status == Game.GAMEOVER),
        float(last_action_valid),
    ], dtype=np.float32)

    padded = np.zeros((4, 4), dtype=np.float32)
    shape = current_piece.shape
    rows, cols = len(shape), len(shape[0])
    padded[:rows, :cols] = np.array(shape)
    shape_vals = padded.flatten()

    remaining_actions_ratio = (max_steps_by_turn - steps_for_current_turn) / max_steps_by_turn

    path_scalars = []
    for i, player in enumerate(game.players):
        h, v = path_progress[i]
        path_scalars.extend([h, v, max(h, v), float(player.score) / grid_cells])

    p1_inventory = _build_piece_inventory(game.players[0])
    p2_inventory = _build_piece_inventory(game.players[1])

    scalars = np.concatenate([
        other_scalars,
        [remaining_actions_ratio],
        shape_vals,
        np.array(path_scalars, dtype=np.float32),
        p1_inventory,
        p2_inventory,
    ])

    return {"grid": grid_array, "scalars": scalars}
