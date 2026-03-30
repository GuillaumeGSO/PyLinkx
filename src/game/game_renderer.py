from pathlib import Path

import pygame

from .game import Game
from .piece import Piece

_FONT = str(Path(__file__).parent.parent / "assets" / "fonts" / "PressStart2P-Regular.ttf")


def _draw_gradient(screen, color_top, color_bottom, rect):
    """Draw a vertical gradient over rect."""
    r1, g1, b1 = color_top
    r2, g2, b2 = color_bottom
    h = rect.height
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(r1 + (r2 - r1) * t)
        g = int(g1 + (g2 - g1) * t)
        b = int(b1 + (b2 - b1) * t)
        pygame.draw.rect(screen, (r, g, b), (rect.x, rect.y + y, rect.width, 1))


class GameRenderer:

    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 600

    BOARD_LEFT = 320
    BOARD_TOP = 240
    BOARD_SIZE = 360  # 360 / 9 = 40px per cell
    BOARD_WIDTH = BOARD_SIZE
    BOARD_HEIGHT = BOARD_SIZE
    HEADER_H = 60
    PANEL_WIDTH = 320

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GOLD = (255, 215, 0)
    BOARD_BG = (20, 10, 35)
    GRID_LINE = (60, 40, 80)
    BLOCK_OUTLINE = (100, 80, 140)
    GHOST_COLOR = (0, 50, 255)
    PANEL_BG = (18, 8, 30)
    PANEL_BORDER = (60, 40, 90)
    ACTIVE_BORDER = (0, 50, 255)
    HIGHLIGHT_FILL = (255, 255, 200)
    DIM = (120, 100, 150)

    """Pure display: knows about fonts, colors, and the screen."""

    def __init__(self, screen, game: Game):
        self.screen = screen
        self.game = game
        self.font_title = pygame.font.Font(_FONT, 26)
        self.font_large = pygame.font.Font(_FONT, 15)
        self.font_medium = pygame.font.Font(_FONT, 13)
        self.font_small = pygame.font.Font(_FONT, 10)
        self.board_rect = pygame.Rect(
            self.BOARD_LEFT, self.BOARD_TOP, self.BOARD_WIDTH, self.BOARD_HEIGHT
        )
        self.block_size = self.BOARD_WIDTH // Game.GRID_SIZE

    # ------------------------------------------------------------------
    # Main draw entry point
    # ------------------------------------------------------------------

    def draw(self):
        self._draw_gradient_background()
        self._draw_header()
        self.draw_board()
        self.draw_grid()

        if self.game.status == Game.PLAYING:
            self.draw_selected_piece(
                self.game.current_piece, self.game.current_piece.color
            )
            self.draw_ghost_piece(self.game.current_piece)
            self._draw_side_panels()
            if self.game.one_extra_turn_remaining:
                self._draw_last_turn_notice()

        elif self.game.status == Game.GAMEOVER:
            self._draw_side_panels()
            self._draw_gameover_overlay()

    # ------------------------------------------------------------------
    # Background & header
    # ------------------------------------------------------------------

    def _draw_gradient_background(self):
        _draw_gradient(
            self.screen,
            (15, 0, 30),
            (0, 0, 0),
            pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT),
        )

    def _draw_header(self):
        header_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.HEADER_H)
        pygame.draw.rect(self.screen, (10, 0, 20), header_rect)
        pygame.draw.line(
            self.screen,
            self.PANEL_BORDER,
            (0, self.HEADER_H - 1),
            (self.SCREEN_WIDTH, self.HEADER_H - 1),
            2,
        )
        title = self.font_title.render("LINKX", True, self.GOLD)
        self.screen.blit(title, (self.SCREEN_WIDTH // 2 - title.get_width() // 2, 6))

    # ------------------------------------------------------------------
    # Board
    # ------------------------------------------------------------------

    def draw_board(self):
        pygame.draw.rect(self.screen, self.BOARD_BG, self.board_rect)
        cell = self.BOARD_WIDTH // Game.GRID_SIZE
        for i in range(1, Game.GRID_SIZE):
            x = self.board_rect.left + i * cell
            pygame.draw.line(
                self.screen,
                self.GRID_LINE,
                (x, self.board_rect.top),
                (x, self.board_rect.bottom),
                1,
            )
        pygame.draw.line(self.screen, self.PANEL_BORDER,
                         (self.board_rect.left, self.board_rect.top),
                         (self.board_rect.right, self.board_rect.top), 2)
        pygame.draw.line(self.screen, self.PANEL_BORDER,
                         (self.board_rect.left, self.board_rect.bottom - 1),
                         (self.board_rect.right, self.board_rect.bottom - 1), 2)

    def draw_grid(self):
        cell_w = self.BOARD_WIDTH // Game.GRID_SIZE
        cell_h = self.BOARD_HEIGHT // Game.GRID_SIZE
        grid = self.game.grid
        pid = self.game.piece_id_grid
        rows = len(grid)
        cols = len(grid[0])
        for row_idx in range(rows):
            for col_idx in range(cols):
                cell_value = grid[row_idx][col_idx]
                if cell_value == 0:
                    continue
                color = self.game.players[cell_value - 1].color
                cx = self.board_rect.left + col_idx * cell_w
                cy = self.board_rect.top + row_idx * cell_h
                piece_id = pid[row_idx][col_idx]
                pygame.draw.rect(self.screen, color, pygame.Rect(cx, cy, cell_w, cell_h))
                if row_idx == 0 or pid[row_idx - 1][col_idx] != piece_id:
                    pygame.draw.line(self.screen, self.BLOCK_OUTLINE, (cx, cy), (cx + cell_w, cy), 1)
                if row_idx == rows - 1 or pid[row_idx + 1][col_idx] != piece_id:
                    pygame.draw.line(self.screen, self.BLOCK_OUTLINE, (cx, cy + cell_h), (cx + cell_w, cy + cell_h), 1)
                if col_idx == 0 or pid[row_idx][col_idx - 1] != piece_id:
                    pygame.draw.line(self.screen, self.BLOCK_OUTLINE, (cx, cy), (cx, cy + cell_h), 1)
                if col_idx == cols - 1 or pid[row_idx][col_idx + 1] != piece_id:
                    pygame.draw.line(self.screen, self.BLOCK_OUTLINE, (cx + cell_w, cy), (cx + cell_w, cy + cell_h), 1)

    def draw_shape(self, piece: Piece, x_pixel, y_pixel, color=WHITE, scale=1.0):
        scaled_block = int(self.block_size * scale)
        for row_idx, row in enumerate(piece.shape):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(
                        x_pixel + col_idx * scaled_block,
                        y_pixel + row_idx * scaled_block,
                        scaled_block,
                        scaled_block,
                    )
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, self.BLOCK_OUTLINE, rect, 1)

    def draw_ghost_piece(self, piece: Piece):
        if self.game.ghost_grid_y is None:
            return
        x_pixel = piece.x * self.block_size + self.board_rect.left
        y_pixel = self.board_rect.top + self.game.ghost_grid_y * self.block_size
        shape = piece.shape
        rows = len(shape)
        cols = len(shape[0])
        bs = self.block_size
        for row_idx in range(rows):
            for col_idx in range(cols):
                if not shape[row_idx][col_idx]:
                    continue
                cx = x_pixel + col_idx * bs
                cy = y_pixel + row_idx * bs
                if row_idx == 0 or not shape[row_idx - 1][col_idx]:
                    pygame.draw.line(self.screen, self.GHOST_COLOR, (cx, cy), (cx + bs, cy), 2)
                if row_idx == rows - 1 or not shape[row_idx + 1][col_idx]:
                    by = min(cy + bs, self.board_rect.bottom - 1)
                    pygame.draw.line(self.screen, self.GHOST_COLOR, (cx, by), (cx + bs, by), 2)
                if col_idx == 0 or not shape[row_idx][col_idx - 1]:
                    pygame.draw.line(self.screen, self.GHOST_COLOR, (cx, cy), (cx, cy + bs), 2)
                if col_idx == cols - 1 or not shape[row_idx][col_idx + 1]:
                    pygame.draw.line(self.screen, self.GHOST_COLOR, (cx + bs, cy), (cx + bs, cy + bs), 2)

    def draw_selected_piece(self, piece, color):
        """Draw the current piece above the board with neon glow."""
        x_pixel = piece.x * self.block_size + self.board_rect.left
        y_pixel = self.board_rect.top - self.block_size * piece.height()
        y_pixel = max(self.HEADER_H + 4, y_pixel)
        self.draw_shape(piece, x_pixel, y_pixel - 7, color=color, scale=1.0)

    # ------------------------------------------------------------------
    # Side panels
    # ------------------------------------------------------------------

    def _draw_side_panels(self):
        p1, p2 = self.game.players[0], self.game.players[1]
        is_p1_active = (
            self.game.status == Game.PLAYING and self.game.current_player == p1
        )
        is_p2_active = (
            self.game.status == Game.PLAYING and self.game.current_player == p2
        )
        left_rect = pygame.Rect(
            0, self.HEADER_H, self.PANEL_WIDTH, self.SCREEN_HEIGHT - self.HEADER_H
        )
        right_rect = pygame.Rect(
            self.BOARD_LEFT + self.BOARD_WIDTH,
            self.HEADER_H,
            self.PANEL_WIDTH,
            self.SCREEN_HEIGHT - self.HEADER_H,
        )
        self._draw_panel(p1, left_rect, is_p1_active)
        self._draw_panel(p2, right_rect, is_p2_active)

    def _draw_panel(self, player, panel_rect, is_active):
        pygame.draw.rect(self.screen, self.PANEL_BG, panel_rect)
        border_color = self.ACTIVE_BORDER if is_active else self.PANEL_BORDER
        pygame.draw.rect(self.screen, border_color, panel_rect, 2)

        margin = 16
        x = panel_rect.left + margin
        y = panel_rect.top + margin

        name_surf = self.font_medium.render(player.name.upper(), True, player.color)
        self.screen.blit(name_surf, (x, y))
        y += name_surf.get_height() + 4

        score_surf = self.font_large.render(f"SCORE: {player.score}", True, self.WHITE)
        self.screen.blit(score_surf, (x, y))
        y += score_surf.get_height() + 6

        if is_active:
            turn_surf = self.font_small.render("> YOUR TURN", True, self.GHOST_COLOR)
            self.screen.blit(turn_surf, (x, y))
        y += self.font_small.get_height() + 10

        pygame.draw.line(
            self.screen,
            self.PANEL_BORDER,
            (panel_rect.left + 8, y),
            (panel_rect.right - 8, y),
            1,
        )
        y += 8

        current_piece = self.game.current_piece if is_active else None
        self._draw_piece_miniatures(player, current_piece, panel_rect, x, y)

    def _draw_piece_miniatures(
        self, player, current_piece, panel_rect, start_x, start_y
    ):
        scale = 0.38
        cell = int(self.block_size * scale)  # ~13px per cell
        col_width = (panel_rect.width - 16) // 2
        gap = 6
        max_piece_h = 4 * cell

        pieces = [p for p in player.pieces]
        mid = (len(pieces) + 1) // 2

        for col_idx, column_pieces in enumerate([pieces[:mid], pieces[mid:]]):
            x = start_x + col_idx * col_width
            y = start_y
            for piece in column_pieces:
                self.draw_shape(piece, x, y, color=player.color, scale=scale)
                y += max_piece_h + gap

                if y > panel_rect.bottom - max_piece_h:
                    break

    # ------------------------------------------------------------------
    # Notices
    # ------------------------------------------------------------------

    def _draw_last_turn_notice(self):
        below_y = self.board_rect.bottom + 12
        surf = self.font_medium.render("LAST TURN!", True, (255, 140, 0))
        x = self.BOARD_LEFT + (self.BOARD_WIDTH - surf.get_width()) // 2
        self.screen.blit(surf, (x, below_y))

    # ------------------------------------------------------------------
    # GAMEOVER overlay
    # ------------------------------------------------------------------

    def _draw_gameover_overlay(self):
        cells = set()
        if self.game.winner:
            if self.game.win_type == "path":
                cells = self.game.winner.get_winning_path_cells(self.game.grid)
            elif self.game.win_type == "score":
                cells = self.game.winner.get_largest_zone_cells(self.game.grid)
        if cells:
            self._draw_winning_highlight(cells)

        overlay = pygame.Surface(
            (self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 170))
        self.screen.blit(overlay, (0, 0))

        cx = self.SCREEN_WIDTH // 2
        cy = self.HEADER_H + 28

        if self.game.winner:
            color = self.game.winner.color
            win_label = "PATH WIN!" if self.game.win_type == "path" else "SCORE WIN!"
            msg1 = self.font_large.render(
                f"{self.game.winner.name.upper()} WINS!", True, color
            )
            msg2 = self.font_medium.render(win_label, True, self.GOLD)
        elif len(self.game.get_players_in_play()) == 0:
            msg1 = self.font_large.render("WINNER BY ZONE!", True, self.GOLD)
            msg2 = self.font_medium.render("", True, self.WHITE)
        else:
            msg1 = self.font_large.render("IT'S A TIE!", True, self.WHITE)
            msg2 = self.font_medium.render("", True, self.WHITE)

        self.screen.blit(msg1, (cx - msg1.get_width() // 2, cy))
        cy += msg1.get_height() + 6
        if msg2.get_width() > 0:
            self.screen.blit(msg2, (cx - msg2.get_width() // 2, cy))
            cy += msg2.get_height() + 18
        else:
            cy += 18

        hint = self.font_small.render(
            "R = Play Again       M / ESC = Menu", True, self.DIM
        )
        self.screen.blit(hint, (cx - hint.get_width() // 2, cy))

    def _draw_winning_highlight(self, cells):
        cell_w = self.BOARD_WIDTH // Game.GRID_SIZE
        cell_h = self.BOARD_HEIGHT // Game.GRID_SIZE
        pid = self.game.piece_id_grid
        rows = Game.GRID_SIZE
        cols = Game.GRID_SIZE

        # Semi-transparent bright overlay on each winning cell
        overlay = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
        overlay.fill((255, 255, 180, 90))
        for row_idx, col_idx in cells:
            cx = self.board_rect.left + col_idx * cell_w
            cy = self.board_rect.top + row_idx * cell_h
            self.screen.blit(overlay, (cx, cy))

        # Gold piece-outline border (3px) around winning cells
        for row_idx, col_idx in cells:
            cx = self.board_rect.left + col_idx * cell_w
            cy = self.board_rect.top + row_idx * cell_h
            piece_id = pid[row_idx][col_idx]
            if row_idx == 0 or pid[row_idx - 1][col_idx] != piece_id:
                pygame.draw.line(self.screen, self.GHOST_COLOR,(cx, cy), (cx + cell_w, cy), 3)
            if row_idx == rows - 1 or pid[row_idx + 1][col_idx] != piece_id:
                by = min(cy + cell_h, self.board_rect.bottom - 1)
                pygame.draw.line(self.screen, self.GHOST_COLOR,(cx, by), (cx + cell_w, by), 3)
            if col_idx == 0 or pid[row_idx][col_idx - 1] != piece_id:
                pygame.draw.line(self.screen, self.GHOST_COLOR,(cx, cy), (cx, cy + cell_h), 3)
            if col_idx == cols - 1 or pid[row_idx][col_idx + 1] != piece_id:
                pygame.draw.line(self.screen, self.GHOST_COLOR,(cx + cell_w, cy), (cx + cell_w, cy + cell_h), 3)

    # ------------------------------------------------------------------
    # Legacy / training helpers (kept for train.py compatibility)
    # ------------------------------------------------------------------

    def draw_scores(self):
        font = pygame.font.Font(None, 28)
        for i, player in enumerate(self.game.players):
            text = font.render(f"{player.name}: {player.score}", True, player.color)
            self.screen.blit(text, (30, 10 + i * 36))

    def draw_player_pieces(
        self, player, hover_idx=None, hover_x=None, rects=None, scale=0.5
    ):
        scaled_block = int(self.block_size * scale)
        piece_widths = [len(p.shape[0]) * scaled_block for p in player.pieces]
        total_width = sum(piece_widths) + (len(piece_widths) - 1) * 20
        x = self.board_rect.left + (self.BOARD_WIDTH - total_width) // 2
        y = self.board_rect.bottom + 20
        if rects is not None:
            rects.clear()
        for idx, piece in enumerate(player.pieces):
            if hover_idx == idx and hover_x is not None:
                if rects is not None:
                    rects.append(
                        pygame.Rect(
                            hover_x, y, piece_widths[idx], piece.height() * scaled_block
                        )
                    )
            else:
                self.draw_shape(piece, x, y, color=player.color, scale=scale)
                if rects is not None:
                    rects.append(
                        pygame.Rect(
                            x, y, piece_widths[idx], piece.height() * scaled_block
                        )
                    )
            x += piece_widths[idx] + 20
