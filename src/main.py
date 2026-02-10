import pygame
import sys
from game import Game
from piece import Piece

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 10
BOARD_TOP_MARGIN = 80  # Space above the board for scores/info
BOARD_MARGIN = 20  # Margin all around the board
# Updated Constants for square cells
BOARD_SIZE = 360  # 360 is divisible by 9 (40px per cell)
BOARD_WIDTH = BOARD_SIZE
BOARD_HEIGHT = BOARD_SIZE


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
INFO_BG = (30, 30, 30)
BOARD_BG = (40, 40, 40)


def draw_grid(screen):
    # Center the board
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT,
    )
    pygame.draw.rect(screen, BOARD_BG, board_rect)
    # Draw 9 columns (visible lines)
    col_width = BOARD_WIDTH // 9
    for col in range(1, 9):
        x = board_rect.left + col * col_width
        pygame.draw.line(screen, WHITE, (x, board_rect.top), (x, board_rect.bottom), 1)
    # 9 rows (not visible, so nothing drawn)


def draw_board(screen, game, board_rect):
    # Calculer la taille précise d'une cellule pour qu'elle tienne dans le rectangle
    cell_w = BOARD_WIDTH // 9
    cell_h = BOARD_HEIGHT // 9

    for row_idx, row in enumerate(game.grid):
        for col_idx, cell_value in enumerate(row):
            if cell_value != 0:
                # On utilise cell_w et cell_h pour remplir exactement la case de la grille
                rect = pygame.Rect(
                    board_rect.left + col_idx * cell_w,
                    board_rect.top + row_idx * cell_h,
                    cell_w,
                    cell_h,
                )
                color = game.players[cell_value - 1].color
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLUE, rect, 1)


def draw_shape(screen, piece: Piece, x, y, block_size, color=WHITE, scale=1.0):
    scaled_block = int(block_size * scale)
    for row_idx, row in enumerate(piece.shape):
        for col_idx, cell in enumerate(row):
            if cell:
                rect = pygame.Rect(
                    x + col_idx * scaled_block,
                    y + row_idx * scaled_block,
                    scaled_block,
                    scaled_block,
                )
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLUE, rect, 1)  # outline


def draw_player_pieces(
    screen, player, hover_idx=None, hover_x=None, rects=None, scale=0.5
):
    # Display player's pieces as Tetris shapes below the grid, centered horizontally, at scale 0.5
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT,
    )
    block_size = BOARD_WIDTH // 9
    scaled_block = int(block_size * scale)
    # Calculate total width of all pieces for centering
    piece_widths = []
    for piece in player.pieces:
        shape = piece.shape
        width = len(shape[0]) * scaled_block
        piece_widths.append(width)
    total_width = sum(piece_widths) + (len(piece_widths) - 1) * 20
    x = board_rect.left + (BOARD_WIDTH - total_width) // 2
    y = board_rect.bottom + 20  # 20px below the board
    if rects is not None:
        rects.clear()
    for idx, piece in enumerate(player.pieces):
        # TODO draw 2 lines of 7 pieces max
        if hover_idx == idx and hover_x is not None:
            # Don't draw the selected piece in the bottom row
            if rects is not None:
                rects.append(
                    pygame.Rect(
                        hover_x, y, piece_widths[idx], piece.height() * scaled_block
                    )
                )
        else:
            draw_shape(screen, piece, x, y, block_size, color=player.color, scale=scale)
            if rects is not None:
                rects.append(
                    pygame.Rect(x, y, piece_widths[idx], piece.height() * scaled_block)
                )
        x += piece_widths[idx] + 20  # Spacing between pieces


def draw_selected_piece(screen, piece, x, y, block_size, color):
    draw_shape(screen, piece, x, y, block_size, color, scale=1.0)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PyLinkx Pygame Project")
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    font = pygame.font.SysFont(None, 36)
    turn = 0  # 0 for player 1, 1 for player 2
    piece_x = 0
    next_piece = game.players[turn].next_piece()

    block_size = BOARD_WIDTH // 9
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT,
    )
    piece_x = board_rect.left  # Start at leftmost position
    piece_y = None  # Not dropped yet

    ghost_grid_y = -1

    while game.running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
            elif event.type == pygame.KEYDOWN:
                # 1. Landing is already calculated (Ghost Y)
                grid_x = (piece_x - board_rect.left) // block_size

                # 2. Draw piece at valid landing spot (Same as ghost)
                ghost_y_pixel = board_rect.top + (ghost_grid_y * block_size)
                draw_shape(screen, next_piece, piece_x, ghost_y_pixel, block_size)

                if event.key == pygame.K_ESCAPE:
                    game.running = False
                elif event.key == pygame.K_TAB:
                    # Cycle through available pieces
                    next_piece = game.players[turn].next_piece()
                    piece_x = board_rect.left
                    piece_y = None
                elif event.key == pygame.K_RETURN:
                    # Flip the piece horizontally
                    next_piece.flip()  # type: ignore
                elif event.key == pygame.K_LEFT:
                    # Move piece left if possible
                    min_x = board_rect.left
                    if piece_x - block_size >= min_x:
                        piece_x -= block_size
                elif event.key == pygame.K_RIGHT:
                    # Move piece right if possible
                    max_x = board_rect.right - next_piece.width() * block_size
                    if piece_x + block_size <= max_x:
                        piece_x += block_size
                elif event.key == pygame.K_UP:
                    # Rotate piece
                    next_piece.rotate()  # type: ignore
                    # Adjust x if needed to keep within board
                    new_width = next_piece.width() * block_size  # type: ignore
                    max_x = board_rect.right - new_width
                    if piece_x > max_x:
                        piece_x = max_x
                elif event.key == pygame.K_DOWN:
                    # Drop the piece with condition
                    if ghost_grid_y != -1:
                        # Place piece only if a valid landing spot was found
                        grid_x = (piece_x - board_rect.left) // block_size
                        game.place_piece_on_grid(next_piece, grid_x, ghost_grid_y, turn)
                        game.players[turn].drop_piece(next_piece)
                        game.update()  # Update game logic
                        
                        # Switch turn
                        turn = (turn + 1) % len(game.players)

                        # Reset selection for next player
                        next_piece = game.players[turn].next_piece()
                        piece_x = board_rect.left

        screen.fill(BLACK)

        # 1. UI: Scores and Player Pieces
        for i, player in enumerate(game.players):
            text = font.render(f"{player.name}: {player.score}", True, WHITE)
            screen.blit(text, (30, 10 + i * 36))
        draw_player_pieces(screen, game.players[turn], font, scale=0.5)

        # 2. GRID: Draw the static board and placed pieces
        draw_grid(screen)
        draw_board(screen, game, board_rect)

        # 3. GHOST PIECE CALCULATION
        grid_x = (piece_x - board_rect.left) // block_size
        ghost_grid_y = -1

        if game.is_valid_move(next_piece, grid_x, 0):
            for y_test in range(9 - next_piece.height() + 1):
                if game.is_valid_move(next_piece, grid_x, y_test):
                    ghost_grid_y = y_test
                else:
                    break
        if ghost_grid_y != -1 and not game.is_fully_supported(
            next_piece, grid_x, ghost_grid_y
        ):
            ghost_grid_y = -1

        # 4. RENDER GHOST
        if ghost_grid_y != -1:
            ghost_y_pixel = board_rect.top + (ghost_grid_y * (BOARD_HEIGHT // 9))
            draw_shape(
                screen,
                next_piece,
                piece_x,
                ghost_y_pixel,
                block_size,
                color=(50, 50, 50),
            )

        # 5. ACTIVE PIECE: Draw the piece controlled by the player
        piece_height = next_piece.height() * block_size
        y = piece_y if piece_y is not None else (board_rect.top - piece_height - 10)
        draw_selected_piece(
            screen,
            type("Tmp", (), {"shape": next_piece.shape})(),
            piece_x,
            y,
            block_size,
            game.players[turn].color,
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
