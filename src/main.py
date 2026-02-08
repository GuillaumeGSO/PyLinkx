import pygame
import sys
from src.game import Game

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30
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


def draw_shape(screen, shape, x, y, block_size, color=WHITE, scale=1.0):
    scaled_block = int(block_size * scale)
    for row_idx, row in enumerate(shape):
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
                        hover_x, y, piece_widths[idx], len(piece.shape) * scaled_block
                    )
                )
        else:
            draw_shape(
                screen, piece.shape, x, y, block_size, color=player.color, scale=scale
            )
            if rects is not None:
                rects.append(
                    pygame.Rect(
                        x, y, piece_widths[idx], len(piece.shape) * scaled_block
                    )
                )
        x += piece_widths[idx] + 20  # Spacing between pieces


def draw_selected_piece(screen, piece, x, y, block_size, color):
    draw_shape(screen, piece.shape, x, y, block_size, color, scale=1.0)


def rotate_shape(shape):
    # Rotate a 2D list (matrix) clockwise
    return [list(row) for row in zip(*shape[::-1])]


def flip_shape(shape):
    # Flip horizontally
    return [list(row[::-1]) for row in shape]


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PyLinkx Pygame Project")
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    font = pygame.font.SysFont(None, 36)
    turn = 1  # 0 for player 1, 1 for player 2
    game_over = False
    winner_text = ""
    selected_piece_idx = 0
    piece_x = 0  # grid-aligned x for the selected piece
    drag_shape = [row[:] for row in game.players[turn].pieces[selected_piece_idx].shape]
    block_size = BOARD_WIDTH // 9
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT,
    )
    piece_x = board_rect.left  # Start at leftmost position
    piece_y = None  # Not dropped yet

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:

                # 1. Calculer la position d'atterrissage (Ghost Y)
                grid_x = (piece_x - board_rect.left) // block_size
                ghost_grid_y = 0
                for y_test in range(10):  # Parcourt la hauteur de la grille
                    if game.is_valid_move(drag_shape, grid_x, y_test):
                        ghost_grid_y = y_test
                    else:
                        break

                # 2. Dessiner l'aperçu à cette position (en gris ou transparent)
                ghost_y_pixel = board_rect.top + (ghost_grid_y * block_size)
                draw_shape(screen, drag_shape, piece_x, ghost_y_pixel, block_size)

                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_TAB:
                    # Cycle through available pieces
                    selected_piece_idx = (selected_piece_idx + 1) % len(
                        game.players[turn].pieces
                    )
                    drag_shape = [
                        row[:]
                        for row in game.players[turn].pieces[selected_piece_idx].shape
                    ]
                    piece_x = board_rect.left
                    piece_y = None
                elif event.key == pygame.K_RETURN:
                    # Flip the piece horizontally
                    drag_shape = flip_shape(drag_shape)
                elif event.key == pygame.K_LEFT:
                    # Move piece left
                    min_x = board_rect.left
                    if piece_x - block_size >= min_x:
                        piece_x -= block_size
                elif event.key == pygame.K_RIGHT:
                    # Move piece right
                    max_x = board_rect.right - len(drag_shape[0]) * block_size
                    if piece_x + block_size <= max_x:
                        piece_x += block_size
                elif event.key == pygame.K_UP:
                    # Rotate piece
                    new_shape = rotate_shape(drag_shape)
                    # Adjust x if needed to keep within board
                    new_width = len(new_shape[0]) * block_size
                    max_x = board_rect.right - new_width
                    if piece_x > max_x:
                        piece_x = max_x
                    drag_shape = new_shape
                elif event.key == pygame.K_DOWN:
                    grid_x = (piece_x - board_rect.left) // block_size

                    # 1. Use the ghost logic to find the landing row
                    ghost_grid_y = -1
                    if game.is_valid_move(drag_shape, grid_x, 0):
                        for y_test in range(9 - len(drag_shape) + 1):
                            if game.is_valid_move(drag_shape, grid_x, y_test):
                                ghost_grid_y = y_test
                            else:
                                break

                    # 2. Place piece only if a valid landing spot was found
                    if ghost_grid_y != -1:
                        game.place_piece(drag_shape, grid_x, ghost_grid_y, turn)
                        game.players[turn].pieces.pop(selected_piece_idx)

                        # 3. Switch turn and reset selection
                        players_with_pieces = [
                            i for i, p in enumerate(game.players) if len(p.pieces) > 0
                        ]
                        if not players_with_pieces:
                            game_over = True
                        else:
                            turn = (turn + 1) % len(game.players)
                            while len(game.players[turn].pieces) == 0:
                                turn = (turn + 1) % len(game.players)

                            # Reset selection for next player
                            selected_piece_idx = 0
                            drag_shape = [
                                row[:]
                                for row in game.players[turn]
                                .pieces[selected_piece_idx]
                                .shape
                            ]
                            piece_x = board_rect.left

                    else:
                        print("Invalid Move! You can't place that there.")
                print(game)
        game.update()  # Update game logic

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

        if game.is_valid_move(drag_shape, grid_x, 0):
            for y_test in range(9 - len(drag_shape) + 1):
                if game.is_valid_move(drag_shape, grid_x, y_test):
                    ghost_grid_y = y_test
                else:
                    break

        # 4. RENDER GHOST
        if ghost_grid_y != -1:
            ghost_y_pixel = board_rect.top + (ghost_grid_y * (BOARD_HEIGHT // 9))
            draw_shape(
                screen,
                drag_shape,
                piece_x,
                ghost_y_pixel,
                block_size,
                color=(50, 50, 50),
            )

        # 5. ACTIVE PIECE: Draw the piece controlled by the player
        piece_height = len(drag_shape) * block_size
        y = piece_y if piece_y is not None else (board_rect.top - piece_height - 10)
        draw_selected_piece(
            screen,
            type("Tmp", (), {"shape": drag_shape})(),
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
