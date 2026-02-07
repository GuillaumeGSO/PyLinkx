import pygame
import sys
from game import Game

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
BOARD_TOP_MARGIN = 80  # Space above the board for scores/info
BOARD_MARGIN = 20      # Margin all around the board
BOARD_WIDTH = 600
BOARD_HEIGHT = 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
INFO_BG = (30, 30, 30)
BOARD_BG = (40, 40, 40)


def draw_grid(screen):
    # Center the board
    board_rect = pygame.Rect(
        (SCREEN_WIDTH - BOARD_WIDTH) // 2,
        BOARD_TOP_MARGIN + (SCREEN_HEIGHT - BOARD_TOP_MARGIN - BOARD_HEIGHT) // 2,
        BOARD_WIDTH,
        BOARD_HEIGHT
    )
    pygame.draw.rect(screen, BOARD_BG, board_rect)
    # Draw 9 columns (visible lines)
    col_width = BOARD_WIDTH // 9
    for col in range(1, 9):
        x = board_rect.left + col * col_width
        pygame.draw.line(screen, WHITE, (x, board_rect.top), (x, board_rect.bottom), 2)
    # 9 rows (not visible, so nothing drawn)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('PyLinkx Pygame Project')
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    font = pygame.font.SysFont(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        game.update()  # Update game logic

        screen.fill(BLACK)
        # Draw area for scores/game info
        pygame.draw.rect(screen, BLACK, (0, 0, SCREEN_WIDTH, BOARD_TOP_MARGIN))
        # Display player names and scores
        for i, player in enumerate(game.players):
            text = font.render(f"{player.name} - Score: {player.score}", True, WHITE)
            screen.blit(text, (30, 10 + i * 36))
        draw_grid(screen)
        # Draw your game objects here, possibly using game state

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()
