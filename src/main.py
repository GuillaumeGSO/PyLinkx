import pygame
import sys
from game import Game
from piece import Piece
from game_renderer import GameRenderer

# Constants
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 10


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("PyLinkx Pygame Project")
    clock = pygame.time.Clock()

    game = Game()  # Initialize game logic
    renderer = GameRenderer(screen, game)
    game.set_current_piece(game.current_player.next_piece())
    running = True

    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game.status == game.PLAYING:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        # Cycle through available pieces
                        game.set_current_piece(game.current_player.next_piece())
                    elif event.key == pygame.K_LEFT:
                        # Move piece left and stop at board edge
                        game.move_piece_left(game.current_piece)
                    elif event.key == pygame.K_RIGHT:
                        # Move piece right and stop at board edge
                        game.move_piece_right(game.current_piece)
                    elif event.key == pygame.K_UP:
                        # Rotate the piece clockwise and adjust position if needed
                        game.rotate_piece(game.current_piece)
                    elif event.key == pygame.K_RETURN:
                        # Flip the piece horizontally
                        game.current_piece.flip()
                    elif event.key == pygame.K_DOWN:
                        # Drop piece if it is legal move, then switch turn
                        if game.play_drop_piece(
                            game.current_piece, game.current_player
                        ):
                            game.current_player = game.get_next_player()
                            game.set_current_piece(game.current_player.next_piece())
                    elif event.key == pygame.K_p:
                        # Press P to give up and switch turn
                        game.give_up_and_check(game.current_player)
                        game.current_player = game.get_next_player()
                        game.set_current_piece(game.current_player.next_piece())
                elif game.status == game.GAMEOVER:
                    # should render button to reset game
                    print("Game Over! Press R to Restart or ESC to Quit.")
                    if event.key == pygame.K_r:
                        game.reset()
                        game.set_current_piece(game.current_player.next_piece())
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                game.update()
        renderer.draw()
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
