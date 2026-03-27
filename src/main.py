import argparse
import asyncio
import sys
import pygame
from game import Game, Actions
from game_renderer import GameRenderer
from game_env import build_observation, compute_action_mask

# Constants
FPS = 10
MAX_STEPS_BY_TURN = 36


def load_ai_model(model_path: str):
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(model_path)


async def main(ai_model=None, ai_delay: int = 150):
    pygame.init()
    screen = pygame.display.set_mode(
        (GameRenderer.SCREEN_WIDTH, GameRenderer.SCREEN_HEIGHT)
    )
    pygame.display.set_caption("PyLinkx Pygame Project")
    clock = pygame.time.Clock()

    game = Game()
    renderer = GameRenderer(screen, game)
    game.start_turn()
    running = True
    gameover_since = None  # ms timestamp when GAMEOVER first detected

    while running:

        # AI turn: execute P1's full turn before processing events
        if ai_model and game.status == game.PLAYING and game.players.index(game.current_player) == 0:
            ai_steps = 0
            while game.status == game.PLAYING and game.players.index(game.current_player) == 0:
                obs = build_observation(game, MAX_STEPS_BY_TURN, ai_steps, True)
                mask = compute_action_mask(game)
                action, _ = ai_model.predict(obs, action_masks=mask, deterministic=True)
                game.execute_action(int(action))
                ai_steps += 1
                if int(action) == Actions.ACTION_DROP or ai_steps >= MAX_STEPS_BY_TURN:
                    ai_steps = 0
                renderer.draw()
                pygame.display.flip()
                await asyncio.sleep(ai_delay / 1000)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if game.status == game.PLAYING:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_TAB:
                        game.execute_action(Actions.ACTION_CYCLE_PIECE)
                    elif event.key == pygame.K_LEFT:
                        game.execute_action(Actions.ACTION_MOVE_LEFT)
                    elif event.key == pygame.K_RIGHT:
                        game.execute_action(Actions.ACTION_MOVE_RIGHT)
                    elif event.key == pygame.K_UP:
                        game.execute_action(Actions.ACTION_ROTATE)
                    elif event.key == pygame.K_RETURN:
                        game.execute_action(Actions.ACTION_FLIP)
                    elif event.key == pygame.K_DOWN:
                        game.execute_action(Actions.ACTION_DROP)
                    elif event.key == pygame.K_p:
                        if not game.player_has_valid_moves(game.current_player):
                            game.force_pass()
                elif game.status == game.GAMEOVER:
                    elapsed = pygame.time.get_ticks() - gameover_since if gameover_since else 0
                    if elapsed >= 3000:
                        if event.key == pygame.K_r:
                            game.reset()
                            game.start_turn()
                            gameover_since = None
                        elif event.key == pygame.K_ESCAPE:
                            running = False

        if game.status == game.GAMEOVER and gameover_since is None:
            gameover_since = pygame.time.get_ticks()

        renderer.draw()
        pygame.display.flip()
        await asyncio.sleep(1.0 / FPS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyLinkx — human vs human or human vs AI")
    parser.add_argument("--ai-model", default=None, help="Path to model zip (AI plays as P1, human as P2)")
    parser.add_argument("--ai-delay", type=int, default=150, help="Milliseconds between AI actions (default: 150)")
    args = parser.parse_args()

    ai_model = None
    if args.ai_model:
        print(f"Loading AI model from {args.ai_model}...")
        ai_model = load_ai_model(args.ai_model)
        print("AI model loaded. You play as P2.")

    asyncio.run(main(ai_model=ai_model, ai_delay=args.ai_delay))
