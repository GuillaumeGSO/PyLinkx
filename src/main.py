import argparse
import asyncio
import sys
from pathlib import Path

# For native execution, add src/ so bare 'game.*' imports resolve.
# In WASM (pygbag) the virtual filesystem root is already src/.
if sys.platform != 'emscripten':
    sys.path.insert(0, str(Path(__file__).parent))
    sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy  # noqa: F401 — imported so pygbag pre-loads it in Pyodide
import pygame
from game.game import Game, Actions
from game.game_renderer import GameRenderer
from game.menu_renderer import MenuRenderer
try:
    from inference.onnx_policy import OnnxPolicy
except ImportError:
    OnnxPolicy = None

try:
    from inference.observation import build_observation, compute_action_mask
except ImportError:
    try:
        from src.inference.observation import build_observation, compute_action_mask
    except ImportError:
        build_observation = None
        compute_action_mask = None

# App states
MENU = "menu"
DIFFICULTY = "difficulty"
HOW_TO_PLAY = "how_to_play"
LOADING = "loading"
PLAYING = "playing"
GAMEOVER = "gameover"

FPS = 10
MAX_STEPS_BY_TURN = 36

# In WASM (pygbag), non-Python files are served under /assets/;
# in a PyInstaller frozen build, data files live under sys._MEIPASS;
# otherwise use __file__-relative path.
if getattr(sys, 'frozen', False):
    _MODELS_DIR = str(Path(sys._MEIPASS) / "models")
else:
    _MODELS_DIR = str(Path(__file__).parent / "models")
MODEL_PATHS = {
    "easy":   f"{_MODELS_DIR}/easy_model.onnx",
    "medium": f"{_MODELS_DIR}/medium_model.onnx",
    "hard":   f"{_MODELS_DIR}/hard_model.onnx",
}
DIFF_KEYS = ["easy", "medium", "hard"]  # maps diff_cursor 0-2 to keys


async def main(ai_model_override=None, ai_delay: int = 150):
    pygame.init()
    screen = pygame.display.set_mode(
        (GameRenderer.SCREEN_WIDTH, GameRenderer.SCREEN_HEIGHT)
    )
    pygame.display.set_caption("PyLinkx")

    menu_renderer = MenuRenderer(screen)

    # If a model was passed via CLI, skip the menu and jump straight to gameplay
    if ai_model_override is not None:
        app_state = PLAYING
        ai_model = ai_model_override
    else:
        app_state = MENU
        ai_model = None

    menu_cursor = 0
    diff_cursor = 0
    loading_label = None
    loading_loader = None      # WasmModelLoader (WASM only)
    loading_error = None       # error message string
    loading_error_timer = 0

    game = None
    renderer = None
    gameover_since = None

    def start_game():
        nonlocal game, renderer, gameover_since
        game = Game()
        game.start_turn()
        renderer = GameRenderer(screen, game)
        gameover_since = None

    if app_state == PLAYING:
        start_game()

    running = True
    while running:

        # ------------------------------------------------------------------
        # AI turn (only in PLAYING, only when it's player index 0's turn)
        # ------------------------------------------------------------------
        if (app_state == PLAYING and ai_model and game and
                game.status == game.PLAYING and
                game.players.index(game.current_player) == 0):
            ai_steps = 0
            while (game.status == game.PLAYING and
                   game.players.index(game.current_player) == 0):
                obs = build_observation(game, MAX_STEPS_BY_TURN, ai_steps, True)
                mask = compute_action_mask(game)
                if asyncio.iscoroutinefunction(ai_model.predict):
                    action, _ = await ai_model.predict(obs, action_masks=mask, deterministic=True)
                else:
                    action, _ = ai_model.predict(obs, action_masks=mask, deterministic=True)
                game.execute_action(int(action))
                ai_steps += 1
                if int(action) == Actions.ACTION_DROP or ai_steps >= MAX_STEPS_BY_TURN:
                    ai_steps = 0
                renderer.draw()
                pygame.display.flip()
                await asyncio.sleep(ai_delay / 1000)

        # ------------------------------------------------------------------
        # Model loading (one step per frame — no internal polling)
        # ------------------------------------------------------------------
        if app_state == LOADING:
            if loading_error:
                loading_error_timer += 1
                if loading_error_timer > FPS * 3:  # 3 seconds
                    loading_error = None
                    app_state = MENU
            elif loading_loader:
                loading_loader.step()
                if loading_loader.done:
                    from inference.wasm_onnx_policy import WasmOnnxPolicy
                    ai_model = WasmOnnxPolicy()
                    start_game()
                    app_state = PLAYING
                    loading_loader = None
                elif loading_loader.error:
                    loading_error = loading_loader.error
                    loading_error_timer = 0
                    loading_loader = None
            else:
                # Native: synchronous load
                try:
                    ai_model = OnnxPolicy(MODEL_PATHS[loading_label])
                    start_game()
                    app_state = PLAYING
                except Exception as e:
                    loading_error = str(e)
                    loading_error_timer = 0

        # ------------------------------------------------------------------
        # Events
        # ------------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:

                if app_state == MENU:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        menu_cursor = (menu_cursor - 1) % len(MenuRenderer.MENU_ITEMS)
                    elif event.key == pygame.K_DOWN:
                        menu_cursor = (menu_cursor + 1) % len(MenuRenderer.MENU_ITEMS)
                    elif event.key == pygame.K_RETURN:
                        chosen = MenuRenderer.MENU_ITEMS[menu_cursor]
                        if chosen == "2 Human Players":
                            ai_model = None
                            start_game()
                            app_state = PLAYING
                        elif chosen == "Human vs Computer":
                            app_state = DIFFICULTY
                        elif chosen == "How to Play":
                            app_state = HOW_TO_PLAY

                elif app_state == DIFFICULTY:
                    if event.key == pygame.K_ESCAPE:
                        app_state = MENU
                    elif event.key == pygame.K_UP:
                        diff_cursor = (diff_cursor - 1) % len(MenuRenderer.DIFF_ITEMS)
                    elif event.key == pygame.K_DOWN:
                        diff_cursor = (diff_cursor + 1) % len(MenuRenderer.DIFF_ITEMS)
                    elif event.key == pygame.K_RETURN:
                        if diff_cursor == len(DIFF_KEYS):  # "← Back"
                            app_state = MENU
                        else:
                            loading_label = DIFF_KEYS[diff_cursor]
                            loading_error = None
                            if sys.platform == "emscripten":
                                from inference.wasm_onnx_policy import WasmModelLoader
                                loading_loader = WasmModelLoader(
                                    MODEL_PATHS[loading_label])
                            else:
                                loading_loader = None
                            app_state = LOADING

                elif app_state == LOADING:
                    if event.key == pygame.K_ESCAPE:
                        loading_loader = None
                        loading_error = None
                        app_state = MENU

                elif app_state == HOW_TO_PLAY:
                    app_state = MENU

                elif app_state == PLAYING:
                    if game and game.status == game.PLAYING:
                        if event.key == pygame.K_ESCAPE:
                            app_state = MENU
                            game = None
                            renderer = None
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

                elif app_state == GAMEOVER:
                    elapsed = pygame.time.get_ticks() - gameover_since if gameover_since else 0
                    if elapsed >= 3000:
                        if event.key == pygame.K_r:
                            game.reset()
                            game.start_turn()
                            gameover_since = None
                            app_state = PLAYING
                        elif event.key in (pygame.K_ESCAPE, pygame.K_m):
                            app_state = MENU
                            game = None
                            renderer = None
                            ai_model = None

        # ------------------------------------------------------------------
        # Detect PLAYING → GAMEOVER transition
        # ------------------------------------------------------------------
        if app_state == PLAYING and game and game.status == game.GAMEOVER:
            app_state = GAMEOVER
            gameover_since = pygame.time.get_ticks()

        # ------------------------------------------------------------------
        # Render
        # ------------------------------------------------------------------
        if app_state == MENU:
            menu_renderer.draw_menu(menu_cursor)
        elif app_state == DIFFICULTY:
            menu_renderer.draw_difficulty(diff_cursor)
        elif app_state == HOW_TO_PLAY:
            menu_renderer.draw_how_to_play()
        elif app_state == LOADING:
            if loading_error:
                menu_renderer.draw_error(loading_error)
            elif loading_loader:
                menu_renderer.draw_loading(loading_loader.label)
            else:
                menu_renderer.draw_loading(loading_label.capitalize())
        elif app_state in (PLAYING, GAMEOVER) and renderer:
            renderer.draw()

        pygame.display.flip()
        await asyncio.sleep(1.0 / FPS)

    pygame.quit()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="PyLinkx — human vs human or human vs AI")
    parser.add_argument("--ai-model", default=None,
                        help="Path to model .onnx (AI plays as P1, human as P2) — bypasses menu")
    parser.add_argument("--ai-delay", type=int, default=150,
                        help="Milliseconds between AI actions (default: 150)")
    args = parser.parse_args()

    ai_model = None
    if args.ai_model:
        print(f"Loading AI model from {args.ai_model}...")
        ai_model = OnnxPolicy(args.ai_model)
        print("AI model loaded. You play as P2.")

    asyncio.run(main(ai_model_override=ai_model, ai_delay=args.ai_delay))
