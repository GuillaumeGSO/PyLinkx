#!/usr/bin/env python3
"""
Basic training script for PyLinkx RL agent using Stable-Baselines3.

This script demonstrates how to:
1. Create and initialize a PyLinkx Gymnasium environment
2. Train an RL agent (PPO) on the environment
3. Save and load trained models
4. Evaluate agent performance
"""

import os
import subprocess
import numpy as np
import sys
from pathlib import Path
import pygame
import torch as th
import torch.nn as nn
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Add parent directory to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.game.game_renderer import GameRenderer

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecNormalize

from src.training.game_env import Actions, PyLinkxEnv


class PyLinkxFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor: CNN for 9x9 grid + MLP for scalars."""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        self.grid_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # -> 32x9x9
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> 64x9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),  # -> 64x7x7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.grid_linear = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )

        scalar_dim = observation_space.spaces["scalars"].shape[0]
        self.scalar_net = nn.Sequential(
            nn.Linear(scalar_dim, 128),
            nn.ReLU(),
        )

    def forward(self, observations):
        grid = observations["grid"].permute(0, 3, 1, 2)  # HWC -> CHW
        grid_features = self.grid_linear(self.grid_cnn(grid))
        scalar_features = self.scalar_net(observations["scalars"])
        return th.cat([grid_features, scalar_features], dim=1)


def linear_schedule(initial_value: float):
    """Linear decay from initial_value to 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class RenderOnBestCallback(BaseCallback):
    def __init__(self, max_steps: int, max_steps_by_turn: int, opponent_model_path: str | None = None,
                 best_model_path: str = "models/best_model.zip"):
        super().__init__(verbose=0)
        self._max_steps = max_steps
        self._max_steps_by_turn = max_steps_by_turn
        self._opponent_model_path = opponent_model_path
        self._best_model_path = best_model_path
        self._render_proc: subprocess.Popen | None = None

    def _on_step(self) -> bool:
        if self._render_proc and self._render_proc.poll() is None:
            return True  # previous render still running, skip
        cmd = [
            sys.executable, __file__,
            "--mode", "evaluate",
            "--model", self._best_model_path,
            "--eval-episodes", "1",
            "--render",
            "--maxsteps", str(self._max_steps),
            "--maxstepsbyturn", str(self._max_steps_by_turn),
        ]
        if self._opponent_model_path:
            cmd.extend(["--opponent-model", self._opponent_model_path])
        self._render_proc = subprocess.Popen(cmd)
        return True


class GameMetricsCallback(BaseCallback):
    """Logs game-specific metrics (win rate, path/score splits) to TensorBoard.

    When baseline_model_path is provided, the eval env uses it as the opponent
    (instead of the training pool) for a stable, comparable plateau signal.
    """

    def __init__(self, eval_env_kwargs: dict, n_eval_episodes: int = 20,
                 eval_freq: int = 20000, baseline_model_path: str | None = None,
                 min_timesteps: int = 0, plateau_window: int = 5,
                 plateau_threshold: float = 0.02, verbose: int = 0):
        super().__init__(verbose)
        self._n_eval_episodes = n_eval_episodes
        self._eval_freq = eval_freq
        self._min_timesteps = min_timesteps
        self._plateau_window = plateau_window
        self._plateau_threshold = plateau_threshold
        self._win_rate_history: list[float] = []

        # Use baseline model as fixed eval opponent if provided, else use training kwargs
        if baseline_model_path:
            base_kwargs = {k: v for k, v in eval_env_kwargs.items()
                          if k not in ("opponent_model_path", "opponent_model_paths", "opponent_weights")}
            self._eval_env_kwargs = {**base_kwargs, "opponent_model_path": baseline_model_path}
        else:
            self._eval_env_kwargs = eval_env_kwargs

    def _on_step(self) -> bool:
        if self.n_calls % self._eval_freq != 0:
            return True

        env = PyLinkxEnv(**self._eval_env_kwargs)
        env = ActionMasker(env, lambda e: e.valid_action_mask())

        wins, path_wins, score_wins, losses, total_p2_drops = 0, 0, 0, 0, 0

        for _ in range(self._n_eval_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                action_masks = env.unwrapped.valid_action_mask()
                action, _ = self.model.predict(obs, deterministic=True, action_masks=action_masks)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated

            if info.get("winner_idx") == 0:
                wins += 1
                if info.get("win_type") == "path":
                    path_wins += 1
                else:
                    score_wins += 1
            elif info.get("winner_idx") == 1:
                losses += 1
            total_p2_drops += info.get("p2_drops", 0)

        env.close()

        n = self._n_eval_episodes
        win_rate = wins / n
        self.logger.record("game/win_rate", win_rate)
        self.logger.record("game/path_win_rate", path_wins / n)
        self.logger.record("game/score_win_rate", score_wins / n)
        self.logger.record("game/loss_rate", losses / n)
        self.logger.record("game/mean_p2_drops", total_p2_drops / n)

        # Plateau detection
        self._win_rate_history.append(win_rate)
        if (self._min_timesteps > 0
                and self.num_timesteps >= self._min_timesteps
                and len(self._win_rate_history) >= self._plateau_window):
            recent = self._win_rate_history[-self._plateau_window:]
            if max(recent) - min(recent) < self._plateau_threshold:
                print(f"\n[Plateau] Win rate stable at {win_rate:.2f} over last "
                      f"{self._plateau_window} evals — stopping early.")
                return False

        return True


def train_agent(
    total_timesteps: int = 100_000,
    eval_episodes: int = 100,
    max_steps: int = 500,
    max_steps_by_turn: int = 20,
    envs: int = max(1, (os.cpu_count() or 4) - 1),
    model_save_path: str = "models/ppo_pylinkx.zip",
    model_save_dir: str | None = None,
    render: bool = False,
    opponent_model_path: str | None = None,
    opponent_model_paths: list[str] | None = None,
    opponent_weights: list[float] | None = None,
    game_eval_freq: int = 20_000,
    baseline_model_path: str | None = None,
    min_timesteps: int = 0,
    plateau_window: int = 5,
    plateau_threshold: float = 0.02,
    tb_log_name: str = "PPO",
):
    """
    Train a PPO agent on the PyLinkx environment.

    Args:
        total_timesteps: Total number of training timesteps
        eval_episodes: Number of episodes per evaluation
        model_save_path: Path to save the trained model
        max_steps: Maximum steps per episode to prevent infinite loops
        max_steps_by_turn: Maximum steps per turn before a drop is forced
        opponent_model_path: Path to opponent model for P2 (None = drop-first fallback)
        game_eval_freq: How often (in timesteps) to run game metrics evaluation
    """
    print("=" * 60)
    print("PyLinkx RL Training Script")
    print("=" * 60)

    # Resolve save directory and paths
    if model_save_dir:
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, "ppo_pylinkx.zip")
        best_model_save_path = model_save_dir
    else:
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        best_model_save_path = str(Path(model_save_path).parent)

    # Create a vectorized environment (for parallel training)
    print("\n1. Creating environment...")
    n_envs = envs  # Number of parallel environments
    if opponent_model_paths:
        weights_desc = "custom" if opponent_weights else "linear"
        opponent_desc = f"pool of {len(opponent_model_paths)} models ({weights_desc} weights)"
    else:
        opponent_desc = opponent_model_path or "drop-first fallback"
    print(f"Using {n_envs} parallel environments (CPU cores: {os.cpu_count()})")
    print(f"Opponent: {opponent_desc}")
    def make_masked_env(**kwargs):
        env = PyLinkxEnv(**kwargs)
        return ActionMasker(env, lambda e: e.valid_action_mask())

    if opponent_model_paths:
        env_kwargs = {"max_steps": max_steps, "max_steps_by_turn": max_steps_by_turn,
                      "opponent_model_paths": opponent_model_paths,
                      "opponent_weights": opponent_weights}
    else:
        env_kwargs = {"max_steps": max_steps, "max_steps_by_turn": max_steps_by_turn,
                      "opponent_model_path": opponent_model_path}
    env = make_vec_env(make_masked_env, n_envs=n_envs, env_kwargs=env_kwargs, wrapper_class=Monitor)
    env = VecNormalize(env, norm_reward=True, norm_obs=False)

    # Create evaluation environment (must be wrapped the same way for EvalCallback)
    eval_env = make_vec_env(make_masked_env, n_envs=1, env_kwargs=env_kwargs, wrapper_class=Monitor)
    eval_env = VecNormalize(eval_env, norm_reward=False, norm_obs=False, training=False)

    # eval_freq is in rollout steps (n_calls), not total timesteps.
    # Divide by n_envs so callbacks fire at the intended total-timestep frequency.
    eval_freq_steps = max(1, game_eval_freq // n_envs)

    # Setup evaluation callback
    render_callback = RenderOnBestCallback(
        max_steps, max_steps_by_turn, opponent_model_path,
        best_model_path=os.path.join(best_model_save_path, "best_model.zip"),
    ) if render else None
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=render_callback,
        best_model_save_path=best_model_save_path,
        log_path="./logs/",
        eval_freq=eval_freq_steps,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    game_metrics_callback = GameMetricsCallback(
        eval_env_kwargs=env_kwargs,
        eval_freq=eval_freq_steps,
        baseline_model_path=baseline_model_path,
        min_timesteps=min_timesteps,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
    )

    callbacks = CallbackList([eval_callback, game_metrics_callback])

    # Create and train the agent
    print("2. Creating MaskablePPO agent...")
    policy_kwargs = {
        "features_extractor_class": PyLinkxFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": False,
    }
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4, #linear_schedule(3e-4),
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./logs",
    )

    print("\n3. Starting training...")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Parallel environments: {n_envs}")
    print(f"   TensorBoard: tensorboard --logdir logs")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name=tb_log_name,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save the model
    print(f"\n4. Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("   Model saved!")

    env.close()

    return model


def evaluate_agent(
    model_path: str, num_episodes: int = 10, render: bool = False, max_steps: int = 100,
    max_steps_by_turn: int = 20, opponent_model_path: str | None = None,
):
    """
    Evaluate a trained agent (P1) against an opponent (P2).

    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        max_steps_by_turn: Maximum steps per turn before a drop is forced
        opponent_model_path: Path to opponent model for P2 (None = drop-first fallback)
    """
    print("\n" + "=" * 60)
    print("Evaluating Agent")
    print("=" * 60)

    # Load the trained model
    print(f"\n1. Loading model from {model_path}...")
    model = MaskablePPO.load(model_path)

    # Create evaluation environment
    env = PyLinkxEnv(
        render_mode="debug" if render else None, max_steps=max_steps,
        max_steps_by_turn=max_steps_by_turn, opponent_model_path=opponent_model_path,
    )

    episode_rewards = []
    episode_lengths = []
    p2_drops_list = []
    wins, path_wins, score_wins, losses = 0, 0, 0, 0

    print(f"\n2. Running {num_episodes} evaluation episodes...")
    print("-" * 60)

    renderer = None
    clock = None
    if render:
        pygame.init()
        info = pygame.display.Info()
        x = info.current_w - GameRenderer.SCREEN_WIDTH - 20
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},50"
        screen = pygame.display.set_mode(
            (GameRenderer.SCREEN_WIDTH, GameRenderer.SCREEN_HEIGHT)
        )
        pygame.display.set_caption("PyLinkx RL Environment - Debug Render")
        renderer = GameRenderer(screen, env.game)
        clock = pygame.time.Clock()

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action_masks = env.valid_action_mask()
            action, _ = model.predict(obs, deterministic=True, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                env.render(renderer, action=int(action))
                (
                    clock.tick(env.metadata["render_fps"]) if clock else None
                )

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        p2_drops_list.append(info.get("p2_drops", 0))

        if info.get("winner_idx") == 0:
            wins += 1
            if info.get("win_type") == "path":
                path_wins += 1
            else:
                score_wins += 1
        elif info.get("winner_idx") == 1:
            losses += 1

        winner = "P1" if info.get("winner_idx") == 0 else ("P2" if info.get("winner_idx") == 1 else "None")
        print(
            f"   Episode {episode + 1:3d}: Reward = {episode_reward:7.2f}, "
            f"Length = {episode_length:4d}, P2 Drops = {info.get('p2_drops', 0)}, "
            f"Winner = {winner} ({info.get('win_type', '-')})"
        )

    # Print statistics
    print("\n3. Evaluation Statistics:")
    print("-" * 60)
    print(
        f"   Mean Reward:     {np.mean(episode_rewards):.2f} "
        f"± {np.std(episode_rewards):.2f}"
    )
    print(
        f"   Mean Length:     {np.mean(episode_lengths):.1f} "
        f"± {np.std(episode_lengths):.1f}"
    )
    print(f"   Max Reward:      {np.max(episode_rewards):.2f}")
    print(f"   Min Reward:      {np.min(episode_rewards):.2f}")
    print(f"   Mean P2 Drops:   {np.mean(p2_drops_list):.1f} ± {np.std(p2_drops_list):.1f}")

    env.close()

    n = num_episodes
    return {
        "win_rate": wins / n,
        "path_win_rate": path_wins / n,
        "score_win_rate": score_wins / n,
        "loss_rate": losses / n,
        "mean_reward": float(np.mean(episode_rewards)),
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "p2_drops": p2_drops_list,
    }


def quick_test():
    """Quick test to verify environment setup."""
    print("\n" + "=" * 60)
    print("Quick Environment Test")
    print("=" * 60)

    _default_envs = max(1, (os.cpu_count() or 4) - 1)
    print(f"Auto-detected parallel environments: {_default_envs} (CPU cores: {os.cpu_count()})")

    print("\n1. Creating environment...")
    env = PyLinkxEnv()

    print("2. Resetting environment...")
    obs, info = env.reset()
    # print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")

    print("3. Running 10 random steps...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"   Step {step + 1}: player={info['current_player_idx']+1}, action={action}, reward={reward:.2f}, "
            f"valid={info['action_valid']}, done={terminated or truncated}"
        )

        if terminated or truncated:
            break

    print("\n   ✓ Environment working correctly!")
    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyLinkx RL Training Script")
    parser.add_argument(
        "--mode",
        choices=["test", "train", "evaluate"],
        default="test",
        help="Script mode: test, train, or evaluate",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Total training timesteps",
    )
    _default_envs = max(1, (os.cpu_count() or 4) - 1)
    parser.add_argument(
        "--envs",
        type=int,
        default=_default_envs,
        help=f"Number of parallel environments (default: auto-detected {_default_envs})",
    )
    parser.add_argument(
        "--maxsteps",
        type=int,
        default=500,
        help="Limit episode length",
    )
    parser.add_argument(
        "--maxstepsbyturn",
        type=int,
        default=36,
        help="Maximum steps per turn before a drop is forced (default: 20)",
    )
    parser.add_argument(
        "--model",
        default="models/ppo_pylinkx.zip",
        help="Path to model file",
    )
    parser.add_argument(
        "--model-save-dir",
        default=None,
        help="Directory to save model and best_model (overrides --model path, used by pipeline)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render evaluation episodes",
    )
    parser.add_argument(
        "--opponent-model",
        default=None,
        help="Path to opponent model for P2 (default: drop-first fallback)",
    )
    parser.add_argument(
        "--opponent-models",
        nargs="+",
        default=None,
        help="Paths to a pool of opponent models (sampled with weights, used by pipeline)",
    )
    parser.add_argument(
        "--opponent-weights",
        nargs="+",
        type=float,
        default=None,
        help="Sampling weights for --opponent-models (must match count; default: linear 1,2,…,N)",
    )
    parser.add_argument(
        "--game-eval-freq",
        type=int,
        default=60000,
        help="How often (in timesteps) to log game metrics to TensorBoard (default: 60000)",
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Fixed reference model for plateau detection eval (default: uses training opponent)",
    )
    parser.add_argument(
        "--min-timesteps",
        type=int,
        default=0,
        help="Minimum timesteps before plateau detection activates (default: 0 = disabled)",
    )
    parser.add_argument(
        "--plateau-window",
        type=int,
        default=5,
        help="Number of game evals to check for plateau (default: 5)",
    )
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=0.02,
        help="Max win rate range within window to declare plateau (default: 0.02)",
    )
    parser.add_argument(
        "--tb-log-name",
        default="PPO",
        help="TensorBoard run name (subfolder under ./logs, default: PPO)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        print(f"\n{'='*60}")
        print(f"Training Parameters")
        print(f"{'='*60}")
        print(f"  timesteps:             {args.timesteps:,}")
        print(f"  min-timesteps:         {args.min_timesteps:,}")
        print(f"  envs:                  {args.envs}")
        print(f"  maxsteps:              {args.maxsteps}")
        print(f"  maxstepsbyturn:        {args.maxstepsbyturn}")
        print(f"  game-eval-freq:        {args.game_eval_freq:,}")
        print(f"  baseline-model:        {args.baseline_model}")
        print(f"  opponent-model:        {args.opponent_model}")
        print(f"  opponent-models:       {args.opponent_models}")
        print(f"  opponent-weights:      {args.opponent_weights}")
        print(f"  model-save-dir:        {args.model_save_dir}")
        print(f"  plateau-window:        {args.plateau_window}")
        print(f"  plateau-threshold:     {args.plateau_threshold}")
        print(f"{'='*60}")

    if args.mode == "test":
        quick_test()
    elif args.mode == "train":
        trained_model = train_agent(
            total_timesteps=args.timesteps,
            model_save_path=args.model,
            model_save_dir=args.model_save_dir,
            max_steps=args.maxsteps,
            max_steps_by_turn=args.maxstepsbyturn,
            envs=args.envs,
            render=args.render,
            opponent_model_path=args.opponent_model,
            opponent_model_paths=args.opponent_models,
            opponent_weights=args.opponent_weights,
            game_eval_freq=args.game_eval_freq,
            baseline_model_path=args.baseline_model,
            min_timesteps=args.min_timesteps,
            plateau_window=args.plateau_window,
            plateau_threshold=args.plateau_threshold,
            tb_log_name=args.tb_log_name,
        )
        print("\n✓ Training completed successfully!")
    elif args.mode == "evaluate":
        results = evaluate_agent(
            model_path=args.model,
            num_episodes=args.eval_episodes,
            max_steps=args.maxsteps,
            max_steps_by_turn=args.maxstepsbyturn,
            render=args.render,
            opponent_model_path=args.opponent_model,
        )
        print("\n✓ Evaluation completed!")
