#!/usr/bin/env python3
"""
Basic training script for PyLinkx RL agent using Stable-Baselines3.

This script demonstrates how to:
1. Create and initialize a PyLinkx Gymnasium environment
2. Train an RL agent (PPO) on the environment
3. Save and load trained models
4. Evaluate agent performance
"""

import numpy as np
import sys
from pathlib import Path
from stable_baselines3.common.monitor import Monitor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from game_env import PyLinkxEnv


def train_agent(
    total_timesteps: int = 100000,
    eval_episodes: int = 100,
    model_save_path: str = "models/ppo_pylinkx.zip",
):
    """
    Train a PPO agent on the PyLinkx environment.

    Args:
        total_timesteps: Total number of training timesteps
        eval_episodes: Number of episodes per evaluation
        model_save_path: Path to save the trained model
    """
    print("=" * 60)
    print("PyLinkx RL Training Script")
    print("=" * 60)

    # Create output directory
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    # Create a vectorized environment (for parallel training)
    print("\n1. Creating environment...")
    n_envs = 4  # Number of parallel environments
    env = make_vec_env(PyLinkxEnv, n_envs=n_envs)

    # Create evaluation environment
    eval_env = Monitor(PyLinkxEnv())

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        eval_freq=1000,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )

    # Create and train the agent
    print("2. Creating PPO agent...")
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print("\n3. Starting training...")
    print(f"   Total timesteps: {total_timesteps}")
    print(f"   Parallel environments: {n_envs}")
    print("-" * 60)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Save the model
    print(f"\n4. Saving model to {model_save_path}...")
    model.save(model_save_path)
    print("   Model saved!")

    env.close()

    return model


def evaluate_agent(model_path: str, num_episodes: int = 10, render: bool = False):
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to the trained model
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
    """
    print("\n" + "=" * 60)
    print("Evaluating Agent")
    print("=" * 60)

    # Load the trained model
    print(f"\n1. Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create evaluation environment
    env = PyLinkxEnv(render_mode="debug" if render else None)

    episode_rewards = []
    episode_lengths = []

    print(f"\n2. Running {num_episodes} evaluation episodes...")
    print("-" * 60)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"   Episode {episode + 1:3d}: Reward = {episode_reward:7.2f}, "
            f"Length = {episode_length:4d}"
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

    env.close()

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
    }


def quick_test():
    """Quick test to verify environment setup."""
    print("\n" + "=" * 60)
    print("Quick Environment Test")
    print("=" * 60)

    print("\n1. Creating environment...")
    env = PyLinkxEnv()

    print("2. Resetting environment...")
    obs, info = env.reset()
    print(f"   Observation shape: {obs.shape}")
    print(f"   Action space: {env.action_space}")

    print("3. Running 10 random steps...")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"   Step {step + 1}: player={info['current_player_idx']+1}, action={action}, reward={reward:.2f}, "
            f"done={terminated or truncated}"
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
    parser.add_argument(
        "--model",
        default="models/ppo_pylinkx.zip",
        help="Path to model file",
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

    args = parser.parse_args()

    if args.mode == "test":
        quick_test()
    elif args.mode == "train":
        trained_model = train_agent(
            total_timesteps=args.timesteps,
            model_save_path=args.model,
        )
        print("\n✓ Training completed successfully!")
    elif args.mode == "evaluate":
        results = evaluate_agent(
            model_path=args.model,
            num_episodes=args.eval_episodes,
            render=args.render,
        )
        print("\n✓ Evaluation completed!")
