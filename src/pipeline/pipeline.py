#!/usr/bin/env python3
"""
Automated self-play training pipeline for PyLinkx.

Runs iterative training loops, versions each model, evaluates against a fixed
baseline, and selects Easy/Medium/Hard difficulty models when done.

Usage:
    python src/pipeline/pipeline.py \
      --max-loops 10 \
      --timesteps 4000000 \
      --min-timesteps 1000000 \     #Remove this line (or 0) to disable plateau check
      --cross-loop-threshold 0.55 \
      --eval-episodes 200 \
      --envs 7 \
      --baseline-model models/base_line_model.zip
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Allow importing train.py from same directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.training.train import evaluate_agent

MANIFEST_PATH = "src/pipeline/manifest.json"
BASELINE_LOOP = 0


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {"baseline_model": None, "loops": [], "difficulty_models": None}


def save_manifest(manifest: dict):
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def bootstrap_manifest(baseline_model: str) -> dict:
    """Register the baseline (loop 5) model into a fresh or existing manifest."""
    manifest = load_manifest()
    manifest["baseline_model"] = baseline_model

    # Register loop 5 if not already present
    if not any(e["loop"] == BASELINE_LOOP for e in manifest["loops"]):
        manifest["loops"].append({
            "loop": BASELINE_LOOP,
            "model_path": baseline_model,
            "opponent": "legacy",
            "timesteps_trained": 6000000,
            "win_rate_vs_prev": None,
            "win_rate_vs_baseline": None,
            "path_win_rate_vs_baseline": None,
            "score_win_rate_vs_baseline": None,
            "mean_reward_vs_baseline": None,
            "note": "baseline model — win_rate_vs_baseline is null by definition",
        })
        save_manifest(manifest)
        print(f"[Bootstrap] Registered loop {BASELINE_LOOP} as baseline: {baseline_model}")
    else:
        print(f"[Bootstrap] Loop {BASELINE_LOOP} already in manifest, skipping.")

    return manifest


def append_loop_to_manifest(manifest: dict, loop_n: int, model_path: str,
                             opponent_pool: list[str], opponent_weights: list[float],
                             timesteps: int, results_vs_prev: dict, results_vs_baseline: dict):
    entry = {
        "loop": loop_n,
        "model_path": model_path,
        "opponent_pool": opponent_pool,
        "opponent_weights": opponent_weights,
        "timesteps_trained": timesteps,
        "win_rate_vs_prev": results_vs_prev["win_rate"],
        "win_rate_vs_baseline": results_vs_baseline["win_rate"],
        "path_win_rate_vs_baseline": results_vs_baseline["path_win_rate"],
        "score_win_rate_vs_baseline": results_vs_baseline["score_win_rate"],
        "mean_reward_vs_baseline": results_vs_baseline["mean_reward"],
    }
    manifest["loops"].append(entry)
    save_manifest(manifest)


def select_difficulty_models(manifest: dict) -> bool:
    """
    Pick the best Easy/Medium/Hard triplet from manifest loops.

    Selection criteria:
      - Monotone strength: win_rate_vs_baseline[i] < [j] < [k]
      - score = strength_spread + style_spread
        - strength_spread = win_rate_vs_baseline[k] - win_rate_vs_baseline[i]
        - style_spread = std of path_win_rate_vs_baseline across the triplet
    """
    eligible = [e for e in manifest["loops"] if e.get("win_rate_vs_baseline") is not None]
    if len(eligible) < 3:
        print(f"[Selection] Only {len(eligible)} loops with baseline metrics — need ≥3. Skipping.")
        return False

    best_score = -1.0
    best_triplet = None

    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            for k in range(j + 1, len(eligible)):
                ei, ej, ek = eligible[i], eligible[j], eligible[k]
                wi = ei["win_rate_vs_baseline"]
                wj = ej["win_rate_vs_baseline"]
                wk = ek["win_rate_vs_baseline"]

                if not (wi < wj < wk):
                    continue

                strength_spread = wk - wi
                path_rates = [
                    ei["path_win_rate_vs_baseline"],
                    ej["path_win_rate_vs_baseline"],
                    ek["path_win_rate_vs_baseline"],
                ]
                style_spread = float(np.std(path_rates))
                score = strength_spread + style_spread

                if score > best_score:
                    best_score = score
                    best_triplet = (ei, ej, ek)

    if best_triplet is None:
        print("[Selection] No valid monotone triplet found.")
        return False

    easy, medium, hard = best_triplet
    manifest["difficulty_models"] = {
        "easy":   {"loop": easy["loop"],   "model_path": easy["model_path"],   "win_rate_vs_baseline": easy["win_rate_vs_baseline"]},
        "medium": {"loop": medium["loop"], "model_path": medium["model_path"], "win_rate_vs_baseline": medium["win_rate_vs_baseline"]},
        "hard":   {"loop": hard["loop"],   "model_path": hard["model_path"],   "win_rate_vs_baseline": hard["win_rate_vs_baseline"]},
    }
    print(f"[Selection] Easy=loop{easy['loop']} ({easy['win_rate_vs_baseline']:.1%}), "
          f"Medium=loop{medium['loop']} ({medium['win_rate_vs_baseline']:.1%}), "
          f"Hard=loop{hard['loop']} ({hard['win_rate_vs_baseline']:.1%}), "
          f"score={best_score:.3f}")
    return True


def build_opponent_pool(manifest: dict, pool_lookback: int) -> tuple[list[str], list[float]]:
    """
    Build the opponent pool for the next training loop.

    Always keeps the baseline (loop 0) for stylistic diversity, then takes the
    last `pool_lookback` RL loops. Weights are linear (1, 2, …, N) so recent
    opponents are sampled more often; the baseline always gets weight 1.

    Returns:
        pool:    list of model paths
        weights: corresponding sampling weights
    """
    all_loops = manifest["loops"]
    baseline = next((e for e in all_loops if e["loop"] == BASELINE_LOOP), None)
    rl_loops = [e for e in all_loops if e["loop"] != BASELINE_LOOP]

    recent_rl = rl_loops[-pool_lookback:] if pool_lookback > 0 else rl_loops
    selected = ([baseline] if baseline else []) + recent_rl

    n = len(selected)
    # Baseline always gets weight 1; RL loops get linearly increasing weights
    # so the most recent opponent is sampled most.
    weights = [1.0] + list(range(1, len(recent_rl) + 1)) if baseline else list(range(1, n + 1))

    pool = [e["model_path"] for e in selected]
    return pool, [float(w) for w in weights]


def run_training_loop(loop_n: int, model_save_dir: str, opponent_pool: list[str],
                      opponent_weights: list[float], args: argparse.Namespace):
    """Launch a training subprocess for one loop."""
    cmd = [
        sys.executable, str(Path(__file__).parent.parent / "training" / "train.py"),
        "--mode", "train",
        "--timesteps", str(args.timesteps),
        "--min-timesteps", str(args.min_timesteps),
        "--plateau-window", str(args.plateau_window),
        "--plateau-threshold", str(args.plateau_threshold),
        "--model-save-dir", model_save_dir,
        "--envs", str(args.envs),
        "--maxsteps", str(args.maxsteps),
        "--maxstepsbyturn", str(args.maxstepsbyturn),
        "--game-eval-freq", str(args.game_eval_freq),
        "--baseline-model", args.baseline_model,
        "--opponent-models", *opponent_pool,
        "--opponent-weights", *[str(w) for w in opponent_weights],
        "--tb-log-name", f"loop_{loop_n:02d}",
    ]
    print(f"\n[Loop {loop_n:02d}] Training against pool: {opponent_pool} (weights: {opponent_weights})")
    print(f"[Loop {loop_n:02d}] Saving to: {model_save_dir}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[Loop {loop_n:02d}] Training subprocess failed (exit {result.returncode}).")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="PyLinkx automated self-play pipeline")
    parser.add_argument("--max-loops", type=int, default=10, help="Maximum number of training loops")
    parser.add_argument("--timesteps", type=int, default=4_000_000, help="Max timesteps per loop")
    parser.add_argument("--min-timesteps", type=int, default=1_000_000, help="Min timesteps before plateau check")
    parser.add_argument("--plateau-window", type=int, default=5, help="Plateau detection window (game evals)")
    parser.add_argument("--plateau-threshold", type=float, default=0.02, help="Plateau win-rate range threshold")
    parser.add_argument("--pool-lookback", type=int, default=3,
                        help="Keep baseline + last N RL loops in opponent pool (default: 3; 0 = keep all)")
    parser.add_argument("--cross-loop-threshold", type=float, default=0.55,
                        help="Min win rate vs prev loop to continue (default: 0.55)")
    parser.add_argument("--eval-episodes", type=int, default=200, help="Episodes for cross-loop evaluation")
    parser.add_argument("--envs", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="Parallel environments per training run")
    parser.add_argument("--maxsteps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--maxstepsbyturn", type=int, default=36, help="Max steps per turn")
    parser.add_argument("--game-eval-freq", type=int, default=60000, help="Game metrics eval frequency")
    parser.add_argument("--baseline-model", default="models/base_line_model.zip",
                        help="Fixed reference model for baseline evaluation")
    parser.add_argument("--start-loop", type=int, default=None,
                        help="Resume from this loop number (default: auto-detect from manifest)")
    args = parser.parse_args()

    if not os.path.exists(args.baseline_model):
        print(f"ERROR: Baseline model not found: {args.baseline_model}")
        sys.exit(1)

    # Bootstrap
    manifest = bootstrap_manifest(args.baseline_model)

    # Determine starting loop
    existing_loops = [e["loop"] for e in manifest["loops"]]
    start_loop = args.start_loop or (max(existing_loops) + 1 if existing_loops else BASELINE_LOOP + 1)
    end_loop = start_loop + args.max_loops - 1

    print(f"\n{'='*60}")
    print(f"Pipeline Parameters")
    print(f"{'='*60}")
    print(f"  baseline-model:        {args.baseline_model}")
    print(f"  start-loop:            {start_loop:02d}  (end: {end_loop:02d})")
    print(f"  max-loops:             {args.max_loops}")
    print(f"  timesteps:             {args.timesteps:,}")
    print(f"  min-timesteps:         {args.min_timesteps:,}")
    print(f"  envs:                  {args.envs}")
    print(f"  maxsteps:              {args.maxsteps}")
    print(f"  maxstepsbyturn:        {args.maxstepsbyturn}")
    print(f"  game-eval-freq:        {args.game_eval_freq:,}")
    print(f"  eval-episodes:         {args.eval_episodes}")
    print(f"  pool-lookback:         {args.pool_lookback} (0 = keep all)")
    print(f"  cross-loop-threshold:  {args.cross_loop_threshold:.0%}")
    print(f"  plateau-window:        {args.plateau_window}")
    print(f"  plateau-threshold:     {args.plateau_threshold}")
    print(f"{'='*60}")

    for loop_n in range(start_loop, end_loop + 1):
        # Build pruned opponent pool: baseline + last N RL loops, with linear weights
        pool, weights = build_opponent_pool(manifest, args.pool_lookback)
        model_save_dir = f"src/pipeline/models/loop_{loop_n}"
        best_model_path = os.path.join(model_save_dir, "best_model.zip")
        fallback_model_path = os.path.join(model_save_dir, "ppo_pylinkx.zip")

        # Train
        success = run_training_loop(loop_n, model_save_dir, pool, weights, args)
        if not success:
            print(f"[Pipeline] Stopping due to training failure at loop {loop_n:02d}.")
            break

        if not os.path.exists(best_model_path):
            if os.path.exists(fallback_model_path):
                print(f"[Loop {loop_n:02d}] No best_model.zip found, using ppo_pylinkx.zip as fallback.")
                best_model_path = fallback_model_path
            else:
                print(f"[Pipeline] No model found in {model_save_dir}. Stopping.")
                break

        # Evaluate vs previous loop (for cross-loop stopping)
        prev_model = manifest["loops"][-1]["model_path"]
        print(f"\n[Loop {loop_n:02d}] Evaluating vs previous ({prev_model})...")
        results_vs_prev = evaluate_agent(
            model_path=best_model_path,
            num_episodes=args.eval_episodes,
            max_steps=args.maxsteps,
            max_steps_by_turn=args.maxstepsbyturn,
            opponent_model_path=prev_model,
        )

        # Evaluate vs fixed baseline (for comparable strength metric)
        print(f"[Loop {loop_n:02d}] Evaluating vs baseline ({args.baseline_model})...")
        results_vs_baseline = evaluate_agent(
            model_path=best_model_path,
            num_episodes=args.eval_episodes,
            max_steps=args.maxsteps,
            max_steps_by_turn=args.maxstepsbyturn,
            opponent_model_path=args.baseline_model,
        )

        # Update manifest
        append_loop_to_manifest(manifest, loop_n, best_model_path, pool, weights,
                                 args.timesteps, results_vs_prev, results_vs_baseline)

        print(f"\n[Loop {loop_n:02d}] win_rate_vs_prev={results_vs_prev['win_rate']:.1%}  "
              f"win_rate_vs_baseline={results_vs_baseline['win_rate']:.1%}")

        # Cross-loop stopping
        if results_vs_prev["win_rate"] < args.cross_loop_threshold:
            print(f"[Pipeline] Loop {loop_n:02d} win rate vs prev "
                  f"{results_vs_prev['win_rate']:.1%} < {args.cross_loop_threshold:.0%} — stopping.")
            break

    # Select difficulty models
    print("\n[Pipeline] Selecting difficulty models...")
    select_difficulty_models(manifest)
    save_manifest(manifest)
    print(f"\n[Pipeline] Done. Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
