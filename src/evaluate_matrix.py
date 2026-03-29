#!/usr/bin/env python3
"""
Round-robin evaluation matrix for PyLinkx models.

Evaluates all model pairs and prints a win-rate table for human inspection.
Used to validate Easy/Medium/Hard model selection before committing.

Usage:
    # Explicit model list
    python src/evaluate_matrix.py \
      --models models/base_line_model.zip models/loop_1/best_model.zip \
        models/loop_2/best_model.zip \
        models/loop_3/best_model.zip \
        models/loop_4/best_model.zip \
      --labels baseline loop1 loop2 loop3 loop4 \
      --episodes 200

    # Pull all loops from manifest
    python src/evaluate_matrix.py --from-manifest models/manifest.json --episodes 200

    # Save results to JSON
    python src/evaluate_matrix.py --from-manifest models/manifest.json --episodes 200 --output models/eval_matrix.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import evaluate_agent


def build_matrix(models: list[str], labels: list[str], episodes: int,
                 maxsteps: int, maxstepsbyturn: int) -> dict:
    """
    Run all ordered pairs (i, j) where i != j.
    Returns a dict: results[(i, j)] = evaluate_agent output dict.
    """
    n = len(models)
    results = {}
    total_pairs = n * (n - 1)
    done = 0

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            done += 1
            print(f"\n[{done}/{total_pairs}] {labels[i]} (P1) vs {labels[j]} (P2)")
            results[(i, j)] = evaluate_agent(
                model_path=models[i],
                opponent_model_path=models[j],
                num_episodes=episodes,
                max_steps=maxsteps,
                max_steps_by_turn=maxstepsbyturn,
            )

    return results


def print_matrix(models: list[str], labels: list[str], results: dict, metric: str = "win_rate"):
    """Print ASCII win-rate matrix. Row = P1, Col = P2."""
    n = len(models)
    col_w = max(len(l) for l in labels) + 2
    row_w = col_w

    header = f"{'':>{row_w}}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\nWin rate of ROW (P1) vs COL (P2) — {metric}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i in range(n):
        row = f"{labels[i]:>{row_w}}"
        for j in range(n):
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                val = results[(i, j)][metric]
                row += f"{val:>{col_w - 1}.1%} "
        print(row)

    print("-" * len(header))


def main():
    parser = argparse.ArgumentParser(description="Round-robin model evaluation matrix")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--models", nargs="+", help="Paths to model zip files")
    group.add_argument("--from-manifest", help="Path to manifest.json (uses all loop models)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels (must match --models count)")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per pair (default: 200)")
    parser.add_argument("--maxsteps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--maxstepsbyturn", type=int, default=36, help="Max steps per turn")
    parser.add_argument("--output", default=None,
                        help="Save full results to this JSON file (e.g. models/eval_matrix.json)")
    args = parser.parse_args()

    # Resolve models and labels
    if args.from_manifest:
        with open(args.from_manifest) as f:
            manifest = json.load(f)
        entries = manifest.get("loops", [])
        models = [e["model_path"] for e in entries]
        labels = [f"loop{e['loop']}" for e in entries]
    else:
        models = args.models
        labels = args.labels or [f"model{i}" for i in range(len(models))]

    if len(labels) != len(models):
        print(f"ERROR: --labels count ({len(labels)}) must match --models count ({len(models)})")
        sys.exit(1)

    # Verify all models exist
    missing = [m for m in models if not os.path.exists(m)]
    if missing:
        print(f"ERROR: Models not found: {missing}")
        sys.exit(1)

    print(f"Models: {list(zip(labels, models))}")
    print(f"Episodes per pair: {args.episodes}")
    print(f"Total pairs: {len(models) * (len(models) - 1)}")

    results = build_matrix(models, labels, args.episodes, args.maxsteps, args.maxstepsbyturn)

    print_matrix(models, labels, results, "win_rate")
    print_matrix(models, labels, results, "path_win_rate")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        serializable = {
            f"{labels[i]}_vs_{labels[j]}": v
            for (i, j), v in results.items()
        }
        with open(args.output, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
