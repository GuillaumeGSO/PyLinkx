---
name: reward_shaping
description: History of reward function changes and lessons learned — critical for avoiding regressions
type: project
---

## score_delta in drop reward — REMOVED (caused score-win bias)

The drop reward used to be `1.0 + 0.1 * score_delta`, where `score_delta` was the change in the player's flood-fill contiguous-area score after a placement.

**Why this was wrong:** That score is literally the score-win metric. Rewarding its growth trained the agent to build large connected blobs — exactly the score-win condition — while ignoring path wins entirely. The terminal reward differential (path win bonus) was too sparse to overcome thousands of steps of dense area-growth shaping.

**How to apply:** Never use the flood-fill score (player.score) as a dense shaping signal. It directly encodes the score-win objective and will crowd out path-win behavior.

## Edge touch bonus — ADDED

A `+2.0` bonus fires the **first time** the acting player's cells touch each grid edge (left/right/top/bottom), tracked per episode via `self._touched_edges` (reset on `env.reset()`). Max possible bonus per player: `+8.0` over a full episode.

**Why:** Guides the agent toward reaching opposite edges (prerequisite for a path win) without rewarding repeated edge contact.

## Current terminal rewards (as of 2026-03-13)

| Outcome | Reward |
|---|---|
| Path win | +100.0 |
| Score win | +20.0 |
| Loss | −20.0 |

Path win is 5× more valuable than score win to make it the dominant objective.
