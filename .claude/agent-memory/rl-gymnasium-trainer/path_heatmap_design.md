---
name: path_heatmap_design
description: Design discussion for adding a path-win heatmap as a second grid observation channel — not yet implemented
type: project
---

## Path-win heatmap as second grid channel (DISCUSSED, not yet implemented)

**Core idea**: Instead of a scalar distance, give the agent a spatial map of which empty cells
it should fill to win by path. Exposed as a second channel in the grid observation so the CNN
can learn "drop when ghost overlaps highlighted cells."

## Algorithm: Dijkstra on 9×9 grid

- Acting player's cells → cost 0 (free to traverse, already placed)
- Empty cells → cost 1 (need to be filled)
- Opponent cells → walls / ∞ (cannot cross — prevents counting impossible routes)

Run from left column → right column (horizontal axis) and top row → bottom row (vertical axis).

**Mark ALL cells on ANY shortest path** — not just one:

```
cell is on shortest path if: dist_from_source[cell] + dist_from_target[cell] == total_min_distance
```

This gives a stable corridor. A single path would flicker whenever tied alternatives exist.

## Observation design

Grid shape: `(9,9,1)` → `(9,9,2)`
- Channel 0: board state (0.0=empty, 0.5=p1, 1.0=p2) — unchanged
- Channel 1: path heatmap for the acting player (1.0=on optimal path, 0.0=not)

The CNN preserves spatial structure, which a flattened scalar cannot.

## Edge cases

| Situation | Behavior |
|---|---|
| Opponent fully blocks the axis | All zeros in channel 1 — agent learns this axis is lost |
| No pieces placed yet | Wide corridor — acts as gentle nudge toward useful area |
| Path win achieved | All zeros (distance = 0, no empty cells needed) |

## Open design questions (not yet decided)

1. **Axes**: Union H+V into one channel, OR two separate channels `(9,9,3)`, OR best axis only?
   - Union: simpler, but agent can't distinguish which axis it's closer to winning on
   - Two channels: richer signal, larger observation
   - Best axis: focuses agent on its strongest option

2. **Perspective**: Own heatmap only, OR also show opponent's heatmap?
   - Own only: simpler, focused on offense
   - Both players: agent can see opponent's optimal path and potentially block

## Critical warnings

- **DO NOT use gap-delta as a reward signal.** Rewarding "gap decreased by N after drop" would
  re-introduce area-filling bias — same root cause as the `score_delta` mistake. Observation only.
- The existing edge-touch bonus reward (`+2.0` per first edge touch) remains valid alongside this.

## Why scalars are insufficient

- `dist(any cell → left edge)` + `dist(any cell → right edge)` counted independently allows
  two disconnected blobs near both edges to score low distance — no connectivity enforced.
- Dijkstra through the player's own cells enforces connectivity: the path must route through
  existing placed cells, not jump between isolated groups.
