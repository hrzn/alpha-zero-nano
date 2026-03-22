# AlphaZero Training Optimizations

Training is slow (~minutes per iteration for chess) because MCTS calls `model.predict()`
once per simulation, per move, per game. With 100 searches × 60 moves × 15 games ≈ 90,000
individual forward passes per iteration, each paying full tensor-creation and device-transfer
overhead.

This document tracks five speedups designed to make chess training feasible on an M1 MacBook Pro.

---

## Optimization Summary

| # | Name | Expected Speedup | Complexity | Status |
|---|------|-----------------|------------|--------|
| 1 | MCTS tree reuse | ~2× | Low | Implemented |
| 2 | Transposition table | ~1.5–3× | Low-medium | Removed (negative impact on chess; superseded by Opt 4 batch deduplication) |
| 3 | Parallel self-play | ~8× (M1 cores) | Medium | Implemented |
| 4 | Batched MCTS inference | ~10–30× | High | Implemented |
| 5 | MPS for training step | ~small | Trivial | Partially done |

---

## Opt 1 — MCTS Tree Reuse (~2×)

**Problem:** Each call to `mcts.search()` builds a brand-new tree from the current root,
discarding all work done in the previous move's search.

**Idea:** After a move is chosen in self-play, reuse the subtree rooted at the chosen
child as the root for the next search, instead of rebuilding from scratch. The subtree
already has visit counts and Q-values from the previous search.

**Changes:**
- `mcts/mcts.py`: `MCTS` stores `self._root: Node | None`. `search()` uses `_root` if set,
  otherwise creates a fresh root. `advance_root(action)` promotes the child at `action` to
  be the new root (detaches parent). If the child wasn't explored, `_root` is set to `None`.
- `train/train.py`: `self_play()` calls `mcts.advance_root(action)` after choosing each move.

**Why it helps:** Simulations that already explored the chosen subtree are not wasted.
For num_searches=100, typically 50–70 of those visits are in the chosen child's subtree.

---

## Opt 2 — Transposition Table (Removed)

**Was:** Cache `(position_hash, player) → (policy, value)` within each MCTS search call.

**Why removed:** Benchmarking showed a slight slowdown for chess (0.98×). Chess's high
branching factor (~35) means 100 simulations rarely revisit the same position, so cache
hits are scarce. Meanwhile, computing `hash(board.fen())` + dict lookup on every
`_evaluate()` call adds overhead that outweighs the savings.

**What remains:** `state_hash()` is kept on `TicTacToe` and `ChessGame` — it's a
one-liner with zero cost when unused, and Opt 4's batch deduplication step will use it
to identify duplicate leaves before the batched forward pass.

**Superseded by:** Opt 4 (batched MCTS inference) handles deduplication more
effectively at the batch level, before the expensive GPU/CPU forward call.

---

## Opt 3 — Parallel Self-Play (~8× on M1)

**Problem:** Self-play games are run sequentially. The M1 chip has 8+ cores that sit idle.

**Idea:** Each self-play game is independent — run N games simultaneously across CPU cores
using `multiprocessing`. The model's CPU inference is the bottleneck, and multiple processes
can run it in parallel.

**Changes:**
- `train/train.py`: Added `parallel_self_play(game, mcts, n_games, max_moves, n_workers)`.
  Uses `multiprocessing.Pool`. Each worker receives the game class, model weights (state_dict),
  and MCTS config; creates its own `ResNet` and `MCTS` instances; and returns examples.
- `model/model.py`: `ResNet` stores `num_res_blocks` and `num_hidden` as instance attributes
  so workers can reconstruct the model architecture.

**Why it helps:** 8 cores → 8 games running simultaneously. Real-world speedup is slightly
less than 8× due to process startup overhead and memory bandwidth limits.

**Usage:** Replace `self_play` loop in `AlphaZero.run` with `parallel_self_play`.

---

## Opt 4 — Batched MCTS Inference (~10–30×)

**Problem:** Even with Opt 3, each MCTS simulation within a game calls `model.predict()`
with batch_size=1. GPU/MPS utilization is extremely low.

**Idea:** Collect leaf nodes from `batch_size` simulations and evaluate them all in one
`model.forward()` call, then backpropagate all values.

**Approach:**
1. Run selection for `batch_size` simulations applying virtual loss at each step so
   different simulations explore different paths.
2. Collect all unique unexpanded non-terminal leaves into a batch.
3. Evaluate the batch in a single `model.forward()` call via `_evaluate_batch()`.
4. Expand all unique leaves, undo virtual loss, then backpropagate real values.

**Changes:**
- `mcts/mcts.py`: `Node` gets `apply_virtual_loss()` / `undo_virtual_loss()` methods.
  `MCTS.__init__` adds `batch_size=1` (default 1 = no change). New `_evaluate_batch()`
  method runs a single batched forward pass. New `_run_one()` extracts the sequential
  simulation body. New `_run_batch()` implements three-phase batched simulation.
  `search()` dispatches to sequential or batched path.
- `benchmark.py`: `--batch-size=N` flag runs batched timing rows alongside sequential ones.

**Usage:** Pass `batch_size=N` to `MCTS(...)`. Recommended N: 16–64.

**Known limitation:** Opt 3 (parallel self-play) + Opt 4 are not combined. Workers in
`parallel_self_play` use the default `batch_size=1`.

---

## Opt 5 — MPS for Training Step (~small, Partially done)

**Problem:** The training step (gradient update on a batch of examples) runs on CPU by
default, missing the M1 GPU (MPS).

**Idea:** Keep MCTS inference on CPU (avoids per-call overhead of MPS tensor transfers),
but move tensors to MPS for the batched `train_step`.

**Status:** The `train_step` function already infers the device from the model parameters.
Calling `model.to("mps")` before training is sufficient. The notebook already supports this.

**Note:** Training is not the bottleneck (self-play is), so this gives only a small overall
speedup. Implementing Opts 1–3 first is higher priority.

---

## Implementation Notes

- Opts 1 and 2 are fully transparent: existing code calling `mcts.search()` benefits
  automatically without any interface changes.
- Opt 3 requires explicitly calling `parallel_self_play` instead of the sequential loop.
- Opts 1–3 are additive and can all be active simultaneously.
- After each optimization, run `uv run pytest` to verify all tests still pass.
