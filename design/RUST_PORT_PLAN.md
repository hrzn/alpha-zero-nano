# Rust Chess Pipeline — Feasibility & Plan

## Context

C4 is working well in the Python pipeline. Chess is the next ambition, but the current setup never gets it past random play on the M1 Pro — not because the algorithm is wrong, but because **wall-clock per training iteration is too slow for the number of iterations chess needs**. The "S" preset benchmarks at ~119 s/iter (15 self-play games × 200 sims/move at batch=32). Reaching anything resembling amateur-level chess requires hundreds to low thousands of iterations; at 2 min/iter that's days to weeks, with no guarantee of convergence.

The question this doc answers: where does wall-clock actually go, and is it worth porting the hot path to a faster language for chess specifically?

## Diagnosis — where the time goes (real numbers)

From `benchmark_results.json` and code inspection of the chess "S" preset (5 res-blocks × 128 hidden, 17×8×8 input):

| Setting | Time per 200 sims | Per-sim |
|---|---|---|
| Unbatched (`batch_size=1`) | **719 ms** | 3.6 ms |
| Batched (`batch_size=32`) | **404 ms** | 2.0 ms |

**Unbatched breakdown per sim (~3.6 ms):**
- Actual NN forward compute: ~1.5 ms (42%)
- PyTorch Python glue (numpy↔tensor, `.to(device)`): ~0.3 ms (8%)
- MCTS tree traversal + PUCT + dict iteration: ~0.3 ms (8%)
- `Node.__init__` allocations during expand: ~0.2 ms (6%)
- python-chess (mostly C++ already) + wrapper glue: ~0.2 ms (6%)
- `encode_state` numpy loop: ~0.1 ms (3%)
- The rest is dict/list overhead and miscellaneous

**Batched breakdown is the regime that actually matters.** With `batch_size=32`, the NN per-sim amortizes from 1.5 ms → ~0.05 ms. Total per sim drops only to 2.0 ms — meaning ~1.95 ms per sim is now **not** NN compute. So in production-batched mode, **NN actual compute is ~3% of MCTS wall-clock; the rest (~97%) is Python overhead**. Batching is always on, so this is the relevant breakdown.

Unbatched analyses can mislead because they make NN look dominant when it isn't in production.

## Is Rust worth it?

Yes — for chess specifically, on this hardware. The expected speedup composes from three independent levers:

1. **Native MCTS tree** — Vec-backed arena of nodes with integer parent/child indices instead of Python objects-in-dicts. Tree traversal goes from ~hundreds of µs to single µs. Estimated **5–10× on the MCTS portion**.
2. **Better batching + threading** — Rust has no GIL. We can run multiple self-play games on a single shared NN inference batcher, instead of multiprocessing (which spawns processes, copies model weights, and pays IPC). Estimated **4–6×** on top of #1 on an 8-core M1 Pro.
3. **Tighter NN inference call** — `tch` (libtorch bindings) or `ort` (ONNX Runtime) eliminate the numpy↔tensor↔.to(device) round-trip per call. Per-call overhead falls from ~0.3 ms to ~tens of µs. Modest contribution when batching is on, but it stacks.

Compounded, **30–80× total speedup** is realistic for self-play wall-clock. That's the regime change from "chess is hopeless" to "chess training is an overnight run."

Cheaper-first alternatives (`torch.compile`, numba on `encode_state`, smarter PUCT loop) are real but at best ~2–3× combined. Worth doing eventually but **won't make chess feasible** on this hardware on their own.

## Architectural shape (recommended)

**Hybrid: Rust for the hot loop, Python for training.** Hyper-specialized for chess.

```
┌──────────────────────────────────────────────────┐
│  Python (existing)                               │
│  - Model definition (PyTorch ResNet)             │
│  - Gradient step / replay buffer (train.py)      │
│  - Checkpoint I/O, eval orchestration            │
└──────┬──────────────────────────────────┬────────┘
       │ exports ONNX                     ▲
       │ + reads training data            │ writes self-play data
       ▼                                  │
┌──────────────────────────────────────────────────┐
│  Rust (new — single binary)                      │
│  - shakmaty: legal moves, terminal detection     │
│  - MCTS: arena-backed tree, PUCT, virtual loss   │
│  - NN inference: ort or tch, batched on leaves   │
│  - Parallel self-play: N threads, shared batcher │
│  - Writes (state, policy_target, value) shards   │
└──────────────────────────────────────────────────┘
```

Critical: the Rust binary **does not train**. It only does self-play data generation. Training stays in the existing Python code, reading the data shards the Rust binary produces. Same iteration loop, just way faster.

This keeps the most painful part (autograd, optimizers, gradient updates) in PyTorch where it's well-tested, and replaces only the bottleneck.

## Phasing

### Phase 0 — Decide & set up (1 day)
Repo layout: new `rust/` directory next to existing `chess_game/` etc. Cargo workspace. The choice of `tch` vs `ort` decides early — likely `ort` for portability (no libtorch system dep) and smaller deploy artifact. Verify ONNX export of the existing chess ResNet works (we already proved this for C4 in the web demo).

### Phase 1 — Move generation + state encoding in Rust (3–4 days)
Use the `shakmaty` crate (de-facto standard, very fast). Implement the 17-channel encoder that mirrors `chess_game/chess_game.py:107-157` exactly — same channel ordering, same Black-side flip via `sq ^ 56`. Action encoding (`from_sq * 64 + to_sq`, 4096 actions) mirrors `chess_game.py:84-105`. Add a Rust-side test harness that loads positions from FEN and asserts encoded tensors byte-for-byte match the Python encoding. This is the foundation; getting it wrong invalidates everything downstream.

### Phase 2 — Single-threaded MCTS in Rust (3–5 days)
Arena-backed tree (`Vec<Node>`, indices for parent/children), PUCT scoring, virtual-loss for batched eval, root reuse. Mirror the algorithm in `mcts/mcts.py` exactly. Numerical agreement test: same root state, same NN outputs, same MCTS policy distribution (modulo float ties). Mirror the test pattern we used for the web port (`web/tools/agreement_test.ts`) — Python produces a reference, Rust runs it, max-abs-diff under tolerance.

### Phase 3 — NN inference + batched eval (2–3 days)
Wire `ort` (or `tch`) to load the ONNX-exported chess model. Implement leaf-batching: collect up to `batch_size` leaves with virtual loss applied, run one forward pass, backpropagate the values. The hand-rolled forward pass in `web/src/model.ts` is a reference for the operation sequence but we're using a real runtime here, not reimplementing it. The agreement test from Phase 2 now uses Rust NN inference instead of stubbed values.

### Phase 4 — Parallel self-play + data shards (3–4 days)
Spawn N self-play threads sharing one inference batcher. Each thread plays games independently, accumulating examples. Periodically flush to a binary shard (msgpack or `.npy` files): `[encoded_state, policy_target, value]` rows. Python's `train.py` already takes numpy arrays — minimal change on that side, just point it at the shard files instead of in-Python self-play.

### Phase 5 — Integration & training loop (3–4 days)
A small Python driver that orchestrates: export current model to ONNX → run `cargo run --release --bin selfplay` for K games → train for some epochs on the resulting shards → save checkpoint → repeat. Same iteration shape as today, but the self-play phase is the Rust binary instead of `parallel_self_play`. Add eval and arena gating on top (we already designed those for C4).

### Phase 6 — Validate end-to-end (few days, mostly waiting)
Run a 50-iter dry run on chess "S" preset, compare to the existing Python pipeline's eval numbers. We expect: same loss curve shape, dramatically lower wall-clock per iter. If the loss curve diverges materially from the Python version, there's a port bug — likely in encoding or sign convention — and we go bug-hunt with the agreement tests from Phase 2.

## Effort estimate

- ~3 weeks of focused work for someone comfortable with Rust + ML tooling.
- ~4–5 weeks if Rust is unfamiliar.
- Add 1 week for the inevitable port-bug hunt before training agrees with Python.

The phasing above is designed to fail fast — Phase 1 alone catches encoding mismatches before any MCTS work; Phase 2 catches MCTS bugs before any NN; Phase 3 catches inference bugs before any parallelism. Each phase has a numerical agreement test against Python as its acceptance criterion.

## Risks & caveats

- **Two codebases to maintain.** The Python pipeline doesn't go away; chess training is now bilingual. Acceptable for a focused chess push, painful long-term.
- **ML tooling in Rust is less mature.** `ort` and `tch` both work, but bug reports take longer to resolve. Pin versions, write tests.
- **Tree-search bugs are easier to introduce in Rust** (lifetimes, borrowing through a tree). The arena + index-based design sidesteps most of the pain.
- **Speedup estimates have wide error bars.** 30× is the floor I'd bet on; 80× the ceiling. Even at the floor, chess becomes feasible. If we somehow only got 5×, this project would be a waste — but the breakdown above shows the Python tax is large enough that 5× is essentially impossible to *miss*.

## Cheap-first alternatives (if deferring Rust)

In priority order:
1. `model.forward = torch.compile(model.forward, mode="reduce-overhead")` — 10–20% on the NN, free.
2. Replace `encode_state` numpy loops with a numba `@njit` version — 5–10% per sim.
3. Replace `Node.children` dict with a fixed-size array (action_size is known up front) — 5–10% on PUCT traversal.

Combined, these are maybe 2–3×. Not enough to crack chess, but worth doing regardless because they help C4 too.

## Files referenced

- `mcts/mcts.py` — algorithm to port (Node + MCTS classes)
- `chess_game/chess_game.py:84-157` — action encoding and 17-channel state encoding to replicate
- `model/model.py` — ResNet definition; ONNX export target
- `train/train.py:148-177` — `train_step`, stays in Python unchanged
- `train/run_training.py` — orchestration; the eventual Phase-5 driver replaces the self-play call here
- `benchmark_results.json` — the numbers backing the diagnosis section

## Verification

End-to-end success looks like: chess "S" preset trains for 200 iters overnight (~3–6 hours wall-clock vs current ~7 hours for 200 iters, but with vastly higher per-iter sim count enabled, so each iter is much more informative). Concretely, the win rate vs random must clearly exceed 50% within 50 iters — something the existing Python pipeline never achieved.

Stretch verification: the trained model holds its own (≥80% non-loss rate) against Stockfish at skill level 0 within a single training run. That's the "fun for a bad amateur" bar.
