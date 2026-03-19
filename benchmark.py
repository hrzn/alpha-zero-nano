"""AlphaZero training benchmark.

Measures key performance metrics and writes results to benchmark_results.json
so successive optimization runs can be compared.

Usage:
    # Baseline (no optimizations):
    uv run python benchmark.py baseline --no-tree-reuse --no-cache

    # With all optimizations active:
    uv run python benchmark.py opts-1-2-3

    # Show speedup table comparing the last two recorded runs:
    uv run python benchmark.py --compare

Flags:
    --no-tree-reuse   Disable Opt 1: force a fresh MCTS tree every move
    --no-cache        Disable Opt 2: bypass the transposition table
    --compare         Print a speedup comparison of the two most recent runs
                      (no new benchmark is run)
"""

import json
import sys
import time
from datetime import datetime

import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_PREDICT_WARMUP = 5
NUM_PREDICT_TRIALS = 20

MCTS_SIM_COUNTS = [50, 100, 200]
MCTS_TRIALS = 3  # average over this many searches

SELF_PLAY_MAX_MOVES = 30
NUM_SELF_PLAY_GAMES_EST = 15  # for iteration-time extrapolation

PARALLEL_N_GAMES = 4          # games to run in parallel benchmark
PARALLEL_N_WORKERS = 4        # worker processes

MODEL_NUM_RES_BLOCKS = 5
MODEL_NUM_HIDDEN = 128

OUTPUT_FILE = "benchmark_results.json"
# ─────────────────────────────────────────────────────────────────────────────


def parse_args():
    """Return (label, no_tree_reuse, no_cache, compare_only) from sys.argv."""
    argv = sys.argv[1:]
    no_tree_reuse = "--no-tree-reuse" in argv
    no_cache = "--no-cache" in argv
    compare_only = "--compare" in argv
    label_parts = [a for a in argv if not a.startswith("--")]
    label = label_parts[0] if label_parts else ("baseline" if no_tree_reuse or no_cache else "optimized")
    return label, no_tree_reuse, no_cache, compare_only


def apply_patches(no_tree_reuse, no_cache):
    """Monkey-patch MCTS to disable selected optimizations.

    This lets us measure a fair baseline without touching production code.
    Patches are applied at the class level before any MCTS instances are created.
    """
    from mcts.mcts import MCTS as _MCTS

    if no_cache:
        # Replace _evaluate with a version that skips the transposition table
        def _evaluate_no_cache(self, state, player):
            if self.model is None:
                return np.ones(self.game.action_size) / self.game.action_size, 0.0
            return self.model.predict(state, player)
        _MCTS._evaluate = _evaluate_no_cache

    if no_tree_reuse:
        # Replace advance_root with a no-op so every search starts from scratch
        def _advance_root_noop(self, action):
            self._root = None
        _MCTS.advance_root = _advance_root_noop


# ── Benchmark functions ───────────────────────────────────────────────────────

def bench_model_predict(model, game):
    """Time a single model.predict() call (batch=1)."""
    state = game.get_initial_state()
    for _ in range(NUM_PREDICT_WARMUP):
        model.predict(state, 1)

    t0 = time.perf_counter()
    for _ in range(NUM_PREDICT_TRIALS):
        model.predict(state, 1)
    return (time.perf_counter() - t0) / NUM_PREDICT_TRIALS * 1000  # ms


def bench_mcts_search(model, game, num_sims):
    """Time one MCTS search (num_sims simulations) from the start position.

    Note: tree reuse has no effect here because _root is reset between trials.
    The transposition table does help within a single search call.
    """
    from mcts.mcts import MCTS
    mcts = MCTS(game, model=model, num_searches=num_sims)
    state = game.get_initial_state()

    times = []
    for _ in range(MCTS_TRIALS):
        mcts._root = None  # fair comparison: each trial starts fresh
        t0 = time.perf_counter()
        mcts.search(state, 1)
        times.append(time.perf_counter() - t0)

    return (sum(times) / len(times)) * 1000  # ms per search


def bench_self_play_game(model, game):
    """Time one complete self-play game (sequential, capped at SELF_PLAY_MAX_MOVES).

    Both tree reuse (Opt 1) and the transposition table (Opt 2) affect this.
    """
    from mcts.mcts import MCTS
    from train.train import self_play
    mcts = MCTS(game, model=model, num_searches=100)
    t0 = time.perf_counter()
    self_play(game, mcts, max_moves=SELF_PLAY_MAX_MOVES)
    return time.perf_counter() - t0  # seconds


def bench_parallel_self_play(model, game, n_workers):
    """Time PARALLEL_N_GAMES games split across n_workers processes.

    Returns wall-clock seconds per game (total / n_games).
    """
    from mcts.mcts import MCTS
    from train.train import parallel_self_play
    mcts = MCTS(game, model=model, num_searches=100)
    t0 = time.perf_counter()
    parallel_self_play(game, mcts, n_games=PARALLEL_N_GAMES,
                       max_moves=SELF_PLAY_MAX_MOVES, n_workers=n_workers)
    return (time.perf_counter() - t0) / PARALLEL_N_GAMES  # s per game


# ── Output helpers ────────────────────────────────────────────────────────────

def _fmt(val, unit):
    if unit == "ms":
        return f"{val:.1f} ms"
    return f"{val:.1f} s"


def print_single(metrics, label):
    w = 38
    print()
    print("=" * 60)
    print(f"  AlphaZero Benchmark — {label}")
    print("=" * 60)
    print(f"  {'model.predict (batch=1)':<{w}} {metrics['predict_ms']:.1f} ms")
    for n in MCTS_SIM_COUNTS:
        key = f"mcts_search_{n}_sims_ms"
        if key in metrics:
            print(f"  {'MCTS search (' + str(n) + ' sims)':<{w}} {metrics[key]:.0f} ms/move")
    print(f"  {'Self-play game (' + str(SELF_PLAY_MAX_MOVES) + ' moves, 1 worker)':<{w}} {metrics['self_play_game_s']:.1f} s")
    if "parallel_1w_game_s" in metrics:
        print(f"  {'Parallel self-play (1 worker/game)':<{w}} {metrics['parallel_1w_game_s']:.1f} s/game")
    if "parallel_Nw_game_s" in metrics:
        print(f"  {'Parallel self-play (' + str(PARALLEL_N_WORKERS) + ' workers/game)':<{w}} {metrics['parallel_Nw_game_s']:.1f} s/game")
    est = metrics["est_iteration_s"]
    print(f"  {'Est. iteration (' + str(NUM_SELF_PLAY_GAMES_EST) + ' games)':<{w}} {est:.0f} s  (~{est/60:.1f} min)")
    print("=" * 60)
    print()


def print_comparison(history):
    """Print a speedup table comparing the two most recent benchmark runs."""
    if len(history) < 2:
        print("Need at least 2 recorded runs to compare. Run the benchmark twice.")
        return

    r1, r2 = history[-2], history[-1]
    m1, m2 = r1["metrics"], r2["metrics"]
    l1, l2 = r1["label"], r2["label"]

    rows = [
        ("predict_ms",             "model.predict (ms)"),
        ("mcts_search_50_sims_ms", "MCTS  50 sims (ms/move)"),
        ("mcts_search_100_sims_ms","MCTS 100 sims (ms/move)"),
        ("mcts_search_200_sims_ms","MCTS 200 sims (ms/move)"),
        ("self_play_game_s",       "self-play game, sequential (s)"),
        ("parallel_1w_game_s",     "parallel self-play, 1 worker (s/game)"),
        ("parallel_Nw_game_s",     f"parallel self-play, {PARALLEL_N_WORKERS} workers (s/game)"),
        ("est_iteration_s",        "est. iteration (s)"),
    ]

    w = 34
    print()
    print("=" * 72)
    print(f"  Speedup comparison:  [{l1}]  →  [{l2}]")
    print("=" * 72)
    print(f"  {'Metric':<{w}} {'[' + l1 + ']':>12} {'[' + l2 + ']':>12} {'speedup':>8}")
    print("  " + "-" * 68)
    for key, name in rows:
        if key not in m1 or key not in m2:
            continue
        v1, v2 = m1[key], m2[key]
        if v2 > 0:
            speedup = f"{v1/v2:.2f}×"
        else:
            speedup = "—"
        print(f"  {name:<{w}} {v1:>12.1f} {v2:>12.1f} {speedup:>8}")
    print("=" * 72)
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    label, no_tree_reuse, no_cache, compare_only = parse_args()

    # Load existing history
    try:
        with open(OUTPUT_FILE) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    if compare_only:
        print_comparison(history)
        return

    # Apply patches before importing MCTS-dependent modules
    apply_patches(no_tree_reuse, no_cache)

    flags = []
    if no_tree_reuse:
        flags.append("no-tree-reuse")
    if no_cache:
        flags.append("no-cache")
    flag_str = f"  flags: {', '.join(flags)}" if flags else "  flags: (all optimizations active)"

    print(f"Building model (res_blocks={MODEL_NUM_RES_BLOCKS}, hidden={MODEL_NUM_HIDDEN})…")
    print(flag_str)

    from chess_game.chess_game import ChessGame
    from model.model import ResNet

    game = ChessGame()
    model = ResNet(game, num_res_blocks=MODEL_NUM_RES_BLOCKS, num_hidden=MODEL_NUM_HIDDEN)
    model.eval()

    metrics = {}

    print("Benchmarking model.predict…")
    metrics["predict_ms"] = bench_model_predict(model, game)

    for n in MCTS_SIM_COUNTS:
        print(f"Benchmarking MCTS search ({n} sims)…")
        metrics[f"mcts_search_{n}_sims_ms"] = bench_mcts_search(model, game, n)

    print(f"Benchmarking sequential self-play game (max_moves={SELF_PLAY_MAX_MOVES})…")
    metrics["self_play_game_s"] = bench_self_play_game(model, game)
    metrics["est_iteration_s"] = metrics["self_play_game_s"] * NUM_SELF_PLAY_GAMES_EST

    print(f"Benchmarking parallel self-play (1 worker, {PARALLEL_N_GAMES} games)…")
    metrics["parallel_1w_game_s"] = bench_parallel_self_play(model, game, n_workers=1)

    print(f"Benchmarking parallel self-play ({PARALLEL_N_WORKERS} workers, {PARALLEL_N_GAMES} games)…")
    metrics["parallel_Nw_game_s"] = bench_parallel_self_play(model, game, n_workers=PARALLEL_N_WORKERS)

    print_single(metrics, label)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "label": label,
        "flags": {"no_tree_reuse": no_tree_reuse, "no_cache": no_cache},
        "config": {
            "num_res_blocks": MODEL_NUM_RES_BLOCKS,
            "num_hidden": MODEL_NUM_HIDDEN,
            "self_play_max_moves": SELF_PLAY_MAX_MOVES,
            "num_self_play_games_est": NUM_SELF_PLAY_GAMES_EST,
            "mcts_sim_counts": MCTS_SIM_COUNTS,
            "parallel_n_games": PARALLEL_N_GAMES,
            "parallel_n_workers": PARALLEL_N_WORKERS,
        },
        "metrics": metrics,
    }

    history.append(record)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Results appended to {OUTPUT_FILE}")

    if len(history) >= 2:
        print_comparison(history)


if __name__ == "__main__":
    main()
