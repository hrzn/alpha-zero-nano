"""AlphaZero training benchmark.

Measures key performance metrics and writes results to benchmark_results.json
so successive optimization runs can be compared.

Usage:
    uv run python benchmark.py
    uv run python benchmark.py --label "after-tree-reuse"
"""

import json
import sys
import time
from datetime import datetime

import numpy as np

from chess_game.chess_game import ChessGame
from mcts.mcts import MCTS
from model.model import ResNet
from train.train import self_play

# ── Configuration ─────────────────────────────────────────────────────────────
NUM_PREDICT_WARMUP = 5
NUM_PREDICT_TRIALS = 20

MCTS_SIM_COUNTS = [50, 100, 200]
MCTS_TRIALS = 3  # average over this many moves

SELF_PLAY_MAX_MOVES = 30
NUM_SELF_PLAY_GAMES_EST = 15  # for extrapolation only

MODEL_NUM_RES_BLOCKS = 5
MODEL_NUM_HIDDEN = 128

OUTPUT_FILE = "benchmark_results.json"


def bench_model_predict(model, game):
    """Time a single model.predict() call (batch=1)."""
    state = game.get_initial_state()

    # Warmup
    for _ in range(NUM_PREDICT_WARMUP):
        model.predict(state, 1)

    start = time.perf_counter()
    for _ in range(NUM_PREDICT_TRIALS):
        model.predict(state, 1)
    elapsed = time.perf_counter() - start

    return (elapsed / NUM_PREDICT_TRIALS) * 1000  # ms per call


def bench_mcts_search(model, game, num_sims):
    """Time one MCTS search (num_sims simulations) from the start position."""
    mcts = MCTS(game, model=model, num_searches=num_sims)
    state = game.get_initial_state()

    times = []
    for _ in range(MCTS_TRIALS):
        mcts._root = None  # reset tree reuse between trials
        t0 = time.perf_counter()
        mcts.search(state, 1)
        times.append(time.perf_counter() - t0)

    return (sum(times) / len(times)) * 1000  # ms per search


def bench_self_play_game(model, game):
    """Time one complete self-play game (capped at SELF_PLAY_MAX_MOVES moves)."""
    mcts = MCTS(game, model=model, num_searches=100)
    t0 = time.perf_counter()
    self_play(game, mcts, max_moves=SELF_PLAY_MAX_MOVES)
    return time.perf_counter() - t0  # seconds


def print_table(metrics):
    print()
    print("=" * 55)
    print("  AlphaZero Benchmark Results")
    print("=" * 55)
    print(f"  {'model.predict (batch=1)':<35} {metrics['predict_ms']:.1f} ms")
    for n in MCTS_SIM_COUNTS:
        key = f"mcts_search_{n}_sims_ms"
        print(f"  {'MCTS search (' + str(n) + ' sims)':<35} {metrics[key]:.0f} ms/move")
    print(f"  {'Self-play game (' + str(SELF_PLAY_MAX_MOVES) + ' moves)':<35} {metrics['self_play_game_s']:.1f} s")
    est = metrics['est_iteration_s']
    print(f"  {'Est. iteration (' + str(NUM_SELF_PLAY_GAMES_EST) + ' games)':<35} {est:.0f} s  (~{est/60:.1f} min)")
    print("=" * 55)
    print()


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    print(f"Building model (res_blocks={MODEL_NUM_RES_BLOCKS}, hidden={MODEL_NUM_HIDDEN})…")
    game = ChessGame()
    model = ResNet(game, num_res_blocks=MODEL_NUM_RES_BLOCKS, num_hidden=MODEL_NUM_HIDDEN)
    model.eval()

    metrics = {}

    print("Benchmarking model.predict…")
    metrics["predict_ms"] = bench_model_predict(model, game)

    for n in MCTS_SIM_COUNTS:
        print(f"Benchmarking MCTS search ({n} sims)…")
        metrics[f"mcts_search_{n}_sims_ms"] = bench_mcts_search(model, game, n)

    print(f"Benchmarking self-play game (max_moves={SELF_PLAY_MAX_MOVES})…")
    metrics["self_play_game_s"] = bench_self_play_game(model, game)
    metrics["est_iteration_s"] = metrics["self_play_game_s"] * NUM_SELF_PLAY_GAMES_EST

    print_table(metrics)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "label": label,
        "config": {
            "num_res_blocks": MODEL_NUM_RES_BLOCKS,
            "num_hidden": MODEL_NUM_HIDDEN,
            "self_play_max_moves": SELF_PLAY_MAX_MOVES,
            "num_self_play_games_est": NUM_SELF_PLAY_GAMES_EST,
            "mcts_sim_counts": MCTS_SIM_COUNTS,
        },
        "metrics": metrics,
    }

    try:
        with open(OUTPUT_FILE) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(record)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Results appended to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
