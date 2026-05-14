# alpha-zero-nano

A (somewhat) minimal, vibe-coded AlphaZero implementation for 2-player games. Pure self-play, no human data — just a neural network learning by playing itself, guided by MCTS.

Trains comfortably on a MacBook Pro (M1/MPS) for **Tic-tac-toe** (trivial) and **Connect 4** (a few hours). **Chess** is implemented too, but reaching non-trivial play would need more compute.

For Connect 4 there's a small web app that lets you play the trained agent in the browser, and visualizes both the MCTS visit distribution and the network's raw policy/value:

**▶ [Play it here](https://hrzn.github.io/alpha-zero-nano/)**

---

## What's inside

- **Game-agnostic MCTS** with PUCT selection (`mcts/`) — no rollouts; the value head replaces them.
- **ResNet policy/value network** (`model/`) — configurable depth/width, shared trunk + two heads for policy and value.
- **Pluggable games** (`tictactoe/`, `connect4/`, `chess_game/`) — each implements the same small interface.
- **Self-play training loop** (`train/`) with preset configurations from "tiny debug" to "full chess run".
- **Connect 4 web demo** (`web/`) — model exported to ONNX, runs entirely in the browser on the client side.

Not-so-minimal ingredients added for speed and stability:

- **MCTS tree reuse** across moves within a self-play game
- **Parallel self-play** across processes
- **Batched MCTS inference** — leaves are collected and evaluated in batches instead of one-by-one
- **Arena gating** — a new model only replaces the current champion if it wins a head-to-head match by a margin

---

## Quick start

The project uses [`uv`](https://github.com/astral-sh/uv) to manage Python deps.

### Train

```bash
# Connect 4 — small model, ~hours on M1, beats random fast and keeps improving
uv run python -m train.run_training --preset C4

# Chess — pick a preset
uv run python -m train.run_training --preset XS   # tiny, for pipeline sanity (~10s/iter)
uv run python -m train.run_training --preset S    # small, first signs of play by iter 20–30
uv run python -m train.run_training --preset M    # full run, hours+
```

Training auto-detects MPS / CUDA / CPU. Checkpoints land in `checkpoints/<preset>/` and resume automatically.

### Play against a trained Connect 4 model in the browser

```bash
# Export the trained champion to ONNX
uv run python tools/export_for_web.py

# Run the web app
cd web
npm install
npm run dev
```

### Train interactively

Jupyter notebooks are included if you'd rather poke around:

- `az-train-tictactoe.ipynb`
- `az-train-chess.ipynb`
- `az-play-connect4.ipynb`

---

## Project layout

```
mcts/           Game-agnostic MCTS (PUCT, tree reuse, batched inference)
model/          ResNet with policy + value heads
train/          Self-play loop, run_training.py with presets, arena gating
tictactoe/      Game implementation + tests
connect4/       Game implementation + tests
chess_game/     Game implementation + tests
tools/          export_for_web.py, etc.
web/            React + ONNX Runtime web demo for Connect 4
design/         Strategy and optimization notes
```

## Tests

```bash
uv run pytest
```
