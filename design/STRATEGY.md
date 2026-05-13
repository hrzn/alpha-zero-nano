# AlphaZero Nano — Implementation Strategy

## Overview

From-scratch AlphaZero implementation, starting with Tic-tac-toe, then scaling to chess. Training target: M1 MacBook Pro.

## Components

### 1. Alpha-MCTS (`mcts/mcts.py`)

Game-agnostic MCTS using PUCT selection (no random rollouts).

**Data structure:** `Node` class with:
- `visit_count`, `value_sum` → Q = value_sum / visit_count
- `prior` — probability from the neural network policy head
- `children` — dict mapping action → child Node
- `state` — board as numpy array

**Four phases per simulation:**
1. **Select** — traverse using PUCT: `Q(s,a) + c_puct * P(s,a) * sqrt(N_parent) / (1 + N(s,a))`
2. **Expand** — at a leaf, call the neural net → `(policy, value)`, create child nodes with priors (masked to valid moves)
3. **Backpropagate** — propagate value up the path, flipping sign at each level (alternating players)
4. **No rollout** — the neural network value estimate replaces random rollouts

**Output:** after N simulations, return policy vector proportional to root visit counts (with temperature parameter).

### 2. Neural Network (`model/model.py`) — PyTorch

PyTorch chosen over JAX for mature M1/MPS backend support.

**Architecture:** ResNet with policy + value heads.
- **Input encoding:** 3 channels (player 1 pieces, player 2 pieces, empty) — generalizes to chess
- **Body:** Parameterized ResNet (`num_res_blocks`, `num_hidden`) — small for TicTacToe (2 blocks, 64 filters), larger for chess (5-10 blocks, 128-256 filters)
- **Policy head:** conv → flatten → linear → `action_size` outputs, softmax
- **Value head:** conv → flatten → linear → tanh (output in [-1, 1])

### 3. Self-Play Training Loop (`train.py`)

1. **Self-play:** MCTS + current network plays games, collecting `(state, mcts_policy, outcome)` tuples
2. **Train:** Loss = cross_entropy(predicted_policy, mcts_policy) + MSE(predicted_value, outcome)
3. **Iterate:** Repeat for N iterations

## File Structure

```
tictactoe/
    tictactoe.py         # Game environment (done)
    test_tictactoe.py    # Tests (done)
mcts/
    mcts.py              # Game-agnostic MCTS with PUCT
model/
    model.py             # ResNet with policy + value heads
train.py                 # Self-play + training loop
az-tictactoe.ipynb       # Interactive play notebook (existing)
```

## Design Principles

- **Game-agnostic interfaces:** MCTS and model take a `game` object matching the TicTacToe interface, so the same code works for chess
- **Parameterized network size:** same ResNet class scales from TicTacToe (~10K params) to chess
- **M1-trainable:** keep chess networks small enough (5-10 res blocks, 128-256 filters) to train on M1 MacBook Pro via PyTorch MPS backend
