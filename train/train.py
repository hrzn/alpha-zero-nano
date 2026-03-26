"""Self-play and training loop for AlphaZero.

Optimizations:
  Opt 1 — Tree reuse: self_play() calls mcts.advance_root(action) after each
           move so the next search reuses the explored subtree.
  Opt 3 — Parallel self-play: parallel_self_play() runs multiple games
           simultaneously across CPU cores using multiprocessing.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from mcts.mcts import MCTS


def self_play(game, mcts: MCTS, max_moves=None, temp_threshold=None):
    """Play one game via MCTS+model and return training examples.

    Returns a list of (encoded_state, policy, outcome) tuples where:
    - encoded_state: (num_channels, rows, cols) float32 array from the acting player's perspective
    - policy: (action_size,) float32 array of MCTS visit-count probabilities
    - outcome: float in [-1, 1] from the acting player's perspective

    max_moves: if set, the game is terminated after this many moves.  Uses the
               model's value estimate of the final position (value bootstrapping)
               rather than declaring a flat draw — this provides learning signal
               even when games don't terminate naturally.
    temp_threshold: if set, use proportional sampling (temperature=1) for the
                    first temp_threshold moves, then play greedily (argmax).
    """
    mcts._root = None  # each game starts with a fresh tree (Opt 1: reuse is within a game)
    examples = []  # (encoded_state, policy, player)
    state = game.get_initial_state()
    player = 1
    move_count = 0

    while True:
        policy = mcts.search(state, player)
        encoded_state = game.encode_state(state, player)
        examples.append((encoded_state, policy, player))

        # Temperature annealing: sample proportionally early, argmax later
        if temp_threshold is not None and move_count >= temp_threshold:
            action = int(np.argmax(policy))
        else:
            action = np.random.choice(game.action_size, p=policy)
        mcts.advance_root(action)  # Opt 1: tree reuse — promote child to root
        state = game.update_state(state, action, player)
        move_count += 1

        value, terminated = game.get_value_and_terminated(state, action)
        if not terminated and max_moves is not None and move_count >= max_moves:
            # Value bootstrap: use model's evaluation instead of declaring a
            # flat draw.  Evaluate from the next-to-move player's perspective,
            # then negate to match the convention (value from last mover's POV).
            _, bootstrap_v = mcts._evaluate(state, game.get_opponent(player))
            value = -float(bootstrap_v)
            terminated = True

        if terminated:
            # Assign outcomes from each acting player's perspective.
            # value is from `player`'s (last mover's) POV; negate for opponent.
            training_examples = []
            for enc_state, pol, acting_player in examples:
                outcome = float(value) if acting_player == player else float(-value)
                training_examples.append((enc_state, pol, outcome))
            return training_examples

        player = game.get_opponent(player)


def _worker_self_play(task):
    """Top-level worker function for parallel_self_play (must be picklable).

    Reconstructs the game, model, and MCTS from serializable arguments so it
    can run in a spawned subprocess without inheriting shared state.
    """
    from model.model import ResNet
    from mcts.mcts import MCTS as _MCTS

    game_cls, state_dict_cpu, num_res_blocks, num_hidden, num_searches, c_puct, batch_size, dirichlet_alpha, dirichlet_epsilon, temp_threshold, max_moves = task
    game = game_cls()
    model = ResNet(game, num_res_blocks=num_res_blocks, num_hidden=num_hidden)
    model.load_state_dict(state_dict_cpu)
    model.eval()
    mcts = _MCTS(
        game,
        model=model,
        num_searches=num_searches,
        c_puct=c_puct,
        batch_size=batch_size,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    return self_play(game, mcts, max_moves=max_moves, temp_threshold=temp_threshold)


def parallel_self_play(game, mcts: MCTS, n_games, max_moves=None, temp_threshold=None, n_workers=4):
    """Play n_games in parallel using multiprocessing (Opt 3).

    Each worker receives the model weights (not the model object) and creates
    its own ResNet + MCTS instances, avoiding shared-state issues.

    Args:
        game: game instance (must be reconstructable via type(game)())
        mcts: MCTS instance with a model attached
        n_games: total number of self-play games to run
        max_moves: passed to self_play in each worker
        temp_threshold: passed to self_play in each worker
        n_workers: number of parallel worker processes

    Returns:
        Combined list of training examples from all n_games games.
    """
    import multiprocessing

    model = mcts.model
    if model is None:
        raise ValueError("parallel_self_play requires MCTS to have a model")

    state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    task = (
        type(game),
        state_dict_cpu,
        model.num_res_blocks,
        model.num_hidden,
        mcts.num_searches,
        mcts.c_puct,
        mcts.batch_size,
        mcts.dirichlet_alpha,
        mcts.dirichlet_epsilon,
        temp_threshold,
        max_moves,
    )
    tasks = [task] * n_games

    if n_workers == 1:
        return sum((_worker_self_play(t) for t in tasks), [])

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        results = pool.map(_worker_self_play, tasks)
    return sum(results, [])


def train_step(model, optimizer, batch):
    """Run one gradient update step on a batch of training examples.

    Args:
        model: ResNet instance
        optimizer: torch optimizer
        batch: (encoded_states, policies, outcomes) as numpy arrays

    Returns:
        loss value as a float
    """
    model.train()
    device = next(model.parameters()).device
    encoded_states, policies, outcomes = batch

    states_t = torch.tensor(encoded_states, dtype=torch.float32).to(device)
    policies_t = torch.tensor(policies, dtype=torch.float32).to(device)
    outcomes_t = torch.tensor(outcomes, dtype=torch.float32).to(device)

    policy_logits, values = model(states_t)

    policy_loss = F.cross_entropy(policy_logits, policies_t)
    value_loss = F.mse_loss(values.squeeze(-1), outcomes_t)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


class AlphaZero:
    def __init__(self, game, model, args):
        """
        args keys:
          num_searches      — MCTS simulations per move
          num_self_play_games — games per iteration
          num_epochs        — training epochs per iteration
          batch_size        — examples per gradient step
          lr                — learning rate
          max_moves         — (optional) draw after this many moves
        """
        self.game = game
        self.model = model
        self.args = args
        self.mcts = MCTS(game, model=model, num_searches=args["num_searches"])
        self.optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    def run(self, num_iterations):
        """Run the full AlphaZero training loop."""
        max_moves = self.args.get("max_moves")
        for _ in range(num_iterations):
            # Self-play: collect examples from all games this iteration
            examples = []
            for _ in range(self.args["num_self_play_games"]):
                examples += self_play(self.game, self.mcts, max_moves=max_moves)

            # Unpack into arrays
            encoded_states = np.array([e[0] for e in examples], dtype=np.float32)
            policies = np.array([e[1] for e in examples], dtype=np.float32)
            outcomes = np.array([e[2] for e in examples], dtype=np.float32)

            # Train for num_epochs, sampling a random batch each epoch
            self.model.train()
            for _ in range(self.args["num_epochs"]):
                batch_size = min(self.args["batch_size"], len(examples))
                idx = np.random.choice(len(examples), size=batch_size, replace=False)
                batch = (encoded_states[idx], policies[idx], outcomes[idx])
                train_step(self.model, self.optimizer, batch)

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
