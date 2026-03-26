"""AlphaZero-style Monte Carlo Tree Search with PUCT selection.

Optimizations implemented:
  Opt 1 — Tree reuse: advance_root() promotes a child to root so the next
           search() call reuses its visit counts instead of starting fresh.
  Opt 4 — Batched inference: _run_batch() collects batch_size leaf nodes and
           evaluates them in a single model.forward() call with virtual loss.
"""

import math

import numpy as np
import torch

_VIRTUAL_LOSS = 1


class Node:
    def __init__(self, game, state, player, prior=0.0, parent=None, action_taken=None):
        self.game = game
        self.state = state
        self.player = player
        self.prior = prior
        self.parent = parent
        self.action_taken = action_taken

        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def select(self, c_puct=1.0):
        """Select the child with the highest PUCT score."""
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            score = child.puct_score(c_puct)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def puct_score(self, c_puct):
        # Negate because value_sum is from this node's player perspective,
        # but the parent (who is selecting) is the opponent
        q_value = 0.0 if self.visit_count == 0 else -self.value_sum / self.visit_count
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)
        return q_value + exploration

    def expand(self, policy):
        """Create child nodes for each valid move, using policy as priors."""
        valid_moves = self.game.get_valid_moves(self.state)
        # Mask and renormalize policy to valid moves
        policy = policy * valid_moves
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum

        for action in range(self.game.action_size):
            if valid_moves[action] == 1:
                child_state = self.state.copy()
                child_state = self.game.update_state(child_state, action, self.player)
                child = Node(
                    game=self.game,
                    state=child_state,
                    player=self.game.get_opponent(self.player),
                    prior=policy[action],
                    parent=self,
                    action_taken=action,
                )
                self.children[action] = child

    def backpropagate(self, value):
        """Propagate value up to root, flipping sign at each level."""
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)

    def apply_virtual_loss(self, vl=_VIRTUAL_LOSS):
        self.visit_count += vl
        self.value_sum += vl          # positive → parent sees Q = -vl/vl = -1.0
        if self.parent is not None:
            self.parent.apply_virtual_loss(vl)

    def undo_virtual_loss(self, vl=_VIRTUAL_LOSS):
        self.visit_count -= vl
        self.value_sum -= vl
        if self.parent is not None:
            self.parent.undo_virtual_loss(vl)


class MCTS:
    def __init__(
        self,
        game,
        model=None,
        num_searches=100,
        c_puct=1.0,
        batch_size=1,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.25,
    ):
        self.game = game
        self.model = model
        self.num_searches = num_searches
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self._root = None  # Opt 1: tree reuse

    def _apply_dirichlet_noise(self, root):
        """Mix Dirichlet noise into the root children's priors (exploration during self-play).

        Called at the start of every search() so each move sees fresh noise.
        No-op when dirichlet_alpha == 0 (default) or root has no children.
        """
        if self.dirichlet_alpha <= 0 or not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        eps = self.dirichlet_epsilon
        for action, eta in zip(actions, noise):
            child = root.children[action]
            child.prior = (1 - eps) * child.prior + eps * eta

    def _evaluate(self, state, player):
        """Get policy and value from the model, or use defaults if no model."""
        if self.model is None:
            return np.ones(self.game.action_size) / self.game.action_size, 0.0
        return self.model.predict(state, player)

    def _evaluate_batch(self, leaves):
        """Evaluate a list of (state, player) tuples in a single forward pass.

        Returns a list of (policy_np, value_float) in the same order as leaves.
        """
        if self.model is None:
            uniform = np.ones(self.game.action_size) / self.game.action_size
            return [(uniform, 0.0) for _ in leaves]
        self.model.eval()
        device = next(self.model.parameters()).device
        x = torch.tensor(
            np.stack([self.game.encode_state(s, p) for s, p in leaves]),
            dtype=torch.float32,
        ).to(device)
        with torch.no_grad():
            policy_logits, values = self.model(x)
        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values_np = values.cpu().numpy()
        return [(policies[i], float(values_np[i])) for i in range(len(leaves))]

    def advance_root(self, action):
        """Opt 1: promote the child at action to be the new root.

        Call this after each move in self-play so the next search() reuses
        the subtree already explored under that child. If the action was
        not in the tree, _root is reset to None (fresh tree on next search).
        """
        if self._root is not None and action in self._root.children:
            self._root = self._root.children[action]
            self._root.parent = None
        else:
            self._root = None

    def _run_one(self, root):
        """Run a single MCTS simulation from root (sequential path)."""
        node = root

        # Select: traverse tree until we find an unexpanded node
        while node.is_expanded():
            _, node = node.select(self.c_puct)

        # Check if this node is terminal
        value, terminated = self.game.get_value_and_terminated(
            node.state, node.action_taken
        )

        if terminated:
            # Value is from the perspective of the player who just moved
            # (the parent's player), but we need it from this node's
            # perspective for backpropagation
            value = -value
        else:
            # Expand and evaluate
            policy, value = self._evaluate(node.state, node.player)
            node.expand(policy)

        # Backpropagate
        node.backpropagate(value)

    def _run_batch(self, root, batch_size):
        """Run batch_size simulations with virtual loss and batched evaluation.

        Phase 1: Select leaf nodes for each sim, applying virtual loss.
        Phase 2: Batch-evaluate unique unexpanded non-terminal leaves.
        Phase 3: Undo virtual loss and backpropagate real values.
        """
        # Phase 1 — Selection with virtual loss
        sim_results = []  # list of (leaf_node, terminated, value)
        for _ in range(batch_size):
            node = root
            while node.is_expanded():
                _, node = node.select(self.c_puct)
            # Apply VL from leaf; the recursive call propagates it up to root
            node.apply_virtual_loss()

            value, terminated = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            if terminated:
                value = -value

            sim_results.append((node, terminated, value))

        # Phase 2 — Batch evaluation of unique unexpanded non-terminal leaves
        unique_nodes = {}  # id(node) → node
        for node, terminated, _ in sim_results:
            if not terminated and not node.is_expanded():
                unique_nodes[id(node)] = node

        node_eval_map = {}
        if unique_nodes:
            node_list = list(unique_nodes.values())
            leaves = [(n.state, n.player) for n in node_list]
            eval_results = self._evaluate_batch(leaves)
            node_eval_map = {
                id(node_list[i]): eval_results[i] for i in range(len(node_list))
            }

            for node_id, node in unique_nodes.items():
                if not node.is_expanded():
                    policy, _ = node_eval_map[node_id]
                    node.expand(policy)

        # Phase 3 — Undo virtual loss + backpropagate
        for node, terminated, value in sim_results:
            node.undo_virtual_loss()
            if not terminated and id(node) in node_eval_map:
                _, value = node_eval_map[id(node)]
            node.backpropagate(value)

    def search(self, state, player):
        """Run MCTS from the given state and return a policy vector."""
        # Opt 1: reuse existing root if available (set via advance_root),
        # otherwise build a fresh root and expand it.
        if self._root is None:
            root = Node(self.game, state.copy(), player)
            policy, _ = self._evaluate(state, player)
            root.expand(policy)
            self._root = root
        else:
            root = self._root

        self._apply_dirichlet_noise(root)

        if self.batch_size == 1:
            for _ in range(self.num_searches):
                self._run_one(root)
        else:
            sims_done = 0
            while sims_done < self.num_searches:
                this_batch = min(self.batch_size, self.num_searches - sims_done)
                if this_batch == 1:
                    self._run_one(root)
                else:
                    self._run_batch(root, this_batch)
                sims_done += this_batch

        # Build policy from visit counts
        action_probs = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs = action_probs / total
        return action_probs
