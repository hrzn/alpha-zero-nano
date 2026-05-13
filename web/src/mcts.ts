// Browser-side port of mcts/mcts.py — sequential PUCT MCTS only.
// Skipped from the Python version (training-only optimizations):
//   - Virtual loss
//   - Batched leaf evaluation
//   - Dirichlet root noise (we want deterministic best-play at inference)

import { Connect4 } from "./connect4";
import type { Model } from "./model";

export class MCTSNode {
  visitCount = 0;
  valueSum = 0;
  children: Map<number, MCTSNode> = new Map();

  constructor(
    readonly game: Connect4,
    readonly state: Int8Array,
    readonly player: number,
    readonly prior: number = 0,
    public parent: MCTSNode | null = null,
    readonly actionTaken: number | null = null,
  ) {}

  isExpanded(): boolean {
    return this.children.size > 0;
  }

  /**
   * PUCT score for selection — `q` is from the parent's perspective
   * (so we negate the child's value_sum, which is from child-player's POV).
   */
  puctScore(cPuct: number): number {
    const q = this.visitCount === 0 ? 0 : -this.valueSum / this.visitCount;
    if (!this.parent) return q;
    const explore = cPuct * this.prior * Math.sqrt(this.parent.visitCount) / (1 + this.visitCount);
    return q + explore;
  }

  selectChild(cPuct: number): { action: number; child: MCTSNode } {
    let bestScore = -Infinity;
    let bestAction = -1;
    let bestChild: MCTSNode | null = null;
    for (const [action, child] of this.children) {
      const s = child.puctScore(cPuct);
      if (s > bestScore) { bestScore = s; bestAction = action; bestChild = child; }
    }
    return { action: bestAction, child: bestChild! };
  }

  /** Build children using a (masked, renormalized) policy as priors. */
  expand(policy: Float32Array): void {
    const valid = this.game.getValidMoves(this.state);
    let sum = 0;
    const masked = new Float32Array(policy.length);
    for (let a = 0; a < policy.length; a++) {
      if (valid[a]) { masked[a] = policy[a]; sum += masked[a]; }
    }
    if (sum > 0) for (let a = 0; a < masked.length; a++) masked[a] /= sum;

    for (let a = 0; a < this.game.actionSize; a++) {
      if (!valid[a]) continue;
      const childState = this.game.copy(this.state);
      this.game.updateState(childState, a, this.player);
      const child = new MCTSNode(
        this.game,
        childState,
        this.game.getOpponent(this.player),
        masked[a],
        this,
        a,
      );
      this.children.set(a, child);
    }
  }

  backpropagate(value: number): void {
    this.valueSum += value;
    this.visitCount += 1;
    if (this.parent) this.parent.backpropagate(-value);
  }
}

export interface MCTSOptions {
  numSearches: number;
  cPuct: number;
}

export class MCTS {
  private root: MCTSNode | null = null;

  constructor(
    readonly game: Connect4,
    readonly model: Model,
    public opts: MCTSOptions,
  ) {}

  /** Promote the chosen child to root so its subtree is reused on the next search. */
  advanceRoot(action: number): void {
    if (this.root && this.root.children.has(action)) {
      this.root = this.root.children.get(action)!;
      this.root.parent = null;
    } else {
      this.root = null;
    }
  }

  resetRoot(): void {
    this.root = null;
  }

  private evaluate(state: Int8Array, player: number): { policy: Float32Array; value: number } {
    const enc = this.game.encodeState(state, player);
    const result = this.model.forward(enc, false);
    return { policy: result.policy, value: result.value };
  }

  private runOne(root: MCTSNode): void {
    let node = root;
    while (node.isExpanded()) {
      const { child } = node.selectChild(this.opts.cPuct);
      node = child;
    }

    let value: number;
    const { value: terminalValue, terminated } = node.actionTaken !== null
      ? this.game.getValueAndTerminated(node.state, node.actionTaken)
      : { value: 0, terminated: false };

    if (terminated) {
      // terminalValue is from the perspective of the player who just moved
      // (the parent's player); flip for this node's POV.
      value = -terminalValue;
    } else {
      const { policy, value: v } = this.evaluate(node.state, node.player);
      node.expand(policy);
      value = v;
    }

    node.backpropagate(value);
  }

  /** Run num_searches simulations from `state`, return action probabilities (visit counts normalized). */
  search(state: Int8Array, player: number): Float32Array {
    if (this.root === null) {
      this.root = new MCTSNode(this.game, this.game.copy(state), player);
      const { policy } = this.evaluate(state, player);
      this.root.expand(policy);
    }
    const root = this.root;

    for (let i = 0; i < this.opts.numSearches; i++) this.runOne(root);

    const probs = new Float32Array(this.game.actionSize);
    let total = 0;
    for (const [action, child] of root.children) {
      probs[action] = child.visitCount;
      total += child.visitCount;
    }
    if (total > 0) for (let i = 0; i < probs.length; i++) probs[i] /= total;
    return probs;
  }

  /** Expose the current root for tree visualization. */
  getRoot(): MCTSNode | null {
    return this.root;
  }
}
