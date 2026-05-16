//! MCTS port of `mcts/mcts.py`.
//!
//! Arena-backed tree with `u32` node indices (no `Rc`, no recursion through
//! lifetimes). The algorithm itself mirrors Python line-for-line: PUCT
//! selection in f64 (same precision as Python floats), strict `>` tie-break,
//! children iterated in action-index order, identical sign-flipping in
//! backpropagation.
//!
//! What's **not** ported in Phase 2:
//!   - 5-fold repetition: shakmaty's `Chess` carries no history, so
//!     `terminal_at` doesn't check it. MCTS test cases stay in the early
//!     game where it cannot trigger. The Python algorithm does the same for
//!     all practical search depths.
//!   - Dirichlet noise at the root: the parity test runs with
//!     `dirichlet_alpha = 0`, matching the fixture generator. Adding noise
//!     later is a few lines.

use rand::Rng;
use rand_distr::{Distribution, Gamma};
use shakmaty::{Chess, Color, Position};

use crate::action::{action_to_move, move_to_action, Action, ACTION_SIZE};
use crate::encoding::Player;
use crate::inference::Evaluator;

pub type NodeId = u32;

/// Virtual-loss value applied during batched leaf selection — same value
/// Python uses at `mcts/mcts.py:15` (_VIRTUAL_LOSS = 1).
const VIRTUAL_LOSS: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct Node {
    pub state: Chess,
    pub prior: f32,
    pub visit_count: u32,
    pub value_sum: f64,
    pub parent: Option<NodeId>,
    pub action_taken: Option<Action>,
    /// (action, child_id) pairs in action-index order. Iteration order
    /// must match Python's dict insertion order at expand time.
    pub children: Vec<(Action, NodeId)>,
}

#[derive(Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: NodeId,
}

impl Tree {
    fn new(root: Node) -> Self {
        Self {
            nodes: vec![root],
            root: 0,
        }
    }

    fn alloc(&mut self, node: Node) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(node);
        id
    }

    fn is_expanded(&self, id: NodeId) -> bool {
        !self.nodes[id as usize].children.is_empty()
    }

}

/// Player-from-position helper. shakmaty's `turn()` is the canonical side
/// to move; we map it to our `Player` enum on demand.
#[inline]
fn player_of(pos: &Chess) -> Player {
    match pos.turn() {
        Color::White => Player::White,
        Color::Black => Player::Black,
    }
}

/// Terminal predicate mirroring `chess_game.py:50-63`, **minus 5-fold
/// repetition** (see module docs). Returns (value, terminated) where value
/// is from the perspective of the side that just moved.
fn terminal_at(pos: &Chess) -> (f32, bool) {
    if pos.is_checkmate() {
        (1.0, true)
    } else if pos.is_stalemate() || pos.is_insufficient_material() || pos.halfmoves() >= 150 {
        (0.0, true)
    } else {
        (0.0, false)
    }
}

/// PUCT score for a child — same formula as `mcts/mcts.py:49-54`.
///
/// All math in f64 to match Python's float precision. The prior is stored
/// as f32 (matches the numpy softmax output the Python algorithm carries)
/// but lifted to f64 for the multiplication, identically to Python.
#[inline]
fn puct_score(child: &Node, parent_visits: u32, c_puct: f64) -> f64 {
    let q = if child.visit_count == 0 {
        0.0
    } else {
        -child.value_sum / child.visit_count as f64
    };
    let exploration = c_puct
        * (child.prior as f64)
        * (parent_visits as f64).sqrt()
        / (1.0 + child.visit_count as f64);
    q + exploration
}

/// Select the child with the highest PUCT score. Strict `>` means the
/// **first** child (in action-index order) wins ties — same as Python's
/// `if score > best_score`.
fn select_best_child(tree: &Tree, id: NodeId, c_puct: f64) -> NodeId {
    let parent_visits = tree.nodes[id as usize].visit_count;
    let children = &tree.nodes[id as usize].children;
    debug_assert!(!children.is_empty(), "select_best_child on leaf");
    let mut best = children[0].1;
    let mut best_score = puct_score(&tree.nodes[best as usize], parent_visits, c_puct);
    for &(_, cid) in children.iter().skip(1) {
        let score = puct_score(&tree.nodes[cid as usize], parent_visits, c_puct);
        if score > best_score {
            best_score = score;
            best = cid;
        }
    }
    best
}

/// Materialise children for `node_id`, given the raw 4096-prior policy
/// vector. Mirrors `mcts/mcts.py:56-77`: mask by legal moves, renormalize,
/// iterate in action-index order, create a child per legal action.
///
/// The expand step is where action-index ordering is established. Selection
/// later iterates these children in insertion order; that order must match
/// Python's `range(action_size)` iteration.
fn expand(tree: &mut Tree, node_id: NodeId, policy: &[f32]) {
    debug_assert_eq!(policy.len(), ACTION_SIZE);
    let parent_state = tree.nodes[node_id as usize].state.clone();
    let stm = player_of(&parent_state);

    // Build the list of (action, raw_prior) for legal moves, in action-index
    // order. Iterate the legal-move generator and bucket by the action index
    // they encode to (deduping promotion variants that collapse to one Q).
    let mut legal_actions: Vec<Action> = parent_state
        .legal_moves()
        .iter()
        .map(|m| move_to_action(m, stm))
        .collect();
    legal_actions.sort_unstable();
    legal_actions.dedup();

    // Mask + renormalize. Use f32 throughout to match Python's numpy f32
    // softmax storage — the renormalised priors are what end up in
    // `child.prior`, which is f32.
    let sum: f32 = legal_actions.iter().map(|&a| policy[a as usize]).sum();
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };

    for action in legal_actions {
        let raw = policy[action as usize];
        let prior = if sum > 0.0 { raw * inv } else { 0.0 };
        // Decode the action → shakmaty Move → apply.
        let m = action_to_move(&parent_state, action)
            .expect("legal_moves emitted an action we cannot decode back to a Move");
        let child_state = parent_state.clone().play(&m).expect("legal move must apply");
        let child = Node {
            state: child_state,
            prior,
            visit_count: 0,
            value_sum: 0.0,
            parent: Some(node_id),
            action_taken: Some(action),
            children: Vec::new(),
        };
        let child_id = tree.alloc(child);
        tree.nodes[node_id as usize].children.push((action, child_id));
    }
}

/// Backpropagate `value` up to the root, flipping sign at each level.
/// Mirrors `mcts/mcts.py:79-84`.
fn backpropagate(tree: &mut Tree, node_id: NodeId, value: f64) {
    let mut current = Some(node_id);
    let mut v = value;
    while let Some(id) = current {
        let node = &mut tree.nodes[id as usize];
        node.value_sum += v;
        node.visit_count += 1;
        current = node.parent;
        v = -v;
    }
}

/// Apply virtual loss (`vl = +1`) along the path from `leaf` up to root.
/// Mirrors `Node.apply_virtual_loss` at `mcts/mcts.py:86-90`: visit_count
/// += vl, value_sum += vl. The +vl on value_sum biases the parent's Q
/// estimate downward for this child, pushing concurrent selectors toward
/// other branches in the same batch.
fn apply_virtual_loss(tree: &mut Tree, leaf: NodeId) {
    let mut current = Some(leaf);
    while let Some(id) = current {
        let node = &mut tree.nodes[id as usize];
        node.visit_count += VIRTUAL_LOSS as u32;
        node.value_sum += VIRTUAL_LOSS;
        current = node.parent;
    }
}

fn undo_virtual_loss(tree: &mut Tree, leaf: NodeId) {
    let mut current = Some(leaf);
    while let Some(id) = current {
        let node = &mut tree.nodes[id as usize];
        node.visit_count -= VIRTUAL_LOSS as u32;
        node.value_sum -= VIRTUAL_LOSS;
        current = node.parent;
    }
}

/// One sequential MCTS simulation. Mirrors `mcts/mcts.py:173-197`.
fn run_one<E: Evaluator>(tree: &mut Tree, c_puct: f64, evaluator: &E) {
    // Selection: descend until we hit an unexpanded node.
    let mut id = tree.root;
    while tree.is_expanded(id) {
        id = select_best_child(tree, id, c_puct);
    }

    let state = tree.nodes[id as usize].state.clone();
    let (term_val, terminated) = terminal_at(&state);
    let value: f64 = if terminated {
        // Value is from the perspective of the player who just moved
        // (parent's player). We need it from this node's perspective for
        // backprop, hence the sign flip — same as `mcts.py:186-190`.
        -(term_val as f64)
    } else {
        let player = player_of(&state);
        let (policy, v) = evaluator.evaluate(&state, player);
        expand(tree, id, &policy);
        v as f64
    };

    backpropagate(tree, id, value);
}

/// One batched MCTS round. Mirrors `mcts/mcts.py:199-248`:
///   Phase 1: select `batch_size` leaves with virtual loss applied each.
///   Phase 2: batch-evaluate unique unexpanded non-terminal leaves once.
///   Phase 3: undo VL, backprop the real value (terminal or NN).
fn run_batch<E: Evaluator>(tree: &mut Tree, c_puct: f64, evaluator: &E, batch_size: u32) {
    // Phase 1 — selection with virtual loss.
    let mut sim_results: Vec<(NodeId, bool, f64)> = Vec::with_capacity(batch_size as usize);
    for _ in 0..batch_size {
        let mut id = tree.root;
        while tree.is_expanded(id) {
            id = select_best_child(tree, id, c_puct);
        }
        apply_virtual_loss(tree, id);
        let state = tree.nodes[id as usize].state.clone();
        let (term_val, terminated) = terminal_at(&state);
        // value is meaningful only when terminated; we overwrite in Phase 3
        // otherwise.
        let value = if terminated { -(term_val as f64) } else { 0.0 };
        sim_results.push((id, terminated, value));
    }

    // Phase 2 — collect unique unexpanded non-terminal leaves.
    // Python uses `id(node)` (object identity); our NodeId is the canonical
    // identity, so we dedupe on it directly. Insertion order matters because
    // it determines the batch order seen by the evaluator — but the
    // evaluator (NN) is permutation-invariant per-row, so order doesn't
    // change outputs.
    let mut unique_order: Vec<NodeId> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &(id, term, _) in &sim_results {
        if !term && !tree.is_expanded(id) && seen.insert(id) {
            unique_order.push(id);
        }
    }

    let mut node_eval_map: std::collections::HashMap<NodeId, (Vec<f32>, f32)> =
        std::collections::HashMap::new();
    if !unique_order.is_empty() {
        let leaves: Vec<(Chess, Player)> = unique_order
            .iter()
            .map(|&id| {
                let state = tree.nodes[id as usize].state.clone();
                let player = player_of(&state);
                (state, player)
            })
            .collect();
        let results = evaluator.evaluate_batch(&leaves);
        for (&id, result) in unique_order.iter().zip(results.into_iter()) {
            // Expand if still unexpanded (it always is at this point).
            if !tree.is_expanded(id) {
                expand(tree, id, &result.0);
            }
            node_eval_map.insert(id, result);
        }
    }

    // Phase 3 — undo VL, backprop.
    for (id, terminated, term_value) in sim_results {
        undo_virtual_loss(tree, id);
        let value = if !terminated {
            // If we evaluated this leaf in Phase 2, use the NN value.
            // Otherwise (duplicate leaf in the batch) the leaf is now
            // expanded and we have no NN value cached for it — fall back to
            // 0.0 (Python falls through and uses the stale `value` from
            // Phase 1, which was 0.0 for non-terminal leaves).
            node_eval_map
                .get(&id)
                .map(|(_, v)| *v as f64)
                .unwrap_or(0.0)
        } else {
            term_value
        };
        backpropagate(tree, id, value);
    }
}

/// Sample from symmetric Dirichlet(alpha, ..., alpha) of length `n` by drawing
/// n iid Gamma(alpha, 1) samples and normalising — the standard construction.
/// rand_distr 0.5's `Dirichlet` is const-generic over N, but chess has a
/// runtime-sized root child list, so we roll it ourselves.
fn sample_symmetric_dirichlet<R: Rng>(alpha: f32, n: usize, rng: &mut R) -> Vec<f64> {
    let gamma = Gamma::<f64>::new(alpha as f64, 1.0).expect("alpha must be > 0");
    let mut samples: Vec<f64> = (0..n).map(|_| gamma.sample(rng)).collect();
    let sum: f64 = samples.iter().sum();
    if sum > 0.0 {
        for s in samples.iter_mut() {
            *s /= sum;
        }
    }
    samples
}

/// Apply Dirichlet noise to root children's priors. Mirrors
/// `mcts/mcts.py:119-132`.
///
///   child.prior = (1 - eps) * child.prior + eps * eta_i
///
/// where `eta` ~ Dirichlet(alpha, ..., alpha) of length = #root children.
/// No-op if alpha <= 0 or root has no children.
fn apply_dirichlet_noise<R: Rng>(
    tree: &mut Tree,
    alpha: f32,
    epsilon: f32,
    rng: &mut R,
) {
    let n = tree.nodes[tree.root as usize].children.len();
    if alpha <= 0.0 || n == 0 {
        return;
    }
    let noise = sample_symmetric_dirichlet(alpha, n, rng);
    let one_minus_eps = (1.0 - epsilon) as f64;
    // Children Vec is in insertion (action-index) order; iterate by index
    // so we don't borrow `tree` mutably and immutably at once.
    for i in 0..n {
        let child_id = tree.nodes[tree.root as usize].children[i].1;
        let prior = tree.nodes[child_id as usize].prior as f64;
        let blended = one_minus_eps * prior + epsilon as f64 * noise[i];
        tree.nodes[child_id as usize].prior = blended as f32;
    }
}

/// Stateless MCTS search without root-noise. Used by the Phase 2 parity
/// fixtures, which need to be bit-identical to Python's MCTS run with
/// `dirichlet_alpha=0` (where `_apply_dirichlet_noise` is a no-op).
pub fn search<E: Evaluator>(
    evaluator: &E,
    state: &Chess,
    num_searches: u32,
    c_puct: f64,
    batch_size: u32,
) -> Vec<f32> {
    let mut tree = build_root_and_expand(evaluator, state);
    run_simulations(&mut tree, evaluator, num_searches, c_puct, batch_size);
    root_visit_policy(&tree)
}

/// Stateless MCTS search with root Dirichlet noise. Used during self-play
/// to broaden exploration at the root — mirrors the production Python call
/// path in `train/train.py::self_play` via the MCTS instance configured
/// with `dirichlet_alpha > 0`.
pub fn search_with_dirichlet<E: Evaluator, R: Rng>(
    evaluator: &E,
    state: &Chess,
    num_searches: u32,
    c_puct: f64,
    batch_size: u32,
    dirichlet_alpha: f32,
    dirichlet_epsilon: f32,
    rng: &mut R,
) -> Vec<f32> {
    let mut tree = build_root_and_expand(evaluator, state);
    apply_dirichlet_noise(&mut tree, dirichlet_alpha, dirichlet_epsilon, rng);
    run_simulations(&mut tree, evaluator, num_searches, c_puct, batch_size);
    root_visit_policy(&tree)
}

fn build_root_and_expand<E: Evaluator>(evaluator: &E, state: &Chess) -> Tree {
    let root_node = Node {
        state: state.clone(),
        prior: 0.0,
        visit_count: 0,
        value_sum: 0.0,
        parent: None,
        action_taken: None,
        children: Vec::new(),
    };
    let mut tree = Tree::new(root_node);
    let player = player_of(state);
    let (root_policy, _root_value) = evaluator.evaluate(state, player);
    let root_id = tree.root;
    expand(&mut tree, root_id, &root_policy);
    tree
}

fn run_simulations<E: Evaluator>(
    tree: &mut Tree,
    evaluator: &E,
    num_searches: u32,
    c_puct: f64,
    batch_size: u32,
) {
    if batch_size <= 1 {
        for _ in 0..num_searches {
            run_one(tree, c_puct, evaluator);
        }
    } else {
        let mut done = 0u32;
        while done < num_searches {
            let this_batch = batch_size.min(num_searches - done);
            if this_batch == 1 {
                run_one(tree, c_puct, evaluator);
            } else {
                run_batch(tree, c_puct, evaluator, this_batch);
            }
            done += this_batch;
        }
    }
}

fn root_visit_policy(tree: &Tree) -> Vec<f32> {
    let mut probs = vec![0.0f32; ACTION_SIZE];
    let mut total = 0.0f32;
    for &(action, child_id) in &tree.nodes[tree.root as usize].children {
        let c = tree.nodes[child_id as usize].visit_count as f32;
        probs[action as usize] = c;
        total += c;
    }
    if total > 0.0 {
        let inv = 1.0 / total;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
    probs
}

