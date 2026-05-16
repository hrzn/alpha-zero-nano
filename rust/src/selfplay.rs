//! Self-play game loop, mirrors `train/train.py::self_play`.
//!
//! One game = MCTS-driven moves with Dirichlet noise at every root, temperature
//! sampling for the first `temp_threshold` moves, argmax after. Each move
//! records `(encoded_state, mcts_policy, acting_player)`; once the game ends
//! (terminal position or move-cap value bootstrap) outcomes are assigned per
//! acting player and we return a vec of training examples.
//!
//! Differences from the Python version:
//!   - **No tree reuse** between moves. The Python MCTS calls `advance_root`
//!     after each move to keep visit counts in the chosen child's subtree;
//!     our Rust MCTS is currently stateless (`search` builds a fresh tree
//!     each call), so we pay a 2× search cost per move. Tree reuse is Opt 1
//!     in `design/OPTIMIZATIONS.md`; revisit in Phase 6 if profiling shows
//!     it matters.
//!   - **f32 throughout for outcomes**, matching the Python numpy dtype.

use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::Rng;
use shakmaty::{Chess, Color, Position};

use crate::action::{action_to_move, Action, ACTION_SIZE};
use crate::encoding::{encode_state, EncodedState, Player};
use crate::inference::Evaluator;
use crate::mcts::search_with_dirichlet;

/// One training example. Matches the tuple shape Python's `train_step`
/// consumes (`train/train.py:148-177`).
#[derive(Debug, Clone)]
pub struct Example {
    pub state: EncodedState,
    pub policy: Vec<f32>,  // length ACTION_SIZE, sums to 1.0
    pub value: f32,         // in [-1, 1], from acting player's perspective
}

/// Self-play configuration. Mirrors the MCTS + game-loop knobs in Python's
/// `train/run_training.py` presets.
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub num_searches: u32,
    pub c_puct: f64,
    pub batch_size: u32,
    pub dirichlet_alpha: f32,
    pub dirichlet_epsilon: f32,
    pub max_moves: u32,
    /// Sample proportionally to MCTS visit counts for the first
    /// `temp_threshold` moves; argmax after. `None` means always sample.
    pub temp_threshold: Option<u32>,
}

fn player_of(pos: &Chess) -> Player {
    match pos.turn() {
        Color::White => Player::White,
        Color::Black => Player::Black,
    }
}

/// Terminal predicate matching `crate::mcts::terminal_at` semantics, kept
/// in sync with `chess_game/chess_game.py:50-63` minus 5-fold repetition
/// (which needs history tracking we don't carry in self-play either).
fn terminal_value(pos: &Chess) -> (f32, bool) {
    if pos.is_checkmate() {
        (1.0, true)
    } else if pos.is_stalemate() || pos.is_insufficient_material() || pos.halfmoves() >= 150 {
        (0.0, true)
    } else {
        (0.0, false)
    }
}

/// Run one self-play game from the standard starting chess position and
/// return all (state, policy, value) tuples collected during the game.
pub fn play_game<E: Evaluator, R: Rng>(
    evaluator: &E,
    cfg: &SelfPlayConfig,
    rng: &mut R,
) -> Vec<Example> {
    let mut state = Chess::default();
    // Pending examples carry (encoded_state, mcts_policy, acting_player);
    // outcomes are assigned once the game ends and the side that won/lost
    // is known. Mirrors `train/train.py:33-69`.
    let mut pending: Vec<(EncodedState, Vec<f32>, Player)> = Vec::new();
    let mut move_count: u32 = 0;
    let final_value: f32;
    let final_player: Player;

    loop {
        let acting = player_of(&state);
        let policy = search_with_dirichlet(
            evaluator,
            &state,
            cfg.num_searches,
            cfg.c_puct,
            cfg.batch_size,
            cfg.dirichlet_alpha,
            cfg.dirichlet_epsilon,
            rng,
        );

        let enc = encode_state(&state, acting);
        pending.push((enc, policy.clone(), acting));

        // Pick an action. Temperature sampling for the opening; argmax after.
        let action = pick_action(&policy, move_count, cfg.temp_threshold, rng);

        // Apply move.
        let mv = action_to_move(&state, action)
            .expect("MCTS-emitted action must decode to a Move");
        state = state.clone().play(&mv).expect("MCTS-emitted move must be legal");
        move_count += 1;

        // Check terminal / move-cap.
        let (term_val, terminated) = terminal_value(&state);
        if terminated {
            // `term_val` is from the side that just moved's perspective; we
            // need it from `acting`'s view, which is also "the side that just
            // moved". So no flip needed at the level of `acting`.
            final_value = term_val;
            final_player = acting;
            break;
        }
        if move_count >= cfg.max_moves {
            // Value bootstrap: ask the network for the next-player's view,
            // negate to get `acting`'s view (matches `train/train.py:54-60`).
            let next_player = acting.opponent();
            let (_, bootstrap_v) = evaluator.evaluate(&state, next_player);
            final_value = -bootstrap_v;
            final_player = acting;
            break;
        }
    }

    // Stamp the outcome onto every recorded example. Examples played by
    // `final_player` get `+final_value`; the opponent's examples get
    // `-final_value`. Mirrors `train/train.py:62-69`.
    pending
        .into_iter()
        .map(|(state, policy, acting)| {
            let value = if acting == final_player { final_value } else { -final_value };
            Example {
                state,
                policy,
                value,
            }
        })
        .collect()
}

fn pick_action<R: Rng>(
    policy: &[f32],
    move_count: u32,
    temp_threshold: Option<u32>,
    rng: &mut R,
) -> Action {
    debug_assert_eq!(policy.len(), ACTION_SIZE);
    let argmax = match temp_threshold {
        Some(t) => move_count >= t,
        None => false,
    };
    if argmax {
        // numpy.argmax tie-break: first index of the max.
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &p) in policy.iter().enumerate() {
            if p > best_v {
                best_v = p;
                best_i = i;
            }
        }
        best_i as u16
    } else {
        // Proportional sampling. WeightedIndex panics on all-zero weights;
        // pre-check (shouldn't happen unless MCTS produced no visits).
        let any_nonzero = policy.iter().any(|&p| p > 0.0);
        if !any_nonzero {
            // Fall back to argmax-of-zeros (action 0). Shouldn't be reached
            // in a healthy MCTS run.
            return 0;
        }
        let dist = WeightedIndex::new(policy).expect("WeightedIndex::new with valid policy");
        dist.sample(rng) as u16
    }
}
