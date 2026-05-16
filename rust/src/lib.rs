//! Rust port of AlphaZero-nano's chess hot path.
//!
//! Phase 0 + Phase 1 scope: 17-channel state encoder, 4096-action codec, and a
//! `ChessGame` façade that mirrors Python's `chess_game/chess_game.py`. Parity
//! is enforced by the integration test at `tests/parity.rs`, which loads the
//! fixture produced by `tools/gen_rust_parity_fixtures.py`.
//!
//! MCTS, NN inference, and self-play are explicitly out of scope here.

pub mod action;
pub mod encoding;
pub mod game;
pub mod inference;
pub mod mcts;
pub mod selfplay;
pub mod shards;

pub use action::{action_to_uci, uci_to_action, Action, ActionError, ACTION_SIZE};
pub use encoding::{
    encode_state, EncodedState, Player, BOARD_H, BOARD_W, NUM_CHANNELS, TENSOR_LEN,
};
pub use game::{ChessGame, GameState};
