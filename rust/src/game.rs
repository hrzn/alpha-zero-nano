//! `ChessGame` façade plus a `GameState` that wraps `shakmaty::Chess` with a
//! Zobrist-keyed repetition counter — needed to mirror python-chess's
//! `is_fivefold_repetition()` (shakmaty does not track repetitions natively).
//!
//! Mirrors the public surface of `chess_game/chess_game.py`. Predicate order
//! in `value_and_terminated` matches `chess_game.py:56-63` exactly.

use std::collections::HashMap;

use shakmaty::{
    fen::Fen,
    uci::UciMove,
    zobrist::{Zobrist64, ZobristHash},
    CastlingMode, Chess, Color, EnPassantMode, Position,
};

use crate::action::{action_to_move, move_to_action, Action, ActionError, ACTION_SIZE};
use crate::encoding::{encode_state, EncodedState, Player};

pub struct ChessGame;

impl ChessGame {
    pub const ROW_COUNT: usize = 8;
    pub const COLUMN_COUNT: usize = 8;
    pub const ACTION_SIZE: usize = ACTION_SIZE;
    pub const NUM_CHANNELS: usize = 17;

    pub fn new() -> Self {
        Self
    }

    pub fn initial_state(&self) -> GameState {
        GameState::initial()
    }

    /// Apply `action` to `state` and return a new state. Never mutates input
    /// (mirrors `chess_game.py:31-36`).
    pub fn update_state(&self, state: &GameState, action: Action) -> Result<GameState, ActionError> {
        let m = action_to_move(&state.pos, action)?;
        let next = state
            .pos
            .clone()
            .play(&m)
            .map_err(|e| ActionError::Illegal(action, format!("{e:?}")))?;
        let mut history = state.history.clone();
        let key = zobrist_key(&next);
        *history.entry(key).or_insert(0) += 1;
        Ok(GameState {
            pos: next,
            history,
            current_key: key,
        })
    }

    /// Legal actions in the side-to-move's spatial frame, sorted & deduped.
    /// Mirrors `chess_game.py:38-48`. Multiple promotion moves for one pawn
    /// collapse to a single action (the decoder always materialises Queen).
    pub fn valid_moves(&self, state: &GameState) -> Vec<Action> {
        let stm = match state.pos.turn() {
            Color::White => Player::White,
            Color::Black => Player::Black,
        };
        let mut out: Vec<Action> = state
            .pos
            .legal_moves()
            .iter()
            .map(|m| move_to_action(m, stm))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// (value, terminated). Predicate order matches `chess_game.py:56-63`.
    pub fn value_and_terminated(&self, state: &GameState) -> (f32, bool) {
        if state.is_checkmate() {
            return (1.0, true);
        }
        if state.is_stalemate()
            || state.is_insufficient_material()
            || state.is_seventyfive_moves()
            || state.is_fivefold_repetition()
        {
            return (0.0, true);
        }
        (0.0, false)
    }

    pub fn encode_state(&self, state: &GameState, player: Player) -> EncodedState {
        encode_state(&state.pos, player)
    }

    pub fn opponent(&self, p: Player) -> Player {
        p.opponent()
    }
}

impl Default for ChessGame {
    fn default() -> Self {
        Self::new()
    }
}

/// Position + move history (Zobrist counts) for 5-fold-repetition detection.
#[derive(Clone, Debug)]
pub struct GameState {
    pub pos: Chess,
    /// Number of times each Zobrist key has been seen *including* the current
    /// position. python-chess counts the current occurrence too, so 5-fold
    /// triggers when the counter reaches 5.
    history: HashMap<u64, u8>,
    current_key: u64,
}

impl GameState {
    pub fn initial() -> Self {
        let pos = Chess::default();
        let key = zobrist_key(&pos);
        let mut history = HashMap::new();
        history.insert(key, 1);
        Self {
            pos,
            history,
            current_key: key,
        }
    }

    /// Build from a FEN string (no history — used when the position is
    /// "anonymous", e.g. a hand-crafted endgame). Repetition detection
    /// from this state only sees the current position.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parsed: Fen = fen.parse().map_err(|e| format!("fen parse: {e:?}"))?;
        let pos: Chess = parsed
            .into_position(CastlingMode::Standard)
            .map_err(|e| format!("fen into_position: {e:?}"))?;
        let key = zobrist_key(&pos);
        let mut history = HashMap::new();
        history.insert(key, 1);
        Ok(Self {
            pos,
            history,
            current_key: key,
        })
    }

    /// Replay a UCI move. Used by parity tests to rebuild positions whose
    /// terminal predicates depend on history (5-fold repetition).
    pub fn push_uci(&mut self, uci: &str) -> Result<(), String> {
        let m = UciMove::from_ascii(uci.as_bytes())
            .map_err(|e| format!("uci parse: {e:?}"))?;
        let mv = m
            .to_move(&self.pos)
            .map_err(|e| format!("uci -> move: {e:?}"))?;
        let next = self
            .pos
            .clone()
            .play(&mv)
            .map_err(|e| format!("play: {e:?}"))?;
        self.pos = next;
        self.current_key = zobrist_key(&self.pos);
        *self.history.entry(self.current_key).or_insert(0) += 1;
        Ok(())
    }

    pub fn turn(&self) -> Player {
        match self.pos.turn() {
            Color::White => Player::White,
            Color::Black => Player::Black,
        }
    }

    pub fn is_checkmate(&self) -> bool {
        self.pos.is_checkmate()
    }

    pub fn is_stalemate(&self) -> bool {
        self.pos.is_stalemate()
    }

    pub fn is_insufficient_material(&self) -> bool {
        self.pos.is_insufficient_material()
    }

    /// Mirrors python-chess: True when the halfmove clock ≥ 150 (75 full
    /// moves with no capture or pawn advance).
    pub fn is_seventyfive_moves(&self) -> bool {
        self.pos.halfmoves() >= 150
    }

    /// Mirrors python-chess: True when the current position has appeared
    /// five or more times in the recorded history.
    pub fn is_fivefold_repetition(&self) -> bool {
        self.history.get(&self.current_key).copied().unwrap_or(0) >= 5
    }
}

fn zobrist_key(pos: &Chess) -> u64 {
    let z: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    u64::from(z)
}
