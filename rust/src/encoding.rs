//! 17-channel state encoder mirroring `chess_game/chess_game.py:107-157`.
//!
//! Layout (CHW, row-major):
//!   ch 0..=5   own pieces (P, N, B, R, Q, K)
//!   ch 6..=11  opponent pieces (same order)
//!   ch 12..=15 castling rights as full planes (own K, own Q, opp K, opp Q)
//!   ch 16      colour plane (1.0 if White to move, 0.0 if Black)
//!
//! For Black-to-move, ranks are flipped (`r = 7 - r`); files are not.
//! The encoder writes only 0.0 / 1.0, so byte-for-byte equality with the
//! Python reference is the right assertion (not tolerance).

use shakmaty::{Bitboard, Board, CastlingSide, Color, Position, Square};

pub const NUM_CHANNELS: usize = 17;
pub const BOARD_H: usize = 8;
pub const BOARD_W: usize = 8;
pub const TENSOR_LEN: usize = NUM_CHANNELS * BOARD_H * BOARD_W;

#[derive(Clone, Debug, PartialEq)]
pub struct EncodedState(pub Vec<f32>);

impl EncodedState {
    pub fn zeros() -> Self {
        Self(vec![0.0; TENSOR_LEN])
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    #[inline]
    pub fn at(&self, ch: usize, r: usize, c: usize) -> f32 {
        self.0[ch * BOARD_H * BOARD_W + r * BOARD_W + c]
    }

    #[inline]
    fn set(&mut self, ch: usize, r: usize, c: usize, v: f32) {
        self.0[ch * BOARD_H * BOARD_W + r * BOARD_W + c] = v;
    }

    fn fill_plane(&mut self, ch: usize, v: f32) {
        let base = ch * BOARD_H * BOARD_W;
        for x in &mut self.0[base..base + BOARD_H * BOARD_W] {
            *x = v;
        }
    }
}

/// White = 1, Black = -1 in the Python convention.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Player {
    White,
    Black,
}

impl Player {
    pub fn from_int(p: i32) -> Self {
        match p {
            1 => Self::White,
            -1 => Self::Black,
            other => panic!("invalid player {other}; expected 1 or -1"),
        }
    }

    pub fn to_color(self) -> Color {
        match self {
            Self::White => Color::White,
            Self::Black => Color::Black,
        }
    }

    pub fn opponent(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

/// Mirrors python-chess `sq ^ 56`: flips rank index (a1↔a8), keeps file.
#[inline]
pub fn flip_sq(sq: u8) -> u8 {
    sq ^ 56
}

/// Index helper: 0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King.
/// Mirrors `chess_game.py:15-16` order.
fn role_bitboard(board: &Board, role_idx: usize) -> Bitboard {
    match role_idx {
        0 => board.pawns(),
        1 => board.knights(),
        2 => board.bishops(),
        3 => board.rooks(),
        4 => board.queens(),
        5 => board.kings(),
        _ => unreachable!(),
    }
}

fn color_bitboard(board: &Board, color: Color) -> Bitboard {
    match color {
        Color::White => board.white(),
        Color::Black => board.black(),
    }
}

/// Encode `pos` from `player`'s perspective into a (17, 8, 8) float32 tensor.
///
/// Mirrors `chess_game.py:107-157` line-by-line:
///   - rank flip for Black at lines 128, 134
///   - castling planes from FEN-derived flags (not "can castle now"), at
///     `chess_game.py:138-152`
///   - colour plane `1.0 if White else 0.0` at `chess_game.py:155`
pub fn encode_state<P: Position>(pos: &P, player: Player) -> EncodedState {
    let mut enc = EncodedState::zeros();
    let white_to_move = player == Player::White;
    let own = player.to_color();
    let opp = player.opponent().to_color();
    let board: &Board = pos.board();
    let own_bb = color_bitboard(board, own);
    let opp_bb = color_bitboard(board, opp);

    for role_idx in 0..6 {
        let role_bb = role_bitboard(board, role_idx);
        for sq in role_bb & own_bb {
            let (r, c) = sq_to_rc(sq, white_to_move);
            enc.set(role_idx, r, c, 1.0);
        }
        for sq in role_bb & opp_bb {
            let (r, c) = sq_to_rc(sq, white_to_move);
            enc.set(6 + role_idx, r, c, 1.0);
        }
    }

    // Castling planes — read FEN-derived flags via Castles::has(color, side).
    // python-chess `has_*_castling_rights` reflects the FEN flag, not "can
    // castle right now" (it does not check attacks). shakmaty's `Castles::has`
    // matches that semantics.
    let castles = pos.castles();
    if castles.has(own, CastlingSide::KingSide) {
        enc.fill_plane(12, 1.0);
    }
    if castles.has(own, CastlingSide::QueenSide) {
        enc.fill_plane(13, 1.0);
    }
    if castles.has(opp, CastlingSide::KingSide) {
        enc.fill_plane(14, 1.0);
    }
    if castles.has(opp, CastlingSide::QueenSide) {
        enc.fill_plane(15, 1.0);
    }

    if white_to_move {
        enc.fill_plane(16, 1.0);
    }

    enc
}

#[inline]
fn sq_to_rc(sq: Square, white_to_move: bool) -> (usize, usize) {
    let r = sq.rank() as usize;
    let c = sq.file() as usize;
    if white_to_move {
        (r, c)
    } else {
        (7 - r, c)
    }
}
