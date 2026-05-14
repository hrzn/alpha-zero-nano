//! Action codec mirroring `chess_game/chess_game.py:75-105, 159-178`.
//!
//! Action = `from_sq * 64 + to_sq` ∈ [0, 4096). For Black-to-move, both
//! squares are flipped via `^ 56` before encoding, so action indices always
//! land in the player's spatial frame (same frame as `encode_state`).
//!
//! Promotion: any pawn-to-back-rank action decodes to a queen promotion,
//! mirroring `chess_game.py:99-104`. The encoder strips the promotion piece,
//! so all four shakmaty `Move::Normal` promotions collapse to one action.

use shakmaty::{uci::UciMove, CastlingSide, EnPassantMode, File, Move, Position, Role, Square};

use crate::encoding::{flip_sq, Player};

pub const ACTION_SIZE: usize = 4096;
pub type Action = u16;

#[derive(Debug, thiserror::Error)]
pub enum ActionError {
    #[error("action {0} out of range (must be < 4096)")]
    OutOfRange(u32),
    #[error("illegal move at action {0}: {1}")]
    Illegal(Action, String),
    #[error("invalid uci '{0}': {1}")]
    UciParse(String, String),
    #[error("no piece on from-square for action {0}")]
    EmptyFromSquare(Action),
}

/// Encode a shakmaty `Move` as an action in `stm`'s spatial frame.
pub fn move_to_action(m: &Move, stm: Player) -> Action {
    let (from, to) = move_squares(m);
    let (f, t) = if stm == Player::Black {
        (flip_sq(u8::from(from)), flip_sq(u8::from(to)))
    } else {
        (u8::from(from), u8::from(to))
    };
    (f as u16) * 64 + (t as u16)
}

/// Decode an action into a concrete shakmaty `Move` legal in `pos`.
///
/// Auto-queens any pawn-to-back-rank move regardless of how the action was
/// produced (mirrors `chess_game.py:99-104`). For king "two-square" moves on
/// the back rank we emit `Move::Castle`; for pawn diagonals to the en passant
/// target we emit `Move::EnPassant`. Everything else is `Move::Normal`.
pub fn action_to_move<P: Position>(pos: &P, action: Action) -> Result<Move, ActionError> {
    if action as usize >= ACTION_SIZE {
        return Err(ActionError::OutOfRange(action as u32));
    }
    let stm_player = match pos.turn() {
        shakmaty::Color::White => Player::White,
        shakmaty::Color::Black => Player::Black,
    };
    let from_idx = (action / 64) as u8;
    let to_idx = (action % 64) as u8;
    let (from_u, to_u) = if stm_player == Player::Black {
        (flip_sq(from_idx), flip_sq(to_idx))
    } else {
        (from_idx, to_idx)
    };
    let from = sq_from_u8(from_u);
    let to = sq_from_u8(to_u);

    let board = pos.board();
    let piece = match board.piece_at(from) {
        Some(p) => p,
        None => return Err(ActionError::EmptyFromSquare(action)),
    };

    // Promotion: pawn reaching rank 0 or 7 → auto-queen.
    if piece.role == Role::Pawn && (to.rank() as u8 == 0 || to.rank() as u8 == 7) {
        let capture = board.role_at(to);
        return Ok(Move::Normal {
            role: Role::Pawn,
            from,
            to,
            capture,
            promotion: Some(Role::Queen),
        });
    }

    // Castle: king two-square horizontal move on the back rank.
    if piece.role == Role::King {
        let dx = (to.file() as i32) - (from.file() as i32);
        if dx.abs() == 2 && from.rank() == to.rank() {
            let side = if dx > 0 {
                CastlingSide::KingSide
            } else {
                CastlingSide::QueenSide
            };
            if pos.castles().has(piece.color, side) {
                let rook_file = if dx > 0 { File::H } else { File::A };
                let rook_sq = Square::from_coords(rook_file, from.rank());
                return Ok(Move::Castle {
                    king: from,
                    rook: rook_sq,
                });
            }
            // Fall through if the FEN doesn't claim that castle right.
        }
    }

    // En passant: pawn diagonal to the EP square with an empty target.
    if piece.role == Role::Pawn {
        let dx = (to.file() as i32) - (from.file() as i32);
        let dy = (to.rank() as i32) - (from.rank() as i32);
        if dx.abs() == 1 && dy.abs() == 1 && board.role_at(to).is_none() {
            if let Some(ep_sq) = pos.ep_square(EnPassantMode::Legal) {
                if ep_sq == to {
                    return Ok(Move::EnPassant { from, to });
                }
            }
        }
    }

    Ok(Move::Normal {
        role: piece.role,
        from,
        to,
        capture: board.role_at(to),
        promotion: None,
    })
}

/// UCI string → action. `player` selects the spatial frame.
pub fn uci_to_action(uci: &str, player: Player) -> Result<Action, ActionError> {
    let m = UciMove::from_ascii(uci.as_bytes())
        .map_err(|e| ActionError::UciParse(uci.to_owned(), e.to_string()))?;
    let (from, to) = match m {
        UciMove::Normal { from, to, .. } => (from, to),
        _ => {
            return Err(ActionError::UciParse(
                uci.to_owned(),
                "only normal UCI moves are supported".into(),
            ))
        }
    };
    let (f, t) = if player == Player::Black {
        (flip_sq(u8::from(from)), flip_sq(u8::from(to)))
    } else {
        (u8::from(from), u8::from(to))
    };
    Ok((f as u16) * 64 + (t as u16))
}

/// Action → UCI string in absolute coordinates (un-flips for Black).
pub fn action_to_uci(action: Action, player: Player) -> Result<String, ActionError> {
    if action as usize >= ACTION_SIZE {
        return Err(ActionError::OutOfRange(action as u32));
    }
    let mut from = (action / 64) as u8;
    let mut to = (action % 64) as u8;
    if player == Player::Black {
        from = flip_sq(from);
        to = flip_sq(to);
    }
    Ok(format!("{}{}", sq_from_u8(from), sq_from_u8(to)))
}

fn sq_from_u8(idx: u8) -> Square {
    // shakmaty Squares match python-chess: A1=0, H8=63. `Square::ALL[idx]` is
    // the safe construction in 0.27.
    Square::ALL[idx as usize]
}

/// Return the (from, to) squares for an arbitrary shakmaty Move.
fn move_squares(m: &Move) -> (Square, Square) {
    match *m {
        Move::Normal { from, to, .. } => (from, to),
        Move::EnPassant { from, to } => (from, to),
        Move::Castle { king, rook } => {
            // Python encodes castling via the king's UCI form (e1g1 / e1c1).
            // Recover the king's destination square from the rook's file.
            let to_file = if rook.file() > king.file() {
                File::G
            } else {
                File::C
            };
            (king, Square::from_coords(to_file, king.rank()))
        }
        Move::Put { .. } => panic!("Put moves unsupported in chess"),
    }
}
