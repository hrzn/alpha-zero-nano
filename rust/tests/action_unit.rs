//! Action codec invariants — round-trips and the auto-queen promotion rule.

use alpha_zero_nano::action::{action_to_move, action_to_uci, uci_to_action};
use alpha_zero_nano::encoding::Player;
use alpha_zero_nano::game::GameState;
use shakmaty::{Move, Role};

#[test]
fn white_uci_roundtrip() {
    for uci in ["e2e4", "g1f3", "b1c3", "d2d4"] {
        let a = uci_to_action(uci, Player::White).unwrap();
        assert!((a as usize) < 4096, "action out of range for {uci}");
        assert_eq!(action_to_uci(a, Player::White).unwrap(), uci);
    }
}

#[test]
fn black_uci_roundtrip() {
    for uci in ["e7e5", "g8f6", "d7d5", "b8c6"] {
        let a = uci_to_action(uci, Player::Black).unwrap();
        assert_eq!(action_to_uci(a, Player::Black).unwrap(), uci);
    }
}

#[test]
fn black_e7e5_lands_in_flipped_frame() {
    // Mirrors `test_chess_game.py:141-147`: black's e7→e5 from-square should
    // sit at encoding row 1 (rank flipped) so the action index is in the same
    // frame as encode_state.
    let a = uci_to_action("e7e5", Player::Black).unwrap();
    let from_sq = (a / 64) as u8;
    let from_row = from_sq / 8;
    let from_col = from_sq % 8;
    assert_eq!(from_row, 1, "expected row 1, got {from_row}");
    assert_eq!(from_col, 4, "e-file → col 4");
}

#[test]
fn auto_queen_promotion() {
    // White pawn on a7, white to move; uci a7a8 has no promotion suffix —
    // action_to_move must auto-queen.
    let state = GameState::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
    let a = uci_to_action("a7a8", Player::White).unwrap();
    let m = action_to_move(&state.pos, a).unwrap();
    match m {
        Move::Normal { promotion, role, .. } => {
            assert_eq!(role, Role::Pawn);
            assert_eq!(promotion, Some(Role::Queen), "must auto-queen");
        }
        other => panic!("expected Move::Normal with queen promotion, got {other:?}"),
    }
}
