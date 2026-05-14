//! Rust-only invariants for the encoder — no fixture, no Python reference.
//! Pins the shakmaty square indexing conventions our port relies on.

use alpha_zero_nano::encoding::{flip_sq, BOARD_H, BOARD_W, NUM_CHANNELS, TENSOR_LEN};
use shakmaty::Square;

#[test]
fn shape_constants() {
    assert_eq!(NUM_CHANNELS, 17);
    assert_eq!(BOARD_H, 8);
    assert_eq!(BOARD_W, 8);
    assert_eq!(TENSOR_LEN, 17 * 8 * 8);
}

#[test]
fn square_indexing_matches_python_chess() {
    // python-chess: A1 = 0, H8 = 63, file = sq & 7, rank = sq >> 3.
    assert_eq!(u8::from(Square::A1), 0);
    assert_eq!(u8::from(Square::H8), 63);
    for sq in Square::ALL {
        let idx = u8::from(sq);
        assert_eq!(sq.file() as u8, idx & 7);
        assert_eq!(sq.rank() as u8, idx >> 3);
    }
}

#[test]
fn flip_sq_is_rank_only_xor_56() {
    assert_eq!(flip_sq(0), 56);  // a1 → a8
    assert_eq!(flip_sq(63), 7);  // h8 → h1
    assert_eq!(flip_sq(12), 52); // e2 → e7
    for sq in 0u8..64 {
        assert_eq!(flip_sq(sq), sq ^ 56);
    }
}
