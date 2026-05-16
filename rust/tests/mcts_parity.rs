//! MCTS parity test against `tools/gen_mcts_parity_fixtures.py`.
//!
//! Each test case carries a starting position, MCTS configuration, a lookup
//! table of every `(fen, player) -> (policy, value)` Python's MCTS queried,
//! and the expected normalized visit-count distribution.
//!
//! Rust runs MCTS with a `FixtureEvaluator` backed by that table. A fixture
//! miss (Rust queries a state Python never saw) is a hard error — it means
//! the algorithm diverged from Python during selection or expansion.
//!
//! Acceptance: top-action match (exact argmax) **and** max-abs-diff on the
//! normalized policy under `TOL`. Visits are integer-valued so equal counts
//! produce bit-exact normalized probs; tolerance >0 only matters if the
//! algorithm itself drifts (e.g. PUCT float ordering on a tie).

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use alpha_zero_nano::game::GameState;
use alpha_zero_nano::inference::{FixtureEntry, FixtureEvaluator};
use alpha_zero_nano::mcts::search;
use serde::Deserialize;

// 1.0 / 65 ≈ 0.0154; one visit out of 64 sims. Anything tighter would flag
// a single-visit drift, which we want — but accommodates floating-point
// ordering edge cases at the very last simulation.
const TOL: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema_version: u32,
    action_size: usize,
    test_cases: Vec<TestCase>,
    #[serde(default)]
    #[allow(dead_code)]
    mcts_source_sha: String,
    #[serde(default)]
    #[allow(dead_code)]
    model: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct TestCase {
    label: String,
    root_fen: String,
    moves_to_reach: Option<Vec<String>>,
    /// Stored for cross-checks against the fixture; Rust derives the side
    /// to move from the position itself, so we never read this field.
    #[allow(dead_code)]
    player: i32,
    num_searches: u32,
    c_puct: f64,
    batch_size: u32,
    expected_visit_policy: SparsePolicy,
    expected_top_action: u32,
    lookup: HashMap<String, LookupEntry>,
}

#[derive(Debug, Deserialize)]
struct SparsePolicy {
    actions: Vec<u32>,
    probs: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct LookupEntry {
    policy_actions: Vec<u16>,
    policy_priors: Vec<f32>,
    value: f32,
}

fn load() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("mcts_parity.json");
    let text = fs::read_to_string(&path).expect("read fixture");
    let f: Fixture = serde_json::from_str(&text).expect("parse fixture json");
    assert_eq!(f.schema_version, 1);
    assert_eq!(f.action_size, 4096);
    f
}

fn build_state(case: &TestCase) -> GameState {
    if let Some(moves) = &case.moves_to_reach {
        let mut s = GameState::initial();
        for uci in moves {
            s.push_uci(uci)
                .unwrap_or_else(|e| panic!("{}: push {uci}: {e}", case.label));
        }
        s
    } else {
        GameState::from_fen(&case.root_fen)
            .unwrap_or_else(|e| panic!("{}: from_fen: {e}", case.label))
    }
}

fn make_evaluator(case: &TestCase) -> FixtureEvaluator {
    let entries: HashMap<String, FixtureEntry> = case
        .lookup
        .iter()
        .map(|(k, e)| {
            (
                k.clone(),
                FixtureEntry {
                    policy_actions: e.policy_actions.clone(),
                    policy_priors: e.policy_priors.clone(),
                    value: e.value,
                },
            )
        })
        .collect();
    FixtureEvaluator::new(entries)
}

fn expected_dense(case: &TestCase, action_size: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; action_size];
    for (&a, &p) in case
        .expected_visit_policy
        .actions
        .iter()
        .zip(case.expected_visit_policy.probs.iter())
    {
        v[a as usize] = p;
    }
    v
}

fn check_case(case: &TestCase, action_size: usize) {
    let state = build_state(case);
    let evaluator = make_evaluator(case);

    let got = search(
        &evaluator,
        &state.pos,
        case.num_searches,
        case.c_puct,
        case.batch_size,
    );

    let want = expected_dense(case, action_size);
    assert_eq!(got.len(), want.len(), "[{}] policy length", case.label);

    // Mirror numpy.argmax tie-breaking: on equal values, return the
    // lowest index. Rust's `max_by` returns the last on Equal, which would
    // disagree with Python whenever the top visit count is tied across
    // several actions (common at low search depths with uniform priors).
    let got_top = {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in got.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i as u32
    };
    assert_eq!(
        got_top, case.expected_top_action,
        "[{}] top action mismatch: got {got_top}, want {}",
        case.label, case.expected_top_action,
    );

    let max_diff = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    if max_diff > TOL {
        // Print top mismatches so the failure points at a concrete action.
        let mut by_diff: Vec<_> = got
            .iter()
            .zip(want.iter())
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs(), *a, *b))
            .filter(|(_, d, _, _)| *d > 0.0)
            .collect();
        by_diff.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let head: Vec<String> = by_diff
            .into_iter()
            .take(8)
            .map(|(i, d, g, w)| format!("a={i:4} diff={d:.6} got={g:.6} want={w:.6}"))
            .collect();
        panic!(
            "[{}] policy max abs diff {max_diff} > tol {TOL}\n  {}",
            case.label,
            head.join("\n  ")
        );
    }
}

fn run_case_by_label(label: &str) {
    let f = load();
    let case = f
        .test_cases
        .iter()
        .find(|c| c.label == label)
        .unwrap_or_else(|| panic!("no test case {label}"));
    check_case(case, f.action_size);
}

#[test]
fn seq_initial_50() {
    run_case_by_label("seq_initial_50");
}

#[test]
fn seq_initial_100_cpuct1_5() {
    run_case_by_label("seq_initial_100_cpuct1.5");
}

#[test]
fn seq_midgame_white_50() {
    run_case_by_label("seq_midgame_white_50");
}

#[test]
fn seq_midgame_black_50() {
    run_case_by_label("seq_midgame_black_50");
}

#[test]
fn batch_initial_64_bs8() {
    run_case_by_label("batch_initial_64_bs8");
}

#[test]
fn batch_midgame_white_64_bs8() {
    run_case_by_label("batch_midgame_white_64_bs8");
}

#[test]
fn all_cases() {
    let f = load();
    for case in &f.test_cases {
        check_case(case, f.action_size);
    }
}
