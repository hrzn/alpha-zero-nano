//! Write a small batch of examples to `.npy` shards, then parse the files
//! byte-for-byte and verify they decode back to the same numbers.
//!
//! We deliberately do NOT use a `.npy` parser crate — the goal here is to
//! prove our hand-rolled `npy` writer produces a file numpy will accept,
//! which means replicating numpy's exact header layout. A roll-your-own
//! decoder catches header alignment / shape / dtype mistakes that a real
//! numpy load would also reject.

use std::fs;
use std::io::Read;
use std::path::Path;

use alpha_zero_nano::encoding::{EncodedState, TENSOR_LEN};
use alpha_zero_nano::selfplay::Example;
use alpha_zero_nano::shards::write_shards;

fn read_npy_f32(path: &Path) -> (Vec<usize>, Vec<f32>) {
    let mut f = fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    assert!(buf.starts_with(b"\x93NUMPY"), "missing NUMPY magic");
    assert_eq!(buf[6], 1, "major version");
    assert_eq!(buf[7], 0, "minor version");
    let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header = std::str::from_utf8(&buf[10..10 + hlen]).unwrap();
    assert!(
        header.contains("'descr': '<f4'"),
        "expected f32 little-endian, got header: {header}"
    );
    assert!(
        header.contains("'fortran_order': False"),
        "expected C order"
    );
    // Pull out the shape tuple, e.g. "'shape': (3, 17, 8, 8),"
    let shape_start = header.find("'shape': (").expect("shape key") + "'shape': (".len();
    let shape_end = shape_start + header[shape_start..].find(')').unwrap();
    let shape: Vec<usize> = header[shape_start..shape_end]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap())
        .collect();

    let total: usize = shape.iter().product();
    let body = &buf[10 + hlen..];
    assert_eq!(body.len(), total * 4, "body length");
    let mut data = Vec::with_capacity(total);
    for chunk in body.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    (shape, data)
}

fn make_example(state_seed: f32, policy_idx: usize, value: f32) -> Example {
    let mut state = vec![0.0f32; TENSOR_LEN];
    // sprinkle a non-trivial pattern so byte-level slips show up.
    for (i, v) in state.iter_mut().enumerate() {
        *v = state_seed + (i as f32) * 1e-3;
    }
    let mut policy = vec![0.0f32; 4096];
    policy[policy_idx] = 1.0;
    Example {
        state: EncodedState(state),
        policy,
        value,
    }
}

#[test]
fn shard_roundtrip_three_examples() {
    let tmp = tempdir_in_target();
    let examples = vec![
        make_example(0.0, 80, 1.0),
        make_example(0.5, 405, -1.0),
        make_example(-0.25, 528, 0.0),
    ];
    write_shards(&tmp, &examples).expect("write_shards");

    let (states_shape, states) = read_npy_f32(&tmp.join("states.npy"));
    let (policies_shape, policies) = read_npy_f32(&tmp.join("policies.npy"));
    let (values_shape, values) = read_npy_f32(&tmp.join("values.npy"));

    assert_eq!(states_shape, vec![3, 17, 8, 8]);
    assert_eq!(policies_shape, vec![3, 4096]);
    assert_eq!(values_shape, vec![3]);

    // Compare per-example data byte-for-byte.
    for (i, ex) in examples.iter().enumerate() {
        let start = i * TENSOR_LEN;
        assert_eq!(&states[start..start + TENSOR_LEN], &ex.state.0[..]);
        let pstart = i * 4096;
        assert_eq!(&policies[pstart..pstart + 4096], &ex.policy[..]);
        assert_eq!(values[i], ex.value);
    }

    // Best-effort cleanup; doesn't matter if it fails on Windows-style locks.
    let _ = fs::remove_dir_all(&tmp);
}

fn tempdir_in_target() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-shards");
    fs::create_dir_all(&base).unwrap();
    let unique = format!("roundtrip-{}", std::process::id());
    let p = base.join(unique);
    if p.exists() {
        fs::remove_dir_all(&p).unwrap();
    }
    fs::create_dir_all(&p).unwrap();
    p
}
