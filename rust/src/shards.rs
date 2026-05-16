//! `.npy` shard writer for self-play training examples.
//!
//! Writes three companion `.npy` files per iteration:
//!
//!   states.npy   shape (N, 17, 8, 8)  dtype float32
//!   policies.npy shape (N, 4096)      dtype float32
//!   values.npy   shape (N,)           dtype float32
//!
//! Python loads them with `np.load(path, mmap_mode="r")` and feeds straight
//! into `train_step` with zero parsing — the whole point of using `.npy`
//! rather than JSON or msgpack.
//!
//! Format spec: https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html
//! Header layout (version 1.0):
//!     magic       "\x93NUMPY"
//!     major u8    1
//!     minor u8    0
//!     header_len  u16 little-endian
//!     header      ASCII dict, padded with 0x20 to total-header-length % 64 == 0,
//!                 terminated with '\n'
//!     data        row-major contiguous f32 little-endian

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::action::ACTION_SIZE;
use crate::selfplay::Example;

const MAGIC: &[u8] = b"\x93NUMPY";
const ALIGN: usize = 64;

/// Write one float32 array to `path` in `.npy` 1.0 format.
fn write_npy_f32(path: &Path, shape: &[usize], data: &[f32]) -> io::Result<()> {
    let total: usize = shape.iter().product();
    assert_eq!(
        total,
        data.len(),
        "shape {shape:?} -> {total} elements but data has {} elements",
        data.len(),
    );

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(MAGIC)?;
    w.write_all(&[1, 0])?; // version 1.0

    // numpy expects a tuple even for 1-D, with a trailing comma for size 1.
    let shape_str = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => {
            let inner = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!("({inner})")
        }
    };
    let header_body = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}"
    );

    // Total prefix = MAGIC (6) + version (2) + header_len (2) = 10 bytes.
    // Header body + padding + '\n' must make the total prefix a multiple of 64.
    let prefix = MAGIC.len() + 2 + 2;
    let unpadded = prefix + header_body.len() + 1; // +1 for trailing '\n'
    let pad = (ALIGN - (unpadded % ALIGN)) % ALIGN;
    let header_total = header_body.len() + pad + 1;
    assert!(header_total <= u16::MAX as usize, "header too large");

    w.write_all(&(header_total as u16).to_le_bytes())?;
    w.write_all(header_body.as_bytes())?;
    for _ in 0..pad {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;

    // Body: contiguous little-endian f32. On all our platforms f32 is already
    // little-endian native; just splat the bytes.
    let body_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    w.write_all(body_bytes)?;
    w.flush()?;
    Ok(())
}

/// Write a batch of `Example`s into three `.npy` files under `out_dir`.
/// The directory must already exist.
pub fn write_shards(out_dir: &Path, examples: &[Example]) -> io::Result<()> {
    let n = examples.len();
    assert!(n > 0, "write_shards: empty examples list");

    let state_size = 17 * 8 * 8;
    let mut states = Vec::with_capacity(n * state_size);
    let mut policies = Vec::with_capacity(n * ACTION_SIZE);
    let mut values = Vec::with_capacity(n);

    for ex in examples {
        debug_assert_eq!(ex.state.0.len(), state_size);
        debug_assert_eq!(ex.policy.len(), ACTION_SIZE);
        states.extend_from_slice(&ex.state.0);
        policies.extend_from_slice(&ex.policy);
        values.push(ex.value);
    }

    write_npy_f32(&out_dir.join("states.npy"), &[n, 17, 8, 8], &states)?;
    write_npy_f32(&out_dir.join("policies.npy"), &[n, ACTION_SIZE], &policies)?;
    write_npy_f32(&out_dir.join("values.npy"), &[n], &values)?;
    Ok(())
}
