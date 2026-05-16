"""Read self-play training shards written by the Rust ``selfplay`` binary.

Phase 4 in ``design/RUST_PORT_PLAN.md``: Rust writes three companion ``.npy``
files per iteration; Python's ``train_step`` already takes numpy arrays so
the integration on the Python side is just an ``np.load`` triple.

Layout written by ``rust/src/shards.rs::write_shards``:

    states.npy    shape (N, 17, 8, 8)   dtype float32
    policies.npy  shape (N, 4096)       dtype float32
    values.npy    shape (N,)            dtype float32
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Shard:
    """A loaded self-play shard. Arrays are read-only views (mmap-backed)
    unless you copy them.
    """
    states: np.ndarray   # (N, 17, 8, 8) float32
    policies: np.ndarray  # (N, 4096) float32
    values: np.ndarray   # (N,) float32

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def as_train_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Triple in the order ``train_step`` consumes (encoded_states,
        policies, outcomes)."""
        return self.states, self.policies, self.values


def load_shard(shard_dir: str, *, mmap: bool = True) -> Shard:
    """Load one shard directory written by the Rust binary.

    Args:
        shard_dir: directory containing states.npy / policies.npy / values.npy
        mmap: if True (default), arrays are mmap'd read-only. Set False if
              you intend to permute or shuffle in place — copy first instead.
    """
    mode: Optional[str] = "r" if mmap else None
    states = np.load(os.path.join(shard_dir, "states.npy"), mmap_mode=mode)
    policies = np.load(os.path.join(shard_dir, "policies.npy"), mmap_mode=mode)
    values = np.load(os.path.join(shard_dir, "values.npy"), mmap_mode=mode)

    # Sanity checks — fail loudly if Rust wrote something unexpected.
    n = states.shape[0]
    assert states.shape == (n, 17, 8, 8), f"states shape {states.shape}"
    assert policies.shape == (n, 4096), f"policies shape {policies.shape}"
    assert values.shape == (n,), f"values shape {values.shape}"
    assert states.dtype == np.float32
    assert policies.dtype == np.float32
    assert values.dtype == np.float32

    return Shard(states=states, policies=policies, values=values)
