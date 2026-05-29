use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use thread_local::ThreadLocal;

use super::TransitionMatRowKey;

/// Sparse transition-count matrix backed by per-row HashMaps with thread-local
/// accumulation buffers to avoid write-lock contention during parallel sampling.
///
/// Usage pattern in a parallel section:
///   - Call `add_local` (no shared-row locking) from rayon threads.
///   - After the parallel section, call `flush_locals` (takes `&mut self`) once
///     from the main thread to merge thread-local buffers into the shared rows.
pub struct TransitionMat {
    rows: Vec<RwLock<HashMap<u64, u32>>>,
    // Per-rayon-thread accumulation buffer.
    // RefCell is Send (not Sync), but ThreadLocal::iter_mut only needs Send.
    local_buf: ThreadLocal<RefCell<HashMap<u64, u32>>>,
}

// Key layout for thread-local buffer: 21 bits each for src, gene, dest.
// Supports up to ~2M cells and ~2M genes.
const BITS: u64 = 21;
const MASK: u64 = (1 << BITS) - 1;

#[inline]
fn encode_local(src: u32, gene: u32, dest_cell: u32) -> u64 {
    ((src as u64) << (2 * BITS)) | ((gene as u64) << BITS) | (dest_cell as u64)
}

/// Pack (gene, dest_cell) into a u64 key for a shared row.
#[inline]
fn encode_row(gene: u32, dest_cell: u32) -> u64 {
    (gene as u64) << 32 | dest_cell as u64
}

#[inline]
fn decode_row(key: u64) -> TransitionMatRowKey {
    TransitionMatRowKey {
        gene: (key >> 32) as u32,
        dest_cell: key as u32,
    }
}

impl TransitionMat {
    pub fn new(nrows: usize) -> Self {
        let rows = (0..nrows)
            .map(|_| RwLock::new(HashMap::new()))
            .collect();
        Self {
            rows,
            local_buf: ThreadLocal::new(),
        }
    }

    /// Accumulate into the calling thread's local buffer — no shared locking.
    /// Must be followed by `flush_locals` after the parallel section.
    #[inline]
    pub fn add_local(&self, src: usize, gene: u32, dest_cell: u32) {
        let buf = self.local_buf.get_or(|| RefCell::new(HashMap::new()));
        *buf.borrow_mut()
            .entry(encode_local(src as u32, gene, dest_cell))
            .or_insert(0) += 1u32;
    }

    /// Merge all thread-local buffers into the shared rows, then clear them.
    /// Call from a single thread after the parallel section completes.
    pub fn flush_locals(&mut self) {
        for buf_cell in self.local_buf.iter_mut() {
            let mut buf = buf_cell.borrow_mut();
            for (k, count) in buf.drain() {
                let src = ((k >> (2 * BITS)) & MASK) as usize;
                let gene = ((k >> BITS) & MASK) as u32;
                let dest = (k & MASK) as u32;
                *self.rows[src]
                    .write()
                    .entry(encode_row(gene, dest))
                    .or_insert(0) += count;
            }
        }
    }

    /// Return all non-zero entries for row `i`, sorted by (gene, dest_cell).
    pub fn iter_row_sorted(&self, i: usize) -> Vec<(TransitionMatRowKey, u32)> {
        let guard = self.rows[i].read();
        let mut entries: Vec<(TransitionMatRowKey, u32)> = guard
            .iter()
            .map(|(&k, &v)| (decode_row(k), v))
            .collect();
        entries.sort_unstable_by_key(|(key, _)| *key);
        entries
    }
}
