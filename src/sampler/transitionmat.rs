use parking_lot::RwLock;
use std::cell::RefCell;
use std::collections::HashMap;
use thread_local::ThreadLocal;

/// Sparse state→state transition-count matrix backed by per-row HashMaps with
/// thread-local accumulation buffers to avoid write-lock contention during
/// parallel sampling.
///
/// Rows and columns are cell indexes, with one extra index (`ncells`) standing
/// for the background state, so both cell↔cell and cell↔background transitions
/// are recorded in the same structure.
///
/// Usage pattern in a parallel section:
///   - Call `add_local` (no shared-row locking) from rayon threads.
///   - After the parallel section, call `flush_locals` (takes `&mut self`) once
///     from the main thread to merge thread-local buffers into the shared rows.
pub struct TransitionMat {
    rows: Vec<RwLock<HashMap<u32, u32>>>,
    // Per-rayon-thread accumulation buffer.
    // RefCell is Send (not Sync), but ThreadLocal::iter_mut only needs Send.
    local_buf: ThreadLocal<RefCell<HashMap<u64, u32>>>,
}

/// Pack (src, dest) into a u64 key for the thread-local buffer.
#[inline]
fn encode_local(src: u32, dest: u32) -> u64 {
    ((src as u64) << 32) | (dest as u64)
}

impl TransitionMat {
    pub fn new(nrows: usize) -> Self {
        let rows = (0..nrows).map(|_| RwLock::new(HashMap::new())).collect();
        Self {
            rows,
            local_buf: ThreadLocal::new(),
        }
    }

    /// Accumulate into the calling thread's local buffer — no shared locking.
    /// Must be followed by `flush_locals` after the parallel section.
    #[inline]
    pub fn add_local(&self, src: u32, dest: u32) {
        let buf = self.local_buf.get_or(|| RefCell::new(HashMap::new()));
        *buf.borrow_mut().entry(encode_local(src, dest)).or_insert(0) += 1u32;
    }

    /// Merge all thread-local buffers into the shared rows, then clear them.
    /// Call from a single thread after the parallel section completes.
    pub fn flush_locals(&mut self) {
        for buf_cell in self.local_buf.iter_mut() {
            let mut buf = buf_cell.borrow_mut();
            for (k, count) in buf.drain() {
                let src = (k >> 32) as usize;
                let dest = k as u32;
                *self.rows[src].write().entry(dest).or_insert(0) += count;
            }
        }
    }

    /// Return all non-zero entries for row `i` as (dest, count), sorted by dest.
    pub fn iter_row_sorted(&self, i: usize) -> Vec<(u32, u32)> {
        let guard = self.rows[i].read();
        let mut entries: Vec<(u32, u32)> = guard.iter().map(|(&k, &v)| (k, v)).collect();
        entries.sort_unstable_by_key(|(dest, _)| *dest);
        entries
    }
}
