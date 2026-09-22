use ahash::AHashMap as HashMap;
use parking_lot::Mutex;
use rayon::prelude::*;

use super::TransitionMatRowKey;

/// Sparse transition-count matrix, indexed as (src state, (gene, dest state)),
/// backed by per-row, individually locked HashMaps.
///
/// During sampling only off-diagonal transitions (src != dest) are recorded,
/// via `add` directly into the shared rows. These are a minority of
/// transcripts, so row-lock contention is low, and it avoids having to merge
/// per-thread buffers after every sample. The diagonal is instead
/// reconstructed once, by `finalize`, from the fixed source (reported) state
/// of each transcript and the number of recorded samples.
pub struct TransitionMat {
    rows: Vec<Mutex<HashMap<u64, u32>>>,
    // Number of recorded samples, i.e. the number of times every transcript
    // contributed one transition (possibly src -> src).
    nsamples: u32,
    finalized: bool,
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
        let rows = (0..nrows).map(|_| Mutex::new(HashMap::new())).collect();
        Self {
            rows,
            nsamples: 0,
            finalized: false,
        }
    }

    pub fn nrows(&self) -> usize {
        self.rows.len()
    }

    /// Record one transition. Diagonal (src == dest_cell) transitions are
    /// implicit and ignored here; see `finalize`.
    #[inline]
    pub fn add(&self, src: usize, gene: u32, dest_cell: u32) {
        if src as u32 == dest_cell {
            return;
        }
        *self.rows[src]
            .lock()
            .entry(encode_row(gene, dest_cell))
            .or_insert(0) += 1;
    }

    /// Mark the end of one recorded sample, during which `add` was called for
    /// every transcript.
    pub fn finish_sample(&mut self) {
        assert!(!self.finalized);
        self.nsamples += 1;
    }

    /// Fill in the diagonal. `src_states` yields the (src state, gene) of every
    /// transcript, which must have been fixed across all recorded samples.
    /// Each transcript made exactly one transition per sample, so the diagonal
    /// entry for (s, g) is nsamples * |{transcripts in (s, g)}| minus the
    /// recorded off-diagonal transitions out of (s, g).
    pub fn finalize<I>(&mut self, src_states: I)
    where
        I: ParallelIterator<Item = (u32, u32)>,
    {
        if self.finalized {
            return;
        }
        self.finalized = true;
        if self.nsamples == 0 {
            return;
        }

        let mut keys: Vec<u64> = src_states
            .map(|(src, gene)| (src as u64) << 32 | gene as u64)
            .collect();
        keys.par_sort_unstable();

        // Split into runs of equal src state, one per row.
        let mut row_runs: Vec<(usize, &[u64])> = Vec::new();
        let mut rest = &keys[..];
        while let Some(&first) = rest.first() {
            let src = first >> 32;
            let len = rest.partition_point(|k| k >> 32 == src);
            row_runs.push((src as usize, &rest[..len]));
            rest = &rest[len..];
        }

        let nsamples = self.nsamples;
        let rows = &self.rows;
        row_runs.into_par_iter().for_each(|(src, run)| {
            let mut row = rows[src].lock();

            let mut offdiag: HashMap<u32, u32> = HashMap::new();
            for (&key, &count) in row.iter() {
                *offdiag.entry((key >> 32) as u32).or_insert(0) += count;
            }

            let mut i = 0;
            while i < run.len() {
                let gene = run[i] as u32;
                let n = run[i..].partition_point(|&k| k as u32 == gene) as u32;
                let off = offdiag.get(&gene).copied().unwrap_or(0);
                let diag = nsamples * n - off;
                if diag > 0 {
                    row.insert(encode_row(gene, src as u32), diag);
                }
                i += n as usize;
            }
        });
    }

    /// Return all non-zero entries for row `i`, sorted by (gene, dest_cell).
    pub fn iter_row_sorted(&self, i: usize) -> Vec<(TransitionMatRowKey, u32)> {
        let guard = self.rows[i].lock();
        let mut entries: Vec<(TransitionMatRowKey, u32)> = guard
            .iter()
            .map(|(&k, &v)| (decode_row(k), v))
            .collect();
        entries.sort_unstable_by_key(|(key, _)| *key);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagonal_is_reconstructed() {
        // States 0, 1, and 2 (background). Transcripts as (src state, gene).
        let transcripts = [(0, 0), (0, 0), (0, 1), (1, 0), (2, 1)];
        // Per sample, the dest state of each transcript.
        let samples = [[0, 1, 0, 1, 1], [2, 0, 0, 0, 2]];

        let mut direct: HashMap<(u32, u32, u32), u32> = HashMap::new();
        let mut mat = TransitionMat::new(3);
        for dests in &samples {
            for (&(src, gene), &dest) in transcripts.iter().zip(dests) {
                *direct.entry((src, gene, dest)).or_insert(0) += 1;
                mat.add(src as usize, gene, dest);
            }
            mat.finish_sample();
        }
        mat.finalize(transcripts.par_iter().copied());

        let mut reconstructed = HashMap::new();
        for src in 0..mat.nrows() {
            for (key, count) in mat.iter_row_sorted(src) {
                reconstructed.insert((src as u32, key.gene, key.dest_cell), count);
            }
        }
        assert_eq!(direct, reconstructed);
    }
}
