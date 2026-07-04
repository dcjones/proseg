use std::sync::atomic::{AtomicU32, Ordering};

// A fixed-length vector of u32 counters supporting concurrent `add`/`sub` from
// many threads via atomics.
//
// This was previously sharded behind an `Arc<RwLock<Vec<u32>>>` per shard, but
// every mutation took a write lock, which was expensive even uncontended. Atomic
// counters give lock-free concurrent updates with no sharding needed. Reads after
// a rayon join see all prior writes (the join is the happens-before barrier), so
// `Relaxed` ordering suffices for these accumulate-then-read count vectors.
pub struct AtomicCountVec {
    data: Vec<AtomicU32>,
}

impl AtomicCountVec {
    pub fn zeros(n: usize) -> Self {
        let data = (0..n).map(|_| AtomicU32::new(0)).collect();
        Self { data }
    }

    pub fn get(&self, index: usize) -> u32 {
        self.data[index].load(Ordering::Relaxed)
    }

    pub fn add(&self, index: usize, value: u32) {
        self.data[index].fetch_add(value, Ordering::Relaxed);
    }

    pub fn sub(&self, index: usize, value: u32) {
        self.data[index].fetch_sub(value, Ordering::Relaxed);
    }

    // Apply a net `+add - sub` in a single atomic operation. Wrapping arithmetic
    // yields the same result as separate add/sub because the counter's true value
    // stays non-negative.
    pub fn add_sub(&self, index: usize, add: u32, sub: u32) {
        self.data[index].fetch_add(add.wrapping_sub(sub), Ordering::Relaxed);
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.data.iter().map(|x| x.load(Ordering::Relaxed))
    }

    pub fn zero(&mut self) {
        for x in &mut self.data {
            *x.get_mut() = 0;
        }
    }
}

impl PartialEq for AtomicCountVec {
    fn eq(&self, other: &AtomicCountVec) -> bool {
        self.data.len() == other.data.len() && self.iter().eq(other.iter())
    }
}
