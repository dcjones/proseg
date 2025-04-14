use num::traits::Zero;
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, Map, ParallelIterator};
use std::collections::BTreeMap;
use std::iter::{IntoIterator, Iterator};
use std::ops::Bound::{Excluded, Included};
use std::ops::{AddAssign, SubAssign};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

// This is a very specific sparse matirx implementation with the purpose
// of storing cell-by-gene count matrices that get updated during sampling.
// Towards that end we have the following goals:
// - Concurrent mutability: the matrix is sharded to allow multiple threads to
//   update entries concurrently. Shards are not random like a sharded hash table
//   but split into concurrent groups of cell indices, where cell indices are
//   spatially correlated.
// - Efficient (read-only) iteration across rows: this is needed in various places in the sampler.
pub struct SparseMat<T, J> {
    shards: Vec<Arc<RwLock<BTreeMap<(u32, J), T>>>>,
    shardsize: usize,
    m: usize,
    n: usize,
}

impl<T, J> SparseMat<T, J>
where
    T: Zero,
{
    pub fn zeros(m: usize, n: usize, shardsize: usize) -> Self {
        let nshards = (m + shardsize - 1) / shardsize;
        let mut shards = Vec::with_capacity(nshards);
        for _ in 0..nshards {
            shards.push(Arc::new(RwLock::new(BTreeMap::new())));
        }
        Self {
            shards,
            shardsize,
            m,
            n,
        }
    }

    pub fn row(&self, i: usize) -> SparseRow<T, J> {
        SparseRow::new(self, i)
    }

    pub fn rows(&self) -> SparseMatRowsIter<'_, T, J> {
        SparseMatRowsIter { mat: self }
    }

    // Zero by clearing the underlying BTreeMap
    pub fn clear(&mut self) {
        for shard in self.shards.iter_mut() {
            shard.write().unwrap().clear();
        }
    }

    // Zero by setting everything to zero without changing the data structure
    pub fn zero(&mut self) {
        for shard in self.shards.iter_mut() {
            shard
                .write()
                .unwrap()
                .values_mut()
                .for_each(|v| *v = T::zero());
        }
    }
}

pub struct SparseMatRowsIter<'a, T, J> {
    mat: &'a SparseMat<T, J>,
}

// Parallel iterator implementations are just trivial delegations.
impl<'a, T, J> ParallelIterator for SparseMatRowsIter<'a, T, J>
where
    T: Send + Sync,
    J: Send + Sync,
{
    type Item = SparseRow<T, J>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        (0..self.mat.m)
            .into_par_iter()
            .map(|i| SparseRow::new(self.mat, i))
            .drive_unindexed(consumer)
    }
}

impl<'a, T, J> IndexedParallelIterator for SparseMatRowsIter<'a, T, J>
where
    T: Send + Sync,
    J: Send + Sync,
{
    fn len(&self) -> usize {
        self.mat.m
    }

    fn drive<C>(self, consumer: C) -> C::Result
    where
        C: Consumer<Self::Item>,
    {
        (0..self.mat.m)
            .into_par_iter()
            .map(|i| SparseRow::new(self.mat, i))
            .drive(consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        (0..self.mat.m)
            .into_par_iter()
            .map(|i| SparseRow::new(self.mat, i))
            .with_producer(callback)
    }
}

pub struct SparseRow<T, J> {
    shard: Arc<RwLock<BTreeMap<(u32, J), T>>>,
    i: usize,
}

impl<T, J> SparseRow<T, J> {
    fn new(mat: &SparseMat<T, J>, i: usize) -> Self {
        let shard = mat.shards[i / mat.shardsize].clone();
        Self { shard, i }
    }

    pub fn read(&self) -> SparseRowReadLock<T, J> {
        let shard = self.shard.read().unwrap();
        SparseRowReadLock { shard, i: self.i }
    }

    pub fn write(&self) -> SparseRowWriteLock<T, J> {
        let shard = self.shard.write().unwrap();
        SparseRowWriteLock { shard, i: self.i }
    }
}

pub struct SparseRowReadLock<'a, T, J> {
    shard: RwLockReadGuard<'a, BTreeMap<(u32, J), T>>,
    i: usize,
}

impl<'a, T, J> IntoIterator for &'a SparseRowReadLock<'a, T, J>
where
    T: Clone + Copy,
    J: Clone + Copy + Ord + Zero,
{
    type Item = (J, T);
    type IntoIter = SparseRowIterator<'a, T, J>;

    fn into_iter(self) -> Self::IntoIter {
        SparseRowIterator::new(self, self.i)
    }
}

pub struct SparseRowWriteLock<'a, T, J> {
    shard: RwLockWriteGuard<'a, BTreeMap<(u32, J), T>>,
    i: usize,
}

impl<'a, T, J> SparseRowWriteLock<'a, T, J>
where
    T: AddAssign + SubAssign + Zero + Eq,
    J: Ord,
{
    pub fn sub(&mut self, j: J, delta: T) {
        let key = (self.i as u32, j);
        let count = self
            .shard
            .get_mut(&key)
            .expect("Subtracting from a 0 entry in a sparse matrix. ");
        *count -= delta;
        if *count == T::zero() {
            self.shard.remove(&key);
        }
    }

    pub fn add(&mut self, j: J, delta: T) {
        let count = self.shard.entry((self.i as u32, j)).or_insert(T::zero());
        *count += delta;
    }
}

pub struct SparseRowIterator<'a, T, J> {
    _row: &'a SparseRowReadLock<'a, T, J>,
    iter: std::collections::btree_map::Range<'a, (u32, J), T>,
}

impl<'a, T, J> SparseRowIterator<'a, T, J>
where
    J: Ord + Zero,
{
    fn new(row: &'a SparseRowReadLock<'a, T, J>, i: usize) -> Self {
        let iter = row.shard.range((
            Included((i as u32, J::zero())),
            Excluded((i as u32 + 1, J::zero())),
        ));

        SparseRowIterator { _row: row, iter }
    }
}

impl<'a, T, J> Iterator for SparseRowIterator<'a, T, J>
where
    T: Clone + Copy,
    J: Clone + Copy,
{
    type Item = (J, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&(_i, j), &v)| (j, v))
    }
}
