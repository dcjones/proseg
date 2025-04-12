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
pub struct SparseMat<T> {
    shards: Vec<Arc<RwLock<BTreeMap<(u32, u32), T>>>>,
    shardsize: usize,
    m: usize,
    n: usize,
}

impl<T> SparseMat<T> {
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

    pub fn row(&self, i: usize) -> SparseRow<T> {
        SparseRow::new(self, i)
    }

    pub fn rows(&self) -> SparseMatRowsIter<'_, T> {
        SparseMatRowsIter { mat: self }
    }

    // TODO:
    //   - Adding and subtracting at specific indices
    //   - shape()
    //   - summing on axes (column iterators would be much more paainful, we'd probably do it one shard at a time)
}

struct SparseMatRowsIter<'a, T> {
    mat: &'a SparseMat<T>,
}

// Parallel iterator implementations are just trivial delegations.
impl<'a, T> ParallelIterator for SparseMatRowsIter<'a, T>
where
    T: Send + Sync,
{
    type Item = SparseRow<T>;

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

impl<'a, T> IndexedParallelIterator for SparseMatRowsIter<'a, T>
where
    T: Send + Sync,
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

pub struct SparseRow<T> {
    shard: Arc<RwLock<BTreeMap<(u32, u32), T>>>,
    i: usize,
}

impl<T> SparseRow<T> {
    fn new(mat: &SparseMat<T>, i: usize) -> Self {
        let shard = mat.shards[i / mat.shardsize].clone();
        Self { shard, i }
    }

    fn read(&self) -> SparseRowReadLock<T> {
        let shard = self.shard.read().unwrap();
        SparseRowReadLock { shard, i: self.i }
    }

    pub fn write(&self) -> SparseRowWriteLock<T> {
        let shard = self.shard.write().unwrap();
        SparseRowWriteLock { shard, i: self.i }
    }
}

struct SparseRowReadLock<'a, T> {
    shard: RwLockReadGuard<'a, BTreeMap<(u32, u32), T>>,
    i: usize,
}

impl<'a, T> IntoIterator for &'a SparseRowReadLock<'a, T>
where
    T: Clone + Copy,
{
    type Item = (u32, T);
    type IntoIter = SparseRowIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        SparseRowIterator::new(self, self.i)
    }
}

pub struct SparseRowWriteLock<'a, T> {
    shard: RwLockWriteGuard<'a, BTreeMap<(u32, u32), T>>,
    i: usize,
}

impl<'a, T> SparseRowWriteLock<'a, T>
where
    T: AddAssign + SubAssign + Zero + Eq,
{
    pub fn sub(&mut self, j: usize, delta: T) {
        let key = (self.i as u32, j as u32);
        let count = self
            .shard
            .get_mut(&key)
            .expect("Subtracting from a 0 entry in a sparse matrix. ");
        *count -= delta;
        if *count == T::zero() {
            self.shard.remove(&key);
        }
    }

    pub fn add(&mut self, j: usize, delta: T) {
        let count = self
            .shard
            .entry((self.i as u32, j as u32))
            .or_insert(T::zero());
        *count += delta;
    }
}

struct SparseRowIterator<'a, T> {
    _row: &'a SparseRowReadLock<'a, T>,
    iter: std::collections::btree_map::Range<'a, (u32, u32), T>,
}

impl<'a, T> SparseRowIterator<'a, T> {
    fn new(row: &'a SparseRowReadLock<'a, T>, i: usize) -> Self {
        let iter = row
            .shard
            .range((Included((i as u32, 0_u32)), Excluded((i as u32 + 1, 0_u32))));

        SparseRowIterator { _row: row, iter }
    }
}

impl<'a, T> Iterator for SparseRowIterator<'a, T>
where
    T: Clone + Copy,
{
    type Item = (u32, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&(_i, j), &v)| (j, v))
    }
}
