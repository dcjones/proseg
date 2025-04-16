use num::traits::{One, Zero};
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, Map, ParallelIterator};
use std::collections::BTreeMap;
use std::iter::{IntoIterator, Iterator, Sum};
use std::ops::Bound::{Excluded, Included};
use std::ops::{Add, AddAssign, SubAssign};
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
    n: J,
}

impl<T, J> SparseMat<T, J>
where
    T: Zero + Copy + AddAssign + Add + Sum,
    J: Copy,
{
    pub fn zeros(m: usize, n: J, shardsize: usize) -> Self {
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

    pub fn shape(&self) -> (usize, J) {
        (self.m, self.n)
    }

    pub fn row(&self, i: usize) -> SparseRow<T, J> {
        SparseRow::new(self, i)
    }

    pub fn par_rows(&self) -> SparseMatRowsParIter<'_, T, J> {
        SparseMatRowsParIter { mat: self }
    }

    pub fn rows(&self) -> SparseMatRowsIter<'_, T, J> {
        SparseMatRowsIter { mat: self, i: 0 }
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

    pub fn sum(&self) -> T {
        let mut accum = T::zero();
        for shard in &self.shards {
            accum += shard.read().unwrap().values().cloned().sum();
        }
        accum
    }
}

pub struct SparseMatRowsIter<'a, T, J> {
    i: usize,
    mat: &'a SparseMat<T, J>,
}

impl<'a, T, J> Iterator for SparseMatRowsIter<'a, T, J>
where
    J: Copy,
{
    type Item = SparseRow<T, J>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.mat.m {
            let item = SparseRow::new(self.mat, self.i);
            self.i += 1;
            Some(item)
        } else {
            None
        }
    }
}

pub struct SparseMatRowsParIter<'a, T, J> {
    mat: &'a SparseMat<T, J>,
}

// Parallel iterator implementations are just trivial delegations.
impl<'a, T, J> ParallelIterator for SparseMatRowsParIter<'a, T, J>
where
    T: Send + Sync,
    J: Send + Sync + Copy,
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

impl<'a, T, J> IndexedParallelIterator for SparseMatRowsParIter<'a, T, J>
where
    T: Send + Sync,
    J: Send + Sync + Copy,
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
    n: J,
    i: usize,
}

impl<T, J> SparseRow<T, J>
where
    J: Copy,
{
    fn new(mat: &SparseMat<T, J>, i: usize) -> Self {
        let shard = mat.shards[i / mat.shardsize].clone();
        Self { shard, n: mat.n, i }
    }

    pub fn read(&self) -> SparseRowReadLock<T, J> {
        let shard = self.shard.read().unwrap();
        SparseRowReadLock {
            shard,
            n: self.n,
            i: self.i,
        }
    }

    pub fn write(&self) -> SparseRowWriteLock<T, J> {
        let shard = self.shard.write().unwrap();
        SparseRowWriteLock {
            shard,
            n: self.n,
            i: self.i,
        }
    }
}

pub struct SparseRowReadLock<'a, T, J> {
    shard: RwLockReadGuard<'a, BTreeMap<(u32, J), T>>,
    n: J,
    i: usize,
}

impl<'a, T, J> SparseRowReadLock<'a, T, J>
where
    T: Clone + Copy,
    J: Clone + Copy + Ord + Zero,
{
    pub fn iter_nonzeros(&'a self) -> SparseRowNonzeroIterator<'a, T, J> {
        SparseRowNonzeroIterator::new(self, self.i)
    }

    pub fn iter_nonzeros_from(&'a self, from: J) -> SparseRowNonzeroIterator<'a, T, J> {
        SparseRowNonzeroIterator::new_from(self, self.i, from)
    }

    pub fn iter_nonzeros_to(&'a self, to: J) -> SparseRowNonzeroIterator<'a, T, J> {
        SparseRowNonzeroIterator::new_to(self, self.i, to)
    }

    pub fn iter(&'a self) -> SparseRowIterator<'a, T, J> {
        SparseRowIterator::new(self, self.i)
    }
}

// impl<'a, T, J> IntoIterator for &'a SparseRowReadLock<'a, T, J>
// where
//     T: Clone + Copy,
//     J: Clone + Copy + Ord + Zero,
// {
//     type Item = (J, T);
//     type IntoIter = SparseRowNonzeroIterator<'a, T, J>;

//     fn into_iter(self) -> Self::IntoIter {
//         SparseRowNonzeroIterator::new(self, self.i)
//     }
// }

pub struct SparseRowWriteLock<'a, T, J> {
    shard: RwLockWriteGuard<'a, BTreeMap<(u32, J), T>>,
    n: J,
    i: usize,
}

impl<'a, T, J> SparseRowWriteLock<'a, T, J>
where
    T: AddAssign + SubAssign + Zero + Eq,
    J: Ord,
{
    pub fn sub(&mut self, j: J, delta: T) {
        assert!(j < self.n);
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
        assert!(j < self.n);
        let count = self.shard.entry((self.i as u32, j)).or_insert(T::zero());
        *count += delta;
    }
}

pub struct SparseRowNonzeroIterator<'a, T, J> {
    _row: &'a SparseRowReadLock<'a, T, J>,
    iter: std::collections::btree_map::Range<'a, (u32, J), T>,
}

impl<'a, T, J> SparseRowNonzeroIterator<'a, T, J>
where
    J: Ord + Zero,
{
    fn new(row: &'a SparseRowReadLock<'a, T, J>, i: usize) -> Self {
        let iter = row.shard.range((
            Included((i as u32, J::zero())),
            Excluded((i as u32 + 1, J::zero())),
        ));

        SparseRowNonzeroIterator { _row: row, iter }
    }

    fn new_from(row: &'a SparseRowReadLock<'a, T, J>, i: usize, from: J) -> Self {
        let iter = row.shard.range((
            Included((i as u32, from)),
            Excluded((i as u32 + 1, J::zero())),
        ));

        SparseRowNonzeroIterator { _row: row, iter }
    }

    fn new_to(row: &'a SparseRowReadLock<'a, T, J>, i: usize, to: J) -> Self {
        let iter = row.shard.range((
            Included((i as u32, J::zero())),
            Excluded((i as u32 + 1, to)),
        ));

        SparseRowNonzeroIterator { _row: row, iter }
    }
}

impl<'a, T, J> Iterator for SparseRowNonzeroIterator<'a, T, J>
where
    T: Clone + Copy,
    J: Clone + Copy,
{
    type Item = (J, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&(_i, j), &v)| (j, v))
    }
}

pub struct SparseRowIterator<'a, T, J> {
    _row: &'a SparseRowReadLock<'a, T, J>,
    j: J,
    n: J,
    buf: Option<(J, T)>,
    iter: std::collections::btree_map::Range<'a, (u32, J), T>,
}

impl<'a, T, J> SparseRowIterator<'a, T, J>
where
    J: Ord + Zero + Copy,
{
    fn new(row: &'a SparseRowReadLock<'a, T, J>, i: usize) -> Self {
        let iter = row.shard.range((
            Included((i as u32, J::zero())),
            Excluded((i as u32 + 1, J::zero())),
        ));

        SparseRowIterator {
            _row: row,
            j: J::zero(),
            n: row.n,
            buf: None,
            iter,
        }
    }
}

impl<'a, T, J> Iterator for SparseRowIterator<'a, T, J>
where
    T: Clone + Copy + Zero,
    J: Clone + Copy + Eq + One + AddAssign + Ord,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.j == self.n {
            None
        } else if let Some((j_buf, item_buf)) = self.buf {
            let value = if self.j < j_buf {
                T::zero()
            } else if self.j == j_buf {
                self.buf = self.iter.next().map(|(&(_i, j), &v)| (j, v));
                item_buf
            } else {
                panic!("Broken iterator assumptions.");
            };

            self.j += J::one();
            Some(value)
        } else {
            self.buf = self.iter.next().map(|(&(_i, j), &v)| (j, v));
            self.next()
        }
    }
}
