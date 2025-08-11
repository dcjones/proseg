// use super::RAYON_CELL_MIN_LEN;
use num::traits::AsPrimitive;
use num::traits::Zero;
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::iter::{Iterator, Sum};
use std::ops::Bound::{Excluded, Included};
use std::ops::{Add, AddAssign, SubAssign};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

type SparseMatShard<T, J> = RwLock<BTreeMap<(u32, J), T>>;

// This is a very specific sparse matirx implementation with the purpose
// of storing cell-by-gene count matrices that get updated during sampling.
// Towards that end we have the following goals:
// - Concurrent mutability: the matrix is sharded to allow multiple threads to
//   update entries concurrently. Shards are not random like a sharded hash table
//   but split into concurrent groups of cell indices, where cell indices are
//   spatially correlated.
// - Efficient (read-only) iteration across rows: this is needed in various places in the sampler.
pub struct SparseMat<T, J> {
    shards: Vec<Arc<SparseMatShard<T, J>>>,
    shardsize: usize,
    pub m: usize,
    pub j_bound: J,
    pub n: J,
}

impl<T, J> SparseMat<T, J>
where
    T: Copy,
    J: Copy + Increment,
{
    pub fn zeros(m: usize, j_bound: J, shardsize: usize) -> Self {
        let nshards = m.div_ceil(shardsize);
        let mut shards = Vec::with_capacity(nshards);
        for _ in 0..nshards {
            shards.push(Arc::new(RwLock::new(BTreeMap::new())));
        }
        Self {
            shards,
            shardsize,
            m,
            j_bound,
            n: j_bound.inc(j_bound),
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
}

impl<T, J> SparseMat<T, J>
where
    T: Send + Sync + Copy + Zero,
    J: Send + Sync + Copy,
{
    pub fn clear(&mut self) {
        self.shards.par_iter().for_each(|shard| {
            shard.write().unwrap().clear();
        });
    }

    pub fn zero(&mut self) {
        self.shards.par_iter().for_each(|shard| {
            shard
                .write()
                .unwrap()
                .values_mut()
                .for_each(|v| *v = T::zero());
        });
    }
}

impl<T, J> SparseMat<T, J>
where
    T: Zero + Copy + AddAssign + Add + Sum + AsPrimitive<u64>,
    J: Copy,
{
    pub fn sum(&self) -> u64 {
        let mut accum = 0;
        for shard in &self.shards {
            accum += shard
                .read()
                .unwrap()
                .values()
                .cloned()
                .map(|x| x.as_())
                .sum::<u64>();
        }

        accum
    }
}

impl<T, J> PartialEq for SparseMat<T, J>
where
    T: Zero + Copy + AddAssign + Add + Sum + AsPrimitive<u64> + PartialEq + Debug,
    J: Clone + Copy + Ord + Zero + AddAssign + Debug + Increment,
{
    fn eq(&self, other: &SparseMat<T, J>) -> bool {
        dbg!(self.m, other.m);
        dbg!(self.n, other.n);

        if self.m != other.m || self.n != other.n {
            return false;
        }

        dbg!(self.sum());
        dbg!(other.sum());

        for (a_i, b_i) in self.rows().zip(other.rows()) {
            for (a_ij, b_ij) in a_i.read().iter().zip(b_i.read().iter()) {
                if a_ij != b_ij {
                    dbg!((a_ij, b_ij));
                    return false;
                }
            }
        }

        true
    }
}

pub struct SparseMatRowsIter<'a, T, J> {
    i: usize,
    mat: &'a SparseMat<T, J>,
}

impl<T, J> Iterator for SparseMatRowsIter<'_, T, J>
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
impl<T, J> ParallelIterator for SparseMatRowsParIter<'_, T, J>
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

impl<T, J> IndexedParallelIterator for SparseMatRowsParIter<'_, T, J>
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

#[derive(Debug)]
pub struct SparseRow<T, J> {
    shard: Arc<RwLock<BTreeMap<(u32, J), T>>>,
    pub j_bound: J,
    pub i: usize,
}

impl<T, J> SparseRow<T, J>
where
    J: Copy,
{
    fn new(mat: &SparseMat<T, J>, i: usize) -> Self {
        let shard = mat.shards[i / mat.shardsize].clone();
        Self {
            shard,
            j_bound: mat.j_bound,
            i,
        }
    }

    pub fn read(&self) -> SparseRowReadLock<T, J> {
        let shard = self.shard.read().unwrap();
        SparseRowReadLock {
            shard,
            j_bound: self.j_bound,
            i: self.i,
        }
    }

    pub fn write(&self) -> SparseRowWriteLock<T, J> {
        let shard = self.shard.write().unwrap();
        SparseRowWriteLock {
            shard,
            j_bound: self.j_bound,
            i: self.i,
        }
    }
}

pub struct SparseRowReadLock<'a, T, J> {
    shard: RwLockReadGuard<'a, BTreeMap<(u32, J), T>>,
    j_bound: J,
    i: usize,
}

impl<'a, T, J> SparseRowReadLock<'a, T, J>
where
    T: Clone + Copy,
    J: Clone + Increment + Copy + Ord + Zero + Debug,
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

impl<T, J> SparseRowReadLock<'_, T, J>
where
    T: Clone + Copy + Zero,
    J: Clone + Increment + Copy + Ord + Zero + Debug,
{
    pub fn get(&self, j: J) -> T {
        assert!(j < self.j_bound.inc(self.j_bound));
        self.shard
            .get(&(self.i as u32, j))
            .copied()
            .unwrap_or(T::zero())
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
    pub shard: RwLockWriteGuard<'a, BTreeMap<(u32, J), T>>,
    pub j_bound: J,
    pub i: usize,
}

impl<T, J> SparseRowWriteLock<'_, T, J>
where
    T: AddAssign + SubAssign + Zero + Eq + PartialOrd,
    J: Ord + Increment + Copy + Debug + Zero,
{
    pub fn sub(&mut self, j: J, delta: T) {
        assert!(j < self.j_bound.inc(self.j_bound));
        let key = (self.i as u32, j);
        let count = self
            .shard
            .get_mut(&key)
            .expect("Subtracting from a 0 entry in a sparse matrix.");
        assert!(*count >= delta);
        *count -= delta;
        if *count == T::zero() {
            self.shard.remove(&key);
        }
    }

    pub fn add(&mut self, j: J, delta: T) {
        assert!(j < self.j_bound.inc(self.j_bound));
        let count = self.shard.entry((self.i as u32, j)).or_insert(T::zero());
        *count += delta;
    }
}

impl<T, J> SparseRowWriteLock<'_, T, J>
where
    J: Ord + Increment + Copy + Debug + Zero,
{
    pub fn update<F, G>(&mut self, j: J, insert_fn: F, update_fn: G)
    where
        F: FnOnce() -> T,
        G: FnOnce(&mut T),
    {
        assert!(j < self.j_bound.inc(self.j_bound));
        let key = (self.i as u32, j);
        let count = self.shard.entry(key).or_insert_with(insert_fn);
        update_fn(count);
    }

    pub fn update_if_present<G>(&mut self, j: J, update_fn: G)
    where
        G: FnOnce(&mut T),
    {
        assert!(j < self.j_bound.inc(self.j_bound));
        let key = (self.i as u32, j);
        if let Some(count) = self.shard.get_mut(&key) {
            update_fn(count);
        }
    }
}

impl<T, J> SparseRowWriteLock<'_, T, J>
where
    J: Ord + Increment + Copy + Debug + Zero,
{
    pub fn iter_nonzeros_mut(&mut self) -> SparseRowMutNonzeroIterator<'_, T, J> {
        SparseRowMutNonzeroIterator::new(self, self.i)
    }
}

pub struct SparseRowNonzeroIterator<'a, T, J> {
    _row: &'a SparseRowReadLock<'a, T, J>,
    iter: std::collections::btree_map::Range<'a, (u32, J), T>,
}

impl<'a, T, J> SparseRowNonzeroIterator<'a, T, J>
where
    J: Ord + Increment + Zero + Copy + Debug,
{
    fn new(row: &'a SparseRowReadLock<'a, T, J>, i: usize) -> Self {
        let iter = row.shard.range((
            Included((i as u32, J::zero())),
            Excluded((i as u32, row.j_bound.inc(row.j_bound))),
        ));

        SparseRowNonzeroIterator { _row: row, iter }
    }

    fn new_from(row: &'a SparseRowReadLock<'a, T, J>, i: usize, from: J) -> Self {
        let iter = row.shard.range((
            Included((i as u32, from)),
            Excluded((i as u32, row.j_bound.inc(row.j_bound))),
        ));

        SparseRowNonzeroIterator { _row: row, iter }
    }

    fn new_to(row: &'a SparseRowReadLock<'a, T, J>, i: usize, to: J) -> Self {
        let iter = row
            .shard
            .range((Included((i as u32, J::zero())), Excluded((i as u32, to))));

        SparseRowNonzeroIterator { _row: row, iter }
    }
}

impl<T, J> Iterator for SparseRowNonzeroIterator<'_, T, J>
where
    T: Clone + Copy,
    J: Clone + Copy,
{
    type Item = (J, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&(_i, j), &v)| (j, v))
    }
}

pub trait Increment {
    fn inc(&self, bound: Self) -> Self;
}

impl Increment for u32 {
    fn inc(&self, _bound: Self) -> Self {
        *self + 1
    }
}

pub struct SparseRowIterator<'a, T, J> {
    _row: &'a SparseRowReadLock<'a, T, J>,
    pub j: J,
    pub j_bound: J,
    pub buf: Option<(J, T)>,
    iter: std::collections::btree_map::Range<'a, (u32, J), T>,
}

impl<'a, T, J> SparseRowIterator<'a, T, J>
where
    J: Ord + Increment + Zero + Copy + Debug,
    T: Copy,
{
    fn new(row: &'a SparseRowReadLock<'a, T, J>, i: usize) -> Self {
        let mut iter = row.shard.range((
            Included((i as u32, J::zero())),
            Excluded((i as u32, row.j_bound.inc(row.j_bound))),
        ));
        let buf = iter.next().map(|(&(_i, j), &v)| (j, v));

        SparseRowIterator {
            _row: row,
            j: J::zero(),
            j_bound: row.j_bound,
            buf,
            iter,
        }
    }
}

impl<T, J> Iterator for SparseRowIterator<'_, T, J>
where
    T: Clone + Copy + Zero,
    J: Debug + Clone + Copy + Eq + AddAssign + Ord + Increment,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.j_bound.inc(self.j_bound);
        if self.j == n {
            None
        } else if let Some((j_buf, item_buf)) = self.buf {
            let value = match self.j.cmp(&j_buf) {
                std::cmp::Ordering::Less => T::zero(),
                std::cmp::Ordering::Equal => {
                    self.buf = self.iter.next().map(|(&(_i, j), &v)| (j, v));
                    item_buf
                }
                std::cmp::Ordering::Greater => {
                    panic!("Broken iterator assumptions.");
                }
            };

            self.j = self.j.inc(self.j_bound);
            Some(value)
        } else if self.j < n {
            self.j = self.j.inc(self.j_bound);
            Some(T::zero())
        } else {
            dbg!((self.j, n, self.buf.is_some()));
            panic!("Incorrect iterator.")
        }
    }
}

pub struct SparseRowMutNonzeroIterator<'a, T, J> {
    pub iter: std::collections::btree_map::RangeMut<'a, (u32, J), T>,
}

impl<'a, T, J> SparseRowMutNonzeroIterator<'a, T, J>
where
    J: Ord + Increment + Zero + Copy + Debug,
{
    fn new(row: &'a mut SparseRowWriteLock<'_, T, J>, i: usize) -> Self {
        let iter = row.shard.range_mut((
            Included((i as u32, J::zero())),
            Excluded((i as u32, row.j_bound.inc(row.j_bound))),
        ));

        SparseRowMutNonzeroIterator { iter }
    }
}

impl<'a, T, J> Iterator for SparseRowMutNonzeroIterator<'a, T, J>
where
    T: Clone + Copy,
    J: Clone + Copy,
{
    type Item = (J, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(&(_i, j), v)| (j, v))
    }
}
