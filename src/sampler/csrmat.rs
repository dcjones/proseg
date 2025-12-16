use super::sparsemat::Increment;
use super::sparsevec::SparseCountVec;
use num::traits::AsPrimitive;
use num::traits::Zero;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::fmt::Debug;
use std::iter::{Iterator, Sum};
use std::ops::{Add, AddAssign, SubAssign};

// CSR sparse matrix using per-row locks and SparseCountVec for each row
pub struct CSRMat<J, T> {
    rows: Vec<RwLock<SparseCountVec<J, T>>>,
    pub m: usize,
    pub j_bound: J,
    pub n: J,
}

impl<J, T> CSRMat<J, T>
where
    T: Copy + Zero,
    J: Copy + Ord + Increment,
{
    pub fn zeros(m: usize, j_bound: J) -> Self {
        let mut rows = Vec::with_capacity(m);
        for _ in 0..m {
            rows.push(RwLock::new(SparseCountVec::new()));
        }
        Self {
            rows,
            m,
            j_bound,
            n: j_bound.inc(j_bound),
        }
    }

    pub fn shape(&self) -> (usize, J) {
        (self.m, self.n)
    }

    pub fn row(&self, i: usize) -> CSRRow<'_, J, T> {
        CSRRow::new(self, i)
    }

    pub fn rows(&self) -> CSRMatRowsIter<'_, J, T> {
        CSRMatRowsIter { mat: self, i: 0 }
    }

    pub fn par_rows(&self) -> CSRMatRowsParIter<'_, J, T> {
        CSRMatRowsParIter { mat: self }
    }
}

impl<J, T> CSRMat<J, T>
where
    T: Send + Sync + Copy + Zero + PartialEq,
    J: Send + Sync + Copy + Ord,
{
    pub fn clear(&mut self) {
        self.rows.par_iter().for_each(|row| {
            *row.write() = SparseCountVec::new();
        });
    }

    pub fn zero(&mut self) {
        self.rows.par_iter().for_each(|row| {
            row.write().zero_all();
        });
    }
}

impl<J, T> CSRMat<J, T>
where
    T: Zero + Copy + AddAssign + Add + Sum + AsPrimitive<u64> + Send + Sync + PartialEq,
    J: Copy + Send + Sync + Ord,
{
    pub fn sum(&self) -> u64 {
        self.rows
            .par_iter()
            .map(|row| row.read().sum_values().as_())
            .sum()
    }
}

impl<J, T> PartialEq for CSRMat<J, T>
where
    T: Zero + Copy + AddAssign + Add + Sum + AsPrimitive<u64> + PartialEq + Debug + Send + Sync,
    J: Clone + Copy + Ord + Zero + AddAssign + Debug + Increment + Send + Sync + PartialEq,
{
    fn eq(&self, other: &CSRMat<J, T>) -> bool {
        if self.m != other.m || self.n != other.n {
            return false;
        }

        for (a_i, b_i) in self.rows().zip(other.rows()) {
            let a_lock = a_i.read();
            let b_lock = b_i.read();

            for ((a_j, a_val), (b_j, b_val)) in a_lock.iter_nonzeros().zip(b_lock.iter_nonzeros()) {
                if a_j != b_j || a_val != b_val {
                    return false;
                }
            }
        }

        true
    }
}

// Sequential row iterator
pub struct CSRMatRowsIter<'a, J, T> {
    i: usize,
    mat: &'a CSRMat<J, T>,
}

impl<'a, J, T> Iterator for CSRMatRowsIter<'a, J, T>
where
    J: Copy,
{
    type Item = CSRRow<'a, J, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i < self.mat.m {
            let item = CSRRow::new(self.mat, self.i);
            self.i += 1;
            Some(item)
        } else {
            None
        }
    }
}

// Parallel row iterator
pub struct CSRMatRowsParIter<'a, J, T> {
    mat: &'a CSRMat<J, T>,
}

impl<'a, J, T> ParallelIterator for CSRMatRowsParIter<'a, J, T>
where
    T: Send + Sync,
    J: Send + Sync + Copy,
{
    type Item = CSRRow<'a, J, T>;

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        (0..self.mat.m)
            .into_par_iter()
            .map(|i| CSRRow::new(self.mat, i))
            .drive_unindexed(consumer)
    }
}

impl<'a, J, T> IndexedParallelIterator for CSRMatRowsParIter<'a, J, T>
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
            .map(|i| CSRRow::new(self.mat, i))
            .drive(consumer)
    }

    fn with_producer<CB>(self, callback: CB) -> CB::Output
    where
        CB: ProducerCallback<Self::Item>,
    {
        (0..self.mat.m)
            .into_par_iter()
            .map(|i| CSRRow::new(self.mat, i))
            .with_producer(callback)
    }
}

// Row handle (like SparseRow)
pub struct CSRRow<'a, J, T> {
    row_lock: &'a RwLock<SparseCountVec<J, T>>,
    pub i: usize,
    pub j_bound: J,
}

impl<'a, J, T> CSRRow<'a, J, T>
where
    J: Copy,
{
    fn new(mat: &'a CSRMat<J, T>, i: usize) -> Self {
        Self {
            row_lock: &mat.rows[i],
            i,
            j_bound: mat.j_bound,
        }
    }

    pub fn read(&self) -> CSRRowReadLock<'a, J, T> {
        CSRRowReadLock {
            guard: self.row_lock.read(),
            j_bound: self.j_bound,
            i: self.i,
        }
    }

    pub fn write(&self) -> CSRRowWriteLock<'a, J, T> {
        CSRRowWriteLock {
            guard: self.row_lock.write(),
            j_bound: self.j_bound,
            i: self.i,
        }
    }
}

// Read lock for a row
pub struct CSRRowReadLock<'a, J, T> {
    guard: RwLockReadGuard<'a, SparseCountVec<J, T>>,
    j_bound: J,
    i: usize,
}

impl<'a, J, T> CSRRowReadLock<'a, J, T>
where
    T: Copy + Zero + PartialEq,
    J: Copy + Ord + Increment + Zero,
{
    pub fn iter_nonzeros(&'a self) -> impl Iterator<Item = (J, T)> + 'a {
        self.guard.iter()
    }

    pub fn iter_nonzeros_from(&'a self, from: J) -> impl Iterator<Item = (J, T)> + 'a {
        self.guard.iter_from(from)
    }

    pub fn iter_nonzeros_to(&'a self, to: J) -> impl Iterator<Item = (J, T)> + 'a {
        self.guard.iter_to(to)
    }

    pub fn iter(&'a self) -> impl Iterator<Item = T> + 'a
    where
        J: Clone + Debug,
    {
        self.guard.iter_dense(self.j_bound)
    }
}

// Write lock for a row
pub struct CSRRowWriteLock<'a, J, T> {
    guard: RwLockWriteGuard<'a, SparseCountVec<J, T>>,
    j_bound: J,
    i: usize,
}

impl<'a, J, T> CSRRowWriteLock<'a, J, T>
where
    T: Copy + Zero + AddAssign + SubAssign + Eq + PartialOrd,
    J: Copy + Ord + Increment + Debug + Zero,
{
    pub fn sub(&mut self, j: J, delta: T) {
        assert!(j <= self.j_bound, "Column index out of bounds");
        self.guard.update_count(j, |v| {
            assert!(*v >= delta, "Subtracting from a value smaller than delta");
            *v -= delta;
        });
    }

    pub fn add(&mut self, j: J, delta: T) {
        assert!(j <= self.j_bound, "Column index out of bounds");
        self.guard.update_count(j, |v| *v += delta);
    }
}

impl<'a, J, T> CSRRowWriteLock<'a, J, T>
where
    J: Copy + Ord + Increment + Debug + Zero,
{
    pub fn update<F, G>(&mut self, j: J, insert_fn: F, update_fn: G)
    where
        F: FnOnce() -> T,
        G: FnOnce(&mut T),
        T: Copy + Zero,
    {
        assert!(j <= self.j_bound, "Column index out of bounds");
        self.guard.update_with_init(j, insert_fn, update_fn);
    }

    pub fn update_if_present<G>(&mut self, j: J, update_fn: G)
    where
        G: FnOnce(&mut T),
        T: Copy + Zero + PartialEq,
    {
        assert!(j <= self.j_bound, "Column index out of bounds");
        self.guard.update_if_present(j, update_fn);
    }
}
