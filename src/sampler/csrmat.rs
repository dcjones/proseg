use super::sparsevec::SparseCountVec;
use super::RAYON_CELL_MIN_LEN;
use num::traits::AsPrimitive;
use num::traits::Zero;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use rayon::iter::plumbing::{Consumer, ProducerCallback, UnindexedConsumer};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::prelude::*;
use std::fmt::Debug;
use std::iter::{Iterator, Sum};
use std::ops::{Add, AddAssign, SubAssign};

// Main trait that the column index type is expected to implement.
pub trait Increment {
    fn inc(&self, bound: Self) -> Self;
}

impl Increment for u32 {
    fn inc(&self, _bound: Self) -> Self {
        *self + 1
    }
}

// CSR sparse matrix using per-row locks and SparseCountVec for each row
//
// # Lock Lifetime Gotcha
//
// When reusing a `CSRRow` handle to acquire multiple locks sequentially,
// Rust's NLL may not always drop the lock guard early enough. To avoid
// potential deadlocks or hanging, use explicit scoping:
//
// ```rust
// let row = mat.row(0);
//
// // Good: explicit scoping ensures lock is dropped
// {
//     let row_read = row.read();
//     // use row_read...
// }
// {
//     let mut row_write = row.write();
//     // use row_write...
// }
//
// // Bad: may hang due to NLL not dropping row_read early enough
// let row_read = row.read();
// let value = row_read.iter_nonzeros().find(...);
// let mut row_write = row.write(); // may deadlock!
// ```
//
// Alternatively, get a fresh `CSRRow` handle for each lock acquisition.
#[derive(Debug)]
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
        self.rows
            .par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|row| {
                row.write().clear();
            });
    }

    pub fn zero(&mut self) {
        self.rows
            .par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|row| {
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
            .with_min_len(RAYON_CELL_MIN_LEN)
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

        for (a_row, b_row) in self.rows().zip(other.rows()) {
            // Use explicit scope to ensure locks are dropped after each row comparison
            let rows_equal = {
                let a_lock = a_row.read();
                let b_lock = b_row.read();

                // Compare iterators element by element without collecting everything
                let mut a_iter = a_lock.iter_nonzeros();
                let mut b_iter = b_lock.iter_nonzeros();

                loop {
                    match (a_iter.next(), b_iter.next()) {
                        (Some(a), Some(b)) => {
                            if a != b {
                                break false;
                            }
                        }
                        (None, None) => break true,
                        _ => break false, // Different lengths
                    }
                }
            }; // Locks dropped here

            if !rows_equal {
                return false;
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
    /// Create a handle to a matrix row.
    ///
    /// Note: When acquiring multiple locks from the same handle sequentially,
    /// use explicit scoping to ensure lock guards are dropped. See CSRMat docs.
    fn new(mat: &'a CSRMat<J, T>, i: usize) -> Self {
        Self {
            row_lock: &mat.rows[i],
            i,
            j_bound: mat.j_bound,
        }
    }

    /// Acquire a read lock on this row.
    ///
    /// When acquiring multiple locks sequentially from the same CSRRow handle,
    /// use explicit scoping to ensure the lock guard is dropped before acquiring
    /// the next lock.
    pub fn read(&self) -> CSRRowReadLock<'a, J, T> {
        CSRRowReadLock {
            guard: self.row_lock.read(),
            j_bound: self.j_bound,
        }
    }

    /// Acquire a write lock on this row.
    ///
    /// When acquiring multiple locks sequentially from the same CSRRow handle,
    /// use explicit scoping to ensure the lock guard is dropped before acquiring
    /// the next lock.
    pub fn write(&self) -> CSRRowWriteLock<'a, J, T> {
        CSRRowWriteLock {
            guard: self.row_lock.write(),
            j_bound: self.j_bound,
        }
    }
}

// Read lock for a row
pub struct CSRRowReadLock<'a, J, T> {
    guard: RwLockReadGuard<'a, SparseCountVec<J, T>>,
    j_bound: J,
}

impl<'a, J, T> CSRRowReadLock<'a, J, T>
where
    T: Copy + Zero + PartialEq,
    J: Copy + Ord + Increment + Zero,
{
    pub fn iter_nonzeros<'b>(&'b self) -> impl Iterator<Item = (J, T)> + 'b {
        self.guard.iter()
    }

    pub fn iter_nonzeros_from<'b>(&'b self, from: J) -> impl Iterator<Item = (J, T)> + 'b {
        self.guard.iter_from(from)
    }

    pub fn iter_nonzeros_to<'b>(&'b self, to: J) -> impl Iterator<Item = (J, T)> + 'b {
        self.guard.iter_to(to)
    }
}

impl<'a, J, T> CSRRowReadLock<'a, J, T>
where
    T: Copy + Zero,
    J: Clone + Debug + Increment + Zero + Copy + Ord,
{
    pub fn iter<'b>(&'b self) -> impl Iterator<Item = T> + 'b {
        self.guard.iter_dense(self.j_bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csrmat_basic() {
        let mat = CSRMat::<u32, i32>::zeros(5, 10);
        assert_eq!(mat.m, 5);
        assert_eq!(mat.n, 11); // j_bound + 1

        // Test row access
        let row = mat.row(2);
        let mut row_write = row.write();
        row_write.add(5, 42);
        row_write.add(7, 13);
        drop(row_write);

        // Test reading back
        let row_read = row.read();
        let nonzeros: Vec<_> = row_read.iter_nonzeros().collect();
        assert_eq!(nonzeros, vec![(5, 42), (7, 13)]);

        // Test dense iteration
        let values: Vec<_> = row_read.iter().collect();
        assert_eq!(values.len(), 11);
        assert_eq!(values[5], 42);
        assert_eq!(values[7], 13);
        assert_eq!(values[0], 0);
    }

    #[test]
    fn test_csrmat_update() {
        // This test demonstrates the need for explicit scoping when reusing
        // a CSRRow handle to acquire multiple locks sequentially.
        let mat = CSRMat::<u32, i32>::zeros(3, 5);
        let row = mat.row(1);

        {
            let mut row_write = row.write();
            row_write.update(2, || 10, |v| *v += 5);
        }

        // Read back to verify
        {
            let row_read = row.read();
            let value = row_read
                .iter_nonzeros()
                .find(|(j, _)| *j == 2)
                .map(|(_, v)| v);
            assert_eq!(value, Some(15));
        }

        // Update again
        {
            let mut row_write = row.write();
            row_write.update(2, || 0, |v| *v += 3);
        }

        // Read back to verify
        {
            let row_read = row.read();
            let value = row_read
                .iter_nonzeros()
                .find(|(j, _)| *j == 2)
                .map(|(_, v)| v);
            assert_eq!(value, Some(18));
        }
    }

    #[test]
    fn test_csrmat_update_no_reuse() {
        let mat = CSRMat::<u32, i32>::zeros(3, 5);

        // First update
        {
            let row = mat.row(1);
            let mut row_write = row.write();
            row_write.update(2, || 10, |v| *v += 5);
        }

        // Read back to verify
        {
            let row = mat.row(1);
            let row_read = row.read();
            let value = row_read
                .iter_nonzeros()
                .find(|(j, _)| *j == 2)
                .map(|(_, v)| v);
            assert_eq!(value, Some(15));
        }

        // Update again
        {
            let row = mat.row(1);
            let mut row_write = row.write();
            row_write.update(2, || 0, |v| *v += 3);
        }

        // Read back to verify
        {
            let row = mat.row(1);
            let row_read = row.read();
            let value = row_read
                .iter_nonzeros()
                .find(|(j, _)| *j == 2)
                .map(|(_, v)| v);
            assert_eq!(value, Some(18));
        }
    }

    #[test]
    fn test_csrmat_parallel() {
        let mat = CSRMat::<u32, i32>::zeros(100, 50);

        // Write values in parallel (add row index to column 10)
        mat.par_rows().for_each(|row| {
            let mut row_write = row.write();
            row_write.add(10, row.i as i32);
        });

        // Read and verify in parallel - sum up all the values we wrote
        let sum: i32 = mat
            .par_rows()
            .map(|row| {
                let row_read = row.read();
                row_read.iter_nonzeros().map(|(_, val)| val).sum::<i32>()
            })
            .sum();

        // Each row i has value i at column 10, so sum is 0+1+2+...+99
        assert_eq!(sum, (0..100).sum::<usize>() as i32);
    }

    #[test]
    fn test_csrmat_sum() {
        let mat = CSRMat::<u32, i32>::zeros(10, 20);

        // Add values sequentially to avoid potential parallel issues during testing
        for i in 0..10 {
            let row = mat.row(i);
            let mut row_write = row.write();
            row_write.add((i * 2) as u32, 100);
            drop(row_write); // Explicitly drop the lock
        }

        assert_eq!(mat.sum(), 1000);
    }

    #[test]
    fn test_csrmat_clear() {
        let mut mat = CSRMat::<u32, i32>::zeros(5, 10);

        // Add some data
        for i in 0..5 {
            let row = mat.row(i);
            let mut row_write = row.write();
            row_write.add(i as u32, 42);
            drop(row_write); // Explicitly drop the lock
        }

        assert_eq!(mat.sum(), 5 * 42);

        // Clear and verify
        mat.clear();
        assert_eq!(mat.sum(), 0);
    }

    #[test]
    fn test_csrmat_zero() {
        let mut mat = CSRMat::<u32, i32>::zeros(5, 10);

        // Add some data
        for i in 0..5 {
            let row = mat.row(i);
            let mut row_write = row.write();
            row_write.add(i as u32, 42);
            drop(row_write); // Explicitly drop the lock
        }

        assert_eq!(mat.sum(), 5 * 42);

        // Zero and verify
        mat.zero();
        assert_eq!(mat.sum(), 0);

        // Verify structure is still intact
        let row = mat.row(2);
        let row_read = row.read();
        let values: Vec<_> = row_read.iter_nonzeros().collect();
        assert_eq!(values.len(), 0); // Should be empty after zero
    }

    #[test]
    fn test_csrmat_equality() {
        let mat1 = CSRMat::<u32, i32>::zeros(3, 5);
        let mat2 = CSRMat::<u32, i32>::zeros(3, 5);
        let mat3 = CSRMat::<u32, i32>::zeros(3, 5);

        // Add same data to mat1 and mat2
        for i in 0..3 {
            let row1 = mat1.row(i);
            let row2 = mat2.row(i);
            let mut row1_write = row1.write();
            let mut row2_write = row2.write();
            row1_write.add((i * 2) as u32, 100);
            row2_write.add((i * 2) as u32, 100);
            drop(row1_write);
            drop(row2_write);
        }

        // Add different data to mat3
        let row3 = mat3.row(1);
        let mut row3_write = row3.write();
        row3_write.add(1, 999);
        drop(row3_write);

        assert_eq!(mat1, mat2);
        assert_ne!(mat1, mat3);
    }

    #[test]
    fn test_csrmat_equality_edge_cases() {
        // Test empty matrices
        let empty1 = CSRMat::<u32, i32>::zeros(3, 5);
        let empty2 = CSRMat::<u32, i32>::zeros(3, 5);
        assert_eq!(empty1, empty2);

        // Test different dimensions
        let mat_3x5 = CSRMat::<u32, i32>::zeros(3, 5);
        let mat_3x6 = CSRMat::<u32, i32>::zeros(3, 6);
        let mat_4x5 = CSRMat::<u32, i32>::zeros(4, 5);
        assert_ne!(mat_3x5, mat_3x6);
        assert_ne!(mat_3x5, mat_4x5);

        // Test sparse vs dense representation of same values
        let mat1 = CSRMat::<u32, i32>::zeros(2, 10);
        let mat2 = CSRMat::<u32, i32>::zeros(2, 10);

        {
            let row1 = mat1.row(0);
            let mut w = row1.write();
            w.add(1, 10);
            w.add(5, 20);
            w.add(9, 30);
        }

        {
            let row2 = mat2.row(0);
            let mut w = row2.write();
            w.add(1, 10);
            w.add(5, 20);
            w.add(9, 30);
        }

        assert_eq!(mat1, mat2);

        // Test different number of nonzeros in same row
        let mat3 = CSRMat::<u32, i32>::zeros(2, 10);
        {
            let row3 = mat3.row(0);
            let mut w = row3.write();
            w.add(1, 10);
            w.add(5, 20);
            // Missing the third element
        }

        assert_ne!(mat1, mat3);

        // Test different values at same positions
        let mat4 = CSRMat::<u32, i32>::zeros(2, 10);
        {
            let row4 = mat4.row(0);
            let mut w = row4.write();
            w.add(1, 10);
            w.add(5, 999); // Different value
            w.add(9, 30);
        }

        assert_ne!(mat1, mat4);

        // Test multiple rows with differences in later rows
        let mat5 = CSRMat::<u32, i32>::zeros(3, 5);
        let mat6 = CSRMat::<u32, i32>::zeros(3, 5);

        for i in 0..3 {
            let row5 = mat5.row(i);
            let row6 = mat6.row(i);
            let mut w5 = row5.write();
            let mut w6 = row6.write();
            w5.add(i as u32, 100);
            w6.add(i as u32, 100);
        }

        assert_eq!(mat5, mat6);

        // Make last row different
        {
            let row5 = mat5.row(2);
            let mut w = row5.write();
            w.add(4, 50); // Add extra element to last row
        }

        assert_ne!(mat5, mat6);
    }
}

// Write lock for a row
pub struct CSRRowWriteLock<'a, J, T> {
    pub guard: RwLockWriteGuard<'a, SparseCountVec<J, T>>,
    j_bound: J,
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
