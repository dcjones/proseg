use num::Zero;
use std::ops::{AddAssign, SubAssign};
use std::sync::{Arc, RwLock};

// TODO:
// - Make this generic on the number of shards
// - Shards don't need to be in a Vec

fn divrem(a: usize, b: usize) -> (usize, usize) {
    (a / b, a % b)
}

pub struct ShardedVec<T> {
    shards: Vec<Arc<RwLock<Vec<T>>>>,
    shardsize: usize,
    n: usize,
}

impl<T> ShardedVec<T>
where
    T: AddAssign + SubAssign + Clone + Zero + Clone + Copy,
{
    pub fn zeros(n: usize, shardsize: usize) -> Self {
        let num_shards = (n + shardsize - 1) / shardsize;
        let shards = (0..num_shards)
            .map(|_| Arc::new(RwLock::new(vec![T::zero(); shardsize])))
            .collect();
        Self {
            shards,
            shardsize,
            n,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn get(&self, index: usize) -> T {
        if index >= self.n {
            panic!(
                "Index {} is out of bounds for ShardedVec of length {}",
                index, self.n
            );
        }
        let (i, j) = divrem(index, self.shardsize);
        let shard_index = i;
        let shard = self.shards[shard_index].read().unwrap();
        shard[j].clone()
    }

    pub fn modify(&self, index: usize, f: impl FnOnce(&mut T)) {
        if index >= self.n {
            panic!(
                "Index {} is out of bounds for ShardedVec of length {}",
                index, self.n
            );
        }
        let (i, j) = divrem(index, self.shardsize);
        let shard_index = i;
        let mut shard = self.shards[shard_index].write().unwrap();
        f(&mut shard[j]);
    }

    pub fn add(&self, index: usize, value: T) {
        if index >= self.n {
            panic!(
                "Index {} is out of bounds for ShardedVec of length {}",
                index, self.n
            );
        }
        let (i, j) = divrem(index, self.shardsize);
        let mut shard = self.shards[i].write().unwrap();
        shard[j] += value;
    }

    pub fn sub(&self, index: usize, value: T) {
        if index >= self.n {
            panic!(
                "Index {} is out of bounds for ShardedVec of length {}",
                index, self.n
            );
        }
        let (i, j) = divrem(index, self.shardsize);
        let mut shard = self.shards[i].write().unwrap();
        shard[j] -= value;
    }

    pub fn set(&self, index: usize, value: T) {
        if index >= self.n {
            panic!(
                "Index {} is out of bounds for ShardedVec of length {}",
                index, self.n
            );
        }

        let (i, j) = divrem(index, self.shardsize);
        let mut shard = self.shards[i].write().unwrap();
        shard[j] = value;
    }

    pub fn copy_from(&mut self, other: &ShardedVec<T>) {
        if self.shardsize != other.shardsize || self.shards.len() != other.shards.len() {
            panic!("ShardedVecs have different shard sizes");
        }

        for (shard, other_shard) in self.shards.iter_mut().zip(other.shards.iter()) {
            let mut shard = shard.write().unwrap();
            let other_shard = other_shard.read().unwrap();
            shard.copy_from_slice(&other_shard);
        }
    }

    pub fn iter(&self) -> ShardedVecIterator<'_, T> {
        ShardedVecIterator {
            vec: self,
            index: 0,
        }
    }

    pub fn zero(&mut self) {
        for shard in &self.shards {
            let mut shard = shard.write().unwrap();
            shard.fill(T::zero());
        }
    }
}

pub struct ShardedVecIterator<'a, T> {
    vec: &'a ShardedVec<T>,
    index: usize,
}

impl<'a, T> Iterator for ShardedVecIterator<'a, T>
where
    T: AddAssign + SubAssign + Clone + Copy + Zero,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.vec.n {
            None
        } else {
            let item = self.vec.get(self.index);
            self.index += 1;
            Some(item)
        }
    }
}
