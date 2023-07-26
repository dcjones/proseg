use rand::rngs::ThreadRng;
use rand::Rng;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::HashMap;
use std::hash::Hash;

// A set you can sample random elements from.
// Inspired by this solution: https://stackoverflow.com/a/53773240
#[derive(Clone)]
pub struct SampleSet<T> {
    set: HashMap<T, usize>,
    vec: Vec<T>,
}

impl<T> SampleSet<T>
where
    T: Eq + Hash + Copy,
{
    pub fn new() -> Self {
        return SampleSet {
            set: HashMap::new(),
            vec: Vec::new(),
        };
    }

    pub fn len(&self) -> usize {
        assert!(self.set.len() == self.vec.len());
        return self.vec.len();
    }

    pub fn is_empty(&self) -> bool {
        return self.len() == 0;
    }

    // pub fn clear(&mut self) {
    //     self.set.clear();
    //     self.vec.clear();
    // }

    pub fn insert(&mut self, value: T) -> bool {
        if let Vacant(entry) = self.set.entry(value) {
            entry.insert(self.vec.len());
            self.vec.push(value);
            return true;
        } else {
            return false;
        }
    }

    pub fn remove(&mut self, value: T) -> bool {
        if let Occupied(entry) = self.set.entry(value) {
            let index = *entry.get();
            self.vec.swap_remove(index);
            self.set.remove(&value);
            // update the index that we just swapped
            if index < self.vec.len() {
                self.set.insert(self.vec[index], index);
            }
            return true;
        } else {
            return false;
        }
    }

    pub fn choose(&self, rng: &mut ThreadRng) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        let index = rng.gen_range(0..self.len());
        return Some(&self.vec[index]);
    }

    // fn contains(&self, value: &T) -> bool {
    //     return self.set.contains_key(value);
    // }

    // fn iter(&self) -> std::slice::Iter<T> {
    //     return self.vec.iter();
    // }
}
