// This is barrowing from the `rle_vec` crate, but with some additional features:
// - Generic on run length type (our runs will always fit in u32s, so we want to
// save space)

use num::{Integer, NumCast};
use std::iter::Iterator;
use std::mem::replace;
use std::ops::{AddAssign, SubAssign};

#[derive(Debug)]
pub struct Run<I, T> {
    pub len: I,
    pub value: T,
}

pub struct RunVec<I, T> {
    pub runs: Vec<Run<I, T>>,
    pub len: usize,
}

pub struct RunVecIter<'a, I, T> {
    it: std::slice::Iter<'a, Run<I, T>>,
    value: Option<&'a T>,
    count: I,
}

impl<'a, I, T> Iterator for RunVecIter<'a, I, T>
where
    I: Integer + NumCast + AddAssign + SubAssign + Copy,
    T: PartialEq,
{
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        if self.count.is_zero() {
            if let Some(next_value) = self.it.next() {
                self.count = next_value.len;
                self.value = Some(&next_value.value);
                self.next()
            } else {
                None
            }
        } else {
            self.count -= I::one();
            self.value
        }
    }
}

impl<I, T> RunVec<I, T>
where
    I: Integer + NumCast + AddAssign + SubAssign + Copy,
    T: Copy + PartialEq,
{
    pub fn new() -> Self {
        RunVec {
            runs: Vec::new(),
            len: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        if let Some(last) = self.runs.last_mut() {
            if last.value == value {
                last.len += I::one();
                self.len += 1;
                return;
            }
        }

        self.runs.push(Run {
            len: I::one(),
            value,
        });
        self.len += 1;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn iter_runs(&self) -> impl Iterator<Item = &Run<I, T>> {
        self.runs.iter()
    }

    pub fn iter_runs_mut(&mut self) -> impl Iterator<Item = &mut Run<I, T>> {
        self.runs.iter_mut()
    }

    pub fn iter(&self) -> RunVecIter<I, T> {
        RunVecIter {
            it: self.runs.iter(),
            value: None,
            count: I::zero(),
        }
    }

    pub fn retain_masked(&mut self, mask: &Vec<bool>) {
        let mut newruns: RunVec<I, T> = RunVec::new();

        for (value, m) in self.iter().zip(mask) {
            if *m {
                newruns.push(*value);
            }
        }

        self.len = newruns.len;
        replace(&mut self.runs, newruns.runs);
    }
}
