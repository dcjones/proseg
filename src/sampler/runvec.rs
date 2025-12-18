// This is barrowing from the `rle_vec` crate, but with some additional features:
// - Generic on run length type (our runs will always fit in u32s, so we want to
// save space)

use num::cast::AsPrimitive;
use num::{Integer, NumCast};
use std::iter::Iterator;
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
    I: Integer + NumCast + AddAssign + SubAssign + Copy + AsPrimitive<usize>,
    T: Copy + PartialEq,
{
    pub fn new() -> Self {
        RunVec {
            runs: Vec::new(),
            len: 0,
        }
    }

    pub fn with_run_capacity(nruns: usize) -> Self {
        RunVec {
            runs: Vec::with_capacity(nruns),
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

    pub fn push_run(&mut self, value: T, len: I) {
        if let Some(last) = self.runs.last_mut() {
            if last.value == value {
                last.len += len;
                self.len += len.as_();
                return;
            }
        }

        self.runs.push(Run { len, value });
        self.len += len.as_();
    }

    pub fn len(&self) -> usize {
        self.len
    }

    // pub fn nruns(&self) -> usize {
    //     self.runs.len()
    // }

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
        let mut newruns: RunVec<I, T> = RunVec::with_run_capacity(self.runs.len());

        for (value, m) in self.iter().zip(mask) {
            if *m {
                newruns.push(*value);
            }
        }

        self.len = newruns.len;
        self.runs = newruns.runs;
    }

    pub fn shrink_to_fit(&mut self) {
        self.runs.shrink_to_fit();
    }
}
