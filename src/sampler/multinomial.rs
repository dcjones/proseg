use num::traits::{AsPrimitive, Float, One, Zero};
use rand::Rng;
use rand_distr::Distribution;
use std::ops::AddAssign;

const BINOMIAL_DIRECT_CUTOFF: u32 = 10;
const BINOMIAL_INVERSE_CUTOFF: f64 = 20.0;

/// Inverse transform with symmetry optimization
/// When p > 0.5, sample from Binomial(n, 1-p) and return n - result
fn binomial_inversion_symmetric<R: Rng>(rng: &mut R, n: u64, p: f64) -> u64 {
    if p == 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }

    // Use symmetry to always work with p <= 0.5
    let (p_eff, flip) = if p > 0.5 { (1.0 - p, true) } else { (p, false) };

    let q = 1.0 - p_eff;
    let s = p_eff / q;
    let mut prob = q.powi(n as i32);
    let mut cdf = prob;
    let u: f64 = rng.random();

    for k in 0..n {
        if u <= cdf {
            return if flip { n - k } else { k };
        }
        prob *= s * (n - k) as f64 / (k + 1) as f64;
        cdf += prob;
    }
    if flip { 0 } else { n }
}

pub fn rand_binomial<R: Rng>(rng: &mut R, p: f64, n: u32) -> u32 {
    if n == 0 || p == 0.0 {
        return 0;
    }

    if p >= 1.0 {
        return n;
    }

    if n <= BINOMIAL_DIRECT_CUTOFF {
        return (0..n).filter(|_| rng.random_bool(p)).count() as u32;
    }

    if n as f64 * p.min(1.0 - p) <= BINOMIAL_INVERSE_CUTOFF {
        return binomial_inversion_symmetric(rng, n as u64, p) as u32;
    }

    rand_distr::Binomial::new(n as u64, p).unwrap().sample(rng) as u32
}

pub struct Multinomial<T> {
    cumprobs: Vec<T>,
}

impl<T> Multinomial<T>
where
    T: Float + Zero + One + AddAssign + AsPrimitive<f64>,
{
    pub fn new(len: usize) -> Self {
        Self {
            cumprobs: vec![T::zero(); len + 1],
        }
    }

    pub fn from_probs(probs: &[T]) -> Self {
        let mut dist = Self::new(probs.len());
        dist.set_probs(probs);
        dist
    }

    pub fn set_probs(&mut self, probs: &[T]) {
        assert!(probs.len() + 1 == self.cumprobs.len());

        let mut cumprob = T::zero();
        for (cumprob_i, prob) in self.cumprobs.iter_mut().zip(probs.iter()) {
            *cumprob_i = cumprob;
            cumprob += *prob;
        }
        *self.cumprobs.last_mut().unwrap() = cumprob;
    }

    pub fn set_probs_from_iter(&mut self, probs: impl IntoIterator<Item = T>) {
        let mut cumprob = T::zero();
        for (cumprob_i, prob) in self.cumprobs.iter_mut().zip(probs.into_iter()) {
            *cumprob_i = cumprob;
            cumprob += prob;
        }
        *self.cumprobs.last_mut().unwrap() = cumprob;
    }

    // Recursive divide-and conqueror sampling. When n is small or outcome probabilities are skewed
    // this lets us prune many branches, speeding up the sampling.
    pub fn sample<R: Rng, F: FnMut(usize, u32)>(&self, rng: &mut R, n: u32, mut report: F) {
        if n == 0 {
            return;
        }

        self.sample_recursion(rng, 0, self.cumprobs.len() - 1, n, &mut report);
    }

    // Sample on the interval [from, to)
    fn sample_recursion<R: Rng, F: FnMut(usize, u32)>(
        &self,
        rng: &mut R,
        from: usize,
        to: usize,
        n: u32,
        report: &mut F,
    ) {
        if n == 0 {
            return;
        }

        if to - from == 1 {
            if n > 0 {
                report(from, n);
            }
            return;
        }

        let mid = (from + to) / 2;
        let left_prob = self.cumprobs[mid] - self.cumprobs[from];
        let total_prob = self.cumprobs[to] - self.cumprobs[from];

        let p = (left_prob / total_prob).min(T::one()).max(T::zero());
        let n_left = rand_binomial(rng, p.as_(), n);

        self.sample_recursion(rng, from, mid, n_left, report);
        self.sample_recursion(rng, mid, to, n - n_left, report);
    }
}
