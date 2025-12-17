use rand::Rng;
use rand_distr::{Binomial, Distribution};

#[derive(Copy, Clone, Debug)]
pub struct MultinomialOutcome {
    pub prob: f32,
    pub index: u32,
}

pub struct Multinomial {
    pub outcomes: Vec<MultinomialOutcome>,
}

// Idea here is to speed up sampling sparse multinomial vectors by sorting the outcome
// in descending order by probability, so we can hopefully bail out early.
impl Multinomial {
    pub fn with_k(k: usize) -> Multinomial {
        Multinomial {
            outcomes: vec![
                MultinomialOutcome {
                    prob: 0.0,
                    index: 0
                };
                k
            ],
        }
    }

    pub fn sample<R, F>(&mut self, rng: &mut R, n: u32, mut nonzero_sample: F)
    where
        R: Rng,
        F: FnMut(usize, u32),
    {
        // sort in descending order by probability
        self.outcomes
            .sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

        let norm = self
            .outcomes
            .iter()
            .fold(0.0, |accum, outcome| accum + outcome.prob as f64);

        let mut ρ = 1.0;
        let mut s = n;
        for outcome in self.outcomes.iter() {
            let p = outcome.prob as f64 / norm;
            let mut x = 0;
            if ρ > 0.0 {
                x = Binomial::new(s as u64, (p / ρ).min(1.0))
                    .unwrap()
                    .sample(rng) as u32;
                if x > 0 {
                    nonzero_sample(outcome.index as usize, x);
                }
            }

            s -= x;
            ρ -= p;

            if s == 0 {
                break;
            }
        }
        assert!(s == 0);
    }
}

// A Binomial sampler with an optimized path for small `n`
struct SmallBinomial {
    p: f64,
    n: u32,
}

impl SmallBinomial {
    fn new(p: f64, n: u32) -> SmallBinomial {
        SmallBinomial { p, n }
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> u32 {
        let mut successes = 0;
        for _ in 0..self.n {
            if rng.random_bool(self.p) {
                successes += 1;
            }
        }
        successes
    }
}

// Multinomial distribution sampler optimized for a smsall n and a small
// number of outcomes.
pub struct SmallMultinomial<'a, R> {
    rng: &'a mut R,
    probs: &'a [f32],
    ρ: f32, // remaining probability
    s: u32, // remaining count
    i: usize,
}

impl<'a, R> SmallMultinomial<'a, R>
where
    R: Rng,
{
    pub fn new(rng: &'a mut R, probs: &'a [f32], n: u32) -> Self {
        SmallMultinomial {
            rng,
            probs,
            ρ: probs.iter().sum(),
            s: n,
            i: 0,
        }
    }
}

impl<'a, R> Iterator for SmallMultinomial<'a, R>
where
    R: Rng,
{
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.i == self.probs.len() {
            return None;
        }

        if self.s == 0 {
            self.i += 1;
            return Some(0);
        }

        if self.i == self.probs.len() - 1 {
            self.i += 1;
            return Some(self.s);
        }

        let pi = self.probs[self.i];
        let r = (pi / self.ρ).min(1.0);
        let x = SmallBinomial::new(r as f64, self.s).sample(self.rng);
        self.ρ -= pi;
        self.s -= x;
        self.i += 1;

        Some(x)
    }
}

// TODO:
// Multinomial distribution sampler optimized for a large number of outcomes,
// when many have low probability.
// struct LargeMultinomial {}
