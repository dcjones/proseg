use rand::Rng;
use rand_distr::{Binomial, Distribution};

#[derive(Copy, Clone, Debug)]
pub struct MultinomialOutcome {
    pub prob: f32,
    pub index: u32,
}

pub struct Multinomial {
    pub outcomes: Vec<MultinomialOutcome>,
    pub n: u32,
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
            n: 0,
        }
    }

    pub fn sample<R, F>(&mut self, rng: &mut R, mut nonzero_sample: F)
    where
        R: Rng,
        F: FnMut(usize, u32),
    {
        // TODO: We could also use a heap here to avoid fulling sorting in most cases.

        // sort in descending order by probability
        self.outcomes
            .sort_unstable_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap());

        let norm = self
            .outcomes
            .iter()
            .fold(0.0, |accum, outcome| accum + outcome.prob as f64);

        let mut ρ = 1.0;
        let mut s = self.n;
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
