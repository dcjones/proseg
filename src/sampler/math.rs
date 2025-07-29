use libm::{erff, lgammaf};
use rand::Rng;
use rand::rngs::ThreadRng;
use rand_distr::{Binomial, Distribution, StandardNormal};
use std::f32;

// pub fn logit(p: f32) -> f32 {
//     return p.ln() - (1.0 - p).ln();
// }

// pub fn sq(x: f32) -> f32 {
//     x * x
// }

pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn odds_to_prob(q: f32) -> f32 {
    q / (1.0 + q)
}

// pub fn relerr(a: f32, b: f32) -> f32 {
//     ((a - b) / a).abs()
// }

// // Partial Student-T log-pdf (just the terms that don't cancel out when doing MH sampling)
// pub fn studentt_logpdf_part(σ2: f32, df: f32, x2: f32) -> f32 {
//     return -((df + 1.0) / 2.0) * ((x2 / σ2) / df).ln_1p();
// }

// const SQRT_TWO_PI: f32 = 2.506_628_3_f32;
const LN_SQRT_TWO_PI: f32 = 0.918_938_5_f32;
const SQRT_2_DIV_SQRT_PI: f32 = 0.797_884_6_f32;

// pub fn normal_x2_pdf(σ: f32, x2: f32) -> f32 {
//     (-x2 / (2.0 * σ.powi(2))).exp() / (σ * SQRT_TWO_PI)
// }

// pub fn normal_x2_logpdf(σ: f32, x2: f32) -> f32 {
//     -x2 / (2.0 * σ.powi(2)) - σ.ln() - LN_SQRT_TWO_PI
// }

// pub fn gamma_logpdf(shape: f32, scale: f32, x: f32) -> f32 {
//     return
//         -lgammaf(shape)
//         - shape * scale.ln()
//         + (shape - 1.0) * x.ln()
//         - x / scale;
// }

pub fn rand_crt(rng: &mut ThreadRng, n: u32, r: f32) -> u32 {
    (0..n)
        .map(|t| rng.random_bool(r as f64 / (r as f64 + t as f64)) as u32)
        .sum()
}

pub fn negbin_logpmf(r: f32, lgamma_r: f32, p: f32, k: u32) -> f32 {
    const MINP: f32 = 0.999999_f32;
    let p = p.min(MINP);

    if k == 0 {
        // handle common case in sparse data efficiently
        r * (-p).ln_1p()
    } else {
        let k_ln_factorial = lgammaf(k as f32 + 1.0);
        let lgamma_rpk = lgammaf(r + k as f32);
        lgamma_rpk - lgamma_r - k_ln_factorial + (k as f32) * p.ln() + r * (-p).ln_1p()
    }
}

// fn normal_cdf(μ: f32, σ: f32, x: f32) -> f32 {
//     return 0.5 * (1.0 + erff((x - μ) / (SQRT2 * σ)));
// }

// fn std_normal_cdf(σ: f32, x: f32) -> f32 {
//     0.5 * (1.0 + erff(x / (f32::consts::SQRT_2 * σ)))
// }

pub fn normal_logpdf(μ: f32, σ: f32, x: f32) -> f32 {
    -LN_SQRT_TWO_PI - σ.ln() - ((x - μ) / σ).powi(2) / 2.0
}

// pub fn lognormal_logpdf(μ: f32, σ: f32, x: f32) -> f32 {
//     let xln = x.ln();
//     -LN_SQRT_TWO_PI - σ.ln() - xln - ((xln - μ) / σ).powi(2) / 2.0
// }

pub fn randn(rng: &mut ThreadRng) -> f32 {
    rng.sample::<f32, StandardNormal>(StandardNormal)
}

pub fn halfnormal_logpdf(σ: f32, x: f32) -> f32 {
    -LN_SQRT_TWO_PI - σ.ln() - x.powi(2) / (2.0 * σ.powi(2))
}

pub fn halfnormal_x2_pdf(σ: f32, x2: f32) -> f32 {
    (SQRT_2_DIV_SQRT_PI / σ) * (-x2 / (2.0 * σ.powi(2))).exp()
}

fn erfint(span: f32, σ: f32) -> f32 {
    -span * erff(span / (f32::consts::SQRT_2 * σ))
        - (f32::consts::SQRT_2 * f32::consts::FRAC_2_SQRT_PI / 2.0)
            * σ
            * (-span.powi(2) / (2.0 * σ.powi(2))).exp()
}

// Suppose x - x0 ~ N(0, σ), yet x and x0 are measured imprecisely, where we only know that
// x0 ∈ [a0, b0]
// x ∈ [a, b]
// This function integrates the N() prior over the uncertain placement of x and x0
pub fn uniformly_imprecise_normal_prob(a: f32, b: f32, a0: f32, b0: f32, σ: f32) -> f32 {
    0.5 * (b0 - a0).recip()
        * (b - a).recip()
        * (erfint(b - b0, σ) + erfint(a - a0, σ) - erfint(b - a0, σ) - erfint(a - b0, σ))
}

// Multinomial iterator
pub struct MultinomialSampler<'a> {
    rng: &'a mut ThreadRng,
    probs: &'a [f64],
    ρ: f64, // remaining probability
    s: u32, // remaining count
    i: usize,
}

impl<'a> MultinomialSampler<'a> {
    pub fn new(rng: &'a mut ThreadRng, probs: &'a [f64], n: u32) -> Self {
        MultinomialSampler {
            rng,
            probs,
            ρ: probs.iter().sum(),
            s: n,
            i: 0,
        }
    }
}

impl<'a> Iterator for MultinomialSampler<'a> {
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
        let x = Binomial::new(self.s as u64, r).unwrap().sample(self.rng) as u32;
        self.ρ -= pi;
        self.s -= x;
        self.i += 1;

        Some(x)
    }
}
