// use libm::{lgammaf, erff};
use libm::lgammaf;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;

// pub fn logit(p: f32) -> f32 {
//     return p.ln() - (1.0 - p).ln();
// }

pub fn sq(x: f32) -> f32 {
    x * x
}

pub fn logistic(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn relerr(a: f32, b: f32) -> f32 {
    ((a - b) / a).abs()
}

pub fn lfact(k: u32) -> f32 {
    lgammaf(k as f32 + 1.0)
}

// // Partial Student-T log-pdf (just the terms that don't cancel out when doing MH sampling)
// pub fn studentt_logpdf_part(σ2: f32, df: f32, x2: f32) -> f32 {
//     return -((df + 1.0) / 2.0) * ((x2 / σ2) / df).ln_1p();
// }

const SQRT_TWO_PI: f32 = 2.506_628_3_f32;
const LN_SQRT_TWO_PI: f32 = 0.918_938_5_f32;

pub fn normal_x2_pdf(σ: f32, x2: f32) -> f32 {
    (-x2 / (2.0 * σ.powi(2))).exp() / (σ * SQRT_TWO_PI)
}

pub fn normal_x2_logpdf(σ: f32, x2: f32) -> f32 {
    -x2 / (2.0 * σ.powi(2)) - σ.ln() - LN_SQRT_TWO_PI
}

// pub fn negbin_logpmf(r: f32, lgamma_r: f32, p: f32, k: u32) -> f32 {
//     let k_ln_factorial = lgammaf(k as f32 + 1.0);
//     let lgamma_rpk = lgammaf(r + k as f32);
//     return negbin_logpmf_fast(r, lgamma_r, lgamma_rpk, p, k, k_ln_factorial);
// }

// const SQRT2: f32 = 1.4142135623730951_f32;

// pub fn normal_cdf(μ: f32, σ: f32, x: f32) -> f32 {
//     return 0.5 * (1.0 + erff((x - μ) / (SQRT2 * σ)));
// }

pub fn normal_pdf(μ: f32, σ: f32, x: f32) -> f32 {
    let xμ = x - μ;
    (-xμ.powi(2) / (2.0 * σ.powi(2))).exp() / (σ * SQRT_TWO_PI)
}

pub fn lognormal_logpdf(μ: f32, σ: f32, x: f32) -> f32 {
    let xln = x.ln();
    -LN_SQRT_TWO_PI - σ.ln() - xln - ((xln - μ) / σ).powi(2) / 2.0
}

// Negative binomial log probability function with capacity for precomputing some values.
pub fn negbin_logpmf_fast(
    r: f32,
    lgamma_r: f32,
    lgamma_rpk: f32,
    p: f32,
    k: u32,
    k_ln_factorial: f32,
) -> f32 {
    const MINP: f32 = 0.999999_f32;
    let p = p.min(MINP);

    if k == 0 {
        // handle common case in sparse data efficiently
        // r * (1.0 - p).ln()
        r * (-p).ln_1p()
    } else {
        lgamma_rpk
            - lgamma_r
            - k_ln_factorial
            // + (k as f32) * p.ln() + r * (1.0 - p).ln()
            + (k as f32) * p.ln() + r * (-p).ln_1p()
    }
}

pub fn randn(rng: &mut ThreadRng) -> f32 {
    return rng.sample::<f32, StandardNormal>(StandardNormal);
}

pub fn rand_crt(rng: &mut ThreadRng, n: u32, r: f32) -> u32 {
    (0..n)
        .map(|t| rng.gen_bool(r as f64 / (r as f64 + t as f64)) as u32)
        .sum()
}

// log-factorial with precomputed values for small numbers
pub struct LogFactorial {
    values: Vec<f32>,
}

impl LogFactorial {
    fn new_with_n(n: usize) -> Self {
        LogFactorial {
            values: Vec::from_iter((0..n as u32).map(lfact)),
        }
    }

    pub fn new() -> Self {
        LogFactorial::new_with_n(100)
    }

    pub fn eval(&self, k: u32) -> f32 {
        self.values
            .get(k as usize)
            .map_or_else(|| lfact(k), |&value| value)
    }
}

// Partially memoized lgamma(r + k), memoized over k.
#[derive(Clone)]
pub struct LogGammaPlus {
    r: f32,
    values: Vec<f32>,
}

impl LogGammaPlus {
    fn new_with_n(r: f32, n: usize) -> Self {
        LogGammaPlus {
            r,
            values: Vec::from_iter((0..n as u32).map(|x| lgammaf(r + x as f32))),
        }
    }

    pub fn new(r: f32) -> Self {
        LogGammaPlus::new_with_n(r, 100)
    }

    pub fn default() -> Self {
        LogGammaPlus::new(0.0)
    }

    pub fn reset(&mut self, r: f32) {
        self.values.iter_mut().enumerate().for_each(|(k, v)| {
            *v = lgammaf(r + k as f32);
        });
        self.r = r;
    }

    pub fn eval(&self, k: u32) -> f32 {
        self.values
            .get(k as usize)
            .map_or_else(|| lgammaf(self.r + k as f32), |&value| value)
    }
}
