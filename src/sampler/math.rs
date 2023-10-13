use libm::lgammaf;
use rand::rngs::ThreadRng;
use rand::Rng;

pub fn relerr(a: f32, b: f32) -> f32 {
    return ((a - b) / a).abs();
}

pub fn lfact(k: u32) -> f32 {
    return lgammaf(k as f32 + 1.0);
}

// pub fn negbin_logpmf(r: f32, lgamma_r: f32, p: f32, k: u32) -> f32 {
//     let k_ln_factorial = lgammaf(k as f32 + 1.0);
//     let lgamma_rpk = lgammaf(r + k as f32);
//     return negbin_logpmf_fast(r, lgamma_r, lgamma_rpk, p, k, k_ln_factorial);
// }

const LN_SQRT_TWO_PI: f32 = 0.9189385332046727_f32;

pub fn lognormal_logpdf(μ: f32, σ: f32, x: f32) -> f32 {
    let xln = x.ln();
    return -LN_SQRT_TWO_PI - σ.ln() - xln - ((xln - μ) / σ).powi(2) / 2.0;
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

    let result = if k == 0 {
        // handle common case in sparse data efficiently
        // r * (1.0 - p).ln()
        r * (-p).ln_1p()
    } else {
        lgamma_rpk
            - lgamma_r
            - k_ln_factorial
            // + (k as f32) * p.ln() + r * (1.0 - p).ln()
            + (k as f32) * p.ln() + r * (-p).ln_1p()
    };

    return result;
}

pub fn rand_crt(rng: &mut ThreadRng, n: u32, r: f32) -> u32 {
    return (0..n)
        .map(|t| rng.gen_bool(r as f64 / (r as f64 + t as f64)) as u32)
        .sum();
}

pub fn odds_to_prob(o: f32) -> f32 {
    return o / (1.0 + o);
}

pub fn prob_to_odds(p: f32) -> f32 {
    return p / (1.0 - p);
}

// log-factorial with precomputed values for small numbers
pub struct LogFactorial {
    values: Vec<f32>,
}

impl LogFactorial {
    fn new_with_n(n: usize) -> Self {
        return LogFactorial {
            values: Vec::from_iter((0..n as u32).map(|x| lfact(x))),
        };
    }

    pub fn new() -> Self {
        return LogFactorial::new_with_n(100);
    }

    pub fn eval(&self, k: u32) -> f32 {
        return self
            .values
            .get(k as usize)
            .map_or_else(|| lfact(k), |&value| value);
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
        return LogGammaPlus {
            r,
            values: Vec::from_iter((0..n as u32).map(|x| lgammaf(r + x as f32))),
        };
    }

    pub fn new(r: f32) -> Self {
        return LogGammaPlus::new_with_n(r, 100);
    }

    pub fn default() -> Self {
        return LogGammaPlus::new(0.0);
    }

    pub fn reset(&mut self, r: f32) {
        self.values.iter_mut().enumerate().for_each(|(k, v)| {
            *v = lgammaf(r + k as f32);
        });
        self.r = r;
    }

    pub fn eval(&self, k: u32) -> f32 {
        return self
            .values
            .get(k as usize)
            .map_or_else(|| lgammaf(self.r + k as f32), |&value| value);
    }
}
