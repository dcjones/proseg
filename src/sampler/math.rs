use libm::lgammaf;
use rand::rngs::ThreadRng;
use rand::Rng;
use rand_distr::StandardNormal;

// pub fn logit(p: f32) -> f32 {
//     return p.ln() - (1.0 - p).ln();
// }

// pub fn sq(x: f32) -> f32 {
//     x * x
// }

pub fn odds_to_prob(q: f32) -> f32 {
    q / (1.0 + q)
}

pub fn relerr(a: f32, b: f32) -> f32 {
    ((a - b) / a).abs()
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
    let k_ln_factorial = lgammaf(k as f32 + 1.0);
    let lgamma_rpk = lgammaf(r + k as f32);
    return negbin_logpmf_fast(r, lgamma_r, lgamma_rpk, p, k, k_ln_factorial);
}

fn negbin_logpmf_fast(
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

// const SQRT2: f32 = 1.4142135623730951_f32;

// pub fn normal_cdf(μ: f32, σ: f32, x: f32) -> f32 {
//     return 0.5 * (1.0 + erff((x - μ) / (SQRT2 * σ)));
// }

pub fn normal_logpdf(μ: f32, σ: f32, x: f32) -> f32 {
    -LN_SQRT_TWO_PI - σ.ln() - ((x - μ) / σ).powi(2) / 2.0
}

pub fn lognormal_logpdf(μ: f32, σ: f32, x: f32) -> f32 {
    let xln = x.ln();
    -LN_SQRT_TWO_PI - σ.ln() - xln - ((xln - μ) / σ).powi(2) / 2.0
}

pub fn randn(rng: &mut ThreadRng) -> f32 {
    return rng.sample::<f32, StandardNormal>(StandardNormal);
}
