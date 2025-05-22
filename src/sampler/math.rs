use libm::{erff, lgammaf};
use rand::Rng;
use rand::rngs::ThreadRng;
use rand_distr::StandardNormal;
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

fn std_normal_cdf(σ: f32, x: f32) -> f32 {
    0.5 * (1.0 + erff(x / (f32::consts::SQRT_2 * σ)))
}

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

// This is a Normal prior over transcript diffusion distance, integrating out
// uncertain transcript positions, given we have only voxel positions. The math
// is not remotely obvious, you'll just have to trust me.
pub fn normal_dist_inter_voxel_marginal(d_min: f32, s: f32, σ: f32) -> f32 {
    let a = normal_dist_inter_voxel_marginal_part(d_min, d_min + s, s, σ);
    let b = s * (std_normal_cdf(σ, d_min + 2.0 * s) - std_normal_cdf(σ, d_min + s))
        - normal_dist_inter_voxel_marginal_part(d_min + s, d_min + 2.0 * s, s, σ);
    a + b
}

fn normal_dist_inter_voxel_marginal_part(d_from: f32, d_to: f32, s: f32, σ: f32) -> f32 {
    let mut result = s * std_normal_cdf(σ, d_to) - (s / 2.0);
    let sqrt2_sigma = f32::consts::SQRT_2 * σ;
    result += 0.5 * (d_from * erff(d_from / sqrt2_sigma) - d_to * erff(d_to / sqrt2_sigma));
    result += 0.5
        * (sqrt2_sigma / f32::consts::PI.sqrt())
        * ((-(d_from / sqrt2_sigma).powi(2)).exp() - (-(d_to / sqrt2_sigma).powi(2)).exp());
    result
}

pub fn normal_dist_intra_voxel_marginal(s: f32, σ: f32) -> f32 {
    2.0 * s * (std_normal_cdf(σ, s) - std_normal_cdf(σ, 0.0))
        - 2.0 * normal_dist_inter_voxel_marginal_part(0.0, s, s, σ)
}
