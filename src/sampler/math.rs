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
// const SQRT_2_DIV_SQRT_PI: f32 = 0.797_884_6_f32;

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

// Digamma function ψ(x) = d/dx ln Γ(x), for x > 0. Recurrence up to x ≥ 6 then
// the standard asymptotic expansion; accurate to well within f32 precision.
pub fn digamma(mut x: f32) -> f32 {
    let mut result = 0.0_f32;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    result + x.ln() - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0)))
}

// Expected Chinese-restaurant-table count: E[CRT(n, r)] = Σ_{t=0}^{n-1} r/(r+t)
// = r·(ψ(r+n) − ψ(r)), extended continuously to fractional n. The deterministic
// (EM) counterpart of the stochastic `rand_crt` draw used by the Gibbs sampler
// for the metagene-dispersion (rφ) update.
pub fn expected_crt(n: f32, r: f32) -> f32 {
    if n <= 0.0 {
        0.0
    } else {
        r * (digamma(r + n) - digamma(r))
    }
}

// Continuous generalization of `negbin_logpmf` accepting a fractional count `k`
// (an EM expected count). Identical to the integer form with `k!` replaced by
// Γ(k+1); reduces to `negbin_logpmf` at integer `k`.
pub fn negbin_logpmf_f(r: f32, lgamma_r: f32, p: f32, k: f32) -> f32 {
    const MINP: f32 = 0.999999_f32;
    let p = p.min(MINP);

    if k == 0.0 {
        r * (-p).ln_1p()
    } else {
        let lgamma_kp1 = lgammaf(k + 1.0);
        let lgamma_rpk = lgammaf(r + k);
        lgamma_rpk - lgamma_r - lgamma_kp1 + k * p.ln() + r * (-p).ln_1p()
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

// pub fn halfnormal_x2_pdf(σ: f32, x2: f32) -> f32 {
//     (SQRT_2_DIV_SQRT_PI / σ) * (-x2 / (2.0 * σ.powi(2))).exp()
// }

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

#[cfg(test)]
mod tests {
    use super::*;
    use libm::lgammaf;

    // Reference values from scipy.stats.nbinom.logpmf(k, r, 1-p).
    // Convention: k successes before r failures, p = per-trial success probability.
    #[test]
    fn negbin_logpmf_values() {
        let cases: &[(f32, f32, u32, f32)] = &[
            (1.0,  0.3,       0,  -0.35667494),
            (1.0,  0.3,       5,  -6.376_538_8),
            (2.5,  0.4,       0,  -1.277_064_1),
            (2.5,  0.4,       3,  -2.144_564_6),
            (2.5,  0.4,      10,  -7.094_719_4),
            (0.5,  0.1,       0,  -0.05268026),
            (0.5,  0.1,       2,  -5.638_679_5),
            (10.0, 0.8,       0, -16.094_38),
            (10.0, 0.8,       7,  -8.311_513),
            // p at the MINP clamp boundary — reference computed with f32 libm
            // (0.999999_f32 != 1 - 1e-6 exactly, so f64 scipy values don't apply)
            (2.0,  0.999999,  0, -27.604_637),
            (2.0,  0.999999,  1, -26.911_491),
        ];

        for &(r, p, k, expected) in cases {
            let result = negbin_logpmf(r, lgammaf(r), p, k);
            assert!(
                (result - expected).abs() < 1e-4,
                "negbin_logpmf(r={r}, p={p}, k={k}): got {result:.8}, expected {expected:.8}",
            );
        }
    }

    // E[CRT(n, r)] = Σ_{t=0}^{n-1} r/(r+t)
    // Var[CRT(n, r)] = Σ_{t=0}^{n-1} (r/(r+t)) · (t/(r+t))   [independent Bernoullis]
    #[test]
    fn rand_crt_mean() {
        let n_samples = 50_000_usize;
        let mut rng = rand::rng();

        for (n, r) in [(1_u32, 1.0_f32), (5, 1.0), (3, 2.0), (10, 0.5), (8, 3.0)] {
            let expected_mean: f32 = (0..n).map(|t| r / (r + t as f32)).sum();
            let variance: f32 = (0..n)
                .map(|t| { let p = r / (r + t as f32); p * (1.0 - p) })
                .sum();

            // n=1 is deterministic: the t=0 term is always Bernoulli(r/r) = 1
            if variance == 0.0 {
                assert_eq!(rand_crt(&mut rng, n, r), 1);
                continue;
            }

            let total: u32 = (0..n_samples).map(|_| rand_crt(&mut rng, n, r)).sum();
            let empirical_mean = total as f32 / n_samples as f32;
            let tol = 5.0 * variance.sqrt() / (n_samples as f32).sqrt();

            assert!(
                (empirical_mean - expected_mean).abs() < tol,
                "rand_crt(n={n}, r={r}): mean {empirical_mean:.4} vs expected \
                 {expected_mean:.4} (tol {tol:.4})",
            );
        }
    }
}
