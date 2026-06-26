// use num_traits::{cast::FromPrimitive, float::Float};
use numeric_literals::replace_float_literals;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Normal, StandardNormal, StandardUniform};

mod float;
use float::Float;

mod common;
mod saddlepoint;
use saddlepoint::sample_polyagamma_saddlepoint;

mod alternate;
use alternate::sample_polyagamma_alternate;

pub struct PolyaGamma<T: Float> {
    h: T,
    z: T,
}

#[replace_float_literals(T::from(literal).unwrap())]
fn sech<T: Float>(x: T) -> T {
    1.0 / x.cosh()
}

impl<T: Float> PolyaGamma<T>
where
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
    Exp1: Distribution<T>,
{
    pub fn new(h: T, z: T) -> Self {
        let eps = T::from(1e-4).unwrap();
        // if h.is_sign_negative() {
        if h < eps {
            panic!("h must be positive (and not too small)")
        }

        Self { h, z }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn mean(&self) -> T {
        if self.z == T::zero() {
            self.h / 4.0
        } else {
            self.h * 0.5 * self.z.recip() * (0.5 * self.z).tanh()
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn var(&self) -> T {
        if self.z == T::zero() {
            self.h / 24.0
        } else if self.z.sinh().is_infinite() {
            self.h
                * 0.25
                * (self.z.powi(3).recip() * 2.0 * self.z.signum()
                    - self.z.recip().powi(2) * sech(0.5 * self.z).powi(2))
        } else {
            self.h
                * 0.25
                * self.z.powi(3).recip()
                * (self.z.sinh() - self.z)
                * sech(0.5 * self.z).powi(2)
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
        assert!(self.h > T::zero(), "h must be non-negative");

        // if self.h >= 50.0 {
        //     return self.sample_normal(rng);
        // } else {
        //     return self.sample_saddlepoint(rng);
        // };

        if self.h >= 50.0 {
            self.sample_normal(rng)
        } else if self.h >= 8.0 || (self.h > 4.0 && self.z <= 4.0) {
            self.sample_saddlepoint(rng)
        } else {
            self.sample_alternate(rng)
        }

        // } else if self.h >= 8.0 || (self.h > 4.0 && self.z <= 4.0) {
        //     return self.sample_saddlepoint(rng);
        // } else if self.h == 1.0 || (self.h == self.h.floor() && self.z <= 1.0) {
        //     return self.sample_devroye(rng);
        // } else {
        //     return self.sample_alternate(rng);
        // }
    }

    fn sample_normal<R: Rng>(&self, rng: &mut R) -> T {
        Normal::new(self.mean(), self.var().sqrt())
            .unwrap()
            .sample(rng)
    }

    fn sample_saddlepoint<R: Rng>(&self, rng: &mut R) -> T {
        sample_polyagamma_saddlepoint(rng, self.h, self.z)
    }

    fn sample_alternate<R: Rng>(&self, rng: &mut R) -> T {
        T::from(sample_polyagamma_alternate(
            rng,
            self.h.as_f64(),
            self.z.as_f64(),
        ))
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Generate n samples from PG(h, z) and assert that the empirical mean and
    // variance are consistent with the analytical formulas.
    //
    // Mean tolerance: 5 standard errors (CLT-based, ~1-in-3.5M false positive).
    // Variance tolerance: 15× the normal-theory SE, which is generous enough to
    // absorb the excess kurtosis of the PG distribution without masking real bugs.
    fn check_moments(h: f64, z: f64, n: usize) {
        let mut rng = rand::rng();
        let pg = PolyaGamma::<f64>::new(h, z);

        let expected_mean = pg.mean();
        let expected_var = pg.var();

        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        for _ in 0..n {
            let x: f64 = pg.sample(&mut rng);
            sum += x;
            sum_sq += x * x;
        }

        let nf = n as f64;
        let empirical_mean = sum / nf;
        let empirical_var = (sum_sq - sum * sum / nf) / (nf - 1.0);

        let mean_tol = 5.0 * expected_var.sqrt() / nf.sqrt();
        let var_tol = 15.0 * expected_var * (2.0 / (nf - 1.0)).sqrt();

        assert!(
            (empirical_mean - expected_mean).abs() < mean_tol,
            "PG({h},{z}): empirical mean {:.6} vs analytical {:.6}, diff {:.2e} (tol {:.2e})",
            empirical_mean,
            expected_mean,
            (empirical_mean - expected_mean).abs(),
            mean_tol,
        );

        assert!(
            (empirical_var - expected_var).abs() < var_tol,
            "PG({h},{z}): empirical var {:.6} vs analytical {:.6}, diff {:.2e} (tol {:.2e})",
            empirical_var,
            expected_var,
            (empirical_var - expected_var).abs(),
            var_tol,
        );
    }

    // alternate path: h < 8 AND (h <= 4 OR z > 4)
    #[test]
    fn moments_alternate() {
        for (h, z) in [
            (1.0_f64, 0.0_f64), // z=0 special case
            (1.0, 1.0),
            (1.0, 5.0),
            (2.0, 0.0),
            (2.0, 2.0),
            (3.0, 5.0),
            (4.0, 0.0),
            (4.0, 8.0),
            (7.0, 8.0), // h > 4 but z > 4, so still alternate
        ] {
            check_moments(h, z, 50_000);
        }
    }

    // saddlepoint path: h < 50 AND (h >= 8 OR (h > 4 AND z <= 4))
    #[test]
    fn moments_saddlepoint() {
        for (h, z) in [
            (5.0_f64, 0.0_f64), // h > 4, z <= 4
            (5.0, 3.0),
            (7.0, 3.0),
            (8.0, 0.0), // h >= 8
            (8.0, 5.0),
            (15.0, 10.0),
            (30.0, 0.0),
            (49.0, 20.0),
        ] {
            check_moments(h, z, 50_000);
        }
    }

    // normal approximation path: h >= 50
    #[test]
    fn moments_normal_approx() {
        for (h, z) in [
            (50.0_f64, 0.0_f64),
            (50.0, 5.0),
            (100.0, 10.0),
        ] {
            check_moments(h, z, 50_000);
        }
    }

    // Two-sample Kolmogorov-Smirnov test against reference quantiles produced by
    // the Python `polyagamma` package (tests/pg_reference.py).  Covers the two
    // exact sampling paths (alternate and saddlepoint); the normal-approximation
    // path (h >= 50) is intentionally approximate and is covered by moment tests.
    //
    // For a one-sample KS test with n = 100_000 at α = 0.001, the critical value
    // is K_{0.001} / sqrt(n) where K_{0.001} ≈ 1.95, giving D_crit ≈ 0.00617.
    // Any real bias or scale error should produce a KS statistic many times larger.
    // Run with: cargo test --release ks_test_against_reference -- --ignored
    // First generate the reference file: python3 tests/pg_reference.py
    #[test]
    #[ignore]
    fn ks_test_against_reference() {
        let json_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/pg_reference_quantiles.json");

        let json_str = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
            panic!("Could not read {}: {e}", json_path.display())
        });

        let cases: Vec<serde_json::Value> = serde_json::from_str(&json_str)
            .expect("Failed to parse pg_reference_quantiles.json");

        let n: usize = 100_000;
        // KS critical value: K_{0.001} / sqrt(n)
        let ks_critical = 1.95_f64 / (n as f64).sqrt();

        let mut rng = rand::rng();

        for case in &cases {
            let h = case["h"].as_f64().unwrap();
            let z = case["z"].as_f64().unwrap();
            let path = case["path"].as_str().unwrap();

            let probs: Vec<f64> = case["probs"]
                .as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap())
                .collect();
            let ref_quantiles: Vec<f64> = case["quantiles"]
                .as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap())
                .collect();

            let pg = PolyaGamma::<f64>::new(h, z);
            let mut samples: Vec<f64> = (0..n).map(|_| pg.sample(&mut rng)).collect();
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // For each reference quantile q at nominal probability p, compute
            // the empirical CDF of our samples at q, then take the max deviation.
            let ks = probs.iter().zip(ref_quantiles.iter())
                .map(|(&p, &q)| {
                    let empirical_p = samples.partition_point(|&x| x <= q) as f64 / n as f64;
                    (empirical_p - p).abs()
                })
                .fold(0.0_f64, f64::max);

            assert!(
                ks < ks_critical,
                "PG({h},{z}) [{path}]: KS statistic {ks:.4e} exceeds critical value \
                 {ks_critical:.4e} (α=0.001, n={n})",
            );
        }
    }
}
