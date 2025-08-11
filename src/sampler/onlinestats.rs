use num::{Signed, Zero, cast::AsPrimitive};
use rayon::prelude::*;
use std::f32;

use super::sparsemat::SparseMat;

// Bookkeeping for P² algorithm
#[derive(Clone, Copy)]
struct P2Values {
    h: [f32; 5], // estimates
    n: [f32; 5], // positions
}

impl P2Values {
    fn new() -> P2Values {
        P2Values {
            h: [f32::INFINITY; 5],
            n: [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    }

    fn update(&mut self, t: usize, dn: &[f32; 5], obs: f32) {
        // if we are still on the first 5 observations, just initialize
        if t < 5 {
            self.h[t] = obs;

            // slide the observation left until we find its place
            let mut j = t;
            while j > 0 {
                if self.h[j] < self.h[j - 1] {
                    self.h.swap(j, j - 1);
                    j -= 1;
                } else {
                    break;
                }
            }

            return;
        }

        let k = if obs < self.h[0] {
            self.h[0] = obs;
            1
        } else if obs < self.h[1] {
            1
        } else if obs < self.h[2] {
            2
        } else if obs < self.h[3] {
            3
        } else if obs <= self.h[4] {
            4
        } else {
            self.h[4] = obs;
            4
        };

        // increment position markers
        for n_i in &mut self.n[k..5] {
            *n_i += 1.0;
        }

        // adjust marker heights
        #[allow(clippy::needless_range_loop)]
        for i in 1..4 {
            let delta = dn[i] - self.n[i];
            if (delta >= 1.0 && self.n[i + 1] - self.n[i] > 1.0)
                || (delta <= -1.0 && self.n[i - 1] - self.n[i] < -1.0)
            {
                let d = delta.signum();

                // parabolic predictor
                let h_adj = self.h[i]
                    + (d / (self.n[i + 1] - self.n[i - 1]))
                        * ((self.n[i] - self.n[i - 1] + d) * (self.h[i + 1] - self.h[i])
                            / (self.n[i + 1] - self.n[i])
                            + (self.n[i + 1] - self.n[i] - d) * (self.h[i] - self.h[i - 1])
                                / (self.n[i] - self.n[i - 1]));

                if self.h[i - 1] < h_adj && h_adj < self.h[i + 1] {
                    self.h[i] = h_adj;
                } else {
                    // linear prediction
                    if delta.is_positive() {
                        self.h[i] += (self.h[i + 1] - self.h[i]) / (self.n[i + 1] - self.n[i]);
                    } else {
                        self.h[i] -= (self.h[i] - self.h[i - 1]) / (self.n[i] - self.n[i - 1]);
                    }
                }

                if delta.is_positive() {
                    self.n[i] += 1.0;
                } else {
                    self.n[i] -= 1.0;
                }
            }
        }
    }

    fn estimate(&self) -> f32 {
        self.h[2]
    }
}

pub struct ScalarQuantileEstimator {
    estimates: P2Values,
    dn: [f32; 5],
    quantile: f32,
    t: usize,
}

impl ScalarQuantileEstimator {
    pub fn new(quantile: f32) -> Self {
        ScalarQuantileEstimator {
            estimates: P2Values::new(),
            dn: [
                1.0,
                1.0 + 2.0 * quantile,
                1.0 + 4.0 * quantile,
                3.0 + 2.0 * quantile,
                5.0,
            ],
            quantile,
            t: 0,
        }
    }

    pub fn update(&mut self, value: f32) {
        // increment desired positions
        self.dn[1] += self.quantile / 2.0;
        self.dn[2] += self.quantile;
        self.dn[3] += (1.0 + self.quantile) / 2.0;
        self.dn[4] += 1.0;
        self.estimates.update(self.t, &self.dn, value);
        self.t += 1;
    }

    pub fn estimate(&self) -> f32 {
        self.estimates.estimate()
    }
}

// Estimating transcript count quantiles using the P² algorithm
pub struct CountQuantileEstimator {
    estimates: SparseMat<P2Values, u32>,
    dn: [f32; 5], // "desired" positions
    quantile: f32,
    t: usize, // iteration number
}

impl CountQuantileEstimator {
    pub fn new(m: usize, n: usize, quantile: f32, shardsize: usize) -> Self {
        CountQuantileEstimator {
            estimates: SparseMat::zeros(m, n as u32 - 1, shardsize),
            dn: [
                1.0,
                1.0 + 2.0 * quantile,
                1.0 + 4.0 * quantile,
                3.0 + 2.0 * quantile,
                5.0,
            ],
            quantile,
            t: 0,
        }
    }

    pub fn update<T>(&mut self, counts: &SparseMat<T, u32>)
    where
        T: AsPrimitive<f32> + Sync + Send + Zero,
    {
        assert!(self.estimates.m == counts.m);
        assert!(self.estimates.n == counts.n);

        // increment desired positions
        self.dn[1] += self.quantile / 2.0;
        self.dn[2] += self.quantile;
        self.dn[3] += (1.0 + self.quantile) / 2.0;
        self.dn[4] += 1.0;

        self.estimates
            .par_rows()
            .zip(counts.par_rows())
            .for_each(|(estimates_c, counts_c)| {
                // update estimates with estimates entries with latest counts
                let counts_c_lock = counts_c.read();
                let mut estimates_c_lock = estimates_c.write();
                for (gene, estimates_cg) in estimates_c_lock.iter_nonzeros_mut() {
                    let count_cg = counts_c_lock.get(gene);
                    estimates_cg.update(self.t, &self.dn, count_cg.as_());
                }

                // update estimates where there is no entry
                for (gene, count_cg) in counts_c_lock.iter_nonzeros() {
                    estimates_c_lock.update(gene, P2Values::new, |est| {
                        est.update(self.t, &self.dn, count_cg.as_())
                    });
                }
            });

        self.t += 1;
    }
}

pub struct CountMeanEstimator {
    pub estimates: SparseMat<f32, u32>,
    t: usize, // iteration number
}

impl CountMeanEstimator {
    pub fn new(m: usize, n: usize, shardsize: usize) -> Self {
        CountMeanEstimator {
            estimates: SparseMat::zeros(m, n as u32 - 1, shardsize),
            t: 0,
        }
    }

    pub fn update<T>(&mut self, counts: &SparseMat<T, u32>)
    where
        T: AsPrimitive<f32> + Sync + Send + Zero,
    {
        self.t += 1;
        let t = self.t as f32;
        self.estimates
            .par_rows()
            .zip(counts.par_rows())
            .for_each(|(estimates_c, counts_c)| {
                let counts_c_lock = counts_c.read();
                let mut estimates_c_lock = estimates_c.write();
                for (gene, count_cg) in counts_c_lock.iter().enumerate() {
                    if count_cg.is_zero() {
                        estimates_c_lock.update_if_present(gene as u32, |est| {
                            *est += (count_cg.as_() - *est) / t
                        });
                    } else {
                        estimates_c_lock.update(
                            gene as u32,
                            || 0.0,
                            |est| *est += (count_cg.as_() - *est) / t,
                        );
                    }
                }
            });
    }
}
