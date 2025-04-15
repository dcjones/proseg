use std::u32;

use super::math::{negbin_logpmf, normal_logpdf, odds_to_prob};
use super::{ModelParams, ModelPriors};
use itertools::{izip, Itertools};
use libm::lgammaf;
use log::trace;
use ndarray::{s, Array1, Array2, Axis, Zip};
use rand::{rng, Rng};
use rand_distr::{Binomial, Distribution, Gamma, Normal, StandardNormal};
use rayon::prelude::*;
use std::cell::RefCell;
use std::time::Instant;

// Setting parallel iterator min length for simple operations
const SIMPLE_PAR_ITER_MIN_LEN: usize = 64;

pub struct ParamSampler {}

impl ParamSampler {
    fn sample(&self, priors: &ModelPriors, params: &mut ModelParams) {
        todo!("Call sampling functions one by one,");
    }

    fn sample_volume_params(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // Parallelization overhead here may outweight the reward. Alternative
        // strategy may be to process this one shard at a time and parallelize within
        // shards.
        params
            .log_cell_volume
            .iter_mut()
            .zip(params.cell_volume.iter())
            .par_bridge()
            .for_each(|(log_cell_volume_c, cell_volume_c)| {
                *log_cell_volume_c = (cell_volume_c as f32 * params.voxel_volume).ln();
            });

        // compute sample means
        params.μ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.log_cell_volume)
            .for_each(|&z, &log_volume| {
                params.μ_volume[z as usize] += log_volume;
            });

        // sample μ parameters
        Zip::from(&mut params.μ_volume)
            .and(&params.σ_volume)
            .and(&params.component_population)
            .into_par_iter()
            .for_each_init(
                || rng(),
                |rng, (μ, &σ, &pop)| {
                    let v = (1_f32 / priors.σ_μ_volume.powi(2) + pop as f32 / σ.powi(2)).recip();
                    *μ = Normal::new(
                        v * (priors.μ_μ_volume / priors.σ_μ_volume.powi(2) + *μ / σ.powi(2)),
                        v.sqrt(),
                    )
                    .unwrap()
                    .sample(rng);
                },
            );

        // compute sample variances
        params.σ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.log_cell_volume)
            .for_each(|&z, &log_volume| {
                params.σ_volume[z as usize] += (params.μ_volume[z as usize] - log_volume).powi(2);
            });

        Zip::from(&mut params.σ_volume)
            .and(&params.component_population)
            .into_par_iter()
            .for_each_init(
                || rng(),
                |rng, (σ, &pop)| {
                    *σ = Gamma::new(
                        priors.α_σ_volume + (pop as f32) / 2.0,
                        (priors.β_σ_volume + *σ / 2.0).recip(),
                    )
                    .unwrap()
                    .sample(rng)
                    .recip()
                    .sqrt();
                },
            );
    }

    fn sample_foreground_background(&self, params: &mut ModelParams, purge: bool) {
        // overwrite params.background_counts with params.unassigned_counts
        for (b_lyr, u_lyr) in params
            .background_counts
            .iter_mut()
            .zip(&params.unassigned_counts)
        {
            b_lyr.copy_from(u_lyr);
        }

        if purge {
            params.foreground_counts.clear();
        } else {
            params.foreground_counts.zero();
        }

        // Generate binomial samples to split cell counts into foreground and background
        params
            .counts
            .rows()
            .zip(params.foreground_counts.rows())
            .enumerate()
            .for_each_init(
                || rng(),
                |rng, (cell, (row, foreground_row))| {
                    let mut foreground_row = foreground_row.write();

                    let mut λ_cg = 0.0;
                    let mut gene = u32::MAX;

                    for (gene_layer, count) in row.read().iter_nonzeros() {
                        if gene_layer.gene != gene {
                            λ_cg = params.λ(cell, gene as usize);
                            gene = gene_layer.gene;
                        }
                        let λ_bg = params.λ_bg[[gene as usize, gene_layer.layer as usize]];

                        let count_fg = Binomial::new(count as u64, (λ_cg / (λ_cg + λ_bg)) as f64)
                            .unwrap()
                            .sample(rng) as u32;
                        let count_bg = count - count_fg;

                        foreground_row.add(gene, count_fg);
                        params.background_counts[gene_layer.layer as usize]
                            .add(gene as usize, count_bg);
                    }
                },
            );
    }

    fn sample_factor_model(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        sample_z: bool,
        burnin: bool,
    ) {
        let t0 = Instant::now();
        self.sample_latent_counts(params);
        trace!("sample_latent_counts: {:?}", t0.elapsed());

        if sample_z {
            let t0 = Instant::now();
            self.sample_z(params);
            trace!("sample_z: {:?}", t0.elapsed());
        }
        self.sample_π(params);

        if priors.use_factorization {
            let t0 = Instant::now();
            self.sample_θ(priors, params);
            trace!("sample_θ: {:?}", t0.elapsed());
        }

        let t0 = Instant::now();
        self.sample_φ(params);
        trace!("sample_φ: {:?}", t0.elapsed());

        let t0 = Instant::now();
        if let Some(dispersion) = priors.dispersion {
            params.rφ.fill(dispersion);
        } else if burnin && priors.burnin_dispersion.is_some() {
            let dispersion = priors.burnin_dispersion.unwrap();
            params.rφ.fill(dispersion);
        } else {
            self.sample_rφ(priors, params);
        }
        trace!("sample_rφ: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_ωck(params);
        self.sample_sφ(priors, params);
        trace!("sample_sφ: {:?}", t0.elapsed());

        if priors.use_cell_scales {
            self.sample_cell_scales(priors, params);
        } else {
            params
                .effective_cell_volume
                .iter_mut()
                .zip(params.cell_volume.iter())
                .for_each(|(ev, v)| {
                    *ev = (v as f32) * params.voxel_volume;
                });
        }
    }

    fn sample_z(&self, params: &mut ModelParams) {
        Zip::from(&mut params.lgamma_rφ)
            .and(&params.rφ)
            .into_par_iter()
            .with_min_len(SIMPLE_PAR_ITER_MIN_LEN)
            .for_each(|(lgamma_r_tk, r_tk)| {
                *lgamma_r_tk = lgammaf(*r_tk);
            });

        let ncomponents = params.ncomponents();
        Zip::indexed(&mut params.z) // for each cell
            .and(params.φ.rows())
            .and(&params.effective_cell_volume)
            .and(&params.log_cell_volume)
            .into_par_iter()
            .for_each_init(
                || rng(),
                |rng, (i, z_c, φ_c, ev_c, log_v_c)| {
                    let x_c_lock = params.cell_latent_counts.row(i);
                    let x_c = x_c_lock.read();
                    let mut z_probs = params
                        .z_probs
                        .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                        .borrow_mut();

                    // compute probability of φ_c under every component

                    // for every component
                    let mut z_probs_sum = 0.0;
                    for (z_probs_t, log_π_t, r_t, lgamma_r_t, s_t, μ_vol_c, σ_vol_c) in izip!(
                        z_probs.iter_mut(),
                        params.log_π.iter(),
                        params.rφ.rows(),
                        params.lgamma_rφ.rows(),
                        params.sφ.rows(),
                        &params.μ_volume,
                        &params.σ_volume
                    ) {
                        *z_probs_t = *log_π_t as f64;

                        // for every hidden dim
                        for (r_tk, lgamma_r_tk, s_tk, θ_k_sum, x_ck) in
                            izip!(r_t, lgamma_r_t, s_t, &params.θksum, x_c.iter())
                        {
                            let p = odds_to_prob(*s_tk * *ev_c * *θ_k_sum);
                            let lp = negbin_logpmf(*r_tk, *lgamma_r_tk, p, x_ck) as f64;
                            *z_probs_t += lp;
                        }

                        *z_probs_t += normal_logpdf(*μ_vol_c, *σ_vol_c, *log_v_c) as f64;
                    }

                    for z_probs_t in z_probs.iter_mut() {
                        *z_probs_t = z_probs_t.exp();
                        z_probs_sum += *z_probs_t;
                    }

                    if !z_probs_sum.is_finite() {
                        dbg!(&z_probs, &φ_c, z_probs_sum);
                    }

                    // cumulative probabilities in-place
                    z_probs.iter_mut().fold(0.0, |mut acc, x| {
                        acc += *x / z_probs_sum;
                        *x = acc;
                        acc
                    });

                    let u = rng.random::<f64>();
                    *z_c = z_probs.partition_point(|x| *x < u) as u32;
                },
            );
    }

    fn sample_θ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        todo!();
    }

    fn sample_φ(&self, params: &mut ModelParams) {
        todo!();
    }

    fn sample_cell_scales(&self, priors: &ModelPriors, params: &mut ModelParams) {
        todo!();
    }

    fn sample_π(&self, params: &mut ModelParams) {
        todo!();
    }

    fn sample_latent_counts(&self, params: &mut ModelParams) {
        todo!();
    }

    fn sample_rφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        todo!();
    }

    fn sample_ωck(&self, params: &mut ModelParams) {
        todo!();
    }

    fn sample_sφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        todo!();
    }

    fn sample_background_rates(&self, priors: &ModelPriors, params: &ModelParams) {
        todo!();
    }

    fn sample_transcript_positions(&self, priors: &ModelPriors, params: &ModelParams) {
        // TODO: PRobably this should live in VoxelSampler, or maybe a different type entirely.
        todo!();
    }
}
