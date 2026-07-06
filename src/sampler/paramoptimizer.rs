// Coordinate-ascent (EM / ICM) counterpart to `ParamSampler`. Where the sampler
// draws each parameter from its conjugate posterior, the optimizer takes the
// posterior *mode* (or mean, where the mode is boundary-degenerate) fed *expected*
// sufficient statistics, hill-climbing toward a high-probability configuration to
// report as the point estimate.
//
// This is "increment 2a": a HARD / ICM E-step. Each transcript count is assigned
// to its single most-likely metagene (argmax of φ_ck·θ_gk) rather than distributed
// by a multinomial draw, so the latent-count accumulators stay integer (`u32`) and
// every downstream consumer that requires integers (`rand_crt`, `negbin_logpmf`,
// the zero-inflation gate tests, the atomic-u32 gene accumulator) is reused
// unchanged. The dispersion / scale nuisance parameters (rφ, ωck, sφ, gate, ξ) are
// still drawn via the shared `ParamSampler` routines; their deterministic MAP
// updates are deferred to a later increment.
//
// The uncertainty phase (Phase B) continues to use `ParamSampler::sample`; the
// optimizer only runs during the point-estimate (Phase A) iterations.

use super::math::{negbin_logpmf, normal_logpdf, odds_to_prob};
use super::paramsampler::ParamSampler;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{TranscriptFixedState, Voxel, VoxelCheckerboard};
use super::{ModelParams, ModelPriors, RAYON_CELL_MIN_LEN, TranscriptAssignment};
use itertools::izip;
use libm::lgammaf;
use ndarray::{Axis, Zip, s};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const SIMPLE_PAR_ITER_MIN_LEN: usize = 64;

// Floor to keep components / metagenes from dying permanently at a mode of
// exactly zero (which would send log π to -inf and make the component
// unrecoverable).
const PI_FLOOR: f32 = 1e-6;

pub struct ParamOptimizer {
    // ParamSampler is a ZST; we hold one to reuse its (stochastic) dispersion /
    // scale updates that don't yet have a deterministic MAP counterpart.
    sampler: ParamSampler,
}

impl ParamOptimizer {
    pub fn new() -> ParamOptimizer {
        ParamOptimizer {
            sampler: ParamSampler::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn optimize(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        voxels: &VoxelCheckerboard,
        burnin: bool,
        sample_z: bool,
        purge_sparse_mats: bool,
    ) {
        let t0 = Instant::now();
        self.optimize_volume_params(priors, params);
        trace_time("optimize_volume_params", t0);

        let t0 = Instant::now();
        self.optimize_foreground_background(priors, params, voxels, purge_sparse_mats);
        trace_time("optimize_foreground_background", t0);

        let t0 = Instant::now();
        self.optimize_factor_model(priors, params, sample_z, burnin, purge_sparse_mats);
        log::info!("optimize_factor_model: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.optimize_background_rates(priors, params);
        trace_time("optimize_background_rates", t0);

        params.t += 1;
    }

    // Posterior mode of the per-component log-volume mean μ and std σ. The mode of
    // the Normal posterior on μ is its mean (the sampler's mean, minus the noise
    // term); the mode of the inverse-gamma posterior on the variance is
    // (β + SS/2) / (α + pop/2 + 1).
    fn optimize_volume_params(&self, priors: &ModelPriors, params: &mut ModelParams) {
        params
            .log_cell_volume
            .iter_mut()
            .zip(params.cell_voxel_count.iter())
            .par_bridge()
            .for_each(|(log_cell_volume_c, cell_volume_c)| {
                *log_cell_volume_c = (cell_volume_c as f32 * params.voxel_volume).ln();
            });

        // accumulate Σ log_volume per component into μ_volume
        params.μ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.log_cell_volume)
            .for_each(|&z, &log_volume| {
                params.μ_volume[z as usize] += log_volume;
            });

        // μ posterior mode (= posterior mean, no noise)
        Zip::from(&mut params.μ_volume)
            .and(&params.σ_volume)
            .and(&params.component_population)
            .for_each(|μ, &σ, &pop| {
                let v = (1_f32 / priors.σ_μ_volume.powi(2) + pop as f32 / σ.powi(2)).recip();
                *μ = v * (priors.μ_μ_volume / priors.σ_μ_volume.powi(2) + *μ / σ.powi(2));
            });

        // accumulate sum of squared deviations per component
        params.σ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.log_cell_volume)
            .for_each(|&z, &log_volume| {
                params.σ_volume[z as usize] += (params.μ_volume[z as usize] - log_volume).powi(2);
            });

        // σ posterior mode: sqrt of the inverse-gamma variance mode
        Zip::from(&mut params.σ_volume)
            .and(&params.component_population)
            .for_each(|σ, &pop| {
                let ss = *σ;
                let var_mode =
                    (priors.β_σ_volume + ss / 2.0) / (priors.α_σ_volume + pop as f32 / 2.0 + 1.0);
                *σ = var_mode.sqrt();
            });
    }

    // Hard foreground/background assignment: a transcript is foreground iff its
    // cell rate exceeds the background rate (the argmax of the two-component
    // Poisson mixture), replacing the Bernoulli draw. Counts stay integer.
    fn optimize_foreground_background(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        voxels: &VoxelCheckerboard,
        purge: bool,
    ) {
        if purge {
            params.foreground_counts.clear();
        } else {
            params.foreground_counts.zero();
        }
        params.background_counts.iter_mut().for_each(|b_d| {
            b_d.iter_mut().for_each(|b_dl| {
                b_dl.zero();
            })
        });

        let states = voxels.states_view();
        let ntranscripts = voxels.transcript_voxel.len();
        (0..ntranscripts).into_par_iter().for_each(|idx| {
            let voxel = Voxel::from_raw(voxels.transcript_voxel[idx].load(Ordering::Relaxed));
            let cell = states.get_voxel_cell(voxel);

            let TranscriptFixedState {
                original_voxel,
                gene,
            } = voxels.transcript_fixed_state[idx];

            let density = voxels.get_voxel_density(original_voxel);
            let k_origin = original_voxel.k() as usize;

            let is_background = if cell == BACKGROUND_CELL {
                true
            } else {
                let is_frozen = params.frozen_cells[cell as usize];

                let λ_cg = if (gene as usize) < params.nunfactored {
                    params.φ[[cell as usize, gene as usize]]
                } else {
                    let φ_c_factored = params.φ.slice(s![cell as usize, params.nunfactored..]);
                    let θ_g_factored = params.θ.slice(s![gene as usize, params.nunfactored..]);
                    φ_c_factored.dot(&θ_g_factored)
                };

                let λ_bg = params.λ_bg[[gene as usize, k_origin, density]];

                if priors.unmodeled_fixed_cells && is_frozen {
                    false
                } else {
                    // hard argmax: background unless the cell rate dominates
                    λ_cg <= λ_bg
                }
            };

            let new_assignment = TranscriptAssignment {
                cell,
                background: is_background,
            };
            params.transcript_state[idx].store(new_assignment);

            if is_background {
                params.background_counts[density][k_origin].add(gene as usize, 1);
            } else {
                params
                    .foreground_counts
                    .row(cell as usize)
                    .write()
                    .add(gene, 1);
            }
        });
    }

    fn optimize_factor_model(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        sample_z: bool,
        burnin: bool,
        purge_sparse_mats: bool,
    ) {
        let zero_inflation = priors.use_zero_inflation && !burnin;

        let t0 = Instant::now();
        self.compute_latent_counts(params, purge_sparse_mats);
        trace_time("compute_latent_counts", t0);

        if sample_z {
            let t0 = Instant::now();
            self.optimize_z(params, zero_inflation);
            trace_time("optimize_z", t0);
        }
        self.optimize_π(params);

        if zero_inflation {
            self.sampler.sample_gate(params);
            self.sampler.sample_ξ(priors, params);
        }

        if priors.use_factorization {
            let t0 = Instant::now();
            self.optimize_θ(priors, params);
            trace_time("optimize_θ", t0);
        }

        let t0 = Instant::now();
        self.optimize_φ(params);
        params.update_phi_theta_dot();
        trace_time("optimize_φ", t0);

        // Dispersion / scale nuisance parameters: reuse the sampler's updates.
        if let Some(dispersion) = priors.dispersion {
            params.rφ.fill(dispersion);
        } else if burnin && priors.burnin_dispersion.is_some() {
            let dispersion = priors.burnin_dispersion.unwrap();
            params.rφ.fill(dispersion);
        } else {
            self.sampler.sample_rφ(priors, params);
        }

        self.sampler.sample_ωck(params);
        self.sampler.sample_sφ(priors, params);

        if priors.use_cell_scales {
            self.optimize_cell_scales(priors, params);
        } else {
            params
                .effective_cell_volume
                .iter_mut()
                .zip(params.cell_voxel_count.iter())
                .for_each(|(ev, v)| {
                    *ev = (v as f32) * params.voxel_volume;
                });
        }
    }

    // Hard E-step: assign every count of a (cell, factored-gene) pair to the single
    // metagene maximizing φ_ck·θ_gk. This replaces the multinomial draw of
    // `ParamSampler::sample_latent_counts`; the dot products are the same, but the
    // per-category prefix sum and RNG draw are gone.
    fn compute_latent_counts(&self, params: &mut ModelParams, purge: bool) {
        if purge {
            params.cell_latent_counts.clear();
        } else {
            params.cell_latent_counts.zero();
        }

        let nhidden = params.nhidden();
        let nfactored_hidden = nhidden - params.nunfactored;
        let nunfactored = params.nunfactored;

        let gene_latent_slice = params.gene_latent_counts.as_slice_mut().unwrap();
        gene_latent_slice.par_iter_mut().for_each(|x| *x = 0);
        // SAFETY: identical to sample_latent_counts — AtomicU32 shares layout with
        // u32, we hold the unique &mut for the view's lifetime, and only access it
        // atomically for concurrent accumulation.
        let gene_latent_atomic: &[AtomicU32] =
            unsafe { &*(gene_latent_slice as *mut [u32] as *const [AtomicU32]) };

        let cell_latent_counts = &params.cell_latent_counts;
        let foreground_counts = &params.foreground_counts;
        let φ = &params.φ;
        let θ = &params.θ;

        cell_latent_counts
            .par_rows()
            .zip(foreground_counts.par_rows())
            .zip(φ.outer_iter())
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|((cell_latent_counts_c, x_c), φ_c)| {
                let x_c = x_c.read();
                let mut cell_latent_counts_c = cell_latent_counts_c.write();

                // unfactored genes map identically into the cell latent counts
                for (g, x_cg) in x_c.iter_nonzeros_to(nunfactored as u32) {
                    if x_cg > 0 {
                        cell_latent_counts_c.add(g, x_cg);
                    }
                }

                // factored genes: hard-assign all counts to the argmax metagene
                let φ_c_factored = φ_c.slice(s![nunfactored..]);
                for (g, x_cg) in x_c.iter_nonzeros_from(nunfactored as u32) {
                    if x_cg == 0 {
                        continue;
                    }

                    let θ_g_factored = θ.slice(s![g as usize, nunfactored..]);

                    let mut best_k = 0usize;
                    let mut best_val = f32::NEG_INFINITY;
                    for (k, (φ_ck, θ_gk)) in
                        φ_c_factored.iter().zip(θ_g_factored.iter()).enumerate()
                    {
                        let v = *φ_ck * *θ_gk;
                        if v > best_val {
                            best_val = v;
                            best_k = k;
                        }
                    }

                    cell_latent_counts_c.add((best_k + nunfactored) as u32, x_cg);
                    gene_latent_atomic[g as usize * nfactored_hidden + best_k]
                        .fetch_add(x_cg, Ordering::Relaxed);
                }
            });

        // component-wise aggregation (unchanged from the sampler; still integer)
        params.component_population.fill(0);
        params.component_volume.fill(0.0);
        params.component_latent_counts.fill(0);
        for ((z_c, v_c), x_c) in params
            .z
            .iter()
            .zip(params.cell_voxel_count.iter())
            .zip(params.cell_latent_counts.rows())
        {
            let z_c = *z_c as usize;
            params.component_population[z_c] += 1;
            params.component_volume[z_c] += (v_c as f32) * params.voxel_volume;
            let mut component_latent_counts_z = params.component_latent_counts.row_mut(z_c);
            for (g, x_cg) in x_c.read().iter_nonzeros() {
                component_latent_counts_z[g as usize] += x_cg;
            }
        }
    }

    // Assign each cell to the component maximizing its posterior (argmax) rather
    // than drawing categorically. Mirrors the log-probability accumulation of
    // `ParamSampler::sample_z`.
    fn optimize_z(&self, params: &mut ModelParams, zero_inflation: bool) {
        Zip::from(&mut params.lgamma_rφ)
            .and(&params.rφ)
            .into_par_iter()
            .with_min_len(SIMPLE_PAR_ITER_MIN_LEN)
            .for_each(|(lgamma_r_tk, r_tk)| {
                *lgamma_r_tk = lgammaf(*r_tk);
            });

        Zip::indexed(&mut params.z)
            .and(&params.effective_cell_volume)
            .and(&params.log_cell_volume)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(i, z_c, ev_c, log_v_c)| {
                let x_c_lock = params.cell_latent_counts.row(i);
                let x_c = x_c_lock.read();

                let mut best_t = 0u32;
                let mut best_lp = f64::NEG_INFINITY;

                for (t, (log_π_t, r_t, lgamma_r_t, s_t, log_ξ_t, log_1m_ξ_t, μ_vol_c, σ_vol_c)) in izip!(
                    params.log_π.iter(),
                    params.rφ.rows(),
                    params.lgamma_rφ.rows(),
                    params.sφ.rows(),
                    params.log_ξ.rows(),
                    params.log_1m_ξ.rows(),
                    &params.μ_volume,
                    &params.σ_volume
                )
                .enumerate()
                {
                    let mut lp = *log_π_t as f64;

                    for (r_tk, lgamma_r_tk, s_tk, &log_ξ_tk, &log_1m_ξ_tk, θ_k_sum, x_ck) in izip!(
                        r_t,
                        lgamma_r_t,
                        s_t,
                        log_ξ_t,
                        log_1m_ξ_t,
                        &params.θksum,
                        x_c.iter()
                    ) {
                        let p = odds_to_prob(*s_tk * *ev_c * *θ_k_sum);
                        let contrib = if zero_inflation {
                            if x_ck == 0 {
                                let a = log_ξ_tk + *r_tk * (-p).ln_1p();
                                let b = log_1m_ξ_tk;
                                let m = a.max(b);
                                (m + ((a - m).exp() + (b - m).exp()).ln()) as f64
                            } else {
                                (log_ξ_tk + negbin_logpmf(*r_tk, *lgamma_r_tk, p, x_ck)) as f64
                            }
                        } else {
                            negbin_logpmf(*r_tk, *lgamma_r_tk, p, x_ck) as f64
                        };
                        lp += contrib;
                    }

                    lp += normal_logpdf(*μ_vol_c, *σ_vol_c, *log_v_c) as f64;

                    if lp > best_lp {
                        best_lp = lp;
                        best_t = t as u32;
                    }
                }

                *z_c = best_t;
            });
    }

    // Dirichlet mode of the component mixing weights: π_t ∝ pop_t, with a small
    // floor so an empty component isn't permanently killed.
    fn optimize_π(&self, params: &mut ModelParams) {
        let mut π_sum = 0.0;
        Zip::from(&mut params.π)
            .and(&params.component_population)
            .for_each(|π_t, &pop_t| {
                *π_t = (pop_t as f32).max(PI_FLOOR);
                π_sum += *π_t;
            });
        params.π.iter_mut().for_each(|π_t| *π_t /= π_sum);
        Zip::from(&mut params.log_π)
            .and(&params.π)
            .for_each(|log_π_t, π_t| *log_π_t = π_t.ln());
    }

    // Posterior *mean* of each metagene's gene distribution, θ_gk ∝ αθ + count_gk,
    // renormalized over genes. The Dirichlet mode ∝ max(αθ + count − 1, 0) is
    // boundary-degenerate for αθ < 1 (it zeros every gene with count ≤ 1 − αθ,
    // sparsifying θ aggressively); the mean keeps every metagene a well-defined
    // distribution, mirroring the mean-not-mode choice made for φ.
    fn optimize_θ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let αθ = priors.αθ;
        let mut θfac = params
            .θ
            .slice_mut(s![params.nunfactored.., params.nunfactored..]);
        let gene_latent_counts_fac = params.gene_latent_counts.slice(s![params.nunfactored.., ..]);

        Zip::from(θfac.axis_iter_mut(Axis(1)))
            .and(gene_latent_counts_fac.axis_iter(Axis(1)))
            .into_par_iter()
            .for_each(|(mut θ_k, x_k)| {
                let mut sum = 0.0_f32;
                for (θ_gk, &x_gk) in θ_k.iter_mut().zip(x_k.iter()) {
                    let w = αθ + x_gk as f32;
                    *θ_gk = w;
                    sum += w;
                }
                let inv = sum.recip();
                θ_k.iter_mut().for_each(|θ_gk| *θ_gk *= inv);
            });

        Zip::from(&mut params.θksum)
            .and(params.θ.axis_iter(Axis(1)))
            .for_each(|θksum, θ_k| {
                *θksum = θ_k.sum();
            });
    }

    // Posterior *mean* of φ (shape·scale), deliberately not the mode: the Gamma
    // mode collapses to zero whenever shape < 1, which drives the degenerate
    // vanishing-cell behavior the design warns about.
    fn optimize_φ(&self, params: &mut ModelParams) {
        Zip::indexed(params.φ.outer_iter_mut())
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .and(params.gate.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(c, φ_c, z_c, v_c, gate_c)| {
                let x_c = params.cell_latent_counts.row(c);
                let z_c = *z_c as usize;

                for (φ_ck, &θ_k_sum, x_ck, &r_k, s_k, &g_ck) in izip!(
                    φ_c,
                    &params.θksum,
                    x_c.read().iter(),
                    &params.rφ.row(z_c),
                    &params.sφ.row(z_c),
                    gate_c
                ) {
                    if !g_ck {
                        *φ_ck = 0.0;
                    } else {
                        let shape = r_k + x_ck as f32;
                        let scale = s_k / (1.0 + s_k * v_c * θ_k_sum);
                        *φ_ck = shape * scale;
                    }
                }
            });

        Zip::from(&mut params.φ_v_dot)
            .and(params.φ.axis_iter(Axis(1)))
            .for_each(|φ_v_dot_k, φ_k| {
                *φ_v_dot_k = φ_k.dot(&params.effective_cell_volume);
            });
    }

    // Deterministic (noise-free) counterpart of `sample_cell_scales`: the mode of
    // the log-normal cell scale is exp(μ).
    fn optimize_cell_scales(&self, priors: &ModelPriors, params: &mut ModelParams) {
        Zip::indexed(&mut params.cell_scale)
            .and(&mut params.effective_cell_volume)
            .and(&params.log_cell_volume)
            .and(&params.z)
            .and(params.ωφ.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(c, a_c, eff_v_c, &log_v_c, &z_c, ω_c)| {
                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);

                let τ = priors.τv + ω_c.sum();
                let σ2 = τ.recip();

                let mut μ = 0.0;
                for (&θ_k_sum, x_ck, &r_tk, &s_tk, &ω_ck) in izip!(
                    &params.θksum,
                    x_c.read().iter(),
                    params.rφ.row(z_c),
                    params.sφ.row(z_c),
                    ω_c
                ) {
                    μ += (x_ck as f32 - r_tk) / 2.0 - ω_ck * ((s_tk * θ_k_sum).ln() + log_v_c);
                }
                μ *= σ2;

                *a_c = μ.exp();
                *eff_v_c = (μ + log_v_c).exp();
            });
    }

    // Posterior mean of the background Poisson rates (α + count)/(β + volume).
    fn optimize_background_rates(&self, priors: &ModelPriors, params: &mut ModelParams) {
        Zip::from(params.λ_bg.axis_iter_mut(Axis(2)))
            .and(&params.background_counts)
            .and(&params.background_region_volume)
            .for_each(|mut λ_d, x_d, &v_d| {
                for (λ_dl, x_dl) in izip!(λ_d.axis_iter_mut(Axis(1)), x_d) {
                    for (λ_dlg, x_dlg) in izip!(λ_dl, x_dl.iter()) {
                        let α = priors.α_bg + x_dlg as f32;
                        let β = priors.β_bg + v_d;
                        *λ_dlg = α / β;
                    }
                }
            });

        Zip::from(&mut params.logλ_bg)
            .and(&params.λ_bg)
            .for_each(|logλ_bg, λ_bg| {
                *logλ_bg = λ_bg.ln();
            });
    }
}

fn trace_time(label: &str, t0: Instant) {
    log::trace!("{label}: {:?}", t0.elapsed());
}
