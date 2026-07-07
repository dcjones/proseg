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

use super::math::{expected_crt, negbin_logpmf_f, normal_logpdf, odds_to_prob, randn};
use super::paramsampler::ParamSampler;
use super::polyagamma::PolyaGamma;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{TranscriptFixedState, Voxel, VoxelCheckerboard};
use super::{ModelParams, ModelPriors, RAYON_CELL_MIN_LEN, TranscriptAssignment};
use itertools::izip;
use libm::lgammaf;
use ndarray::{Array2, Axis, Zip, s};
use rand::{Rng, rng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::time::Instant;
use thread_local::ThreadLocal;

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
            self.optimize_gate(params);
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

        // Metagene dispersion rφ. Freely maximizing likelihood drives rφ small
        // (high overdispersion), which fits per-cell counts better but washes out
        // the shared metagene structure that separates cell types — an overfit that
        // hurts segmentation despite raising the likelihood. So by default the
        // optimizer PINS rφ to a regularizing value (`optimizer_dispersion`, ~5),
        // which concentrates the factorization and matches/beats the Gibbs point
        // estimate. An explicit `--dispersion` overrides; `--optimizer-free-dispersion`
        // re-enables the experimental EM update (`optimize_rφ`).
        if let Some(dispersion) = priors.dispersion {
            params.rφ.fill(dispersion);
        } else if priors.optimizer_free_dispersion {
            self.optimize_rφ(priors, params);
        } else {
            params.rφ.fill(priors.optimizer_dispersion);
        }

        // Scale updates, reimplemented on the fractional f32 counts.
        self.optimize_ωck(params);
        self.optimize_sφ(priors, params);

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

    // Soft EM E-step: distribute each (cell, factored-gene) count over metagenes by
    // the responsibilities E[z_cgk] = x_cg · φ_ck·θ_gk / Σ_k'(φ_ck'·θ_gk'), the
    // conditional expectation the multinomial draws in the sampler. These
    // fractional counts feed all downstream updates (θ, φ, z, and the dispersion /
    // scale updates rφ/ωck/sφ, which are reimplemented in this struct to consume
    // f32 directly). Rounding fractional responsibilities per metagene would zero
    // the many small (<0.5) entries — for the 79% of pairs with a single count the
    // responsibilities never exceed ~0.4 — losing nearly all factored mass, so we
    // keep everything fractional.
    fn compute_latent_counts(&self, params: &mut ModelParams, purge: bool) {
        if purge {
            params.cell_latent_counts_f.clear();
        } else {
            params.cell_latent_counts_f.zero();
        }

        let nhidden = params.nhidden();
        let nfactored_hidden = nhidden - params.nunfactored;
        let nunfactored = params.nunfactored;
        let ngenes = params.θ.shape()[0];

        // Zero the per-thread gene accumulators (persisted across iterations).
        for tl in params.gene_latent_counts_f_tl.iter_mut() {
            tl.borrow_mut().fill(0.0);
        }

        let cell_latent_counts_f = &params.cell_latent_counts_f;
        let foreground_counts = &params.foreground_counts;
        let φ = &params.φ;
        let θ = &params.θ;
        let gene_tl = &params.gene_latent_counts_f_tl;

        cell_latent_counts_f
            .par_rows()
            .zip(foreground_counts.par_rows())
            .zip(φ.outer_iter())
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|((cf_c, x_c), φ_c)| {
                let mut gene_acc = gene_tl
                    .get_or(|| RefCell::new(Array2::zeros((ngenes, nfactored_hidden))))
                    .borrow_mut();

                let x_c = x_c.read();
                let mut cf_c = cf_c.write();

                // Unfactored genes map identically into the cell latent counts.
                for (g, x_cg) in x_c.iter_nonzeros_to(nunfactored as u32) {
                    if x_cg > 0 {
                        cf_c.update(g, || 0.0, |v| *v += x_cg as f32);
                    }
                }

                // Factored genes: distribute counts over metagenes by responsibility.
                let φ_c_factored = φ_c.slice(s![nunfactored..]);
                for (g, x_cg) in x_c.iter_nonzeros_from(nunfactored as u32) {
                    if x_cg == 0 {
                        continue;
                    }
                    let θ_g_factored = θ.slice(s![g as usize, nunfactored..]);

                    let mut sum = 0.0_f32;
                    for (φ_ck, θ_gk) in φ_c_factored.iter().zip(θ_g_factored.iter()) {
                        sum += *φ_ck * *θ_gk;
                    }
                    if sum <= 0.0 {
                        // Degenerate (no metagene supports this gene in this cell);
                        // skip — leaves the count unassigned this iteration.
                        continue;
                    }
                    let scale = x_cg as f32 / sum;

                    let mut gene_acc_g = gene_acc.row_mut(g as usize);
                    for (k, (φ_ck, θ_gk)) in
                        φ_c_factored.iter().zip(θ_g_factored.iter()).enumerate()
                    {
                        let e = *φ_ck * *θ_gk * scale;
                        if e > 0.0 {
                            cf_c.update((k + nunfactored) as u32, || 0.0, |v| *v += e);
                            gene_acc_g[k] += e;
                        }
                    }
                }
            });

        // Reduce the per-thread gene accumulators into gene_latent_counts_f.
        params.gene_latent_counts_f.fill(0.0);
        for tl in params.gene_latent_counts_f_tl.iter_mut() {
            params.gene_latent_counts_f += &*tl.borrow();
        }

        // component-wise aggregation of population / volume (component_latent_counts
        // has no downstream consumer, so it is not recomputed here).
        params.component_population.fill(0);
        params.component_volume.fill(0.0);
        for (z_c, v_c) in params.z.iter().zip(params.cell_voxel_count.iter()) {
            let z_c = *z_c as usize;
            params.component_population[z_c] += 1;
            params.component_volume[z_c] += (v_c as f32) * params.voxel_volume;
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
                let x_c_lock = params.cell_latent_counts_f.row(i);
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
                            if x_ck == 0.0 {
                                let a = log_ξ_tk + *r_tk * (-p).ln_1p();
                                let b = log_1m_ξ_tk;
                                let m = a.max(b);
                                (m + ((a - m).exp() + (b - m).exp()).ln()) as f64
                            } else {
                                (log_ξ_tk + negbin_logpmf_f(*r_tk, *lgamma_r_tk, p, x_ck)) as f64
                            }
                        } else {
                            negbin_logpmf_f(*r_tk, *lgamma_r_tk, p, x_ck) as f64
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
        let gene_latent_counts_fac = params
            .gene_latent_counts_f
            .slice(s![params.nunfactored.., ..]);

        Zip::from(θfac.axis_iter_mut(Axis(1)))
            .and(gene_latent_counts_fac.axis_iter(Axis(1)))
            .into_par_iter()
            .for_each(|(mut θ_k, x_k)| {
                let mut sum = 0.0_f32;
                for (θ_gk, &x_gk) in θ_k.iter_mut().zip(x_k.iter()) {
                    let w = αθ + x_gk;
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
                let x_c = params.cell_latent_counts_f.row(c);
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
                        let shape = r_k + x_ck;
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
                let x_c = params.cell_latent_counts_f.row(c);

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
                    μ += (x_ck - r_tk) / 2.0 - ω_ck * ((s_tk * θ_k_sum).ln() + log_v_c);
                }
                μ *= σ2;

                *a_c = μ.exp();
                *eff_v_c = (μ + log_v_c).exp();
            });
    }

    // Deterministic EM update of the metagene dispersion rφ. The Gibbs sampler
    // draws an integer CRT latent l_ck = CRT(x_ck, r_k) then samples
    // rφ ~ Gamma(eφ + Σl, (1/fφ + Σ ln1p(s·v·θksum))^-1). The EM counterpart
    // replaces the CRT draw with its expectation E[CRT] = r·(ψ(r+x) − ψ(r))
    // (fractional-count safe) and takes the Gamma-posterior mode. Per-component
    // sums are accumulated over cells via thread-local reduction (small
    // [ncomponents, nhidden] buffers), avoiding the sampler's O(ncomp·nhidden·ncells)
    // rescan.
    fn optimize_rφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let ncomp = params.ncomponents();
        let nhidden = params.nhidden();
        // Per-thread [ncomponents, nhidden] accumulators: expected CRT counts and
        // the log1p scale-inverse term.
        let mut lsum_tl: ThreadLocal<RefCell<Array2<f32>>> = ThreadLocal::new();
        let mut sinv_tl: ThreadLocal<RefCell<Array2<f32>>> = ThreadLocal::new();

        let rφ = &params.rφ;
        let sφ = &params.sφ;
        let θksum = &params.θksum;
        let cell_latent_counts_f = &params.cell_latent_counts_f;

        Zip::indexed(&params.z)
            .and(&params.effective_cell_volume)
            .and(params.gate.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(c, &z_c, &v_c, gate_c)| {
                let z_c = z_c as usize;
                let mut lsum = lsum_tl
                    .get_or(|| RefCell::new(Array2::zeros((ncomp, nhidden))))
                    .borrow_mut();
                let mut sinv = sinv_tl
                    .get_or(|| RefCell::new(Array2::zeros((ncomp, nhidden))))
                    .borrow_mut();

                let x_c = cell_latent_counts_f.row(c);
                let x_c = x_c.read();
                let mut lsum_z = lsum.row_mut(z_c);
                let mut sinv_z = sinv.row_mut(z_c);

                for (lsum_zk, sinv_zk, x_ck, &r_k, &s_k, &θk, &g_ck) in izip!(
                    lsum_z.iter_mut(),
                    sinv_z.iter_mut(),
                    x_c.iter(),
                    rφ.row(z_c),
                    sφ.row(z_c),
                    θksum,
                    gate_c
                ) {
                    *lsum_zk += expected_crt(x_ck, r_k);
                    // Only on-cells (gate) are NB observations contributing to the
                    // dispersion rate term.
                    if g_ck {
                        *sinv_zk += (s_k * v_c * θk).ln_1p();
                    }
                }
            });

        let mut lsum_total = Array2::<f32>::zeros((ncomp, nhidden));
        let mut sinv_total = Array2::<f32>::zeros((ncomp, nhidden));
        for tl in lsum_tl.iter_mut() {
            lsum_total += &*tl.borrow();
        }
        for tl in sinv_tl.iter_mut() {
            sinv_total += &*tl.borrow();
        }

        let eφ = priors.eφ;
        let inv_fφ = 1.0 / priors.fφ;
        let min_rφ = priors.min_rφ;
        Zip::from(&mut params.rφ)
            .and(&lsum_total)
            .and(&sinv_total)
            .for_each(|r_tk, &lsum, &sinv| {
                // Mode of Gamma(shape = eφ + Σl, rate = 1/fφ + Σln1p) = (shape−1)/rate.
                let shape = eφ + lsum;
                let rate = inv_fφ + sinv;
                let mode = (shape - 1.0).max(0.0) / rate;
                *r_tk = mode.max(min_rφ);
            });
    }

    // Zero-inflation gate update on fractional counts (mirrors sample_gate; only
    // active when zero-inflation is enabled). A cell/metagene with any expected
    // count is on; otherwise the gate is drawn from its Bernoulli posterior.
    fn optimize_gate(&self, params: &mut ModelParams) {
        Zip::indexed(params.gate.outer_iter_mut())
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, gate_c, &z_c, &ev_c)| {
                let z_c = z_c as usize;
                let x_c_lock = params.cell_latent_counts_f.row(c);
                let x_c = x_c_lock.read();
                let r_t = params.rφ.row(z_c);
                let s_t = params.sφ.row(z_c);
                let log_ξ_t = params.log_ξ.row(z_c);
                let log_1m_ξ_t = params.log_1m_ξ.row(z_c);

                for (g_ck, x_ck, &r_tk, &s_tk, &log_ξ_tk, &log_1m_ξ_tk, &θ_k_sum) in izip!(
                    gate_c,
                    x_c.iter(),
                    r_t,
                    s_t,
                    log_ξ_t,
                    log_1m_ξ_t,
                    &params.θksum
                ) {
                    if x_ck > 0.0 {
                        *g_ck = true;
                    } else {
                        let p = odds_to_prob(s_tk * ev_c * θ_k_sum);
                        let log_on = log_ξ_tk + r_tk * (-p).ln_1p();
                        let log_off = log_1m_ξ_tk;
                        let p_on = 1.0 / (1.0 + (log_off - log_on).exp());
                        *g_ck = rng.random::<f32>() < p_on;
                    }
                }
            });
    }

    // Polya-Gamma augmentation draw for the sφ update, on fractional counts.
    // Mirrors sample_ωck; kept stochastic (the auxiliary variable's deterministic
    // MAP is deferred with the rest of the dispersion/scale machinery).
    fn optimize_ωck(&self, params: &mut ModelParams) {
        Zip::indexed(params.ωφ.outer_iter_mut())
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .and(params.gate.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, ω_c, &z_c, &v_c, gate_c)| {
                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts_f.row(c);

                for (ω_ck, x_ck, &r_k, &s_k, &θ_k_sum, &g_ck) in izip!(
                    ω_c,
                    x_c.read().iter(),
                    params.rφ.row(z_c),
                    params.sφ.row(z_c),
                    &params.θksum,
                    gate_c
                ) {
                    if g_ck {
                        let ε = (s_k * v_c * θ_k_sum).ln();
                        *ω_ck = PolyaGamma::new(x_ck + r_k, ε).sample(rng);
                    } else {
                        *ω_ck = 0.0;
                    }
                }
            });
    }

    // sφ (per-component metagene scale) update on fractional counts. Mirrors
    // sample_sφ exactly, reading the f32 latent counts.
    fn optimize_sφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let ncomponents = params.ncomponents();
        let nhidden = params.nhidden();

        for x in params.sφ_work_tl.iter_mut() {
            x.borrow_mut().fill(0.0);
        }
        Zip::from(&params.z)
            .and(params.ωφ.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(&z_c, ω_c)| {
                let mut τ_sφ_tl = params
                    .sφ_work_tl
                    .get_or(|| RefCell::new(Array2::zeros((ncomponents, nhidden))))
                    .borrow_mut();
                let z_c = z_c as usize;
                let mut τ_sφ_k = τ_sφ_tl.row_mut(z_c);
                τ_sφ_k.scaled_add(1.0, &ω_c);
            });

        params.τ_sφ.fill(priors.τφ);
        for x in params.sφ_work_tl.iter_mut() {
            params.τ_sφ.scaled_add(1.0, &x.borrow());
        }

        for x in params.sφ_work_tl.iter_mut() {
            x.borrow_mut().fill(0.0);
        }
        Zip::indexed(&params.z)
            .and(&params.effective_cell_volume)
            .and(params.ωφ.outer_iter())
            .and(params.gate.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(c, &z_c, &v_c, ω_c, gate_c)| {
                let mut μ_sφ_tl = params
                    .sφ_work_tl
                    .get_or(|| RefCell::new(Array2::zeros((ncomponents, nhidden))))
                    .borrow_mut();

                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts_f.row(c);
                let r_t = params.rφ.row(z_c);
                let μ_sφ_t = μ_sφ_tl.row_mut(z_c);

                for (μ_sφ_tk, x_ck, &ω_ck, &r_tk, &θ_k_sum, &g_ck) in
                    izip!(μ_sφ_t, x_c.read().iter(), ω_c, r_t, &params.θksum, gate_c)
                {
                    if g_ck {
                        *μ_sφ_tk += (x_ck - r_tk) / 2.0 - ω_ck * (v_c * θ_k_sum).ln();
                    }
                }
            });

        params.μ_sφ.fill(priors.μφ * priors.τφ);
        for x in params.sφ_work_tl.iter_mut() {
            params.μ_sφ.scaled_add(1.0, &x.borrow());
        }

        Zip::from(&mut params.sφ)
            .and(&params.μ_sφ)
            .and(&params.τ_sφ)
            .into_par_iter()
            .for_each_init(rng, |rng, (s_tk, &μ_tk, &τ_tk)| {
                let σ2_tk = τ_tk.recip();
                let μ_tk = μ_tk * σ2_tk;
                *s_tk = (μ_tk + σ2_tk.sqrt() * randn(rng)).exp();
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
