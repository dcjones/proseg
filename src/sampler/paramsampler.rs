use super::math::{negbin_logpmf, normal_logpdf, odds_to_prob, rand_crt, randn};
use super::multinomial::Multinomial;
use super::polyagamma::PolyaGamma;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{TranscriptFixedState, Voxel, VoxelCheckerboard};
use super::{FlowStats, ModelParams, ModelPriors, RAYON_CELL_MIN_LEN, TranscriptAssignment};
use itertools::izip;
use libm::lgammaf;
use log::{info, trace};
use ndarray::{Array2, Axis, Zip, s};
use rand::{Rng, rng};
use rand_distr::{Distribution, Gamma, Normal};
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

// Setting parallel iterator min length for simple operations
const SIMPLE_PAR_ITER_MIN_LEN: usize = 64;

pub struct ParamSampler {}

impl ParamSampler {
    pub fn new() -> ParamSampler {
        ParamSampler {}
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        voxels: &VoxelCheckerboard,
        burnin: bool,
        _temperature: f32,
        record_samples: bool,
        sample_z: bool,
        purge_sparse_mats: bool,
    ) {
        let t0 = Instant::now();
        self.sample_volume_params(priors, params);
        trace!("sample_volume_params: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_foreground_background(
            priors,
            params,
            voxels,
            purge_sparse_mats,
            record_samples,
        );
        trace!("sample_foreground_background: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_factor_model(priors, params, sample_z, burnin, purge_sparse_mats);
        info!("sample_factor_model: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_background_rates(priors, params);
        trace!("sample_background_rates: {:?}", t0.elapsed());

        if !burnin && record_samples {
            // params
            //     .foreground_counts_lower
            //     .update(&params.foreground_counts);
            // params
            //     .foreground_counts_upper
            //     .update(&params.foreground_counts);
            params
                .foreground_counts_mean
                .update(&params.foreground_counts);
        }

        params.t += 1;
    }

    fn sample_volume_params(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // Index-parallel map over cells. (`par_bridge` here funnels through a shared
        // sequential iterator whose contention grows with thread count, so it
        // actually regressed with more threads; a split range does not.)
        let voxel_volume = params.voxel_volume;
        let cell_voxel_count = &params.cell_voxel_count;
        Zip::indexed(&mut params.log_cell_volume)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(c, log_cell_volume_c)| {
                *log_cell_volume_c = (cell_voxel_count.get(c) as f32 * voxel_volume).ln();
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
            .for_each_init(rng, |rng, (μ, &σ, &pop)| {
                let v = (1_f32 / priors.σ_μ_volume.powi(2) + pop as f32 / σ.powi(2)).recip();
                *μ = Normal::new(
                    v * (priors.μ_μ_volume / priors.σ_μ_volume.powi(2) + *μ / σ.powi(2)),
                    v.sqrt(),
                )
                .unwrap()
                .sample(rng);
            });

        // compute sample variances
        params.σ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.log_cell_volume)
            .for_each(|&z, &log_volume| {
                params.σ_volume[z as usize] += (params.μ_volume[z as usize] - log_volume).powi(2);
            });

        let mut rng = rng();
        Zip::from(&mut params.σ_volume)
            .and(&params.component_population)
            .for_each(|σ, &pop| {
                *σ = Gamma::new(
                    priors.α_σ_volume + (pop as f32) / 2.0,
                    (priors.β_σ_volume + *σ / 2.0).recip(),
                )
                .unwrap()
                .sample(&mut rng)
                .recip()
                .sqrt();
            });
    }

    pub fn sample_foreground_background(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        voxels: &VoxelCheckerboard,
        purge: bool,
        record_samples: bool,
    ) {
        // Sample foreground/background state
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

        // Iterate over all transcripts in parallel (over the flat position array).
        // States are read-only here, so take a single lock-free view rather than
        // locking on every per-transcript cell lookup.
        let states = voxels.states_view();
        let ntranscripts = voxels.transcript_voxel.len();
        (0..ntranscripts).into_par_iter().for_each_init(rng, |rng, idx| {
            {
                // Current voxel and its cell assignment.
                let voxel = Voxel::from_raw(
                    voxels.transcript_voxel[idx].load(std::sync::atomic::Ordering::Relaxed),
                );
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

                    let fg_prob = if priors.unmodeled_fixed_cells && is_frozen {
                        1.0
                    } else {
                        λ_cg / (λ_cg + λ_bg)
                    };

                    rng.random::<f32>() > fg_prob
                };

                let new_assignment = TranscriptAssignment {
                    cell,
                    background: is_background,
                };
                params.transcript_state[idx].store(new_assignment);

                // Update transition counts
                if record_samples {
                    let ncells = voxels.ncells as u32;
                    let src_state = params.reported_transcript_state[idx].load();

                    if !src_state.background && src_state != new_assignment {
                        let mut inflow_row =
                            params.expected_inflow.row(src_state.cell as usize).write();

                        inflow_row.update(gene, FlowStats::default, |v| {
                            v.sample_count += 1;
                            v.count += 1;
                        });
                    }

                    if !new_assignment.background && src_state != new_assignment {
                        let mut outflow_row = params
                            .expected_outflow
                            .row(new_assignment.cell as usize)
                            .write();
                        outflow_row.update(gene, FlowStats::default, |v| {
                            v.sample_count += 1;
                            v.count += 1;
                        });
                    }

                    // Heterotypic uncertainty: record off-diagonal, foreground
                    // cell→cell moves relative to the point estimate (src_state =
                    // reported cell, `cell` = currently-sampled cell). Always on.
                    if !src_state.background && !is_background && src_state.cell != cell {
                        params
                            .het_transitions
                            .row(src_state.cell as usize)
                            .write()
                            .add(cell, 1);
                    }

                    if priors.record_state_transitions {
                        let src_state = if src_state.background {
                            ncells
                        } else {
                            src_state.cell
                        };

                        let dest_state = if is_background { ncells } else { cell };

                        params
                            .state_transitions
                            .add_local(src_state as usize, gene, dest_state);
                    }

                    if let Some(ref counts) = params.transcript_assignment_counts {
                        let assigned_cell = if is_background { BACKGROUND_CELL } else { cell };
                        let mut map = counts[idx].lock();
                        *map.entry(assigned_cell).or_insert(0) += 1;
                    }
                }

                // Update count matrices
                if is_background {
                    // if cell != BACKGROUND_CELL {
                    params.background_counts[density][k_origin].add(gene as usize, 1);
                    // }
                } else {
                    params
                        .foreground_counts
                        .row(cell as usize)
                        .write()
                        .add(gene, 1);
                }
            }
        });

        if priors.record_state_transitions && record_samples {
            params.state_transitions.flush_locals();
        }
    }

    fn sample_factor_model(
        &self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        sample_z: bool,
        burnin: bool,
        purge_sparse_mats: bool,
    ) {
        // Zero-inflation is only active after burn-in, once the metagene
        // archetypes have settled enough to trust a gate-off decision.
        let zero_inflation = priors.use_zero_inflation && !burnin;

        let t0 = Instant::now();
        self.sample_latent_counts(params, purge_sparse_mats);
        trace!("sample_latent_counts: {:?}", t0.elapsed());
        if sample_z {
            let t0 = Instant::now();
            self.sample_z(params, zero_inflation);
            trace!("sample_z: {:?}", t0.elapsed());
        }
        self.sample_π(params);

        if zero_inflation {
            let t0 = Instant::now();
            self.sample_gate(params);
            self.sample_ξ(priors, params);
            trace!("sample_gate/sample_ξ: {:?}", t0.elapsed());
        }

        if priors.use_factorization {
            let t0 = Instant::now();
            self.sample_θ(priors, params);
            trace!("sample_θ: {:?}", t0.elapsed());
        }

        let t0 = Instant::now();
        self.sample_φ(params);
        params.update_phi_theta_dot();
        trace!("sample_φ: {:?}", t0.elapsed());

        let t0 = Instant::now();
        if let Some(dispersion) = priors.dispersion {
            params.rφ.fill(dispersion);
        } else if burnin
            && let Some(dispersion) = priors.burnin_dispersion
        {
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
                .zip(params.cell_voxel_count.iter())
                .for_each(|(ev, v)| {
                    *ev = (v as f32) * params.voxel_volume;
                });
        }
    }

    fn sample_z(&self, params: &mut ModelParams, zero_inflation: bool) {
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
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (i, z_c, φ_c, ev_c, log_v_c)| {
                let x_c_lock = params.cell_latent_counts.row(i);
                let x_c = x_c_lock.read();

                let mut z_probs = params
                    .z_probs
                    .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                    .borrow_mut();

                // compute probability of φ_c under every component

                // for every component
                let mut z_probs_sum = 0.0;
                for (z_probs_t, log_π_t, r_t, lgamma_r_t, s_t, log_ξ_t, log_1m_ξ_t, μ_vol_c, σ_vol_c) in izip!(
                    z_probs.iter_mut(),
                    params.log_π.iter(),
                    params.rφ.rows(),
                    params.lgamma_rφ.rows(),
                    params.sφ.rows(),
                    params.log_ξ.rows(),
                    params.log_1m_ξ.rows(),
                    &params.μ_volume,
                    &params.σ_volume
                ) {
                    *z_probs_t = *log_π_t as f64;

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
                        let lp = if zero_inflation {
                            if x_ck == 0 {
                                // log( ξ·NB(0) + (1-ξ) ), NB(0) = (1-p)^r
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
            });
    }

    // Sample the zero-inflation gates b_ck. A cell with any counts assigned to
    // metagene k must be "on"; cells with zero counts are drawn from the
    // posterior P(on | x=0) ∝ ξ·NB(0), P(off | x=0) ∝ (1-ξ).
    pub(crate) fn sample_gate(&self, params: &mut ModelParams) {
        Zip::indexed(params.gate.outer_iter_mut()) // for each cell
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, gate_c, &z_c, &ev_c)| {
                let z_c = z_c as usize;
                let x_c_lock = params.cell_latent_counts.row(c);
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
                    if x_ck > 0 {
                        *g_ck = true;
                    } else {
                        let p = odds_to_prob(s_tk * ev_c * θ_k_sum);
                        let log_on = log_ξ_tk + r_tk * (-p).ln_1p(); // NB(0) = (1-p)^r
                        let log_off = log_1m_ξ_tk;
                        let p_on = 1.0 / (1.0 + (log_off - log_on).exp());
                        *g_ck = rng.random::<f32>() < p_on;
                    }
                }
            });
    }

    // Sample the per-component, per-metagene activation probability ξ from its
    // Beta(a_ξ + #on, b_ξ + #off) posterior.
    pub(crate) fn sample_ξ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let ncomponents = params.ncomponents();
        let nhidden = params.nhidden();

        // Tally "on" cells per (component, metagene). One cheap pass; small
        // relative to the Gibbs sweep, so kept sequential to avoid reduction.
        let mut on = Array2::<u32>::zeros((ncomponents, nhidden));
        for (z_c, gate_c) in params.z.iter().zip(params.gate.outer_iter()) {
            let mut on_row = on.row_mut(*z_c as usize);
            for (on_tk, &g_ck) in on_row.iter_mut().zip(gate_c) {
                *on_tk += g_ck as u32;
            }
        }

        let mut rng = rng();
        let component_population = &params.component_population;
        Zip::indexed(&mut params.ξ)
            .and(&mut params.log_ξ)
            .and(&mut params.log_1m_ξ)
            .and(&on)
            .for_each(|(t, _k), ξ_tk, log_ξ_tk, log_1m_ξ_tk, &on_tk| {
                let pop = component_population[t] as f32;
                let off = (pop - on_tk as f32).max(0.0);
                let g1 = Gamma::new(priors.a_ξ + on_tk as f32, 1.0)
                    .unwrap()
                    .sample(&mut rng);
                let g2 = Gamma::new(priors.b_ξ + off, 1.0).unwrap().sample(&mut rng);
                let ξ = (g1 / (g1 + g2)).clamp(1e-6, 1.0 - 1e-6);
                *ξ_tk = ξ;
                *log_ξ_tk = ξ.ln();
                *log_1m_ξ_tk = (1.0 - ξ).ln();
            });
    }

    fn sample_latent_counts(&self, params: &mut ModelParams, purge: bool) {
        let t0 = Instant::now();
        if purge {
            params.cell_latent_counts.clear();
        } else {
            params.cell_latent_counts.zero();
        }

        let nhidden = params.nhidden();
        let nfactored_hidden = nhidden - params.nunfactored;
        let nunfactored = params.nunfactored;

        // Zero the gene latent counts, then accumulate directly into it from every
        // worker thread through an atomic view. This replaces a large per-thread
        // dense accumulator that had to be zeroed and reduced every iteration (both
        // memory-bandwidth bound on ngenes*nfactored_hidden*nthreads). Zeroing is
        // now a single pass over one matrix and the reduction is gone entirely.
        let gene_latent_slice = params.gene_latent_counts.as_slice_mut().unwrap();
        gene_latent_slice.par_iter_mut().for_each(|x| *x = 0);
        // SAFETY: AtomicU32 has the same size, alignment, and representation as u32.
        // We hold the unique `&mut` for the lifetime of this view and only ever
        // access the memory atomically through it, so exposing it as shared atomics
        // for concurrent accumulation is sound.
        let gene_latent_atomic: &[AtomicU32] =
            unsafe { &*(gene_latent_slice as *mut [u32] as *const [AtomicU32]) };

        // Bind the other participating fields as disjoint shared borrows so the
        // parallel closure doesn't capture all of `params` (which would collide
        // with the mutable borrow backing `gene_latent_atomic`).
        let cell_latent_counts = &params.cell_latent_counts;
        let foreground_counts = &params.foreground_counts;
        let φ = &params.φ;
        let θ = &params.θ;
        let multinomials = &params.multinomials;

        cell_latent_counts
            .par_rows()
            .zip(foreground_counts.par_rows())
            .zip(φ.outer_iter())
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, ((cell_latent_counts_c, x_c), φ_c)| {
                let mut multinomial = multinomials
                    .get_or(|| RefCell::new(Multinomial::new(nfactored_hidden)))
                    .borrow_mut();

                let x_c = x_c.read();
                let mut cell_latent_counts_c = cell_latent_counts_c.write();

                // assign counts from unfactored genes. These map identically into
                // the cell latent counts; the gene-level counts are never consumed,
                // so we don't accumulate them.
                for (g, x_cg) in x_c.iter_nonzeros_to(nunfactored as u32) {
                    if x_cg > 0 {
                        cell_latent_counts_c.add(g, x_cg);
                    }
                }

                // distribute counts from factored genes
                let φ_c_factored = φ_c.slice(s![nunfactored..]);
                for (g, x_cg) in x_c.iter_nonzeros_from(nunfactored as u32) {
                    if x_cg == 0 {
                        continue;
                    }

                    let θ_g_factored = θ.slice(s![g as usize, nunfactored..]);

                    let prob_iter = φ_c_factored
                        .iter()
                        .zip(θ_g_factored.iter())
                        .map(|(φ_ck, θ_gk)| *φ_ck * *θ_gk);
                    multinomial.set_probs_from_iter(prob_iter);

                    let row_base = g as usize * nfactored_hidden;
                    multinomial.sample(rng, x_cg, |k, x| {
                        cell_latent_counts_c.add((k + nunfactored) as u32, x);
                        gene_latent_atomic[row_base + k].fetch_add(x, Ordering::Relaxed);
                    });
                }
            });

        // compute component-wise counts
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
        trace!("sample_latent_counts: {:?}", t0.elapsed());
    }

    fn sample_θ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let mut θfac = params
            .θ
            .slice_mut(s![params.nunfactored.., params.nunfactored..]);
        // gene_latent_counts already holds only the factored hidden columns.
        let gene_latent_counts_fac = params.gene_latent_counts.slice(s![params.nunfactored.., ..]);

        // Sampling with Dirichlet prior on θ (I think Gamma makes more
        // sense, but this is an alternative to consider)
        Zip::from(θfac.axis_iter_mut(Axis(1)))
            .and(gene_latent_counts_fac.axis_iter(Axis(1)))
            .into_par_iter()
            .for_each_init(rng, |rng, (mut θ_k, x_k)| {
                // dirichlet sampling by normalizing gammas
                Zip::from(&mut θ_k).and(x_k).for_each(|θ_gk, x_gk| {
                    *θ_gk = Gamma::new(priors.αθ + *x_gk as f32, 1.0)
                        .unwrap()
                        .sample(rng);
                });

                let θsum = θ_k.sum();
                θ_k *= θsum.recip();
            });

        Zip::from(&mut params.θksum)
            .and(params.θ.axis_iter(Axis(1)))
            .for_each(|θksum, θ_k| {
                *θksum = θ_k.sum();
            });
    }

    fn sample_φ(&self, params: &mut ModelParams) {
        Zip::indexed(params.φ.outer_iter_mut()) // for each cell
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .and(params.gate.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, φ_c, z_c, v_c, gate_c)| {
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
                        // structural zero: metagene off in this cell
                        *φ_ck = 0.0;
                    } else {
                        let shape = r_k + x_ck as f32;
                        let scale = s_k / (1.0 + s_k * v_c * θ_k_sum);
                        *φ_ck = Gamma::new(shape, scale).unwrap().sample(rng);
                    }
                }
            });

        Zip::from(&mut params.φ_v_dot)
            .and(params.φ.axis_iter(Axis(1)))
            .for_each(|φ_v_dot_k, φ_k| {
                *φ_v_dot_k = φ_k.dot(&params.effective_cell_volume);
            });
    }

    fn sample_cell_scales(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // for each cell
        Zip::indexed(&mut params.cell_scale)
            .and(&mut params.effective_cell_volume)
            .and(&params.log_cell_volume)
            .and(&params.z)
            .and(params.ωφ.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, a_c, eff_v_c, &log_v_c, &z_c, ω_c)| {
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

                let log_a_c = μ + σ2.sqrt() * randn(rng);
                *a_c = log_a_c.exp();
                *eff_v_c = (log_a_c + log_v_c).exp();
            });
    }

    fn sample_π(&self, params: &mut ModelParams) {
        let mut rng = rand::rng();
        let mut π_sum = 0.0;
        Zip::from(&mut params.π)
            .and(&params.component_population)
            .for_each(|π_t, pop_t| {
                *π_t = Gamma::new(1.0 + *pop_t as f32, 1.0)
                    .unwrap()
                    .sample(&mut rng);
                π_sum += *π_t;
            });

        // normalize to get dirichlet posterior
        params.π.iter_mut().for_each(|π_t| *π_t /= π_sum);

        Zip::from(&mut params.log_π)
            .and(&params.π)
            .for_each(|log_π_t, π_t| *log_π_t = π_t.ln());
    }

    pub(crate) fn sample_rφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // for each cell
        Zip::indexed(params.lφ.outer_iter_mut()) // for every cell
            .and(&params.z)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, l_c, &z_c)| {
                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);

                for (l_ck, x_ck, &r_k) in izip!(l_c, x_c.read().iter(), &params.rφ.row(z_c)) {
                    *l_ck = rand_crt(rng, x_ck, r_k);
                }
            });

        Zip::indexed(params.rφ.outer_iter_mut()) // for each component
            .and(params.sφ.outer_iter())
            .into_par_iter()
            .for_each_init(rng, |rng, (t, r_t, s_t)| {
                Zip::from(r_t) // each hidden dim
                    .and(s_t)
                    .and(params.lφ.axis_iter(Axis(1)))
                    .and(params.gate.axis_iter(Axis(1)))
                    .and(&params.θksum)
                    .for_each(|r_tk, s_tk, l_k, gate_k, θ_k_sum| {
                        // summing elements of lφ in component t. Off-cells
                        // (gate false) have x_ck = 0, hence l_ck = 0, so they
                        // drop out of this sum automatically.
                        let lsum = l_k
                            .iter()
                            .zip(&params.z)
                            .filter(|(_l_ck, z_c)| **z_c as usize == t)
                            .map(|(l_ck, _z_c)| *l_ck)
                            .sum::<u32>();

                        let shape = priors.eφ + lsum as f32;

                        // Only on-cells are NB observations; structural zeros
                        // must not contribute to the dispersion rate term.
                        let scale_inv = (1.0 / priors.fφ)
                            + izip!(&params.z, &params.effective_cell_volume, gate_k.iter())
                                .filter(|(z_c, _v_c, g_ck)| **z_c as usize == t && **g_ck)
                                .map(|(_z_c, v_c, _g_ck)| (*s_tk * v_c * *θ_k_sum).ln_1p())
                                .sum::<f32>();
                        let scale = scale_inv.recip();
                        *r_tk = Gamma::new(shape, scale).unwrap().sample(rng);
                        *r_tk = r_tk.max(priors.min_rφ);
                    });
            });
    }

    pub(crate) fn sample_ωck(&self, params: &mut ModelParams) {
        // for every cell
        Zip::indexed(params.ωφ.outer_iter_mut()) // for every cell
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .and(params.gate.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, ω_c, &z_c, &v_c, gate_c)| {
                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);

                for (ω_ck, x_ck, &r_k, &s_k, &θ_k_sum, &g_ck) in izip!(
                    ω_c,
                    x_c.read().iter(),
                    params.rφ.row(z_c),
                    params.sφ.row(z_c),
                    &params.θksum,
                    gate_c
                ) {
                    // Off-cells contribute nothing to the sφ posterior; zeroing
                    // ω keeps them out of the τ_sφ accumulation.
                    if g_ck {
                        let ε = (s_k * v_c * θ_k_sum).ln();
                        *ω_ck = PolyaGamma::new(x_ck as f32 + r_k, ε).sample(rng);
                    } else {
                        *ω_ck = 0.0;
                    }
                }
            });
    }

    pub(crate) fn sample_sφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let ncomponents = params.ncomponents();
        let nhidden = params.nhidden();

        // compute posterior precision
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

        // compute posterior means
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
                let x_c = params.cell_latent_counts.row(c);
                let r_t = params.rφ.row(z_c);
                let μ_sφ_t = μ_sφ_tl.row_mut(z_c);

                for (μ_sφ_tk, x_ck, &ω_ck, &r_tk, &θ_k_sum, &g_ck) in
                    izip!(μ_sφ_t, x_c.read().iter(), ω_c, r_t, &params.θksum, gate_c)
                {
                    // Skip structural zeros so they don't bias sφ downward.
                    if g_ck {
                        *μ_sφ_tk += (x_ck as f32 - r_tk) / 2.0 - ω_ck * (v_c * θ_k_sum).ln();
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

    fn sample_background_rates(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // TODO: worth doing ethier of these loops in parallel?
        let mut rng = rng();
        Zip::from(params.λ_bg.axis_iter_mut(Axis(2)))
            .and(&params.background_counts)
            .and(&params.background_region_volume)
            .for_each(|mut λ_d, x_d, &v_d| {
                for (λ_dl, x_dl) in izip!(λ_d.axis_iter_mut(Axis(1)), x_d) {
                    for (λ_dlg, x_dlg) in izip!(λ_dl, x_dl.iter()) {
                        let α = priors.α_bg + x_dlg as f32;
                        let β = priors.β_bg + v_d;
                        *λ_dlg = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) as f32;
                    }
                }
            });

        // // TODO: Crude hack to see what things look like if we don't vary background rates by layer
        // let mut background_counts: Array1<u32> = Array1::zeros(params.ngenes());
        // for x_l in params.background_counts.iter() {
        //     for (x_lg, y_g) in x_l.iter().zip(background_counts.iter_mut()) {
        //         *y_g += x_lg;
        //     }
        // }

        // let nlayers = params.nlayers();
        // Zip::from(params.λ_bg.columns_mut()).for_each(|λ_l| {
        //     for (λ_lg, x_lg) in izip!(λ_l, background_counts.iter()) {
        //         let α = priors.α_bg + *x_lg as f32;
        //         let β = priors.β_bg + params.layer_volume * nlayers as f32;
        //         *λ_lg = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) as f32;
        //     }
        // });

        Zip::from(&mut params.logλ_bg)
            .and(&params.λ_bg)
            .for_each(|logλ_bg, λ_bg| {
                *logλ_bg = λ_bg.ln();
            });
    }
}
