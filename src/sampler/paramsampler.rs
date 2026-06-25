use super::math::{negbin_logpmf, normal_logpdf, odds_to_prob, rand_crt, randn};
use super::multinomial::Multinomial;
use super::polyagamma::PolyaGamma;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{TranscriptFixedState, VoxelCheckerboard};
use super::{FlowStats, ModelParams, ModelPriors, RAYON_CELL_MIN_LEN, TranscriptAssignment};
use itertools::izip;
use libm::lgammaf;
use log::{info, trace};
use ndarray::{Array2, Axis, Zip, s};
use rand::{Rng, rng};
use rand_distr::{Distribution, Gamma, Normal};
use rayon::prelude::*;
use std::cell::RefCell;
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
        // Parallelization overhead here may outweight the reward. Alternative
        // strategy may be to process this one shard at a time and parallelize within
        // shards.
        params
            .log_cell_volume
            .iter_mut()
            .zip(params.cell_voxel_count.iter())
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

        // Iterate over quads in parallel to sync, sample, and record
        voxels.quads.par_iter().for_each(|((_u, _v), quad)| {
            let transcripts = quad.transcripts.read().unwrap();
            let voxel_states = quad.states.read().unwrap();
            let mut rng = rand::rng();

            for transcript in transcripts.transcripts.iter() {
                let idx = transcript.transcript_idx as usize;

                // Get destination cell from the current voxel's state in the checkerboard
                let cell = voxel_states.get_voxel_cell(transcript.voxel);

                let &TranscriptFixedState {
                    original_voxel,
                    gene,
                } = voxels.transcript_fixed_state.get(idx as u32);

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
        let t0 = Instant::now();
        self.sample_latent_counts(params, purge_sparse_mats);
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
        params.update_phi_theta_dot();
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
                .zip(params.cell_voxel_count.iter())
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
            .for_each(|(lgamma_r_ck, r_ck)| {
                *lgamma_r_ck = lgammaf(*r_ck);
            });

        let ncomponents = params.ncomponents();
        Zip::indexed(&mut params.z) // for each cell
            .and(params.φ.rows())
            .and(&params.effective_cell_volume)
            .and(&params.log_cell_volume)
            .and(params.rφ.rows())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (i, z_c, φ_c, ev_c, log_v_c, r_c)| {
                let x_c_lock = params.cell_latent_counts.row(i);
                let x_c = x_c_lock.read();
                let lgamma_r_c = params.lgamma_rφ.row(i);

                let mut z_probs = params
                    .z_probs
                    .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                    .borrow_mut();

                // compute probability of φ_c under every component
                // (rφ is per-cell, so it stays fixed across candidate
                // components; only the component-level scale sφ varies)

                // for every component
                let mut z_probs_sum = 0.0;
                for (z_probs_t, log_π_t, s_t, μ_vol_c, σ_vol_c) in izip!(
                    z_probs.iter_mut(),
                    params.log_π.iter(),
                    params.sφ.rows(),
                    &params.μ_volume,
                    &params.σ_volume
                ) {
                    *z_probs_t = *log_π_t as f64;

                    for (r_ck, lgamma_r_ck, s_tk, θ_k_sum, x_ck) in
                        izip!(&r_c, &lgamma_r_c, s_t, &params.θksum, x_c.iter())
                    {
                        let p = odds_to_prob(*s_tk * *ev_c * *θ_k_sum);
                        let lp = negbin_logpmf(*r_ck, *lgamma_r_ck, p, x_ck) as f64;
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

    fn sample_latent_counts(&self, params: &mut ModelParams, purge: bool) {
        let t0 = Instant::now();
        if purge {
            params.cell_latent_counts.clear();
        } else {
            params.cell_latent_counts.zero();
        }

        // zero out thread local gene latent counts
        for x in params.gene_latent_counts_tl.iter_mut() {
            x.get_mut().fill(0);
        }

        let ngenes = params.ngenes();
        let nhidden = params.nhidden();
        params
            .cell_latent_counts
            .par_rows()
            .zip(params.foreground_counts.par_rows())
            .zip(params.φ.outer_iter())
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, ((cell_latent_counts_c, x_c), φ_c)| {
                let mut multinomial = params
                    .multinomials
                    .get_or(|| RefCell::new(Multinomial::new(nhidden - params.nunfactored)))
                    .borrow_mut();

                let mut gene_latent_counts_tl = params
                    .gene_latent_counts_tl
                    .get_or(|| RefCell::new(Array2::zeros((ngenes, nhidden))))
                    .borrow_mut();

                let x_c = x_c.read();
                let mut cell_latent_counts_c = cell_latent_counts_c.write();

                // assign counts from unfactored genes
                for (g, x_cg) in x_c.iter_nonzeros_to(params.nunfactored as u32) {
                    if x_cg > 0 {
                        cell_latent_counts_c.add(g, x_cg);
                        gene_latent_counts_tl[[g as usize, g as usize]] += x_cg;
                    }
                }

                // distribute counts from factored genes
                let φ_c_factored = φ_c.slice(s![params.nunfactored..]);
                for (g, x_cg) in x_c.iter_nonzeros_from(params.nunfactored as u32) {
                    if x_cg == 0 {
                        continue;
                    }

                    let θ_g_factored = params.θ.slice(s![g as usize, params.nunfactored..]);

                    let prob_iter = φ_c_factored
                        .iter()
                        .zip(θ_g_factored.iter())
                        .map(|(φ_ck, θ_gk)| *φ_ck * *θ_gk);
                    multinomial.set_probs_from_iter(prob_iter);

                    let mut gene_latent_counts_g = gene_latent_counts_tl.row_mut(g as usize);
                    let nunfactored = params.nunfactored;
                    multinomial.sample(rng, x_cg, |k, x| {
                        cell_latent_counts_c.add((k + nunfactored) as u32, x);
                        gene_latent_counts_g[k + nunfactored] += x;
                    });
                }
            });

        // accumulate from thread locate matrices
        let tl_matrices: Vec<&Array2<u32>> = params
            .gene_latent_counts_tl
            .iter_mut()
            .map(|x| &*x.get_mut())
            .collect();

        params
            .gene_latent_counts
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(g, mut row)| {
                row.fill(0);
                for tl in &tl_matrices {
                    row += &tl.row(g);
                }
            });
        // marginal count along the hidden axis
        Zip::from(&mut params.latent_counts)
            .and(params.gene_latent_counts.columns())
            .par_for_each(|lc, glc| {
                *lc = glc.sum();
            });

        let count = params.latent_counts.mapv(|v| v as u64).sum();
        assert!(params.gene_latent_counts.mapv(|v| v as u64).sum() == count);
        assert!(params.cell_latent_counts.sum() == count);

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
        info!("sample_latent_counts: accumulation: {:?}", t0.elapsed());

        info!("component_population: {:?}", &params.component_population);
    }

    fn sample_θ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        let mut θfac = params
            .θ
            .slice_mut(s![params.nunfactored.., params.nunfactored..]);
        let gene_latent_counts_fac = params
            .gene_latent_counts
            .slice(s![params.nunfactored.., params.nunfactored..]);

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
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, φ_c, z_c, v_c)| {
                let x_c = params.cell_latent_counts.row(c);
                let z_c = *z_c as usize;

                for (φ_ck, &θ_k_sum, x_ck, &r_k, s_k) in izip!(
                    φ_c,
                    &params.θksum,
                    x_c.read().iter(),
                    &params.rφ.row(c),
                    &params.sφ.row(z_c)
                ) {
                    let shape = r_k + x_ck as f32;
                    let scale = s_k / (1.0 + s_k * v_c * θ_k_sum);
                    *φ_ck = Gamma::new(shape, scale).unwrap().sample(rng);
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
                for (&θ_k_sum, x_ck, &r_ck, &s_tk, &ω_ck) in izip!(
                    &params.θksum,
                    x_c.read().iter(),
                    params.rφ.row(c),
                    params.sφ.row(z_c),
                    ω_c
                ) {
                    μ += (x_ck as f32 - r_ck) / 2.0 - ω_ck * ((s_tk * θ_k_sum).ln() + log_v_c);
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

    fn sample_rφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // CRT auxiliary variables, using each cell's own rφ (rather than a
        // shared per-component value)
        Zip::indexed(params.lφ.outer_iter_mut()) // for every cell
            .and(params.rφ.outer_iter())
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, l_c, r_c)| {
                let x_c = params.cell_latent_counts.row(c);

                for (l_ck, x_ck, &r_ck) in izip!(l_c, x_c.read().iter(), &r_c) {
                    *l_ck = rand_crt(rng, x_ck, r_ck);
                }
            });

        // rφ is now per-cell rather than pooled across an entire component,
        // so each cell's posterior only ever sees that cell's one CRT draw.
        // This keeps the prior (eφ, fφ) from being overwhelmed as the number
        // of cells in a component grows.
        Zip::indexed(params.rφ.outer_iter_mut()) // for every cell
            .and(&params.z)
            .and(params.lφ.outer_iter())
            .and(&params.effective_cell_volume)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (_c, r_c, &z_c, l_c, &v_c)| {
                let z_c = z_c as usize;
                let s_t = params.sφ.row(z_c);

                Zip::from(r_c) // each hidden dim
                    .and(l_c)
                    .and(s_t)
                    .and(&params.θksum)
                    .for_each(|r_ck, &l_ck, &s_tk, &θ_k_sum| {
                        let shape = priors.eφ + l_ck as f32;
                        let scale_inv = (1.0 / priors.fφ) + (s_tk * v_c * θ_k_sum).ln_1p();
                        let scale = scale_inv.recip();
                        *r_ck = Gamma::new(shape, scale).unwrap().sample(rng);
                        *r_ck = r_ck.max(2e-4);
                    });
            });
    }

    fn sample_ωck(&self, params: &mut ModelParams) {
        // for every cell
        Zip::indexed(params.ωφ.outer_iter_mut()) // for every cell
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (c, ω_c, &z_c, &v_c)| {
                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);

                for (ω_ck, x_ck, &r_k, &s_k, &θ_k_sum) in izip!(
                    ω_c,
                    x_c.read().iter(),
                    params.rφ.row(c),
                    params.sφ.row(z_c),
                    &params.θksum
                ) {
                    let ε = (s_k * v_c * θ_k_sum).ln();
                    *ω_ck = PolyaGamma::new(x_ck as f32 + r_k, ε).sample(rng);
                }
            });
    }

    fn sample_sφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
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
            .into_par_iter()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each(|(c, &z_c, &v_c, ω_c)| {
                let mut μ_sφ_tl = params
                    .sφ_work_tl
                    .get_or(|| RefCell::new(Array2::zeros((ncomponents, nhidden))))
                    .borrow_mut();

                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);
                let r_c = params.rφ.row(c);
                let μ_sφ_t = μ_sφ_tl.row_mut(z_c);

                for (μ_sφ_tk, x_ck, &ω_ck, &r_ck, &θ_k_sum) in
                    izip!(μ_sφ_t, x_c.read().iter(), ω_c, r_c, &params.θksum)
                {
                    *μ_sφ_tk += (x_ck as f32 - r_ck) / 2.0 - ω_ck * (v_c * θ_k_sum).ln();
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
