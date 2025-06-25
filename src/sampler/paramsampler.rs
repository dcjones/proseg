use super::math::{negbin_logpmf, normal_logpdf, odds_to_prob, rand_crt, randn};
use super::multinomial::Multinomial;
use super::polyagamma::PolyaGamma;
use super::{ModelParams, ModelPriors, RAYON_CELL_MIN_LEN};
use itertools::izip;
use libm::lgammaf;
use log::{info, trace};
use ndarray::{Array1, Array2, Axis, Zip, s};
use rand::{Rng, rng};
use rand_distr::{Binomial, Distribution, Gamma, Normal};
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
        self.sample_foreground_background(params, purge_sparse_mats);
        trace!("sample_foreground_background: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_factor_model(priors, params, sample_z, burnin, purge_sparse_mats);
        info!("sample_factor_model: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_background_rates(priors, params);
        trace!("sample_background_rates: {:?}", t0.elapsed());

        if !burnin && record_samples {
            // TODO: temporarily disabling this to try to save as much memory as I can.
            //     params
            //         .foreground_counts_lower
            //         .update(&params.foreground_counts);
            //     params
            //         .foreground_counts_upper
            //         .update(&params.foreground_counts);
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
            .par_rows()
            .zip(params.foreground_counts.par_rows())
            .enumerate()
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, (cell, (row, foreground_row))| {
                let mut foreground_row = foreground_row.write();

                let mut λ_cg = 0.0;
                let mut gene = u32::MAX;

                for (gene_layer, count) in row.read().iter_nonzeros() {
                    if gene_layer.gene != gene {
                        gene = gene_layer.gene;
                        λ_cg = params.λ(cell, gene as usize);
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
            });

        // let mut nunassigned = 0;
        // for x_l in &params.unassigned_counts {
        //     for count in x_l.iter() {
        //         nunassigned += count;
        //     }
        // }

        // let mut nbackground = 0;
        // for x_l in &params.background_counts {
        //     for count in x_l.iter() {
        //         nbackground += count;
        //     }
        // }
        // info!("sum(unassigned_counts): {}", nunassigned);
        // info!("sum(background_counts): {}", nbackground);
        // info!("sum(foreground_counts): {}", params.foreground_counts.sum());
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

        params.λ.clear();
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
            x.borrow_mut().fill(0);
        }

        let ngenes = params.ngenes();
        let nhidden = params.nhidden();
        info!("sample_latent_counts: init: {:?}", t0.elapsed());

        let t0 = Instant::now();
        params
            .cell_latent_counts
            .par_rows()
            .zip(params.foreground_counts.par_rows())
            .zip(params.φ.outer_iter())
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |rng, ((cell_latent_counts_c, x_c), φ_c)| {
                let mut multinomial = params
                    .multinomials
                    .get_or(|| RefCell::new(Multinomial::with_k(nhidden - params.nunfactored)))
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

                // distribute counts from fatored genes
                for ((g, x_cg), θ_g) in x_c
                    .iter_nonzeros_from(params.nunfactored as u32)
                    .zip(params.θ.slice(s![params.nunfactored.., ..]).outer_iter())
                {
                    if x_cg == 0 {
                        continue;
                    }

                    for (k, outcome, &φ_ck, &θ_gk) in izip!(
                        params.nunfactored..nhidden,
                        multinomial.outcomes.iter_mut(),
                        &φ_c.slice(s![params.nunfactored..]),
                        &θ_g.slice(s![params.nunfactored..])
                    ) {
                        outcome.prob = φ_ck * θ_gk;
                        outcome.index = k as u32;
                    }

                    let mut gene_latent_counts_g = gene_latent_counts_tl.row_mut(g as usize);
                    multinomial.sample(rng, x_cg, |k, x| {
                        cell_latent_counts_c.add(k as u32, x);
                        gene_latent_counts_g[k] += x;
                    });
                }
            });
        info!("sample_latent_counts: sample: {:?}", t0.elapsed());

        // accumulate from thread locate matrices
        let t0 = Instant::now();
        params.gene_latent_counts.fill(0);
        for x in params.gene_latent_counts_tl.iter_mut() {
            params.gene_latent_counts.scaled_add(1, &x.borrow());
        }

        // marginal count along the hidden axis
        Zip::from(&mut params.latent_counts)
            .and(params.gene_latent_counts.columns())
            .for_each(|lc, glc| {
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
            .for_each_init(rng, |rng, (c, φ_c, z_c, v_c)| {
                let x_c = params.cell_latent_counts.row(c);
                let z_c = *z_c as usize;

                for (φ_ck, &θ_k_sum, x_ck, &r_k, s_k) in izip!(
                    φ_c,
                    &params.θksum,
                    x_c.read().iter(),
                    &params.rφ.row(z_c),
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
    }

    fn sample_rφ(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // for each cell
        Zip::indexed(params.lφ.outer_iter_mut()) // for every cell
            .and(&params.z)
            .into_par_iter()
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
                    .and(&params.θksum)
                    .for_each(|r_tk, s_tk, l_k, θ_k_sum| {
                        // summing elements of lφ in component t
                        let lsum = l_k
                            .iter()
                            .zip(&params.z)
                            .filter(|(_l_ck, z_c)| **z_c as usize == t)
                            .map(|(l_ck, _z_c)| *l_ck)
                            .sum::<u32>();

                        let shape = priors.eφ + lsum as f32;

                        let scale_inv = (1.0 / priors.fφ)
                            + params
                                .z
                                .iter()
                                .zip(&params.effective_cell_volume)
                                .filter(|(z_c, _v_c)| **z_c as usize == t)
                                .map(|(_z_c, v_c)| (*s_tk * v_c * *θ_k_sum).ln_1p())
                                .sum::<f32>();
                        let scale = scale_inv.recip();
                        *r_tk = Gamma::new(shape, scale).unwrap().sample(rng);
                        *r_tk = r_tk.max(2e-4);
                    });
            });
    }

    fn sample_ωck(&self, params: &mut ModelParams) {
        // for every cell
        Zip::indexed(params.ωφ.outer_iter_mut()) // for every cell
            .and(&params.z)
            .and(&params.effective_cell_volume)
            .into_par_iter()
            .for_each_init(rng, |rng, (c, ω_c, &z_c, &v_c)| {
                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);

                for (ω_ck, x_ck, &r_k, &s_k, &θ_k_sum) in izip!(
                    ω_c,
                    x_c.read().iter(),
                    params.rφ.row(z_c),
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
            .par_for_each(|&z_c, ω_c| {
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
            .par_for_each(|c, &z_c, &v_c, ω_c| {
                let mut μ_sφ_tl = params
                    .sφ_work_tl
                    .get_or(|| RefCell::new(Array2::zeros((ncomponents, nhidden))))
                    .borrow_mut();

                let z_c = z_c as usize;
                let x_c = params.cell_latent_counts.row(c);
                let r_t = params.rφ.row(z_c);
                let μ_sφ_t = μ_sφ_tl.row_mut(z_c);

                for (μ_sφ_tk, x_ck, &ω_ck, &r_tk, &θ_k_sum) in
                    izip!(μ_sφ_t, x_c.read().iter(), ω_c, r_t, &params.θksum)
                {
                    *μ_sφ_tk += (x_ck as f32 - r_tk) / 2.0 - ω_ck * (v_c * θ_k_sum).ln();
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
        Zip::from(params.λ_bg.columns_mut())
            .and(&params.background_counts)
            .for_each(|λ_l, x_l| {
                for (λ_lg, x_lg) in izip!(λ_l, x_l.iter()) {
                    let α = priors.α_bg + x_lg as f32;
                    let β = priors.β_bg + params.layer_volume;
                    *λ_lg = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) as f32;
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
