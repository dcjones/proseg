use log::trace;
use rand_distr::{Binomial, Distribution};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashSet;
use std::ops::{DerefMut, Neg};
use std::time::Instant;

use super::math::{MultinomialSampler, uniformly_imprecise_normal_prob};
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{VoxelCheckerboard, VoxelCountKey, VoxelOffset, VoxelQuad};
use super::{CountMatRowKey, ModelParams, ModelPriors};

use rand::rng;
use rand::rngs::ThreadRng;

pub struct TranscriptRepo {
    prior_near: VoxelDiffusionPrior,
    prior_far: VoxelDiffusionPrior,
    prior_z: VoxelDiffusionPrior,
    proposal_xy_probs: Vec<f64>,
    proposal_z_probs: Vec<f64>,
}

impl TranscriptRepo {
    pub fn new(priors: &ModelPriors, voxelsize: f32, voxelsize_z: f32) -> Self {
        const EPS: f32 = 1e-5;

        let mut proposal_xy_probs =
            VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_proposal, EPS)
                .pmf
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>();
        proposal_xy_probs.reverse();
        let n = proposal_xy_probs.len();
        proposal_xy_probs.resize(2 * n - 1, 0.0);
        for i in 1..n {
            proposal_xy_probs[(n - 1) + i] = proposal_xy_probs[(n - 1) - i];
        }

        let mut proposal_z_probs =
            VoxelDiffusionPrior::new(voxelsize_z, priors.σ_z_diffusion_proposal, EPS)
                .pmf
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>();
        proposal_z_probs.reverse();
        let n = proposal_z_probs.len();
        proposal_z_probs.resize(2 * n - 1, 0.0);
        for i in 1..n {
            proposal_z_probs[(n - 1) + i] = proposal_z_probs[(n - 1) - i];
        }

        TranscriptRepo {
            prior_near: VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_near, EPS),
            prior_far: VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_far, EPS),
            prior_z: VoxelDiffusionPrior::new(voxelsize, priors.σ_z_diffusion, EPS),
            proposal_xy_probs,
            proposal_z_probs,
        }
    }

    pub fn set_voxel_size(&mut self, priors: &ModelPriors, voxelsize: f32, voxelsize_z: f32) {
        let eps = self.prior_near.eps;

        self.proposal_xy_probs =
            VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_proposal, eps)
                .pmf
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>();
        self.proposal_xy_probs.reverse();
        let n = self.proposal_xy_probs.len();
        self.proposal_xy_probs.resize(2 * n - 1, 0.0);
        for i in 1..n {
            self.proposal_xy_probs[(n - 1) + i] = self.proposal_xy_probs[(n - 1) - i];
        }

        self.proposal_z_probs =
            VoxelDiffusionPrior::new(voxelsize_z, priors.σ_z_diffusion_proposal, eps)
                .pmf
                .iter()
                .map(|v| *v as f64)
                .collect::<Vec<_>>();
        self.proposal_z_probs.reverse();
        let n = self.proposal_z_probs.len();
        self.proposal_z_probs.resize(2 * n - 1, 0.0);
        for i in 1..n {
            self.proposal_z_probs[(n - 1) + i] = self.proposal_z_probs[(n - 1) - i];
        }

        self.prior_near =
            VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_near, self.prior_near.eps);
        self.prior_far =
            VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_far, self.prior_far.eps);
        self.prior_z = VoxelDiffusionPrior::new(voxelsize_z, priors.σ_z_diffusion, eps);
    }

    pub fn sample(
        &self,
        voxels: &mut VoxelCheckerboard,
        priors: &ModelPriors,
        params: &mut ModelParams,
        temperature: f32,
    ) {
        let t0 = Instant::now();
        voxels
            .quads
            .par_iter()
            .for_each_init(rng, |rng, (_key, quad)| {
                self.quad_transcript_repo(
                    voxels,
                    rng,
                    priors,
                    params,
                    quad,
                    &voxels.quads_coords,
                    voxels.quadsize as u32,
                    voxels.voxelsize,
                    temperature,
                );
            });
        trace!("transcript repo: compute deltas: {:?}", t0.elapsed());

        let t0 = Instant::now();
        voxels.merge_counts_deltas(params);
        trace!("transcript repo: merge deltas: {:?}", t0.elapsed());
    }

    #[allow(clippy::too_many_arguments)]
    fn quad_transcript_repo(
        &self,
        voxels: &VoxelCheckerboard,
        rng: &mut ThreadRng,
        priors: &ModelPriors,
        params: &ModelParams,
        quad: &VoxelQuad,
        quads_coords: &HashSet<(u32, u32)>,
        quadsize: u32,
        _voxelsize: f32,
        _temperature: f32,
    ) {
        let quad_states = quad.states.read().unwrap();
        let mut quad_counts = quad.counts.write().unwrap();
        let quad_counts_ref = quad_counts.deref_mut();

        quad_counts_ref.counts_deltas.clear();

        // let mut proposed_total = [0; 26];
        // let mut accept_total = [0; 26];
        // let mut proposed_total = 0;
        // let mut accept_total = 0;

        // let mut inc_proposal_total = 0;
        // let mut dec_proposal_total = 0;
        // let mut eq_proposal_total = 0;

        for (
            VoxelCountKey {
                voxel,
                gene,
                offset,
            },
            count,
        ) in quad_counts_ref.counts.iter_mut()
        {
            if *count == 0 {
                continue;
            }

            let gene = *gene as usize;
            let origin = voxel.offset(offset.neg());
            let k0 = voxel.k();
            let [di0, dj0, dk0] = offset.coords();
            let k_origin = origin.k();

            let cell = quad_states
                .states
                .get(voxel)
                .map(|state| state.cell)
                .unwrap_or(BACKGROUND_CELL);

            let density = voxels.get_voxel_density_hint(quad, origin);

            let λ_bg = params.λ_bg[[gene, k_origin as usize, density]];

            let mut λ_current = λ_bg;
            if cell != BACKGROUND_CELL {
                λ_current += params.λ(cell as usize, gene);
            }

            let dist_prob_current = self.diffusion_distance_prior(priors, di0, dj0, dk0);
            let current_prob = dist_prob_current * λ_current;

            let mut total_moved = 0;
            for (dk, c_k) in
                MultinomialSampler::new(&mut rng.clone(), &self.proposal_z_probs, *count)
                    .enumerate()
            {
                if c_k == 0 {
                    continue;
                }

                let dk = (dk as i32) - ((self.proposal_z_probs.len() - 1) / 2) as i32;

                if k0 + dk < 0 || k0 + dk > quad.kmax {
                    continue;
                }

                for (dj, c_jk) in
                    MultinomialSampler::new(&mut rng.clone(), &self.proposal_xy_probs, c_k)
                        .enumerate()
                {
                    if c_jk == 0 {
                        continue;
                    }

                    let dj = (dj as i32) - ((self.proposal_xy_probs.len() - 1) / 2) as i32;

                    for (di, c_ijk) in
                        MultinomialSampler::new(&mut rng.clone(), &self.proposal_xy_probs, c_jk)
                            .enumerate()
                    {
                        if c_ijk == 0 {
                            continue;
                        }

                        let di = (di as i32) - ((self.proposal_xy_probs.len() - 1) / 2) as i32;
                        let neighbor = voxel.offset_coords(di, dj, dk);
                        if neighbor.is_oob() {
                            continue;
                        }

                        // don't repo into a quad that doesn't exist
                        let u = neighbor.i() as u32 / quadsize;
                        let v = neighbor.j() as u32 / quadsize;
                        if !quads_coords.contains(&(u, v)) {
                            continue;
                        }

                        // sample from another binomial to determine how many of these move we accept
                        let mut λ_proposed = λ_bg;
                        let neighbor_cell = if quad.voxel_in_bounds(neighbor) {
                            quad_states
                                .states
                                .get(&neighbor)
                                .map(|state| state.cell)
                                .unwrap_or(BACKGROUND_CELL)
                        } else {
                            voxels.get_voxel_cell(neighbor)
                        };

                        if neighbor_cell != BACKGROUND_CELL {
                            λ_proposed += params.λ(neighbor_cell as usize, gene);
                        }

                        let di = di + di0;
                        let dj = dj + dj0;
                        let dk = dk + dk0;

                        let dist_prob_proposed = self.diffusion_distance_prior(priors, di, dj, dk);

                        let proposal_prob = dist_prob_proposed * λ_proposed;

                        // if proposal_prob == current_prob {
                        //     eq_proposal_total += c_ijk;
                        // } else if proposal_prob < current_prob {
                        //     dec_proposal_total += c_ijk;
                        // } else {
                        //     inc_proposal_total += c_ijk;
                        // }

                        let accept_prob = (proposal_prob.ln() - current_prob.ln()).exp() as f64;
                        let accepted_count = Binomial::new(c_ijk as u64, accept_prob.min(1.0))
                            .unwrap()
                            .sample(rng) as u32;

                        // proposed_total += c_ijk as usize;
                        // accept_total += accepted_count as usize;

                        if accepted_count == 0 {
                            continue;
                        }

                        quad_counts_ref.counts_deltas.push((
                            VoxelCountKey {
                                voxel: neighbor,
                                gene: gene as u32,
                                offset: VoxelOffset::new(di, dj, dk),
                            },
                            accepted_count,
                        ));

                        total_moved += accepted_count;
                    }
                }
            }

            if total_moved > 0 {
                assert!(total_moved <= *count);
                *count -= total_moved;
                if cell == BACKGROUND_CELL {
                    params.unassigned_counts[density][k_origin as usize].sub(gene, total_moved);
                } else {
                    let counts_c = params.counts.row(cell as usize);
                    counts_c.write().sub(
                        CountMatRowKey::new(gene as u32, k_origin as u32, density as u8),
                        total_moved,
                    );
                }
            }
        }
        // let accept_rate = accept_total as f64 / proposed_total as f64;
        // dbg!((
        //     accept_total,
        //     proposed_total,
        //     accept_rate,
        //     inc_proposal_total,
        //     eq_proposal_total,
        //     dec_proposal_total
        // ));

        // Clear out any zeros
        quad_counts.counts.retain(|_key, count| *count > 0);
    }

    fn diffusion_distance_prior(&self, priors: &ModelPriors, di: i32, dj: i32, dk: i32) -> f32 {
        let z_prob = self.prior_z.prob(dk);
        let xy_prob = priors.p_diffusion * self.prior_far.prob(di) * self.prior_far.prob(dj)
            + (1.0 - priors.p_diffusion) * self.prior_near.prob(di) * self.prior_near.prob(dj);

        z_prob * xy_prob
    }
}

struct VoxelDiffusionPrior {
    eps: f32,
    pub pmf: Vec<f32>,
}

// Simple memoized discrete distance prior
impl VoxelDiffusionPrior {
    fn new(voxelsize: f32, σ: f32, eps: f32) -> VoxelDiffusionPrior {
        let mut pmf = Vec::new();

        let mut d = 0.0;
        loop {
            let p =
                voxelsize * uniformly_imprecise_normal_prob(0.0, voxelsize, d, d + voxelsize, σ);
            pmf.push(p);
            if p < eps {
                break;
            }
            d += voxelsize;
        }

        VoxelDiffusionPrior { eps, pmf }
    }

    fn prob(&self, d: i32) -> f32 {
        let dist = d.unsigned_abs() as usize;
        if dist < self.pmf.len() {
            self.pmf[dist]
        } else {
            self.eps
        }
    }
}
