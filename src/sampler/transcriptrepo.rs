use log::trace;
use rand_distr::{Binomial, Distribution};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashSet;
use std::ops::DerefMut;
use std::time::{Duration, Instant};

use super::math::{halfnormal_x2_pdf, uniformly_imprecise_normal_prob};
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{
    MOORE_OFFSETS, RADIUS2_OFFSETS, RADIUS3_OFFSETS, VoxelCheckerboard, VoxelCountKey, VoxelOffset,
    VoxelQuad,
};
use super::{CountMatRowKey, ModelParams, ModelPriors};

use rand::rngs::ThreadRng;
use rand::{Rng, rng};

// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 7] = VON_NEUMANN_AND_SELF_OFFSETS;
// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 27] = MOORE_AND_SELF_OFFSETS;
// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 15] = RADIUS2_AND_SELF_OFFSETS;
// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 6] = VON_NEUMANN_OFFSETS;
// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 26] = MOORE_OFFSETS;
// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 14] = RADIUS2_OFFSETS;
const REPO_NEIGHBORHOOD: [(i32, i32, i32); 26] = RADIUS3_OFFSETS;

pub struct TranscriptRepo {
    prior_near: VoxelDiffusionPrior,
    prior_far: VoxelDiffusionPrior,
}

impl TranscriptRepo {
    pub fn new(priors: &ModelPriors, voxelsize: f32) -> Self {
        const EPS: f32 = 1e-5;
        TranscriptRepo {
            prior_near: VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_near, EPS),
            prior_far: VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_far, EPS),
        }
    }

    pub fn set_voxel_size(&mut self, priors: &ModelPriors, voxelsize: f32) {
        self.prior_near =
            VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_near, self.prior_near.eps);
        self.prior_far =
            VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_far, self.prior_far.eps);
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
                    voxels.voxelsize_z,
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
        voxelsize_z: f32,
        temperature: f32,
    ) {
        let quad_states = quad.states.read().unwrap();
        let mut quad_counts = quad.counts.write().unwrap();
        let quad_counts_ref = quad_counts.deref_mut();

        quad_counts_ref.counts_deltas.clear();

        // let mut proposed_total = [0; 26];
        // let mut accept_total = [0; 26];

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

            let k0 = voxel.k();
            let gene = *gene as usize;
            let [di0, dj0, dk0] = offset.coords();

            let cell = quad_states
                .states
                .get(voxel)
                .map(|state| state.cell)
                .unwrap_or(BACKGROUND_CELL);

            // let t0 = Instant::now();
            let mut λ_current = params.λ_bg[[gene, k0 as usize]];
            if cell != BACKGROUND_CELL {
                λ_current += params.λ(cell as usize, gene);
            }

            let dist_prob_current =
                self.diffusion_distance_prior(priors, voxelsize_z, di0, dj0, dk0);
            let current_prob = dist_prob_current * λ_current;

            let mut ρ = 1.0;
            let mut s = *count;
            let prop_prob = 1.0 / REPO_NEIGHBORHOOD.len() as f64;
            let mut total_moved = 0;
            REPO_NEIGHBORHOOD
                .iter()
                .enumerate()
                .for_each(|(_neighbor_idx, &(di, dj, dk))| {
                    if s == 0 {
                        return;
                    }

                    // sample from a marginal binomial to determine how many transcript we are
                    // proposing to move to this neighbor.
                    let r = (prop_prob / ρ).min(1.0);
                    let diffused_count = Binomial::new(s as u64, r).unwrap().sample(rng) as u32;
                    ρ -= prop_prob;
                    s -= diffused_count;

                    if diffused_count == 0 {
                        return;
                    }

                    // proposed_total[neighbor_idx] += diffused_count;

                    let neighbor = voxel.offset_coords(di, dj, dk);
                    let k = neighbor.k();
                    if neighbor.is_oob() || k < 0 || k > quad.kmax {
                        return;
                    }

                    let u = neighbor.i() as u32 / quadsize;
                    let v = neighbor.j() as u32 / quadsize;
                    if !quads_coords.contains(&(u, v)) {
                        return;
                    }

                    // sample from another binomial to determine how many of these move we accept
                    let mut λ_proposed = params.λ_bg[[gene, k as usize]];
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

                    let dist_prob_proposed =
                        self.diffusion_distance_prior(priors, voxelsize_z, di, dj, dk);

                    let proposal_prob = dist_prob_proposed * λ_proposed;

                    let accept_prob =
                        ((proposal_prob.ln() - current_prob.ln()) / temperature).exp() as f64;
                    let accepted_count = Binomial::new(diffused_count as u64, accept_prob.min(1.0))
                        .unwrap()
                        .sample(rng) as u32;

                    if accepted_count == 0 {
                        return;
                    }

                    // accept_total[neighbor_idx] += accepted_count;

                    quad_counts_ref.counts_deltas.push((
                        VoxelCountKey {
                            voxel: neighbor,
                            gene: gene as u32,
                            offset: VoxelOffset::new(di, dj, dk),
                        },
                        accepted_count,
                    ));

                    total_moved += accepted_count;
                });
            assert!(s == 0);
            assert!(total_moved <= *count);

            // dbg!(proposed_total);
            // dbg!(accept_total);

            if total_moved > 0 {
                *count -= total_moved;
                if cell == BACKGROUND_CELL {
                    params.unassigned_counts[k1 as usize].sub(gene, total_moved);
                } else {
                    let counts_c = params.counts.row(cell as usize);
                    counts_c
                        .write()
                        .sub(CountMatRowKey::new(gene as u32, k0 as u32), total_moved);
                }
            }
        }

        // Clear out any zeros
        quad_counts.counts.retain(|_key, count| *count > 0);
    }

    fn diffusion_distance_prior(
        &self,
        priors: &ModelPriors,
        voxelsize_z: f32,
        di: i32,
        dj: i32,
        dk: i32,
    ) -> f32 {
        // TODO: We need to know how many layers to normalize to [0, 1]

        let dz = dk as f32 * voxelsize_z;
        let z_prob = halfnormal_x2_pdf(priors.σ_z_diffusion, dz * dz);

        let xy_prob = priors.p_diffusion * self.prior_far.prob(di) * self.prior_far.prob(dj)
            + (1.0 - priors.p_diffusion) * self.prior_near.prob(di) * self.prior_near.prob(dj);

        z_prob * xy_prob
    }
}

struct VoxelDiffusionPrior {
    eps: f32,
    pmf: Vec<f32>,
}

// Simple memoized discrete distance prior
impl VoxelDiffusionPrior {
    fn new(voxelsize: f32, σ: f32, eps: f32) -> VoxelDiffusionPrior {
        let mut pmf = Vec::new();

        let mut d = 0;
        loop {
            let p =
                uniformly_imprecise_normal_prob(0.0, voxelsize, d as f32, d as f32 + voxelsize, σ);
            pmf.push(p);
            if p < eps {
                break;
            }
            d += 1;
        }

        dbg!(voxelsize, σ, &pmf);

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
