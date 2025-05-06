use log::info;
use rand_distr::{Binomial, Distribution};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::collections::HashSet;
use std::ops::DerefMut;
use std::time::{Duration, Instant};

use super::math::halfnormal_x2_pdf;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{
    VON_NEUMANN_AND_SELF_OFFSETS, VoxelCheckerboard, VoxelCountKey, VoxelOffset, VoxelQuad,
};
use super::{CountMatRowKey, ModelParams, ModelPriors};

use rand::rng;
use rand::rngs::ThreadRng;

const REPO_NEIGHBORHOOD: [(i32, i32, i32); 7] = VON_NEUMANN_AND_SELF_OFFSETS;
// const REPO_NEIGHBORHOOD: [(i32, i32, i32); 27] = MOORE_AND_SELF_OFFSETS;

pub struct TranscriptRepo {}

impl TranscriptRepo {
    pub fn new() -> Self {
        TranscriptRepo {}
    }

    pub fn sample(
        &self,
        voxels: &mut VoxelCheckerboard,
        priors: &ModelPriors,
        params: &mut ModelParams,
    ) {
        let t0 = Instant::now();
        voxels
            .quads
            .par_iter()
            .for_each_init(rng, |rng, (_key, quad)| {
                let mut quad_lock = quad.write().unwrap();
                let quad_lock_ref = quad_lock.deref_mut();
                quad_transcript_repo(
                    rng,
                    priors,
                    params,
                    quad_lock_ref,
                    &voxels.quads_coords,
                    voxels.quadsize as u32,
                    voxels.voxelsize,
                    voxels.voxelsize_z,
                );
            });
        info!("transcript repo: compute deltas: {:?}", t0.elapsed());

        let t0 = Instant::now();
        voxels.merge_counts_deltas(params);
        info!("transcript repo: merge deltas: {:?}", t0.elapsed());
    }
}

#[allow(clippy::too_many_arguments)]
fn quad_transcript_repo(
    rng: &mut ThreadRng,
    priors: &ModelPriors,
    params: &ModelParams,
    quad: &mut VoxelQuad,
    quads_coords: &HashSet<(u32, u32)>,
    quadsize: u32,
    voxelsize: f32,
    voxelsize_z: f32,
) {
    quad.counts_deltas.clear();

    let mut compute_probs_elapsed = Duration::ZERO;
    let mut multinomial_sampling_elapsed = Duration::ZERO;

    for (
        VoxelCountKey {
            voxel,
            gene,
            offset,
        },
        count,
    ) in quad.counts.iter_mut()
    {
        if *count == 0 {
            continue;
        }

        let k = voxel.k();
        let gene = *gene as usize;
        let [di0, dj0, dk0] = offset.coords();

        let cell = quad
            .states
            .get(voxel)
            .map(|state| state.cell)
            .unwrap_or(BACKGROUND_CELL);

        let t0 = Instant::now();
        let λ_bg = params.λ_bg[[gene, k as usize]];
        let neighbor_probs = REPO_NEIGHBORHOOD.map(|(di, dj, dk)| {
            let neighbor = voxel.offset_coords(di, dj, dk);
            let k = neighbor.k();
            if neighbor.is_oob() || k < 0 || k > quad.kmax {
                return 0_f32;
            }

            let u = neighbor.i() as u32 / quadsize;
            let v = neighbor.j() as u32 / quadsize;
            if !quads_coords.contains(&(u, v)) {
                return 0_f32;
            }

            let mut λ = λ_bg;
            let neighbor_cell = quad
                .states
                .get(&neighbor)
                .map(|state| state.cell)
                .unwrap_or(BACKGROUND_CELL);

            if neighbor_cell != BACKGROUND_CELL {
                λ += params.λ(neighbor_cell as usize, gene);
            }

            let di = di + di0;
            let dj = dj + dj0;
            let dk = dk + dk0;

            let sq_dist_xy = di * di + dj * dj;
            let sq_dist_z = dk * dk;

            let mut sq_dist_prob = priors.p_diffusion
                * halfnormal_x2_pdf(
                    priors.σ_xy_diffusion,
                    sq_dist_xy as f32 * (voxelsize * voxelsize),
                )
                * halfnormal_x2_pdf(
                    priors.σ_z_diffusion,
                    sq_dist_z as f32 * (voxelsize_z * voxelsize_z),
                );
            if sq_dist_xy == 0 && sq_dist_z == 0 {
                sq_dist_prob += 1.0 - priors.p_diffusion;
            }

            sq_dist_prob * λ
        });
        assert!(neighbor_probs[0] > 0.0);
        compute_probs_elapsed += t0.elapsed();

        let sum_probs = neighbor_probs.iter().map(|v| *v as f64).sum::<f64>();
        if sum_probs == 0.0 {
            continue;
        }

        let t0 = Instant::now();
        // multinomial sampling (by sampling from Binomial marginals)
        {
            let mut ρ = 1.0;
            let mut s = *count;

            for (step, (&(di, dj, dk), p)) in
                REPO_NEIGHBORHOOD.iter().zip(neighbor_probs).enumerate()
            {
                let p = p as f64 / sum_probs;
                let diffused_count = if step == REPO_NEIGHBORHOOD.len() - 1 {
                    s
                } else if ρ > 0.0 {
                    let r = (p / ρ).min(1.0);
                    Binomial::new(s as u64, r).unwrap().sample(rng) as u32
                } else {
                    0
                };

                if di == 0 && dj == 0 && dk == 0 {
                    if diffused_count < *count {
                        let delta = *count - diffused_count;
                        if cell == BACKGROUND_CELL {
                            params.unassigned_counts[k as usize].sub(gene, delta);
                        } else {
                            let counts_c = params.counts.row(cell as usize);
                            counts_c
                                .write()
                                .sub(CountMatRowKey::new(gene as u32, k as u32), delta);
                        }
                        *count = diffused_count;
                    }
                } else if diffused_count > 0 {
                    let neighbor = voxel.offset_coords(di, dj, dk);
                    quad.counts_deltas.push((
                        VoxelCountKey {
                            voxel: neighbor,
                            gene: gene as u32,
                            offset: VoxelOffset::new(di0 + di, dj0 + dj, dk0 + dk),
                        },
                        diffused_count,
                    ));
                }

                s -= diffused_count;
                ρ -= p;

                if s == 0 {
                    break;
                }
            }
            assert!(s == 0);
        }
        multinomial_sampling_elapsed += t0.elapsed();
    }

    info!(
        "transcript repo timings: {:?}",
        (compute_probs_elapsed, multinomial_sampling_elapsed)
    );

    // Clear out any zeros
    quad.counts.retain(|_key, count| *count > 0);
}
