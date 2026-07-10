use log::trace;
use ndarray::s;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::time::Instant;

use crate::sampler::voxelcheckerboard::TranscriptFixedState;

use super::math::uniformly_imprecise_normal_prob;
use super::multinomial::Multinomial;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{QuadStatesView, Voxel, VoxelCheckerboard};
use super::{CountMatRowKey, ModelParams, ModelPriors};

use rand::rngs::ThreadRng;
use rand::{Rng, rng};

pub struct TranscriptRepo {
    proposal_near_xy_probs: Vec<f32>,
    proposal_far_xy_probs: Vec<f32>,
    proposal_z_probs: Vec<f32>,
    proposal_near_xy: Multinomial<f32>,
    proposal_far_xy: Multinomial<f32>,
    proposal_z: Multinomial<f32>,
}

// Turn a one-sided distance pmf (index = |distance| in voxels) into a
// symmetric two-sided pmf centered at zero, so a sampled index minus the
// center yields a signed offset.
fn two_sided_pmf(mut pmf: Vec<f32>) -> Vec<f32> {
    pmf.reverse();
    let n = pmf.len();
    pmf.resize(2 * n - 1, 0.0);
    for i in 1..n {
        pmf[(n - 1) + i] = pmf[(n - 1) - i];
    }
    pmf
}

impl TranscriptRepo {
    pub fn new(priors: &ModelPriors, voxelsize: f32, voxelsize_z: f32) -> Self {
        const EPS: f32 = 1e-5;

        // The proposal distribution *is* the diffusion prior. Proposing new
        // positions relative to each transcript's original voxel (an
        // independence sampler) means the prior terms cancel in the
        // Metropolis-Hastings ratio, leaving only the likelihood ratio.
        let proposal_near_xy_probs =
            two_sided_pmf(VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_near, EPS).pmf);
        let proposal_far_xy_probs =
            two_sided_pmf(VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_far, EPS).pmf);
        let proposal_z_probs =
            two_sided_pmf(VoxelDiffusionPrior::new(voxelsize_z, priors.σ_z_diffusion, EPS).pmf);

        let proposal_near_xy = Multinomial::from_probs(&proposal_near_xy_probs);
        let proposal_far_xy = Multinomial::from_probs(&proposal_far_xy_probs);
        let proposal_z = Multinomial::from_probs(&proposal_z_probs);

        TranscriptRepo {
            proposal_near_xy_probs,
            proposal_far_xy_probs,
            proposal_z_probs,
            proposal_near_xy,
            proposal_far_xy,
            proposal_z,
        }
    }

    pub fn set_voxel_size(&mut self, priors: &ModelPriors, voxelsize: f32, voxelsize_z: f32) {
        const EPS: f32 = 1e-5;

        self.proposal_near_xy_probs =
            two_sided_pmf(VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_near, EPS).pmf);
        self.proposal_far_xy_probs =
            two_sided_pmf(VoxelDiffusionPrior::new(voxelsize, priors.σ_xy_diffusion_far, EPS).pmf);
        self.proposal_z_probs =
            two_sided_pmf(VoxelDiffusionPrior::new(voxelsize_z, priors.σ_z_diffusion, EPS).pmf);

        self.proposal_near_xy = Multinomial::from_probs(&self.proposal_near_xy_probs);
        self.proposal_far_xy = Multinomial::from_probs(&self.proposal_far_xy_probs);
        self.proposal_z = Multinomial::from_probs(&self.proposal_z_probs);
    }

    pub fn sample(
        &self,
        voxels: &mut VoxelCheckerboard,
        priors: &ModelPriors,
        params: &mut ModelParams,
        temperature: f32,
        record_samples: bool,
    ) {
        let t0 = Instant::now();
        // All updates go through interior mutability: the position array is
        // AtomicU64 and the count matrices are row-locked / atomic, so we only
        // need a shared borrow and can iterate every transcript in parallel
        // (no quad partition, no cross-quad routing, no delta merge).
        let voxels: &VoxelCheckerboard = voxels;
        // States are fixed during repositioning, so take one lock-free view rather
        // than locking on every per-transcript cell lookup (two per transcript).
        let states = voxels.states_view();
        let ntranscripts = voxels.transcript_voxel.len();
        (0..ntranscripts)
            .into_par_iter()
            .for_each_init(rng, |rng, idx| {
                self.repo_transcript(
                    voxels,
                    &states,
                    rng,
                    priors,
                    params,
                    idx,
                    temperature,
                    record_samples,
                );
            });
        trace!("transcript repo: {:?}", t0.elapsed());
    }

    // Propose and (if accepted) apply a repositioning move for a single
    // transcript. Voxel→cell assignments are fixed during repositioning, so
    // every transcript's move is independent of the others: it reads its own
    // current voxel/cell, proposes from its original voxel, and on acceptance
    // updates its position and moves its count between cells.
    #[allow(clippy::too_many_arguments)]
    fn repo_transcript(
        &self,
        voxels: &VoxelCheckerboard,
        states: &QuadStatesView,
        rng: &mut ThreadRng,
        priors: &ModelPriors,
        params: &ModelParams,
        idx: usize,
        temperature: f32,
        record_samples: bool,
    ) {
        use std::sync::atomic::Ordering::Relaxed;

        let TranscriptFixedState {
            original_voxel,
            gene,
        } = voxels.transcript_fixed_state[idx];
        let gene = gene as usize;

        let current_voxel = Voxel::from_raw(voxels.transcript_voxel[idx].load(Relaxed));
        let cell = states.get_voxel_cell(current_voxel);

        // Independence proposal: draw a displacement from the diffusion prior
        // relative to the transcript's *original* voxel. The xy prior is a
        // mixture of two isotropic components, so pick the component once and
        // draw both axes from it (rather than sampling each axis from the
        // marginal mixture, which would decouple the components).
        let (proposal_xy, proposal_xy_probs) = if rng.random::<f32>() < priors.p_diffusion {
            (&self.proposal_far_xy, &self.proposal_far_xy_probs)
        } else {
            (&self.proposal_near_xy, &self.proposal_near_xy_probs)
        };
        let xy_center = ((proposal_xy_probs.len() - 1) / 2) as i32;
        let di = proposal_xy.sample1(rng) as i32 - xy_center;
        let dj = proposal_xy.sample1(rng) as i32 - xy_center;

        let z_center = ((self.proposal_z_probs.len() - 1) / 2) as i32;
        let dk = self.proposal_z.sample1(rng) as i32 - z_center;

        if original_voxel.k() + dk < 0 || original_voxel.k() + dk > voxels.kmax {
            return;
        }

        let neighbor = original_voxel.offset_coords(di, dj, dk);
        if neighbor.is_oob() {
            return;
        }

        // don't repo into a quad that doesn't exist
        let u = neighbor.i() as u32 / voxels.quadsize as u32;
        let v = neighbor.j() as u32 / voxels.quadsize as u32;
        if !voxels.quads_coords.contains(&(u, v)) {
            return;
        }

        let neighbor_cell = states.get_voxel_cell(neighbor);

        // If the move stays within the same cell (including
        // background→background), the Poisson rates are identical — same cell,
        // gene, and original voxel — so the acceptance ratio is exactly 1 and we
        // can accept without evaluating either λ dot product.
        let accept_prob = if cell == neighbor_cell {
            1.0
        } else {
            let density = voxels.get_voxel_density(original_voxel);
            let λ_bg = params.λ_bg[[gene, original_voxel.k() as usize, density]];
            let θ_g_factored = if gene < params.nunfactored {
                None
            } else {
                Some(params.θ.slice(s![gene, params.nunfactored..]))
            };

            let mut λ_current = λ_bg;
            if cell != BACKGROUND_CELL {
                λ_current += if gene < params.nunfactored {
                    params.φ[[cell as usize, gene]]
                } else {
                    params
                        .φ
                        .slice(s![cell as usize, params.nunfactored..])
                        .dot(&θ_g_factored.unwrap())
                };
            }

            let mut λ_proposed = λ_bg;
            if neighbor_cell != BACKGROUND_CELL {
                λ_proposed += if gene < params.nunfactored {
                    params.φ[[neighbor_cell as usize, gene]]
                } else {
                    params
                        .φ
                        .slice(s![neighbor_cell as usize, params.nunfactored..])
                        .dot(&θ_g_factored.unwrap())
                };
            }

            let ratio = λ_proposed / λ_current;
            // Annealing toward a greedy (hill-climbing) point estimate: tempering
            // the likelihood-ratio acceptance by 1/temperature sharpens it as
            // temperature → 0 (improving moves accepted, worsening moves rejected),
            // and recovers the exact Metropolis independence sampler at
            // temperature == 1. The prior terms already cancel (independence
            // proposal from the diffusion prior), so the ratio is purely the
            // likelihood ratio and tempering it is the correct annealed kernel.
            if temperature < 1.0 {
                ratio.powf(1.0 / temperature)
            } else {
                ratio
            }
        };

        if rng.random::<f32>() > accept_prob {
            return;
        }

        // Record the transition only when sampling posteriors. Acquiring the row
        // lock lazily here (rather than eagerly per transcript) avoids a lock
        // acquisition for every non-background transcript, which is pure overhead
        // during the point-estimate phase (record_samples == false).
        if record_samples && cell != BACKGROUND_CELL && neighbor_cell != BACKGROUND_CELL {
            params
                .transition_counts
                .row(cell as usize)
                .write()
                .add(neighbor_cell, 1);
        }

        // Move the transcript's count between cells. Same-cell moves (including
        // background→background) leave the count table unchanged — same cell,
        // gene, and original voxel → same density/layer key — so we skip them.
        if cell != neighbor_cell {
            let k_origin = original_voxel.k() as usize;
            let density = voxels.get_voxel_density(original_voxel);
            let key = CountMatRowKey::new(gene as u32, k_origin as u32, density as u8);

            if cell == BACKGROUND_CELL {
                params.unassigned_counts[density][k_origin].sub(gene, 1);
            } else {
                params.counts.row(cell as usize).write().sub(key, 1);
            }

            if neighbor_cell == BACKGROUND_CELL {
                params.unassigned_counts[density][k_origin].add(gene, 1);
            } else {
                params.counts.row(neighbor_cell as usize).write().add(key, 1);
            }
        }

        voxels.transcript_voxel[idx].store(neighbor.raw(), Relaxed);
    }
}

struct VoxelDiffusionPrior {
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

        VoxelDiffusionPrior { pmf }
    }
}
