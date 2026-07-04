use ahash::AHashSet as HashSet;
use log::trace;
use ndarray::s;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::ops::DerefMut;
use std::time::Instant;

use crate::sampler::voxelcheckerboard::{TranscriptFixedState, VoxelTranscript};

use super::math::uniformly_imprecise_normal_prob;
use super::multinomial::Multinomial;
use super::transcripts::BACKGROUND_CELL;
use super::voxelcheckerboard::{VoxelCheckerboard, VoxelQuad};
use super::{ModelParams, ModelPriors};

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
                    record_samples,
                );
            });
        trace!("transcript repo: compute deltas: {:?}", t0.elapsed());

        let t0 = Instant::now();
        voxels.merged_moved_transcripts(params);
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
        record_samples: bool,
    ) {
        let quad_states = quad.states.read().unwrap();
        let mut quad_transcripts = quad.transcripts.write().unwrap();
        let quad_transcripts_ref = quad_transcripts.deref_mut();

        assert!(quad_transcripts_ref.outgoing_transcripts.is_empty());

        for &VoxelTranscript {
            voxel,
            transcript_idx,
        } in quad_transcripts_ref.transcripts.iter()
        {
            let TranscriptFixedState {
                original_voxel,
                gene,
            } = voxels.transcript_fixed_state[transcript_idx as usize];
            let gene = gene as usize;

            let cell = quad_states
                .states
                .get(&voxel)
                .map(|state| state.cell)
                .unwrap_or(BACKGROUND_CELL);

            let transition_counts_row = if cell != BACKGROUND_CELL {
                Some(params.transition_counts.row(cell as usize))
            } else {
                None
            };

            let mut transition_counts_row_write =
                transition_counts_row.as_ref().map(|row| row.write());

            let density = voxels.get_voxel_density_hint(quad, original_voxel);

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

            // Independence proposal: draw a displacement from the diffusion
            // prior relative to the transcript's *original* voxel. The xy prior
            // is a mixture of two isotropic components, so pick the component
            // once and draw both axes from it (rather than sampling each axis
            // from the marginal mixture, which would decouple the components).
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

            if original_voxel.k() + dk < 0 || original_voxel.k() + dk > quad.kmax {
                continue;
            }

            let neighbor = original_voxel.offset_coords(di, dj, dk);
            if neighbor.is_oob() {
                continue;
            }

            // don't repo into a quad that doesn't exist
            let u = neighbor.i() as u32 / quadsize;
            let v = neighbor.j() as u32 / quadsize;
            if !quads_coords.contains(&(u, v)) {
                continue;
            }

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
                λ_proposed += if gene < params.nunfactored {
                    params.φ[[neighbor_cell as usize, gene]]
                } else {
                    params
                        .φ
                        .slice(s![neighbor_cell as usize, params.nunfactored..])
                        .dot(&θ_g_factored.unwrap())
                };
            }

            let accept_prob = λ_proposed / λ_current;

            if rng.random::<f32>() > accept_prob {
                continue;
            }
            if record_samples
                && neighbor_cell != BACKGROUND_CELL
                && let Some(transition_counts_row_write) = transition_counts_row_write.as_mut()
            {
                transition_counts_row_write.add(neighbor_cell, 1);
            }

            quad_transcripts_ref.outgoing_transcripts.push((
                voxel,
                VoxelTranscript {
                    voxel: neighbor,
                    transcript_idx,
                },
            ));
        }
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
