use crate::sampler::transcripts::{CellIndex, BACKGROUND_CELL};

use super::math::{halfnormal_logpdf, lognormal_logpdf, normal_logpdf};
use super::voxelcheckerboard::{Voxel, VoxelCheckerboard, VoxelCountKey, VoxelQuad, VoxelState};
use super::{ModelParams, ModelPriors};
use rand::{rng, Rng};
use rayon::prelude::*;
use std::f32;
use std::thread::current;

pub fn inv_isoperimetric_quotient(surface_area: f32, volume: f32) -> f32 {
    (surface_area).powi(3) / (36.0 * f32::consts::PI * volume.powi(2))
}

#[derive(Debug, Clone, Copy)]
struct Proposal {
    voxel: Voxel,
    current_state: Option<VoxelState>,
    proposed_cell: CellIndex,
    log_proposal_imbalance: f32,
    current_cell_surface_area_delta: i8,
    proposed_cell_surface_area_delta: i8,
}

pub struct VoxelSampler {
    t: usize,  // iteration number
    kmin: i32, // minimum voxel z-layer
    kmax: i32, // maximum voxel z-layer
    bubble_formation_prob: f32,
}

impl VoxelSampler {
    fn new(kmin: i32, kmax: i32, bubble_formation_prob: f32) -> VoxelSampler {
        VoxelSampler {
            t: 0,
            kmin,
            kmax,
            bubble_formation_prob,
        }
    }

    fn sample(
        &mut self,
        voxels: &mut VoxelCheckerboard,
        priors: &ModelPriors,
        params: &ModelParams,
    ) {
        let parity = self.t % 4;

        voxels
            .quads
            .par_iter()
            .filter_map(|(&(i, j), quad)| {
                let quad_parity = 2 * (i % 2) + (j % 2);
                if quad_parity as usize == parity {
                    Some(quad)
                } else {
                    None
                }
            })
            .for_each(|quad| {
                let mut quad = quad.write().unwrap();

                let proposal = self.generate_proposal(&*quad);
                if proposal.is_none() {
                    return;
                }
                let proposal = proposal.unwrap();

                let logu = self.evaluate_proposal(&*quad, priors, params, proposal);
                if rng().random::<f32>().ln() < logu {
                    self.accept_proposal(&mut *quad, proposal);
                    // TODO: We are going to have to do some bookkeeping to
                    // facilitate efficient cross-quad consistency updates,
                    // and to update transcript count matrices.
                }
            });

        // TODO:
        // Whan post-processing will we need:
        //  - Update cell-by-gene sparse count matrix (we could put this in a sharded hash table)
        //  - Update cell perimiter (could also be in a sharded hash table)
        //  - Recompute redundant quad edge voxel states. (I guess we compute updates for each quad then pull them as needed)
        //    NOTE: This may necessitate updating the edge sets on these neighboring quads!
        //
        // We need to optimize the fuck out of this post-processing or be the victim of Amdahls law.

        self.t += 1;
    }

    fn generate_proposal(&self, quad: &VoxelQuad) -> Option<Proposal> {
        let mut rng = rng();
        let edge = *quad.mismatch_edges.choose(&mut rng)?;

        // choose direction of undirected edge uniformly at random
        let (source, target) = if rng.random::<f32>() < 0.5 {
            (edge.a, edge.b)
        } else {
            (edge.b, edge.a)
        };

        let mut proposed_cell = quad.get_cell(source);
        let current_state = quad.get(target).map(|&state| state);
        let current_cell = current_state
            .map(|state| state.cell)
            .unwrap_or(BACKGROUND_CELL);
        assert!(proposed_cell != current_cell);

        if proposed_cell != BACKGROUND_CELL && rng.random::<f32>() < self.bubble_formation_prob {
            proposed_cell = BACKGROUND_CELL;
        }

        // TODO: Connectivity checking
        // I'm skipping this for now, because it may not be necessary if our
        // perimiter penalization is rigorous enough. If we decide we do need
        // this, I think we just rewrite the existing code, but use u128s to store
        // node sets (visited and cell), so there's no allocation or hashing.
        // We also have to expand how much redudant information is mirrored
        // across quads to account for the 2x moore neighborhood used for this.

        let target_neighbor_cells = target.von_neumann_neighborhood().map(|voxel| {
            let k = voxel.k();
            if k < self.kmin || k > self.kmax {
                None
            } else {
                Some(quad.get_cell(voxel))
            }
        });

        // multiply by 2 because edges are undirected and we only store one direction
        let num_mismatching_edges = 2 * quad.mismatch_edges.len();

        let num_target_cell_neighbors = target_neighbor_cells
            .iter()
            .filter(|cell| cell.is_some_and(|cell| cell == current_cell))
            .count();

        let num_source_cell_neighbors = target_neighbor_cells
            .iter()
            .filter(|cell| cell.is_some_and(|cell| cell == proposed_cell))
            .count();

        let mut proposal_prob = (1.0 - self.bubble_formation_prob as f64)
            * (num_source_cell_neighbors as f64 / num_mismatching_edges as f64);

        if proposed_cell == BACKGROUND_CELL {
            let num_mismatching_neighbors = target_neighbor_cells
                .iter()
                .filter(|cell| cell.is_some_and(|cell| cell != current_cell))
                .count();
            proposal_prob += (self.bubble_formation_prob as f64)
                * (num_mismatching_neighbors as f64 / num_mismatching_edges as f64);
        }

        let post_accept_num_mismatching_edges =
            num_mismatching_edges + 2 * num_target_cell_neighbors - 2 * num_source_cell_neighbors;

        let mut reverse_proposal_prob = (1.0 - self.bubble_formation_prob as f64)
            * (num_target_cell_neighbors as f64 / post_accept_num_mismatching_edges as f64);

        if current_cell == BACKGROUND_CELL {
            let num_mismatching_neighbors = target_neighbor_cells
                .iter()
                .filter(|cell| cell.is_some_and(|cell| cell != proposed_cell))
                .count();
            reverse_proposal_prob += (self.bubble_formation_prob as f64)
                * (num_mismatching_neighbors as f64 / post_accept_num_mismatching_edges as f64);
        }

        let log_proposal_imbalance = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;

        // TODO: Figure out surface area deltas
        let mut current_cell_surface_area_delta = 0;
        let mut proposed_cell_surface_area_delta = 0;
        for neighbor_cell in target_neighbor_cells {
            // skip oob neighbors
            if neighbor_cell.is_none() {
                continue;
            }
            let neighbor_cell = neighbor_cell.unwrap();

            if neighbor_cell == proposed_cell {
                proposed_cell_surface_area_delta -= 1;
            }
            if neighbor_cell != proposed_cell {
                proposed_cell_surface_area_delta += 1;
            }
            if neighbor_cell == current_cell {
                current_cell_surface_area_delta += 1;
            }
            if neighbor_cell != current_cell {
                current_cell_surface_area_delta -= 1;
            }
        }

        Some(Proposal {
            voxel: target,
            current_state,
            proposed_cell,
            log_proposal_imbalance,
            current_cell_surface_area_delta,
            proposed_cell_surface_area_delta,
        })
    }

    fn evaluate_proposal(
        &self,
        quad: &VoxelQuad,
        priors: &ModelPriors,
        params: &ModelParams,
        proposal: Proposal,
    ) -> f32 {
        let mut δ = 0.0; // Metropolis-Hastings ratio

        let voxel = proposal.voxel;
        let proposed_cell = proposal.proposed_cell;
        let (current_cell, prior_cell, prior_prob) =
            if let Some(current_state) = proposal.current_state {
                (
                    current_state.cell,
                    current_state.prior_cell,
                    current_state.prior,
                )
            } else {
                (BACKGROUND_CELL, BACKGROUND_CELL, 0.0)
            };

        // voxel prior penalties
        // prior_prob = 0, in cases where our prior in neutral on what the voxel
        // should be assigned to
        if prior_prob != 0.0 {
            let prior_log_prob = prior_prob.ln();
            if current_cell == prior_cell {
                δ -= prior_log_prob;
            } else {
                δ += prior_log_prob;
            }

            if proposed_cell == prior_cell {
                δ += prior_log_prob;
            } else {
                δ -= prior_log_prob;
            }
        }

        let k = voxel.k();
        let λ_bg_k = params.λ_bg.column(k as usize);
        let φ_current = if current_cell == BACKGROUND_CELL {
            None
        } else {
            Some(params.φ.row(current_cell as usize))
        };
        let φ_proposed = if proposed_cell == BACKGROUND_CELL {
            None
        } else {
            Some(params.φ.row(proposed_cell as usize))
        };
        for (
            &VoxelCountKey {
                voxel: _,
                gene,
                offset: _,
            },
            &count,
        ) in quad.voxel_counts(proposal.voxel)
        {
            if count == 0 {
                continue;
            }

            let λ_current_g = φ_current
                .map(|φ_current| {
                    if (gene as usize) < params.nunfactored {
                        φ_current[gene as usize]
                    } else {
                        let θ_g = params.θ.row(gene as usize);
                        φ_current.dot(&θ_g)
                    }
                })
                .unwrap_or(0.0);

            let λ_proposed_g = φ_proposed
                .map(|φ_proposed| {
                    if (gene as usize) < params.nunfactored {
                        φ_proposed[gene as usize]
                    } else {
                        let θ_g = params.θ.row(gene as usize);
                        φ_proposed.dot(&θ_g)
                    }
                })
                .unwrap_or(0.0);

            let λ_bg = λ_bg_k[gene as usize];
            δ -= (count as f32) * (λ_current_g + λ_bg).ln();
            δ += (count as f32) * (λ_proposed_g + λ_bg).ln();
        }

        if current_cell != BACKGROUND_CELL {
            let z = params.z[current_cell as usize];
            let current_volume = params.cell_volume[current_cell as usize];
            let proposed_volume = current_volume - params.voxel_volume;

            // Simplification of log(N(proposed_volume, μ, σ)) - log(N(current_volume, μ, σ))
            let μ_vol_z = params.μ_volume[z as usize];
            let σ_vol_z = params.σ_volume[z as usize];
            δ += ((current_volume - μ_vol_z).powi(2) - (proposed_volume - μ_vol_z).powi(2))
                / σ_vol_z;

            let current_surface_area = params.cell_surface_area[current_cell as usize];
            δ -= halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(current_surface_area, current_volume),
            );

            let proposed_surface_area =
                current_surface_area + proposal.proposed_cell_surface_area_delta as f32;
            // TODO: Need to iterate over voxels neighbors again! I guess we should
            // compute this when generating the proposal.
            δ += halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(proposed_surface_area, current_volume),
            );
        }

        if proposed_cell != BACKGROUND_CELL {
            let z = params.z[proposed_cell as usize];
            let current_volume = params.cell_volume[proposed_cell as usize];
            let proposed_volume = current_volume + params.voxel_volume;

            // Simplification of log(N(proposed_volume, μ, σ)) - log(N(current_volume, μ, σ))
            let μ_vol_z = params.μ_volume[z as usize];
            let σ_vol_z = params.σ_volume[z as usize];
            δ += ((current_volume - μ_vol_z).powi(2) - (proposed_volume - μ_vol_z).powi(2))
                / σ_vol_z;

            let current_surface_area = params.cell_surface_area[proposed_cell as usize];
            δ -= halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(current_surface_area, current_volume),
            );

            let proposed_surface_area =
                current_surface_area + proposal.current_cell_surface_area_delta as f32;
            // TODO: Need to iterate over voxels neighbors again! I guess we should
            // compute this when generating the proposal.
            δ += halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(proposed_surface_area, current_volume),
            );
        }

        δ + proposal.log_proposal_imbalance
    }

    fn accept_proposal(&self, quad: &mut VoxelQuad, proposal: Proposal) {
        //   - On acceptance:
        //     - Update voxel state
        //     - Updaate edge sets
        todo!();
    }
}

// Does actual processing of VoxelQuads need to be a member function of VoxelSample?
// Might as well, just in case we need something.
