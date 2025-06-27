use crate::sampler::transcripts::{BACKGROUND_CELL, CellIndex};
use crate::sampler::voxelcheckerboard::UndirectedVoxelPair;

use super::math::halfnormal_logpdf;
use super::voxelcheckerboard::{Voxel, VoxelCheckerboard, VoxelCountKey, VoxelQuad, VoxelState};
use super::{CountMatRowKey, ModelParams, ModelPriors};
use log::trace;
use rand::{Rng, rng};
use rayon::prelude::*;
use std::f32;
use std::time::Instant;

// fn inv_isoperimetric_quotient(surface_area: u32, volume: u32) -> f32 {
//     (surface_area as f32).powi(3) / (36.0 * f32::consts::PI * (volume as f32).powi(2))
// }

fn inv_isoperimetric_quotient(surface_area: f32, volume: u32) -> f32 {
    surface_area.powi(2) / (4.0 * f32::consts::PI * (volume as f32).powi(2))
}

fn count_matching_neighbors(
    neighbor_cells: &[Option<CellIndex>; 10],
    current_cell: CellIndex,
    proposed_cell: CellIndex,
) -> (u32, u32, u32) {
    let mut current_cell_neighbors = 0;
    let mut proposed_cell_neighbors = 0;
    let mut other_neighbors = 0;
    for neighbor_cell in neighbor_cells {
        let neighbor_cell = neighbor_cell.unwrap_or(BACKGROUND_CELL);
        if neighbor_cell == current_cell {
            current_cell_neighbors += 1;
        } else if neighbor_cell == proposed_cell {
            proposed_cell_neighbors += 1;
        } else {
            other_neighbors += 1;
        }
    }
    (
        current_cell_neighbors,
        proposed_cell_neighbors,
        other_neighbors,
    )
}

#[derive(Debug, Clone, Copy)]
struct Proposal {
    voxel: Voxel,
    current_state: Option<VoxelState>,
    proposed_cell: CellIndex,
    log_proposal_imbalance: f32,
    neighbor_cells: [Option<CellIndex>; 6],
    current_cell_neighbors: u32,
    proposed_cell_neighbors: u32,
    other_cell_neighbors: u32,
}

pub struct VoxelSampler {
    t: usize,  // iteration number
    kmin: i32, // minimum voxel z-layer
    kmax: i32, // maximum voxel z-layer
    bubble_formation_prob: f32,
}

impl VoxelSampler {
    pub fn new(kmin: i32, kmax: i32, bubble_formation_prob: f32) -> VoxelSampler {
        VoxelSampler {
            t: 0,
            kmin,
            kmax,
            bubble_formation_prob,
        }
    }

    pub fn sample(
        &mut self,
        voxels: &mut VoxelCheckerboard,
        priors: &ModelPriors,
        params: &ModelParams,
        temperature: f32,
    ) {
        let t0 = Instant::now();

        let voxelsize_z = voxels.voxelsize_z;

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
                let proposal = self.generate_proposal(quad);
                if proposal.is_none() {
                    // TODO: We may be here because we randomly generated an oob proposal.
                    // In that case we should just regenerate, but we have to be careful we
                    // don't get stuck in an infinite loop.
                    return;
                }
                let proposal = proposal.unwrap();

                let mut logu = self.evaluate_proposal(quad, priors, params, voxelsize_z, proposal);
                if temperature < 1.0 {
                    logu /= temperature;
                } else {
                    logu += proposal.log_proposal_imbalance
                }
                let s = rng().random::<f32>().ln();

                if s < logu {
                    self.accept_proposal(voxels, quad, params, proposal);
                }
            });

        self.t += 1;

        trace!("sample voxels: {:?}", t0.elapsed());
    }

    fn generate_proposal(&self, quad: &VoxelQuad) -> Option<Proposal> {
        let quad_states = quad.states.read().unwrap();
        let mut connectivity = quad.connectivity.write().unwrap();
        let mut rng = rng();
        let edge = *quad_states.mismatch_edges.choose(&mut rng)?;

        // choose direction of undirected edge uniformly at random
        let (source, target) = if rng.random::<f32>() < 0.5 {
            (edge.a, edge.b)
        } else {
            (edge.b, edge.a)
        };

        if !quad.voxel_in_bounds(target) {
            return None;
        }

        let mut proposed_cell = quad_states.get_voxel_cell(source);
        let current_state = quad_states.get_voxel_state(target).copied();
        let current_cell = current_state
            .map(|state| state.cell)
            .unwrap_or(BACKGROUND_CELL);
        assert!(proposed_cell != current_cell);

        if proposed_cell != BACKGROUND_CELL && rng.random::<f32>() < self.bubble_formation_prob {
            proposed_cell = BACKGROUND_CELL;
        }

        // Connectivity checking
        let moore_neigbor_is_current: [bool; 26] = target.moore_neighborhood().map(|neighbor| {
            quad_states
                .get_voxel_state(neighbor)
                .is_some_and(|state| state.cell == current_cell)
        });

        let moore_neigbor_is_proposed: [bool; 26] = target.moore_neighborhood().map(|neighbor| {
            quad_states
                .get_voxel_state(neighbor)
                .is_some_and(|state| state.cell == proposed_cell)
        });

        let is_articulation_current = connectivity.is_articulation(&moore_neigbor_is_current);
        let is_articulation_proposed = connectivity.is_articulation(&moore_neigbor_is_proposed);

        if is_articulation_current || is_articulation_proposed {
            return None;
        }

        let target_neighbor_cells = target.von_neumann_neighborhood().map(|voxel| {
            let k = voxel.k();
            if k < self.kmin || k > self.kmax {
                None
            } else {
                Some(quad_states.get_voxel_cell(voxel))
            }
        });

        // multiply by 2 because edges are undirected and we only store one direction
        let num_mismatching_edges = 2 * quad_states.mismatch_edges.len();

        let num_current_cell_neighbors = target_neighbor_cells
            .iter()
            .filter(|cell| cell.is_some_and(|cell| cell == current_cell))
            .count();

        let num_proposed_cell_neighbors = target_neighbor_cells
            .iter()
            .filter(|cell| cell.is_some_and(|cell| cell == proposed_cell))
            .count();

        let mut proposal_prob = (1.0 - self.bubble_formation_prob as f64)
            * (num_proposed_cell_neighbors as f64 / num_mismatching_edges as f64);

        if proposed_cell == BACKGROUND_CELL {
            let num_mismatching_neighbors = target_neighbor_cells
                .iter()
                .filter(|cell| cell.is_some_and(|cell| cell != current_cell))
                .count();
            proposal_prob += (self.bubble_formation_prob as f64)
                * (num_mismatching_neighbors as f64 / num_mismatching_edges as f64);
        }

        let post_accept_num_mismatching_edges = num_mismatching_edges
            + 2 * num_current_cell_neighbors
            - 2 * num_proposed_cell_neighbors;

        let mut reverse_proposal_prob = (1.0 - self.bubble_formation_prob as f64)
            * (num_current_cell_neighbors as f64 / post_accept_num_mismatching_edges as f64);

        if current_cell == BACKGROUND_CELL {
            let num_mismatching_neighbors = target_neighbor_cells
                .iter()
                .filter(|cell| cell.is_some_and(|cell| cell != proposed_cell))
                .count();
            reverse_proposal_prob += (self.bubble_formation_prob as f64)
                * (num_mismatching_neighbors as f64 / post_accept_num_mismatching_edges as f64);
        }

        let log_proposal_imbalance = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;

        let (current_cell_neighbors, proposed_cell_neighbors, other_cell_neighbors) =
            count_matching_neighbors(
                &target.moorish_neighborhood().map(|voxel| {
                    let k = voxel.k();
                    if k < self.kmin || k > self.kmax {
                        None
                    } else {
                        Some(quad_states.get_voxel_cell(voxel))
                    }
                }),
                current_cell,
                proposed_cell,
            );

        Some(Proposal {
            voxel: target,
            current_state,
            proposed_cell,
            log_proposal_imbalance,
            neighbor_cells: target_neighbor_cells,
            current_cell_neighbors,
            proposed_cell_neighbors,
            other_cell_neighbors,
        })
    }

    fn evaluate_proposal(
        &self,
        quad: &VoxelQuad,
        priors: &ModelPriors,
        params: &ModelParams,
        voxelsize_z: f32,
        proposal: Proposal,
    ) -> f32 {
        let mut δ = 0.0; // Metropolis-Hastings ratio
        let quad_counts = quad.counts.read().unwrap();

        let proposed_cell = proposal.proposed_cell;
        let (current_cell, prior_cell, log_prior_prob, log_1m_prior_prob) =
            if let Some(current_state) = proposal.current_state {
                (
                    current_state.cell,
                    current_state.prior_cell,
                    current_state.log_prior.to_f32(),
                    current_state.log_1m_prior.to_f32(),
                )
            } else {
                (BACKGROUND_CELL, BACKGROUND_CELL, f32::NAN, f32::NAN)
            };

        // voxel prior penalties
        // prior_prob = 0, in cases where our prior in neutral on what the voxel
        // should be assigned to
        if log_prior_prob.is_finite() {
            if current_cell == prior_cell {
                δ -= log_prior_prob;
            } else {
                δ -= log_1m_prior_prob;
            }

            if proposed_cell == prior_cell {
                δ += log_prior_prob;
            } else {
                δ += log_1m_prior_prob;
            }
        }

        for (
            &VoxelCountKey {
                voxel,
                gene,
                offset,
            },
            &count,
        ) in quad_counts.voxel_counts(proposal.voxel)
        {
            let k = voxel.k() - offset.dk();
            let λ_bg_k = params.λ_bg.column(k as usize);

            if count == 0 {
                continue;
            }

            let λ_current_g = params.λ(current_cell as usize, gene as usize);
            let λ_proposed_g = params.λ(proposed_cell as usize, gene as usize);

            let λ_bg = λ_bg_k[gene as usize];
            δ -= (count as f32) * (λ_current_g + λ_bg).ln();
            δ += (count as f32) * (λ_proposed_g + λ_bg).ln();
        }

        if current_cell != BACKGROUND_CELL {
            let z = params.z[current_cell as usize];
            let current_volume = params.cell_voxel_count.get(current_cell as usize);
            let proposed_volume = current_volume - 1;
            let current_volume_μm = current_volume as f32 * params.voxel_volume;
            let proposed_volume_μm = proposed_volume as f32 * params.voxel_volume;
            let log_current_volume_μm = current_volume_μm.ln();
            let log_proposed_volume_μm = proposed_volume_μm.ln();

            // poisson point process normalization
            let scale = params.cell_scale[current_cell as usize];
            let volume_delta = proposed_volume_μm - current_volume_μm;
            δ -= scale * volume_delta * params.φ.row(current_cell as usize).dot(&params.θksum);

            // Simplification of log(N(proposed_volume, μ, σ)) - log(N(current_volume, μ, σ))
            let μ_vol_z = params.μ_volume[z as usize];
            let σ_vol_z = params.σ_volume[z as usize];
            δ += ((log_current_volume_μm - μ_vol_z).powi(2)
                - (log_proposed_volume_μm - μ_vol_z).powi(2))
                / (2.0 * σ_vol_z.powi(2))
                + log_current_volume_μm
                - log_proposed_volume_μm;

            let current_surface_area = params.cell_surface_area.get(current_cell as usize);

            // We scale by voxelsize_z here, because that's always fractional in [0,1], so the effect is
            // averaging 2d perimeters across voxel layers.
            δ -= halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(
                    voxelsize_z * current_surface_area as f32,
                    current_volume,
                ),
            );

            let other_cell_neighbors =
                proposal.proposed_cell_neighbors + proposal.other_cell_neighbors;
            let proposed_surface_area =
                current_surface_area + proposal.current_cell_neighbors - other_cell_neighbors;

            δ += halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(
                    voxelsize_z * proposed_surface_area as f32,
                    proposed_volume,
                ),
            );
        }

        if proposed_cell != BACKGROUND_CELL {
            let z = params.z[proposed_cell as usize];
            let current_volume = params.cell_voxel_count.get(proposed_cell as usize);
            let proposed_volume = current_volume + 1;
            let current_volume_μm = current_volume as f32 * params.voxel_volume;
            let proposed_volume_μm = proposed_volume as f32 * params.voxel_volume;
            let log_current_volume_μm = current_volume_μm.ln();
            let log_proposed_volume_μm = proposed_volume_μm.ln();

            // poisson point process normalization
            let scale = params.cell_scale[proposed_cell as usize];
            let volume_delta = proposed_volume_μm - current_volume_μm;
            δ -= scale * volume_delta * params.φ.row(proposed_cell as usize).dot(&params.θksum);

            // Simplification of log(N(proposed_volume, μ, σ)) - log(N(current_volume, μ, σ))
            let μ_vol_z = params.μ_volume[z as usize];
            let σ_vol_z = params.σ_volume[z as usize];
            δ += ((log_current_volume_μm - μ_vol_z).powi(2)
                - (log_proposed_volume_μm - μ_vol_z).powi(2))
                / (2.0 * σ_vol_z.powi(2))
                + log_current_volume_μm
                - log_proposed_volume_μm;

            let current_surface_area = params.cell_surface_area.get(proposed_cell as usize);
            δ -= halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(
                    voxelsize_z * current_surface_area as f32,
                    current_volume,
                ),
            );

            let other_cell_neighbors =
                proposal.current_cell_neighbors + proposal.other_cell_neighbors;
            let proposed_surface_area =
                current_surface_area - proposal.proposed_cell_neighbors + other_cell_neighbors;

            δ += halfnormal_logpdf(
                priors.σ_iiq,
                inv_isoperimetric_quotient(
                    voxelsize_z * proposed_surface_area as f32,
                    proposed_volume,
                ),
            );
        }

        δ
    }

    fn accept_proposal(
        &self,
        voxels: &VoxelCheckerboard,
        quad: &VoxelQuad,
        params: &ModelParams,
        proposal: Proposal,
    ) {
        let mut quad_states = quad.states.write().unwrap();
        let quad_counts = quad.counts.read().unwrap();
        let voxel = proposal.voxel;
        let proposed_cell = proposal.proposed_cell;
        let current_cell = proposal
            .current_state
            .map(|state| state.cell)
            .unwrap_or(BACKGROUND_CELL);

        quad_states.set_voxel_cell(voxel, proposed_cell);
        for (neighbor, neighbor_cell) in proposal
            .voxel
            .von_neumann_neighborhood()
            .iter()
            .cloned()
            .zip(proposal.neighbor_cells)
        {
            if let Some(neighbor_cell) = neighbor_cell {
                let edge = UndirectedVoxelPair::new(voxel, neighbor);
                if neighbor_cell == proposed_cell {
                    quad_states.mismatch_edges.remove(edge);
                } else if neighbor_cell == current_cell {
                    quad_states.mismatch_edges.insert(edge);
                };
            }
        }

        // Update neighboring quads to mirror voxel states on the edge
        let (u, v) = (quad.u, quad.v);
        let [i, j, _k] = proposal.voxel.coords();

        let (min_i, max_i, min_j, max_j) = quad.bounds();
        assert!((min_i..max_i + 1).contains(&i) && (min_j..max_j + 1).contains(&j));

        voxels.for_each_quad_neighbor_states(u, v, |neighbor_quad, neighbor_quad_states| {
            let (min_i, max_i, min_j, max_j) = neighbor_quad.bounds();
            if (min_i - 1..max_i + 2).contains(&i) && (min_j - 1..max_j + 2).contains(&j) {
                neighbor_quad_states.update_voxel_cell(
                    neighbor_quad,
                    voxel,
                    current_cell,
                    proposed_cell,
                );
            }
        });

        // Updating count matrices
        if current_cell != BACKGROUND_CELL {
            params
                .cell_voxel_count
                .modify(current_cell as usize, |volume| *volume -= 1);

            let other_cell_neighbors =
                proposal.proposed_cell_neighbors + proposal.other_cell_neighbors;
            params
                .cell_surface_area
                .modify(current_cell as usize, |surface_area| {
                    *surface_area += proposal.current_cell_neighbors;
                    *surface_area -= other_cell_neighbors;
                });

            let counts_row = params.counts.row(current_cell as usize);
            let mut counts_row_write = counts_row.write();
            for (key, &count) in quad_counts.voxel_counts(voxel) {
                let k_origin = key.voxel.k() - key.offset.dk();
                counts_row_write.sub(CountMatRowKey::new(key.gene, k_origin as u32), count);
            }
        } else {
            for (key, &count) in quad_counts.voxel_counts(voxel) {
                let k_origin = key.voxel.k() - key.offset.dk();
                let background_counts_k = &params.unassigned_counts[k_origin as usize];
                background_counts_k.sub(key.gene as usize, count);
            }
        }

        if proposed_cell != BACKGROUND_CELL {
            params
                .cell_voxel_count
                .modify(proposed_cell as usize, |volume| *volume += 1);

            let other_cell_neighbors =
                proposal.current_cell_neighbors + proposal.other_cell_neighbors;
            params
                .cell_surface_area
                .modify(proposed_cell as usize, |surface_area| {
                    *surface_area += other_cell_neighbors;
                    *surface_area -= proposal.proposed_cell_neighbors;
                });

            let counts_row = params.counts.row(proposed_cell as usize);
            let mut counts_row_write = counts_row.write();
            for (key, &count) in quad_counts.voxel_counts(voxel) {
                let k_origin = key.voxel.k() - key.offset.dk();
                counts_row_write.add(CountMatRowKey::new(key.gene, k_origin as u32), count);
            }
        } else {
            for (key, &count) in quad_counts.voxel_counts(voxel) {
                let k_origin = key.voxel.k() - key.offset.dk();
                let background_counts_k = &params.unassigned_counts[k_origin as usize];
                background_counts_k.add(key.gene as usize, count);
            }
        }
    }
}

// Does actual processing of VoxelQuads need to be a member function of VoxelSample?
// Might as well, just in case we need something.
