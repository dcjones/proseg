use crate::sampler::transcripts::{CellIndex, BACKGROUND_CELL};

use super::voxelcheckerboard::{Voxel, VoxelCheckerboard, VoxelQuad};
use rand::{rng, Rng};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct Proposal {
    voxel: Voxel,
    cell: CellIndex,
    log_imbalance: f32,
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

    fn sample(&mut self, voxels: &mut VoxelCheckerboard) {
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

                let logu = self.evaluate_proposal(&*quad, proposal);
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

        let mut source_cell = quad.get_cell(source);
        let target_cell = quad.get_cell(target);
        assert!(source_cell != target_cell);

        if source_cell != BACKGROUND_CELL && rng.random::<f32>() < self.bubble_formation_prob {
            source_cell = BACKGROUND_CELL;
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
            .filter(|cell| cell.is_some_and(|cell| cell == target_cell))
            .count();

        let num_source_cell_neighbors = target_neighbor_cells
            .iter()
            .filter(|cell| cell.is_some_and(|cell| cell == source_cell))
            .count();

        let mut proposal_prob = (1.0 - self.bubble_formation_prob as f64)
            * (num_source_cell_neighbors as f64 / num_mismatching_edges as f64);

        if source_cell == BACKGROUND_CELL {
            let num_mismatching_neighbors = target_neighbor_cells
                .iter()
                .filter(|cell| cell.is_some_and(|cell| cell != target_cell))
                .count();
            proposal_prob += (self.bubble_formation_prob as f64)
                * (num_mismatching_neighbors as f64 / num_mismatching_edges as f64);
        }

        let post_accept_num_mismatching_edges =
            num_mismatching_edges + 2 * num_target_cell_neighbors - 2 * num_source_cell_neighbors;

        let mut reverse_proposal_prob = (1.0 - self.bubble_formation_prob as f64)
            * (num_target_cell_neighbors as f64 / post_accept_num_mismatching_edges as f64);

        if target_cell == BACKGROUND_CELL {
            let num_mismatching_neighbors = target_neighbor_cells
                .iter()
                .filter(|cell| cell.is_some_and(|cell| cell != source_cell))
                .count();
            reverse_proposal_prob += (self.bubble_formation_prob as f64)
                * (num_mismatching_neighbors as f64 / post_accept_num_mismatching_edges as f64);
        }

        let log_imbalance = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;

        Some(Proposal {
            voxel: target,
            cell: source_cell,
            log_imbalance,
        })
    }

    fn evaluate_proposal(&self, quad: &VoxelQuad, proposal: Proposal) -> f32 {
        let mut δ = 0.0; // Metropolis-Hastings ratio

        // TODO:
        //   - need to pass Params to this
        //   - dot products to compute rate vectors for source and target cells
        //   - computing likelhood under background
        //   - prior segmentation
        //   - volume distributions
        //   - perimiter ratio penalities

        δ
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
