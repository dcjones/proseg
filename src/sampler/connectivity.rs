use crate::sampler::voxelcheckerboard::MOORE_OFFSETS;

use lazy_static::lazy_static;
use ndarray::Array2;
use ndarray::linalg::general_mat_mul;
use std::collections::HashMap;

lazy_static! {
    static ref MOORE_NEIGHBORHOOD_ADJACENCY: Array2<f32> = {
        let mut rev_moore_offsets = HashMap::new();
        for (u, (udi, udj, udk)) in MOORE_OFFSETS.iter().enumerate() {
            rev_moore_offsets.insert((*udi, *udj, *udk), u);
        }

        let mut adj = Array2::zeros((26, 26));
        for (u, (udi, udj, udk)) in MOORE_OFFSETS.iter().enumerate() {
            for (vdi, vdj, vdk) in MOORE_OFFSETS.iter() {
                // By doing these lookups we are excluded out of bound edges and
                // edges involving (0,0).
                if let Some(v) = rev_moore_offsets.get(&(udi + vdi, udj + vdj, udk + vdk)) {
                    adj[[u, *v]] = 1.0;
                }
            }

            // We do want self edges so raising to powers computes connectivity
            adj[[u, u]] = 1.0;
        }

        adj
    };
}

pub struct MooreConnectivityChecker {
    cell_mask: Array2<f32>,
    cell_adj: Array2<f32>,
    cell_adj_k: Array2<f32>,
    cell_adj_kp1: Array2<f32>,
}

impl MooreConnectivityChecker {
    pub fn new() -> Self {
        MooreConnectivityChecker {
            cell_mask: Array2::zeros((26, 26)),
            cell_adj: MOORE_NEIGHBORHOOD_ADJACENCY.clone(),
            cell_adj_k: MOORE_NEIGHBORHOOD_ADJACENCY.clone(),
            cell_adj_kp1: MOORE_NEIGHBORHOOD_ADJACENCY.clone(),
        }
    }

    pub fn is_articulation(&mut self, cell_mask: &[bool; 26]) -> bool {
        // bulid f3 cell mask matrix
        self.cell_mask.fill(0.0);
        self.cell_mask
            .diag_mut()
            .iter_mut()
            .zip(cell_mask)
            .for_each(|(mask, &is_cell)| {
                *mask = (is_cell as u8) as f32;
            });

        // mask the moore neighboorhood adjacency matrix
        // to only consider edges between voxels of the cell being considered
        general_mat_mul(
            1.0,
            &self.cell_mask,
            &MOORE_NEIGHBORHOOD_ADJACENCY,
            0.0,
            &mut self.cell_adj,
        );

        // The longest possibly connecting path we have to consider is k=4
        // so we compute connectivity info for that many steps.
        self.cell_adj_k.fill(0.0);
        self.cell_adj_k.diag_mut().fill(1.0);
        for _ in 0..4 {
            general_mat_mul(
                1.0,
                &self.cell_adj_k,
                &self.cell_adj,
                0.0,
                &mut self.cell_adj_kp1,
            );
            self.cell_adj_k.assign(&self.cell_adj_kp1);
        }

        // it's sufficient now to choose an arbitrary neighbor and check
        // that it's connected to every other neighbor
        for (i, &mask_i) in cell_mask.iter().enumerate() {
            if !mask_i {
                continue;
            }

            for (j, &mask_j) in cell_mask.iter().enumerate() {
                // j in unreachable from i without (0, 0), so it is an articulation point
                if mask_j && self.cell_adj_k[[i, j]] == 0.0 {
                    return true;
                }
            }
            break;
        }

        false
    }
}
