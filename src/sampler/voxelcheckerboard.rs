// Data structures for maintaining a set of voxels each with an associated
// sparse transcript vector.

use super::sampleset::SampleSet;
use super::transcripts::{CellIndex, TranscriptDataset, BACKGROUND_CELL};

use std::cmp::{Ordering, PartialOrd};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

// Store a voxel offset in compact form
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
struct VoxelOffset {
    // offsets (di, dj, dk) in each dimension packed into a single u32 with
    // 12, 12, and 8 bits, respectively.
    offset: u32,
}

// interpret the lower 12-bits of the u32 as an i12
// and convert to an i32
fn i12_to_i32(u: u32) -> i32 {
    // if u & 0x800 != 0 {
    //     u = 0xFFFFF000 | u
    // }
    // u as i32

    // exploiting signed integer shift behavior
    (((u & 0xFFF) as i32) << 20) >> 20
}

// interpret the lower 8-bits of the u32 as an i8
// and convert to an i32
fn i8_to_i32(u: u32) -> i32 {
    // if u & 0x80 != 0 {
    //     u = 0xFFFFFF00 | u
    // }
    // u as i32

    // exploiting signed integer shift behavior
    (((u & 0xFF) as i32) << 24) >> 24
}

impl VoxelOffset {
    fn new(di: i32, dj: i32, dk: i32) -> VoxelOffset {
        VoxelOffset {
            offset: ((di as u32 & 0xFFF) << 20) | ((dj as u32 & 0xFFF) << 8) | (dk as u32 & 0xFF),
        }
    }

    fn zero() -> VoxelOffset {
        VoxelOffset { offset: 0 }
    }

    fn di(&self) -> i32 {
        i12_to_i32((self.offset >> 20) & 0xFFF)
    }

    fn dj(&self) -> i32 {
        i12_to_i32((self.offset >> 8) & 0xFFF)
    }

    fn dk(&self) -> i32 {
        i8_to_i32(self.offset & 0xFF)
    }
}

// Index of a single voxel
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Voxel {
    // i, j, k coordinates packaed into a single integer
    // with 24, 24, and 16-bits, respectively.
    index: u64,
}

// Using a special value to represent any out of bound voxel.
const OOB_VOXEL: u64 = 0xffffffffffffffff;

// TODO: we could just u32 for coordinates.
impl Voxel {
    fn new(i: i32, j: i32, k: i32) -> Voxel {
        if i < 0
            || (i as u64) >= (1 << 24)
            || j < 0
            || (j as u64) >= (1 << 24)
            || k < 0
            || (k as u64) >= (1 << 16)
        {
            return Voxel { index: OOB_VOXEL };
        }

        Voxel {
            index: ((i as u64 & 0xFFFFFF) << 48)
                | ((j as u64 & 0xFFFFFF) << 24)
                | (k as u64 & 0xFFFF),
        }
    }

    fn oob() -> Voxel {
        Voxel { index: OOB_VOXEL }
    }

    fn is_oob(&self) -> bool {
        self.index == OOB_VOXEL
    }

    fn coords(&self) -> [i32; 3] {
        [self.i(), self.j(), self.k()]
    }

    fn u32_coords(&self) -> [u32; 3] {
        [self.i() as u32, self.j() as u32, self.k() as u32]
    }

    fn i(&self) -> i32 {
        ((self.index >> 48) & 0xFFFFFF) as i32
    }

    fn j(&self) -> i32 {
        ((self.index >> 24) & 0xFFFFFF) as i32
    }

    fn k(&self) -> i32 {
        (self.index & 0xFFFF) as i32
    }

    fn offset(&self, d: VoxelOffset) -> Voxel {
        self.offset_coords(d.di(), d.dj(), d.dk())
    }

    fn offset_coords(&self, di: i32, dj: i32, dk: i32) -> Voxel {
        let new_i = self.i() + di;
        let new_j = self.j() + dj;
        let new_k = self.k() + dk;

        if new_i < 0
            || new_i >= (1 << 24)
            || new_j < 0
            || new_j >= (1 << 24)
            || new_k < 0
            || new_k >= (1 << 16)
        {
            return Voxel { index: OOB_VOXEL };
        }

        Voxel::new(new_i, new_j, new_k)
    }

    pub fn von_neumann_neighborhood(&self) -> [Voxel; 6] {
        let [i, j, k] = self.coords();
        [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    }

    pub fn moore_neighborhood(&self) -> [Voxel; 26] {
        let [i, j, k] = self.coords();
        [
            // top layer
            (-1, 0, -1),
            (0, 0, -1),
            (1, 0, -1),
            (-1, 1, -1),
            (0, 1, -1),
            (1, 1, -1),
            (-1, -1, -1),
            (0, -1, -1),
            (1, -1, -1),
            // middle layer
            (-1, 0, 0),
            (1, 0, 0),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            // bottom layer
            (-1, 0, 1),
            (0, 0, 1),
            (1, 0, 1),
            (-1, 1, 1),
            (0, 1, 1),
            (1, 1, 1),
            (-1, -1, 1),
            (0, -1, 1),
            (1, -1, 1),
        ]
        .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    }
}

// Z-order curve comparison function. Following from:
// https://en.wikipedia.org/wiki/Z-order_curve#Efficiently_building_quadtrees_and_octrees
impl Ord for Voxel {
    fn cmp(&self, other: &Voxel) -> Ordering {
        // self.index.cmp(&other.index)
        fn less_msb(a: u32, b: u32) -> bool {
            a < b && a < (a ^ b)
        }

        if self.index == other.index {
            return Ordering::Equal;
        }

        // xor then extract coords rather than vice versa to save a few ops
        let xor = self.index ^ other.index;
        let xi = ((xor >> 48) & 0xFFFFFF) as u32;
        let xj = ((xor >> 24) & 0xFFFFFF) as u32;
        let xk = (xor & 0xFFFF) as u32;

        // just doing a bunch of branches here an trusting the compiler
        // to generate cmovs
        let islt = if less_msb(xi, xj) {
            if less_msb(xj, xk) {
                self.k() < other.k()
            } else {
                self.j() < other.j()
            }
        } else {
            if less_msb(xi, xk) {
                self.k() < other.k()
            } else {
                self.i() < other.i()
            }
        };

        if islt {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    }
}

impl PartialOrd for Voxel {
    fn partial_cmp(&self, other: &Voxel) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

type GeneIndex = u32;

// TODO: we could probably get away with packing this into one u32,
// as well, but that probably gets a bit sketchier on visium and the like.
struct GeneCount {
    count: u32,
    noise: u32,
}

impl GeneCount {
    fn zero() -> Self {
        GeneCount { count: 0, noise: 0 }
    }
}

// TODO: Honestly, what is the point of having a separate B-tree for every voxel?
// What if we were to just have one huge BTree that's like
// BTreeMap<(Voxel, VoxelOffset, GeneIndex), GeneCount>
//
// Wouldn't that have less overhead? We could even linearize the Voxel indexe
// to get better locality?
//
// We might even index by gene first, like:
// Vec<BTreeMap<(Voxel, VoxelOffset), GeneCount>>
//
// Let's think about acces patterns. When evaluating voxel copy proposals
// we need to get the counts for gene for the voxel. Breaking up by
// gene would involve a lot of lookups, so probably shouldn't do this.

// Represent a sparse count vector indexed by gene, and by
// a voxel offset, to keep track of repositioned transcripts.
// struct SparseTranscriptsVec {
//     // transcript count for transcripts repositioned to this voxel
//     counts: BTreeMap<(VoxelOffset, GeneIndex), GeneCount>,
// }

// Ok, so each key is 8 + 8 + 4 = 20 bytes
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
struct VoxelCountKey {
    voxel: Voxel,
    gene: GeneIndex,
    offset: VoxelOffset,
}

#[derive(Clone, Copy)]
struct VoxelState {
    cell: CellIndex,
    prior_cell: CellIndex,
    prior: f32,
}

// This will represent one square of the checkerboard
struct VoxelQuad {
    // Voxel state and prior.
    states: BTreeMap<Voxel, VoxelState>,

    // This is essentially one giant sparse matrix for the entire
    // voxel set. We also have to keep track of repositioned transcripts here.
    counts: BTreeMap<VoxelCountKey, GeneCount>,

    cell_edge_voxels: SampleSet<Voxel>,
    // TODO: Do we ever need to actually know the bounds? When daoes this come up?
    // bounds: (Voxel, Voxel),
}

impl VoxelQuad {
    // Initialize empty voxel set
    fn new() -> VoxelQuad {
        return VoxelQuad {
            states: BTreeMap::new(),
            counts: BTreeMap::new(),
            cell_edge_voxels: SampleSet::new(),
            // bounds: (from, to),
        };
    }

    // TODO: initialize with

    // Maybe we wait to see what functionality we actually need here
}

pub struct VoxelCheckerboard {
    // number of voxels in each direction
    quadsize: usize,

    // Main thing is we'll need to look up arbitrary Voxels,
    // which means first looking up which VoxelSet this is in.
    //
    // We also need to keep track of whether indexe parities to
    // do staggered updates. So how do we want to organize this.
    quads: HashMap<(u32, u32), RwLock<VoxelQuad>>,
}

impl VoxelCheckerboard {
    pub fn from_prior_transcript_assignments(
        dataset: &TranscriptDataset,
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        nucprior: f32,
        cellprior: f32,
    ) -> VoxelCheckerboard {
        let (xmin, _xmax, ymin, _ymax, zmin, zmax) = dataset.coordinate_span();
        let voxelsize_z = (zmax - zmin) / nzlayers as f32;

        let coords_to_voxel = |x: f32, y: f32, z: f32| {
            let i = ((x - xmin) / voxelsize).round().max(0.0) as i32;
            let j = ((y - ymin) / voxelsize).round().max(0.0) as i32;
            let k = (((z - zmin) / voxelsize_z) as i32)
                .min(nzlayers as i32 - 1)
                .max(0);
            Voxel::new(i, j, k)
        };

        // tally votes: count transcripts in each voxel aggregated by prior cell assignment
        let t0 = Instant::now();
        let mut nuc_votes: BTreeMap<(Voxel, CellIndex), u32> = BTreeMap::new();
        let mut cell_votes: BTreeMap<(Voxel, CellIndex), u32> = BTreeMap::new();
        for (transcript, prior) in dataset.transcripts.iter().zip(dataset.priorseg.iter()) {
            let voxel = coords_to_voxel(transcript.x, transcript.y, transcript.z);

            if prior.nucleus != BACKGROUND_CELL {
                let key = (voxel, prior.nucleus);
                *nuc_votes.entry(key).or_insert(0) += 1;
            }

            if prior.cell != BACKGROUND_CELL {
                let key = (voxel, prior.cell);
                *cell_votes.entry(key).or_insert(0) += 1;
            }
        }

        let mut checkerboard = VoxelCheckerboard {
            quadsize: (quadsize / voxelsize).round().max(1.0) as usize,
            quads: HashMap::new(),
        };

        // assign voxels based on vote winners
        let mut used_cells = HashMap::new();
        let mut current_voxel = Voxel::oob();
        let mut vote_winner = BACKGROUND_CELL;
        let mut vote_winner_count: u32 = 0;
        for ((voxel, cell), count) in nuc_votes {
            if voxel != current_voxel {
                if !current_voxel.is_oob() {
                    checkerboard.insert_state(
                        current_voxel,
                        VoxelState {
                            cell: vote_winner,
                            prior_cell: vote_winner,
                            prior: nucprior,
                        },
                    );
                    let next_cell_id = used_cells.len() as CellIndex;
                    used_cells.entry(vote_winner).or_insert(next_cell_id);
                }
                vote_winner = cell;
                vote_winner_count = count;
                current_voxel = voxel;
            } else {
                if count > vote_winner_count {
                    vote_winner = cell;
                    vote_winner_count = count;
                }
            }
        }
        if !current_voxel.is_oob() {
            checkerboard.insert_state(
                current_voxel,
                VoxelState {
                    cell: vote_winner,
                    prior_cell: vote_winner,
                    prior: nucprior,
                },
            );

            let next_cell_id = used_cells.len() as CellIndex;
            used_cells.entry(vote_winner).or_insert(next_cell_id);
        }
        println!("initialized voxel state: {:?}", t0.elapsed());
        dbg!(checkerboard.quads.len());
        dbg!(dataset.ncells);
        dbg!(used_cells.len());

        // re-assign cell indices so that there are no cells without any assigned voxel
        for quad in &mut checkerboard.quads.values() {
            let mut quad = quad.write().unwrap();
            for state in quad.states.values_mut() {
                let cell = *used_cells.get(&state.cell).unwrap();
                state.cell = cell;
                state.prior_cell = cell;
            }
        }

        // assign cell priors, by once again voting
        let mut current_voxel = Voxel::oob();
        let mut vote_winner = BACKGROUND_CELL;
        let mut vote_winner_count: u32 = 0;
        for ((voxel, cell), count) in cell_votes {
            if voxel != current_voxel {
                if !current_voxel.is_oob() {
                    if let Some(&vote_winner) = used_cells.get(&vote_winner) {
                        checkerboard.insert_state_if_missing(current_voxel, || VoxelState {
                            cell: BACKGROUND_CELL,
                            prior_cell: vote_winner,
                            prior: cellprior,
                        });
                    }
                }
                vote_winner = cell;
                vote_winner_count = count;
                current_voxel = voxel;
            } else {
                if count > vote_winner_count {
                    vote_winner = cell;
                    vote_winner_count = count;
                }
            }
        }
        if !current_voxel.is_oob() {
            if let Some(&vote_winner) = used_cells.get(&vote_winner) {
                checkerboard.insert_state_if_missing(current_voxel, || VoxelState {
                    cell: BACKGROUND_CELL,
                    prior_cell: vote_winner,
                    prior: cellprior,
                });
            }
        }
        println!("assigned cell priors: {:?}", t0.elapsed());

        // assign voxel counts
        let t0 = Instant::now();
        for run in dataset.transcripts.iter_runs() {
            let transcript = &run.value;
            let voxel = coords_to_voxel(transcript.x, transcript.y, transcript.z);
            let key = VoxelCountKey {
                voxel,
                gene: transcript.gene,
                offset: VoxelOffset::zero(),
            };
            checkerboard
                .write_quad(voxel)
                .counts
                .entry(key)
                .or_insert(GeneCount::zero())
                .count += run.len;
        }
        println!("assigned voxel counts: {:?}", t0.elapsed());

        // initialize edge voxel sets
        checkerboard.mirror_quad_edges();
        checkerboard.build_edge_sets();

        checkerboard
    }

    fn quad_index(&self, voxel: Voxel) -> (u32, u32) {
        let u = voxel.i() as u32 / self.quadsize as u32;
        let v = voxel.j() as u32 / self.quadsize as u32;
        (u, v)
    }

    fn write_quad(&mut self, voxel: Voxel) -> RwLockWriteGuard<VoxelQuad> {
        self.write_quad_index(self.quad_index(voxel))
    }

    fn write_quad_index(&mut self, index: (u32, u32)) -> RwLockWriteGuard<VoxelQuad> {
        self.quads
            .entry(index)
            .or_insert_with(|| RwLock::new(VoxelQuad::new()))
            .write()
            .unwrap()
    }

    // fn read_quad(&mut self, voxel: Voxel) -> RwLockReadGuard<VoxelQuad> {
    //     self.quads
    //         .entry(self.quad_index(voxel))
    //         .or_insert_with(|| RwLock::new(VoxelQuad::new()))
    //         .read()
    //         .unwrap()
    // }

    fn insert_state(&mut self, voxel: Voxel, state: VoxelState) {
        self.write_quad(voxel).states.insert(voxel, state);
    }

    fn insert_state_if_missing(&mut self, voxel: Voxel, f: impl FnOnce() -> VoxelState) {
        self.write_quad(voxel).states.entry(voxel).or_insert_with(f);
    }

    // We keep redundant copies of the states of voxels along the edges of each
    // quad. This way we can always stay within the quad to check neighborhoods.
    fn mirror_quad_edges(&mut self) {
        for (&(u, v), quad) in &self.quads {
            let mut quad = quad.write().unwrap();

            let quad_imin = u * self.quadsize as u32;
            let quad_imax = ((u + 1) * self.quadsize as u32) - 1;
            let quad_jmin = v * self.quadsize as u32;
            let quad_jmax = ((v + 1) * self.quadsize as u32) - 1;

            // copy states from our left neighbor
            if let Some(left_quad) = self.quads.get(&(u - 1, v)) {
                let left_quad = left_quad.read().unwrap();
                for (voxel, state) in &left_quad.states {
                    if voxel.i() as u32 == quad_imin - 1 {
                        quad.states.insert(voxel.clone(), state.clone());
                    }
                }
            }

            // copy states from our top neighbor
            if let Some(top_quad) = self.quads.get(&(u, v - 1)) {
                let top_quad = top_quad.read().unwrap();
                for (voxel, state) in &top_quad.states {
                    if voxel.j() as u32 == quad_jmin - 1 {
                        quad.states.insert(voxel.clone(), state.clone());
                    }
                }
            }

            // copy states from our right neighbor
            if let Some(right_quad) = self.quads.get(&(u + 1, v)) {
                let right_quad = right_quad.read().unwrap();
                for (voxel, state) in &right_quad.states {
                    if voxel.i() as u32 == quad_imax + 1 {
                        quad.states.insert(voxel.clone(), state.clone());
                    }
                }
            }

            // copy states from our bottom neighbor
            if let Some(bottom_quad) = self.quads.get(&(u, v + 1)) {
                let bottom_quad = bottom_quad.read().unwrap();
                for (voxel, state) in &bottom_quad.states {
                    if voxel.j() as u32 == quad_jmax + 1 {
                        quad.states.insert(voxel.clone(), state.clone());
                    }
                }
            }

            // TODO: We also have to check our corners!
        }
    }

    fn build_edge_sets(&mut self) {
        // have to do this to get around a double borrow issue
        let mut cell_edge_voxels = Vec::new();
        for quad in self.quads.values() {
            let mut quad = quad.write().unwrap();
            cell_edge_voxels.clear();
            for (voxel, state) in &quad.states {
                let cell = state.cell;
                if cell == BACKGROUND_CELL {
                    continue;
                }

                for neighbor in voxel.von_neumann_neighborhood() {
                    let neighbor_cell = quad
                        .states
                        .get(&neighbor)
                        .map_or(BACKGROUND_CELL, |state| state.cell);

                    if cell != neighbor_cell {
                        cell_edge_voxels.push(*voxel);
                        break;
                    }
                }
            }

            quad.cell_edge_voxels.clear();
            quad.cell_edge_voxels.extend(&cell_edge_voxels);
        }
    }
}

// TODO: Ok, now we have to think about how we are actually sampling voxels.
// I'm thinking I'm going to have rewrite everything in voxelsampler and sampler
// gets turned into paramsampler

// TODO: Data structure to hold a grid of `VoxelSets`
// Design considerations:
//   - Should this be sparse? Would we allow for empty quads? If so,
//     we may need to lazily add quads if transcripts get moved there.
//   - Write locks on all quads
//   - Probably need to access specific voxels
//   - Processing message passing between quads

// TODO:
//  Seems like the `lindel` crate is a good one to try
//  to use to do the spacefill stuff.

// TODO:
//  After each iteration, we are going to have to aggregate
//  a cell x gene matrix. We should think about how do that while
//  avoiding tons of allocation or contention.
//  I think this may be a good place to use try to use `dashmap`
//  and just store a big COO matrix or CSR matrix.
//

// TODO:
// Also have to think about how to message pass cell boundary updates
// and transcript repo updates between VoxelSets. This may be another
// case where we can use DashMap.
//
