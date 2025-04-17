// Data structures for maintaining a set of voxels each with an associated
// sparse transcript vector.

use super::sampleset::SampleSet;
use super::shardedvec::ShardedVec;
use super::sparsemat::SparseMat;
use super::transcripts::{CellIndex, TranscriptDataset, BACKGROUND_CELL};
use super::CountMatRowKey;

use itertools::Itertools;
use log::trace;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cmp::{Ordering, PartialOrd};
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound::Included;
use std::ops::DerefMut;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::time::Instant;

// Store a voxel offset in compact form
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct VoxelOffset {
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

    pub fn zero() -> Voxel {
        Voxel::new(0, 0, 0)
    }

    pub fn oob() -> Voxel {
        Voxel { index: OOB_VOXEL }
    }

    pub fn is_oob(&self) -> bool {
        self.index == OOB_VOXEL
    }

    pub fn coords(&self) -> [i32; 3] {
        [self.i(), self.j(), self.k()]
    }

    fn u32_coords(&self) -> [u32; 3] {
        [self.i() as u32, self.j() as u32, self.k() as u32]
    }

    pub fn i(&self) -> i32 {
        ((self.index >> 48) & 0xFFFFFF) as i32
    }

    pub fn j(&self) -> i32 {
        ((self.index >> 24) & 0xFFFFFF) as i32
    }

    pub fn k(&self) -> i32 {
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

#[derive(Hash, PartialEq, Eq, Copy, Clone)]
pub struct UndirectedVoxelPair {
    pub a: Voxel,
    pub b: Voxel,
}

impl UndirectedVoxelPair {
    pub fn new(a: Voxel, b: Voxel) -> UndirectedVoxelPair {
        if a <= b {
            UndirectedVoxelPair { a, b }
        } else {
            UndirectedVoxelPair { a: b, b: a }
        }
    }
}

type GeneIndex = u32;

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
pub struct VoxelCountKey {
    pub voxel: Voxel,
    pub gene: GeneIndex,
    pub offset: VoxelOffset,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelState {
    pub cell: CellIndex,
    pub prior_cell: CellIndex,
    pub prior: f32,
}

impl VoxelState {
    fn new_cell_only(cell: CellIndex) -> VoxelState {
        VoxelState {
            cell,
            prior_cell: BACKGROUND_CELL,
            prior: 0.0,
        }
    }
}

// This will represent one square of the checkerboard
pub struct VoxelQuad {
    // Voxel state and prior.
    pub states: BTreeMap<Voxel, VoxelState>,

    // This is essentially one giant sparse matrix for the entire
    // voxel set. We also have to keep track of repositioned transcripts here.
    pub counts: BTreeMap<VoxelCountKey, u32>,

    pub mismatch_edges: SampleSet<UndirectedVoxelPair>,

    pub kmax: i32,
    quadsize: usize,

    // quad coordinates
    pub u: u32,
    pub v: u32,
}

impl VoxelQuad {
    // Initialize empty voxel set
    fn new(kmax: i32, quadsize: usize, u: u32, v: u32) -> VoxelQuad {
        return VoxelQuad {
            states: BTreeMap::new(),
            counts: BTreeMap::new(),
            mismatch_edges: SampleSet::new(),
            kmax,
            quadsize,
            u,
            v,
        };
    }

    pub fn get_voxel_state(&self, voxel: Voxel) -> Option<&VoxelState> {
        self.states.get(&voxel)
    }

    pub fn set_voxel_cell(&mut self, voxel: Voxel, cell: CellIndex) {
        if cell == BACKGROUND_CELL {
            self.states.remove(&voxel);
        } else {
            self.states
                .entry(voxel)
                .and_modify(|state| state.cell = cell)
                .or_insert_with(|| VoxelState::new_cell_only(cell));
        }
    }

    pub fn get_voxel_cell(&self, voxel: Voxel) -> CellIndex {
        self.states
            .get(&voxel)
            .map(|state| state.cell)
            .unwrap_or(BACKGROUND_CELL)
    }

    // Inclusive bounds: (min_i, max_i, min_j, max_j)
    pub fn bounds(&self) -> (i32, i32, i32, i32) {
        (
            (self.u as usize * self.quadsize) as i32,
            (((self.u + 1) as usize * self.quadsize) as i32) - 1,
            (self.v as usize * self.quadsize) as i32,
            (((self.v + 1) as usize * self.quadsize) as i32) - 1,
        )
    }

    pub fn voxel_in_bounds(&self, voxel: Voxel) -> bool {
        let [i, j, _k] = voxel.coords();
        let (min_i, max_i, min_j, max_j) = self.bounds();
        min_i <= i && i <= max_i && min_j <= j && j <= max_j
    }

    pub fn update_voxel_cell(
        &mut self,
        voxel: Voxel,
        current_cell: CellIndex,
        proposed_cell: CellIndex,
    ) {
        self.set_voxel_cell(voxel, proposed_cell);

        for neighbor in voxel.von_neumann_neighborhood() {
            let k = neighbor.k();
            if k < 0 || k > self.kmax {
                continue;
            }

            let neighbor_cell = self.get_voxel_cell(neighbor);
            if neighbor_cell == proposed_cell {
                self.mismatch_edges
                    .remove(UndirectedVoxelPair::new(voxel, neighbor));
            } else if neighbor_cell == current_cell {
                self.mismatch_edges
                    .insert(UndirectedVoxelPair::new(voxel, neighbor));
            }
        }
    }
}

impl<'a> VoxelQuad {
    pub fn voxel_counts(
        &'a self,
        voxel: Voxel,
    ) -> std::collections::btree_map::Range<'a, VoxelCountKey, u32> {
        let from = VoxelCountKey {
            voxel,
            gene: 0,
            offset: VoxelOffset::zero(),
        };
        let to = VoxelCountKey {
            voxel,
            gene: GeneIndex::MAX,
            offset: VoxelOffset::zero(),
        };
        self.counts.range((Included(from), Included(to)))
    }
}

pub struct VoxelCheckerboard {
    // number of voxels in each x/y direction
    quadsize: usize,

    // number of cells (excluding any initialized without voxels)
    pub ncells: usize,

    // number of genes
    pub ngenes: usize,

    // maximum z layer
    pub kmax: i32,

    // volume of a single voxel
    pub voxel_volume: f32,

    // Main thing is we'll need to look up arbitrary Voxels,
    // which means first looking up which VoxelSet this is in.
    //
    // We also need to keep track of whether indexe parities to
    // do staggered updates. So how do we want to organize this.
    pub quads: HashMap<(u32, u32), RwLock<VoxelQuad>>,
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
        let voxel_volume = voxelsize * voxelsize * voxelsize_z;

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
            kmax: (nzlayers - 1) as i32,
            ncells: 0,
            ngenes: dataset.ngenes(),
            voxel_volume,
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
        trace!("initialized voxel state: {:?}", t0.elapsed());
        checkerboard.ncells = used_cells.len();

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
        trace!("assigned cell priors: {:?}", t0.elapsed());

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

            let mut quad = checkerboard.write_quad(voxel);
            let count = quad.counts.entry(key).or_insert(0_u32);
            *count += run.len;
        }
        trace!("assigned voxel counts: {:?}", t0.elapsed());

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
        let (u, v) = index;

        self.quads
            .entry(index)
            .or_insert_with(|| RwLock::new(VoxelQuad::new(self.kmax, self.quadsize, u, v)))
            .write()
            .unwrap()
    }

    fn quad_bounds(&self, index: (u32, u32)) -> (Voxel, Voxel) {
        let min_ij = Voxel::new(
            index.0 as i32 * self.quadsize as i32,
            index.1 as i32 * self.quadsize as i32,
            0,
        );

        let max_ij = Voxel::new(
            (index.0 + 1) as i32 * self.quadsize as i32 - 1,
            (index.1 + 1) as i32 * self.quadsize as i32 - 1,
            0,
        );

        (min_ij, max_ij)
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
            let quad = quad.read().unwrap();
            self.for_each_quad_neighbor(u, v, |neighbor_quad| {
                let (min_i, max_i, min_j, max_j) = neighbor_quad.bounds();
                for (voxel, state) in &quad.states {
                    let [i, j, k] = voxel.coords();
                    if i + 1 == min_i || i - 1 == max_i || j + 1 == min_j || j - 1 == max_j {
                        neighbor_quad.states.insert(voxel.clone(), state.clone());
                    }
                }
            });
        }
    }

    fn build_edge_sets(&mut self) {
        // have to do this to get around a double borrow issue
        let mut mismatch_edges = Vec::new();
        for quad in self.quads.values() {
            let mut quad = quad.write().unwrap();
            mismatch_edges.clear();
            for (&voxel, state) in &quad.states {
                let cell = state.cell;
                if cell == BACKGROUND_CELL {
                    continue;
                }

                for neighbor in voxel.von_neumann_neighborhood() {
                    let k = neighbor.k();
                    if k < 0 || k > self.kmax {
                        continue;
                    }

                    let neighbor_cell = quad
                        .states
                        .get(&neighbor)
                        .map_or(BACKGROUND_CELL, |state| state.cell);

                    if cell != neighbor_cell {
                        mismatch_edges.push(UndirectedVoxelPair::new(voxel, neighbor));
                    }
                }
            }

            quad.mismatch_edges.clear();
            quad.mismatch_edges.extend(&mismatch_edges);
        }
    }

    pub fn for_each_quad_neighbor<F>(&self, u: u32, v: u32, f: F)
    where
        F: Fn(&mut VoxelQuad) -> (),
    {
        let offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        for (delta_u, delta_v) in offsets {
            let neighbor_u = u as i32 + delta_u;
            let neighbor_v = v as i32 + delta_v;

            if neighbor_u < 0 || neighbor_v < 0 {
                continue;
            }

            let neighbor_u = neighbor_u as u32;
            let neighbor_v = neighbor_v as u32;

            if let Some(neighbor_quad) = self.quads.get(&(neighbor_u, neighbor_v)) {
                f(neighbor_quad.write().unwrap().deref_mut())
            }
        }
    }

    pub fn compute_cell_volume_surface_area(
        &self,
        volume: &mut ShardedVec<u32>,
        surface_area: &mut ShardedVec<u32>,
    ) {
        volume.zero();
        surface_area.zero();

        self.quads.par_iter().for_each(|((_u, _v), quad)| {
            let quad = quad.read().unwrap();
            for (&voxel, state) in &quad.states {
                // We mirror state on the border. We need to skip these mirrored voxels to avoid double-counting.
                if !quad.voxel_in_bounds(voxel) {
                    continue;
                }

                if state.cell != BACKGROUND_CELL {
                    volume.add(state.cell as usize, 1);

                    let mut voxel_surface_area = 0;
                    for neighbor in voxel.von_neumann_neighborhood() {
                        let k = neighbor.k();
                        let neighbor_cell = if k >= 0 && k <= quad.kmax {
                            quad.get_voxel_cell(neighbor)
                        } else {
                            BACKGROUND_CELL
                        };

                        if state.cell != neighbor_cell {
                            voxel_surface_area += 1;
                        }
                    }
                    surface_area.add(state.cell as usize, voxel_surface_area);
                }
            }
        });
    }

    pub fn compute_counts(&self, counts: &mut SparseMat<u32, CountMatRowKey>) {
        counts.zero();
        self.quads.par_iter().for_each(|((_u, _v), quad)| {
            let quad = quad.read().unwrap();
            for (
                &VoxelCountKey {
                    voxel,
                    gene,
                    offset: _offset,
                },
                &count,
            ) in &quad.counts
            {
                let cell = quad.get_voxel_cell(voxel);
                if cell != BACKGROUND_CELL {
                    counts
                        .row(cell as usize)
                        .write()
                        .add(CountMatRowKey::new(gene, voxel.k() as u32), count);
                }
            }
        });
    }
}
