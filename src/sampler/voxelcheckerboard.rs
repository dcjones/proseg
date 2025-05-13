// Data structures for maintaining a set of voxels each with an associated:
// sparse transcript vector.

use super::math::logistic;
use super::polygons::{PolygonBuilder, union_all_into_multipolygon};
use super::sampleset::SampleSet;
use super::shardedvec::ShardedVec;
use super::sparsemat::SparseMat;
use super::transcripts::{BACKGROUND_CELL, CellIndex, TranscriptDataset};
use super::{CountMatRowKey, ModelParams};

use flate2::read::GzDecoder;
use geo::geometry::{MultiPolygon, Polygon};
use half::f16;
use log::info;
use log::trace;
use ndarray::{Array2, Zip};
use ndarray_npy::{ReadNpyExt, read_npy};
use rand::rng;
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cell::RefCell;
use std::cmp::{Ordering, PartialOrd};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem::drop;
use std::ops::Bound::Included;
use std::ops::{Add, DerefMut};
use std::sync::{RwLock, RwLockWriteGuard};
use std::time::Instant;
use thread_local::ThreadLocal;

pub type CellPolygon = MultiPolygon<f32>;
pub type CellPolygonLayers = Vec<(i32, CellPolygon)>;

// Simple affine transform matrix to map pixel coordinates onto slide microns
pub struct PixelTransform {
    pub tx: [f32; 3],
    pub ty: [f32; 3],
}

impl PixelTransform {
    pub fn scale(s: f32) -> PixelTransform {
        PixelTransform {
            tx: [s, 0.0, 0.0],
            ty: [0.0, s, 0.0],
        }
    }

    fn transform(&self, i: usize, j: usize) -> (f32, f32) {
        let (i, j) = (i as f32, j as f32);

        (
            i * self.tx[0] + j * self.tx[1] + self.tx[2],
            i * self.ty[0] + j * self.ty[1] + self.ty[2],
        )
    }

    fn det(&self) -> f32 {
        self.tx[0] * self.ty[1] - self.tx[1] * self.ty[0]
    }
}

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
    pub fn new(di: i32, dj: i32, dk: i32) -> VoxelOffset {
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

    pub fn coords(&self) -> [i32; 3] {
        [self.di(), self.dj(), self.dk()]
    }
}

impl Add for VoxelOffset {
    type Output = VoxelOffset;

    fn add(self, other: VoxelOffset) -> VoxelOffset {
        let [di_a, dj_a, dk_a] = self.coords();
        let [di_b, dj_b, dk_b] = other.coords();

        VoxelOffset::new(di_a + di_b, dj_a + dj_b, dk_a + dk_b)
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

pub const VON_NEUMANN_AND_SELF_OFFSETS: [(i32, i32, i32); 7] = [
    (0, 0, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

// pub const MOORE_AND_SELF_OFFSETS: [(i32, i32, i32); 27] = [
//     (0, 0, 0),
//     // top layer
//     (-1, 0, -1),
//     (0, 0, -1),
//     (1, 0, -1),
//     (-1, 1, -1),
//     (0, 1, -1),
//     (1, 1, -1),
//     (-1, -1, -1),
//     (0, -1, -1),
//     (1, -1, -1),
//     // middle layer
//     (-1, 0, 0),
//     (1, 0, 0),
//     (-1, 1, 0),
//     (0, 1, 0),
//     (1, 1, 0),
//     (-1, -1, 0),
//     (0, -1, 0),
//     (1, -1, 0),
//     // bottom layer
//     (-1, 0, 1),
//     (0, 0, 1),
//     (1, 0, 1),
//     (-1, 1, 1),
//     (0, 1, 1),
//     (1, 1, 1),
//     (-1, -1, 1),
//     (0, -1, 1),
//     (1, -1, 1),
// ];

// TODO: we could just u32 for coordinates.
impl Voxel {
    pub fn new(i: i32, j: i32, k: i32) -> Voxel {
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

    // pub fn zero() -> Voxel {
    //     Voxel::new(0, 0, 0)
    // }

    pub fn oob() -> Voxel {
        Voxel { index: OOB_VOXEL }
    }

    pub fn is_oob(&self) -> bool {
        self.index == OOB_VOXEL
    }

    pub fn coords(&self) -> [i32; 3] {
        [self.i(), self.j(), self.k()]
    }

    // fn u32_coords(&self) -> [u32; 3] {
    //     [self.i() as u32, self.j() as u32, self.k() as u32]
    // }

    pub fn i(&self) -> i32 {
        ((self.index >> 48) & 0xFFFFFF) as i32
    }

    pub fn j(&self) -> i32 {
        ((self.index >> 24) & 0xFFFFFF) as i32
    }

    pub fn k(&self) -> i32 {
        (self.index & 0xFFFF) as i32
    }

    pub fn offset(&self, d: VoxelOffset) -> Voxel {
        self.offset_coords(d.di(), d.dj(), d.dk())
    }

    pub fn offset_coords(&self, di: i32, dj: i32, dk: i32) -> Voxel {
        let new_i = self.i() + di;
        let new_j = self.j() + dj;
        let new_k = self.k() + dk;

        if !(0..(1 << 24)).contains(&new_i)
            || !(0..(1 << 24)).contains(&new_j)
            || !(0..(1 << 16)).contains(&new_k)
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

    pub fn von_neumann_neighborhood_xy(&self) -> [Voxel; 4] {
        let [i, j, k] = self.coords();
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
            .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    }

    // pub fn moore_neighborhood(&self) -> [Voxel; 26] {
    //     let [i, j, k] = self.coords();
    //     [
    //         // top layer
    //         (-1, 0, -1),
    //         (0, 0, -1),
    //         (1, 0, -1),
    //         (-1, 1, -1),
    //         (0, 1, -1),
    //         (1, 1, -1),
    //         (-1, -1, -1),
    //         (0, -1, -1),
    //         (1, -1, -1),
    //         // middle layer
    //         (-1, 0, 0),
    //         (1, 0, 0),
    //         (-1, 1, 0),
    //         (0, 1, 0),
    //         (1, 1, 0),
    //         (-1, -1, 0),
    //         (0, -1, 0),
    //         (1, -1, 0),
    //         // bottom layer
    //         (-1, 0, 1),
    //         (0, 0, 1),
    //         (1, 0, 1),
    //         (-1, 1, 1),
    //         (0, 1, 1),
    //         (1, 1, 1),
    //         (-1, -1, 1),
    //         (0, -1, 1),
    //         (1, -1, 1),
    //     ]
    //     .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    // }

    // // Gives the line segment defining the edge between two voxels bordering on
    // // the xy plane. (Panics if the voxels don't border.)
    // pub fn edge_xy(&self, other: Voxel) -> ((i32, i32), (i32, i32)) {
    //     let [i_a, j_a, k_a] = self.coords();
    //     let [i_b, j_b, k_b] = other.coords();

    //     if k_a != k_b {
    //         dbg!(((i_a, j_a, k_a), (j_b, j_b, k_b)));
    //     }

    //     assert!(k_a == k_b);
    //     assert!((i_a - i_b).abs() + (j_a - j_b).abs() == 1);

    //     let i0 = i_a.max(i_b);
    //     let j0 = j_a.max(j_b);

    //     let i1 = i0 + (j_b - j_a).abs();
    //     let j1 = j0 + (i_b - i_a).abs();

    //     ((i0, j0), (i1, j1))
    // }

    // Gives the line segment defining the edge between two voxels bordering on
    // the xy plane. (Panics if the voxels don't border.)
    pub fn offset_edge_xy(&self, offset: VoxelOffset) -> ((i32, i32), (i32, i32)) {
        let [i, j, _k] = self.coords();

        let [di, dj, dk] = offset.coords();

        assert!(dk == 0);
        assert!(di.abs() + dj.abs() == 1);

        let i0 = i.max(i + di);
        let j0 = j.max(j + dj);

        let i1 = i0 + dj.abs();
        let j1 = j0 + di.abs();

        ((i0, j0), (i1, j1))
    }
}

pub fn von_neumann_neighborhood_xy_offsets() -> [VoxelOffset; 4] {
    [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)].map(|(di, dj, dk)| VoxelOffset::new(di, dj, dk))
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
        } else if less_msb(xi, xk) {
            self.k() < other.k()
        } else {
            self.i() < other.i()
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
    pub log_prior: f16,    // log(p)
    pub log_1m_prior: f16, // log(1-p)
}

impl VoxelState {
    fn new_cell_only(cell: CellIndex) -> VoxelState {
        VoxelState {
            cell,
            prior_cell: BACKGROUND_CELL,
            log_prior: f16::NAN,
            log_1m_prior: f16::NAN,
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

    pub counts_deltas: Vec<(VoxelCountKey, u32)>,

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
        VoxelQuad {
            states: BTreeMap::new(),
            counts: BTreeMap::new(),
            counts_deltas: Vec::new(),
            mismatch_edges: SampleSet::new(),
            kmax,
            quadsize,
            u,
            v,
        }
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

    // If the given voxel in is bounds of this quad and is a bubble (i.e. von
    // neummann neighborhood all share the same state, that differs from voxel's),
    // return Some((voxel_cell, neighbor_cell)), otherwise returns None.
    pub fn is_bubble(&self, voxel: Voxel) -> Option<(CellIndex, CellIndex)> {
        if !self.voxel_in_bounds(voxel) {
            return None;
        }

        let cell = self.get_voxel_cell(voxel);

        let neighbor_cells = voxel
            .von_neumann_neighborhood_xy()
            .map(|neighbor| self.get_voxel_cell(neighbor));
        let neighbor_cell = neighbor_cells[0];
        if !neighbor_cells.iter().all(|&c| c == neighbor_cell) {
            return None;
        }

        if cell != neighbor_cell {
            Some((cell, neighbor_cell))
        } else {
            None
        }
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
        let [i, j, k] = voxel.coords();
        let (min_i, max_i, min_j, max_j) = self.bounds();
        min_i <= i && i <= max_i && min_j <= j && j <= max_j && 0 <= k && k <= self.kmax
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
    pub quadsize: usize,

    // number of cells (excluding any initialized without voxels)
    pub ncells: usize,

    // number of genes
    pub ngenes: usize,

    // number of voxel z layers
    nzlayers: usize,

    // maximum z layer
    pub kmax: i32,

    // volume of a single voxel
    pub voxel_volume: f32,

    // kept for coordinate transforms back to microns
    xmin: f32,
    ymin: f32,
    zmin: f32,
    pub voxelsize: f32,
    pub voxelsize_z: f32,

    // boolean mask of cells from initialization that are used in inference
    pub used_cell_mask: Vec<bool>,

    // Main thing is we'll need to look up arbitrary Voxels,
    // which means first looking up which VoxelSet this is in.
    //
    // We also need to keep track of whether indexe parities to
    // do staggered updates. So how do we want to organize this.
    pub quads: HashMap<(u32, u32), RwLock<VoxelQuad>>,

    // Set of keys in `quads`. This is obviously redundant, but a
    // contrivance to avoid multiple borrows of quads in some places.
    pub quads_coords: HashSet<(u32, u32)>,
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

        let voxelsize_z = if zmin == zmax {
            1.0
        } else {
            (zmax - zmin) / nzlayers as f32
        };
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
            nzlayers,
            voxel_volume,
            xmin,
            ymin,
            zmin,
            voxelsize,
            voxelsize_z,
            used_cell_mask: vec![false; dataset.ncells],
            quads: HashMap::new(),
            quads_coords: HashSet::new(),
        };

        // assign voxels based on vote winners
        let log_nucprior = f16::from_f32(nucprior.ln());
        let log_1m_nucprior = f16::from_f32((1.0 - nucprior).ln());
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
                            log_prior: log_nucprior,
                            log_1m_prior: log_1m_nucprior,
                        },
                    );
                    let next_cell_id = used_cells.len() as CellIndex;
                    used_cells.entry(vote_winner).or_insert(next_cell_id);
                    checkerboard.used_cell_mask[vote_winner as usize] = true;
                }
                vote_winner = cell;
                vote_winner_count = count;
                current_voxel = voxel;
            } else if count > vote_winner_count {
                vote_winner = cell;
                vote_winner_count = count;
            }
        }
        if !current_voxel.is_oob() {
            checkerboard.insert_state(
                current_voxel,
                VoxelState {
                    cell: vote_winner,
                    prior_cell: vote_winner,
                    log_prior: log_nucprior,
                    log_1m_prior: log_1m_nucprior,
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
        let log_cellprior = f16::from_f32(cellprior.ln());
        let log_1m_cellprior = f16::from_f32((1.0 - cellprior).ln());
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
                            log_prior: log_cellprior,
                            log_1m_prior: log_1m_cellprior,
                        });
                    }
                }
                vote_winner = cell;
                vote_winner_count = count;
                current_voxel = voxel;
            } else if count > vote_winner_count {
                vote_winner = cell;
                vote_winner_count = count;
            }
        }
        if !current_voxel.is_oob() {
            if let Some(&vote_winner) = used_cells.get(&vote_winner) {
                checkerboard.insert_state_if_missing(current_voxel, || VoxelState {
                    cell: BACKGROUND_CELL,
                    prior_cell: vote_winner,
                    log_prior: log_cellprior,
                    log_1m_prior: log_1m_cellprior,
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
        checkerboard.quads_coords.extend(checkerboard.quads.keys());

        // initialize edge voxel sets
        checkerboard.mirror_quad_edges();
        checkerboard.build_edge_sets();

        checkerboard
    }

    /// Reads an NPY file that may or may not be gzipped
    fn read_npy_gz<T: ndarray_npy::ReadableElement>(
        filename: &str,
    ) -> Result<Array2<T>, Box<dyn std::error::Error>> {
        // Open the file
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        // Read the first few bytes to check if it's gzipped
        let mut magic_bytes = [0u8; 2];
        let peek_result = {
            let buf_reader_ref = &mut reader;
            buf_reader_ref.read_exact(&mut magic_bytes)
        };

        // Rewind the reader to the beginning of the file
        let file = File::open(filename)?;
        let reader = BufReader::new(file);

        if peek_result.is_ok() && magic_bytes == [0x1f, 0x8b] {
            // File is gzipped
            let gz_decoder = GzDecoder::new(reader);
            Ok(Array2::<T>::read_npy(gz_decoder)?)
        } else {
            // Not gzipped, read normally
            Ok(read_npy(filename)?)
        }
    }

    fn tally_cellpose_pixels_with_probs(
        cellprob_filename: &str,
        cellprob_discount: f32,
        masks: &Array2<u32>,
        pixel_transform: &PixelTransform,
        cell_votes: &mut BTreeMap<(Voxel, CellIndex), f32>,
        coords_to_voxel: impl Fn(f32, f32, f32) -> Voxel,
        zmid: f32,
    ) {
        let cellprobs: Array2<f32> = Self::read_npy_gz(cellprob_filename).unwrap_or_else(
            |_err| panic!("Unable to read cellpose cellprob from {}. Make sure it an npy file (possibly gzipped) containing a float32 matrix", cellprob_filename)
        );
        assert!(masks.shape() == cellprobs.shape());

        Zip::indexed(masks)
            .and(&cellprobs)
            .for_each(|(i, j), &masks_ij, &cellprobs_ij| {
                if masks_ij == 0 {
                    return;
                }

                let (x, y) = pixel_transform.transform(j, i);
                let voxel = coords_to_voxel(x, y, zmid);
                let cell = masks_ij - 1;
                let prior = logistic(cellprobs_ij);

                let vote = cell_votes.entry((voxel, cell)).or_insert(0.0);
                *vote += cellprob_discount * prior;
            });
    }

    fn tally_cellpose_pixels_without_probs(
        prior: f32,
        masks: &Array2<u32>,
        pixel_transform: &PixelTransform,
        cell_votes: &mut BTreeMap<(Voxel, CellIndex), f32>,
        coords_to_voxel: impl Fn(f32, f32, f32) -> Voxel,
        zmid: f32,
    ) {
        Zip::indexed(masks).for_each(|(i, j), &masks_ij| {
            if masks_ij == 0 {
                return;
            }

            let (x, y) = pixel_transform.transform(j, i);
            let voxel = coords_to_voxel(x, y, zmid);
            let cell = masks_ij - 1;

            let vote = cell_votes.entry((voxel, cell)).or_insert(0.0);
            *vote += prior;
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_cellpose_masks(
        dataset: &mut TranscriptDataset,
        masks_filename: &str,
        cellprob_filename: &Option<String>,
        cellprob_discount: f32,
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        pixel_transform: &PixelTransform,
        cellprior: f32,
    ) -> VoxelCheckerboard {
        let masks: Array2<u32> = Self::read_npy_gz(masks_filename).unwrap_or_else(
            |_err| panic!("Unable to read cellpose masks from {}. Make sure it an npy file (possibly gzipped) containing a uint32 matrix", masks_filename)
        );
        info!(
            "Read {} by {} cellpose mask.",
            masks.shape()[0],
            masks.shape()[1]
        );

        let (xmin, _xmax, ymin, _ymax, zmin, zmax) = dataset.coordinate_span();
        let voxelsize_z = if zmin == zmax {
            1.0
        } else {
            (zmax - zmin) / nzlayers as f32
        };
        let voxel_volume = voxelsize * voxelsize * voxelsize_z;
        let zmid = dataset.z_mean();

        let coords_to_voxel = |x: f32, y: f32, z: f32| {
            let i = ((x - xmin) / voxelsize).round().max(0.0) as i32;
            let j = ((y - ymin) / voxelsize).round().max(0.0) as i32;
            let k = (((z - zmin) / voxelsize_z) as i32)
                .min(nzlayers as i32 - 1)
                .max(0);
            Voxel::new(i, j, k)
        };

        let t0 = Instant::now();
        let mut cell_votes: BTreeMap<(Voxel, CellIndex), f32> = BTreeMap::new();

        if let Some(cellprob_filename) = cellprob_filename {
            Self::tally_cellpose_pixels_with_probs(
                cellprob_filename,
                cellprob_discount,
                &masks,
                pixel_transform,
                &mut cell_votes,
                coords_to_voxel,
                zmid,
            );
        } else {
            Self::tally_cellpose_pixels_without_probs(
                cellprior,
                &masks,
                pixel_transform,
                &mut cell_votes,
                coords_to_voxel,
                zmid,
            );
        }
        trace!("Voting on voxel states: {:?}", t0.elapsed());

        // save memory where we can
        drop(masks);

        let mut checkerboard = VoxelCheckerboard {
            quadsize: (quadsize / voxelsize).round().max(1.0) as usize,
            kmax: (nzlayers - 1) as i32,
            ncells: 0,
            ngenes: dataset.ngenes(),
            nzlayers,
            voxel_volume,
            xmin,
            ymin,
            zmin,
            voxelsize,
            voxelsize_z,
            used_cell_mask: vec![false; 0],
            quads: HashMap::new(),
            quads_coords: HashSet::new(),
        };

        let t0 = Instant::now();
        let pixels_per_voxel = pixel_transform.det().abs().recip();
        let mut used_cells = HashMap::new();
        let mut current_voxel = Voxel::oob();
        let mut vote_winner = BACKGROUND_CELL;
        let mut vote_winner_prior_sum: f32 = 0.0;
        for ((voxel, cell), prior_sum) in cell_votes {
            if voxel != current_voxel {
                if !current_voxel.is_oob() {
                    let prior = prior_sum / pixels_per_voxel;
                    checkerboard.insert_state(
                        current_voxel,
                        VoxelState {
                            cell: vote_winner,
                            prior_cell: vote_winner,
                            log_prior: f16::from_f32(prior.ln()),
                            log_1m_prior: f16::from_f32((1.0 - prior).ln()),
                        },
                    );
                    let next_cell_id = used_cells.len() as CellIndex;
                    used_cells.entry(vote_winner).or_insert(next_cell_id);
                }
                vote_winner = cell;
                vote_winner_prior_sum = prior_sum;
                current_voxel = voxel;
            } else if prior_sum > vote_winner_prior_sum {
                vote_winner = cell;
                vote_winner_prior_sum = prior_sum;
            }
        }
        trace!("Set voxel states: {:?}", t0.elapsed());

        if !current_voxel.is_oob() {
            let prior = vote_winner_prior_sum / pixels_per_voxel;
            checkerboard.insert_state(
                current_voxel,
                VoxelState {
                    cell: vote_winner,
                    prior_cell: vote_winner,
                    log_prior: f16::from_f32(prior.ln()),
                    log_1m_prior: f16::from_f32((1.0 - prior).ln()),
                },
            );

            let next_cell_id = used_cells.len() as CellIndex;
            used_cells.entry(vote_winner).or_insert(next_cell_id);
        }

        checkerboard.ncells = used_cells.len();

        // Rewrite original ids
        let mut cell_id_pairs: Vec<_> = used_cells
            .iter()
            .map(|(original_cell_id, cell_id)| (*cell_id, *original_cell_id))
            .collect();
        cell_id_pairs.sort();
        dataset.original_cell_ids.clear();
        dataset.original_cell_ids.extend(
            cell_id_pairs
                .iter()
                .map(|(_, original_cell_id)| (original_cell_id + 1).to_string()),
        );
        checkerboard.used_cell_mask = vec![true; used_cells.len()];

        checkerboard.quads.values().for_each(|quad| {
            let mut quad_lock = quad.write().unwrap();

            // re-assign cell indices so that there are no cells without any assigned voxel
            for state in quad_lock.states.values_mut() {
                let cell = *used_cells.get(&state.cell).unwrap();
                state.cell = cell;
                state.prior_cell = cell;
            }

            // copy the state along the z-axis
            let states_2d = quad_lock.states.clone();
            for (voxel, state) in states_2d {
                let [i, j, k] = voxel.coords();
                for k_down in (0..k).rev() {
                    quad_lock.states.insert(Voxel::new(i, j, k_down), state);
                }

                for k_up in (k + 1)..(quad_lock.kmax + 1) {
                    quad_lock.states.insert(Voxel::new(i, j, k_up), state);
                }
            }
        });

        // assign voxel counts
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

        checkerboard.quads_coords.extend(checkerboard.quads.keys());

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

    // fn quad_bounds(&self, index: (u32, u32)) -> (Voxel, Voxel) {
    //     let min_ij = Voxel::new(
    //         index.0 as i32 * self.quadsize as i32,
    //         index.1 as i32 * self.quadsize as i32,
    //         0,
    //     );

    //     let max_ij = Voxel::new(
    //         (index.0 + 1) as i32 * self.quadsize as i32 - 1,
    //         (index.1 + 1) as i32 * self.quadsize as i32 - 1,
    //         0,
    //     );

    //     (min_ij, max_ij)
    // }

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

    // fn update_state(
    //     &mut self,
    //     voxel: Voxel,
    //     init: impl FnOnce() -> VoxelState,
    //     update: impl FnOnce(&mut VoxelState),
    // ) {
    //     update(
    //         self.write_quad(voxel)
    //             .states
    //             .entry(voxel)
    //             .or_insert_with(init),
    //     )
    // }

    // We keep redundant copies of the states of voxels along the edges of each
    // quad. This way we can always stay within the quad to check neighborhoods.
    fn mirror_quad_edges(&mut self) {
        for (&(u, v), quad) in &self.quads {
            let quad = quad.read().unwrap();
            self.for_each_quad_neighbor(u, v, |neighbor_quad| {
                let (min_i, max_i, min_j, max_j) = neighbor_quad.bounds();
                for (voxel, state) in &quad.states {
                    let [i, j, _k] = voxel.coords();
                    if i + 1 == min_i || i - 1 == max_i || j + 1 == min_j || j - 1 == max_j {
                        neighbor_quad.states.insert(*voxel, *state);
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
        F: Fn(&mut VoxelQuad),
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

    pub fn compute_counts(
        &self,
        counts: &mut SparseMat<u32, CountMatRowKey>,
        unassigned_counts: &mut [ShardedVec<u32>],
    ) {
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
                } else {
                    unassigned_counts[voxel.k() as usize].add(gene as usize, count);
                }
            }
        });
    }

    pub fn cell_centroids(&self, params: &ModelParams) -> Array2<f32> {
        let mut centroids = Array2::zeros((self.ncells, 3));

        self.quads.values().for_each(|quad| {
            let quad_lock = quad.read().unwrap();
            for (voxel, state) in quad_lock.states.iter() {
                if state.cell != BACKGROUND_CELL {
                    let [i, j, k] = voxel.coords();

                    let x = ((i as f32) + 0.5) * self.voxelsize + self.xmin;
                    let y = ((j as f32) + 0.5) * self.voxelsize + self.ymin;
                    let z = ((k as f32) + 0.5) * self.voxelsize_z + self.zmin;

                    let mut centroids_c = centroids.row_mut(state.cell as usize);
                    centroids_c[0] += x;
                    centroids_c[1] += y;
                    centroids_c[2] += z;
                }
            }
        });

        centroids
            .rows_mut()
            .into_iter()
            .zip(params.cell_voxel_count.iter())
            .for_each(|(mut centroids_c, voxel_count_c)| {
                centroids_c /= voxel_count_c as f32;
            });

        centroids
    }

    pub fn cell_polygons(&self) -> (Vec<CellPolygonLayers>, Vec<CellPolygon>) {
        let mut cell_voxels = vec![HashSet::new(); self.ncells];
        self.quads.values().for_each(|quad| {
            quad.read()
                .unwrap()
                .states
                .iter()
                .for_each(|(&voxel, &state)| {
                    if state.cell != BACKGROUND_CELL {
                        cell_voxels[state.cell as usize].insert(voxel);
                    }
                });
        });

        let voxel_corner_to_world_pos = |voxel: Voxel| -> (f32, f32, f32) {
            let [i, j, k] = voxel.coords();
            (
                self.xmin + (i as f32) * self.voxelsize,
                self.ymin + (j as f32) * self.voxelsize,
                self.zmin + (k as f32) * self.voxelsize_z,
            )
        };

        let polygon_builder = ThreadLocal::new();
        let cell_polygons: Vec<Vec<(i32, MultiPolygon<f32>)>> = cell_voxels
            .par_iter()
            .map(|voxels| {
                let mut polygon_builder = polygon_builder
                    .get_or(|| RefCell::new(PolygonBuilder::new()))
                    .borrow_mut();

                polygon_builder.cell_voxels_to_polygons(voxel_corner_to_world_pos, voxels)
            })
            .collect();

        let cell_flattened_polygons: Vec<_> = cell_polygons
            .par_iter()
            .map(|polys| {
                let mut flat_polys: Vec<Polygon<f32>> = Vec::new();
                for (_k, poly) in polys {
                    flat_polys.extend(poly.iter().cloned());
                }
                union_all_into_multipolygon(flat_polys, true)
            })
            .collect();

        (cell_polygons, cell_flattened_polygons)
    }

    pub fn consensus_cell_polygons(&self) -> Vec<CellPolygon> {
        let mut voxel_votes = HashMap::new();
        let mut top_voxel: HashMap<CellIndex, (Voxel, u32)> = HashMap::new();
        self.quads.values().for_each(|quad| {
            let quad_lock = quad.read().unwrap();
            quad_lock.states.iter().for_each(|(&voxel, &state)| {
                let cell = state.cell;
                if cell == BACKGROUND_CELL || !quad_lock.voxel_in_bounds(voxel) {
                    return;
                }
                let [i, j, _k] = voxel.coords();

                let transcript_count = quad_lock
                    .voxel_counts(voxel)
                    .fold(0, |accum, (_key, &count)| accum + count);

                voxel_votes
                    .entry((i, j))
                    .or_insert_with(|| Vec::with_capacity(self.nzlayers))
                    .push((cell, transcript_count));

                top_voxel
                    .entry(cell)
                    .and_modify(|e| {
                        if transcript_count > e.1 {
                            *e = (Voxel::new(i, j, 0), transcript_count)
                        }
                    })
                    .or_insert((Voxel::new(i, j, 0), transcript_count));
            })
        });

        let mut cell_voxels = vec![HashSet::new(); self.ncells];
        for ((voxel_i, voxel_j), mut votes) in voxel_votes {
            votes.sort();

            let mut winner = BACKGROUND_CELL;
            let mut winner_count = 0;

            let mut i = 0;
            while i < votes.len() {
                let mut count = 0;
                let mut j = i;
                while j < votes.len() && votes[j].0 == votes[i].0 {
                    count += votes[j].1;
                    j += 1;
                }

                if count > winner_count || winner == BACKGROUND_CELL {
                    winner = votes[i].0;
                    winner_count = count;
                }
                i = j;
            }

            assert!(winner != BACKGROUND_CELL);
            cell_voxels[winner as usize].insert(Voxel::new(voxel_i, voxel_j, 0));
        }

        // Issues arise with some downstream tools (e.g. xeniumranger) if there
        // are empty cell polygons, which can happen with this consensus approach.
        // Here we try to fix those cases by including at least on voxel.
        for (cell, voxels) in cell_voxels.iter_mut().enumerate() {
            if voxels.is_empty() {
                let cell = cell as u32;
                voxels.insert(top_voxel[&cell].0);
            }
        }

        let voxel_corner_to_world_pos = |voxel: Voxel| -> (f32, f32, f32) {
            let [i, j, k] = voxel.coords();
            (
                self.xmin + (i as f32) * self.voxelsize,
                self.ymin + (j as f32) * self.voxelsize,
                self.zmin + (k as f32) * self.voxelsize_z,
            )
        };

        let polygon_builder = ThreadLocal::new();
        let cell_polygons: Vec<CellPolygon> = cell_voxels
            .par_iter()
            .map(|voxels| {
                let mut polygon_builder = polygon_builder
                    .get_or(|| RefCell::new(PolygonBuilder::new()))
                    .borrow_mut();

                let polygons =
                    polygon_builder.cell_voxels_to_polygons(voxel_corner_to_world_pos, voxels);
                if polygons.is_empty() {
                    CellPolygon::new(vec![])
                } else {
                    assert!(polygons.len() == 1);
                    let (_k, polygon) = polygons.first().unwrap();
                    polygon.clone()
                }
            })
            .collect();

        cell_polygons
    }

    // Copy occupied voxel states to unoccupied neighbors
    fn expand_cells(&mut self) {
        self.quads.par_iter().for_each(|(_quad_pos, quad)| {
            let mut quad_lock = quad.write().unwrap();

            let mut state_changes = Vec::new();
            quad_lock.states.iter().for_each(|(voxel, state)| {
                if state.cell == BACKGROUND_CELL {
                    return;
                }

                for neighbor in voxel.von_neumann_neighborhood() {
                    if !neighbor.is_oob()
                        && quad_lock.voxel_in_bounds(neighbor)
                        && quad_lock.get_voxel_cell(neighbor) == BACKGROUND_CELL
                    {
                        state_changes.push((neighbor, state.cell));
                    }
                }
            });

            let mut rng = rng();
            state_changes.shuffle(&mut rng);

            for (neighbor, cell) in state_changes {
                quad_lock.set_voxel_cell(neighbor, cell);
            }
        });
    }

    pub fn expand_cells_n(&mut self, n: usize) {
        for _ in 0..n {
            self.expand_cells();
        }
        self.build_edge_sets();
    }

    pub fn pop_bubbles(&mut self) {
        self.quads.par_iter().for_each(|(_quad_pos, quad)| {
            let mut quad_lock = quad.write().unwrap();
            let mut bubbles = HashSet::new();

            for edge in quad_lock.mismatch_edges.iter() {
                if let Some((cell, neighbor_cell)) = quad_lock.is_bubble(edge.a) {
                    if cell == BACKGROUND_CELL {
                        bubbles.insert((edge.a, neighbor_cell));
                    }
                } else if let Some((cell, neighbor_cell)) = quad_lock.is_bubble(edge.b) {
                    if cell == BACKGROUND_CELL {
                        bubbles.insert((edge.b, neighbor_cell));
                    }
                }
            }

            for (voxel, neighbor_cell) in bubbles {
                quad_lock.update_voxel_cell(voxel, BACKGROUND_CELL, neighbor_cell);
            }
        });
    }

    pub fn merge_counts_deltas(&mut self, params: &mut ModelParams) {
        // Move any counts delta that is in the wrong quad to the correct one
        let t0 = Instant::now();
        for key in self.quads.keys() {
            let quad = &self.quads[key];
            let mut quad_lock = quad.write().unwrap();
            let quad_lock_ref = quad_lock.deref_mut();
            let (min_i, max_i, min_j, max_j) = quad_lock_ref.bounds();

            for (key, count_delta) in quad_lock_ref.counts_deltas.iter_mut() {
                let neighbor_key = self.quad_index(key.voxel);
                let neighbor_quad = &self.quads[&neighbor_key];
                let [i, j, _k] = key.voxel.coords();

                if i < min_i || i > max_i || j < min_j || j > max_j {
                    neighbor_quad
                        .write()
                        .unwrap()
                        .counts_deltas
                        .push((*key, *count_delta));
                    *count_delta = 0;
                }
            }
        }
        info!("merge counts cleanup: {:?}", t0.elapsed());

        // now we can update quads in parallel
        let t0 = Instant::now();
        self.quads.par_iter().for_each(|(_key, quad)| {
            let mut quad_lock = quad.write().unwrap();
            let quad_lock_ref = quad_lock.deref_mut();
            for (key, count_delta) in quad_lock_ref.counts_deltas.drain(..) {
                if count_delta != 0 {
                    quad_lock_ref
                        .counts
                        .entry(key)
                        .and_modify(|count| *count += count_delta)
                        .or_insert(count_delta);

                    let cell = quad_lock_ref
                        .states
                        .get(&key.voxel)
                        .map(|state| state.cell)
                        .unwrap_or(BACKGROUND_CELL);

                    let k = key.voxel.k();
                    if cell == BACKGROUND_CELL {
                        params.unassigned_counts[k as usize].add(key.gene as usize, count_delta);
                    } else {
                        let counts_c = params.counts.row(cell as usize);
                        counts_c
                            .write()
                            .add(CountMatRowKey::new(key.gene, k as u32), count_delta);
                    }
                }
            }
        });
        info!("merge counts merge: {:?}", t0.elapsed());
    }
}
