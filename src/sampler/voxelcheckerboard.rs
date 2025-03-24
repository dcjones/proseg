// Data structures for maintaining a set of voxels each with an associated
// sparse transcript vector.

use super::transcripts::{CellIndex, BACKGROUND_CELL};

use std::cmp::{Ordering, PartialOrd};
use std::collections::{BTreeMap, HashSet};
use std::sync::RwLock;

// Store a voxel offset in compact form
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq, Hash)]
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
impl PartialOrd for Voxel {
    fn partial_cmp(&self, other: &Voxel) -> Option<Ordering> {
        fn less_msb(a: u32, b: u32) -> bool {
            a < b && a < (a ^ b)
        }

        if self.index == other.index {
            return Some(Ordering::Equal);
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
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

type GeneIndex = u32;

// TODO: we could probably get away with packing this into one u32,
// as well, but that probably gets a bit sketchier on visium and the like.
struct GeneCount {
    count: u32,
    noise: u32,
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
#[derive(Clone, Copy, Debug, PartialOrd, PartialEq)]
struct VoxelCountKey {
    voxel: Voxel,
    gene: GeneIndex,
    offset: VoxelOffset,
}

struct VoxelState {
    cell: CellIndex,
    prior_cell: CellIndex,
    prior: f32,
}

// This will represent one square of the checkerboard
struct VoxelSet {
    // Voxel state and prior.
    states: BTreeMap<Voxel, VoxelState>,

    // This is essentially one giant sparse matrix for the entire
    // voxel set. We also have to keep track of repositioned transcripts here.
    counts: BTreeMap<VoxelCountKey, GeneCount>,

    cell_edge_voxels: HashSet<Voxel>,
    bounds: (Voxel, Voxel),
}

impl VoxelSet {
    // Initialize empty voxel set
    fn new(from: Voxel, to: Voxel) -> VoxelSet {
        return VoxelSet {
            states: BTreeMap::new(),
            counts: BTreeMap::new(),
            cell_edge_voxels: HashSet::new(),
            bounds: (from, to),
        };
    }

    // TODO: initialize with

    // Maybe we wait to see what functionality we actually need here
}

struct VoxelCheckerboard {
    // number of voxels in each direction
    quad_x_size: usize,
    quad_y_size: usize,

    // Main thing is we'll need to look up arbitrary Voxels,
    // which means first looking up which VoxelSet this is in.
    //
    // We also need to keep track of whether indexe parities to
    // do staggered updates. So how do we want to organize this.
    voxel_sets: Vec<RwLock<VoxelSet>>,
}

// impl VoxelCheckerboard {
//     fn voxel_state(voxel: Voxel) -> CellIndex {
//         // TODO: lookup the quad
//         // TODO: linearize the voxel
//         // TODO: check the quad
//     }
// }

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
