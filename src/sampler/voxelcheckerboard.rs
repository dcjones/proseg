// Data structures for maintaining a set of voxels each with an associated:
// sparse transcript vector.

use super::super::spatialdata_input::CellPolygons;
use super::connectivity::MooreConnectivityChecker;
use super::math::logistic;
use super::onlinestats::ScalarQuantileEstimator;
use super::polygons::{PolygonBuilder, union_all_into_multipolygon};
use super::runvec::RunVec;
use super::sampleset::SampleSet;
use super::shardedvec::ShardedVec;
use super::sparsemat::SparseMat;
use super::transcripts::{BACKGROUND_CELL, CellIndex, Transcript, TranscriptDataset};
use super::{CountMatRowKey, ModelParams};

use arrow::array::RecordBatch;
use arrow::csv;
use arrow::datatypes::{DataType, Field, Schema};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use geo::algorithm::{BoundingRect, Contains};
use geo::geometry::{MultiPolygon, Point, Polygon};
use half::f16;
use itertools::izip;
use log::info;
use log::trace;
use ndarray::{Array1, Array2, Zip};
use ndarray_npy::{ReadNpyExt, read_npy};

use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::rng;
use rand::seq::SliceRandom;
use rayon::iter::{
    IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelDrainFull, ParallelIterator,
};
use rstar::primitives::GeomWithData;
use rstar::{PointDistance, RTree};
use std::cell::RefCell;
use std::cmp::{Ordering, PartialOrd};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::f32;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem::drop;
use std::ops::Bound::{Excluded, Included};
use std::ops::{Add, DerefMut, Neg};
use std::sync::Arc;
use std::sync::{Mutex, RwLock, RwLockWriteGuard};
use std::time::Instant;
use thread_local::ThreadLocal;

pub type CellPolygon = MultiPolygon<f32>;
pub type CellPolygonLayers = Vec<(i32, CellPolygon)>;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct TranscriptMetadata {
    pub offset: VoxelOffset,
    pub cell: CellIndex,
    pub foreground: bool,
}

// Simple affine transform matrix to map pixel coordinates onto slide microns
#[derive(Debug, Clone, Copy)]
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

        // probably makes more sense to consider the pixel center
        let (i, j) = (i + 0.5, j + 0.5);

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

    pub fn dk(&self) -> i32 {
        i8_to_i32(self.offset & 0xFF)
    }

    pub fn coords(&self) -> [i32; 3] {
        [self.di(), self.dj(), self.dk()]
    }
}

impl Neg for VoxelOffset {
    type Output = VoxelOffset;

    fn neg(self) -> VoxelOffset {
        VoxelOffset::new(-self.di(), -self.dj(), -self.dk())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_offset_zero() {
        let offset = VoxelOffset::zero();
        assert_eq!(offset.coords(), [0, 0, 0]);
    }

    #[test]
    fn test_voxel_offset_positive_values() {
        let offset = VoxelOffset::new(100, 200, 50);
        assert_eq!(offset.coords(), [100, 200, 50]);
    }

    #[test]
    fn test_voxel_offset_negative_values() {
        let offset = VoxelOffset::new(-100, -200, -50);
        assert_eq!(offset.coords(), [-100, -200, -50]);
    }

    #[test]
    fn test_voxel_offset_mixed_values() {
        let offset = VoxelOffset::new(-100, 200, -50);
        assert_eq!(offset.coords(), [-100, 200, -50]);

        let offset2 = VoxelOffset::new(100, -200, 50);
        assert_eq!(offset2.coords(), [100, -200, 50]);
    }

    #[test]
    fn test_voxel_offset_12bit_boundaries() {
        // Test 12-bit signed integer boundaries for di and dj
        // Range: -2048 to 2047

        // Maximum positive values
        let offset_max = VoxelOffset::new(2047, 2047, 127);
        assert_eq!(offset_max.coords(), [2047, 2047, 127]);

        // Maximum negative values
        let offset_min = VoxelOffset::new(-2048, -2048, -128);
        assert_eq!(offset_min.coords(), [-2048, -2048, -128]);

        // Test edge cases around zero
        let offset_pos_one = VoxelOffset::new(1, 1, 1);
        assert_eq!(offset_pos_one.coords(), [1, 1, 1]);

        let offset_neg_one = VoxelOffset::new(-1, -1, -1);
        assert_eq!(offset_neg_one.coords(), [-1, -1, -1]);
    }

    #[test]
    fn test_voxel_offset_8bit_boundaries() {
        // Test 8-bit signed integer boundaries for dk
        // Range: -128 to 127

        let offset_max_k = VoxelOffset::new(0, 0, 127);
        assert_eq!(offset_max_k.coords(), [0, 0, 127]);

        let offset_min_k = VoxelOffset::new(0, 0, -128);
        assert_eq!(offset_min_k.coords(), [0, 0, -128]);
    }

    #[test]
    fn test_voxel_offset_addition() {
        let offset1 = VoxelOffset::new(10, 20, 5);
        let offset2 = VoxelOffset::new(5, -15, 3);
        let result = offset1 + offset2;
        assert_eq!(result.coords(), [15, 5, 8]);
    }

    #[test]
    fn test_voxel_offset_addition_with_negatives() {
        let offset1 = VoxelOffset::new(-10, -20, -5);
        let offset2 = VoxelOffset::new(-5, 15, -3);
        let result = offset1 + offset2;
        assert_eq!(result.coords(), [-15, -5, -8]);
    }

    #[test]
    fn test_voxel_offset_individual_accessors() {
        let offset = VoxelOffset::new(123, -456, 78);
        assert_eq!(offset.di(), 123);
        assert_eq!(offset.dj(), -456);
        assert_eq!(offset.dk(), 78);
    }

    #[test]
    fn test_i12_to_i32_conversion() {
        // Test positive values
        assert_eq!(i12_to_i32(100), 100);
        assert_eq!(i12_to_i32(2047), 2047);

        // Test negative values (with sign bit set)
        assert_eq!(i12_to_i32(0xFFF), -1); // All bits set = -1
        assert_eq!(i12_to_i32(0x800), -2048); // Sign bit only = -2048
        assert_eq!(i12_to_i32(0x801), -2047); // Sign bit + 1 = -2047
    }

    #[test]
    fn test_i8_to_i32_conversion() {
        // Test positive values
        assert_eq!(i8_to_i32(100), 100);
        assert_eq!(i8_to_i32(127), 127);

        // Test negative values (with sign bit set)
        assert_eq!(i8_to_i32(0xFF), -1); // All bits set = -1
        assert_eq!(i8_to_i32(0x80), -128); // Sign bit only = -128
        assert_eq!(i8_to_i32(0x81), -127); // Sign bit + 1 = -127
    }

    #[test]
    fn test_voxel_offset_bit_packing() {
        // Test that values are correctly packed and unpacked
        let offset = VoxelOffset::new(0x123, 0x456, 0x78);

        // Verify individual components
        assert_eq!(offset.di(), 0x123);
        assert_eq!(offset.dj(), 0x456);
        assert_eq!(offset.dk(), 0x78);

        // Verify the packed representation
        let expected_packed =
            ((0x123u32 & 0xFFF) << 20) | ((0x456u32 & 0xFFF) << 8) | (0x78u32 & 0xFF);
        assert_eq!(offset.offset, expected_packed);
    }

    #[test]
    fn test_voxel_offset_sign_extension() {
        // Test that negative values are correctly sign-extended

        // Test di with negative value
        let offset1 = VoxelOffset::new(-1, 0, 0);
        assert_eq!(offset1.di(), -1);

        // Test dj with negative value
        let offset2 = VoxelOffset::new(0, -1, 0);
        assert_eq!(offset2.dj(), -1);

        // Test dk with negative value
        let offset3 = VoxelOffset::new(0, 0, -1);
        assert_eq!(offset3.dk(), -1);
    }

    #[test]
    fn test_voxel_offset_roundtrip() {
        // Test a variety of values to ensure perfect roundtrip storage/recovery
        let test_cases = vec![
            (0, 0, 0),
            (1, 1, 1),
            (-1, -1, -1),
            (100, -200, 50),
            (-100, 200, -50),
            (2047, 2047, 127),
            (-2048, -2048, -128),
            (1000, -1000, 100),
            (-1000, 1000, -100),
        ];

        for (di, dj, dk) in test_cases {
            let offset = VoxelOffset::new(di, dj, dk);
            let recovered = offset.coords();
            assert_eq!(
                recovered,
                [di, dj, dk],
                "Roundtrip failed for ({di}, {dj}, {dk}): got {recovered:?}",
            );
        }
    }

    #[test]
    fn test_voxel_offset_basic() {
        // Test basic offset functionality with simple cases
        let voxel = Voxel::new(100, 200, 50);

        // Test zero offset - should return same voxel
        let offset_zero = VoxelOffset::new(0, 0, 0);
        let result = voxel.offset(offset_zero);
        assert_eq!(result.coords(), [100, 200, 50]);
        assert!(!result.is_oob());

        // Test positive offsets
        let offset_pos = VoxelOffset::new(1, 2, 3);
        let result = voxel.offset(offset_pos);
        assert_eq!(result.coords(), [101, 202, 53]);
        assert!(!result.is_oob());

        // Test negative offsets
        let offset_neg = VoxelOffset::new(-5, -10, -15);
        let result = voxel.offset(offset_neg);
        assert_eq!(result.coords(), [95, 190, 35]);
        assert!(!result.is_oob());

        // Test mixed offsets
        let offset_mixed = VoxelOffset::new(50, -25, 10);
        let result = voxel.offset(offset_mixed);
        assert_eq!(result.coords(), [150, 175, 60]);
        assert!(!result.is_oob());
    }

    #[test]
    fn test_voxel_offset_moore_neighbors() {
        // Test that offset correctly computes Moore neighborhood (3x3x3 cube minus center)
        let center = Voxel::new(500, 600, 100);

        for &(di, dj, dk) in MOORE_OFFSETS.iter() {
            let offset = VoxelOffset::new(di, dj, dk);
            let neighbor = center.offset(offset);
            let expected = [500 + di, 600 + dj, 100 + dk];
            assert_eq!(
                neighbor.coords(),
                expected,
                "Moore neighbor offset ({di}, {dj}, {dk}) failed"
            );
            assert!(!neighbor.is_oob());
        }
    }

    #[test]
    fn test_voxel_offset_boundary_conditions() {
        // Test offset behavior near coordinate boundaries

        // Test near origin
        let near_origin = Voxel::new(1, 1, 1);
        let offset_neg = VoxelOffset::new(-1, -1, -1);
        let result = near_origin.offset(offset_neg);
        assert_eq!(result.coords(), [0, 0, 0]);
        assert!(!result.is_oob());

        // Test at origin going negative (should be OOB)
        let origin = Voxel::new(0, 0, 0);
        let result = origin.offset(offset_neg);
        assert!(result.is_oob());

        // Test near maximum k boundary
        let max_k = (1 << 16) - 1; // 65535
        let near_max_k = Voxel::new(100, 100, max_k - 1);
        let offset_pos_k = VoxelOffset::new(0, 0, 1);
        let result = near_max_k.offset(offset_pos_k);
        assert_eq!(result.coords(), [100, 100, max_k]);
        assert!(!result.is_oob());

        // Test at maximum k boundary going positive (should be OOB)
        let at_max_k = Voxel::new(100, 100, max_k);
        let result = at_max_k.offset(offset_pos_k);
        assert!(result.is_oob());
    }

    #[test]
    fn test_voxel_offset_out_of_bounds() {
        // Test various scenarios that should result in OOB voxels

        // Test negative coordinates
        let voxel = Voxel::new(5, 5, 5);
        let large_neg = VoxelOffset::new(-10, 0, 0);
        let result = voxel.offset(large_neg);
        assert!(result.is_oob());

        let large_neg_j = VoxelOffset::new(0, -10, 0);
        let result = voxel.offset(large_neg_j);
        assert!(result.is_oob());

        let large_neg_k = VoxelOffset::new(0, 0, -10);
        let result = voxel.offset(large_neg_k);
        assert!(result.is_oob());

        // Test coordinates that would exceed maximum bounds
        let max_coord_24bit = (1 << 24) - 1; // For i and j coordinates
        let max_coord_16bit = (1 << 16) - 1; // For k coordinate

        let near_max_i = Voxel::new(max_coord_24bit - 1, 100, 100);
        let large_pos_i = VoxelOffset::new(10, 0, 0);
        let result = near_max_i.offset(large_pos_i);
        assert!(result.is_oob());

        let near_max_j = Voxel::new(100, max_coord_24bit - 1, 100);
        let large_pos_j = VoxelOffset::new(0, 10, 0);
        let result = near_max_j.offset(large_pos_j);
        assert!(result.is_oob());

        let near_max_k = Voxel::new(100, 100, max_coord_16bit - 1);
        let large_pos_k = VoxelOffset::new(0, 0, 10);
        let result = near_max_k.offset(large_pos_k);
        assert!(result.is_oob());
    }

    #[test]
    fn test_voxel_offset_large_offsets() {
        // Test with larger offset values within VoxelOffset limits
        let center = Voxel::new(10000, 10000, 1000);

        // Test maximum positive VoxelOffset values
        let max_offset = VoxelOffset::new(2047, 2047, 127);
        let result = center.offset(max_offset);
        assert_eq!(result.coords(), [12047, 12047, 1127]);
        assert!(!result.is_oob());

        // Test maximum negative VoxelOffset values
        let min_offset = VoxelOffset::new(-2048, -2048, -128);
        let result = center.offset(min_offset);
        assert_eq!(result.coords(), [7952, 7952, 872]);
        assert!(!result.is_oob());
    }

    #[test]
    fn test_voxel_offset_consistency_with_coords() {
        // Test that offset() and offset_coords() produce identical results
        let voxel = Voxel::new(1234, 5678, 999);

        let test_offsets = [
            (0, 0, 0),
            (1, 2, 3),
            (-1, -2, -3),
            (100, -50, 25),
            (-100, 50, -25),
        ];

        for (di, dj, dk) in test_offsets {
            let offset_obj = VoxelOffset::new(di, dj, dk);
            let result_offset = voxel.offset(offset_obj);
            let result_coords = voxel.offset_coords(di, dj, dk);

            assert_eq!(
                result_offset.coords(),
                result_coords.coords(),
                "offset() and offset_coords() disagree for ({di}, {dj}, {dk})"
            );
            assert_eq!(
                result_offset.is_oob(),
                result_coords.is_oob(),
                "OOB status differs between offset() and offset_coords() for ({di}, {dj}, {dk})",
            );
        }
    }

    #[test]
    fn test_voxel_offset_chaining() {
        // Test that multiple offsets can be chained correctly
        let start = Voxel::new(1000, 2000, 500);

        // Apply a series of offsets
        let offset1 = VoxelOffset::new(10, 20, 5);
        let offset2 = VoxelOffset::new(-5, 15, -2);
        let offset3 = VoxelOffset::new(100, -50, 25);

        let result = start.offset(offset1).offset(offset2).offset(offset3);

        // Should be equivalent to applying the sum of all offsets
        let total_offset = VoxelOffset::new(10 + (-5) + 100, 20 + 15 + (-50), 5 + (-2) + 25);
        let expected = start.offset(total_offset);

        assert_eq!(result.coords(), expected.coords());
        assert_eq!(result.is_oob(), expected.is_oob());
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

// pub const VON_NEUMANN_OFFSETS: [(i32, i32, i32); 6] = [
//     (-1, 0, 0),
//     (1, 0, 0),
//     (0, -1, 0),
//     (0, 1, 0),
//     (0, 0, -1),
//     (0, 0, 1),
// ];

// pub const RADIUS2_OFFSETS: [(i32, i32, i32); 14] = [
//     (0, -2, 0),
//     (-1, -1, 0),
//     (0, -1, 0),
//     (1, -1, 0),
//     (-2, 0, 0),
//     (-1, 0, 0),
//     (1, 0, 0),
//     (2, 0, 0),
//     (-1, 1, 0),
//     (0, 1, 0),
//     (1, 1, 0),
//     (0, 2, 0),
//     (0, 0, -1),
//     (0, 0, 1),
// ];

// pub const SELF_RADIUS2_2D_OFFSETS: [(i32, i32, i32); 13] = [
//     (0, 0, 0),
//     (0, -2, 0),
//     (-1, -1, 0),
//     (0, -1, 0),
//     (1, -1, 0),
//     (-2, 0, 0),
//     (-1, 0, 0),
//     (1, 0, 0),
//     (2, 0, 0),
//     (-1, 1, 0),
//     (0, 1, 0),
//     (1, 1, 0),
//     (0, 2, 0),
// ];

pub const SELF_RADIUS3_2D_OFFSETS: [(i32, i32, i32); 25] = [
    (0, 0, 0),
    (0, -3, 0),
    (-1, -2, 0),
    (0, -2, 0),
    (1, -2, 0),
    (-2, -1, 0),
    (-1, -1, 0),
    (0, -1, 0),
    (1, -1, 0),
    (2, -1, 0),
    (-3, 0, 0),
    (-2, 0, 0),
    (-1, 0, 0),
    (1, 0, 0),
    (2, 0, 0),
    (3, 0, 0),
    (-2, 1, 0),
    (-1, 1, 0),
    (0, 1, 0),
    (1, 1, 0),
    (2, 1, 0),
    (-1, 2, 0),
    (0, 2, 0),
    (1, 2, 0),
    (0, 3, 0),
];

// pub const RADIUS3_OFFSETS: [(i32, i32, i32); 26] = [
//     (0, -3, 0),
//     (-1, -2, 0),
//     (0, -2, 0),
//     (1, -2, 0),
//     (-2, -1, 0),
//     (-1, -1, 0),
//     (0, -1, 0),
//     (1, -1, 0),
//     (2, -1, 0),
//     (-3, 0, 0),
//     (-2, 0, 0),
//     (-1, 0, 0),
//     (1, 0, 0),
//     (2, 0, 0),
//     (3, 0, 0),
//     (-2, 1, 0),
//     (-1, 1, 0),
//     (0, 1, 0),
//     (1, 1, 0),
//     (2, 1, 0),
//     (-1, 2, 0),
//     (0, 2, 0),
//     (1, 2, 0),
//     (0, 3, 0),
//     (0, 0, -1),
//     (0, 0, 1),
// ];

pub const MOORE_OFFSETS: [(i32, i32, i32); 26] = [
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
];

// pub const VON_NEUMANN_AND_SELF_OFFSETS: [(i32, i32, i32); 7] = [
//     (0, 0, 0),
//     (-1, 0, 0),
//     (1, 0, 0),
//     (0, -1, 0),
//     (0, 1, 0),
//     (0, 0, -1),
//     (0, 0, 1),
// ];

// pub const RADIUS2_AND_SELF_OFFSETS: [(i32, i32, i32); 15] = [
//     (0, 0, 0),
//     (0, -2, 0),
//     (-1, -1, 0),
//     (0, -1, 0),
//     (1, -1, 0),
//     (-2, 0, 0),
//     (-1, 0, 0),
//     (1, 0, 0),
//     (2, 0, 0),
//     (-1, 1, 0),
//     (0, 1, 0),
//     (1, 1, 0),
//     (0, 2, 0),
//     (0, 0, -1),
//     (0, 0, 1),
// ];

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
            index: ((i as u64 & 0xFFFFFF) << 40)
                | ((j as u64 & 0xFFFFFF) << 16)
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
        ((self.index >> 40) & 0xFFFFFF) as i32
    }

    pub fn j(&self) -> i32 {
        ((self.index >> 16) & 0xFFFFFF) as i32
    }

    pub fn k(&self) -> i32 {
        (self.index & 0xFFFF) as i32
    }

    pub fn setk(&self, k: i32) -> Voxel {
        let new_index = (self.index & !(0xFFFF)) | (k as u64);
        Voxel { index: new_index }
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

    pub fn z_neighborhood(&self) -> [Voxel; 2] {
        let [i, j, k] = self.coords();
        [Voxel::new(i, j, k - 1), Voxel::new(i, j, k + 1)]
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

    // pub fn radius2_neighborhood(&self) -> [Voxel; 14] {
    //     let [i, j, k] = self.coords();
    //     [
    //         (0, -2, 0),
    //         (-1, -1, 0),
    //         (0, -1, 0),
    //         (1, -1, 0),
    //         (-2, 0, 0),
    //         (-1, 0, 0),
    //         (1, 0, 0),
    //         (2, 0, 0),
    //         (-1, 1, 0),
    //         (0, 1, 0),
    //         (1, 1, 0),
    //         (0, 2, 0),
    //         (0, 0, -1),
    //         (0, 0, 1),
    //     ]
    //     .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    // }

    pub fn von_neumann_neighborhood_xy(&self) -> [Voxel; 4] {
        let [i, j, k] = self.coords();
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
            .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    }

    pub fn moore2d_neighborhood(&self) -> [Voxel; 8] {
        let [i, j, k] = self.coords();
        [
            (-1, 0, 0),
            (1, 0, 0),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
        ]
        .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    }

    // pub fn moorish_neighborhood(&self) -> [Voxel; 10] {
    //     let [i, j, k] = self.coords();
    //     [
    //         (-1, 0, 0),
    //         (1, 0, 0),
    //         (-1, 1, 0),
    //         (0, 1, 0),
    //         (1, 1, 0),
    //         (-1, -1, 0),
    //         (0, -1, 0),
    //         (1, -1, 0),
    //         (0, 0, -1),
    //         (0, 0, 1),
    //     ]
    //     .map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    // }

    pub fn moore_neighborhood(&self) -> [Voxel; 26] {
        let [i, j, k] = self.coords();
        MOORE_OFFSETS.map(|(di, dj, dk)| Voxel::new(i + di, j + dj, k + dk))
    }

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
        let xi = ((xor >> 40) & 0xFFFFFF) as u32;
        let xj = ((xor >> 16) & 0xFFFFFF) as u32;
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

// Ok, so each key is 8 + 8 + 4 = 20 bytes
#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub struct VoxelCountKey {
    pub voxel: Voxel,
    pub gene: GeneIndex,
    pub offset: VoxelOffset,
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

pub struct QuadStates {
    // Voxel state and prior.
    pub states: BTreeMap<Voxel, VoxelState>,
    pub mismatch_edges: SampleSet<UndirectedVoxelPair>,
}

impl QuadStates {
    pub fn new() -> QuadStates {
        QuadStates {
            states: BTreeMap::new(),
            mismatch_edges: SampleSet::new(),
        }
    }

    pub fn get_voxel_state(&self, voxel: Voxel) -> Option<&VoxelState> {
        self.states.get(&voxel)
    }

    pub fn set_voxel_cell(&mut self, voxel: Voxel, cell: CellIndex) {
        if cell == BACKGROUND_CELL {
            let mut remove = false;
            if let Some(state) = self.states.get_mut(&voxel) {
                // Don't remove the state if it's being used to store a prior
                if state.log_prior.is_finite() {
                    state.cell = cell;
                } else {
                    remove = true;
                }
            }
            if remove {
                self.states.remove(&voxel);
            }
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
    pub fn is_bubble(&self, quad: &VoxelQuad, voxel: Voxel) -> Option<(CellIndex, CellIndex)> {
        if !quad.voxel_in_bounds(voxel) {
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

    pub fn update_voxel_cell(
        &mut self,
        quad: &VoxelQuad,
        voxel: Voxel,
        current_cell: CellIndex,
        proposed_cell: CellIndex,
    ) {
        self.set_voxel_cell(voxel, proposed_cell);

        for neighbor in voxel.von_neumann_neighborhood() {
            let k = neighbor.k();
            if k < 0 || k > quad.kmax {
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

pub struct QuadCounts {
    // This is essentially one giant sparse matrix for the entire
    // voxel set. We also have to keep track of repositioned transcripts here.
    pub counts: BTreeMap<VoxelCountKey, u32>,

    pub counts_deltas: Vec<(VoxelCountKey, u32)>,
}

impl QuadCounts {
    pub fn new() -> QuadCounts {
        QuadCounts {
            counts: BTreeMap::new(),
            counts_deltas: Vec::new(),
        }
    }
}

impl<'a> QuadCounts {
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

// This will represent one square of the checkerboard
pub struct VoxelQuad {
    pub states: RwLock<QuadStates>,

    // This is essentially one giant sparse matrix for the entire
    // voxel set. We also have to keep track of repositioned transcripts here.
    pub counts: RwLock<QuadCounts>,

    // Local transcript density, used for noise rate estimation
    pub densities: RwLock<BTreeMap<Voxel, f32>>,

    // Allocates some matrices to be re-used for connectivity checks
    pub connectivity: RwLock<MooreConnectivityChecker>,

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
            states: RwLock::new(QuadStates::new()),
            counts: RwLock::new(QuadCounts::new()),
            densities: RwLock::new(BTreeMap::new()),
            connectivity: RwLock::new(MooreConnectivityChecker::new()),
            kmax,
            quadsize,
            u,
            v,
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
}

impl VoxelQuad {}

pub struct VoxelCheckerboard {
    // number of voxels in each x/y direction
    pub quadsize: usize,

    // number of cells (excluding any initialized without voxels)
    pub ncells: usize,

    // number of genes
    pub ngenes: usize,

    // number of voxel z layers
    nzlayers: usize,

    // number of transcript density bins
    pub density_nbins: usize,

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

    // map cells indexes after initialization to their indexed prior to removing usused cells
    pub used_cells_map: Vec<u32>,

    // Main thing is we'll need to look up arbitrary Voxels,
    // which means first looking up which VoxelSet this is in.
    //
    // We also need to keep track of whether indexe parities to
    // do staggered updates. So how do we want to organize this.
    pub quads: HashMap<(u32, u32), VoxelQuad>,

    // Set of keys in `quads`. This is obviously redundant, but a
    // contrivance to avoid multiple borrows of quads in some places.
    pub quads_coords: HashSet<(u32, u32)>,

    // [ncells] False where morphology updates are prohibited.
    pub frozen_cells: Vec<bool>,

    // [nxpixels, nypixels] If initialized using a auxiliary frozen mask, we neeed to
    // store it here so we can re-initialize at a higher resolution when `increase_resuliton` is called.
    frozen_masks: Option<Array2<u32>>,
    frozen_masks_transform: Option<PixelTransform>,
}

impl VoxelCheckerboard {
    #[allow(clippy::too_many_arguments)]
    fn empty(
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        density_nbins: usize,
        ngenes: usize,
        xmin: f32,
        ymin: f32,
        zmin: f32,
    ) -> VoxelCheckerboard {
        let voxelsize_z = 1.0 / nzlayers as f32;
        let voxel_volume = voxelsize * voxelsize * voxelsize_z;
        VoxelCheckerboard {
            quadsize: (quadsize / voxelsize).round().max(1.0) as usize,
            kmax: (nzlayers - 1) as i32,
            ncells: 0,
            ngenes,
            nzlayers,
            density_nbins,
            voxel_volume,
            xmin,
            ymin,
            zmin,
            voxelsize,
            voxelsize_z,
            used_cells_map: Vec::new(),
            quads: HashMap::new(),
            quads_coords: HashSet::new(),
            frozen_cells: Vec::new(),
            frozen_masks: None,
            frozen_masks_transform: None,
        }
    }

    fn initialize_counts(&mut self, dataset: &TranscriptDataset) {
        let t0 = Instant::now();
        for run in dataset.transcripts.iter_runs() {
            let transcript = &run.value;
            let voxel = self.coords_to_voxel(transcript.x, transcript.y, transcript.z);
            let key = VoxelCountKey {
                voxel,
                gene: transcript.gene,
                offset: VoxelOffset::zero(),
            };

            let mut quad_counts = self.write_quad_counts(voxel);
            let count = quad_counts.counts.entry(key).or_insert(0_u32);
            *count += run.len;
        }
        trace!("assigned voxel counts: {:?}", t0.elapsed());
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_prior_transcript_assignments(
        dataset: &TranscriptDataset,
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        nucprior: f32,
        cellprior: f32,
        expansion: usize,
        density_bandwidth: f32,
        density_nbins: usize,
    ) -> VoxelCheckerboard {
        let (xmin, _xmax, ymin, _ymax, zmin, _zmax) = dataset.coordinate_span();

        let mut checkerboard = VoxelCheckerboard::empty(
            voxelsize,
            quadsize,
            nzlayers,
            density_nbins,
            dataset.ngenes(),
            xmin,
            ymin,
            zmin,
        );

        // tally votes: count transcripts in each voxel aggregated by prior cell assignment
        let t0 = Instant::now();
        let mut nuc_votes: BTreeMap<(Voxel, CellIndex), u32> = BTreeMap::new();
        let mut cell_votes: BTreeMap<(Voxel, CellIndex), u32> = BTreeMap::new();
        for (transcript, prior) in dataset.transcripts.iter().zip(dataset.priorseg.iter()) {
            let voxel = checkerboard.coords_to_voxel(transcript.x, transcript.y, transcript.z);

            if prior.nucleus != BACKGROUND_CELL {
                let key = (voxel, prior.nucleus);
                *nuc_votes.entry(key).or_insert(0) += 1;
            }

            if prior.cell != BACKGROUND_CELL {
                let key = (voxel, prior.cell);
                *cell_votes.entry(key).or_insert(0) += 1;
            }
        }

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
        checkerboard.frozen_cells = vec![false; checkerboard.ncells];
        checkerboard.used_cells_map.resize(checkerboard.ncells, 0);
        for (old_cell_id, new_cell_id) in used_cells.iter() {
            checkerboard.used_cells_map[*new_cell_id as usize] = *old_cell_id;
        }

        // re-assign cell indices so that there are no cells without any assigned voxel
        for quad in &mut checkerboard.quads.values() {
            let mut quad_states = quad.states.write().unwrap();
            for state in quad_states.states.values_mut() {
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

        checkerboard.finish_initialization(dataset, expansion, density_bandwidth, density_nbins);
        checkerboard
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_visium_barcode_mappings(
        dataset: &mut TranscriptDataset,
        barcode_mappings_filename: &str,
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        nucprior: f32,
        expansion: usize,
        density_bandwidth: f32,
        density_nbins: usize,
    ) -> VoxelCheckerboard {
        let (xmin, _xmax, ymin, _ymax, zmin, _zmax) = dataset.coordinate_span();
        let zmid = dataset.z_mean();
        let log_nucprior = f16::from_f32(nucprior.ln());
        let log_1m_nucprior = f16::from_f32((1.0 - nucprior).ln());

        let mut checkerboard = VoxelCheckerboard::empty(
            voxelsize,
            quadsize,
            nzlayers,
            density_nbins,
            dataset.ngenes(),
            xmin,
            ymin,
            zmin,
        );

        let barcode_positions = dataset.barcode_positions.as_ref().unwrap();

        let barcode_mapping_file = File::open(barcode_mappings_filename)
            .unwrap_or_else(|_| panic!("Unable to open '{}'.", &barcode_mappings_filename));
        let builder = ParquetRecordBatchReaderBuilder::try_new(barcode_mapping_file).unwrap();
        let schema = builder.schema().as_ref().clone();

        let rdr = builder.build().unwrap_or_else(|_| {
            panic!("Unable to read parquet data from {barcode_mappings_filename}")
        });

        let barcode_col_idx = schema.index_of("square_002um").unwrap();
        let cell_id_col_idx = schema.index_of("cell_id").unwrap();
        let in_nucleus_col_idx = schema.index_of("in_nucleus").unwrap();
        let mut cell_id_map = HashMap::new();

        for rec_batch in rdr {
            let rec_batch = rec_batch.expect("Unable to read record batch.");

            let barcodes = rec_batch
                .column(barcode_col_idx)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();

            let cell_ids = rec_batch
                .column(cell_id_col_idx)
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();

            let in_nucleus = rec_batch
                .column(in_nucleus_col_idx)
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .unwrap();

            for (barcode, cell_id_str, in_nuc) in izip!(barcodes, cell_ids, in_nucleus) {
                if !in_nuc.unwrap() {
                    continue;
                }

                if let Some(cell_id_str) = cell_id_str {
                    let (x, y) = barcode_positions[barcode.unwrap()];

                    let voxel = checkerboard.coords_to_voxel(x, y, zmid);

                    if !voxel.is_oob() {
                        checkerboard.insert_state_if_missing(voxel, || {
                            let next_cell_id = cell_id_map.len() as CellIndex;
                            let cell_id = *cell_id_map
                                .entry(cell_id_str.to_string())
                                .or_insert(next_cell_id);
                            VoxelState {
                                cell: cell_id,
                                prior_cell: cell_id,
                                log_prior: log_nucprior,
                                log_1m_prior: log_1m_nucprior,
                            }
                        });
                    }
                }
            }
        }

        checkerboard.ncells = cell_id_map.len();
        checkerboard.frozen_cells = vec![false; checkerboard.ncells];

        dataset.original_cell_ids = vec![String::new(); cell_id_map.len()];
        for (cell_id, i) in cell_id_map {
            dataset.original_cell_ids[i as usize] = cell_id;
        }
        checkerboard.used_cells_map = (0..checkerboard.ncells as u32).collect();

        checkerboard.finish_initialization(dataset, expansion, density_bandwidth, density_nbins);
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

    fn tally_seg_mask_pixels(
        &self,
        masks: &Array2<u32>,
        cellprobs: &Array2<f32>,
        frozen_masks: &Array2<u32>,
        pixel_transform: &PixelTransform,
        cell_votes: &mut BTreeMap<(Voxel, CellIndex, bool), f32>,
        zmid: f32,
    ) {
        Zip::indexed(masks)
            .and(frozen_masks)
            .and(cellprobs)
            .for_each(|(i, j), &masks_ij, &frozen_masks_ij, &cellprobs_ij| {
                if masks_ij == 0 && frozen_masks_ij == 0 {
                    return;
                }

                let (x, y) = pixel_transform.transform(j, i);
                let voxel = self.coords_to_voxel(x, y, zmid);

                if masks_ij != 0 {
                    let vote = cell_votes.entry((voxel, masks_ij, false)).or_insert(0.0);
                    *vote += cellprobs_ij;
                } else if frozen_masks_ij != 0 {
                    let vote = cell_votes
                        .entry((voxel, frozen_masks_ij, true))
                        .or_insert(0.0);
                    *vote += 1.0;
                }
            });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_seg_masks(
        dataset: &mut TranscriptDataset,
        masks_filename: &str,
        cellprob_filename: &Option<String>,
        cellprob_discount: f32,
        frozen_masks_filename: &Option<String>,
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        pixel_transform: &PixelTransform,
        cellprior: f32,
        expansion: usize,
        density_bandwidth: f32,
        density_nbins: usize,
    ) -> VoxelCheckerboard {
        let masks: Array2<u32> = Self::read_npy_gz(masks_filename).unwrap_or_else(
            |_err| panic!("Unable to read cell masks from {masks_filename}. Make sure it an npy file (possibly gzipped) containing a uint32 matrix")
        );
        info!(
            "Read {} by {} cell segmentation mask.",
            masks.shape()[0],
            masks.shape()[1]
        );

        let cellprobs = if let Some(cellprob_filename) = cellprob_filename {
            let mut cellprobs: Array2<f32> = Self::read_npy_gz(cellprob_filename).unwrap_or_else(
                |_err| panic!("Unable to read cellpose cellprob from {cellprob_filename}. Make sure it an npy file (possibly gzipped) containing a float32 matrix")
            );
            if cellprobs.shape() != masks.shape() {
                panic!("Cellprobs must have the same shape as the segmentation mask");
            }

            // transform to probs, discount, and scale everything to be in [0.5, 1.0]
            cellprobs.map_inplace(|v| *v = logistic(*v));
            cellprobs *= 0.5 * cellprob_discount;
            cellprobs += 0.5;
            cellprobs
        } else {
            Array2::from_elem((masks.shape()[0], masks.shape()[1]), cellprior)
        };

        let mut frozen_masks = if let Some(frozen_masks_filename) = frozen_masks_filename {
            let frozen_masks: Array2<u32> = Self::read_npy_gz(frozen_masks_filename).unwrap_or_else(
                |_err| panic!("Unable to read fixed cell masks from {masks_filename}. Make sure it an npy file (possibly gzipped) containing a uint32 matrix")
            );
            if frozen_masks.shape() != masks.shape() {
                panic!("Fixed cell masks must have the same shape as the segmentation mask");
            }

            frozen_masks
        } else {
            Array2::zeros((masks.shape()[0], masks.shape()[1]))
        };

        let (xmin, _xmax, ymin, _ymax, zmin, _zmax) = dataset.coordinate_span();
        let zmid = dataset.z_mean();

        let mut checkerboard = VoxelCheckerboard::empty(
            voxelsize,
            quadsize,
            nzlayers,
            density_nbins,
            dataset.ngenes(),
            xmin,
            ymin,
            zmin,
        );

        let t0 = Instant::now();
        let mut cell_votes: BTreeMap<(Voxel, CellIndex, bool), f32> = BTreeMap::new();

        checkerboard.tally_seg_mask_pixels(
            &masks,
            &cellprobs,
            &frozen_masks,
            pixel_transform,
            &mut cell_votes,
            zmid,
        );

        trace!("Voting on voxel states: {:?}", t0.elapsed());

        // save memory where we can
        drop(masks);
        drop(cellprobs);

        let t0 = Instant::now();
        let pixels_per_voxel = pixel_transform.det().abs().recip();
        let mut used_cells = HashMap::new();
        let mut current_voxel = Voxel::oob();
        let mut vote_winner = (BACKGROUND_CELL, false);
        let mut vote_winner_prior_sum: f32 = 0.0;
        for ((voxel, cell, frozen), prior_sum) in cell_votes {
            if voxel != current_voxel {
                if !current_voxel.is_oob() {
                    let prior = (vote_winner_prior_sum / pixels_per_voxel).min(0.99);
                    let next_cell_id = used_cells.len() as CellIndex;
                    let cell_id = *used_cells.entry(vote_winner).or_insert(next_cell_id);
                    checkerboard.insert_state(
                        current_voxel,
                        VoxelState {
                            cell: cell_id,
                            prior_cell: cell_id,
                            log_prior: f16::from_f32(prior.ln()),
                            log_1m_prior: f16::from_f32((1.0 - prior).ln()),
                        },
                    );
                }
                vote_winner = (cell, frozen);
                vote_winner_prior_sum = prior_sum;
                current_voxel = voxel;
            } else if prior_sum > vote_winner_prior_sum {
                vote_winner = (cell, frozen);
                vote_winner_prior_sum = prior_sum;
            }
        }
        trace!("Set voxel states: {:?}", t0.elapsed());

        if !current_voxel.is_oob() {
            let prior = (vote_winner_prior_sum / pixels_per_voxel).min(0.99);
            let next_cell_id = used_cells.len() as CellIndex;
            let cell_id = *used_cells.entry(vote_winner).or_insert(next_cell_id);
            checkerboard.insert_state(
                current_voxel,
                VoxelState {
                    cell: cell_id,
                    prior_cell: cell_id,
                    log_prior: f16::from_f32(prior.ln()),
                    log_1m_prior: f16::from_f32((1.0 - prior).ln()),
                },
            );
        }

        if frozen_masks_filename.is_some() {
            // Rewriting the frozen segmentation mask to use assigned cell ids so we can more easily re-use
            // it when changing voxel resolution.
            frozen_masks.iter_mut().for_each(|value| {
                *value = used_cells
                    .get(&(*value, true))
                    .cloned()
                    .unwrap_or(BACKGROUND_CELL);
            });
            checkerboard.frozen_masks = Some(frozen_masks);
            checkerboard.frozen_masks_transform = Some(*pixel_transform);
        } else {
            drop(frozen_masks);
        }

        checkerboard.ncells = used_cells.len();
        checkerboard.frozen_cells = vec![false; checkerboard.ncells];
        for (&(_original_cell_id, frozen), &cell_id) in used_cells.iter() {
            checkerboard.frozen_cells[cell_id as usize] = frozen;
        }

        // Rewrite original ids
        let mut cell_id_pairs: Vec<_> = used_cells
            .iter()
            .map(|((original_cell_id, frozen), cell_id)| (*cell_id, *original_cell_id, *frozen))
            .collect();
        cell_id_pairs.sort();
        dataset.original_cell_ids.clear();
        dataset.original_cell_ids.extend(cell_id_pairs.iter().map(
            |&(_, original_cell_id, frozen)| {
                if frozen {
                    format!("{original_cell_id}-fixed")
                } else {
                    original_cell_id.to_string()
                }
            },
        ));
        checkerboard.used_cells_map = (0..checkerboard.ncells as u32).collect();

        // TODO: I don't think any of this stuff is necessary anymore, but let's make sure.
        // let mut minprior = f32::INFINITY;
        // let mut maxprior = f32::NEG_INFINITY;
        // checkerboard.quads.values().for_each(|quad| {
        //     let mut quad_states = quad.states.write().unwrap();

        //     // TODO: I don't think we should have to do this.
        //     //
        //     // re-assign cell indices so that there are no cells without any assigned voxel
        //     for state in quad_states.states.values_mut() {
        //         let cell = *used_cells.get(&state.cell).unwrap();
        //         state.cell = cell;
        //         state.prior_cell = cell;

        //         let prior = state.log_prior.to_f32().exp();
        //         minprior = minprior.min(prior);
        //         maxprior = maxprior.max(prior);
        //     }

        //     // TODO: Aren't we doing this already when we finalize???
        //     // copy the state along the z-axis
        //     let states_2d = quad_states.states.clone();
        //     for (voxel, state) in states_2d {
        //         let [i, j, k] = voxel.coords();
        //         for k_down in (0..k).rev() {
        //             quad_states.states.insert(Voxel::new(i, j, k_down), state);
        //         }

        //         for k_up in (k + 1)..(quad.kmax + 1) {
        //             quad_states.states.insert(Voxel::new(i, j, k_up), state);
        //         }
        //     }
        // });
        //

        checkerboard.finish_initialization(dataset, expansion, density_bandwidth, density_nbins);
        checkerboard
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_cell_polygons(
        dataset: &mut TranscriptDataset,
        cell_polygons: &CellPolygons,
        voxelsize: f32,
        quadsize: f32,
        nzlayers: usize,
        prior: f32,
        expansion: usize,
        density_bandwidth: f32,
        density_nbins: usize,
    ) -> VoxelCheckerboard {
        let (mut xmin, _xmax, mut ymin, _ymax, zmin, _zmax) = dataset.coordinate_span();
        let (xmin_poly, _xmax_poly, ymin_poly, _ymax_poly) = cell_polygons.bounding_box();
        xmin = xmin.min(xmin_poly);
        ymin = ymin.min(ymin_poly);
        let log_prior = f16::from_f32(prior.ln());
        let log_1m_prior = f16::from_f32((1.0 - prior).ln());

        let mut checkerboard = VoxelCheckerboard::empty(
            voxelsize,
            quadsize,
            nzlayers,
            density_nbins,
            dataset.ngenes(),
            xmin,
            ymin,
            zmin,
        );

        // TODO: Maybe we can shuffle the order of the polygons and do this more efficiently parallel.
        for (cell_id, cell_polygon) in cell_polygons
            .cells
            .iter()
            .zip(cell_polygons.polygons.iter())
        {
            let bounding_rect = cell_polygon.bounding_rect().unwrap();
            let (poly_x_min, poly_y_min) = bounding_rect.min().x_y();
            let (poly_x_max, poly_y_max) = bounding_rect.max().x_y();

            // converting the bounding box to voxel coordinates
            let poly_i_min = ((poly_x_min - xmin) / voxelsize).floor().max(0.0) as i32;
            let poly_i_max = ((poly_x_max - xmin) / voxelsize).ceil().max(0.0) as i32;
            let poly_j_min = ((poly_y_min - ymin) / voxelsize).floor().max(0.0) as i32;
            let poly_j_max = ((poly_y_max - ymin) / voxelsize).ceil().max(0.0) as i32;

            // test every voxel in the bounding box to see if it intersects the cell's polygon
            for i in poly_i_min..poly_i_max + 1 {
                let x = xmin + (0.5 + i as f32) * voxelsize;
                for j in poly_j_min..poly_j_max {
                    let y = ymin + (0.5 + j as f32) * voxelsize;
                    if cell_polygon.contains(&Point::new(x, y)) {
                        // Note: this can overwrite an existing state
                        checkerboard.insert_state(
                            Voxel::new(i, j, 0),
                            VoxelState {
                                cell: *cell_id,
                                prior_cell: *cell_id,
                                log_prior,
                                log_1m_prior,
                            },
                        );
                    }
                }
            }
        }

        // figure out which cells we are actually using
        let mut used_cells = HashMap::new();
        checkerboard.quads.values().for_each(|quad| {
            for (_voxel, state) in quad.states.read().unwrap().states.iter() {
                if state.cell != BACKGROUND_CELL {
                    let next_cell_id = used_cells.len() as u32;
                    used_cells.entry(state.cell).or_insert(next_cell_id);
                }
            }
        });
        checkerboard.ncells = used_cells.len();
        checkerboard.frozen_cells = vec![false; checkerboard.ncells];

        // reassign cell ids
        checkerboard.used_cells_map.resize(checkerboard.ncells, 0);
        dataset.original_cell_ids = vec![String::new(); checkerboard.ncells];
        for (old_cell_id, new_cell_id) in used_cells.iter() {
            checkerboard.used_cells_map[*new_cell_id as usize] = *old_cell_id;
            dataset.original_cell_ids[*new_cell_id as usize] =
                cell_polygons.original_cell_ids[*old_cell_id as usize].clone();
        }
        checkerboard.used_cells_map = (0..checkerboard.ncells as u32).collect();

        // re-assign cell indices so that there are no cells without any assigned voxel
        for quad in &mut checkerboard.quads.values() {
            let mut quad_states = quad.states.write().unwrap();
            for state in quad_states.states.values_mut() {
                let cell = *used_cells.get(&state.cell).unwrap();
                state.cell = cell;
                state.prior_cell = cell;
            }
        }

        checkerboard.finish_initialization(dataset, expansion, density_bandwidth, density_nbins);
        checkerboard
    }

    fn finish_initialization(
        &mut self,
        dataset: &TranscriptDataset,
        expansion: usize,
        density_bandwidth: f32,
        density_nbins: usize,
    ) {
        self.initialize_counts(dataset);
        self.quads_coords.extend(self.quads.keys());
        self.estimate_local_transcript_density(dataset, density_bandwidth, density_nbins);
        self.expand_cells_n(expansion);
        self.expand_cells_vertically(false);
        self.pop_bubbles();
        self.mirror_quad_edges();
        self.build_edge_sets();
    }

    fn quad_index(&self, voxel: Voxel) -> (u32, u32) {
        let u = voxel.i() as u32 / self.quadsize as u32;
        let v = voxel.j() as u32 / self.quadsize as u32;
        (u, v)
    }

    fn write_quad_states(&mut self, voxel: Voxel) -> RwLockWriteGuard<QuadStates> {
        self.write_quad_index_states(self.quad_index(voxel))
    }

    fn write_quad_index_states(&mut self, index: (u32, u32)) -> RwLockWriteGuard<QuadStates> {
        let (u, v) = index;

        self.quads
            .entry(index)
            .or_insert_with(|| VoxelQuad::new(self.kmax, self.quadsize, u, v))
            .states
            .write()
            .unwrap()
    }

    fn write_quad_counts(&mut self, voxel: Voxel) -> RwLockWriteGuard<QuadCounts> {
        self.write_quad_index_counts(self.quad_index(voxel))
    }

    fn write_quad_index_counts(&mut self, index: (u32, u32)) -> RwLockWriteGuard<QuadCounts> {
        let (u, v) = index;

        self.quads
            .entry(index)
            .or_insert_with(|| VoxelQuad::new(self.kmax, self.quadsize, u, v))
            .counts
            .write()
            .unwrap()
    }

    pub fn get_voxel_cell(&self, voxel: Voxel) -> CellIndex {
        self.quads
            .get(&self.quad_index(voxel))
            .map(|quad| quad.states.read().unwrap().get_voxel_cell(voxel))
            .unwrap_or(BACKGROUND_CELL)
    }

    pub fn get_voxel_density(&self, voxel: Voxel) -> usize {
        let voxel = voxel.setk(0);
        self.quads
            .get(&self.quad_index(voxel))
            .map(|quad| quad.densities.read().unwrap()[&voxel] as usize)
            .unwrap()
    }

    // Same as get_voxel_density, but faster when the voxel is probably in the given quad
    pub fn get_voxel_density_hint(&self, quad: &VoxelQuad, voxel: Voxel) -> usize {
        let voxel = voxel.setk(0);
        if quad.voxel_in_bounds(voxel) {
            quad.densities.read().unwrap()[&voxel] as usize
        } else {
            self.get_voxel_density(voxel)
        }
    }

    fn coords_to_voxel(&self, x: f32, y: f32, z: f32) -> Voxel {
        let i = ((x - self.xmin) / self.voxelsize).floor().max(0.0) as i32;
        let j = ((y - self.ymin) / self.voxelsize).floor().max(0.0) as i32;
        let k = (((z - self.zmin) / self.voxelsize_z) as i32)
            .min(self.nzlayers as i32 - 1)
            .max(0);
        Voxel::new(i, j, k)
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
        self.write_quad_states(voxel).states.insert(voxel, state);
    }

    fn insert_state_if_missing(&mut self, voxel: Voxel, f: impl FnOnce() -> VoxelState) {
        self.write_quad_states(voxel)
            .states
            .entry(voxel)
            .or_insert_with(f);
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
            let quad_states = quad.states.read().unwrap();
            self.for_each_quad_neighbor_states(u, v, |neighbor_quad, neighbor_quad_states| {
                let (min_i, max_i, min_j, max_j) = neighbor_quad.bounds();
                for (voxel, state) in &quad_states.states {
                    let [i, j, _k] = voxel.coords();
                    if (min_i - 1..max_i + 2).contains(&i) && (min_j - 1..max_j + 2).contains(&j) {
                        neighbor_quad_states.states.insert(*voxel, *state);
                    }
                }
            });
        }
    }

    pub fn check_mirrored_quad_edges(&self) {
        for (&(u, v), quad) in &self.quads {
            let quad_states = quad.states.read().unwrap();
            self.for_each_quad_neighbor_states(u, v, |neighbor_quad, neighbor_quad_states| {
                let (min_i, max_i, min_j, max_j) = neighbor_quad.bounds();
                for (voxel, state) in &quad_states.states {
                    let [i, j, _k] = voxel.coords();

                    if (min_i - 1..max_i + 2).contains(&i) && (min_j - 1..max_j + 2).contains(&j) {
                        if state.cell == BACKGROUND_CELL
                            && !neighbor_quad_states.states.contains_key(voxel)
                        {
                            continue;
                        }

                        let mirrored_state = neighbor_quad_states.states.get(voxel).unwrap();
                        assert!(mirrored_state.cell == state.cell);
                        assert!(mirrored_state.prior_cell == state.prior_cell);
                    }
                }
            });
        }
    }

    pub fn check_mismatch_edges(&self) {
        let mut mismatch_edges = Vec::new();
        let mut mismatch_edge_set = SampleSet::new();
        for quad in self.quads.values() {
            let quad_states = quad.states.read().unwrap();

            mismatch_edges.clear();
            self.build_quad_edge_sets(&quad_states, &mut mismatch_edges);

            mismatch_edge_set.clear();
            mismatch_edge_set.extend(&mismatch_edges);

            assert!(quad_states.mismatch_edges == mismatch_edge_set);
        }
    }

    fn build_edge_sets(&mut self) {
        // have to do this to get around a double borrow issue
        let mut mismatch_edges = Vec::new();
        for quad in self.quads.values() {
            let mut quad_states = quad.states.write().unwrap();
            self.build_quad_edge_sets(&quad_states, &mut mismatch_edges);

            quad_states.mismatch_edges.clear();
            quad_states.mismatch_edges.extend(&mismatch_edges);
        }
    }

    fn build_quad_edge_sets(
        &self,
        quad_states: &QuadStates,
        mismatch_edges: &mut Vec<UndirectedVoxelPair>,
    ) {
        mismatch_edges.clear();
        for (&voxel, state) in &quad_states.states {
            let cell = state.cell;
            if cell == BACKGROUND_CELL {
                continue;
            }

            for neighbor in voxel.von_neumann_neighborhood() {
                let k = neighbor.k();
                if k < 0 || k > self.kmax {
                    continue;
                }

                let neighbor_cell = quad_states
                    .states
                    .get(&neighbor)
                    .map_or(BACKGROUND_CELL, |state| state.cell);

                if cell != neighbor_cell {
                    mismatch_edges.push(UndirectedVoxelPair::new(voxel, neighbor));
                }
            }
        }
    }

    pub fn for_each_quad_neighbor_states<F>(&self, u: u32, v: u32, f: F)
    where
        F: Fn(&VoxelQuad, &mut QuadStates),
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
                f(
                    neighbor_quad,
                    neighbor_quad.states.write().unwrap().deref_mut(),
                )
            }
        }
    }

    fn estimate_local_transcript_density(
        &mut self,
        dataset: &TranscriptDataset,
        bandwidth: f32,
        nbins: usize,
    ) {
        let mut rtree = RTree::new();
        for run in dataset.transcripts.iter_runs() {
            rtree.insert(GeomWithData::new([run.value.x, run.value.y], run.len));
        }

        let bandwidth_sq = bandwidth * bandwidth;
        let kernel_norm = 1.0 / (bandwidth * (2.0 * f32::consts::PI).sqrt());
        let eps = 1e-3_f32;
        let max_distance = -2.0 * bandwidth_sq * eps.ln();
        self.quads.par_iter_mut().for_each(|((_u, _v), quad)| {
            let counts = quad.counts.read().unwrap();
            let mut densities = quad.densities.write().unwrap();
            counts.counts.iter().for_each(|(count_key, _count)| {
                let [i, j, _k] = count_key.voxel.coords();
                let x = ((i as f32) + 0.5) * self.voxelsize + self.xmin;
                let y = ((j as f32) + 0.5) * self.voxelsize + self.ymin;

                let voxel = count_key.voxel.setk(0);
                for (di, dj, dk) in SELF_RADIUS3_2D_OFFSETS {
                    let neighbor = voxel.offset_coords(di, dj, dk);
                    densities.entry(neighbor).or_insert_with(|| {
                        let mut voxel_density = 0.0;

                        for neighbor in rtree.locate_within_distance([x, y], max_distance) {
                            let d2 = neighbor.distance_2(&[x, y]);
                            let count = neighbor.data;
                            voxel_density += (count as f32) * (-d2 / (2.0 * bandwidth_sq)).exp();
                        }
                        voxel_density / kernel_norm
                    });
                }
            });
        });

        let mut quant_est = (1..nbins + 1)
            .map(|i| ScalarQuantileEstimator::new((i as f32) / (nbins as f32)))
            .collect::<Vec<_>>();

        self.quads.iter().for_each(|((_u, _v), quad)| {
            let densities = quad.densities.read().unwrap();
            for (_voxel, &density) in densities.iter() {
                for quant_est_q in quant_est.iter_mut() {
                    quant_est_q.update(density);
                }
            }
        });

        let quants = quant_est
            .iter()
            .map(|quant_est_q| quant_est_q.estimate())
            .collect::<Vec<_>>();

        info!("Density quantiles: {quants:?}");

        // Replace densities with their quantiles
        self.quads.par_iter_mut().for_each(|((_u, _v), quad)| {
            let mut densities = quad.densities.write().unwrap();
            for (_voxel, density) in densities.iter_mut() {
                let mut quantile_index = 0;
                for (idx, &quant) in quants.iter().enumerate() {
                    if quant <= *density {
                        quantile_index = idx + 1;
                    } else {
                        break;
                    }
                }
                quantile_index = quantile_index.min(quants.len() - 1);
                *density = quantile_index as f32;
            }
        });
    }

    pub fn compute_cell_volume_surface_area(
        &self,
        volume: &mut ShardedVec<u32>,
        layer_volume: &mut [ShardedVec<u32>],
        layer_surface_area: &mut [ShardedVec<u32>],
    ) {
        volume.zero();
        layer_volume.iter_mut().for_each(|v_k| v_k.zero());
        layer_surface_area.iter_mut().for_each(|a_k| a_k.zero());

        self.quads.par_iter().for_each(|((_u, _v), quad)| {
            let quad_states = quad.states.read().unwrap();
            for (&voxel, state) in &quad_states.states {
                // We mirror state on the border. We need to skip these mirrored voxels to avoid double-counting.
                if !quad.voxel_in_bounds(voxel) {
                    continue;
                }

                let k = voxel.k() as usize;

                if state.cell != BACKGROUND_CELL {
                    volume.add(state.cell as usize, 1);
                    layer_volume[k].add(state.cell as usize, 1);

                    let mut voxel_surface_area = 0;
                    for neighbor in voxel.moore2d_neighborhood() {
                        let k = neighbor.k();
                        let neighbor_cell = if k >= 0 && k <= quad.kmax {
                            quad_states.get_voxel_cell(neighbor)
                        } else {
                            BACKGROUND_CELL
                        };

                        if state.cell != neighbor_cell {
                            voxel_surface_area += 1;
                        }
                    }
                    layer_surface_area[k].add(state.cell as usize, voxel_surface_area);
                }
            }
        });
    }

    pub fn compute_counts(
        &self,
        counts: &mut SparseMat<u32, CountMatRowKey>,
        unassigned_counts: &mut [Vec<ShardedVec<u32>>],
    ) {
        counts.zero();
        unassigned_counts.iter_mut().for_each(|c_d| {
            c_d.iter_mut().for_each(|c_ld| {
                c_ld.zero();
            })
        });

        self.quads.par_iter().for_each(|((_u, _v), quad)| {
            let quad_states = quad.states.read().unwrap();
            let quad_counts = quad.counts.read().unwrap();
            for (
                &VoxelCountKey {
                    voxel,
                    gene,
                    offset,
                },
                &count,
            ) in &quad_counts.counts
            {
                let origin = voxel.offset(-offset);
                let k_origin = origin.k();
                let density = self.get_voxel_density_hint(quad, origin);

                let cell = quad_states.get_voxel_cell(voxel);
                if cell != BACKGROUND_CELL {
                    counts.row(cell as usize).write().add(
                        CountMatRowKey::new(gene, k_origin as u32, density as u8),
                        count,
                    );
                } else {
                    unassigned_counts[density][k_origin as usize].add(gene as usize, count);
                }
            }
        });
    }

    pub fn compute_background_region_volumes(&self, background_region_volume: &mut Array1<f32>) {
        background_region_volume.fill(0.0);

        let mut background_region_voxel_count =
            Array1::<u32>::zeros(background_region_volume.len());

        for ((_u, _v), quad) in self.quads.iter() {
            let densities = quad.densities.read().unwrap();
            for (_voxel, &density) in densities.iter() {
                background_region_voxel_count[density as usize] += 1;
            }
        }

        Zip::from(background_region_volume)
            .and(&background_region_voxel_count)
            .for_each(|volume, &voxel_count| {
                *volume = self.voxel_volume * (voxel_count as f32);
            });
    }

    pub fn cell_centroids(&self, params: &ModelParams) -> Array2<f32> {
        let mut centroids = Array2::zeros((self.ncells, 3));

        self.quads.values().for_each(|quad| {
            let quad_states = quad.states.read().unwrap();
            let (i_min, i_max, j_min, j_max) = quad.bounds();
            for (voxel, state) in quad_states.states.iter() {
                if state.cell != BACKGROUND_CELL {
                    let [i, j, k] = voxel.coords();

                    // don't count mirrored edge states
                    if i < i_min || i > i_max || j < j_min || j > j_max {
                        continue;
                    }

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
            quad.states
                .read()
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
            let quad_states = quad.states.read().unwrap();
            let quad_counts = quad.counts.read().unwrap();

            quad_states.states.iter().for_each(|(&voxel, &state)| {
                let cell = state.cell;
                if cell == BACKGROUND_CELL || !quad.voxel_in_bounds(voxel) {
                    return;
                }
                let [i, j, _k] = voxel.coords();

                let transcript_count = quad_counts
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

    // Construct transcript metadata by matching observed transcripts up to voxelized counts.
    pub fn transcript_metadata(
        &self,
        params: &ModelParams,
        transcripts: &RunVec<u32, Transcript>,
    ) -> RunVec<u32, TranscriptMetadata> {
        // Clone the count structures so we can decrement transcripts as they are encountered
        // We also have to re-index by the observed voxel in order to look up transcripts by their
        // observed position.
        let mut counts = BTreeMap::new();
        self.quads.iter().for_each(|(_quad_index, quad)| {
            quad.counts
                .read()
                .unwrap()
                .counts
                .iter()
                .for_each(|(key, count)| {
                    // Rebuild the hash map indexing on observed voxel
                    let mut newkey = *key;
                    newkey.voxel = key.voxel.offset(-key.offset);
                    counts.insert(newkey, *count);
                });
        });

        // Similarly, we need to clone foreground counts so we can keep track
        let mut foreground_counts: HashMap<(u32, u32), u32> = HashMap::new();
        for row in params.foreground_counts.rows() {
            let row_lock = row.read();
            let cell = row.i;
            for (gene, count) in row_lock.iter_nonzeros() {
                if count > 0 {
                    foreground_counts.insert((cell as CellIndex, gene), count);
                }
            }
        }

        let mut metadata = RunVec::new();
        for transcript in transcripts.iter() {
            let voxel = self.coords_to_voxel(transcript.x, transcript.y, transcript.z);

            let from = VoxelCountKey {
                voxel,
                gene: transcript.gene,
                offset: VoxelOffset::zero(),
            };

            let to = VoxelCountKey {
                voxel,
                gene: transcript.gene + 1,
                offset: VoxelOffset::zero(),
            };

            // look for foreground count
            let mut key_match = None;
            let mut cell = BACKGROUND_CELL;
            for (key, count) in counts.range_mut((Included(from), Excluded(to))) {
                if *count == 0 {
                    continue;
                }
                let voxel_cell = self.get_voxel_cell(voxel.offset(key.offset));
                if let Some(c) = foreground_counts.get_mut(&(voxel_cell, transcript.gene)) {
                    if *c > 0 {
                        *c -= 1;
                        key_match = Some(*key);
                        cell = voxel_cell;
                        break;
                    }
                }
            }

            if let Some(key) = key_match {
                counts.entry(key).and_modify(|c| *c -= 1);
                metadata.push(TranscriptMetadata {
                    offset: key.offset,
                    cell,
                    foreground: true,
                });
            } else {
                // non for a matching non-background -count
                let mut found = false;
                for (key, count) in counts.range_mut((Included(from), Excluded(to))) {
                    if *count > 0 {
                        *count -= 1;
                        found = true;
                        metadata.push(TranscriptMetadata {
                            offset: key.offset,
                            cell: BACKGROUND_CELL,
                            foreground: false,
                        });
                        break;
                    }
                }
                if !found {
                    panic!("Unable to find a matching transcript. Inconsistent count structure.");
                }
            }
        }

        let remaining_counts = counts.values().map(|v| *v as usize).sum::<usize>();
        let remaining_foreground_counts = foreground_counts
            .values()
            .map(|v| *v as usize)
            .sum::<usize>();

        assert!(remaining_counts == 0);
        assert!(remaining_foreground_counts == 0);

        metadata
    }

    fn expand_cells_vertically(&mut self, only_frozen: bool) {
        for _ in 0..self.nzlayers - 1 {
            self.quads.par_iter().for_each(|(_quad_pos, quad)| {
                let mut quad_states = quad.states.write().unwrap();

                let mut state_changes = Vec::new();
                quad_states.states.iter().for_each(|(voxel, state)| {
                    if state.cell == BACKGROUND_CELL {
                        return;
                    }

                    if only_frozen && !self.frozen_cells[state.cell as usize] {
                        return;
                    }

                    for neighbor in voxel.z_neighborhood() {
                        if !neighbor.is_oob()
                            && quad.voxel_in_bounds(neighbor)
                            && quad_states.get_voxel_cell(neighbor) == BACKGROUND_CELL
                        {
                            state_changes.push((neighbor, state.cell));
                        }
                    }
                });

                let mut rng = rng();
                state_changes.shuffle(&mut rng);

                for (neighbor, cell) in state_changes {
                    quad_states.set_voxel_cell(neighbor, cell);
                }
            });
        }
    }

    // Copy occupied voxel states to unoccupied neighbors
    fn expand_cells(&mut self) {
        self.quads.par_iter().for_each(|(_quad_pos, quad)| {
            let mut quad_states = quad.states.write().unwrap();

            let mut state_changes = Vec::new();
            quad_states.states.iter().for_each(|(voxel, state)| {
                if state.cell == BACKGROUND_CELL || self.frozen_cells[state.cell as usize] {
                    return;
                }

                for neighbor in voxel.von_neumann_neighborhood() {
                    if !neighbor.is_oob()
                        && quad.voxel_in_bounds(neighbor)
                        && quad_states.get_voxel_cell(neighbor) == BACKGROUND_CELL
                    {
                        state_changes.push((neighbor, state.cell));
                    }
                }
            });

            let mut rng = rng();
            state_changes.shuffle(&mut rng);

            for (neighbor, cell) in state_changes {
                quad_states.set_voxel_cell(neighbor, cell);
            }
        });
    }

    fn expand_cells_n(&mut self, n: usize) {
        for _ in 0..n {
            self.expand_cells();
        }
        self.build_edge_sets();
    }

    fn pop_bubbles(&mut self) {
        self.quads.par_iter().for_each(|(_quad_pos, quad)| {
            let mut quad_states = quad.states.write().unwrap();
            let mut bubbles = HashSet::new();

            for edge in quad_states.mismatch_edges.iter() {
                if let Some((cell, neighbor_cell)) = quad_states.is_bubble(quad, edge.a) {
                    if cell == BACKGROUND_CELL {
                        bubbles.insert((edge.a, neighbor_cell));
                    }
                } else if let Some((cell, neighbor_cell)) = quad_states.is_bubble(quad, edge.b) {
                    if cell == BACKGROUND_CELL {
                        bubbles.insert((edge.b, neighbor_cell));
                    }
                }
            }

            for (voxel, neighbor_cell) in bubbles {
                quad_states.update_voxel_cell(quad, voxel, BACKGROUND_CELL, neighbor_cell);
            }
        });
    }

    pub fn merge_counts_deltas(&mut self, params: &mut ModelParams) {
        // Move any counts delta that is in the wrong quad to the correct one
        let t0 = Instant::now();
        for key in self.quads.keys() {
            let quad = &self.quads[key];
            let mut quad_counts = quad.counts.write().unwrap();
            let quad_lock_ref = quad_counts.deref_mut();
            let (min_i, max_i, min_j, max_j) = quad.bounds();

            for (key, count_delta) in quad_lock_ref.counts_deltas.iter_mut() {
                let neighbor_key = self.quad_index(key.voxel);
                let neighbor_quad = &self.quads[&neighbor_key];
                let [i, j, _k] = key.voxel.coords();

                if i < min_i || i > max_i || j < min_j || j > max_j {
                    neighbor_quad
                        .counts
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
            let mut quad_counts = quad.counts.write().unwrap();
            let quad_states = quad.states.read().unwrap();
            let quad_lock_ref = quad_counts.deref_mut();
            for (key, count_delta) in quad_lock_ref.counts_deltas.drain(..) {
                if count_delta != 0 {
                    quad_lock_ref
                        .counts
                        .entry(key)
                        .and_modify(|count| *count += count_delta)
                        .or_insert(count_delta);

                    let cell = quad_states
                        .states
                        .get(&key.voxel)
                        .map(|state| state.cell)
                        .unwrap_or(BACKGROUND_CELL);

                    let origin = key.voxel.offset(-key.offset);
                    let k_origin = origin.k() as usize;
                    let density = self.get_voxel_density_hint(quad, origin);

                    if cell == BACKGROUND_CELL {
                        params.unassigned_counts[density][k_origin]
                            .add(key.gene as usize, count_delta);
                    } else {
                        let counts_c = params.counts.row(cell as usize);
                        counts_c.write().add(
                            CountMatRowKey::new(key.gene, k_origin as u32, density as u8),
                            count_delta,
                        );
                    }
                }
            }
        });
        info!("merge counts merge: {:?}", t0.elapsed());
    }

    // Scale the number of voxels on the x/y axis by `scale`
    pub fn increase_resolution(
        mut self,
        scale: usize,
        params: &mut ModelParams,
        dataset: &TranscriptDataset,
        density_bandwidth: f32,
        density_nbins: usize,
    ) -> VoxelCheckerboard {
        let quadsize = scale * self.quadsize;
        let voxelsize = (scale as f32).recip() * self.voxelsize;
        let voxelsize_z = self.voxelsize_z;
        let voxel_volume = voxelsize * voxelsize * voxelsize_z;

        let reinit_frozen = self.frozen_masks.is_some();

        let quads = Mutex::new(HashMap::new());
        self.quads.par_drain().for_each(|((u, v), old_quad)| {
            let new_quad = VoxelQuad::new(self.kmax, quadsize, u, v);
            {
                let old_quad_states = old_quad.states.read().unwrap();
                let mut new_quad_states = new_quad.states.write().unwrap();
                old_quad_states.states.iter().for_each(|(voxel, state)| {
                    if state.cell != BACKGROUND_CELL
                        && self.frozen_cells[state.cell as usize]
                        && reinit_frozen
                    {
                        return;
                    }

                    let [i, j, k] = voxel.coords();
                    for s in 0..scale as i32 {
                        for t in 0..scale as i32 {
                            let subvoxel =
                                Voxel::new(i * (scale as i32) + s, j * (scale as i32) + t, k);

                            // excluding mirrored edge states
                            if new_quad.voxel_in_bounds(subvoxel) {
                                new_quad_states.states.insert(subvoxel, *state);
                            }
                        }
                    }
                });
            }

            quads.lock().unwrap().insert((u, v), new_quad);
        });

        let quads = quads.into_inner().unwrap();
        let quads_coords = quads.keys().cloned().collect();

        let mut new_checkerboard = VoxelCheckerboard {
            quadsize,
            kmax: self.kmax,
            ncells: self.ncells,
            ngenes: dataset.ngenes(),
            nzlayers: self.nzlayers,
            density_nbins: self.density_nbins,
            voxel_volume,
            xmin: self.xmin,
            ymin: self.ymin,
            zmin: self.zmin,
            voxelsize,
            voxelsize_z,
            used_cells_map: self.used_cells_map,
            quads,
            quads_coords,
            frozen_cells: self.frozen_cells,
            frozen_masks: self.frozen_masks,
            frozen_masks_transform: self.frozen_masks_transform,
        };

        // If we are using an auxiliary frozen segmentation mask, we need to
        // reinitialize those cells at this higher resolution.
        if let Some(frozen_masks) = &new_checkerboard.frozen_masks {
            let pixel_transform = &new_checkerboard.frozen_masks_transform.unwrap();
            let pixels_per_voxel = pixel_transform.det().abs().recip();
            let zmid = new_checkerboard.zmin
                + (new_checkerboard.nzlayers as f32) * new_checkerboard.voxelsize_z;

            let mut cell_votes = BTreeMap::new();
            Zip::indexed(frozen_masks).for_each(|(i, j), &frozen_masks_ij| {
                if frozen_masks_ij == BACKGROUND_CELL {
                    return;
                }

                let (x, y) = pixel_transform.transform(j, i);
                let voxel = new_checkerboard.coords_to_voxel(x, y, zmid);
                let vote = cell_votes
                    .entry((voxel, frozen_masks_ij))
                    .or_insert(0.0_f32);
                *vote += 1_f32;
            });

            let mut current_voxel = Voxel::oob();
            let mut vote_winner = BACKGROUND_CELL;
            let mut vote_winner_prior_sum: f32 = 0.0;
            for ((voxel, cell), prior_sum) in cell_votes {
                if voxel != current_voxel {
                    if !current_voxel.is_oob() {
                        let prior = (vote_winner_prior_sum / pixels_per_voxel).min(0.99);
                        new_checkerboard.insert_state_if_missing(current_voxel, || VoxelState {
                            cell: vote_winner,
                            prior_cell: vote_winner,
                            log_prior: f16::from_f32(prior.ln()),
                            log_1m_prior: f16::from_f32((1.0 - prior).ln()),
                        });
                    }

                    vote_winner = cell;
                    vote_winner_prior_sum = prior_sum;
                    current_voxel = voxel;
                } else if prior_sum > vote_winner_prior_sum {
                    vote_winner = cell;
                    vote_winner_prior_sum = prior_sum;
                }
            }

            if !current_voxel.is_oob() {
                let prior = (vote_winner_prior_sum / pixels_per_voxel).min(0.99);
                new_checkerboard.insert_state_if_missing(current_voxel, || VoxelState {
                    cell: vote_winner,
                    prior_cell: vote_winner,
                    log_prior: f16::from_f32(prior.ln()),
                    log_1m_prior: f16::from_f32((1.0 - prior).ln()),
                });
            }

            new_checkerboard.expand_cells_vertically(true);
        }

        new_checkerboard.initialize_counts(dataset);

        // initialize edge voxel sets
        new_checkerboard.mirror_quad_edges();
        new_checkerboard.build_edge_sets();

        new_checkerboard.compute_cell_volume_surface_area(
            &mut params.cell_voxel_count,
            &mut params.cell_layer_voxel_count,
            &mut params.cell_layer_surface_area,
        );

        new_checkerboard.estimate_local_transcript_density(
            dataset,
            density_bandwidth,
            density_nbins,
        );

        params.voxel_volume = voxel_volume;

        // Need to recompute these count matrices because density values will have changed for some voxels.
        new_checkerboard.compute_counts(&mut params.counts, &mut params.unassigned_counts);
        new_checkerboard.compute_background_region_volumes(&mut params.background_region_volume);

        new_checkerboard
    }

    pub fn dump_counts(&self, transcripts: &TranscriptDataset, filename: &str) {
        let file = File::create(filename).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());

        // TODO: This isn't very useful unless we convert to slide coordinates.
        // What info do I need for that.

        let schema_fields = vec![
            Field::new("gene", DataType::Utf8, false),
            Field::new("count", DataType::UInt32, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("z", DataType::Float32, false),
            Field::new("dx", DataType::Float32, false),
            Field::new("dy", DataType::Float32, false),
            Field::new("dz", DataType::Float32, false),
        ];
        let schema = Schema::new(schema_fields);

        let mut gene_col = Vec::new();
        let mut count_col = Vec::new();
        let mut x_col = Vec::new();
        let mut y_col = Vec::new();
        let mut z_col = Vec::new();
        let mut dx_col = Vec::new();
        let mut dy_col = Vec::new();
        let mut dz_col = Vec::new();

        for quad in self.quads.values() {
            let quad_counts = quad.counts.read().unwrap();
            for (key, &count) in quad_counts.counts.iter() {
                if count == 0 {
                    continue;
                }

                let [i, j, k] = key.voxel.coords();
                let [di, dj, dk] = key.offset.coords();

                let x = ((i as f32) + 0.5) * self.voxelsize + self.xmin;
                let y = ((j as f32) + 0.5) * self.voxelsize + self.ymin;
                let z = ((k as f32) + 0.5) * self.voxelsize_z + self.zmin;

                let dx = (di as f32) * self.voxelsize;
                let dy = (dj as f32) * self.voxelsize;
                let dz = (dk as f32) * self.voxelsize_z;

                gene_col.push(Some(transcripts.gene_names[key.gene as usize].clone()));
                count_col.push(Some(count));
                x_col.push(Some(x));
                y_col.push(Some(y));
                z_col.push(Some(z));
                dx_col.push(Some(dx));
                dy_col.push(Some(dy));
                dz_col.push(Some(dz));
            }
        }

        let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(arrow::array::StringArray::from(gene_col)),
            Arc::new(arrow::array::UInt32Array::from(count_col)),
            Arc::new(arrow::array::Float32Array::from(x_col)),
            Arc::new(arrow::array::Float32Array::from(y_col)),
            Arc::new(arrow::array::Float32Array::from(z_col)),
            Arc::new(arrow::array::Float32Array::from(dx_col)),
            Arc::new(arrow::array::Float32Array::from(dy_col)),
            Arc::new(arrow::array::Float32Array::from(dz_col)),
        ];

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        let mut writer = csv::WriterBuilder::new().with_header(true).build(encoder);
        writer.write(&batch).unwrap();
    }
}
