use super::connectivity::ConnectivityChecker;
use super::math::relerr;
use super::polygons::{union_all_into_multipolygon, PolygonBuilder};
use super::sampleset::SampleSet;
use super::transcripts::{coordinate_span, CellIndex, Transcript, BACKGROUND_CELL};
use super::{chunkquad, perimeter_bound, ModelParams, ModelPriors, Proposal, Sampler};

// use hexx::{Hex, HexLayout, HexOrientation, Vec2};
// use arrow;
use geo::geometry::{MultiPolygon, Polygon};
use itertools::Itertools;
use ndarray::{Array2, Zip};
use rand::{rng, Rng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::{Ord, Ordering, PartialEq, PartialOrd};
use std::collections::{HashMap, HashSet};
use std::f32;
use std::sync::{Arc, Mutex};
use thread_local::ThreadLocal;

pub type CellPolygon = MultiPolygon<f32>;
pub type CellPolygonLayers = Vec<(i32, CellPolygon)>;

// use std::time::Instant;

fn clip_z_position(position: (f32, f32, f32), zmin: f32, zmax: f32) -> (f32, f32, f32) {
    let eps = (zmax - zmin) * 1e-6;
    (
        position.0,
        position.1,
        position.2.max(zmin + eps).min(zmax - eps),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Voxel {
    pub i: i32,
    pub j: i32,
    pub k: i32,
}

impl Voxel {
    pub fn new(i: i32, j: i32, k: i32) -> Voxel {
        Voxel { i, j, k }
    }

    fn default() -> Voxel {
        Voxel { i: 0, j: 0, k: 0 }
    }

    pub fn moore_neighborhood(&self) -> [Voxel; 26] {
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
        .map(|(di, dj, dk)| Voxel::new(self.i + di, self.j + dj, self.k + dk))
    }

    // pub fn moore_neighborhood_xy(&self) -> [Voxel; 8] {
    //     [
    //         (-1, 0, 0),
    //         (1, 0, 0),
    //         (-1, 1, 0),
    //         (0, 1, 0),
    //         (1, 1, 0),
    //         (-1, -1, 0),
    //         (0, -1, 0),
    //         (1, -1, 0),
    //     ]
    //     .map(|(di, dj, dk)| Voxel::new(self.i + di, self.j + dj, self.k + dk))
    // }

    pub fn von_neumann_neighborhood(&self) -> [Voxel; 6] {
        [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        .map(|(di, dj, dk)| Voxel::new(self.i + di, self.j + dj, self.k + dk))
    }

    pub fn von_neumann_neighborhood_xy(&self) -> [Voxel; 4] {
        [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0)]
            .map(|(di, dj, dk)| Voxel::new(self.i + di, self.j + dj, self.k + dk))
    }

    pub fn radius2_xy_neighborhood(&self) -> [Voxel; 12] {
        [
            (0, -2, 0),
            (-1, -1, 0),
            (0, -1, 0),
            (1, -1, 0),
            (-2, 0, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (2, 0, 0),
            (-1, 1, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 2, 0),
        ]
        .map(|(di, dj, dk)| Voxel::new(self.i + di, self.j + dj, self.k + dk))
    }

    fn double_resolution_and_layers_children(&self) -> [Voxel; 8] {
        [
            Voxel::new(2 * self.i, 2 * self.j, 2 * self.k),
            Voxel::new(2 * self.i + 1, 2 * self.j, 2 * self.k),
            Voxel::new(2 * self.i, 2 * self.j + 1, 2 * self.k),
            Voxel::new(2 * self.i + 1, 2 * self.j + 1, 2 * self.k),
            Voxel::new(2 * self.i, 2 * self.j, 2 * self.k + 1),
            Voxel::new(2 * self.i + 1, 2 * self.j, 2 * self.k + 1),
            Voxel::new(2 * self.i, 2 * self.j + 1, 2 * self.k + 1),
            Voxel::new(2 * self.i + 1, 2 * self.j + 1, 2 * self.k + 1),
        ]
    }

    fn double_resolution_children(&self) -> [Voxel; 4] {
        [
            Voxel::new(2 * self.i, 2 * self.j, self.k),
            Voxel::new(2 * self.i + 1, 2 * self.j, self.k),
            Voxel::new(2 * self.i, 2 * self.j + 1, self.k),
            Voxel::new(2 * self.i + 1, 2 * self.j + 1, self.k),
        ]
    }

    fn inbounds(&self, nlayers: usize) -> bool {
        self.k >= 0 && self.k < nlayers as i32
    }

    pub fn edge_xy(&self, other: &Voxel) -> ((i32, i32), (i32, i32)) {
        assert!(self.k == other.k);
        assert!((self.i - other.i).abs() + (self.j - other.j).abs() == 1);

        let i0 = self.i.max(other.i);
        let j0 = self.j.max(other.j);

        let i1 = i0 + (other.j - self.j).abs();
        let j1 = j0 + (other.i - self.i).abs();

        ((i0, j0), (i1, j1))
    }
}

impl PartialOrd for Voxel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Voxel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.k
            .cmp(&other.k)
            .then(self.j.cmp(&other.j))
            .then(self.i.cmp(&other.i))
    }
}

#[derive(Debug)]
pub struct VoxelLayout {
    origin: (f32, f32, f32),
    size: (f32, f32, f32),
}

impl VoxelLayout {
    fn double_resolution(&self) -> VoxelLayout {
        VoxelLayout {
            origin: (self.origin.0, self.origin.1, self.origin.2),
            size: (self.size.0 / 2.0, self.size.1 / 2.0, self.size.2),
        }
    }

    fn double_resolution_and_layers(&self) -> VoxelLayout {
        VoxelLayout {
            origin: (self.origin.0, self.origin.1, self.origin.2),
            size: (self.size.0 / 2.0, self.size.1 / 2.0, self.size.2 / 2.0),
        }
    }

    pub fn voxel_corner_to_world_pos(&self, voxel: Voxel) -> (f32, f32, f32) {
        (
            self.origin.0 + (voxel.i as f32) * self.size.0,
            self.origin.1 + (voxel.j as f32) * self.size.1,
            self.origin.2 + (voxel.k as f32) * self.size.2,
        )
    }

    fn voxel_to_world_pos(&self, voxel: Voxel) -> (f32, f32, f32) {
        (
            self.origin.0 + (0.5 + voxel.i as f32) * self.size.0,
            self.origin.1 + (0.5 + voxel.j as f32) * self.size.1,
            self.origin.2 + (0.5 + voxel.k as f32) * self.size.2,
        )
    }

    fn world_pos_to_voxel(&self, pos: (f32, f32, f32)) -> Voxel {
        Voxel::new(
            ((pos.0 - self.origin.0) / self.size.0).floor() as i32,
            ((pos.1 - self.origin.1) / self.size.1).floor() as i32,
            ((pos.2 - self.origin.2) / self.size.2).floor() as i32,
        )
    }

    fn voxel_to_world_coords(&self, voxel: Voxel) -> (f32, f32, f32, f32, f32, f32) {
        let x0 = self.origin.0 + voxel.i as f32 * self.size.0;
        let y0 = self.origin.1 + voxel.j as f32 * self.size.1;
        let z0 = self.origin.2 + voxel.k as f32 * self.size.2;

        (
            x0,
            y0,
            z0,
            x0 + self.size.0,
            y0 + self.size.1,
            z0 + self.size.2,
        )
    }
}

type VoxelEdgeSampleSet = SampleSet<(Voxel, Voxel)>;

#[derive(Clone, Debug)]
struct VoxelBin {
    voxel: Voxel,
    transcripts: Arc<Mutex<Vec<usize>>>,
}

struct ChunkQuadMap {
    layout: VoxelLayout,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,
}

impl ChunkQuadMap {
    fn get(&self, voxel: Voxel) -> (u32, u32) {
        let voxel_xyz = self.layout.voxel_to_world_pos(voxel);
        chunkquad(
            voxel_xyz.0,
            voxel_xyz.1,
            self.xmin,
            self.ymin,
            self.chunk_size,
            self.nxchunks,
        )
    }
}

struct VoxelCellMap {
    index: HashMap<Voxel, CellIndex>,
}

impl VoxelCellMap {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    fn insert(&mut self, voxel: Voxel, cell: CellIndex) {
        self.index.insert(voxel, cell);
    }

    fn get(&self, voxel: Voxel) -> CellIndex {
        match self.index.get(&voxel) {
            Some(cell) => *cell,
            None => BACKGROUND_CELL,
        }
    }

    fn set(&mut self, voxel: Voxel, cell: CellIndex) {
        self.index.insert(voxel, cell);
    }

    // fn len(&self) -> usize {
    //     return self.index.len();
    // }

    // fn count(&self, cell: u32) -> usize {
    //     return self.index.values().filter(|&&c| c == cell).count();
    // }

    fn iter(&self) -> std::collections::hash_map::Iter<Voxel, CellIndex> {
        return self.index.iter();
    }
}

// Initial binning of the transcripts
fn bin_transcripts(
    transcripts: &Vec<Transcript>,
    scale: f32,
    zlayers: usize,
) -> (VoxelLayout, Vec<VoxelBin>) {
    let (_, _, _, _, zmin, zmax) = coordinate_span(transcripts);

    let mut height = zmax - zmin;
    if height == 0.0 {
        height = 1.0;
    }

    let voxel_height = height / zlayers as f32;

    let voxel_size = scale;
    let layout = VoxelLayout {
        origin: (0.0, 0.0, zmin),
        size: (voxel_size, voxel_size, voxel_height),
    };

    let mut voxel_transcripts = transcripts
        .par_iter()
        .enumerate()
        .map(|(i, t)| {
            let position = clip_z_position((t.x, t.y, t.z), zmin, zmax);
            let voxel = layout.world_pos_to_voxel(position);
            (voxel, i)
        })
        .collect::<Vec<_>>();

    voxel_transcripts.par_sort_unstable_by_key(|(voxel, _)| *voxel);

    let mut voxel_bins = Vec::new();
    voxel_transcripts
        .iter()
        .group_by(|(voxel, _)| *voxel)
        .into_iter()
        .for_each(|(voxel, group)| {
            let transcripts = group.map(|(_, transcript)| *transcript).collect::<Vec<_>>();
            voxel_bins.push(VoxelBin {
                voxel,
                transcripts: Arc::new(Mutex::new(transcripts)),
            });
        });

    (layout, voxel_bins)
}

fn voxel_assignments(voxel_bins: &Vec<VoxelBin>, cell_assignments: &[CellIndex]) -> VoxelCellMap {
    let mut voxel_assignments = HashMap::new();
    let mut voxel_cells = VoxelCellMap::new();
    for voxel_bin in voxel_bins {
        voxel_assignments.clear();

        // vote on rect assignment
        for &t in voxel_bin.transcripts.lock().unwrap().iter() {
            if cell_assignments[t] != BACKGROUND_CELL {
                voxel_assignments
                    .entry(cell_assignments[t])
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        // Need to break ties deterministically here, otherwise
        // weird bugs emerge.
        let winner = voxel_assignments
            .iter()
            .max_by_key(|(cell, count)| (*count, *cell))
            .map(|(cell, _)| *cell)
            .unwrap_or(BACKGROUND_CELL);

        if winner != BACKGROUND_CELL {
            voxel_cells.insert(voxel_bin.voxel, winner);
        }
    }

    voxel_cells
}

pub struct VoxelSampler {
    chunkquad: ChunkQuadMap,
    transcript_genes: Vec<u32>,
    transcript_voxels: Vec<Voxel>,
    transcript_voxel_ord: Vec<usize>,
    transcript_layers: Vec<u32>,
    nlayers: usize,

    mismatch_edges: [Vec<Arc<Mutex<VoxelEdgeSampleSet>>>; 4],

    // assignment of rectbins to cells
    // (Unassigned cells are either absent or set to `BACKGROUND_CELL`)
    voxel_cells: VoxelCellMap,

    // need to track the per z-layer cell population and perimeter in order
    // to implement perimeter constraints.
    voxel_layers: usize,
    cell_population: Array2<f32>, // [voxellayers, ncells]
    cell_perimeter: Array2<f32>,  // [voxellayers, ncells]

    proposals: Vec<VoxelProposal>,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,

    zmin: f32,
    zmax: f32,

    voxel_volume: f32,
    quad: usize,
}

#[allow(clippy::too_many_arguments)]
impl VoxelSampler {
    pub fn new(
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        ngenes: usize,
        voxellayers: usize,
        nlayers: usize,
        z0: f32,
        layer_depth: f32,
        scale: f32,
        chunk_size: f32,
    ) -> Self {
        let (xmin, xmax, ymin, ymax, zmin, zmax) = coordinate_span(transcripts);
        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;

        let (layout, voxel_bins) = bin_transcripts(transcripts, scale, voxellayers);

        let transcript_genes = transcripts.iter().map(|t| t.gene).collect::<Vec<_>>();
        let transcript_layers = transcripts
            .iter()
            .map(|t| ((t.z - z0) / layer_depth) as u32)
            .collect::<Vec<_>>();

        assert!(layout.size.0 == layout.size.1);
        let voxel_volume = layout.size.0 * layout.size.1 * layout.size.2;

        // initialize mismatch_edges
        let mut mismatch_edges = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for chunks in mismatch_edges.iter_mut() {
            for _ in 0..nchunks {
                chunks.push(Arc::new(Mutex::new(VoxelEdgeSampleSet::new())));
            }
        }

        // initial voxel assignments
        let voxel_cells = voxel_assignments(&voxel_bins, &params.cell_assignments);

        // TODO: debugging
        let mut used_cell_ids: HashMap<CellIndex, CellIndex> = HashMap::new();
        for (_, cell_id) in voxel_cells.iter() {
            if *cell_id != BACKGROUND_CELL {
                let next_id = used_cell_ids.len() as CellIndex;
                used_cell_ids.entry(*cell_id).or_insert(next_id);
            }
        }

        // build index
        let mut voxel_index = HashMap::new();
        for voxel_bin in voxel_bins {
            voxel_index.insert(voxel_bin.voxel, voxel_bin);
        }

        // homogenize the rect: assign every transcript in the rect to the winner
        for voxel_bin in voxel_index.values() {
            let cell = voxel_cells.get(voxel_bin.voxel);
            for &t in voxel_bin.transcripts.lock().unwrap().iter() {
                if params.cell_assignments[t] != BACKGROUND_CELL {
                    params.cell_population[params.cell_assignments[t] as usize] -= 1;
                }
                if cell != BACKGROUND_CELL {
                    params.cell_population[cell as usize] += 1;
                }
                params.cell_assignments[t] = cell;
            }
        }

        // for (transcript, &cell) in transcripts.iter().zip(params.cell_assignments.iter()) {
        //     // let position = clip_z_position(
        //     //     (transcript.x, transcript.y, transcript.z), zmin, zmax);
        //     // let loc_cell = cubecells.get(layout.world_pos_to_cube(position));
        //     // assert!(loc_cell == cell);
        // }

        let cell_population = Array2::from_elem((voxellayers, params.ncells()), 0.0_f32);
        let cell_perimeter = Array2::from_elem((voxellayers, params.ncells()), 0.0_f32);

        let proposals = vec![VoxelProposal::new(ngenes, nlayers); nchunks];
        let connectivity_checker = ThreadLocal::new();
        // let transcript_x_pos = (0..transcripts.len()).collect::<Vec<_>>();
        let transcript_voxels = vec![Voxel::default(); transcripts.len()];
        let transcript_voxel_ord = (0..transcripts.len()).collect::<Vec<_>>();

        let mut sampler = VoxelSampler {
            chunkquad: ChunkQuadMap {
                layout,
                xmin,
                ymin,
                chunk_size,
                nxchunks,
            },
            transcript_genes,
            transcript_voxels,
            transcript_voxel_ord,
            transcript_layers,
            nlayers,
            mismatch_edges,
            voxel_cells,
            voxel_layers: voxellayers,
            cell_population,
            cell_perimeter,
            proposals,
            connectivity_checker,
            zmin,
            zmax,
            voxel_volume,
            quad: 0,
        };

        sampler.recompute_cell_population();
        sampler.recompute_cell_perimeter();
        sampler.recompute_cell_volume(priors, params);
        params.effective_cell_volume.assign(&params.cell_volume);
        sampler.populate_mismatches();
        sampler.update_transcript_positions(
            &vec![true; transcripts.len()],
            &params.transcript_positions,
        );

        sampler
    }

    fn ncells(&self) -> usize {
        self.cell_population.shape()[1]
    }

    // Allocate a new RectBinSampler with the same state as this one, but
    // grid resolution doubled (i.e. rect size halved).
    pub fn double_resolution(&self, params: &ModelParams, double_z_layers: bool) -> VoxelSampler {
        let nchunks = self.mismatch_edges[0].len();
        let ngenes = self.proposals[0].genepop.shape()[0];
        let voxel_volume = if double_z_layers {
            self.voxel_volume / 8.0
        } else {
            self.voxel_volume / 4.0
        };

        let layout = if double_z_layers {
            self.chunkquad.layout.double_resolution_and_layers()
        } else {
            self.chunkquad.layout.double_resolution()
        };

        let proposals = vec![VoxelProposal::new(ngenes, self.nlayers); nchunks];
        let connectivity_checker = ThreadLocal::new();

        let mut voxel_cells = VoxelCellMap::new();

        // 1.3s
        // let t0 = Instant::now();
        // Build a set of every cube that is either populated with transcripts
        // or assigned to a cell.
        let mut voxel_set = HashSet::<Voxel>::new();
        for (&voxel, &cell) in self.voxel_cells.iter() {
            if cell != BACKGROUND_CELL {
                voxel_set.insert(voxel);
            }
        }
        // println!("cubeset: {:?}", t0.elapsed());

        // 15.3s
        // let t0 = Instant::now();
        if double_z_layers {
            for voxel in voxel_set {
                let cell = self.voxel_cells.get(voxel);
                if cell != BACKGROUND_CELL {
                    let subvoxels = voxel.double_resolution_and_layers_children();
                    for subvoxel in &subvoxels {
                        voxel_cells.insert(*subvoxel, cell);
                    }
                }
            }
        } else {
            for voxel in voxel_set {
                let cell = self.voxel_cells.get(voxel);
                if cell != BACKGROUND_CELL {
                    let subvoxels = voxel.double_resolution_children();
                    for subvoxel in &subvoxels {
                        voxel_cells.insert(*subvoxel, cell);
                    }
                }
            }
        }
        // println!("cubebins: {:?}", t0.elapsed());

        // initialize mismatch_edges
        let mut mismatch_edges = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for chunks in mismatch_edges.iter_mut() {
            for _ in 0..nchunks {
                chunks.push(Arc::new(Mutex::new(VoxelEdgeSampleSet::new())));
            }
        }

        let voxellayers = if double_z_layers {
            self.voxel_layers * 2
        } else {
            self.voxel_layers
        };

        let cell_population =
            Array2::from_elem((voxellayers, self.cell_population.shape()[1]), 0.0_f32);
        let cell_perimeter =
            Array2::from_elem((voxellayers, self.cell_perimeter.shape()[1]), 0.0_f32);

        let mut sampler = VoxelSampler {
            chunkquad: ChunkQuadMap {
                layout,
                xmin: self.chunkquad.xmin,
                ymin: self.chunkquad.ymin,
                chunk_size: self.chunkquad.chunk_size,
                nxchunks: self.chunkquad.nxchunks,
            },
            transcript_genes: self.transcript_genes.clone(),
            transcript_voxels: self.transcript_voxels.clone(),
            transcript_voxel_ord: self.transcript_voxel_ord.clone(),
            transcript_layers: self.transcript_layers.clone(),
            nlayers: self.nlayers,
            mismatch_edges,
            voxel_cells,
            voxel_layers: voxellayers,
            cell_population,
            cell_perimeter,
            proposals,
            connectivity_checker,
            zmin: self.zmin,
            zmax: self.zmax,
            voxel_volume,
            quad: 0,
        };

        // 11.3s
        // let t0 = Instant::now();
        sampler.populate_mismatches();
        // println!("populate_mismatches: {:?}", t0.elapsed());

        // 141ms
        // let t0 = Instant::now();
        sampler.recompute_cell_population();
        // println!("recompute_cell_population: {:?}", t0.elapsed());

        // 85.4s
        // let t0 = Instant::now();
        sampler.recompute_cell_perimeter();
        // println!("recompute_cell_perimeter: {:?}", t0.elapsed());

        sampler.update_transcript_positions(
            &vec![true; params.transcript_positions.len()],
            &params.transcript_positions,
        );

        sampler
    }

    fn recompute_cell_volume(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // recompute cell areas as the sum of rect areas
        params.cell_volume.fill(0.0_f32);
        for (_, &cell) in self.voxel_cells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }
            params.cell_volume[cell as usize] += self.voxel_volume;
        }

        for cell_volume in params.cell_volume.iter_mut() {
            assert!(*cell_volume > 0.0);
            *cell_volume = cell_volume.max(priors.min_cell_volume);
        }

        Zip::from(&mut params.log_cell_volume)
            .and(&params.cell_volume)
            .into_par_iter()
            .with_min_len(50)
            .for_each(|(log_volume, &volume)| {
                *log_volume = volume.ln();
            });
    }

    fn recompute_cell_population(&mut self) {
        self.cell_population.fill(0.0_f32);
        for (&voxel, &cell) in self.voxel_cells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }
            self.cell_population[[voxel.k as usize, cell as usize]] += 1.0_f32;
        }
    }

    fn recompute_cell_perimeter(&mut self) {
        // TODO: This function is super expensive. It's hard to update in parallel
        // though, each cell consists of many voxels. So we'd either need mutexes
        // or more likelily, organize things by cell first.

        self.cell_perimeter.fill(0.0_f32);
        for (&voxel, &cell) in self.voxel_cells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }

            for neighbor in voxel.radius2_xy_neighborhood() {
                let neighbor_cell = self.voxel_cells.get(neighbor);
                if neighbor_cell != cell {
                    self.cell_perimeter[[voxel.k as usize, cell as usize]] += 1.0_f32;
                }
            }
        }
    }

    fn populate_mismatches(&mut self) {
        for (&voxel, &cell) in self.voxel_cells.iter() {
            let (chunk, quad) = self.chunkquad.get(voxel);
            for neighbor in voxel.von_neumann_neighborhood() {
                // don't consider neighbors that are out of bounds on the z-axis
                if neighbor.k < 0 || neighbor.k >= self.voxel_layers as i32 {
                    continue;
                }

                let neighbor_cell = self.voxel_cells.get(neighbor);
                if cell != neighbor_cell {
                    let (neighbor_chunk, neighbor_quad) = self.chunkquad.get(neighbor);

                    let mismatch_edges = &self.mismatch_edges[quad as usize];
                    if (chunk as usize) < mismatch_edges.len() {
                        mismatch_edges[chunk as usize]
                            .lock()
                            .unwrap()
                            .insert((voxel, neighbor));
                    }

                    let mismatch_edges = &self.mismatch_edges[neighbor_quad as usize];
                    if (neighbor_chunk as usize) < mismatch_edges.len() {
                        mismatch_edges[neighbor_chunk as usize]
                            .lock()
                            .unwrap()
                            .insert((neighbor, voxel));
                    }
                }
            }
        }
    }

    pub fn voxels(&self) -> impl Iterator<Item = (CellIndex, (f32, f32, f32, f32, f32, f32))> + '_ {
        return self
            .voxel_cells
            .iter()
            .filter(|(_, &cell)| cell != BACKGROUND_CELL)
            .map(|(voxel, cell)| (*cell, self.chunkquad.layout.voxel_to_world_coords(*voxel)));
    }

    pub fn cell_centroids(&self) -> Vec<(f32, f32, f32)> {
        let mut centroids = vec![(0.0, 0.0, 0.0); self.ncells()];
        let mut counts = vec![0; self.ncells()];
        for (&voxel, &cell) in self.voxel_cells.iter() {
            if cell != BACKGROUND_CELL {
                let (x0, y0, z0, x1, y1, z1) = self.chunkquad.layout.voxel_to_world_coords(voxel);
                centroids[cell as usize].0 += (x0 + x1) / 2.0;
                centroids[cell as usize].1 += (y0 + y1) / 2.0;
                centroids[cell as usize].2 += (z0 + z1) / 2.0;
                counts[cell as usize] += 1;
            }
        }

        for (i, count) in counts.iter().enumerate() {
            if *count > 0 {
                centroids[i].0 /= *count as f32;
                centroids[i].1 /= *count as f32;
                centroids[i].2 /= *count as f32;
            }
        }

        centroids
    }

    pub fn cell_polygons(&self) -> (Vec<CellPolygonLayers>, Vec<CellPolygon>) {
        // Build sets of voxels for each cell
        let mut cell_voxels = vec![HashSet::new(); self.ncells()];
        for (voxel, &cell) in self.voxel_cells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }

            cell_voxels[cell as usize].insert(*voxel);
        }

        let polygon_builder = ThreadLocal::new();
        let cell_polygons: Vec<Vec<(i32, MultiPolygon<f32>)>> = cell_voxels
            .par_iter()
            .map(|voxels| {
                let mut polygon_builder = polygon_builder
                    .get_or(|| RefCell::new(PolygonBuilder::new()))
                    .borrow_mut();

                polygon_builder.cell_voxels_to_polygons(&self.chunkquad.layout, voxels)
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
        // let t0 = Instant::now();
        let mut voxel_votes = HashMap::new();
        let mut top_voxel: HashMap<CellIndex, (Voxel, usize)> = HashMap::new();
        for (voxel, &cell) in self.voxel_cells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }

            // count transcripts in this voxel
            let transcript_range_start = self
                .transcript_voxel_ord
                .partition_point(|&t| self.transcript_voxels[t] < *voxel);

            let mut transcript_range_end = transcript_range_start;
            for &t in self.transcript_voxel_ord[transcript_range_start..].iter() {
                if self.transcript_voxels[t] != *voxel {
                    break;
                }
                transcript_range_end += 1;
            }

            let transcript_count = transcript_range_end - transcript_range_start;

            voxel_votes
                .entry((voxel.i, voxel.j))
                .or_insert_with(|| Vec::with_capacity(self.voxel_layers))
                .push((cell, transcript_count));

            top_voxel
                .entry(cell)
                .and_modify(|e| {
                    if transcript_count > e.1 {
                        *e = (
                            Voxel {
                                i: voxel.i,
                                j: voxel.j,
                                k: 0,
                            },
                            transcript_count,
                        )
                    }
                })
                .or_insert((
                    Voxel {
                        i: voxel.i,
                        j: voxel.j,
                        k: 0,
                    },
                    transcript_count,
                ));
        }
        // println!("index voxels: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        let mut cell_voxels = vec![HashSet::new(); self.ncells()];
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
            cell_voxels[winner as usize].insert(Voxel {
                i: voxel_i,
                j: voxel_j,
                k: 0,
            });
        }
        // println!("tally votes: {:?}", t0.elapsed());

        // Issues arise with some downstream tools (e.g. xeniumranger) if there
        // are empty cell polygons, which can happen with this consensus approach.
        // Here we try to fix those cases by including at least on voxel.
        for (cell, voxels) in cell_voxels.iter_mut().enumerate() {
            if voxels.is_empty() {
                let cell = cell as u32;
                voxels.insert(top_voxel[&cell].0);
            }
        }

        // let t0 = Instant::now();
        let polygon_builder = ThreadLocal::new();
        let cell_polygons: Vec<CellPolygon> = cell_voxels
            .par_iter()
            .map(|voxels| {
                let mut polygon_builder = polygon_builder
                    .get_or(|| RefCell::new(PolygonBuilder::new()))
                    .borrow_mut();

                let polygons =
                    polygon_builder.cell_voxels_to_polygons(&self.chunkquad.layout, voxels);
                if polygons.is_empty() {
                    CellPolygon::new(vec![])
                } else {
                    assert!(polygons.len() == 1);
                    let (_k, polygon) = polygons.first().unwrap();
                    polygon.clone()
                }
            })
            .collect();
        // println!("build polygons: {:?}", t0.elapsed());

        return cell_polygons;
    }

    // pub fn mismatch_edge_stats(&self) -> (usize, usize) {
    //     let mut num_cell_cell_edges = 0;
    //     let mut num_cell_bg_edges = 0;
    //     for mismatch_edges_quad in self.mismatch_edges.iter() {
    //         for mismatch_edges_chunk in mismatch_edges_quad.iter() {
    //             for (from, to) in mismatch_edges_chunk.lock().unwrap().iter() {
    //                 let cell_from = self.cubecells.get(*from);
    //                 let cell_to = self.cubecells.get(*to);
    //                 assert!(cell_from != cell_to);
    //                 if cell_from == BACKGROUND_CELL || cell_to == BACKGROUND_CELL {
    //                     num_cell_bg_edges += 1;
    //                 } else {
    //                     num_cell_cell_edges += 1;
    //                 }
    //             }
    //         }
    //     }

    //     return (num_cell_cell_edges, num_cell_bg_edges);
    // }

    // pub fn check_perimeter_bounds(&self, priors: &ModelPriors) {
    //     let mut count = 0;
    //     Zip::from(&self.cell_perimeter)
    //         .and(&self.cell_population)
    //         .for_each(|&perimiter, &pop| {
    //             let bound = perimeter_bound(priors.perimeter_eta, priors.perimeter_bound, pop);
    //             if perimiter > bound {
    //                 count += 1;
    //             }
    //         });
    //     println!("perimeter bound violations: {}", count);
    // }

    pub fn check_consistency(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        self.check_cell_volume(priors, params);
        self.check_cell_perimeter();
        self.check_cell_population();
    }

    // Since we have to update various values as we sample, here we are checking to
    // make sure no inconsistencies emerged.
    fn check_cell_volume(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        let current_cell_volume = params.cell_volume.clone();
        self.recompute_cell_volume(priors, params);

        let ε = 1e-5_f32;
        for (v0, v1) in params.cell_volume.iter().zip(current_cell_volume.iter()) {
            if relerr(*v0, *v1) > ε {
                dbg!(v0, v1);
            }
        }
        // TODO: A small few are still off by one cubevolume

        assert!(params
            .cell_volume
            .iter()
            .zip(current_cell_volume.iter())
            .all(|(&v0, &v1)| relerr(v0, v1) < ε));
    }

    fn check_cell_perimeter(&mut self) {
        let current_cell_perimeter = self.cell_perimeter.clone();
        self.recompute_cell_perimeter();
        for (p0, p1) in self
            .cell_perimeter
            .iter()
            .zip(current_cell_perimeter.iter())
        {
            if p0 != p1 {
                dbg!(p0, p1);
            }
        }

        assert!(self
            .cell_perimeter
            .iter()
            .zip(current_cell_perimeter.iter())
            .all(|(new_perimeter, old_perimeter)| new_perimeter == old_perimeter));
    }

    fn check_cell_population(&mut self) {
        let current_cell_population = self.cell_population.clone();
        self.recompute_cell_population();
        for (p0, p1) in self
            .cell_population
            .iter()
            .zip(current_cell_population.iter())
        {
            if p0 != p1 {
                dbg!(p0, p1);
            }
        }

        assert!(self
            .cell_population
            .iter()
            .zip(current_cell_population.iter())
            .all(|(new_population, old_population)| new_population == old_population));
    }

    // fn chunkquad(&self, hex: Hex) -> (u32, u32) {
    //     let hex_xy = self.layout.hex_to_world_pos(hex);
    //     return chunkquad(hex_xy.x, hex_xy.y, self.xmin, self.ymin, self.chunk_size, self.nxchunks);
    // }
}

impl Sampler<VoxelProposal> for VoxelSampler {
    fn repopulate_proposals(&mut self, priors: &ModelPriors, params: &ModelParams) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.01;

        self.proposals
            .par_iter_mut()
            // .iter_mut()
            .zip(&self.mismatch_edges[self.quad])
            .for_each(|(proposal, mismatch_edges)| {
                proposal.old_cell = BACKGROUND_CELL;
                proposal.new_cell = BACKGROUND_CELL;
                proposal.ignore = false;
                proposal.accept = false;

                let mismatch_edges = mismatch_edges.lock().unwrap();
                if mismatch_edges.is_empty() {
                    proposal.ignore = true;
                    return;
                }

                let mut rng = rng();
                let (i, j) = mismatch_edges.choose(&mut rng).unwrap();

                let cell_from = self.voxel_cells.get(*i);
                let mut cell_to = self.voxel_cells.get(*j);
                assert!(cell_from != cell_to);

                let from_unassigned = cell_from == BACKGROUND_CELL;
                let to_unassigned = cell_to == BACKGROUND_CELL;

                // don't let the cell grow in the z-dimension past the extents
                // of the data. Not strictly necessary to restrict it this way,
                // but lessens the tendency to produce weird shaped cells.
                if from_unassigned && !to_unassigned {
                    let (_, _, z0, _, _, z1) = self.chunkquad.layout.voxel_to_world_coords(*i);
                    if z1 < self.zmin || z0 > self.zmax {
                        proposal.ignore = true;
                        return;
                    }
                }

                if !from_unassigned && rng.random::<f64>() < UNASSIGNED_PROPOSAL_PROB {
                    cell_to = BACKGROUND_CELL;
                }

                // Local connectivity condition: don't propose changes that render increase the
                // number of connected components of either the cell_from or cell_to
                // neighbors subgraphs.
                if priors.enforce_connectivity {
                    let mut connectivity_checker = self
                        .connectivity_checker
                        .get_or(|| RefCell::new(ConnectivityChecker::new()))
                        .borrow_mut();

                    let art_from = connectivity_checker.voxel_isarticulation(
                        *i,
                        |voxel| self.voxel_cells.get(voxel),
                        cell_from,
                    );

                    let art_to = connectivity_checker.voxel_isarticulation(
                        *i,
                        |voxel| self.voxel_cells.get(voxel),
                        cell_to,
                    );

                    if (art_from && !from_unassigned) || (art_to && !to_unassigned) {
                        proposal.ignore = true;
                        return;
                    }
                }

                // Don't propose removing the last voxel from a cell. (This is
                // breaks the markov chain balance, since there's no path back to the previous state.)
                //
                // We could allow this if we introduce a spontaneous nucleation move.
                if !from_unassigned
                    && params.cell_volume[cell_from as usize] - self.voxel_volume
                        < priors.min_cell_volume
                {
                    proposal.ignore = true;
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = mismatch_edges.len();

                let num_new_state_neighbors = i
                    .von_neumann_neighborhood()
                    .iter()
                    .filter(|&&j| {
                        j.inbounds(self.voxel_layers) && self.voxel_cells.get(j) == cell_to
                    })
                    .count();

                let num_prev_state_neighbors = i
                    .von_neumann_neighborhood()
                    .iter()
                    .filter(|&&j| {
                        j.inbounds(self.voxel_layers) && self.voxel_cells.get(j) == cell_from
                    })
                    .count();

                let mut proposal_prob = (1.0 - UNASSIGNED_PROPOSAL_PROB)
                    * (num_new_state_neighbors as f64 / num_mismatching_edges as f64);

                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if to_unassigned {
                    let num_mismatching_neighbors = i
                        .von_neumann_neighborhood()
                        .iter()
                        .filter(|&&j| self.voxel_cells.get(j) != cell_from)
                        .count();
                    proposal_prob += UNASSIGNED_PROPOSAL_PROB
                        * (num_mismatching_neighbors as f64 / num_mismatching_edges as f64);
                }

                let new_num_mismatching_edges = num_mismatching_edges
                    + 2*num_prev_state_neighbors // edges that are newly mismatching
                    - 2*num_new_state_neighbors; // edges that are newly matching

                let mut reverse_proposal_prob = (1.0 - UNASSIGNED_PROPOSAL_PROB)
                    * (num_prev_state_neighbors as f64 / new_num_mismatching_edges as f64);

                // If this is a proposal from unassigned, account for multiple ways of reversing it
                if from_unassigned {
                    let new_num_mismatching_neighbors = i
                        .von_neumann_neighborhood()
                        .iter()
                        .filter(|&&j| self.voxel_cells.get(j) != cell_to)
                        .count();
                    reverse_proposal_prob += UNASSIGNED_PROPOSAL_PROB
                        * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64);
                }

                // if proposal_prob > 0.5 || reverse_proposal_prob > 0.5 {
                // if from_unassigned {
                //     dbg!(
                //         from_unassigned,
                //         to_unassigned,
                //         proposal_prob,
                //         reverse_proposal_prob,
                //         num_mismatching_edges,
                //         new_num_mismatching_edges,
                //     );
                // }

                proposal.voxel = *i;
                // if let Some(transcripts) = transcripts {
                //     proposal.transcripts.clone_from(&transcripts.lock().unwrap());
                // } else {
                //     proposal.transcripts.clear();
                // }
                proposal.old_cell = cell_from;
                proposal.new_cell = cell_to;
                proposal.log_weight = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;
                proposal.ignore = false;
                proposal.accept = false;
                proposal.old_cell_volume_delta = -self.voxel_volume;
                proposal.new_cell_volume_delta = self.voxel_volume;

                proposal.old_cell_perimeter_delta = 0.0;
                proposal.new_cell_perimeter_delta = 0.0;

                for neighbor in i.radius2_xy_neighborhood() {
                    let neighbor_cell = self.voxel_cells.get(neighbor);

                    // cube i's new mismatches
                    if neighbor_cell != cell_to {
                        proposal.new_cell_perimeter_delta += 1.0;
                    }

                    // neighbors for whom i is no longer a mismatch
                    if neighbor_cell == cell_to {
                        proposal.new_cell_perimeter_delta -= 1.0;
                    }

                    // neighbors for whom i is now a mismatch
                    if neighbor_cell == cell_from {
                        proposal.old_cell_perimeter_delta += 1.0;
                    }

                    // neighbors of i that were previously counted as a mismatch
                    if neighbor_cell != cell_from {
                        proposal.old_cell_perimeter_delta -= 1.0;
                    }
                }

                // reject in advance if the perimeter for this layer surpases
                // the limit.
                if cell_from != BACKGROUND_CELL {
                    let prev_bound = perimeter_bound(
                        priors.perimeter_eta,
                        priors.perimeter_bound,
                        self.cell_population[[i.k as usize, cell_from as usize]],
                    );
                    let prev_bound_ratio =
                        self.cell_perimeter[[i.k as usize, cell_from as usize]] / prev_bound;

                    let old_cell_perimeter = self.cell_perimeter
                        [[i.k as usize, cell_from as usize]]
                        + proposal.old_cell_perimeter_delta;
                    let pop = self.cell_population[[i.k as usize, cell_from as usize]] - 1.0;
                    let bound = perimeter_bound(priors.perimeter_eta, priors.perimeter_bound, pop);
                    let bound_ratio = old_cell_perimeter / bound;

                    // if we violate the perimiter bound, and make things worse, reject
                    if old_cell_perimeter > bound && bound_ratio > prev_bound_ratio {
                        proposal.ignore = true;
                        return;
                    }
                }

                if cell_to != BACKGROUND_CELL {
                    let prev_bound = perimeter_bound(
                        priors.perimeter_eta,
                        priors.perimeter_bound,
                        self.cell_population[[i.k as usize, cell_to as usize]],
                    );
                    let prev_bound_ratio =
                        self.cell_perimeter[[i.k as usize, cell_to as usize]] / prev_bound;

                    let new_cell_perimeter = self.cell_perimeter[[i.k as usize, cell_to as usize]]
                        + proposal.new_cell_perimeter_delta;
                    let pop = self.cell_population[[i.k as usize, cell_to as usize]] + 1.0;
                    let bound = perimeter_bound(priors.perimeter_eta, priors.perimeter_bound, pop);
                    let bound_ratio = new_cell_perimeter / bound;

                    if new_cell_perimeter > bound && bound_ratio > prev_bound_ratio {
                        proposal.ignore = true;
                        return;
                    }
                }

                // find transcripts within the voxel
                let transcript_range_start = self
                    .transcript_voxel_ord
                    .partition_point(|&t| self.transcript_voxels[t] < *i);

                let mut transcript_range_end = transcript_range_start;
                proposal.genepop.fill(0);
                for &t in self.transcript_voxel_ord[transcript_range_start..].iter() {
                    if self.transcript_voxels[t] != *i {
                        break;
                    }
                    transcript_range_end += 1;
                    let layer = self.transcript_layers[t] as usize;
                    proposal.genepop[[self.transcript_genes[t] as usize, layer]] += 1;
                }

                proposal.transcripts.clear();
                proposal.transcripts.extend_from_slice(
                    &self.transcript_voxel_ord[transcript_range_start..transcript_range_end],
                );
            });

        // Increment so we run updates on the next quad
        self.quad = (self.quad + 1) % 4;
    }

    fn proposals<'a, 'b>(&'a self) -> &'b [VoxelProposal]
    where
        'a: 'b,
    {
        &self.proposals
    }

    fn proposals_mut<'a, 'b>(&'a mut self) -> &'b mut [VoxelProposal]
    where
        'a: 'b,
    {
        &mut self.proposals
    }

    fn update_sampler_state(&mut self, _: &ModelParams) {
        for proposal in self.proposals.iter().filter(|p| !p.ignore && p.accept) {
            self.voxel_cells.set(proposal.voxel, proposal.new_cell);

            // update cell population and perimeter
            if proposal.old_cell != BACKGROUND_CELL {
                self.cell_population[[proposal.voxel.k as usize, proposal.old_cell as usize]] -=
                    1.0_f32;
                self.cell_perimeter[[proposal.voxel.k as usize, proposal.old_cell as usize]] +=
                    proposal.old_cell_perimeter_delta;
            }

            if proposal.new_cell != BACKGROUND_CELL {
                self.cell_population[[proposal.voxel.k as usize, proposal.new_cell as usize]] +=
                    1.0_f32;
                self.cell_perimeter[[proposal.voxel.k as usize, proposal.new_cell as usize]] +=
                    proposal.new_cell_perimeter_delta;
            }
        }

        self.proposals
            .par_iter()
            .filter(|p| !p.ignore && p.accept)
            .for_each(|proposal| {
                let (chunk, quad) = self.chunkquad.get(proposal.voxel);

                // update mismatch edges
                for neighbor in proposal.voxel.von_neumann_neighborhood() {
                    if neighbor.k < 0 || neighbor.k >= self.voxel_layers as i32 {
                        continue;
                    }

                    let (neighbor_chunk, neighbor_quad) = self.chunkquad.get(neighbor);
                    let neighbor_cell = self.voxel_cells.get(neighbor);
                    if proposal.new_cell == neighbor_cell {
                        let mismatch_edges = &self.mismatch_edges[quad as usize];
                        if (chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[chunk as usize]
                                .lock()
                                .unwrap()
                                .remove((proposal.voxel, neighbor));
                        }

                        let mismatch_edges = &self.mismatch_edges[neighbor_quad as usize];
                        if (neighbor_chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[neighbor_chunk as usize]
                                .lock()
                                .unwrap()
                                .remove((neighbor, proposal.voxel));
                        }
                    } else {
                        let mismatch_edges = &self.mismatch_edges[quad as usize];
                        if (chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[chunk as usize]
                                .lock()
                                .unwrap()
                                .insert((proposal.voxel, neighbor));
                        }

                        let mismatch_edges = &self.mismatch_edges[neighbor_quad as usize];
                        if (neighbor_chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[neighbor_chunk as usize]
                                .lock()
                                .unwrap()
                                .insert((neighbor, proposal.voxel));
                        }
                    }
                }
            });
    }

    fn cell_at_position(&self, position: (f32, f32, f32)) -> u32 {
        let position = clip_z_position(position, self.zmin, self.zmax);
        let cubindex = self.chunkquad.layout.world_pos_to_voxel(position);
        self.voxel_cells.get(cubindex)
    }

    fn update_transcript_positions(&mut self, updated: &[bool], positions: &[(f32, f32, f32)]) {
        // let t0 = Instant::now();
        self.transcript_voxels
            .par_iter_mut()
            .zip(positions)
            .zip(updated)
            .for_each(|((voxel, position), &updated)| {
                if updated {
                    let position = clip_z_position(*position, self.zmin, self.zmax);
                    *voxel = self.chunkquad.layout.world_pos_to_voxel(position);
                }
            });
        // println!("    REPO: compute voxels {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.transcript_voxel_ord
            .par_sort_by_key(|&t| (self.transcript_voxels[t], self.transcript_genes[t]));
        // println!("    REPO: sort on voxel assignment {:?}", t0.elapsed());
    }

    // fn update_transcript_position(&mut self, i: usize, prev_pos: (f32, f32, f32), new_pos: (f32, f32, f32)) {
    //     let prev_pos = clip_z_position(prev_pos, self.zmin, self.zmax);
    //     let old_cube = self
    //         .chunkquad.layout.world_pos_to_cube(prev_pos);

    //     let new_pos = clip_z_position(new_pos, self.zmin, self.zmax);
    //     let new_cube = self
    //         .chunkquad.layout.world_pos_to_cube(new_pos);

    //     if old_cube == new_cube {
    //         return;
    //     }

    //     // remove transcript from old cube
    //     {
    //         let cubeindex = self.cubeindex.read().unwrap();
    //         let old_cubebin = cubeindex.get(&old_cube).unwrap();
    //         let transcripts = &mut old_cubebin.transcripts.lock().unwrap();
    //         let idx = transcripts.iter().position(|t| *t == i).unwrap();
    //         transcripts.swap_remove(idx);
    //     }

    //     // insert transcript into old cube
    //     {
    //         let cubeindex = &mut self.cubeindex.write().unwrap();
    //         let new_cubebin = cubeindex
    //             .entry(new_cube)
    //             .or_insert_with(|| CubeBin::new(new_cube));
    //         new_cubebin.transcripts.lock().unwrap().push(i);
    //     }
    // }

    // // TODO: This can't really be effectively parallelized because there's too
    // // much contention on `cubeindex`.
    // //
    // // I don't know what can be done about this. Moving transcripts around creates
    // // a bunch of new cubes. There's not much we can do about that. Any way to
    // // speed this up seems like it would require an entirely different design.
    // //
    // // 1. Just keep track of which transcripts are in which chunk. There would
    // //    still be a lot of contention because we need to need to move transcripts
    // //    in and out of chunks.
    // //
    // // 2. Don't keep track of anything. Each proposal will just loop through the
    // //    transcript positions selecting transcripts. I'm afraid this would slow
    // //    things down a lot.
    // //
    // // 3. Index positions somehow (maybe a kd_tree) before generating proposals.
    // //    This seems like the most promising approach. Just have to figure
    // //    out the indexing scheme.

    // fn update_transcript_positions(
    //     &mut self,
    //     accept: &Vec<bool>,
    //     positions: &Vec<(f32, f32, f32)>,
    //     proposed_positions: &Vec<(f32, f32, f32)>)
    // {
    //     accept
    //         .par_iter()
    //         .zip(positions)
    //         .zip(proposed_positions)
    //         .enumerate()
    //         .for_each(|(i, ((accept, position), proposed_position))| {
    //             if !*accept {
    //                 return;
    //             }

    //             let prev_pos = clip_z_position(*position, self.zmin, self.zmax);
    //             let old_cube = self
    //                 .chunkquad.layout.world_pos_to_cube(prev_pos);

    //             let new_pos = clip_z_position(*proposed_position, self.zmin, self.zmax);
    //             let new_cube = self
    //                 .chunkquad.layout.world_pos_to_cube(new_pos);

    //             if old_cube == new_cube {
    //                 return;
    //             }

    //             // remove transcript from old cube
    //             {
    //                 let cubeindex = self.cubeindex.read().unwrap();
    //                 let old_cubebin = cubeindex.get(&old_cube).unwrap();
    //                 let transcripts = &mut old_cubebin.transcripts.lock().unwrap();
    //                 let idx = transcripts.iter().position(|t| *t == i).unwrap();
    //                 transcripts.swap_remove(idx);
    //             }

    //             // insert transcript into old cube
    //             {
    //                 // if let Some(new_cubebin) = self.cubeindex.read().unwrap().get(&old_cube) {
    //                 //     new_cubebin.transcripts.lock().unwrap().push(i);
    //                 // } else {
    //                     let cubeindex = &mut self.cubeindex.write().unwrap();
    //                     let new_cubebin = cubeindex
    //                         .entry(new_cube)
    //                         .or_insert_with(|| CubeBin::new(new_cube));
    //                     new_cubebin.transcripts.lock().unwrap().push(i);
    //                 // }
    //             }
    //         });
    // }
}

#[derive(Clone, Debug)]
pub struct VoxelProposal {
    voxel: Voxel,
    transcripts: Vec<usize>,

    // [ngenes, nlayers] gene count for this rect
    genepop: Array2<u32>,

    old_cell: u32,
    new_cell: u32,

    // metroplis-hastings proposal weight weight
    log_weight: f32,

    ignore: bool,
    accept: bool,

    // updated cell volumes and logprobs if the proposal is accepted
    old_cell_volume_delta: f32,
    new_cell_volume_delta: f32,

    old_cell_perimeter_delta: f32,
    new_cell_perimeter_delta: f32,
}

impl VoxelProposal {
    fn new(ngenes: usize, nlayers: usize) -> VoxelProposal {
        VoxelProposal {
            voxel: Voxel::new(0, 0, 0),
            transcripts: Vec::new(),
            genepop: Array2::from_elem((ngenes, nlayers), 0),
            old_cell: 0,
            new_cell: 0,
            log_weight: 0.0,
            ignore: false,
            accept: false,
            old_cell_volume_delta: 0.0,
            new_cell_volume_delta: 0.0,
            old_cell_perimeter_delta: 0.0,
            new_cell_perimeter_delta: 0.0,
        }
    }
}

impl Proposal for VoxelProposal {
    fn accept(&mut self) {
        self.accept = true;
    }
    fn reject(&mut self) {
        self.accept = false;
    }

    fn ignored(&self) -> bool {
        self.ignore
    }
    fn accepted(&self) -> bool {
        self.accept
    }

    fn old_cell(&self) -> u32 {
        self.old_cell
    }

    fn new_cell(&self) -> u32 {
        self.new_cell
    }

    fn old_cell_volume_delta(&self) -> f32 {
        self.old_cell_volume_delta
    }

    fn new_cell_volume_delta(&self) -> f32 {
        self.new_cell_volume_delta
    }

    fn log_weight(&self) -> f32 {
        self.log_weight
    }

    fn transcripts<'b, 'c>(&'b self) -> &'c [usize]
    where
        'b: 'c,
    {
        self.transcripts.as_slice()
    }

    fn gene_count<'b, 'c>(&'b self) -> &'c Array2<u32>
    where
        'b: 'c,
    {
        &self.genepop
    }
}

// We need to exclude cells that can't be initalized with a non-zero number of voxels.
pub fn filter_sparse_cells(
    scale: f32,
    voxellayers: usize,
    transcripts: &Vec<Transcript>,
    nucleus_assignments: &mut [CellIndex],
    cell_assignments: &mut [CellIndex],
    nucleus_population: &mut Vec<usize>,
    original_cell_ids: &mut Vec<String>,
) {
    // let t0 = Instant::now();
    let (_layout, voxel_bins) = bin_transcripts(transcripts, scale, voxellayers);
    // println!("bin_transcripts: {:?}", t0.elapsed());

    // let t0 = Instant::now();
    let voxel_cells = voxel_assignments(&voxel_bins, nucleus_assignments);
    // println!("cube_assignments: {:?}", t0.elapsed());

    // let t0 = Instant::now();
    let mut used_cell_ids: HashMap<CellIndex, CellIndex> = HashMap::new();
    for (_, cell_id) in voxel_cells.iter() {
        if *cell_id != BACKGROUND_CELL {
            let next_id = used_cell_ids.len() as CellIndex;
            used_cell_ids.entry(*cell_id).or_insert(next_id);
        }
    }

    if used_cell_ids.len() != nucleus_population.len() {
        for cell_id in nucleus_assignments.iter_mut() {
            if *cell_id != BACKGROUND_CELL {
                if let Some(new_cell_id) = used_cell_ids.get(cell_id) {
                    *cell_id = *new_cell_id;
                } else {
                    *cell_id = BACKGROUND_CELL;
                }
            }
        }
        for cell_id in cell_assignments.iter_mut() {
            if *cell_id != BACKGROUND_CELL {
                if let Some(new_cell_id) = used_cell_ids.get(cell_id) {
                    *cell_id = *new_cell_id;
                } else {
                    *cell_id = BACKGROUND_CELL;
                }
            }
        }

        let old_original_cell_ids = original_cell_ids.clone();
        original_cell_ids.resize(used_cell_ids.len(), String::new());
        for (old_cell_id, new_cell_id) in used_cell_ids.iter() {
            original_cell_ids[*new_cell_id as usize] =
                old_original_cell_ids[*old_cell_id as usize].clone();
        }
    }
    // println!("index assignments {:?}", t0.elapsed());

    nucleus_population.resize(used_cell_ids.len(), 0);
    nucleus_population.fill(0);
    for cell_id in nucleus_assignments.iter() {
        if *cell_id != BACKGROUND_CELL {
            nucleus_population[*cell_id as usize] += 1;
        }
    }
}
