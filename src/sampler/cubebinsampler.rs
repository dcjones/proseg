use super::connectivity::ConnectivityChecker;
use super::math::relerr;
use super::sampleset::SampleSet;
use super::transcripts::{coordinate_span, CellIndex, Transcript, BACKGROUND_CELL};
use super::{chunkquad, perimeter_bound, ModelParams, ModelPriors, Proposal, Sampler};

// use hexx::{Hex, HexLayout, HexOrientation, Vec2};
// use arrow;
use geo::geometry::{LineString, MultiPolygon, Polygon};
use geo::algorithm::simplify::Simplify;
use geo::BooleanOps;
use ndarray::{Array1, Array2, Axis};
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::f32;
use std::sync::{Arc, Mutex, RwLock};
use thread_local::ThreadLocal;

use std::time::Instant;

// taken from: https://github.com/a-b-street/abstreet
fn union_all_into_multipolygon(mut list: Vec<Polygon<f32>>) -> MultiPolygon<f32> {
    // TODO Not sure why this happened, or if this is really valid to construct...
    if list.is_empty() {
        return MultiPolygon(Vec::new());
    }

    let mut result = geo::MultiPolygon(vec![list.pop().unwrap().into()]);
    for p in list {
        result = result.union(&p.into());
    }
    result
}


fn clip_z_position(position: (f32, f32, f32), zmin: f32, zmax: f32) -> (f32, f32, f32) {
    let eps = (zmax - zmin) * 1e-6;
    return (
        position.0,
        position.1,
        position.2.max(zmin + eps).min(zmax - eps),
    );
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Cube {
    i: i32,
    j: i32,
    k: i32,
}

impl Cube {
    fn new(i: i32, j: i32, k: i32) -> Cube {
        return Cube { i, j, k };
    }

    pub fn moore_neighborhood(&self) -> [Cube; 26] {
        return [
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
        .map(|(di, dj, dk)| Cube::new(self.i + di, self.j + dj, self.k + dk));
    }

    pub fn von_neumann_neighborhood(&self) -> [Cube; 6] {
        return [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ]
        .map(|(di, dj, dk)| Cube::new(self.i + di, self.j + dj, self.k + dk));
    }

    pub fn radius2_xy_neighborhood(&self) -> [Cube; 12] {
        return [
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
        .map(|(di, dj, dk)| Cube::new(self.i + di, self.j + dj, self.k + dk));
    }

    fn double_resolution_children(&self) -> [Cube; 8] {
        return [
            Cube::new(2 * self.i, 2 * self.j, 2 * self.k),
            Cube::new(2 * self.i + 1, 2 * self.j, 2 * self.k),
            Cube::new(2 * self.i, 2 * self.j + 1, 2 * self.k),
            Cube::new(2 * self.i + 1, 2 * self.j + 1, 2 * self.k),
            Cube::new(2 * self.i, 2 * self.j, 2 * self.k + 1),
            Cube::new(2 * self.i + 1, 2 * self.j, 2 * self.k + 1),
            Cube::new(2 * self.i, 2 * self.j + 1, 2 * self.k + 1),
            Cube::new(2 * self.i + 1, 2 * self.j + 1, 2 * self.k + 1),
        ];
    }

    fn inbounds(&self, nlayers: usize) -> bool {
        return self.k >= 0 && self.k < nlayers as i32;
    }
}

#[derive(Debug)]
struct CubeLayout {
    origin: (f32, f32, f32),
    cube_size: (f32, f32, f32),
}

impl CubeLayout {
    fn double_resolution(&self) -> CubeLayout {
        return CubeLayout {
            origin: (self.origin.0, self.origin.1, self.origin.2),
            cube_size: (
                self.cube_size.0 / 2.0,
                self.cube_size.1 / 2.0,
                self.cube_size.2 / 2.0,
            ),
        };
    }

    fn cube_to_world_pos(&self, cube: Cube) -> (f32, f32, f32) {
        return (
            self.origin.0 + (0.5 + cube.i as f32) * self.cube_size.0,
            self.origin.1 + (0.5 + cube.j as f32) * self.cube_size.1,
            self.origin.2 + (0.5 + cube.k as f32) * self.cube_size.2,
        );
    }

    fn world_pos_to_cube(&self, pos: (f32, f32, f32)) -> Cube {
        return Cube::new(
            ((pos.0 - self.origin.0) / self.cube_size.0).floor() as i32,
            ((pos.1 - self.origin.1) / self.cube_size.1).floor() as i32,
            ((pos.2 - self.origin.2) / self.cube_size.2).floor() as i32,
        );
    }

    fn cube_to_world_coords(&self, cube: Cube) -> (f32, f32, f32, f32, f32, f32) {
        let x0 = self.origin.0 + cube.i as f32 * self.cube_size.0;
        let y0 = self.origin.1 + cube.j as f32 * self.cube_size.1;
        let z0 = self.origin.2 + cube.k as f32 * self.cube_size.2;

        return (
            x0,
            y0,
            z0,
            x0 + self.cube_size.0,
            y0 + self.cube_size.1,
            z0 + self.cube_size.2,
        );
    }
}

type CubeEdgeSampleSet = SampleSet<(Cube, Cube)>;

#[derive(Clone, Debug)]
struct CubeBin {
    cube: Cube,
    transcripts: Arc<Mutex<Vec<usize>>>,
}

impl CubeBin {
    fn new(cube: Cube) -> Self {
        Self {
            cube,
            transcripts: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

struct ChunkQuadMap {
    layout: CubeLayout,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,
}

impl ChunkQuadMap {
    fn get(&self, cube: Cube) -> (u32, u32) {
        let cube_xyz = self.layout.cube_to_world_pos(cube);
        return chunkquad(
            cube_xyz.0,
            cube_xyz.1,
            self.xmin,
            self.ymin,
            self.chunk_size,
            self.nxchunks,
        );
    }
}

struct CubeCellMap {
    index: HashMap<Cube, CellIndex>,
}

impl CubeCellMap {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    fn insert(&mut self, cube: Cube, cell: CellIndex) {
        self.index.insert(cube, cell);
    }

    fn get(&self, cube: Cube) -> CellIndex {
        match self.index.get(&cube) {
            Some(cell) => *cell,
            None => BACKGROUND_CELL,
        }
    }

    fn set(&mut self, cube: Cube, cell: CellIndex) {
        self.index.insert(cube, cell);
    }

    // fn len(&self) -> usize {
    //     return self.index.len();
    // }

    // fn count(&self, cell: u32) -> usize {
    //     return self.index.values().filter(|&&c| c == cell).count();
    // }

    fn iter(&self) -> std::collections::hash_map::Iter<Cube, CellIndex> {
        return self.index.iter();
    }
}

// Initial binning of the transcripts
fn bin_transcripts(transcripts: &Vec<Transcript>, scale: f32) -> (CubeLayout, Vec<CubeBin>) {
    let (_, _, _, _, zmin, zmax) = coordinate_span(&transcripts);

    let mut height = zmax - zmin;
    if height == 0.0 {
        height = 1.0;
    }

    let cube_size = scale;
    let layout = CubeLayout {
        origin: (0.0, 0.0, zmin),
        cube_size: (cube_size, cube_size, height),
    };

    // Bin transcripts into CubeBins
    let mut cube_index = HashMap::new();

    for (i, transcript) in transcripts.iter().enumerate() {
        let position = clip_z_position(
            (transcript.x, transcript.y, transcript.z), zmin, zmax);
        let cube = layout.world_pos_to_cube(position);

        cube_index
            .entry(cube)
            .or_insert_with(|| CubeBin::new(cube))
            .transcripts
            .lock()
            .unwrap()
            .push(i);
    }

    let cubebins = cube_index.values().cloned().collect::<Vec<_>>();

    return (layout, cubebins);
}

fn cube_assignments(cubebins: &Vec<CubeBin>, cell_assignments: &Vec<CellIndex>) -> CubeCellMap {
    let mut cube_assignments = HashMap::new();
    let mut cubecells = CubeCellMap::new();
    for cubebin in cubebins {
        cube_assignments.clear();

        // vote on rect assignment
        for &t in cubebin.transcripts.lock().unwrap().iter() {
            if cell_assignments[t] != BACKGROUND_CELL {
                cube_assignments
                    .entry(cell_assignments[t])
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }

        let winner = cube_assignments
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(cell, _)| *cell)
            .unwrap_or(BACKGROUND_CELL);

        if winner != BACKGROUND_CELL {
            cubecells.insert(cubebin.cube, winner);
        }
    }

    return cubecells;
}

pub struct CubeBinSampler {
    chunkquad: ChunkQuadMap,
    transcript_genes: Vec<u32>,
    transcript_layers: Vec<u32>,
    density: Array1<f32>,
    nlayers: usize,

    mismatch_edges: [Vec<Arc<Mutex<CubeEdgeSampleSet>>>; 4],
    cubeindex: Arc<RwLock<HashMap<Cube, CubeBin>>>,

    // assignment of rectbins to cells
    // (Unassigned cells are either absent or set to `BACKGROUND_CELL`)
    cubecells: CubeCellMap,

    // need to track the per z-layer cell population and perimeter in order
    // to implement perimeter constraints.
    nzbins: usize,
    cell_population: Array2<f32>, // [nzbins, ncells]
    cell_perimeter: Array2<f32>,  // [nzbins, ncells]

    proposals: Vec<CubeBinProposal>,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,

    zmin: f32,
    zmax: f32,
    nvoxel_layers: usize,

    cubevolume: f32,
    quad: usize,
}

impl CubeBinSampler {
    pub fn new(
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        density: Array1<f32>,
        ngenes: usize,
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

        let (layout, cubebins) = bin_transcripts(transcripts, scale);

        let transcript_genes = transcripts.iter().map(|t| t.gene).collect::<Vec<_>>();
        let transcript_layers = transcripts
            .iter()
            .map(|t| ((t.z - z0) / layer_depth) as u32)
            .collect::<Vec<_>>();

        assert!(layout.cube_size.0 == layout.cube_size.1);
        let cubevolume = layout.cube_size.0 * layout.cube_size.1 * layout.cube_size.2;

        // initialize mismatch_edges
        let mut mismatch_edges = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for chunks in mismatch_edges.iter_mut() {
            for _ in 0..nchunks {
                chunks.push(Arc::new(Mutex::new(CubeEdgeSampleSet::new())));
            }
        }

        // initial cube assignments
        let cubecells = cube_assignments(&cubebins, &params.cell_assignments);

        // build index
        let mut cubeindex = HashMap::new();
        for cubebin in cubebins {
            cubeindex.insert(cubebin.cube, cubebin);
        }

        // homogenize the rect: assign every transcript in the rect to the winner
        for cubebin in cubeindex.values() {
            let cell = cubecells.get(cubebin.cube);

            for &t in cubebin.transcripts.lock().unwrap().iter() {
                if params.cell_assignments[t] != BACKGROUND_CELL {
                    params.cell_population[params.cell_assignments[t] as usize] -= 1;
                }
                if cell != BACKGROUND_CELL {
                    params.cell_population[cell as usize] += 1;
                }
                params.cell_assignments[t] = cell;
            }
        }

        params.recompute_counts(transcripts);

        // for (transcript, &cell) in transcripts.iter().zip(params.cell_assignments.iter()) {
        //     // let position = clip_z_position(
        //     //     (transcript.x, transcript.y, transcript.z), zmin, zmax);
        //     // let loc_cell = cubecells.get(layout.world_pos_to_cube(position));
        //     // assert!(loc_cell == cell);
        // }


        let nzbins = 1;
        let cell_population = Array2::from_elem((nzbins, params.ncells()), 0.0_f32);
        let cell_perimeter = Array2::from_elem((nzbins, params.ncells()), 0.0_f32);

        let proposals = vec![CubeBinProposal::new(ngenes, nlayers); nchunks];
        let connectivity_checker = ThreadLocal::new();

        let mut sampler = CubeBinSampler {
            chunkquad: ChunkQuadMap {
                layout,
                xmin,
                ymin,
                chunk_size,
                nxchunks,
            },
            transcript_genes,
            transcript_layers,
            density,
            nlayers,
            mismatch_edges,
            cubeindex: Arc::new(RwLock::new(cubeindex)),
            cubecells,
            nzbins,
            cell_population,
            cell_perimeter,
            proposals,
            connectivity_checker,
            zmin,
            zmax,
            nvoxel_layers: 1,
            cubevolume,
            quad: 0,
        };

        sampler.recompute_cell_population();
        sampler.recompute_cell_perimeter();
        sampler.recompute_cell_volume(priors, params);
        sampler.populate_mismatches();

        return sampler;
    }

    fn ncells(&self) -> usize {
        return self.cell_population.shape()[1];
    }

    // Allocate a new RectBinSampler with the same state as this one, but
    // grid resolution doubled (i.e. rect size halved).
    pub fn double_resolution(&self, params: &ModelParams) -> CubeBinSampler {
        let nchunks = self.mismatch_edges[0].len();
        let ngenes = self.proposals[0].genepop.shape()[0];
        let cubevolume = self.cubevolume / 8.0;
        let layout = self.chunkquad.layout.double_resolution();

        let proposals = vec![CubeBinProposal::new(ngenes, self.nlayers); nchunks];
        let connectivity_checker = ThreadLocal::new();

        let mut cubecells = CubeCellMap::new();
        let mut cubebins = Vec::new();

        // 1.3s
        let t0 = Instant::now();
        // Build a set of every cube that is either populated with transcripts
        // or assigned to a cell.
        let mut cubeset = HashSet::<Cube>::new();
        for (&cube, &cell) in self.cubecells.iter() {
            if cell != BACKGROUND_CELL {
                cubeset.insert(cube);
            }
        }

        for cubebin in self.cubeindex.read().unwrap().values() {
            cubeset.insert(cubebin.cube);
        }
        println!("cubeset: {:?}", t0.elapsed());

        // 15.3s
        let t0 = Instant::now();
        for cube in cubeset {
            let cell = self.cubecells.get(cube);
            let subcubes = cube.double_resolution_children();

            if cell != BACKGROUND_CELL {
                // set cell states
                for subcube in &subcubes {
                    cubecells.insert(subcube.clone(), cell);
                }
            }

            // if this is a populated cube, need to build sub cube bins
            // if let Some(&cubebin_idx) = self.cubeindex.get(&cube) {
            //     let cubebin = &self.cubebins[cubebin_idx];
            if let Some(cubebin) = self.cubeindex.read().unwrap().get(&cube) {

                let mut subcubebins = [
                    CubeBin::new(subcubes[0]),
                    CubeBin::new(subcubes[1]),
                    CubeBin::new(subcubes[2]),
                    CubeBin::new(subcubes[3]),
                    CubeBin::new(subcubes[4]),
                    CubeBin::new(subcubes[5]),
                    CubeBin::new(subcubes[6]),
                    CubeBin::new(subcubes[7]),
                ];

                // allocate transcripts to children
                for &t in cubebin.transcripts.lock().unwrap().iter() {
                    let position = clip_z_position(
                        params.transcript_positions[t],
                        self.zmin, self.zmax);
                    let tcube =
                        layout.world_pos_to_cube(position);

                    let mut found = false;
                    for subcubebin in subcubebins.iter_mut() {
                        if subcubebin.cube == tcube {
                            subcubebin.transcripts.lock().unwrap().push(t);
                            found = true;
                            break;
                        }
                    }

                    assert!(found, "Unable to allocate transcript to sub-voxel");
                }

                // TODO:
                // Why only add cubebins when they are populated? Doesn't this
                // prevent mismatch edges from being added?

                // add to index
                for subcubebin in subcubebins {
                    if !subcubebin.transcripts.lock().unwrap().is_empty() {
                        cubebins.push(subcubebin);
                    }
                }
            }
        }
        println!("cubebins: {:?}", t0.elapsed());

        // build index
        // 1.2s
        let t0 = Instant::now();
        let mut cubeindex = HashMap::new();
        for cubebin in cubebins {
            cubeindex.insert(cubebin.cube, cubebin);
        }
        println!("cubeindex: {:?}", t0.elapsed());

        // initialize mismatch_edges
        let mut mismatch_edges = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for chunks in mismatch_edges.iter_mut() {
            for _ in 0..nchunks {
                chunks.push(Arc::new(Mutex::new(CubeEdgeSampleSet::new())));
            }
        }

        let nzbins = 2 * self.nzbins;
        let cell_population = Array2::from_elem((nzbins, self.cell_population.shape()[1]), 0.0_f32);
        let cell_perimeter = Array2::from_elem((nzbins, self.cell_perimeter.shape()[1]), 0.0_f32);

        let mut sampler = CubeBinSampler {
            chunkquad: ChunkQuadMap {
                layout,
                xmin: self.chunkquad.xmin,
                ymin: self.chunkquad.ymin,
                chunk_size: self.chunkquad.chunk_size,
                nxchunks: self.chunkquad.nxchunks,
            },
            transcript_genes: self.transcript_genes.clone(),
            transcript_layers: self.transcript_layers.clone(),
            density: self.density.clone(),
            nlayers: self.nlayers,
            mismatch_edges,
            cubeindex: Arc::new(RwLock::new(cubeindex)),
            cubecells,
            nzbins,
            cell_population,
            cell_perimeter,
            proposals,
            connectivity_checker,
            zmin: self.zmin,
            zmax: self.zmax,
            nvoxel_layers: 2 * self.nvoxel_layers,
            cubevolume,
            quad: 0,
        };

        // 11.3s
        let t0 = Instant::now();
        sampler.populate_mismatches();
        println!("populate_mismatches: {:?}", t0.elapsed());

        // 141ms
        let t0 = Instant::now();
        sampler.recompute_cell_population();
        println!("recompute_cell_population: {:?}", t0.elapsed());

        // 85.4s
        let t0 = Instant::now();
        sampler.recompute_cell_perimeter();
        println!("recompute_cell_perimeter: {:?}", t0.elapsed());

        return sampler;
    }

    fn recompute_cell_volume(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // recompute cell areas as the sum of rect areas
        params.cell_volume.fill(0.0_f32);
        for (_, &cell) in self.cubecells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }
            params.cell_volume[cell as usize] += self.cubevolume;
        }
        for cell_volume in params.cell_volume.iter_mut() {
            *cell_volume = cell_volume.max(priors.min_cell_volume);
        }
    }

    fn recompute_cell_population(&mut self) {
        self.cell_population.fill(0.0_f32);
        for (&cube, &cell) in self.cubecells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }
            self.cell_population[[cube.k as usize, cell as usize]] += 1.0_f32;
        }
    }

    fn recompute_cell_perimeter(&mut self) {
        // TODO: This function is super expensive. It's hard to update in parallel
        // though, each cell consists of many cubes. So we'd either need mutexes
        // or more likelily, organize things by cell first.

        self.cell_perimeter.fill(0.0_f32);
        for (&cube, &cell) in self.cubecells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }

            for neighbor in cube.radius2_xy_neighborhood() {
                let neighbor_cell = self.cubecells.get(neighbor);
                if neighbor_cell != cell {
                    self.cell_perimeter[[cube.k as usize, cell as usize]] += 1.0_f32;
                }
            }
        }
    }

    fn populate_mismatches(&mut self) {
        for (&cube, &cell) in self.cubecells.iter() {
            let (chunk, quad) = self.chunkquad.get(cube);
            for neighbor in cube.von_neumann_neighborhood() {
                // don't consider neighbors that are out of bounds on the z-axis
                if neighbor.k < 0 || neighbor.k >= self.nzbins as i32 {
                    continue;
                }

                let neighbor_cell = self.cubecells.get(neighbor);
                if cell != neighbor_cell {
                    let (neighbor_chunk, neighbor_quad) = self.chunkquad.get(neighbor);

                    let mismatch_edges = &self.mismatch_edges[quad as usize];
                    if (chunk as usize) < mismatch_edges.len() {
                        mismatch_edges[chunk as usize]
                            .lock()
                            .unwrap()
                            .insert((cube, neighbor));
                    }

                    let mismatch_edges = &self.mismatch_edges[neighbor_quad as usize];
                    if (neighbor_chunk as usize) < mismatch_edges.len() {
                        mismatch_edges[neighbor_chunk as usize]
                            .lock()
                            .unwrap()
                            .insert((neighbor, cube));
                    }
                }
            }
        }
    }

    pub fn cubes(&self) -> impl Iterator<Item = (CellIndex, (f32, f32, f32, f32, f32, f32))> + '_ {
        return self
            .cubecells
            .iter()
            .filter(|(_, &cell)| cell != BACKGROUND_CELL)
            .map(|(cube, cell)| (*cell, self.chunkquad.layout.cube_to_world_coords(*cube)));
    }

    pub fn cell_polygons(&self) -> Vec<MultiPolygon<f32>> {
        let mut cell_polys: Vec<Vec<Polygon<f32>>> = vec![Vec::new(); self.ncells()];
        for (cell, (x0, y0, _z0, x1, y1, _z1)) in self.cubes() {
            cell_polys[cell as usize].push(Polygon::<f32>::new(
                LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
                vec![],
            ))
        }

        let cell_polys = cell_polys
            .into_par_iter()
            .map(union_all_into_multipolygon)
            .map(|p| p.simplify(&1e-2))
            .collect();

        return cell_polys;
    }

    pub fn cell_layered_polygons(&self) -> Vec<(i32, Vec<MultiPolygon<f32>>)> {
        let mut cell_polys = HashMap::new();

        for (cube, &cell) in self.cubecells.iter() {
            if cell == BACKGROUND_CELL {
                continue;
            }

            let (x0, y0, _z0, x1, y1, _z1) = self.chunkquad.layout.cube_to_world_coords(*cube);

            let cell_polys = cell_polys
                .entry(cube.k)
                .or_insert_with(|| vec![Vec::new(); self.ncells()]);
            cell_polys[cell as usize].push(Polygon::<f32>::new(
                LineString::from(vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]),
                vec![],
            ))
        }

        let cell_polys = cell_polys
            .into_iter()
            .map(|(k, polys)| {
                (
                    k,
                    polys
                        .into_par_iter()
                        .map(union_all_into_multipolygon)
                        .map(|p| p.simplify(&1e-2))
                        .collect(),
                )
            })
            .collect();

        return cell_polys;
    }

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

impl Sampler<CubeBinProposal> for CubeBinSampler {
    fn repopulate_proposals(&mut self, priors: &ModelPriors, params: &ModelParams) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.05;

        self.proposals
            .par_iter_mut()
            .zip(&self.mismatch_edges[self.quad])
            .for_each(|(proposal, mismatch_edges)| {
                let mismatch_edges = mismatch_edges.lock().unwrap();
                if mismatch_edges.is_empty() {
                    proposal.ignore = true;
                    return;
                }

                let mut rng = thread_rng();
                let (i, j) = mismatch_edges.choose(&mut rng).unwrap();

                let cell_from = self.cubecells.get(*i);
                let mut cell_to = self.cubecells.get(*j);
                assert!(cell_from != cell_to);

                let from_unassigned = cell_from == BACKGROUND_CELL;
                let to_unassigned = cell_to == BACKGROUND_CELL;

                // don't let the cell grow in the z-dimension past the extents
                // of the data. Not strictly necessary to restrict it this way,
                // but lessens the tendency to produce weird shaped cells.
                if from_unassigned && !to_unassigned {
                    let (_, _, z0, _, _, z1) = self.chunkquad.layout.cube_to_world_coords(*i);
                    if z1 < self.zmin || z0 > self.zmax {
                        proposal.ignore = true;
                        return;
                    }
                }

                if !from_unassigned && rng.gen::<f64>() < UNASSIGNED_PROPOSAL_PROB {
                    cell_to = BACKGROUND_CELL;
                }

                let cubeindex = self.cubeindex.read().unwrap();
                let cubebin = cubeindex.get(i);

                let transcripts = cubebin.map(|cubebin| &cubebin.transcripts);
                let i_pop = transcripts.map(|ts| ts.lock().unwrap().len()).unwrap_or(0);

                // Don't propose removing the last transcript from a cell. (This is
                // breaks the markov chain balance, since there's no path back to the previous state.)
                //
                // We could allow this if we introduce a spontaneous nucleation move.
                if !from_unassigned && params.cell_population[cell_from as usize] == i_pop {
                    proposal.ignore = true;
                    return;
                }

                // Local connectivity condition: don't propose changes that render increase the
                // number of connected components of either the cell_from or cell_to
                // neighbors subgraphs.
                let mut connectivity_checker = self
                    .connectivity_checker
                    .get_or(|| RefCell::new(ConnectivityChecker::new()))
                    .borrow_mut();

                let art_from = connectivity_checker.cube_isarticulation(
                    *i,
                    |cube| self.cubecells.get(cube),
                    cell_from,
                );

                let art_to = connectivity_checker.cube_isarticulation(
                    *i,
                    |cube| self.cubecells.get(cube),
                    cell_to,
                );

                if (art_from && !from_unassigned) || (art_to && !to_unassigned) {
                    proposal.ignore = true;
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = mismatch_edges.len();

                let num_new_state_neighbors = i
                    .von_neumann_neighborhood()
                    .iter()
                    .filter(|&&j| {
                        j.inbounds(self.nvoxel_layers) && self.cubecells.get(j) == cell_to
                    })
                    .count();

                let num_prev_state_neighbors = i
                    .von_neumann_neighborhood()
                    .iter()
                    .filter(|&&j| {
                        j.inbounds(self.nvoxel_layers) && self.cubecells.get(j) == cell_from
                    })
                    .count();

                let mut proposal_prob =
                    num_new_state_neighbors as f64 / num_mismatching_edges as f64;

                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if to_unassigned {
                    let num_mismatching_neighbors = i
                        .von_neumann_neighborhood()
                        .iter()
                        .filter(|&&j| self.cubecells.get(j) != cell_from)
                        .count();

                    proposal_prob = UNASSIGNED_PROPOSAL_PROB
                        * (num_mismatching_neighbors as f64 / num_mismatching_edges as f64)
                        + (1.0 - UNASSIGNED_PROPOSAL_PROB) * proposal_prob;
                }

                let new_num_mismatching_edges = num_mismatching_edges
                    + 2*num_prev_state_neighbors // edges that are newly mismatching
                    - 2*num_new_state_neighbors; // edges that are newly matching

                let mut reverse_proposal_prob =
                    num_prev_state_neighbors as f64 / new_num_mismatching_edges as f64;

                // If this is a proposal from unassigned, account for multiple ways of reversing it
                if from_unassigned {
                    let new_num_mismatching_neighbors = i
                        .von_neumann_neighborhood()
                        .iter()
                        .filter(|&&j| self.cubecells.get(j) != cell_to)
                        .count();
                    reverse_proposal_prob = UNASSIGNED_PROPOSAL_PROB
                        * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64)
                        + (1.0 - UNASSIGNED_PROPOSAL_PROB) * reverse_proposal_prob;
                }

                proposal.cube = *i;
                if let Some(transcripts) = transcripts {
                    proposal.transcripts.clone_from(&transcripts.lock().unwrap());
                } else {
                    proposal.transcripts.clear();
                }
                proposal.old_cell = cell_from;
                proposal.new_cell = cell_to;
                proposal.log_weight = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;
                proposal.ignore = false;
                proposal.accept = false;
                proposal.old_cell_volume_delta = -self.cubevolume;
                proposal.new_cell_volume_delta = self.cubevolume;

                proposal.old_cell_perimeter_delta = 0.0;
                proposal.new_cell_perimeter_delta = 0.0;

                for neighbor in i.radius2_xy_neighborhood() {
                    let neighbor_cell = self.cubecells.get(neighbor);

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
                if cell_from != BACKGROUND_CELL && proposal.old_cell_perimeter_delta > 0.0 {
                    let old_cell_perimeter = self.cell_perimeter
                        [[i.k as usize, cell_from as usize]]
                        + proposal.old_cell_perimeter_delta;
                    let pop = self.cell_population[[i.k as usize, cell_from as usize]] - 1.0;
                    let bound = perimeter_bound(priors.perimeter_eta, priors.perimeter_bound, pop);
                    if old_cell_perimeter > bound {
                        proposal.ignore = true;
                        return;
                    }
                }

                if cell_to != BACKGROUND_CELL && proposal.new_cell_perimeter_delta > 0.0 {
                    let new_cell_perimeter = self.cell_perimeter[[i.k as usize, cell_to as usize]]
                        + proposal.new_cell_perimeter_delta;
                    let pop = self.cell_population[[i.k as usize, cell_to as usize]] + 1.0;
                    let bound = perimeter_bound(priors.perimeter_eta, priors.perimeter_bound, pop);
                    if new_cell_perimeter > bound {
                        proposal.ignore = true;
                        return;
                    }
                }

                proposal.genepop.fill(0);
                if let Some(transcripts) = transcripts {
                    for &t in transcripts.lock().unwrap().iter() {
                        let layer = self.transcript_layers[t] as usize;
                        proposal.genepop[[self.transcript_genes[t] as usize, layer]] += 1;
                    }
                }

                // average the density from the constitutive transcripts
                proposal.density.fill(1e-2);
                if let Some(transcripts) = transcripts {
                    let genepop = proposal.genepop.sum_axis(Axis(1));

                    for &t in transcripts.lock().unwrap().iter() {
                        let g = self.transcript_genes[t] as usize;
                        proposal.density[g] += self.density[t] / genepop[g] as f32;
                    }
                }
            });

        // Increment so we run updates on the next quad
        self.quad = (self.quad + 1) % 4;
    }

    fn proposals<'a, 'b>(&'a self) -> &'b [CubeBinProposal]
    where
        'a: 'b,
    {
        return &self.proposals;
    }

    fn proposals_mut<'a, 'b>(&'a mut self) -> &'b mut [CubeBinProposal]
    where
        'a: 'b,
    {
        return &mut self.proposals;
    }

    fn update_sampler_state(&mut self, _: &ModelParams) {
        for proposal in self.proposals.iter().filter(|p| !p.ignore && p.accept) {
            self.cubecells.set(proposal.cube, proposal.new_cell);

            // update cell population and perimeter
            if proposal.old_cell != BACKGROUND_CELL {
                self.cell_population[[proposal.cube.k as usize, proposal.old_cell as usize]] -=
                    1.0_f32;
                self.cell_perimeter[[proposal.cube.k as usize, proposal.old_cell as usize]] +=
                    proposal.old_cell_perimeter_delta;
            }

            if proposal.new_cell != BACKGROUND_CELL {
                self.cell_population[[proposal.cube.k as usize, proposal.new_cell as usize]] +=
                    1.0_f32;
                self.cell_perimeter[[proposal.cube.k as usize, proposal.new_cell as usize]] +=
                    proposal.new_cell_perimeter_delta;
            }
        }

        self.proposals
            .par_iter()
            .filter(|p| !p.ignore && p.accept)
            .for_each(|proposal| {
                let (chunk, quad) = self.chunkquad.get(proposal.cube);

                // update mismatch edges
                for neighbor in proposal.cube.von_neumann_neighborhood() {
                    if neighbor.k < 0 || neighbor.k >= self.nzbins as i32 {
                        continue;
                    }

                    let (neighbor_chunk, neighbor_quad) = self.chunkquad.get(neighbor);
                    let neighbor_cell = self.cubecells.get(neighbor);
                    if proposal.new_cell == neighbor_cell {
                        let mismatch_edges = &self.mismatch_edges[quad as usize];
                        if (chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[chunk as usize]
                                .lock()
                                .unwrap()
                                .remove((proposal.cube, neighbor));
                        }

                        let mismatch_edges = &self.mismatch_edges[neighbor_quad as usize];
                        if (neighbor_chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[neighbor_chunk as usize]
                                .lock()
                                .unwrap()
                                .remove((neighbor, proposal.cube));
                        }
                    } else {
                        let mismatch_edges = &self.mismatch_edges[quad as usize];
                        if (chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[chunk as usize]
                                .lock()
                                .unwrap()
                                .insert((proposal.cube, neighbor));
                        }

                        let mismatch_edges = &self.mismatch_edges[neighbor_quad as usize];
                        if (neighbor_chunk as usize) < mismatch_edges.len() {
                            mismatch_edges[neighbor_chunk as usize]
                                .lock()
                                .unwrap()
                                .insert((neighbor, proposal.cube));
                        }
                    }
                }
            });
    }

    fn cell_at_position(&self, position: (f32, f32, f32)) -> u32 {
        let position = clip_z_position(position, self.zmin, self.zmax);
        let cubindex = self.chunkquad.layout.world_pos_to_cube(position);
        return self.cubecells.get(cubindex);
    }

    fn update_transcript_position(&mut self, i: usize, prev_pos: (f32, f32, f32), new_pos: (f32, f32, f32)) {
        let prev_pos = clip_z_position(prev_pos, self.zmin, self.zmax);
        let old_cube = self
            .chunkquad.layout.world_pos_to_cube(prev_pos);

        let new_pos = clip_z_position(new_pos, self.zmin, self.zmax);
        let new_cube = self
            .chunkquad.layout.world_pos_to_cube(new_pos);

        if old_cube == new_cube {
            return;
        }

        // remove transcript from old cube
        {
            let cubeindex = self.cubeindex.read().unwrap();
            let old_cubebin = cubeindex.get(&old_cube).unwrap();
            let transcripts = &mut old_cubebin.transcripts.lock().unwrap();
            let idx = transcripts.iter().position(|t| *t == i).unwrap();
            transcripts.swap_remove(idx);
        }

        // insert transcript into old cube
        {
            let cubeindex = &mut self.cubeindex.write().unwrap();
            let new_cubebin = cubeindex
                .entry(new_cube)
                .or_insert_with(|| CubeBin::new(new_cube));
            new_cubebin.transcripts.lock().unwrap().push(i);
        }
    }

    // TODO: This can't really be effectively parallelized because there's too
    // much contention on `cubeindex`.
    //
    // I don't know what can be done about this. Moving transcripts around creates
    // a bunch of new cubes. There's not much we can do about that. Any way to
    // speed this up seems like it would require an entirely different design.
    //
    // 1. Just keep track of which transcripts are in which chunk. There would
    //    still be a lot of contention because we need to need to move transcripts
    //    in and out of chunks.
    //
    // 2. Don't keep track of anything. Each proposal will just loop through the
    //    transcript positions selecting transcripts. I'm afraid this would slow
    //    things down a lot.
    //
    // 3. Index positions somehow (maybe a kd_tree) before generating proposals.
    //    This seems like the most promising approach. Just have to figure
    //    out the indexing scheme.

    fn update_transcript_positions(
        &mut self,
        accept: &Vec<bool>,
        positions: &Vec<(f32, f32, f32)>,
        proposed_positions: &Vec<(f32, f32, f32)>)
    {
        accept
            .par_iter()
            .zip(positions)
            .zip(proposed_positions)
            .enumerate()
            .for_each(|(i, ((accept, position), proposed_position))| {
                if !*accept {
                    return;
                }

                let prev_pos = clip_z_position(*position, self.zmin, self.zmax);
                let old_cube = self
                    .chunkquad.layout.world_pos_to_cube(prev_pos);

                let new_pos = clip_z_position(*proposed_position, self.zmin, self.zmax);
                let new_cube = self
                    .chunkquad.layout.world_pos_to_cube(new_pos);

                if old_cube == new_cube {
                    return;
                }

                // remove transcript from old cube
                {
                    let cubeindex = self.cubeindex.read().unwrap();
                    let old_cubebin = cubeindex.get(&old_cube).unwrap();
                    let transcripts = &mut old_cubebin.transcripts.lock().unwrap();
                    let idx = transcripts.iter().position(|t| *t == i).unwrap();
                    transcripts.swap_remove(idx);
                }

                // insert transcript into old cube
                {
                    // if let Some(new_cubebin) = self.cubeindex.read().unwrap().get(&old_cube) {
                    //     new_cubebin.transcripts.lock().unwrap().push(i);
                    // } else {
                        let cubeindex = &mut self.cubeindex.write().unwrap();
                        let new_cubebin = cubeindex
                            .entry(new_cube)
                            .or_insert_with(|| CubeBin::new(new_cube));
                        new_cubebin.transcripts.lock().unwrap().push(i);
                    // }
                }
            });
    }


}

#[derive(Clone, Debug)]
pub struct CubeBinProposal {
    cube: Cube,
    transcripts: Vec<usize>,

    // [ngenes, nlayers] gene count for this rect
    genepop: Array2<u32>,

    // transcript density
    density: Array1<f32>,

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

impl CubeBinProposal {
    fn new(ngenes: usize, nlayers: usize) -> CubeBinProposal {
        return CubeBinProposal {
            cube: Cube::new(0, 0, 0),
            transcripts: Vec::new(),
            genepop: Array2::from_elem((ngenes, nlayers), 0),
            density: Array1::zeros(ngenes),
            old_cell: 0,
            new_cell: 0,
            log_weight: 0.0,
            ignore: false,
            accept: false,
            old_cell_volume_delta: 0.0,
            new_cell_volume_delta: 0.0,
            old_cell_perimeter_delta: 0.0,
            new_cell_perimeter_delta: 0.0,
        };
    }
}

impl Proposal for CubeBinProposal {
    fn accept(&mut self) {
        self.accept = true;
    }
    fn reject(&mut self) {
        self.accept = false;
    }

    fn ignored(&self) -> bool {
        return self.ignore;
    }
    fn accepted(&self) -> bool {
        return self.accept;
    }

    fn old_cell(&self) -> u32 {
        return self.old_cell;
    }

    fn new_cell(&self) -> u32 {
        return self.new_cell;
    }

    fn old_cell_volume_delta(&self) -> f32 {
        return self.old_cell_volume_delta;
    }

    fn new_cell_volume_delta(&self) -> f32 {
        return self.new_cell_volume_delta;
    }

    fn log_weight(&self) -> f32 {
        return self.log_weight;
    }

    fn transcripts<'b, 'c>(&'b self) -> &'c [usize]
    where
        'b: 'c,
    {
        return self.transcripts.as_slice();
    }

    fn gene_count<'b, 'c>(&'b self) -> &'c Array2<u32>
    where
        'b: 'c,
    {
        return &self.genepop;
    }

    fn density<'b, 'c>(&'b self) -> &'c Array1<f32>
    where
        'b: 'c,
    {
        return &self.density;
    }
}

// We need to exclude cells that can't be initalized with a non-zero number of voxels.
pub fn filter_sparse_cells(
    scale: f32,
    transcripts: &Vec<Transcript>,
    cell_assignments: &mut Vec<CellIndex>,
    cell_population: &mut Vec<usize>,
) {
    let (_, cubebins) = bin_transcripts(transcripts, scale);
    let cubecells = cube_assignments(&cubebins, &cell_assignments);

    let mut used_cell_ids: HashMap<CellIndex, CellIndex> = HashMap::new();
    for (_, cell_id) in cubecells.iter() {
        if *cell_id != BACKGROUND_CELL {
            let next_id = used_cell_ids.len() as CellIndex;
            used_cell_ids.entry(*cell_id).or_insert(next_id);
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

    cell_population.resize(used_cell_ids.len(), 0);
    cell_population.fill(0);
    for cell_id in cell_assignments.iter() {
        if *cell_id != BACKGROUND_CELL {
            cell_population[*cell_id as usize] += 1;
        }
    }
}
