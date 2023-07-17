

use super::transcripts::{Transcript, coordinate_span, BACKGROUND_CELL};
use super::{Sampler, ModelPriors, ModelParams, Proposal, chunkquad};
use super::sampleset::SampleSet;
use super::connectivity::ConnectivityChecker;

// use hexx::{Hex, HexLayout, HexOrientation, Vec2};
use std::collections::HashMap;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use thread_local::ThreadLocal;
use rand::{thread_rng, Rng};
use rayon::prelude::*;


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rect {
    i: i32,
    j: i32,
}

impl Rect {
    fn new(i: i32, j: i32) -> Rect {
        return Rect {
            i: i,
            j: j,
        };
    }

    pub fn moore_neighborhood(&self) -> [Rect; 8] {
        return [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)].map(
                |(di, dj)| Rect::new(self.i + di, self.j + dj));
    }

    pub fn von_neumann_neighborhood(&self) -> [Rect; 4] {
        return [(-1, 0), (1, 0), (0, -1), (0, 1)].map(
            |(di, dj)| Rect::new(self.i + di, self.j + dj));
    }

    fn double_resolution_children(&self) -> [Rect; 4] {
        return [
            Rect::new(2*self.i, 2*self.j),
            Rect::new(2*self.i + 1, 2*self.j),
            Rect::new(2*self.i, 2*self.j + 1),
            Rect::new(2*self.i + 1, 2*self.j + 1),
        ];
    }
}

struct RectLayout {
    origin: (f32, f32),
    rect_size: (f32, f32),
}

impl RectLayout {
    // fn new(origin: (f32, f32), rect_size: (f32, f32)) -> RectLayout {
    //     return RectLayout {
    //         origin,
    //         rect_size,
    //     };
    // }

    fn double_resolution(&self) -> RectLayout {
        return RectLayout {
            origin: (self.origin.0, self.origin.1),
            rect_size: (self.rect_size.0 / 2.0, self.rect_size.1 / 2.0),
        };
    }

    fn rect_to_world_pos(&self, rect: Rect) -> (f32, f32) {
        return (
            self.origin.0 + (0.5 + rect.i as f32) * self.rect_size.0,
            self.origin.1 + (0.5 + rect.j as f32) * self.rect_size.1);
    }

    fn world_pos_to_rect(&self, pos: (f32, f32)) -> Rect {
        return Rect::new(
            ((pos.0 - self.origin.0) / self.rect_size.0).floor() as i32,
            ((pos.1 - self.origin.1) / self.rect_size.1).floor() as i32);
    }
}

type RectEdgeSampleSet = SampleSet<(Rect, Rect)>;


#[derive(Clone, Debug)]
struct RectBin {
    rect: Rect,
    transcripts: Vec<usize>,
}

impl RectBin {
    fn new(rect: Rect) -> Self {
        Self {
            rect,
            transcripts: Vec::new(),
        }
    }
}

struct ChunkQuadMap {
    layout: RectLayout,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,
}


impl ChunkQuadMap {
    fn get(&self, rect: Rect) -> (u32, u32) {
        let rect_xy = self.layout.rect_to_world_pos(rect);
        return chunkquad(rect_xy.0, rect_xy.1, self.xmin, self.ymin, self.chunk_size, self.nxchunks);
    }
}


struct RectCellMap {
    index: HashMap<Rect, u32>,
}


impl RectCellMap {
    fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    fn insert(&mut self, rect: Rect, cell: u32) {
        self.index.insert(rect, cell);
    }

    fn get(&self, rect: Rect) -> u32 {
        match self.index.get(&rect) {
            Some(cell) => *cell,
            None => BACKGROUND_CELL,
        }
    }

    fn set(&mut self, rect: Rect, cell: u32) {
        self.index.insert(rect, cell);
    }

    // fn count(&self, cell: u32) -> usize {
    //     return self.index.values().filter(|&&c| c == cell).count();
    // }
}


fn bin_transcripts(transcripts: &Vec<Transcript>, full_area: f32, avgpop: f32) -> (RectLayout, Vec<RectBin>) {
    let density = transcripts.len() as f32 / full_area;
    let target_area = avgpop / density;
    let rect_size = target_area.sqrt();

    let layout = RectLayout {
        origin: (0.0, 0.0),
        rect_size: (rect_size, rect_size),
    };

    // Bin transcripts into RectBins
    let mut rect_index = HashMap::new();

    for (i, transcript) in transcripts.iter().enumerate() {
        let rect = layout.world_pos_to_rect((transcript.x, transcript.y));

        rect_index.entry(rect)
            .or_insert_with(|| RectBin::new(rect))
            .transcripts.push(i);
    }

    let rectbins = rect_index.values().cloned().collect::<Vec<_>>();

    return (layout, rectbins);
}

pub struct RectBinSampler {
    chunkquad: ChunkQuadMap,
    transcript_genes: Vec<u32>,

    mismatch_edges: [Vec<Arc<Mutex<RectEdgeSampleSet>>>; 4],
    rectbins: Vec<RectBin>,
    rectindex: HashMap<Rect, usize>,

    // assignment of rectbins to cells
    // (Unassigned cells are either absent or set to `BACKGROUND_CELL`)
    rectcells: RectCellMap,

    proposals: Vec<RectBinProposal>,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,

    rectarea: f32,
    quad: usize,
}


impl RectBinSampler {
    pub fn new(
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        ngenes: usize,
        full_area: f32,
        avgrectpop: f32,
        chunk_size: f32) -> Self
    {
        let (xmin, xmax, ymin, ymax) = coordinate_span(transcripts);
        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;

        let (layout, rectbins ) = bin_transcripts(transcripts, full_area, avgrectpop);

        let transcript_genes = transcripts.iter().map(|t| t.gene).collect::<Vec<_>>();

        assert!(layout.rect_size.0 == layout.rect_size.1);
        let rect_size = layout.rect_size.0;

        dbg!(rect_size);
        let rectarea = rect_size.powi(2);

        // build index
        let mut rectindex = HashMap::new();
        for (i, rectbin) in rectbins.iter().enumerate() {
            rectindex.insert(rectbin.rect, i);
        }

        // initialize mismatch_edges
        let mut mismatch_edges = [
            Vec::new(), Vec::new(), Vec::new(), Vec::new() ];
        for chunks in mismatch_edges.iter_mut() {
            for _ in 0..nchunks {
                chunks.push(Arc::new(Mutex::new(RectEdgeSampleSet::new())));
            }
        }

        // initial assignments
        let mut rect_assignments = HashMap::new();
        let mut rectcells = RectCellMap::new();
        for rectbin in &rectbins {
            rect_assignments.clear();

            // vote on rect assignment
            for &t in rectbin.transcripts.iter() {
                rect_assignments
                    .entry(params.cell_assignments[t])
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }

            let winner = rect_assignments.iter()
                .max_by_key(|(_, count)| *count)
                .map(|(cell, _)| *cell).unwrap_or(BACKGROUND_CELL);

            if winner != BACKGROUND_CELL {
                rectcells.insert(rectbin.rect, winner);
            }

            // homogenize the rect: assign every transcript in the rect to the winner
            for &t in rectbin.transcripts.iter() {
                if params.cell_assignments[t] != BACKGROUND_CELL {
                    params.cell_population[params.cell_assignments[t] as usize] -= 1;
                }
                if winner != BACKGROUND_CELL {
                    params.cell_population[winner as usize] += 1;
                }
                params.cell_assignments[t] = winner;
            }
        }
        params.recompute_counts(transcripts);


        let proposals = vec![RectBinProposal::new(ngenes); nchunks];
        let connectivity_checker = ThreadLocal::new();

        let mut sampler = RectBinSampler {
            chunkquad: ChunkQuadMap {
                layout,
                xmin,
                ymin,
                chunk_size,
                nxchunks,
            },
            transcript_genes,
            mismatch_edges,
            rectbins,
            rectindex,
            rectcells,
            proposals,
            connectivity_checker,
            rectarea,
            quad: 0,
        };

        sampler.recompute_cell_areas(priors, params);
        sampler.populate_mismatches();

        return sampler;
    }

    // Allocate a new RectBinSampler with the same state as this one, but
    // grid resolution doubled (i.e. rect size halved).
    pub fn double_resolution(&self, transcripts: &Vec<Transcript>) -> RectBinSampler {
        let nchunks = self.mismatch_edges[0].len();
        let ngenes = self.proposals[0].genepop.len();
        let rectarea = self.rectarea / 4.0;
        dbg!(rectarea.sqrt());
        let layout = self.chunkquad.layout.double_resolution();

        let proposals = vec![RectBinProposal::new(ngenes); nchunks];
        let connectivity_checker = ThreadLocal::new();

        let mut rectcells = RectCellMap::new();
        let mut rectbins = Vec::new();
        for rectbin in &self.rectbins {
            let cell = self.rectcells.get(rectbin.rect);
            if rectbin.transcripts.is_empty() && cell == BACKGROUND_CELL {
                continue;
            }

            let subrects = rectbin.rect.double_resolution_children();

            let mut subrectbins = [
                RectBin::new(subrects[0]),
                RectBin::new(subrects[1]),
                RectBin::new(subrects[2]),
                RectBin::new(subrects[3]),
            ];

            // allocate transcripts to children
            for &t in rectbin.transcripts.iter() {
                let transcript = &transcripts[t];
                let trect = layout.world_pos_to_rect((transcript.x, transcript.y));

                for subrectbin in subrectbins.iter_mut() {
                    if subrectbin.rect == trect {
                        subrectbin.transcripts.push(t);
                        break;
                    }
                }
            }

            // set cell states
            for subrectbin in &subrectbins {
                rectcells.insert(subrectbin.rect, cell);
            }

            // add to index
            for subrectbin in subrectbins {
                rectbins.push(subrectbin);
            }
        }

        // DEBUG: check that all transcripts are accounted for
        let mut prev_ntranscripts = 0;
        for rectbin in &self.rectbins {
            prev_ntranscripts += rectbin.transcripts.len();
        }
        let mut curr_ntranscripts = 0;
        for rectbin in &rectbins {
            curr_ntranscripts += rectbin.transcripts.len();
        }
        dbg!(prev_ntranscripts, curr_ntranscripts);

        // build index
        let mut rectindex = HashMap::new();
        for (i, rectbin) in rectbins.iter().enumerate() {
            rectindex.insert(rectbin.rect, i);
        }

        // initialize mismatch_edges
        let mut mismatch_edges = [
            Vec::new(), Vec::new(), Vec::new(), Vec::new() ];
        for chunks in mismatch_edges.iter_mut() {
            for _ in 0..nchunks {
                chunks.push(Arc::new(Mutex::new(RectEdgeSampleSet::new())));
            }
        }

        let mut sampler = RectBinSampler {
            chunkquad: ChunkQuadMap {
                layout,
                xmin: self.chunkquad.xmin,
                ymin: self.chunkquad.ymin,
                chunk_size: self.chunkquad.chunk_size,
                nxchunks: self.chunkquad.nxchunks,
            },
            transcript_genes: self.transcript_genes.clone(),
            mismatch_edges,
            rectbins,
            rectindex,
            rectcells,
            proposals,
            connectivity_checker,
            rectarea,
            quad: 0,
        };

        // TODO: areas should not change if we did this correctly
        // sampler.recompute_cell_areas(priors, params);

        sampler.populate_mismatches();

        return sampler;
    }

    fn recompute_cell_areas(&self, priors: &ModelPriors, params: &mut ModelParams) {
        // recompute cell areas as the sum of rect areas
        params.cell_areas.fill(0.0_f32);
        for rectbin in &self.rectbins {
            let cell = self.rectcells.get(rectbin.rect);
            if cell != BACKGROUND_CELL {
                params.cell_areas[cell as usize] += self.rectarea;
            }
        }
        for cell_area in params.cell_areas.iter_mut() {
            *cell_area = cell_area.max(priors.min_cell_area);
        }
    }

    fn populate_mismatches(&mut self) {
        // compute initial mismatch edges
        for rectbin in &self.rectbins {
            let cell = self.rectcells.get(rectbin.rect);
            let (chunk, quad) = self.chunkquad.get(rectbin.rect);

            for neighbor in rectbin.rect.von_neumann_neighborhood() {
                let neighbor_cell = self.rectcells.get(neighbor);
                if cell != neighbor_cell {
                    let (neighbor_chunk, neighbor_quad) = self.chunkquad.get(neighbor);

                    self.mismatch_edges[quad as usize][chunk as usize]
                        .lock()
                        .unwrap()
                        .insert((rectbin.rect, neighbor));

                    self.mismatch_edges[neighbor_quad as usize][neighbor_chunk as usize]
                        .lock()
                        .unwrap()
                        .insert((neighbor, rectbin.rect));
                }
            }
        }
    }

    // fn chunkquad(&self, hex: Hex) -> (u32, u32) {
    //     let hex_xy = self.layout.hex_to_world_pos(hex);
    //     return chunkquad(hex_xy.x, hex_xy.y, self.xmin, self.ymin, self.chunk_size, self.nxchunks);
    // }
}


impl Sampler<RectBinProposal> for RectBinSampler {
    fn repopulate_proposals(&mut self, params: &ModelParams) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.05;

        self.proposals
            .par_iter_mut()
            .zip(&self.mismatch_edges[self.quad])
            .for_each(|(proposal, mismatch_edges)|
            {
                let mismatch_edges = mismatch_edges.lock().unwrap();
                if mismatch_edges.is_empty() {
                    proposal.ignore = true;
                    return;
                }

                let mut rng = thread_rng();
                let (i, j) = mismatch_edges.choose(&mut rng).unwrap();

                let cell_from = self.rectcells.get(*i);
                let mut cell_to = self.rectcells.get(*j);
                assert!(cell_from != cell_to);

                let from_unassigned = cell_from == BACKGROUND_CELL;
                let to_unassigned = cell_to == BACKGROUND_CELL;

                if !from_unassigned && rng.gen::<f64>() < UNASSIGNED_PROPOSAL_PROB {
                    cell_to = BACKGROUND_CELL;
                }

                let rectbin = self.rectindex.get(i).map(|i| &self.rectbins[*i]);

                let transcripts = rectbin.map(
                    |rectbin| &rectbin.transcripts);
                let i_pop = transcripts.map(|ts| ts.len()).unwrap_or(0);

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

                let art_from = connectivity_checker.rect_isarticulation(
                    *i,
                    |rect| self.rectcells.get(rect),
                    cell_from);

                let art_to = connectivity_checker.rect_isarticulation(
                    *i,
                    |rect| self.rectcells.get(rect),
                    cell_to);

                if art_from || art_to {
                    proposal.ignore = true;
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = mismatch_edges.len();

                let num_new_state_neighbors = i.von_neumann_neighborhood().iter()
                    .filter(|&&j| self.rectcells.get(j) == cell_to)
                    .count();

                let num_prev_state_neighbors = i.von_neumann_neighborhood().iter()
                    .filter(|&&j| self.rectcells.get(j) == cell_from)
                    .count();

                let mut proposal_prob =
                    num_new_state_neighbors as f64 / num_mismatching_edges as f64;

                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if to_unassigned {
                    let num_mismatching_neighbors = i.von_neumann_neighborhood().iter()
                        .filter(|&&j| self.rectcells.get(j) != cell_from)
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
                    let new_num_mismatching_neighbors = i.von_neumann_neighborhood().iter()
                        .filter(|&&j| self.rectcells.get(j) != cell_to)
                        .count();
                    reverse_proposal_prob = UNASSIGNED_PROPOSAL_PROB
                        * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64)
                        + (1.0 - UNASSIGNED_PROPOSAL_PROB) * reverse_proposal_prob;
                }

                proposal.rect = *i;
                if let Some(transcripts) = transcripts {
                    proposal.transcripts.clone_from(transcripts);
                } else {
                    proposal.transcripts.clear();
                }
                proposal.old_cell = cell_from;
                proposal.new_cell = cell_to;
                proposal.log_weight = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;
                proposal.ignore = false;
                proposal.accept = false;
                proposal.old_cell_area_delta = -self.rectarea;
                proposal.new_cell_area_delta = self.rectarea;

                proposal.genepop.fill(0);
                if let Some(transcripts) = transcripts {
                    for &t in transcripts.iter() {
                        proposal.genepop[self.transcript_genes[t] as usize] += 1;
                    }
                }
            });

        // Increment so we run updates on the next quad
        self.quad = (self.quad + 1) % 4;
    }

    fn proposals<'a, 'b>(&'a self) -> &'b [RectBinProposal] where 'a: 'b {
        return &self.proposals;
    }

    fn proposals_mut<'a, 'b>(&'a mut self) -> &'b mut [RectBinProposal] where 'a: 'b {
        return &mut self.proposals;
    }

    fn update_sampler_state(&mut self, _: &ModelParams) {
        for proposal in self.proposals.iter().filter(|p| !p.ignore && p.accept) {
            self.rectcells.set(proposal.rect, proposal.new_cell);
        }

        self.proposals.par_iter()
            .filter(|p| !p.ignore && p.accept)
            .for_each(|proposal| {
                let (chunk, quad) = self.chunkquad.get(proposal.rect);

                // update mismatch edges
                for neighbor in proposal.rect.von_neumann_neighborhood() {
                    let (neighbor_chunk, neighbor_quad) = self.chunkquad.get(neighbor);
                    let neighbor_cell = self.rectcells.get(neighbor);
                    if proposal.new_cell == neighbor_cell {
                        self.mismatch_edges[quad as usize][chunk as usize]
                            .lock()
                            .unwrap()
                            .remove((proposal.rect, neighbor));
                        self.mismatch_edges[neighbor_quad as usize][neighbor_chunk as usize]
                            .lock()
                            .unwrap()
                            .remove((neighbor, proposal.rect));
                    } else {
                        self.mismatch_edges[quad as usize][chunk as usize]
                            .lock()
                            .unwrap()
                            .insert((proposal.rect, neighbor));
                        self.mismatch_edges[neighbor_quad as usize][neighbor_chunk as usize]
                            .lock()
                            .unwrap()
                            .insert((neighbor, proposal.rect));
                    }
                }
            });
    }
}

#[derive(Clone, Debug)]
pub struct RectBinProposal {
    rect: Rect,
    transcripts: Vec<usize>,

    // gene count for this rect
    genepop: Vec<u32>,

    old_cell: u32,
    new_cell: u32,

    // metroplis-hastings proposal weight weight
    log_weight: f32,

    ignore: bool,
    accept: bool,

    // updated cell areas and logprobs if the proposal is accepted
    old_cell_area_delta: f32,
    new_cell_area_delta: f32,
}

impl RectBinProposal {
    fn new(ngenes: usize) -> RectBinProposal {
        return RectBinProposal {
            rect: Rect::new(0, 0),
            transcripts: Vec::new(),
            genepop: vec![0; ngenes],
            old_cell: 0,
            new_cell: 0,
            log_weight: 0.0,
            ignore: false,
            accept: false,
            old_cell_area_delta: 0.0,
            new_cell_area_delta: 0.0,
        };
    }
}

impl Proposal for RectBinProposal {
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

    fn old_cell_area_delta(&self) -> f32 {
        return self.old_cell_area_delta;
    }

    fn new_cell_area_delta(&self) -> f32 {
        return self.new_cell_area_delta;
    }

    fn log_weight(&self) -> f32 {
        return self.log_weight;
    }

    fn transcripts<'b, 'c>(&'b self) -> &'c [usize] where 'b: 'c {
        return self.transcripts.as_slice();
    }

    fn gene_count<'b, 'c>(&'b self) -> &'c [u32] where 'b: 'c {
        return self.genepop.as_slice();
    }

}

