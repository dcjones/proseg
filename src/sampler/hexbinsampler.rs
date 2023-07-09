

use crate::sampler::connectivity;

use super::transcripts::{Transcript, coordinate_span};
use super::{Sampler, ModelPriors, ModelParams, Proposal, ChunkQuad, chunkquad};
use super::sampleset::SampleSet;
use super::connectivity::ConnectivityChecker;

use hexx::{Hex, HexLayout, HexOrientation, Vec2};
use std::arch::x86_64::_popcnt32;
use std::collections::HashMap;
use std::cell::{RefCell, RefMut};
use std::iter::{Cloned, Enumerate};
use std::slice;
use thread_local::ThreadLocal;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

#[derive(Clone, Debug)]
struct HexBin {
    hex: Hex,
    transcripts: Vec<usize>,
}

impl HexBin {
    fn new(hex: Hex) -> Self {
        Self {
            hex,
            transcripts: Vec::new(),
        }
    }
}

// struct ChunkedHexBin {
//     chunk: u32,
//     quad: u32,
//     hexbin: HexBin,
// }

// fn chunk_hexbins(
//     layout: &HexLayout,
//     hexbins: &Vec<HexBin>,
//     xmin: f32,
//     ymin: f32,
//     chunk_size: f32,
//     nxchunks: usize) -> Vec<ChunkedHexBin>
// {
//     return hexbins
//         .iter()
//         .map(|hexbin| {
//             let hex_xy = layout.hex_to_world_pos(hexbin.hex);
//             let (chunk, quad) =
//                 chunkquad(hex_xy.x, hex_xy.y, xmin, ymin, chunk_size, nxchunks);

//             ChunkedHexBin {
//                 hexbin: hexbin.clone(),
//                 chunk,
//                 quad,
//             }
//         })
//         .collect();
// }


struct HexCellMap {
    index: HashMap<Hex, u32>,
    ncells: usize,
}


impl HexCellMap {
    fn new(ncells: usize) -> Self {
        Self {
            index: HashMap::new(),
            ncells,
        }
    }

    fn insert(&mut self, hex: Hex, cell: u32) {
        self.index.insert(hex, cell);
    }

    fn get(&self, hex: Hex) -> u32 {
        match self.index.get(&hex) {
            Some(cell) => *cell,
            None => self.ncells as u32,
        }
    }

    fn set(&mut self, hex: Hex, cell: u32) {
        self.index.insert(hex, cell);
    }

    fn count(&self, cell: u32) -> usize {
        return self.index.values().filter(|&&c| c == cell).count();
    }
}


fn bin_transcripts(transcripts: &Vec<Transcript>, full_area: f32, avgpop: f32) -> (HexLayout, Vec<HexBin>) {
    let density = transcripts.len() as f32 / full_area;
    let target_area = avgpop / density;
    let hex_size = (target_area * 2.0 / (3.0 * (3.0 as f32).sqrt())).sqrt();

    let layout = HexLayout {
        orientation: HexOrientation::Flat,
        origin: Vec2::ZERO,
        hex_size: Vec2::new(hex_size, hex_size),
    };

    // Bin transcripts into HexBins
    let mut hex_index = HashMap::new();

    for (i, transcript) in transcripts.iter().enumerate() {
        let hex = layout.world_pos_to_hex(Vec2::new(transcript.x, transcript.y));

        hex_index.entry(hex)
            .or_insert_with(|| HexBin::new(hex))
            .transcripts.push(i);
    }

    let hexbins = hex_index.values().cloned().collect::<Vec<_>>();

    return (layout, hexbins);
}

pub struct HexBinSampler {
    layout: HexLayout,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,

    transcript_genes: Vec<u32>,

    chunkquads: [Vec<ChunkQuad<Hex>>; 4],
    hexbins: Vec<HexBin>,
    hexindex: HashMap<Hex, usize>,

    // assignment of hexbins to cells
    // (Unassigned cells are either absent or set to `ncells`)
    hexcells: HexCellMap,

    proposals: Vec<HexBinProposal>,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,

    hexarea: f32,
    ncells: usize,
    quad: usize,
}


impl HexBinSampler {
    pub fn new(
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        ncells: usize,
        ngenes: usize,
        full_area: f32,
        avghexpop: f32,
        chunk_size: f32) -> Self
    {
        let (xmin, xmax, ymin, ymax) = coordinate_span(transcripts);
        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;

        let (layout, hexbins ) = bin_transcripts(transcripts, full_area, avghexpop);

        let transcript_genes = transcripts.iter().map(|t| t.gene).collect::<Vec<_>>();

        assert!(layout.hex_size.x == layout.hex_size.y);
        let hex_size = layout.hex_size.x;

        let hexarea = (hex_size*hex_size) * 3.0 * (3.0_f32).sqrt() / 2.0;

        // build index
        let mut hexindex = HashMap::new();
        for (i, hexbin) in hexbins.iter().enumerate() {
            hexindex.insert(hexbin.hex, i);
        }

        // initialize chunkquads
        let mut chunkquads = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (quad, chunks) in chunkquads.iter_mut().enumerate() {
            for chunk in 0..nchunks {
                chunks.push(ChunkQuad::<Hex> {
                    chunk: chunk as u32,
                    quad: quad as u32,
                    mismatch_edges: SampleSet::new(),
                });
            }
        }

        // initial assignments
        let mut hex_assignments = HashMap::new();
        let mut hexcells = HexCellMap::new(ncells);
        for hexbin in &hexbins {
            hex_assignments.clear();

            // vote on hex assignment
            for &t in hexbin.transcripts.iter() {
                hex_assignments
                    .entry(params.cell_assignments[t])
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }

            let winner = hex_assignments.iter()
                .max_by_key(|(_, count)| *count)
                .map(|(cell, _)| *cell).unwrap_or(ncells as u32);

            if winner != ncells as u32 {
                hexcells.insert(hexbin.hex, winner);
            }

            // homogenize the hex: assign every transcript in the hex to the winner
            for &t in hexbin.transcripts.iter() {
                params.cell_population[params.cell_assignments[t] as usize] -= 1;
                params.cell_population[winner as usize] += 1;
                params.cell_assignments[t] = winner;
            }
        }
        params.recompute_counts(ncells, transcripts);

        // recompute cell areas as the sum of hexagon areas
        params.cell_areas.fill(0.0_f32);
        for hexbin in &hexbins {
            let cell = hexcells.get(hexbin.hex);
            if cell < ncells as u32 {
                params.cell_areas[cell as usize] += hexarea;
            }
        }
        for cell_area in params.cell_areas.iter_mut() {
            *cell_area = cell_area.max(priors.min_cell_area);
        }

        let proposals = vec![HexBinProposal::new(ngenes); nchunks];
        let connectivity_checker = ThreadLocal::new();

        let mut sampler = HexBinSampler {
            layout,
            xmin,
            ymin,
            chunk_size,
            nxchunks,
            transcript_genes,
            chunkquads,
            hexbins,
            hexindex,
            hexcells,
            proposals,
            connectivity_checker,
            hexarea,
            ncells,
            quad: 0,
        };

        sampler.populate_mismatches();

        return sampler;
    }

    fn populate_mismatches(&mut self) {
        // compute initial mismatch edges
        let mut nmismatches = 0;
        for hexbin in &self.hexbins {
            let cell = self.hexcells.get(hexbin.hex);
            let (chunk, quad) = self.chunkquad(hexbin.hex);

            for neighbor in hexbin.hex.all_neighbors() {
                let neighbor_cell = self.hexcells.get(neighbor);
                if cell != neighbor_cell {
                    let (neighbor_chunk, neighbor_quad) = self.chunkquad(neighbor);

                    self.chunkquads[quad as usize][chunk as usize]
                        .mismatch_edges.insert((hexbin.hex, neighbor));

                    self.chunkquads[neighbor_quad as usize][neighbor_chunk as usize]
                        .mismatch_edges.insert((neighbor, hexbin.hex));
                }
            }
        }
    }

    fn chunkquad(&self, hex: Hex) -> (u32, u32) {
        let hex_xy = self.layout.hex_to_world_pos(hex);
        return chunkquad(hex_xy.x, hex_xy.y, self.xmin, self.ymin, self.chunk_size, self.nxchunks);
    }
}


impl Sampler<HexBinProposal> for HexBinSampler {
    fn repopulate_proposals(&mut self, params: &ModelParams) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.05;

        self.proposals
            .par_iter_mut()
            .zip(&self.chunkquads[self.quad])
            .for_each(|(proposal, chunkquad)|
            {
                if chunkquad.mismatch_edges.is_empty() {
                    proposal.ignore = true;
                    return;
                }

                let mut rng = thread_rng();
                let (i, j) = chunkquad.mismatch_edges.choose(&mut rng).unwrap();

                // TODO: Ok, this has borrow issues.
                let cell_from = self.hexcells.get(*i);
                let mut cell_to = self.hexcells.get(*j);
                assert!(cell_from != cell_to);

                let from_unassigned = cell_from == self.ncells as u32;
                let to_unassigned = cell_to == self.ncells as u32;

                if !from_unassigned && rng.gen::<f64>() < UNASSIGNED_PROPOSAL_PROB {
                    cell_to = self.ncells as u32;
                }

                let hexbin = self.hexindex.get(i).map(|i| &self.hexbins[*i]);

                let transcripts = hexbin.map(
                    |hexbin| &hexbin.transcripts);
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

                let art_from = connectivity_checker.hex_isarticulation(
                    *i,
                    |hex| self.hexcells.get(hex),
                    cell_from);

                let art_to = connectivity_checker.hex_isarticulation(
                    *i,
                    |hex| self.hexcells.get(hex),
                    cell_to);

                if art_from || art_to {
                    proposal.ignore = true;
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = chunkquad.mismatch_edges.len();

                let num_new_state_neighbors = i.all_neighbors().iter()
                    .filter(|&&j| self.hexcells.get(j) == cell_to)
                    .count();

                let num_prev_state_neighbors = i.all_neighbors().iter()
                    .filter(|&&j| self.hexcells.get(j) == cell_from)
                    .count();

                let mut proposal_prob =
                    num_new_state_neighbors as f64 / num_mismatching_edges as f64;

                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if to_unassigned {
                    let num_mismatching_neighbors = i.all_neighbors().iter()
                        .filter(|&&j| self.hexcells.get(j) != cell_from)
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
                    let new_num_mismatching_neighbors = i.all_neighbors().iter()
                        .filter(|&&j| self.hexcells.get(j) != cell_to)
                        .count();
                    reverse_proposal_prob = UNASSIGNED_PROPOSAL_PROB
                        * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64)
                        + (1.0 - UNASSIGNED_PROPOSAL_PROB) * reverse_proposal_prob;
                }

                proposal.hex = *i;
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
                proposal.old_cell_area_delta = -self.hexarea;
                proposal.new_cell_area_delta = self.hexarea;

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

    fn proposals<'a, 'b>(&'a self) -> &'b [HexBinProposal] where 'a: 'b {
        return &self.proposals;
    }

    fn proposals_mut<'a, 'b>(&'a mut self) -> &'b mut [HexBinProposal] where 'a: 'b {
        return &mut self.proposals;
    }

    fn update_sampler_state(&mut self, _: &ModelParams) {
        for proposal in &self.proposals {
            if proposal.accept {
                let (chunk, quad) = self.chunkquad(proposal.hex);

            // update hex cell assignments
            self.hexcells.set(proposal.hex, proposal.new_cell);

            // update mismatch edges
            for neighbor in proposal.hex.all_neighbors() {
                let (neighbor_chunk, neighbor_quad) = self.chunkquad(neighbor);
                let neighbor_cell = self.hexcells.get(neighbor);
                if proposal.new_cell == neighbor_cell {
                    self.chunkquads[quad as usize][chunk as usize]
                        .mismatch_edges.remove((proposal.hex, neighbor));
                    self.chunkquads[neighbor_quad as usize][neighbor_chunk as usize]
                        .mismatch_edges.remove((neighbor, proposal.hex));
                } else {
                    self.chunkquads[quad as usize][chunk as usize]
                        .mismatch_edges.insert((proposal.hex, neighbor));
                    self.chunkquads[neighbor_quad as usize][neighbor_chunk as usize]
                        .mismatch_edges.insert((neighbor, proposal.hex));
                }
            }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct HexBinProposal {
    hex: Hex,
    transcripts: Vec<usize>,

    // gene count for this hexagon
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

impl HexBinProposal {
    fn new(ngenes: usize) -> HexBinProposal {
        return HexBinProposal {
            hex: Hex::new(0, 0),
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

impl Proposal for HexBinProposal {
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

