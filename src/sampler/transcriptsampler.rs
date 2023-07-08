
use super::{Proposal, Sampler, ModelPriors, ModelParams, ChunkQuad, chunkquad};
use super::sampleset::SampleSet;
use super::connectivity::ConnectivityChecker;
use super::transcripts::{Transcript, NeighborhoodGraph, coordinate_span};

use thread_local::ThreadLocal;
use std::iter::{Map, Once, once};
use std::cell::{RefCell, RefMut};
use rayon::prelude::*;
use petgraph::visit::IntoNeighbors;

#[derive(Clone, Debug)]
struct TranscriptProposal {
    i: usize, // transcript

    gene: u32,
    ngenes: usize,

    old_cell: u32,
    new_cell: u32,

    // metroplis-hastings proposal weight weight
    log_weight: f32,

    ignore: bool,
    accept: bool,

    // updated cell areas and logprobs if the proposal is accepted
    old_cell_area: f32,
    new_cell_area: f32,
}


impl TranscriptProposal {
    fn new() -> Self {
        TranscriptProposal {
            i: 0,
            gene: 0,
            ngenes: 0,
            old_cell: 0,
            new_cell: 0,
            log_weight: 0.0,
            ignore: true,
            accept: false,
            old_cell_area: 0.0,
            new_cell_area: 0.0,
        }
    }
}

impl Proposal<Once<usize>, Once<(u32, u32)>> for TranscriptProposal {
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

    fn set_old_cell_area(&mut self, area: f32) {
        self.old_cell_area = area;
    }

    fn set_new_cell_area(&mut self, area: f32) {
        self.new_cell_area = area;
    }

    fn old_cell_area(&self) -> f32 {
        return self.old_cell_area;
    }

    fn new_cell_area(&self) -> f32 {
        return self.new_cell_area;
    }

    fn log_weight(&self) -> f32 {
        return self.log_weight;
    }

    fn transcripts(&self) -> Once<usize> {
        return once(self.i);
    }

    fn gene_count(&self) -> Once<(u32, u32)> {
        return once((self.gene, 1));
    }
}



struct ChunkedTranscript {
    transcript: Transcript,
    chunk: u32,
    quad: u32,
}


impl ChunkQuad<usize> {
    fn contains(&self, transcript: &ChunkedTranscript) -> bool {
        return self.chunk == transcript.chunk && self.quad == transcript.quad;
    }
}


// Figure out every transcript's chunk and quadrant.
fn chunk_transcripts(
    transcripts: &Vec<Transcript>,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,
) -> Vec<ChunkedTranscript> {
    return transcripts
        .iter()
        .map(|transcript| {
            let (chunk, quad) =
                chunkquad(transcript.x, transcript.y, xmin, ymin, chunk_size, nxchunks);

            ChunkedTranscript {
                transcript: transcript.clone(),
                chunk,
                quad,
            }
        })
        .collect();
}


pub struct TranscriptSampler {
    chunkquads: Vec<Vec<ChunkQuad<usize>>>,
    transcripts: Vec<ChunkedTranscript>,
    proposals: Vec<TranscriptProposal>,
    adjacency: NeighborhoodGraph,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,

    quad: usize,
    sample_num: usize,

    ncells: usize,
    ngenes: usize,
    background_cell: u32,
}

impl TranscriptSampler {
    pub fn new(
        adjacency: NeighborhoodGraph,
        priors: &ModelPriors,
        params: &ModelParams,
        transcripts: &Vec<Transcript>,
        transcript_areas: &Vec<f32>,
        ngenes: usize,
        ncells: usize,
        ncomponents: usize,
        full_area: f32,
        chunk_size: f32,
    ) -> TranscriptSampler {
        let (xmin, xmax, ymin, ymax) = coordinate_span(transcripts);

        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;
        let chunked_transcripts =
            chunk_transcripts(transcripts, xmin, ymin, chunk_size, nxchunks);

        let mut chunkquads = Vec::with_capacity(4);
        for quad in 0..4 {
            let mut chunks = Vec::with_capacity(nchunks);
            for chunk in 0..nchunks {
                chunks.push(ChunkQuad::<usize> {
                    chunk: chunk as u32,
                    quad: quad as u32,
                    mismatch_edges: SampleSet::new(),
                });
            }
            chunkquads.push(chunks);
        }

        // need to be able to look up a quad chunk given its indexes
        let mut nmismatchedges = 0;
        for i in 0..adjacency.node_count() {
            for j in adjacency.neighbors(i) {
                if params.cell_assignments[i] != params.cell_assignments[j] {
                    let ti = &chunked_transcripts[i];
                    chunkquads[ti.quad as usize][ti.chunk as usize]
                        .mismatch_edges
                        .insert((i, j));
                    nmismatchedges += 1;
                }
            }
        }
        println!(
            "Made initial cell assignments with {} mismatch edges",
            nmismatchedges
        );

        let proposals = vec![Proposal::new(); nchunks];

        return TranscriptSampler {
            chunkquads,
            transcripts: chunked_transcripts,
            adjacency,
            connectivity_checker: ThreadLocal::new(),
            proposals,
            quad: 0,
            sample_num: 0,
            ncells,
            ngenes,
            background_cell: ncells as u32,
        };
    }

    fn repoulate_proposals(&mut self, params: &ModelParams) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.05;
        let ncells = params.ncells();
        self.proposals
            .par_iter_mut()
            .zip(&mut self.chunkquads[self.quad])
            .for_each(|(proposal, chunkquad)| {
                let mut rng = rand::thread_rng();

                // So we have a lack of mismatch_edges in a lot of places...?
                if chunkquad.mismatch_edges.is_empty() {
                    proposal.ignore = true;
                    return;
                }

                let (i, j) = chunkquad.mismatch_edges.choose(&mut rng).unwrap();
                assert!(params.cell_assignments[*i] != params.cell_assignments[*j]);

                let cell_from = params.cell_assignments[*i];
                let mut cell_to = params.cell_assignments[*j];
                let isunassigned = cell_from == ncells as u32;

                // TODO: For correct balance, should I do UNASSIGNED_PROPOSAL_PROB
                // chance of background proposal regardless of isunassigned?
                if !isunassigned && rng.gen::<f64>() < UNASSIGNED_PROPOSAL_PROB {
                    cell_to = ncells as u32;
                }

                // Don't propose removing the last transcript from a cell. (This is
                // breaks the markov chain balance, since there's no path back to the previous state.)
                if !isunassigned && params.cell_population[params.cell_assignments[*i] as usize] == 1 {
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

                let art_from = connectivity_checker.isarticulation(
                    &self.adjacency,
                    &params.cell_assignments,
                    *i,
                    cell_from,
                );

                let art_to = connectivity_checker.isarticulation(
                    &self.adjacency,
                    &params.cell_assignments,
                    *i,
                    cell_to,
                );

                if art_from || art_to {
                    proposal.ignore = true;
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = chunkquad.mismatch_edges.len();

                let num_new_state_neighbors = self.adjacency
                    .neighbors(*i)
                    .filter(|j| params.cell_assignments[*j] == cell_to)
                    .count();

                let num_prev_state_neighbors = self.adjacency
                    .neighbors(*i)
                    .filter(|j| params.cell_assignments[*j] == cell_from)
                    .count();

                let mut proposal_prob =
                    num_new_state_neighbors as f64 / num_mismatching_edges as f64;
                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if cell_to == ncells as u32 {
                    let num_mismatching_neighbors = self.adjacency
                        .neighbors(*i)
                        .filter(|j| params.cell_assignments[*j] != cell_from)
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
                if params.cell_assignments[*i] == ncells as u32 {
                    let new_num_mismatching_neighbors = self.adjacency
                        .neighbors(*i)
                        .filter(|j| params.cell_assignments[*j] != cell_to)
                        .count();
                    reverse_proposal_prob = UNASSIGNED_PROPOSAL_PROB
                        * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64)
                        + (1.0 - UNASSIGNED_PROPOSAL_PROB) * reverse_proposal_prob;
                }

                proposal.i = *i;
                proposal.gene = self.transcripts[*i].transcript.gene;
                proposal.ngenes = self.ngenes;
                proposal.old_cell = cell_from;
                proposal.new_cell = cell_to;
                proposal.log_weight = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;
                proposal.accept = false;
                proposal.ignore = false;

                // TODO: It seems like we have some unpoppable bubbles.
                // if !proposal.log_weight.is_finite() {
                //     // let num_neighbors = seg.adjacency.neighbors(*i).count();
                //     let neighbor_states: Vec<u32> = seg.adjacency.neighbors(*i).map(|j| seg.cell_assignments[j]).collect();
                //     dbg!(proposal.log_weight, num_prev_state_neighbors,
                //         &neighbor_states,
                //         cell_from, cell_to, reverse_proposal_prob, proposal_prob.ln());
                // }
            });

        self.quad = (self.quad + 1) % 4;
    }
}


impl Sampler<TranscriptProposal, Once<usize>, Once<(u32,u32)>> for TranscriptSampler {

    fn generate_proposals<'a> (&'a mut self, params: &ModelParams) -> &'a [TranscriptProposal] {
        self.repoulate_proposals(params);
        return &self.proposals;
    }

    fn update_sampler_state(&mut self, params: &ModelParams, proposals: &[TranscriptProposal]) {
        // Update mismatch edges
        for quad in 0..4 {
            self.chunkquads[quad]
                .par_iter_mut()
                .for_each(|chunkquad| {
                    for proposal in proposals.iter().filter(|p| p.accept) {
                        let i = proposal.i;
                        for j in self.adjacency.neighbors(i) {
                            if params.cell_assignments[i] != params.cell_assignments[j] {
                                if chunkquad.contains(&self.transcripts[j]) {
                                    chunkquad.mismatch_edges.insert((j, i));
                                }
                                if chunkquad.contains(&self.transcripts[i]) {
                                    chunkquad.mismatch_edges.insert((i, j));
                                }
                            } else {
                                if chunkquad.contains(&self.transcripts[j]) {
                                    chunkquad.mismatch_edges.remove((j, i));
                                }
                                if chunkquad.contains(&self.transcripts[i]) {
                                    chunkquad.mismatch_edges.remove((i, j));
                                }
                            }
                        }
                    }
                });
        }
    }

}