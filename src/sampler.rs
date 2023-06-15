
pub mod transcripts;

use transcripts::{Transcript, NucleiCentroid, NeighborhoodGraph, coordinate_span};
use kiddo::float::kdtree::KdTree;
use kiddo::float::distance::squared_euclidean;
use std::collections::HashSet;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use petgraph::visit::IntoNeighbors;


struct ChunkedTranscript {
    transcript: Transcript,
    chunk: u32,
    quad: u32,
}


// Compute chunk and quadrant for a single a single (x,y) point.
fn chunkquad(x: f32, y: f32, xmin: f32, ymin: f32, chunk_size: f32, nxchunks: usize) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size/2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size/2.0)).floor() as u32;

    let chunk = (xchunkquad/4) + (ychunkquad/4)*(nxchunks as u32);
    let quad = (xchunkquad%2) + (ychunkquad%2)*2;

    return (chunk, quad);
}


// Figure out every transcript's chunk and quadrant.
fn chunk_transcripts(
    transcripts: &Vec<Transcript>,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize) -> Vec<ChunkedTranscript>
{
    return transcripts.iter().map(|transcript| {
        let (chunk, quad) = chunkquad(
            transcript.x, transcript.y, xmin, ymin, chunk_size, nxchunks);

        ChunkedTranscript {
            transcript: transcript.clone(),
            chunk: chunk,
            quad: quad,
        }
    }).collect();
}


pub struct Segmentation<'a> {
    transcripts: &'a Vec<Transcript>,
    nuclei_centroids: &'a Vec<NucleiCentroid>,
    adjacency: &'a NeighborhoodGraph,
    cell_assignments: Vec<u32>,
}

impl<'a> Segmentation<'a> {
    pub fn new(transcripts: &'a Vec<Transcript>, nuclei_centroids: &'a Vec<NucleiCentroid>, adjacency: &'a NeighborhoodGraph) -> Segmentation<'a> {
        let cell_assignments = init_cell_assignments(transcripts, nuclei_centroids, 15);
        return Segmentation {
            transcripts,
            nuclei_centroids,
            adjacency,
            cell_assignments,
        }
    }

    pub fn apply_local_updates(&mut self, sampler: &mut Sampler) {
        println!("Updating with {} proposals", sampler.proposals.len());

        // TODO: check if we are doing multiple updates on the same cell and warn
        // about it.

        // Update cell assignments
        for proposal in sampler.proposals.iter().filter(|p| p.accept) {
            self.cell_assignments[proposal.i] = proposal.state;
        }

        // Update mismatch edges
        for quad in 0..4 {
            sampler.chunkquads[quad].par_iter_mut().for_each(|chunkquad| {
                for proposal in sampler.proposals.iter().filter(|p| p.accept) {
                    let i = proposal.i;
                    for j in self.adjacency.neighbors(i) {
                        if self.cell_assignments[i] != self.cell_assignments[j] {
                            if chunkquad.contains(&sampler.transcripts[j]) {
                                chunkquad.mismatch_edges.insert((j, i));
                            }
                            if chunkquad.contains(&sampler.transcripts[i]) {
                                chunkquad.mismatch_edges.insert((i, j));
                            }
                        } else {
                            if chunkquad.contains(&sampler.transcripts[j]) {
                                chunkquad.mismatch_edges.remove(&(j, i));
                            }
                            if chunkquad.contains(&sampler.transcripts[i]) {
                                chunkquad.mismatch_edges.remove(&(i, j));
                            }
                        }
                    }
                }
            });
        }

    }
}

#[derive(Clone)]
struct ChunkQuad {
    chunk: u32,
    quad: u32,
    mismatch_edges: HashSet<(usize, usize)>,
    shuffled_mismatch_edges: Vec<(usize, usize)>,
}

impl ChunkQuad {
    fn contains(&self, transcript: &ChunkedTranscript) -> bool {
        return self.chunk == transcript.chunk && self.quad == transcript.quad;
    }
}


pub struct Sampler {
    chunkquads: Vec<Vec<ChunkQuad>>,
    transcripts: Vec<ChunkedTranscript>,
    proposals: Vec<Proposal>,
    quad: usize,
    sample_num: usize,
}

impl Sampler {
    pub fn new(seg: &Segmentation, chunk_size: f32) -> Sampler {
        let (xmin, xmax, ymin, ymax) = coordinate_span(seg.transcripts, seg.nuclei_centroids);

        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;
        let chunked_transcripts = chunk_transcripts(seg.transcripts, xmin, ymin, chunk_size, nxchunks);

        let ncells = seg.nuclei_centroids.len();

        let mut chunkquads = Vec::with_capacity(4);
        for quad in 0..4 {
            let mut chunks= Vec::with_capacity(nchunks);
            for chunk in 0..nchunks {
                chunks.push(ChunkQuad {
                    chunk: chunk as u32,
                    quad: quad as u32,
                    mismatch_edges: HashSet::new(),
                    shuffled_mismatch_edges: Vec::new(),
                });
            }
            chunkquads.push(chunks);
        }

        // need to be able to look up a quad chunk given its indexes
        let mut nmismatchedges = 0;
        for i in 0..seg.adjacency.node_count() {
            for j in seg.adjacency.neighbors(i) {
                if seg.cell_assignments[i] != seg.cell_assignments[j] {
                    let ti = &chunked_transcripts[i];
                    chunkquads[ti.quad as usize][ti.chunk as usize].mismatch_edges.insert((i, j));
                    nmismatchedges += 1;
                }
            }
        }
        println!("Made initial cell assignments with {} mismatch edges", nmismatchedges);

        let proposals = vec![Proposal {
            i: 0,
            state: ncells as u32,
            accept: false,
            ignore: true,
        }; nchunks];

        return Sampler {
            chunkquads,
            transcripts: chunked_transcripts,
            proposals: proposals,
            quad: 0,
            sample_num: 0,
        }
    }

    pub fn sample_local_updates(&mut self, seg: &Segmentation) {
        self.repoulate_proposals(seg);

        self.proposals.par_iter_mut().for_each(|proposal| {
            proposal.evaluate(seg);
        });

        self.sample_num += 1;
    }

    fn repoulate_proposals(&mut self, seg: &Segmentation) {
        self.proposals.par_iter_mut()
            .zip(&mut self.chunkquads[self.quad])
            .for_each(|(proposal, chunkquad)| {
            let mut rng = rand::thread_rng();

            if chunkquad.mismatch_edges.is_empty() {
                proposal.ignore = true;
                return;
            }

            chunkquad.shuffled_mismatch_edges.clear();
            chunkquad.shuffled_mismatch_edges.extend(chunkquad.mismatch_edges.iter().cloned());
            let (i, j) = chunkquad.shuffled_mismatch_edges.choose(&mut rng).unwrap();
            proposal.i = *i;
            proposal.state = seg.cell_assignments[*j];
            proposal.accept = false;
            proposal.ignore = false;
        });

        self.quad = (self.quad + 1) % 4;
    }

    pub fn sample_global_params(&mut self, seg: &Segmentation) {
        // TODO:
    }

}


#[derive(Clone)]
struct Proposal {
    i: usize,
    state: u32,

    ignore: bool,
    accept: bool

    // We need to keep track of some other stuff, preventing this from being
    // some pre-allocated structure.
}


impl Proposal {
    fn evaluate(&mut self, seg: &Segmentation) {
        if self.ignore {
            self.accept = false;
            return;
        }
        // TODO: actually evaluate
        self.accept = true;
    }
}


fn init_cell_assignments(transcripts: &Vec<Transcript>, nuclei_centroids: &Vec<NucleiCentroid>, k: usize) -> Vec<u32> {
    let mut kdtree: KdTree<f32, usize, 2, 32, u32> = KdTree::with_capacity(transcripts.len());

    for (i, transcript) in transcripts.iter().enumerate() {
        kdtree.add(&[transcript.x, transcript.y], i);
    }

    let ncells = nuclei_centroids.len();
    let ntranscripts = transcripts.len();
    let mut cell_assignments = vec![ncells as u32; ntranscripts];

    for (i, centroid) in nuclei_centroids.iter().enumerate() {
        for neighbor in kdtree.nearest_n(&[centroid.x, centroid.y], k, &squared_euclidean) {
            cell_assignments[neighbor.item] = i as u32;
        }
    }

    return cell_assignments;
}

