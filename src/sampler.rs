
pub mod transcripts;

use transcripts::{Transcript, NucleiCentroid};
use sprs::CsMat;
use kiddo::float::kdtree::KdTree;
use kiddo::float::distance::squared_euclidean;
use std::collections::HashSet;


pub struct Sampler<'a> {
    transcripts: &'a Vec<Transcript>,
    nuclei_centroids: &'a Vec<NucleiCentroid>,
    adjacency: &'a CsMat<f32>,
    cell_assignments: Vec<u32>,
    mismatch_edges: HashSet<(usize, usize)>,
}

impl<'a> Sampler<'a> {
    pub fn new(transcripts: &'a Vec<Transcript>, nuclei_centroids: &'a Vec<NucleiCentroid>, adjacency: &'a CsMat<f32>) -> Sampler<'a> {

        let cell_assignments = init_cell_assignments(transcripts, nuclei_centroids, 15);

        let mut mismatch_edges = HashSet::new();
        for (_, (i, j)) in adjacency.iter() {
            if cell_assignments[i] != cell_assignments[j] {
                mismatch_edges.insert((i, j));
            }
        }
        println!("Made initial cell assignments with {} mismatch edges", mismatch_edges.len());

        return Sampler {
            transcripts,
            nuclei_centroids,
            adjacency,
            cell_assignments,
            mismatch_edges
        }
    }

    pub fn iterate(&mut self) {
        // TODO:
        //  1. Select as many mismatch edges as we can. Process mismatch edges
        //  in random order, add proposals when there is no conflict, and pass
        //  over when there is.
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


