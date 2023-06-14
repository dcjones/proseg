
pub mod transcripts;

use transcripts::{Transcript, NucleiCentroid};
use sprs::CsMat;
use kiddo::float::kdtree::KdTree;
use kiddo::float::distance::squared_euclidean;
use std::collections::HashSet;
use bitvec::prelude::*;
use rand::seq::SliceRandom;
use rand::RngCore;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;


pub struct Segmentation<'a> {
    transcripts: &'a Vec<Transcript>,
    nuclei_centroids: &'a Vec<NucleiCentroid>,
    adjacency: &'a CsMat<f32>,
    cell_assignments: Vec<u32>,
}

impl<'a> Segmentation<'a> {
    pub fn new(transcripts: &'a Vec<Transcript>, nuclei_centroids: &'a Vec<NucleiCentroid>, adjacency: &'a CsMat<f32>) -> Segmentation<'a> {
        let cell_assignments = init_cell_assignments(transcripts, nuclei_centroids, 15);
        return Segmentation {
            transcripts,
            nuclei_centroids,
            adjacency,
            cell_assignments,
        }
    }

    pub fn apply_local_updates(&mut self, sampler: &Sampler) {
        println!("Updating with {} proposals", sampler.proposals.len());
        for proposal in sampler.proposals.iter() {
            if proposal.accept {
                proposal.apply(self);
            }
        }
    }
}


pub struct Sampler {
    mismatch_edges: HashSet<(usize, usize)>,
    shuffled_mismatch_edges: Vec<(u32, usize, usize)>,
    proposal_cell_blacklist: BitVec,
    proposals: Vec<Proposal>,
    sample_num: usize,
}

impl Sampler {
    pub fn new(seg: &Segmentation) -> Sampler {
        let ncells = seg.nuclei_centroids.len();

        let mut mismatch_edges = HashSet::new();
        for (_, (i, j)) in seg.adjacency.iter() {
            if seg.cell_assignments[i] != seg.cell_assignments[j] {
                mismatch_edges.insert((i, j));
            }
        }
        println!("Made initial cell assignments with {} mismatch edges", mismatch_edges.len());

        let proposal_cell_blacklist = bitvec![0; ncells+1];

        return Sampler {
            mismatch_edges,
            shuffled_mismatch_edges: Vec::new(),
            proposal_cell_blacklist,
            proposals: Vec::new(),
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
        let ncells = seg.nuclei_centroids.len();

        self.proposal_cell_blacklist.fill(false);
        let mut rng = rand::thread_rng();

        // Strategy here is to approximately shuffle the mismatch edges by
        // assigning a random priority to each edge, then sorting on that.
        self.shuffled_mismatch_edges.clear();
        self.shuffled_mismatch_edges.extend(self.mismatch_edges.iter().map(|(i, j)| {
            (rng.next_u32(), *i, *j)
        }));
        self.shuffled_mismatch_edges.par_sort_unstable_by(|(a, _, _), (b, _, _)| {
            a.cmp(b)
        });

        self.proposals.clear();
        let mut nblacklisted = 0;

        for (_, i, j) in self.shuffled_mismatch_edges.iter() {
            let i_cell = seg.cell_assignments[*i];
            let i_cell_blacklisted = self.proposal_cell_blacklist[i_cell as usize];
            if i_cell_blacklisted {
                continue;
            }

            let j_cell = seg.cell_assignments[*j];
            let j_cell_blacklisted = self.proposal_cell_blacklist[j_cell as usize];
            if j_cell_blacklisted {
                continue;
            }

            // never blacklist the background "cell"
            if i_cell < ncells as u32 {
                self.proposal_cell_blacklist.set(i_cell as usize, true);
                nblacklisted += 1;
            }
            if j_cell < ncells as u32 {
                self.proposal_cell_blacklist.set(j_cell as usize, true);
                nblacklisted += 1;
            }

            self.proposals.push(Proposal{i: *i, j: *j, accept: false});

            // Quit early if we can. This is a sufficient but not necessary
            // condition. There may be fewer proposals possible than there are
            // cells in which case we end up going through every mismatch edge.
            if nblacklisted == ncells {
                break;
            }
        }

    }

    pub fn sample_global_params(&mut self, seg: &Segmentation) {
        // TODO:
    }

}


struct Proposal {
    i: usize,
    j: usize,

    accept: bool

    // We need to keep track of some other stuff, preventing this from being
    // some pre-allocated structure.
}


impl Proposal {
    fn evaluate(&mut self, seg: &Segmentation) {
        // TODO: actually evaluate
        self.accept = true;
    }

    fn apply(&self, seg: &mut Segmentation) {
        // TODO:
        //   1. swap states
        //   2. update mismatch edges

        // This second part is going to be a little expensive, we have to look up
        // all of i and j's neighbors and see if they are now mismatched.
    }
}


fn hash_value<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    return s.finish();
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

