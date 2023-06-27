pub mod transcripts;
mod hull;
mod distributions;
mod sampleset;
mod connectivity;

use kiddo::float::distance::squared_euclidean;
use kiddo::float::kdtree::KdTree;
use petgraph::visit::{IntoNeighbors, EdgeRef};
use rand::{Rng, thread_rng};
use rand::distributions::Distribution;
use rayon::prelude::*;
use std::collections::HashSet;
use transcripts::{coordinate_span, NeighborhoodGraph, NucleiCentroid, Transcript};
use hull::convex_hull_area;
use thread_local::ThreadLocal;
use std::cell::{RefCell, RefMut};
use distributions::{lfact, odds_to_prob, prob_to_odds, rand_pois, lognormal_logpdf, negbin_logpmf, negbin_logpmf_fast};
use statrs::distribution::{Beta, Dirichlet, InverseGamma, Gamma, Normal};
use itertools::izip;
use sampleset::SampleSet;
use ndarray::{Array1, Array2, AsArray, Zip};
use libm::{log1pf, lgammaf};
use std::fs::File;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::io::Write;
use connectivity::ConnectivityChecker;

// use std::time::Instant;


#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub min_cell_size: f32,
    pub background_logprob: f32,
    pub foreground_logprob: f32,

    // params for normal prior
    pub μ_μ_a: f32,
    pub σ_μ_a: f32,

    // params for inverse-gamma prior
    pub α_σ_a: f32,
    pub β_σ_a: f32,

    pub α_w: f32,
    pub β_w: f32,

    // gamma rate prior
    pub e_r: f32,
    pub f_r: f32,
}


pub struct ModelParams {
    π: Vec<f32>, // mixing proportions over components

    μ_a: Vec<f32>, // area dist mean param by component
    σ_a: Vec<f32>, // area dist std param by component


    // TODO: If we follow the paper, we have no coefficients we are regressing
    // on. We are just modeling logistic(p) ~ LogNormal(). Let's make sure
    // we understand how to do updates.
    //
    // Beta is conjugate prior with p, so we can just use that with α, β = 1.0
    //
    // The real issue is how do we sample `r`

    // [ngenes] NB r parameters.
    r: Array1<f32>,

    lgamma_r: Array1<f32>,

    // [ngenes, ncomponents] NB p parameters.
    w: Array2<f32>,
}


impl ModelParams {
    // initialize model parameters, with random cell assignments
    // and other parameterz unninitialized.
    fn new(priors: &ModelPriors, ncomponents: usize, ngenes: usize) -> Self {
        let r = Array1::<f32>::from_elem(ngenes, 0.1_f32);
        let lgamma_r = Array1::<f32>::from_iter(r.iter().map(|&x| lgammaf(x) ));
        return ModelParams {
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_a: vec![priors.μ_μ_a; ncomponents],
            σ_a: vec![priors.σ_μ_a; ncomponents],
            r: r,
            lgamma_r: lgamma_r,
            w: Array2::<f32>::from_elem((ngenes, ncomponents), 0.1),
        };
    }

    fn ncomponents(&self) -> usize {
        return self.π.len();
    }

    fn cell_logprob<'a, A: AsArray<'a, u32>>(&self, z: usize, cell_area: f32, counts: A) -> f32
    {
        let counts = counts.into();
        let count_logprob =
            Zip::from(&self.r)
            .and(&self.lgamma_r)
            .and(self.w.column(z))
            .and(counts)
            .fold(0_f32, |acc, r, lgamma_r, w, c| {
                acc + negbin_logpmf(*r, *lgamma_r, odds_to_prob(*w * cell_area), *c)
            });

        let area_logprob = lognormal_logpdf(
            self.μ_a[z],
            self.σ_a[z],
            cell_area);

        return count_logprob + area_logprob;
        // return count_logprob;
    }

    // with pre-computed log(factorial(counts))
    fn cell_logprob_fast<'a, A: AsArray<'a, u32>, B: AsArray<'a, f32>>(
        &self, z: usize, cell_area: f32, counts: A, counts_ln_factorial: B) -> f32
    {
        let counts = counts.into();
        let counts_ln_factorial = counts_ln_factorial.into();
        // TODO: This is super slow, and it doesn't seem to have that much to do
        // with the actual function being computed here. It seems just really expensive
        // to do this iteration. Why would that be?

        let count_logprob = Zip::from(&self.r)
            .and(&self.lgamma_r)
            .and(self.w.column(z))
            .and(counts)
            .and(counts_ln_factorial)
            .fold(0_f32, |acc, r, lgamma_r, w, c, clf| {
                acc + negbin_logpmf_fast(*r, *lgamma_r, odds_to_prob(*w * cell_area), *c, *clf)
            });

        let area_logprob = lognormal_logpdf(
            self.μ_a[z],
            self.σ_a[z],
            cell_area);

        return count_logprob + area_logprob;
        // return count_logprob;
    }
}


struct ChunkedTranscript {
    transcript: Transcript,
    chunk: u32,
    quad: u32,
}

// Compute chunk and quadrant for a single a single (x,y) point.
fn chunkquad(x: f32, y: f32, xmin: f32, ymin: f32, chunk_size: f32, nxchunks: usize) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size / 2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size / 2.0)).floor() as u32;

    let chunk = (xchunkquad / 4) + (ychunkquad / 4) * (nxchunks as u32);
    let quad = (xchunkquad % 2) + (ychunkquad % 2) * 2;

    return (chunk, quad);
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
                chunk: chunk,
                quad: quad,
            }
        })
        .collect();
}

// Holds the segmentation state.
// This is kept outside of Sample mainly so we can borrow this in multiple threads
// while modifying the sampler when generating proposals.
pub struct Segmentation<'a> {
    transcripts: &'a Vec<Transcript>,
    pub nuclei_centroids: &'a Vec<NucleiCentroid>,
    adjacency: &'a NeighborhoodGraph,
    pub cell_assignments: Vec<u32>,
    cell_population: Vec<usize>,
    pub cell_logprobs: Vec<f32>,
}


impl<'a> Segmentation<'a> {
    pub fn new(
        transcripts: &'a Vec<Transcript>,
        nuclei_centroids: &'a Vec<NucleiCentroid>,
        adjacency: &'a NeighborhoodGraph,
    ) -> Segmentation<'a> {

        // initialize cell assignments so roughly half of the transcripts are assigned.
        const INIT_NEIGHBORHOOD_PROPORTION: f64 = 0.25;
        let k = (INIT_NEIGHBORHOOD_PROPORTION * (transcripts.len() as f64) / (nuclei_centroids.len() as f64)).ceil() as usize;

        let (cell_assignments, cell_population) =
            init_cell_assignments(transcripts, nuclei_centroids, k);

        let cell_logprobs = vec![0.0; nuclei_centroids.len()];

        return Segmentation {
            transcripts,
            nuclei_centroids,
            adjacency,
            cell_assignments,
            cell_population,
            cell_logprobs,
        };
    }

    pub fn nunassigned(&self) -> usize {
        let ncells = self.ncells() as u32;
        return self.cell_assignments.iter().filter(|&c| *c == ncells).count();
    }

    fn ncells(&self) -> usize {
        return self.nuclei_centroids.len();
    }

    pub fn apply_local_updates(&mut self, sampler: &mut Sampler) {
        // let accept_count = sampler.proposals.iter().filter(|p| p.accept).count();
        // let unignored_count = sampler.proposals.iter().filter(|p| !p.ignore).count();
        // println!("Applying {} of {} proposals", accept_count, unignored_count);

        // TODO: check if we are doing multiple updates on the same cell and warn
        // about it.

        // let mut background_to_cell = 0;
        // let mut cell_to_background = 0;
        // let mut cell_to_cell = 0;

        // Update cell assignments
        for proposal in sampler.proposals.iter().filter(|p| p.accept) {
            let prev_state = self.cell_assignments[proposal.i];
            let gene = self.transcripts[proposal.i].gene;

            self.cell_population[prev_state as usize] -= 1;
            self.cell_population[proposal.state as usize] += 1;
            self.cell_assignments[proposal.i] = proposal.state;

            if prev_state as usize != self.ncells() {
                sampler.cell_areas[prev_state as usize] = proposal.from_cell_area;
                self.cell_logprobs[prev_state as usize] = proposal.from_cell_logprob;
                sampler.counts[[gene as usize, prev_state as usize]] -= 1;
                sampler.counts_ln_factorial[[gene as usize, prev_state as usize]] =
                    lfact(sampler.counts[[gene as usize, prev_state as usize]]);
            }

            if proposal.state as usize != self.ncells() {
                sampler.cell_areas[proposal.state as usize] = proposal.to_cell_area;
                self.cell_logprobs[proposal.state as usize] = proposal.to_cell_logprob;
                sampler.counts[[gene as usize, proposal.state as usize]] += 1;
                sampler.counts_ln_factorial[[gene as usize, proposal.state as usize]] =
                    lfact(sampler.counts[[gene as usize, proposal.state as usize]]);
            }

            // if prev_state as usize == self.ncells() && proposal.state as usize != self.ncells() {
            //     background_to_cell += 1;
            // } else if prev_state as usize != self.ncells() && proposal.state as usize == self.ncells() {
            //     cell_to_background += 1;
            // } else if prev_state as usize != self.ncells() && proposal.state as usize != self.ncells() {
            //     cell_to_cell += 1;
            // }
        }

        // dbg!(background_to_cell, cell_to_background, cell_to_cell);

        // Update mismatch edges
        for quad in 0..4 {
            sampler.chunkquads[quad]
                .par_iter_mut()
                .for_each(|chunkquad| {
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
                                    chunkquad.mismatch_edges.remove((j, i));
                                }
                                if chunkquad.contains(&sampler.transcripts[i]) {
                                    chunkquad.mismatch_edges.remove((i, j));
                                }
                            }
                        }
                    }
                });
        }
    }

    pub fn write_cell_hulls(&self, filename: &str) {

        // We are not maintaining any kind of per-cell array, so I guess I have
        // no choice but to compute such a thing here.
        // TODO: We area already keeping track of this in Sampler!!
        let mut cell_transcripts: Vec<Vec<usize>> = vec![Vec::new(); self.ncells()];
        for (i, &cell) in self.cell_assignments.iter().enumerate() {
            if cell as usize != self.ncells() {
                cell_transcripts[cell as usize].push(i);
            }
        }

        let file = File::create(filename).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        writeln!(encoder, "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": [").unwrap();

        let mut vertices: Vec<(f32, f32)> = Vec::new();
        let mut hull: Vec<(f32, f32)> = Vec::new();

        let mut vertices_refcell = RefCell::new(vertices);
        let mut vertices_ref = vertices_refcell.borrow_mut();

        let mut hull_refcell = RefCell::new(hull);
        let mut hull_ref = hull_refcell.borrow_mut();

        for (i, transcripts) in cell_transcripts.iter().enumerate() {
            vertices_ref.clear();
            for j in transcripts {
                let transcript = self.transcripts[*j];
                vertices_ref.push((transcript.x, transcript.y));
            }

            let area = convex_hull_area(&mut vertices_ref, &mut hull_ref);

            writeln!(
                encoder,
                concat!(
                "    {{\n",
                "      \"type\": \"Feature\",\n",
                "      \"properties\": {{\n",
                "        \"cell\": {},\n",
                "        \"area\": {}\n",
                "      }},\n",
                "      \"geometry\": {{\n",
                "        \"type\": \"Polygon\",\n",
                "        \"coordinates\": [",
                "          ["), i, area).unwrap();
            for (i, (x, y)) in hull_ref.iter().enumerate() {
                writeln!(encoder, "            [{}, {}]", x, y).unwrap();
                if i < hull_ref.len() - 1 {
                    write!(encoder, ",").unwrap();
                }
            }
            write!(encoder, concat!(
                "          ]\n", // polygon
                "        ]\n", // coordinates
                "      }}\n", // geometry
                "    }}\n", // feature
            )).unwrap();

            if i < cell_transcripts.len() - 1 {
                write!(encoder, ",").unwrap();
            }
        }

        writeln!(encoder, "\n  ]\n}}").unwrap();

    }

}

#[derive(Clone)]
struct ChunkQuad {
    chunk: u32,
    quad: u32,
    mismatch_edges: SampleSet<(usize, usize)>,
}

impl ChunkQuad {
    fn contains(&self, transcript: &ChunkedTranscript) -> bool {
        return self.chunk == transcript.chunk && self.quad == transcript.quad;
    }
}


// Pre-allocated thread-local storage used when computing cell areas.
struct AreaCalcStorage {
    vertices: Vec<(f32, f32)>,
    hull: Vec<(f32, f32)>,
}

impl AreaCalcStorage {
    fn new() -> Self {
        return AreaCalcStorage {
            vertices: Vec::new(),
            hull: Vec::new(),
        };
    }
}


pub struct Sampler {
    priors: ModelPriors,
    chunkquads: Vec<Vec<ChunkQuad>>,
    transcripts: Vec<ChunkedTranscript>,
    params: ModelParams,
    proposals: Vec<Proposal>,
    cell_transcripts: Vec<HashSet<usize>>,
    cell_areas: Array1<f32>,
    cell_area_calc_storage: ThreadLocal<RefCell<AreaCalcStorage>>,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,
    pub z: Array1<u32>, // assignment of cells to components
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,
    counts: Array2<u32>,
    counts_ln_factorial: Array2<f32>,
    component_counts: Array2<u32>,
    component_counts_per_area: Array2<f32>,
    quad: usize,
    sample_num: usize,
}

impl Sampler {
    pub fn new(priors: ModelPriors, seg: &mut Segmentation, ncomponents: usize, ngenes: usize, chunk_size: f32) -> Sampler {
        let (xmin, xmax, ymin, ymax) = coordinate_span(seg.transcripts, seg.nuclei_centroids);

        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;
        let chunked_transcripts =
            chunk_transcripts(seg.transcripts, xmin, ymin, chunk_size, nxchunks);

        let ncells = seg.nuclei_centroids.len();

        let mut chunkquads = Vec::with_capacity(4);
        for quad in 0..4 {
            let mut chunks = Vec::with_capacity(nchunks);
            for chunk in 0..nchunks {
                chunks.push(ChunkQuad {
                    chunk: chunk as u32,
                    quad: quad as u32,
                    mismatch_edges: SampleSet::new(),
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

        let params = ModelParams::new(&priors, ncomponents, ngenes);

        let mut cell_transcripts = vec![HashSet::new(); ncells];
        for (i, cell) in seg.cell_assignments.iter().enumerate() {
            if (*cell as usize) < ncells {
                cell_transcripts[*cell as usize].insert(i);
            }
        }

        let cell_areas = Array1::<f32>::zeros(ncells);
        let proposals = vec![Proposal::new(ngenes); nchunks];

        let mut rng = rand::thread_rng();
        let z = (0..ncells).map(|_| rng.gen_range(0..ncomponents) as u32).collect::<Vec<_>>().into();

        let mut sampler = Sampler {
            priors,
            chunkquads,
            transcripts: chunked_transcripts,
            params: params,
            cell_transcripts: cell_transcripts,
            cell_areas: cell_areas,
            cell_area_calc_storage: ThreadLocal::new(),
            connectivity_checker: ThreadLocal::new(),
            z_probs: ThreadLocal::new(),
            z: z,
            counts: Array2::<u32>::zeros((ngenes, ncells)),
            counts_ln_factorial: Array2::<f32>::zeros((ngenes, ncells)),
            component_counts: Array2::<u32>::zeros((ngenes, ncomponents)),
            component_counts_per_area: Array2::<f32>::zeros((ngenes, ncomponents)),
            proposals: proposals,
            quad: 0,
            sample_num: 0,
        };

        sampler.compute_cell_areas();
        sampler.compute_counts(seg);
        sampler.sample_global_params();
        sampler.compute_cell_logprobs(seg);

        return sampler;
    }

    pub fn counts(&self) -> Array2<u32> {
        return self.counts.clone();
    }

    pub fn sample_local_updates(&mut self, seg: &Segmentation) {
        // let t0 = Instant::now();
        self.repoulate_proposals(seg);
        // let t1 = Instant::now();
        // println!("repoulate_proposals took {:?}", t1 - t0);

        self.proposals.par_iter_mut().for_each(|proposal| {
            let areacalc = self.cell_area_calc_storage.get_or(
                || RefCell::new(AreaCalcStorage::new()) ).borrow_mut();

            proposal.evaluate(
                seg, &self.priors, &self.params, &self.z, &self.counts,
                &self.counts_ln_factorial, &self.cell_transcripts, areacalc);
        });
        // let t2 = Instant::now();
        // println!("evaluate took {:?}", t2 - t1);

        self.sample_num += 1;
    }

    fn repoulate_proposals(&mut self, seg: &Segmentation) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.1;
        let ncells = seg.ncells();
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
                assert!(seg.cell_assignments[*i] != seg.cell_assignments[*j]);

                let cell_from = seg.cell_assignments[*i];
                let mut cell_to = seg.cell_assignments[*j];
                let isunassigned = cell_from == ncells as u32;


                // TODO: For correct balance, should I do UNASSIGNED_PROPOSAL_PROB
                // chance of background proposal regardless of isunassigned?
                if !isunassigned && rng.gen::<f64>() < UNASSIGNED_PROPOSAL_PROB {
                    cell_to = ncells as u32;
                }

                // Don't propose removing the last transcript from a cell. (This is
                // breaks the markov chain balance, since there's no path back to the previous state.)
                if !isunassigned && seg.cell_population[seg.cell_assignments[*i] as usize] == 1 {
                    proposal.ignore = true;
                    return;
                }

                // Local connectivity condition: don't propose changes that render increase the
                // number of connected components of either the cell_from or cell_to
                // neighbors subgraphs.
                let mut connectivity_checker = self.connectivity_checker.get_or(
                    || RefCell::new(ConnectivityChecker::new())).borrow_mut();

                let art_from = connectivity_checker.isarticulation(
                    &seg.adjacency, &seg.cell_assignments,
                    *i, seg.cell_assignments[*i]);

                let art_to = connectivity_checker.isarticulation(
                    &seg.adjacency, &seg.cell_assignments,
                    *i, seg.cell_assignments[*j]);

                if art_from || art_to {
                    proposal.ignore = true;
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = chunkquad.mismatch_edges.len();

                let num_new_state_neighbors = seg
                    .adjacency
                    .neighbors(*i)
                    .filter(|j| seg.cell_assignments[*j] == cell_to)
                    .count();

                let num_prev_state_neighbors = seg
                    .adjacency
                    .neighbors(*i)
                    .filter(|j| seg.cell_assignments[*j] == cell_from)
                    .count();

                let mut proposal_prob = num_new_state_neighbors as f64 / num_mismatching_edges as f64;
                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if cell_to == ncells as u32 {
                    let num_mismatching_neighbors = seg
                        .adjacency
                        .neighbors(*i)
                        .filter(|j| seg.cell_assignments[*j] != cell_from)
                        .count();
                    proposal_prob =
                        UNASSIGNED_PROPOSAL_PROB * (num_mismatching_neighbors as f64 / num_mismatching_edges as f64) +
                        (1.0 - UNASSIGNED_PROPOSAL_PROB) * proposal_prob;
                }

                let new_num_mismatching_edges = num_mismatching_edges
                    + 2*num_prev_state_neighbors // edges that are newly mismatching
                    - 2*num_new_state_neighbors; // edges that are newly matching

                let mut reverse_proposal_prob = num_prev_state_neighbors as f64 / new_num_mismatching_edges as f64;

                // If this is a proposal from unassigned, account for multiple ways of reversing it
                if seg.cell_assignments[*i] == ncells as u32 {
                    let new_num_mismatching_neighbors = seg
                        .adjacency
                        .neighbors(*i)
                        .filter(|j| seg.cell_assignments[*j] != cell_to)
                        .count();
                    reverse_proposal_prob =
                        UNASSIGNED_PROPOSAL_PROB * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64) +
                        (1.0 - UNASSIGNED_PROPOSAL_PROB) * reverse_proposal_prob;
                }

                // let unassign_proposal {
                //     // TODO: 
                // }

                // TODO: In particular, we need make sure "unassigned" is proposed
                // with some probability, etherwise there is no reverse path and proposals
                // to assign the last unassigned transcript are automatically rejected.

                proposal.i = *i;
                proposal.state = cell_to;
                proposal.log_weight = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;
                proposal.accept = false;
                proposal.ignore = false;
            });

        self.quad = (self.quad + 1) % 4;
    }

    pub fn sample_global_params(&mut self) {
        // TODO: we are doing some allocation in this function that can be avoided
        // by pre-allocating and storing in Sampler.

        let mut rng = thread_rng();
        let ncomponents = self.params.ncomponents();

        Zip::from(self.counts.columns())
            .and(self.counts_ln_factorial.columns())
            .and(&mut self.z)
            .and(&self.cell_areas)
            .par_for_each(|cs, clfs, z_i, cell_area| {
                let mut z_probs = self.z_probs.get_or(|| RefCell::new(vec![0_f64; ncomponents])).borrow_mut();


                // TODO: This is basically the majority of the run time. Evaluating
                // NB log pmf is super expensive. Pre-computing lgamma(r) is
                // one pretty big optimization, but we also have to compute
                // lgamma(r + k)

                z_probs.iter_mut().enumerate().for_each(|(j, zp)| {
                    *zp = (self.params.π[j] as f64) *
                        (self.params.cell_logprob_fast(j as usize, *cell_area, &cs, &clfs) as f64).exp();
                });

                // TODO: Nope, still slow as fuck!

                let z_prob_sum = z_probs.iter().sum::<f64>();

                assert!(z_prob_sum.is_finite());

                // dbg!(&z_probs);

                // cumsum in-place
                z_probs.iter_mut().fold(0.0, |mut acc, x| {
                    acc += *x / z_prob_sum;
                    *x = acc;
                    acc
                });

                // dbg!(&z_probs);
                // panic!();


                let rng = &mut thread_rng();
                let u = rng.gen::<f64>();
                *z_i = z_probs.partition_point(|x| *x < u) as u32;
        });

        // sample π
        let mut α = vec![1_f64; self.params.ncomponents()];
        for z_i in self.z.iter() {
            α[*z_i as usize] += 1.0;
        }

        self.params.π.clear();
        self.params.π.extend(Dirichlet::new(α).unwrap().sample(&mut rng).iter().map(|x| *x as f32));

        // sample μ_a
        let mut μ_μ_a = vec![self.priors.μ_μ_a / (self.priors.σ_μ_a.powi(2)) ; self.params.ncomponents()];
        // let mut component_population = vec![0; self.params.ncomponents()];
        let mut component_population = Array1::<u32>::from_elem(self.params.ncomponents(), 0_u32);

        for (z_i, cell_area) in self.z.iter().zip(&self.cell_areas) {
            μ_μ_a[*z_i as usize] += cell_area.ln() / self.params.σ_a[*z_i as usize].powi(2);
            component_population[*z_i as usize] += 1;
        }
        dbg!(&component_population);

        μ_μ_a.iter_mut().enumerate().for_each(|(i, μ)| {
            *μ /= (1_f32/self.priors.σ_μ_a.powi(2)) +
                (component_population[i] as f32 / self.params.σ_a[i].powi(2));
        });

        let σ2_μ_a: Vec<f32> = izip!(&self.params.σ_a, &component_population).map(|(σ, n)| {
            (1_f32 / self.priors.σ_μ_a.powi(2) + (*n as f32) / σ.powi(2)).recip()
        }).collect();

        self.params.μ_a.clear();
        self.params.μ_a.extend(
            izip!(&μ_μ_a, &σ2_μ_a).map(|(μ, σ2)| Normal::new(*μ as f64, σ2.sqrt() as f64).unwrap().sample(&mut rng) as f32)
        );

        // sample σ_a
        let mut δ2s = vec![0_f32; self.params.ncomponents()];
        for (z_i, cell_area) in self.z.iter().zip(&self.cell_areas) {
            δ2s[*z_i as usize] += (cell_area.ln() - μ_μ_a[*z_i as usize]).powi(2);
        }

        // sample σ_a
        self.params.σ_a.clear();
        self.params.σ_a.extend(izip!(&component_population, &δ2s).map(|(n, δ2)| {
            InverseGamma::new(
                self.priors.α_σ_a as f64 + 0.5 * *n as f64,
                self.priors.β_σ_a as f64 + 0.5 * *δ2 as f64
            ).unwrap().sample(&mut rng).sqrt() as f32
        }));

        // dbg!(&self.params.μ_a);
        // dbg!(&self.params.σ_a);
        // dbg!(&self.cell_areas);

        // Sample p

        // total component area
        let mut component_cell_area = vec![0_f32; self.params.ncomponents()];
        self.cell_areas.iter().zip(&self.z).for_each(|(area, z_i)| {
            component_cell_area[*z_i as usize] += *area;
        });

        // compute per component transcript counts
        self.component_counts.fill(0);
        Zip::from(self.component_counts.rows_mut())
            .and(self.counts.rows()).par_for_each(|mut compc, cellc| {
                for (c, component) in cellc.iter().zip(&self.z) {
                    compc[*component as usize] += *c;
                }
            });

        self.component_counts_per_area.fill(0_f32);
        Zip::from(self.component_counts_per_area.rows_mut())
            .and(self.counts.rows())
            .par_for_each(|mut compc, cellc| {
                for (c, component, area) in izip!(cellc.iter(), &self.z, &self.cell_areas) {
                    compc[*component as usize] += *c as f32 / *area;
                }
            });

        // dbg!(self.params.p.shape());
        // dbg!(self.component_counts_per_area.shape());
        // dbg!(component_population.shape());
        // dbg!(self.params.r.shape());


        // TODO: This seems to reduce the probability...

        Zip::from(self.params.w.rows_mut())
            .and(self.component_counts.rows())
            .and(&self.params.r)
            .par_for_each(|ws, cs, r| {
                let mut rng = thread_rng();
                for (w, c, a) in izip!(ws, cs, &component_cell_area) {
                    *w = prob_to_odds(Beta::new(
                        (self.priors.α_w + *c as f32) as f64,
                        (self.priors.β_w + *a * *r) as f64,
                    ).unwrap().sample(&mut rng) as f32);

                    *w = w.max(1e-6);

                    // if *w > 1000.0 {
                    //     dbg!(w, c, a, r);
                    //     panic!("w is too large");
                    // }
                }
            });

        // TODO: Ok, so the origin is really that we generate some huge values of w,
        // which in turn makes p 1.0, which breaks things.
        //
        // This seems to happen because `r` is allowed to get very small.

        // let w_min = self.params.w.fold(f32::INFINITY, |acc, w| acc.min(*w));
        // let w_max = self.params.w.fold(f32::NEG_INFINITY, |acc, w| acc.max(*w));
        // dbg!(w_min, w_max);

        //
        // Origin of the huge values of `w` seem to be from 

        // Zip::from(self.params.p.rows_mut())
        //     // .and(self.component_counts.rows())
        //     .and(self.component_counts_per_area.rows())
        //     .and(&self.params.r)
        //     .par_for_each(|ps, cs, r| {
        //         let mut rng = thread_rng();
        //         for (p, c, pop) in izip!(ps, cs, &component_population) {
        //             *p = Beta::new(
        //                 self.priors.α_p as f64 + *c as f64,
        //                 self.priors.β_p as f64 + (*pop as f64) * *r as f64
        //             ).unwrap().sample(&mut rng) as f32;
        //         }
        //     });


        // TODO: This part is absolutely fucked.

        // Sample r
        Zip::from(&mut self.params.r)
            .and(&mut self.params.lgamma_r)
            .and(self.params.w.rows())
            // .and(self.counts.rows())
            .par_for_each(|r, lgamma_r, ws| {
                let mut rng = thread_rng();

                // TODO: I must need to account for cell_area somehow here.
                // I guess in expectation I would be dividing counts by cell_area,
                // but the crt distribution necessitates it being an integer.

                // let u = cs.iter().zip(&self.cell_areas).map(|(c, cell_area)| rand_crt(&mut rng, *c, *r * cell_area)).sum::<u32>() as f32;
                // let v = ws.iter().map(|w| odds_to_prob(*w)).map(|p| log1pf(-p) ).sum::<f32>();

                // Ok, the other Zhang paper uses a difference procedure.

                // TODO: optimize this by not redundantly computing log1pf(-odds_to_prob(w * *a))

                // TODO: seems it runs for a while then we get stuck sampling the poisson here


                let u =
                    Zip::from(&self.z)
                        .and(&self.cell_areas)
                        .fold(0, |accum, z, a| {
                            let w = ws[*z as usize];
                            let λ = -*r * log1pf(-odds_to_prob(w * *a));
                            // assert!(λ >= 0.0);
                            // accum + Poisson::new(λ as f64).unwrap().sample(&mut rng)
                            accum + rand_pois(&mut rng, λ)
                        }) as f32;
                // let v = ws.iter().map(|w| odds_to_prob(*w)).map(|p| log1pf(-p) ).sum::<f32>();
                // Ohh, I think the issue is that this should be done over cells not components.
                let v =
                    Zip::from(&self.z)
                        .and(&self.cell_areas)
                        .fold(0.0, |accum, z, a| {
                            let w = ws[*z as usize];
                            accum + log1pf(-odds_to_prob(w * *a))
                        });

                *r = Gamma::new(
                    (self.priors.e_r + u) as f64,
                    (self.priors.f_r - v) as f64
                ).unwrap().sample(&mut rng) as f32;

                *lgamma_r = lgammaf(*r);

                // TODO: any better solution here?
                *r = r.min(100.0).max(1e-4);
            });

        // dbg!(&self.params.p);
        // dbg!(&self.params.r);

    }

    fn compute_cell_areas(&mut self) {
        Zip::indexed(&mut self.cell_areas).par_for_each(|i, area| {
            let mut areacalc = self.cell_area_calc_storage.get_or(
                || RefCell::new(AreaCalcStorage::new()) ).borrow_mut();
            areacalc.vertices.clear();

            for j in self.cell_transcripts[i].iter() {
                let transcript = self.transcripts[*j].transcript;
                areacalc.vertices.push((transcript.x, transcript.y));
            }

            let (mut vertices, mut hull) =
                RefMut::map_split(areacalc, |areacalc| (&mut areacalc.vertices, &mut areacalc.hull));
            *area = convex_hull_area(&mut vertices, &mut hull).max(self.priors.min_cell_size);
        });

        let mut minarea: f32 = 1e12;
        let mut maxarea: f32 = 0.0;
        for area in &self.cell_areas {
            if *area < minarea {
                minarea = *area;
            }
            if *area > maxarea {
                maxarea = *area;
            }
        }
        println!("Area span: {}, {}", minarea, maxarea);
    }

    fn compute_counts(&mut self, seg: &Segmentation) {
        self.counts.fill(0);
        let ncells = seg.ncells() as u32;
        for (transcript, j) in self.transcripts.iter().zip(&seg.cell_assignments) {
            if *j == ncells {
                continue;
            }
            let i = transcript.transcript.gene as usize;
            self.counts[[i, *j as usize]] += 1;
        }

        Zip::from(&mut self.counts_ln_factorial)
            .and(&self.counts)
            .for_each(|clf, c| {
                *clf = lfact(*c);
            });
    }

    pub fn compute_cell_logprobs(&self, seg: &mut Segmentation) {
        // Assume at this point that cell areas and transcript counts are up to date
        seg.cell_logprobs.par_iter_mut().enumerate().for_each(|(i, logprob)| {
            let cell_area = self.cell_areas[i];
            *logprob = self.params.cell_logprob(self.z[i] as usize, cell_area, self.counts.column(i));
        });

        // dbg!(&seg.cell_logprobs);
    }

    pub fn check_mismatch_edges(&self, seg: &Segmentation) {
        // check that mismatch edges are symmetric and really are mismatching
        for quad in 0..4 {
            for quadchunk in &self.chunkquads[quad] {
                for (i, j) in quadchunk.mismatch_edges.iter() {
                    assert!(seg.cell_assignments[*i] != seg.cell_assignments[*j]);

                    let quad_j = self.transcripts[*j].quad as usize;
                    let chunk_j = self.transcripts[*j].chunk as usize;
                    let quadchunk_j = &self.chunkquads[quad_j][chunk_j];
                    assert!(quadchunk_j.mismatch_edges.contains(&(*j, *i)));
                }
            }
        }

        // check that every mismatched edge is present in a set
        for i in 0..self.transcripts.len() {
            for edge in seg.adjacency.edges(i) {
                let j = edge.target();
                if seg.cell_assignments[i] != seg.cell_assignments[j] {
                    let quad_i = self.transcripts[i].quad as usize;
                    let chunk_i = self.transcripts[i].chunk as usize;
                    let quadchunk_i = &self.chunkquads[quad_i][chunk_i];
                    assert!(quadchunk_i.mismatch_edges.contains(&(i, j)));
                }
            }
        }
    }


}

#[derive(Clone, Debug)]
struct Proposal {
    i: usize,
    state: u32,

    // metroplis-hastings proposal weight weight
    log_weight: f32,

    ignore: bool,
    accept: bool,

    // updated cell areas and logprobs if the proposal is accepted
    to_cell_area: f32,
    to_cell_logprob: f32,

    from_cell_area: f32,
    from_cell_logprob: f32,

    // tempory transcript counts array
    counts: Array1<u32>,
    counts_ln_factorial: Array1<f32>,
}

impl Proposal {
    fn new(ngenes: usize) -> Self {
        Proposal {
            i: 0,
            state: 0,
            log_weight: 0.0,
            ignore: true,
            accept: false,
            to_cell_area: 0.0,
            to_cell_logprob: 0.0,
            from_cell_area: 0.0,
            from_cell_logprob: 0.0,
            counts: Array1::zeros(ngenes),
            counts_ln_factorial: Array1::zeros(ngenes),
        }
    }

    fn evaluate(
            &mut self,
            seg: &Segmentation,
            priors: &ModelPriors,
            params: &ModelParams,
            z: &Array1<u32>,
            counts: &Array2<u32>,
            counts_ln_factorial: &Array2<f32>,
            cell_transcripts: &Vec<HashSet<usize>>,
            areacalc: RefMut<AreaCalcStorage>) {

        if self.ignore || seg.cell_assignments[self.i] == self.state {
            self.accept = false;
            return;
        }

        let (mut areacalc_vertices, mut areacalc_hull) =
            RefMut::map_split(areacalc, |areacalc| (&mut areacalc.vertices, &mut areacalc.hull));

        let prev_state = seg.cell_assignments[self.i];
        let from_background = prev_state == seg.ncells() as u32;
        let to_background = self.state == seg.ncells() as u32;
        let this_transcript = &seg.transcripts[self.i];

        // Current total log prob
        let mut current_logprob = 0.0_f32;
        let mut proposal_logprob = 0.0;

        if from_background {
            current_logprob += priors.background_logprob;

        } else {
            current_logprob += priors.foreground_logprob;

            current_logprob += seg.cell_logprobs[prev_state as usize];

            // recompute area for `self.state` after reassigning this transcript
            areacalc_vertices.clear();
            for j in cell_transcripts[prev_state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                if transcript != this_transcript {
                    areacalc_vertices.push((transcript.x, transcript.y));
                }
            }

            self.from_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull).max(priors.min_cell_size);

            self.counts.assign(&counts.column(prev_state as usize));
            self.counts[this_transcript.gene as usize] -= 1;

            self.counts_ln_factorial.assign(&counts_ln_factorial.column(prev_state as usize));
            self.counts_ln_factorial[this_transcript.gene as usize] =
                lfact(self.counts[this_transcript.gene as usize]);

            self.from_cell_logprob = params.cell_logprob_fast(
                z[prev_state as usize] as usize, self.from_cell_area, &self.counts, &self.counts_ln_factorial);

            proposal_logprob += self.from_cell_logprob;
        }

        if to_background {
            proposal_logprob += priors.background_logprob;

        } else {
            proposal_logprob += priors.foreground_logprob;

            current_logprob += seg.cell_logprobs[self.state as usize];

            // recompute area for `self.state` after reassigning this transcript
            areacalc_vertices.clear();
            for j in cell_transcripts[self.state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                areacalc_vertices.push((transcript.x, transcript.y));
            }
            areacalc_vertices.push((this_transcript.x, this_transcript.y));

            self.to_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull).max(priors.min_cell_size);

            self.counts.assign(&counts.column(self.state as usize));
            self.counts[this_transcript.gene as usize] += 1;

            self.counts_ln_factorial.assign(&counts_ln_factorial.column(self.state as usize));
            self.counts_ln_factorial[this_transcript.gene as usize] =
                lfact(self.counts[this_transcript.gene as usize]);

            self.to_cell_logprob = params.cell_logprob_fast(
                z[self.state as usize] as usize, self.to_cell_area, &self.counts, &self.counts_ln_factorial);

            proposal_logprob += self.to_cell_logprob;
        }

        let mut rng = thread_rng();
        let logu = rng.gen::<f32>().ln();

        // println!("Eval: {} {} {} {} {}", prev_state, self.state, proposal_logprob, current_logprob, self.log_weight);

        self.accept = logu <= (proposal_logprob - current_logprob) + self.log_weight;

    }
}

fn init_cell_assignments(
    transcripts: &Vec<Transcript>,
    nuclei_centroids: &Vec<NucleiCentroid>,
    k: usize,
) -> (Vec<u32>, Vec<usize>) {
    let mut kdtree: KdTree<f32, usize, 2, 32, u32> = KdTree::with_capacity(transcripts.len());

    for (i, transcript) in transcripts.iter().enumerate() {
        kdtree.add(&[transcript.x, transcript.y], i);
    }

    let ncells = nuclei_centroids.len();
    let ntranscripts = transcripts.len();
    let mut cell_assignments = vec![ncells as u32; ntranscripts];
    let mut cell_population = vec![0; ncells + 1];
    cell_population[ncells] = ntranscripts;

    for (i, centroid) in nuclei_centroids.iter().enumerate() {
        for neighbor in kdtree.nearest_n(&[centroid.x, centroid.y], k, &squared_euclidean) {
            cell_assignments[neighbor.item] = i as u32;
            cell_population[ncells] -= 1;
            cell_population[i] += 1;
        }
    }

    return (cell_assignments, cell_population);
}
