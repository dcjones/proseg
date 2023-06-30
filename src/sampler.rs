mod connectivity;
mod math;
mod hull;
mod sampleset;
pub mod transcripts;

use connectivity::ConnectivityChecker;
use core::fmt::Debug;
use math::{
    negbin_logpmf_fast,
    odds_to_prob, prob_to_odds, rand_pois,
    LogFactorial,
    LogGammaPlus,
};
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::izip;
use libm::{lgammaf, log1pf};
use ndarray::{Array1, Array2, Zip};
use petgraph::visit::IntoNeighbors;
use rand::{thread_rng, Rng};
use rand_distr::{Binomial, Distribution};
use rayon::prelude::*;
use sampleset::SampleSet;
use statrs::distribution::{Beta, Dirichlet, Gamma, InverseGamma, Normal};
use std::cell::{RefCell, RefMut};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use thread_local::ThreadLocal;
use transcripts::{coordinate_span, NeighborhoodGraph, Transcript};

// use std::time::Instant;

#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub min_cell_size: f32,

    // params for normal prior
    pub μ_μ_a: f32,
    pub σ_μ_a: f32,

    // params for inverse-gamma prior
    pub α_σ_a: f32,
    pub β_σ_a: f32,

    pub α_θ: f32,
    pub β_θ: f32,

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

    // Precomputing lgamma(r)
    lgamma_r: Array1<f32>,

    // [ngenes, ncomponents] NB p parameters.
    θ: Array2<f32>,

    // // log(ods_to_prob(θ))
    // logp: Array2<f32>,

    // // log(1 - ods_to_prob(θ))
    // log1mp: Array2<f32>,

    // [ngenes, ncells] Poisson rates
    λ: Array2<f32>,

    γ_bg: Array1<f32>,
    γ_fg: Array1<f32>,

    // background rate
    λ_bg: Array1<f32>,
}

impl ModelParams {
    // initialize model parameters, with random cell assignments
    // and other parameterz unninitialized.
    fn new(priors: &ModelPriors, ncomponents: usize, ncells: usize, ngenes: usize) -> Self {
        let r = Array1::<f32>::from_elem(ngenes, 100.0_f32);
        let lgamma_r = Array1::<f32>::from_iter(r.iter().map(|&x| lgammaf(x)));
        return ModelParams {
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_a: vec![priors.μ_μ_a; ncomponents],
            σ_a: vec![priors.σ_μ_a; ncomponents],
            r,
            lgamma_r,
            θ: Array2::<f32>::from_elem((ngenes, ncomponents), 0.1),
            λ: Array2::<f32>::from_elem((ngenes, ncells), 0.1),
            γ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            γ_fg: Array1::<f32>::from_elem(ngenes, 0.0),
            λ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
        };
    }

    fn ncomponents(&self) -> usize {
        return self.π.len();
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

    let chunk = (xchunkquad / 2) + (ychunkquad / 2) * (nxchunks as u32);
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
                chunk,
                quad,
            }
        })
        .collect();
}

// Holds the segmentation state.
// This is kept outside of Sample mainly so we can borrow this in multiple threads
// while modifying the sampler when generating proposals.
pub struct Segmentation<'a> {
    transcripts: &'a Vec<Transcript>,
    adjacency: &'a NeighborhoodGraph,
    pub cell_assignments: Vec<u32>,
    cell_population: Vec<usize>,
    pub cell_logprobs: Vec<f32>,
}

impl<'a> Segmentation<'a> {
    pub fn new(
        transcripts: &'a Vec<Transcript>,
        adjacency: &'a NeighborhoodGraph,
        init_cell_assignments: Vec<u32>,
        init_cell_population: Vec<usize>,
    ) -> Segmentation<'a> {

        let ncells = init_cell_population.len() - 1;
        let cell_logprobs = vec![0.0; ncells];

        return Segmentation {
            transcripts,
            adjacency,
            cell_assignments: init_cell_assignments,
            cell_population: init_cell_population,
            cell_logprobs,
        };
    }

    pub fn nunassigned(&self) -> usize {
        let ncells = self.ncells() as u32;
        return self
            .cell_assignments
            .iter()
            .filter(|&c| *c == ncells)
            .count();
    }

    fn ncells(&self) -> usize {
        return self.cell_population.len() - 1;
    }

    pub fn apply_local_updates(&mut self, sampler: &mut Sampler) {
        // let accept_count = sampler.proposals.iter().filter(|p| p.accept).count();
        // let unignored_count = sampler.proposals.iter().filter(|p| !p.ignore).count();
        // println!("Applying {} of {} proposals", accept_count, unignored_count);

        // TODO: check if we are doing multiple updates on the same cell and warn
        // about it.

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
            }

            if proposal.state as usize != self.ncells() {
                sampler.cell_areas[proposal.state as usize] = proposal.to_cell_area;
                self.cell_logprobs[proposal.state as usize] = proposal.to_cell_logprob;
                sampler.counts[[gene as usize, proposal.state as usize]] += 1;
            }

            if prev_state as usize == self.ncells() && proposal.state as usize != self.ncells() {
                sampler.proposal_stats.background_to_cell_accept += 1;
            } else if prev_state as usize != self.ncells() && proposal.state as usize == self.ncells() {
                sampler.proposal_stats.cell_to_background_accept += 1;
            } else if prev_state as usize != self.ncells() && proposal.state as usize != self.ncells() {
                sampler.proposal_stats.cell_to_cell_accept += 1;
            }
        }

        for proposal in sampler.proposals.iter().filter(|p| !p.accept) {
            let prev_state = self.cell_assignments[proposal.i];
            if prev_state as usize == self.ncells() && proposal.state as usize != self.ncells() {
                sampler.proposal_stats.background_to_cell_reject += 1;
                if proposal.ignore {
                    sampler.proposal_stats.background_to_cell_ignore += 1;
                }
            } else if prev_state as usize != self.ncells() && proposal.state as usize == self.ncells() {
                sampler.proposal_stats.cell_to_background_reject += 1;
            } else if prev_state as usize != self.ncells() && proposal.state as usize != self.ncells() {
                sampler.proposal_stats.cell_to_cell_reject += 1;
            }
        }

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
        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

        let vertices: Vec<(f32, f32)> = Vec::new();
        let hull: Vec<(f32, f32)> = Vec::new();

        let vertices_refcell = RefCell::new(vertices);
        let mut vertices_ref = vertices_refcell.borrow_mut();

        let hull_refcell = RefCell::new(hull);
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
                    "          ["
                ),
                i, area
            )
            .unwrap();
            for (i, (x, y)) in hull_ref.iter().enumerate() {
                writeln!(encoder, "            [{}, {}]", x, y).unwrap();
                if i < hull_ref.len() - 1 {
                    write!(encoder, ",").unwrap();
                }
            }
            write!(
                encoder,
                concat!(
                    "          ]\n", // polygon
                    "        ]\n",   // coordinates
                    "      }}\n",    // geometry
                    "    }}\n",      // feature
                )
            )
            .unwrap();

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
    pub proposal_stats: ProposalStats,
    cell_transcripts: Vec<HashSet<usize>>,
    cell_areas: Array1<f32>,
    cell_area_calc_storage: ThreadLocal<RefCell<AreaCalcStorage>>,
    full_area: f32,
    connectivity_checker: ThreadLocal<RefCell<ConnectivityChecker>>,
    pub z: Array1<u32>, // assignment of cells to components
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,
    counts: Array2<u32>,
    foreground_counts: Array2<u32>,
    background_counts: Array1<u32>,
    total_gene_counts: Array1<u32>,
    component_counts: Array2<u32>,
    quad: usize,
    sample_num: usize,
    ncells: usize,
    background_cell: u32,
    logfact: LogFactorial,
    loggammaplus: Vec<LogGammaPlus>,
}

impl Sampler {
    pub fn new(
        priors: ModelPriors,
        seg: &mut Segmentation,
        ncomponents: usize,
        ngenes: usize,
        chunk_size: f32,
    ) -> Sampler {
        let (xmin, xmax, ymin, ymax) = coordinate_span(seg.transcripts);

        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;
        let chunked_transcripts =
            chunk_transcripts(seg.transcripts, xmin, ymin, chunk_size, nxchunks);

        let ncells = seg.ncells();

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

        let params = ModelParams::new(&priors, ncomponents, ncells, ngenes);

        let mut cell_transcripts = vec![HashSet::new(); ncells];
        for (i, cell) in seg.cell_assignments.iter().enumerate() {
            if (*cell as usize) < ncells {
                cell_transcripts[*cell as usize].insert(i);
            }
        }

        let cell_areas = Array1::<f32>::zeros(ncells);
        let proposals = vec![Proposal::new(); nchunks];

        let mut rng = rand::thread_rng();
        let z = (0..ncells)
            .map(|_| rng.gen_range(0..ncomponents) as u32)
            .collect::<Vec<_>>()
            .into();

        let mut sampler = Sampler {
            priors,
            chunkquads,
            transcripts: chunked_transcripts,
            params,
            cell_transcripts,
            cell_areas,
            cell_area_calc_storage: ThreadLocal::new(),
            full_area: 0_f32,
            connectivity_checker: ThreadLocal::new(),
            z_probs: ThreadLocal::new(),
            z,
            counts: Array2::<u32>::zeros((ngenes, ncells)),
            foreground_counts: Array2::<u32>::zeros((ngenes, ncells)),
            background_counts: Array1::<u32>::zeros(ngenes),
            total_gene_counts: Array1::<u32>::zeros(ngenes),
            component_counts: Array2::<u32>::zeros((ngenes, ncomponents)),
            proposals,
            proposal_stats: ProposalStats::new(),
            quad: 0,
            sample_num: 0,
            ncells,
            background_cell: ncells as u32,
            logfact: LogFactorial::new(),
            loggammaplus: Vec::from_iter((0..ngenes).map(|_| LogGammaPlus::default())),
        };

        sampler.full_area = sampler.compute_full_area();
        dbg!(sampler.full_area);
        // sampler.pop_bubbles(seg);
        sampler.compute_cell_areas();
        sampler.compute_counts(seg);
        sampler.sample_global_params();

        return sampler;
    }

    pub fn counts(&self) -> Array2<u32> {
        return self.counts.clone();
    }

    pub fn log_likelihood(&self) -> f32 {
        let mut ll = Zip::from(self.params.λ.columns())
            .and(&self.cell_areas)
            .and(self.counts.columns())
            .fold(0_f32, |accum, λs, cell_area, cs| {
                accum + Zip::from(λs)
                    .and(&self.params.λ_bg)
                    .and(cs)
                    .fold(0_f32, |accum, λ, λ_bg, &c| {
                        accum + (c as f32) * (λ + λ_bg).ln() - λ * cell_area
                    })
            });

        // background terms
        ll += Zip::from(&self.total_gene_counts)
            .and(self.counts.rows())
            .and(&self.params.λ_bg)
            .fold(0_f32, |accum, c_total, cs, &λ| {
                let c_bg = c_total - cs.sum();
                accum + (c_bg as f32) * λ.ln() - λ * self.full_area
            });

        return ll;
    }

    fn compute_full_area(&self) -> f32 {
        let vertices = Vec::from_iter(
            self.transcripts
                .iter()
                .map(|t| (t.transcript.x, t.transcript.y)),
        );
        let hull = Vec::new();

        let vertices_refcell = RefCell::new(vertices);
        let mut vertices_ref = vertices_refcell.borrow_mut();

        let hull_refcell = RefCell::new(hull);
        let mut hull_ref = hull_refcell.borrow_mut();

        return convex_hull_area(&mut vertices_ref, &mut hull_ref);
    }

    pub fn sample_local_updates(&mut self, seg: &Segmentation) {
        // let t0 = Instant::now();
        self.repoulate_proposals(seg);
        // let t1 = Instant::now();
        // println!("repoulate_proposals took {:?}", t1 - t0);

        self.proposals.par_iter_mut().for_each(|proposal| {
            let areacalc = self
                .cell_area_calc_storage
                .get_or(|| RefCell::new(AreaCalcStorage::new()))
                .borrow_mut();

            proposal.evaluate(
                seg,
                &self.priors,
                &self.params,
                &self.cell_transcripts,
                &self.cell_areas,
                areacalc,
                &self.counts,
            );
        });
        // let t2 = Instant::now();
        // println!("evaluate took {:?}", t2 - t1);

        self.sample_num += 1;
    }

    fn repoulate_proposals(&mut self, seg: &Segmentation) {
        const UNASSIGNED_PROPOSAL_PROB: f64 = 0.05;
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
                let mut connectivity_checker = self
                    .connectivity_checker
                    .get_or(|| RefCell::new(ConnectivityChecker::new()))
                    .borrow_mut();

                let art_from = connectivity_checker.isarticulation(
                    &seg.adjacency,
                    &seg.cell_assignments,
                    *i,
                    cell_from,
                );

                let art_to = connectivity_checker.isarticulation(
                    &seg.adjacency,
                    &seg.cell_assignments,
                    *i,
                    cell_to,
                );

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

                let mut proposal_prob =
                    num_new_state_neighbors as f64 / num_mismatching_edges as f64;
                // If this is an unassigned proposal, account for multiple ways of doing unassigned proposals
                if cell_to == ncells as u32 {
                    let num_mismatching_neighbors = seg
                        .adjacency
                        .neighbors(*i)
                        .filter(|j| seg.cell_assignments[*j] != cell_from)
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
                if seg.cell_assignments[*i] == ncells as u32 {
                    let new_num_mismatching_neighbors = seg
                        .adjacency
                        .neighbors(*i)
                        .filter(|j| seg.cell_assignments[*j] != cell_to)
                        .count();
                    reverse_proposal_prob = UNASSIGNED_PROPOSAL_PROB
                        * (new_num_mismatching_neighbors as f64 / new_num_mismatching_edges as f64)
                        + (1.0 - UNASSIGNED_PROPOSAL_PROB) * reverse_proposal_prob;
                }

                // let unassign_proposal {
                //     // TODO:
                // }

                proposal.i = *i;
                proposal.state = cell_to;
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

    pub fn sample_global_params(&mut self) {
        // TODO: we are doing some allocation in this function that can be avoided
        // by pre-allocating and storing in Sampler.

        let mut rng = thread_rng();
        let ncomponents = self.params.ncomponents();


        // sample μ_a
        let mut μ_μ_a =
            vec![self.priors.μ_μ_a / (self.priors.σ_μ_a.powi(2)); self.params.ncomponents()];
        // let mut component_population = vec![0; self.params.ncomponents()];
        let mut component_population = Array1::<u32>::from_elem(self.params.ncomponents(), 0_u32);

        for (z_i, cell_area) in self.z.iter().zip(&self.cell_areas) {
            μ_μ_a[*z_i as usize] += cell_area.ln() / self.params.σ_a[*z_i as usize].powi(2);
            component_population[*z_i as usize] += 1;
        }
        // dbg!(&component_population);

        μ_μ_a.iter_mut().enumerate().for_each(|(i, μ)| {
            *μ /= (1_f32 / self.priors.σ_μ_a.powi(2))
                + (component_population[i] as f32 / self.params.σ_a[i].powi(2));
        });

        let σ2_μ_a: Vec<f32> = izip!(&self.params.σ_a, &component_population)
            .map(|(σ, n)| (1_f32 / self.priors.σ_μ_a.powi(2) + (*n as f32) / σ.powi(2)).recip())
            .collect();

        self.params.μ_a.clear();
        self.params
            .μ_a
            .extend(izip!(&μ_μ_a, &σ2_μ_a).map(|(μ, σ2)| {
                Normal::new(*μ as f64, σ2.sqrt() as f64)
                    .unwrap()
                    .sample(&mut rng) as f32
            }));

        // sample σ_a
        let mut δ2s = vec![0_f32; self.params.ncomponents()];
        for (z_i, cell_area) in self.z.iter().zip(&self.cell_areas) {
            δ2s[*z_i as usize] += (cell_area.ln() - μ_μ_a[*z_i as usize]).powi(2);
        }

        // sample σ_a
        self.params.σ_a.clear();
        self.params
            .σ_a
            .extend(izip!(&component_population, &δ2s).map(|(n, δ2)| {
                InverseGamma::new(
                    self.priors.α_σ_a as f64 + 0.5 * *n as f64,
                    self.priors.β_σ_a as f64 + 0.5 * *δ2 as f64,
                )
                .unwrap()
                .sample(&mut rng)
                .sqrt() as f32
            }));

        // Sample background/foreground counts
        // (This seems a little pointless. As devised now, if a transcript is under a cell it has a
        // very high probability of coming from that cell)
        self.background_counts.assign(&self.total_gene_counts);
        Zip::from(self.counts.rows())
            .and(self.foreground_counts.rows_mut())
            .and(&mut self.background_counts)
            .and(self.params.λ.rows())
            .and(&self.params.λ_bg)
            .par_for_each(|cs, fcs, bc, λs, λ_bg| {
                let mut rng = thread_rng();

                let p_bg = λ_bg * (-λ_bg * self.full_area).exp();

                for (c, fc, λ, cell_area) in izip!(cs, fcs, λs, &self.cell_areas) {

                    // let p_fg = λ * (-λ * cell_area).exp();
                    // let p = p_fg / (p_fg + p_bg);
                    //
                    let p = λ / (λ + λ_bg);


                    // if p == 0.0 {
                    //     *fc = 0;
                    // } else if p == 1.0 {
                    //     *fc = *c;
                    // } else {
                    //     *fc = Binomial::new(*c as u64, p as f64).unwrap().sample(&mut rng) as u32;
                    // }

                    *fc = Binomial::new(*c as u64, p as f64).unwrap().sample(&mut rng) as u32;

                    // TODO: This seems fucked overall. I get
                    // *fc = *c;

                    *bc -= *fc;
                }
            });

        // total component area
        let mut component_cell_area = vec![0_f32; self.params.ncomponents()];
        self.cell_areas
            .iter()
            .take(self.ncells)
            .zip(&self.z)
            .for_each(|(area, z_i)| {
                component_cell_area[*z_i as usize] += *area;
            });

        // compute per component transcript counts
        self.component_counts.fill(0);
        Zip::from(self.component_counts.rows_mut())
            .and(self.foreground_counts.rows())
            .par_for_each(|mut compc, cellc| {
                for (c, component) in cellc.iter().zip(&self.z) {
                    compc[*component as usize] += *c;
                }
            });

        // Sample θ
        Zip::from(self.params.θ.rows_mut())
            .and(self.component_counts.rows())
            .and(&self.params.r)
            .par_for_each(|θs, cs, r| {
                let mut rng = thread_rng();
                for (θ, c, a) in izip!(θs, cs, &component_cell_area) {
                    *θ = prob_to_odds(
                        Beta::new(
                            (self.priors.α_θ + *c as f32) as f64,
                            (self.priors.β_θ + *a * *r) as f64,
                        )
                        .unwrap()
                        .sample(&mut rng) as f32,
                    );

                    *θ = θ.max(1e-6);

                    // if *w > 1000.0 {
                    //     dbg!(w, c, a, r);
                    //     panic!("w is too large");
                    // }
                }
            });

        // println!("θ span {} {} {}",
        //     self.params.θ.mean().unwrap(),
        //     self.params.θ.fold(f32::MAX, |accum, &x| accum.min(x)),
        //     self.params.θ.fold(f32::MIN, |accum, &x| accum.max(x)));

        // Sample r
        Zip::from(&mut self.params.r)
            .and(&mut self.params.lgamma_r)
            .and(&mut self.loggammaplus)
            .and(self.params.θ.rows())
            // .and(self.counts.rows())
            .par_for_each(|r, lgamma_r, loggammaplus, θs| {
                let mut rng = thread_rng();

                // self.cell_areas.slice(0..self.ncells)

                let u = Zip::from(&self.z)
                    .and(&self.cell_areas)
                    .fold(0, |accum, z, a| {
                        let θ = θs[*z as usize];
                        let λ = -*r * log1pf(-odds_to_prob(θ * *a));
                        // assert!(λ >= 0.0);
                        // accum + Poisson::new(λ as f64).unwrap().sample(&mut rng)
                        accum + rand_pois(&mut rng, λ)
                    }) as f32;
                let v = Zip::from(&self.z)
                    .and(&self.cell_areas)
                    .fold(0.0, |accum, z, a| {
                        let w = θs[*z as usize];
                        accum + log1pf(-odds_to_prob(w * *a))
                    });

                *r = Gamma::new((self.priors.e_r + u) as f64, (self.priors.f_r - v) as f64)
                    .unwrap()
                    .sample(&mut rng) as f32;

                *lgamma_r = lgammaf(*r);
                loggammaplus.reset(*r);

                // TODO: any better solution here?
                *r = r.min(100.0).max(1e-4);
            });

        // println!("r span {} {} {}",
        //     self.params.r.mean().unwrap(),
        //     self.params.r.fold(f32::MAX, |accum, &x| accum.min(x)),
        //     self.params.r.fold(f32::MIN, |accum, &x| accum.max(x)));

        // Sample λ
        Zip::from(self.params.λ.rows_mut())
            .and(self.foreground_counts.rows())
            .and(self.params.θ.rows())
            .and(&self.params.r)
            .par_for_each(|mut λs, cs, θs, r| {
                let mut rng = thread_rng();

                // TODO: Afraid this is where we'll get killed on performance. Look for
                // a Gamma distribution sampler that runs as f32 precision. Maybe in rand_distr

                for (λ, z, c, cell_area) in izip!(
                    &mut λs,
                    &self.z,
                    cs,
                    &self.cell_areas
                ) {
                    let θ = θs[*z as usize];
                    *λ = Gamma::new(
                        *r as f64 + *c as f64,
                        // (θ / (cell_area * θ + 1.0)) as f64
                        ((cell_area * θ + 1.0) / θ) as f64,
                    )
                    .unwrap()
                    .sample(&mut rng)
                    .max(1e-9) as f32;

                    // dbg!(*λ, *r, θ, *c, cell_area, *c as f32 / cell_area);
                }
            });

        // Sample z
        Zip::from(
            self.foreground_counts
                .columns(),
        )
        .and(&mut self.z)
        .and(&self.cell_areas)
        .par_for_each(|cs, z_i, cell_area| {
            let mut z_probs = self
                .z_probs
                .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                .borrow_mut();

            // loop over components
            for (zp, π, θs) in izip!(z_probs.iter_mut(), &self.params.π, self.params.θ.columns())
            {
                // TODO: This is a big bottleneck due to negbinom pmf being so expensive. Because
                // most counts are small, we should be able to precompute some values.
                //  1. Precompute log(odds_to_prob(θ)) and log(1 - odds_to_prob(θ))
                //  (Maybe not worth it)
                //  2. Implement a lookup table for log factorials (DONE)
                //  3. Implement a lookup table for lgamma(r + k) (DONE)

                // sum over genes
                *zp = (*π as f64)
                    * (Zip::from(cs)
                        .and(&self.params.r)
                        .and(&self.params.lgamma_r)
                        .and(&self.loggammaplus)
                        .and(&θs)
                        .fold(0_f32, |accum, c, r, lgamma_r, lgammaplus, θ| {
                            accum + negbin_logpmf_fast(
                                *r, *lgamma_r, lgammaplus.eval(*c),
                                odds_to_prob(*θ * cell_area), *c, self.logfact.eval(*c))
                        }) as f64)
                        .exp();
            }

            // z_probs.iter_mut().enumerate().for_each(|(j, zp)| {
            //     *zp = (self.params.π[j] as f64) *
            //         negbin_logpmf(r, lgamma_r, p, k)
            //         // (self.params.cell_logprob_fast(j as usize, *cell_area, &cs, &clfs) as f64).exp();
            // });

            let z_prob_sum = z_probs.iter().sum::<f64>();
            assert!(z_prob_sum.is_finite());

            // cumulative probabilities in-place
            z_probs.iter_mut().fold(0.0, |mut acc, x| {
                acc += *x / z_prob_sum;
                *x = acc;
                acc
            });

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
        self.params.π.extend(
            Dirichlet::new(α)
                .unwrap()
                .sample(&mut rng)
                .iter()
                .map(|x| *x as f32),
        );

        // Special case for background noise.
        Zip::from(&mut self.params.λ_bg)
            .and(&self.background_counts)
            .for_each(|λ, c| {
                *λ = (*c as f32) / self.full_area;
            });

        // TODO: What if we instead force the background to be some
        // proportion of the transcripts.
        // Zip::from(&mut self.params.λ_bg)
        //     .and(&self.total_gene_counts)
        //     .for_each(|λ, c| {
        //         *λ = 0.05 * (*c as f32) / self.full_area;
        //     });

        // dbg!(&self.background_counts, &self.params.λ_bg);

        // Comptue γ_bg
        Zip::from(&mut self.params.γ_bg)
            .and(&self.params.λ_bg)
            .for_each(|γ, λ| {
                *γ = λ * self.full_area;
            });

        // Compute γ_fg
        Zip::from(&mut self.params.γ_fg)
            .and(self.params.λ.rows())
            .par_for_each(|γ, λs| {
                *γ = 0.0;
                for (λ, cell_area) in izip!(λs, &self.cell_areas) {
                    *γ += *λ * *cell_area;
                }
            });
    }

    fn compute_cell_areas(&mut self) {
        Zip::indexed(&mut self.cell_areas).par_for_each(|i, area| {
            if i == self.background_cell as usize {
                return;
            }

            let mut areacalc = self
                .cell_area_calc_storage
                .get_or(|| RefCell::new(AreaCalcStorage::new()))
                .borrow_mut();
            areacalc.vertices.clear();

            for j in self.cell_transcripts[i].iter() {
                let transcript = self.transcripts[*j].transcript;
                areacalc.vertices.push((transcript.x, transcript.y));
            }

            let (mut vertices, mut hull) = RefMut::map_split(areacalc, |areacalc| {
                (&mut areacalc.vertices, &mut areacalc.hull)
            });
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
        for (transcript, &j) in self.transcripts.iter().zip(&seg.cell_assignments) {
            let i = transcript.transcript.gene as usize;
            if j != self.background_cell {
                self.counts[[i, j as usize]] += 1;
            }
            self.total_gene_counts[i] += 1;
        }
    }

    // // Bubbles (a transcript of state u, with no state u neighbors) are impossible
    // // to burst while retaining detail balance, so we burst them on initialization
    // // and try to avoid introducing any by preserving local connectivity.
    // fn pop_bubbles(&mut self, seg: &Segmentation) {
    //     let mut bubble_count = 0;
    //     for i in 0..self.ncells {
    //         let u = seg.cell_assignments[i];
    //         let isbubble = !seg.adjacency.neighbors(i).any(|j| {
    //             let v = seg.cell_assignments[j];
    //             v == self.background_cell || v == u
    //         });

    //         if isbubble {
    //             bubble_count += 1;
    //         }
    //     };

    //     // TODO: We should implement this, but there are only a few bubbles, so
    //     // this isn't a huge issue currently.

    //     // dbg!(bubble_count);
    //     // panic!();
    // }

    // TODO: This should go in a test. Too expensive to survail like this in regular usage.
    // pub fn check_mismatch_edges(&self, seg: &Segmentation) {
    //     // check that mismatch edges are symmetric and really are mismatching
    //     for quad in 0..4 {
    //         for quadchunk in &self.chunkquads[quad] {
    //             for (i, j) in quadchunk.mismatch_edges.iter() {
    //                 assert!(seg.cell_assignments[*i] != seg.cell_assignments[*j]);

    //                 let quad_j = self.transcripts[*j].quad as usize;
    //                 let chunk_j = self.transcripts[*j].chunk as usize;
    //                 let quadchunk_j = &self.chunkquads[quad_j][chunk_j];
    //                 assert!(quadchunk_j.mismatch_edges.contains(&(*j, *i)));
    //             }
    //         }
    //     }

    //     // check that every mismatched edge is present in a set
    //     for i in 0..self.transcripts.len() {
    //         for edge in seg.adjacency.edges(i) {
    //             let j = edge.target();
    //             if seg.cell_assignments[i] != seg.cell_assignments[j] {
    //                 let quad_i = self.transcripts[i].quad as usize;
    //                 let chunk_i = self.transcripts[i].chunk as usize;
    //                 let quadchunk_i = &self.chunkquads[quad_i][chunk_i];
    //                 assert!(quadchunk_i.mismatch_edges.contains(&(i, j)));
    //             }
    //         }
    //     }
    // }
}


#[derive(Clone, Debug)]
pub struct ProposalStats {
    cell_to_cell_accept: usize,
    cell_to_cell_reject: usize,
    background_to_cell_accept: usize,
    background_to_cell_reject: usize,
    background_to_cell_ignore: usize,
    cell_to_background_accept: usize,
    cell_to_background_reject: usize,
}


impl ProposalStats {
    fn new() -> Self {
        ProposalStats {
            cell_to_cell_accept: 0,
            cell_to_cell_reject: 0,
            background_to_cell_accept: 0,
            background_to_cell_reject: 0,
            background_to_cell_ignore: 0,
            cell_to_background_accept: 0,
            cell_to_background_reject: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cell_to_cell_accept = 0;
        self.cell_to_cell_reject = 0;
        self.background_to_cell_accept = 0;
        self.background_to_cell_reject = 0;
        self.background_to_cell_ignore = 0;
        self.cell_to_background_accept = 0;
        self.cell_to_background_reject = 0;
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
}

impl Proposal {
    fn new() -> Self {
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
        }
    }

    fn evaluate(
        &mut self,
        seg: &Segmentation,
        priors: &ModelPriors,
        params: &ModelParams,
        cell_transcripts: &Vec<HashSet<usize>>,
        cell_areas: &Array1<f32>,
        areacalc: RefMut<AreaCalcStorage>,
        counts: &Array2<u32>,
    ) {
        if self.ignore || seg.cell_assignments[self.i] == self.state {
            self.accept = false;
            return;
        }

        let (mut areacalc_vertices, mut areacalc_hull) = RefMut::map_split(areacalc, |areacalc| {
            (&mut areacalc.vertices, &mut areacalc.hull)
        });

        let prev_state = seg.cell_assignments[self.i];
        let from_background = prev_state == seg.ncells() as u32;
        let to_background = self.state == seg.ncells() as u32;
        let this_transcript = &seg.transcripts[self.i];

        // Log Metropolis-Hastings acceptance ratio
        let mut δ = 0.0;

        if !from_background {
            // recompute area for `prev_state` after reassigning this transcript
            areacalc_vertices.clear();
            for j in cell_transcripts[prev_state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                if transcript != this_transcript {
                    areacalc_vertices.push((transcript.x, transcript.y));
                }
            }

            self.from_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull)
                .max(priors.min_cell_size);

            let current_from_cell_area = cell_areas[prev_state as usize];

            δ += Zip::from(params.λ.column(prev_state as usize))
                    .fold(0_f32, |accum, λ| accum + λ * (current_from_cell_area - self.from_cell_area));
        }

        if !to_background {
            // recompute area for `self.state` after reassigning this transcript
            areacalc_vertices.clear();
            for j in cell_transcripts[self.state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                areacalc_vertices.push((transcript.x, transcript.y));
            }
            areacalc_vertices.push((this_transcript.x, this_transcript.y));

            self.to_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull)
               .max(priors.min_cell_size);

            let current_to_cell_area = cell_areas[self.state as usize];

            δ += Zip::from(params.λ.column(self.state as usize))
                    .fold(0_f32, |accum, λ| accum + λ * (current_to_cell_area - self.to_cell_area));
        }

        let λ_bg = params.λ_bg[this_transcript.gene as usize];

        if from_background {
            δ -= λ_bg.ln();
        } else {
            let λ_from = params.λ[[this_transcript.gene as usize, prev_state as usize]];
            δ -= (λ_from + λ_bg).ln();
        }

        if to_background {
            δ += λ_bg.ln();
        } else {
            let λ_to = params.λ[[this_transcript.gene as usize, self.state as usize]];
            δ += (λ_to + λ_bg).ln();
        }


        // TODO: Add to δ differences from cell size prior.

        // dbg!(δ);

        let mut rng = thread_rng();
        let logu = rng.gen::<f32>().ln();

        // println!("Eval: {} {} {} {} {}", prev_state, self.state, proposal_logprob, current_logprob, self.log_weight);

        self.accept = logu <= δ + self.log_weight;

        // DEBUG: Why are we rejecting from background proposals?
        // if !self.accept && from_background {
        //     if rng.gen::<f64>() < 0.001 {
        //         let to_cell_count = counts.column(self.state as usize).sum();
        //         let λ_to = params.λ[[this_transcript.gene as usize, self.state as usize]];
        //         let current_to_cell_area = cell_areas[self.state as usize];
        //         dbg!(self.log_weight, δ,
        //             λ_bg, λ_to, 
        //             current_to_cell_area,
        //             self.to_cell_area,
        //             to_cell_count,
        //         );
        //     }
        // }

        // DEBUG: Why are we accepting to_background proposals?
        // if self.accept && to_background {
        //     dbg!(
        //         self.log_weight, proposal_logprob, current_logprob,
        //         seg.cell_logprobs[prev_state as usize],
        //         self.from_cell_logprob, self.to_cell_logprob);
        // }
    }
}

