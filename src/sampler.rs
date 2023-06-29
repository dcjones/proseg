mod connectivity;
mod distributions;
mod hull;
mod sampleset;
pub mod transcripts;

use connectivity::ConnectivityChecker;
use core::fmt::Debug;
use distributions::{
    gamma_logpdf, lfact, logaddexp, lognormal_logpdf, negbin_logpmf, negbin_logpmf_fast,
    odds_to_prob, prob_to_odds, rand_pois,
    LogFactorial,
    LogGammaPlus,
};
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::izip;
use kiddo::float::distance::squared_euclidean;
use kiddo::float::kdtree::KdTree;
use libm::{lgammaf, log1pf};
use ndarray::{s, Array1, Array2, AsArray, Zip};
use petgraph::visit::{EdgeRef, IntoNeighbors};
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
use transcripts::{coordinate_span, NeighborhoodGraph, NucleiCentroid, Transcript};

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

    // [ngenes, ncells] Cell (and background) mixing coefficients.
    t: Array2<f32>,

    γ_bg: Array1<f32>,
    γ_fg: Array1<f32>,

    // background rate
    λ_bg: Array1<f32>,

    // background relative probability
    p_bg: Array1<f32>,
    log_p_bg: Array1<f32>,
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
            r: r,
            lgamma_r: lgamma_r,
            θ: Array2::<f32>::from_elem((ngenes, ncomponents), 0.1),
            λ: Array2::<f32>::from_elem((ngenes, ncells), 0.1),
            t: Array2::<f32>::from_elem((ngenes, ncells), (ncells as f32).recip()),
            γ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            γ_fg: Array1::<f32>::from_elem(ngenes, 0.0),
            λ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            p_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            log_p_bg: Array1::<f32>::from_elem(ngenes, 0.0),
        };
    }

    fn ncomponents(&self) -> usize {
        return self.π.len();
    }

    fn cell_logprob<'a, A: AsArray<'a, f32> + Debug, B: AsArray<'a, u32> + Debug>(
        &self,
        ts: A,
        cell_area: f32,
        z: u32,
        counts: B,
    ) -> f32 {
        let ts = ts.into();
        let counts = counts.into();

        let count_logprob = Zip::from(&counts)
            .and(&ts)
            .and(&self.log_p_bg)
            .fold(0_f32, |acc, c, t, &log_p_bg| {
                acc + (*c as f32) * logaddexp(log_p_bg, t.ln() - cell_area.ln())
            });

        // if (!count_logprob.is_finite()) {
        //     let tmin = ts.fold(f32::INFINITY, |acc, t| acc.min(*t));
        //     let tmax = ts.fold(f32::NEG_INFINITY, |acc, t| acc.max(*t));
        //     dbg!(count_logprob, &ts, tmin, tmax, cell_area, &counts, z);
        // }
        //
        // TODO: looks like this fails because background has area zero somehow.
        assert!(count_logprob.is_finite());

        // let count_logprob =
        //     Zip::from(&self.r)
        //     .and(&self.lgamma_r)
        //     .and(self.w.column(z))
        //     .and(counts)
        //     .fold(0_f32, |acc, r, lgamma_r, w, c| {
        //         acc + negbin_logpmf(*r, *lgamma_r, odds_to_prob(*w * cell_area), *c)
        //     });

        // this condition is false when we are dealing with the background cell
        if (z as usize) < self.μ_a.len() {
            let area_logprob =
                lognormal_logpdf(self.μ_a[z as usize], self.σ_a[z as usize], cell_area);
            return count_logprob + area_logprob;
        }

        return count_logprob;

        // return count_logprob;
    }

    // // with pre-computed log(factorial(counts))
    // fn cell_logprob_fast<'a, A: AsArray<'a, u32>, B: AsArray<'a, f32>>(
    //     &self, z: usize, cell_area: f32, counts: A, counts_ln_factorial: B) -> f32
    // {
    //     let counts = counts.into();
    //     let counts_ln_factorial = counts_ln_factorial.into();
    //     // TODO: This is super slow, and it doesn't seem to have that much to do
    //     // with the actual function being computed here. It seems just really expensive
    //     // to do this iteration. Why would that be?

    //     let count_logprob = Zip::from(&self.r)
    //         .and(&self.lgamma_r)
    //         .and(self.w.column(z))
    //         .and(counts)
    //         .and(counts_ln_factorial)
    //         .fold(0_f32, |acc, r, lgamma_r, w, c, clf| {
    //             acc + negbin_logpmf_fast(*r, *lgamma_r, odds_to_prob(*w * cell_area), *c, *clf)
    //         });

    //     let area_logprob = lognormal_logpdf(
    //         self.μ_a[z],
    //         self.σ_a[z],
    //         cell_area);

    //     return count_logprob + area_logprob;
    //     // return count_logprob;
    // }
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

        // initialize cell assignments so roughly half of the transcripts are assigned.
        const INIT_NEIGHBORHOOD_PROPORTION: f64 = 0.5;
        let k = (INIT_NEIGHBORHOOD_PROPORTION * (transcripts.len() as f64)
            / (ncells as f64))
            .ceil() as usize;

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
            }

            if proposal.state as usize != self.ncells() {
                sampler.cell_areas[proposal.state as usize] = proposal.to_cell_area;
                self.cell_logprobs[proposal.state as usize] = proposal.to_cell_logprob;
                sampler.counts[[gene as usize, proposal.state as usize]] += 1;
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
        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

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
        let proposals = vec![Proposal::new(ngenes); nchunks];

        let mut rng = rand::thread_rng();
        let z = (0..ncells)
            .map(|_| rng.gen_range(0..ncomponents) as u32)
            .collect::<Vec<_>>()
            .into();

        let mut sampler = Sampler {
            priors,
            chunkquads,
            transcripts: chunked_transcripts,
            params: params,
            cell_transcripts: cell_transcripts,
            cell_areas: cell_areas,
            cell_area_calc_storage: ThreadLocal::new(),
            full_area: 0_f32,
            connectivity_checker: ThreadLocal::new(),
            z_probs: ThreadLocal::new(),
            z: z,
            counts: Array2::<u32>::zeros((ngenes, ncells)),
            foreground_counts: Array2::<u32>::zeros((ngenes, ncells)),
            background_counts: Array1::<u32>::zeros(ngenes),
            total_gene_counts: Array1::<u32>::zeros(ngenes),
            component_counts: Array2::<u32>::zeros((ngenes, ncomponents)),
            proposals: proposals,
            quad: 0,
            sample_num: 0,
            ncells: ncells,
            background_cell: ncells as u32,
            logfact: LogFactorial::new(),
            loggammaplus: Vec::from_iter((0..ngenes).map(|_| LogGammaPlus::default())),
        };

        sampler.full_area = sampler.compute_full_area();
        dbg!(sampler.full_area);
        sampler.pop_bubbles(seg);
        sampler.compute_cell_areas();
        sampler.compute_counts(seg);
        sampler.sample_global_params();
        sampler.compute_cell_logprobs(seg);

        return sampler;
    }

    pub fn counts(&self) -> Array2<u32> {
        return self.counts.clone();
    }

    pub fn log_likelihood(&self, seg: &Segmentation) -> f32 {
        let γ_factor = Zip::from(&self.total_gene_counts)
            .and(&self.params.γ_fg)
            .fold(0_f32, |accum, &c, γ_fg| {
                accum + (c as f32) * γ_fg.ln()
            });

        let bg_log_p = Zip::from(&self.params.log_p_bg)
            .and(self.counts.rows())
            .and(&self.total_gene_counts)
            .fold(0_f32, |accum, log_p_bg, cs, tc| {
                accum + log_p_bg * ((tc - cs.sum()) as f32)
            });

        return
            - self.params.γ_bg.sum()
            - self.params.γ_fg.sum()
            + γ_factor
            + seg.cell_logprobs.iter().sum::<f32>()
            + bg_log_p;
    }

    fn compute_full_area(&self) -> f32 {
        let mut vertices = Vec::from_iter(
            self.transcripts
                .iter()
                .map(|t| (t.transcript.x, t.transcript.y)),
        );
        let mut hull = Vec::new();

        let mut vertices_refcell = RefCell::new(vertices);
        let mut vertices_ref = vertices_refcell.borrow_mut();

        let mut hull_refcell = RefCell::new(hull);
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
                &self.z,
                &self.counts,
                &self.cell_transcripts,
                areacalc,
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
        let ngenes = self.params.λ.shape()[0];

        // // sample π
        // let mut α = vec![1_f64; self.params.ncomponents()];
        // for z_i in self.z.iter() {
        //     α[*z_i as usize] += 1.0;
        // }

        // self.params.π.clear();
        // self.params.π.extend(Dirichlet::new(α).unwrap().sample(&mut rng).iter().map(|x| *x as f32));

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
        self.background_counts.assign(&self.total_gene_counts);
        Zip::from(self.counts.rows())
            .and(self.foreground_counts.rows_mut())
            .and(self.params.t.rows())
            .and(&mut self.background_counts)
            .and(&self.params.p_bg)
            .par_for_each(|cs, fcs, ts, bc, p_bg| {
                let mut rng = thread_rng();
                for (c, fc, t, cell_area) in izip!(cs, fcs, ts, &self.cell_areas) {
                    let p_fg = t / cell_area;
                    let p = p_fg / (p_fg + p_bg);
                    *fc = Binomial::new(*c as u64, p as f64).unwrap().sample(&mut rng) as u32;
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
                }
            });

        // TODO: I think because we are wildly overdispersed, we just end
        // up with some extreme values of λ.
        // dbg!(self.cell_areas[0]);
        // println!();
        // dbg!(self.counts.slice(s![0..ngenes, 0]));
        // println!();
        // dbg!(self.params.λ.slice(s![0..ngenes, 0]));
        // println!();
        // dbg!(self.params.θ.slice(s![0..ngenes, self.z[0] as usize]));
        // println!();
        // dbg!(&self.params.r);
        // panic!();

        // Sample z (assignment of cells to components)

        // TODO: It seems super unstable to do this.

        // outer loop is over cells
        // Zip::from(&mut self.z)
        //     .and(self.params.λ.slice(s![0..ngenes, 0..self.ncells]).columns())
        //     .par_for_each(|z, λs| {
        //         let mut z_probs = self.z_probs.get_or(|| RefCell::new(vec![0_f64; ncomponents])).borrow_mut();

        //         // loop over components
        //         for (zp, π, θs) in izip!(z_probs.iter_mut(), &self.params.π, self.params.θ.columns()) {
        //             // sum over genes
        //             *zp = (*π as f64) *
        //                 (Zip::from(λs)
        //                 .and(&self.params.r)
        //                 .and(&θs)
        //                 .fold(0_f32, |accum, λ, r, θ| {
        //                     // TODO: seem to be getting some very large λ values
        //                     // dbg!(*r, *θ, *λ, gamma_logpdf(*r, *θ, *λ));
        //                     accum + gamma_logpdf(*r, *θ, *λ)
        //                 }) as f64).exp();

        //             let thing = (Zip::from(λs)
        //                 .and(&self.params.r)
        //                 .and(&θs)
        //                 .fold(0_f32, |accum, λ, r, θ| {
        //                     // TODO: seem to be getting some very large λ values
        //                     dbg!(*r, *θ, *λ, gamma_logpdf(*r, *θ, *λ));
        //                     accum + gamma_logpdf(*r, *θ, *λ)
        //                 }) as f64);
        //             dbg!(thing);

        //             // TODO: What the fuck. It's actually too large. I guess from
        //             // numbers near 0?
        //             //
        //             // How can this work if we occasionally get stupidly large
        //             // probability densities. It's just going to be constantly
        //             // swapping components.

        //             // Try to understand why we are sampling such tiny values.

        //             dbg!(*zp);
        //         }
        //         let z_prob_sum = z_probs.iter().sum::<f64>();
        //         assert!(z_prob_sum.is_finite());

        //         // cumulative probabilities in-place
        //         z_probs.iter_mut().fold(0.0, |mut acc, x| {
        //             acc += *x / z_prob_sum;
        //             *x = acc;
        //             acc
        //         });

        //         let rng = &mut thread_rng();
        //         let u = rng.gen::<f64>();
        //         *z = z_probs.partition_point(|x| *x < u) as u32;
        //     });

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
                //  2. Implement a lookup table for log factorials
                //  3. Possibly for each r we can compute lgamma(r + k) values of k.

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
        // Zip::from(&mut self.params.λ_bg)
        //     .and(&self.background_counts)
        //     .for_each(|λ, c| {
        //         *λ = (*c as f32) / self.full_area;
        //     });

        // TODO: What if we instead force the background to be some
        // proportion of the transcripts.
        Zip::from(&mut self.params.λ_bg)
            .and(&self.total_gene_counts)
            .for_each(|λ, c| {
                *λ = 0.05 * (*c as f32) / self.full_area;
            });

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

        // Compute relative background probability (p_bg)
        Zip::from(&mut self.params.p_bg)
            .and(&mut self.params.log_p_bg)
            .and(&self.params.γ_bg)
            .and(&self.params.γ_fg)
            .for_each(|p, lp, γ_bg, γ_fg| {
                *p = (*γ_bg / *γ_fg) / self.full_area;
                *lp = p.ln();
            });

        // Compute t (per-gene cell mixture coeficients)
        Zip::from(self.params.t.rows_mut())
            .and(self.params.λ.rows())
            .and(&self.params.γ_fg)
            .par_for_each(|mut ts, λs, γ| {
                for (t, λ, cell_area) in izip!(&mut ts, λs, &self.cell_areas) {
                    *t = (*λ * cell_area) / *γ;
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

    pub fn compute_cell_logprobs(&self, seg: &mut Segmentation) {
        // Assume at this point that cell areas and transcript counts are up to date
        seg.cell_logprobs
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, logprob)| {
                let cell_area = self.cell_areas[i];
                *logprob = self.params.cell_logprob(
                    self.params.t.column(i),
                    cell_area,
                    self.z[i],
                    self.counts.column(i),
                );
            });

        // dbg!(&seg.cell_logprobs);
    }
    
    // Bubbles (a transcript of state u, with no state u neighbors) are impossible
    // to burst while retaining detail balance, so we burst them on initialization
    // and try to avoid introducing any by preserving local connectivity.
    fn pop_bubbles(&mut self, seg: &Segmentation) {
        let mut bubble_count = 0;
        for i in 0..self.ncells {
            let u = seg.cell_assignments[i];
            let isbubble = !seg.adjacency.neighbors(i).any(|j| {
                let v = seg.cell_assignments[j];
                v == self.background_cell || v == u
            });

            if isbubble {
                bubble_count += 1;
            }
        };

        // TODO: We should implement this, but there are only a few bubbles, so
        // this isn't a huge issue currently.

        // dbg!(bubble_count);
        // panic!();
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
        }
    }

    fn evaluate(
        &mut self,
        seg: &Segmentation,
        priors: &ModelPriors,
        params: &ModelParams,
        z: &Array1<u32>,
        counts: &Array2<u32>,
        cell_transcripts: &Vec<HashSet<usize>>,
        areacalc: RefMut<AreaCalcStorage>,
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
        let log_p_bg = params.log_p_bg[this_transcript.gene as usize];

        // Current total log prob
        let current_logprob = if from_background {
            log_p_bg
        } else {
            seg.cell_logprobs[prev_state as usize]
        } + if to_background {
            0_f32
        } else {
            seg.cell_logprobs[self.state as usize]
        };

        // let mut proposal_logprob = 0.0;
        //
        //

        // TODO: figure out what to do here wrt to background relative prob.

        if from_background {
            self.from_cell_logprob = 0.0;
        } else {
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

            self.counts.assign(&counts.column(prev_state as usize));
            self.counts[this_transcript.gene as usize] -= 1;

            let z_prev = if from_background {
                u32::MAX
            } else {
                z[prev_state as usize]
            };
            self.from_cell_logprob = params.cell_logprob(
                params.t.column(prev_state as usize),
                self.from_cell_area,
                z_prev,
                &self.counts,
            );
        }

        if to_background {
            self.to_cell_logprob = log_p_bg;
        } else {
            // recompute area for `self.state` after reassigning this transcript
            areacalc_vertices.clear();
            for j in cell_transcripts[self.state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                areacalc_vertices.push((transcript.x, transcript.y));
            }
            areacalc_vertices.push((this_transcript.x, this_transcript.y));

            self.to_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull)
               .max(priors.min_cell_size);

            self.counts.assign(&counts.column(self.state as usize));
            self.counts[this_transcript.gene as usize] += 1;

            let z_to = if to_background {
                u32::MAX
            } else {
                z[self.state as usize]
            };
            self.to_cell_logprob = params.cell_logprob(
                params.t.column(self.state as usize),
                self.to_cell_area,
                z_to,
                &self.counts,
            );
        }

        let mut rng = thread_rng();
        let logu = rng.gen::<f32>().ln();

        // println!("Eval: {} {} {} {} {}", prev_state, self.state, proposal_logprob, current_logprob, self.log_weight);

        let proposal_logprob = self.from_cell_logprob + self.to_cell_logprob;

        self.accept = logu <= (proposal_logprob - current_logprob) + self.log_weight;

        // DEBUG: Why are we rejecting from background proposals?
        // if !self.accept && from_background {
        //     dbg!(self.log_weight, proposal_logprob, current_logprob,
        //         // seg.cell_logprobs[prev_state as usize],
        //         seg.cell_logprobs[self.state as usize],
        //         self.from_cell_logprob, self.to_cell_logprob);
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
