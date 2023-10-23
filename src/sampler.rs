mod connectivity;
pub mod cubebinsampler;
pub mod density;
pub mod hull;
mod math;
mod sampleset;
pub mod transcripts;

use core::fmt::Debug;
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::{izip, Itertools};
use libm::{lgammaf, log1pf};
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use math::{
    normal_pdf,
    lognormal_logpdf, negbin_logpmf_fast, odds_to_prob, prob_to_odds, rand_crt, LogFactorial,
    LogGammaPlus,
};
use ndarray::{Array1, Array2, Array3, Axis, Zip};
use rand::{thread_rng, Rng};
use rand_distr::{Beta, Dirichlet, Distribution, Gamma, Normal, StandardNormal};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f32;
use std::fs::File;
use std::io::Write;
use std::iter::Iterator;
use thread_local::ThreadLocal;
use transcripts::{CellIndex, Transcript, BACKGROUND_CELL};

// use std::time::Instant;

// Bounding perimeter as some multiple of the perimiter of a sphere with the
// same volume. This of course is all on a lattice, so it's approximate.
// `eta` is the scaling factor between number of mismatching neighbors on the
// lattice and perimeter.
//
// This is inspired by:
// Magno,R., Grieneisen,V.A. and Marée,A.F. (2015) The biophysical nature of
// cells: potential cell behaviours revealed by analytical and computational
// studies of cell surface mechanics. BMC Biophys., 8, 8.
pub fn perimeter_bound(eta: f32, bound: f32, population: f32) -> f32 {
    return bound * eta * (2.0 * (f32::consts::PI * population).sqrt());
}

// Compute chunk and quadrant for a single a single (x,y) point.
fn chunkquad(x: f32, y: f32, xmin: f32, ymin: f32, chunk_size: f32, nxchunks: usize) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size / 2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size / 2.0)).floor() as u32;

    let chunk = (xchunkquad / 2) + (ychunkquad / 2) * (nxchunks as u32);
    let quad = (xchunkquad % 2) + (ychunkquad % 2) * 2;

    return (chunk, quad);
}

// Model prior parameters.
#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub dispersion: Option<f32>,
    pub burnin_dispersion: Option<f32>,

    pub min_cell_volume: f32,

    // params for normal prior
    pub μ_μ_volume: f32,
    pub σ_μ_volume: f32,

    // params for inverse-gamma prior
    pub α_σ_volume: f32,
    pub β_σ_volume: f32,

    pub α_θ: f32,
    pub β_θ: f32,

    // gamma rate prior
    pub e_r: f32,

    pub e_h: f32,
    pub f_h: f32,

    // gamma prior for background rates
    pub α_bg: f32,
    pub β_bg: f32,

    // scaling factor for circle perimeters
    pub perimeter_eta: f32,
    pub perimeter_bound: f32,

    pub nuclear_reassignment_log_prob: f32,
    pub nuclear_reassignment_1mlog_prob: f32,

    // mixture between diffusion prior components
    pub p_diffusion: f32,
    pub σ_diffusion_proposal: f32,
    pub σ_diffusion_near: f32,
    pub σ_diffusion_far: f32,

    pub σ_z_diffusion_proposal: f32,
    pub σ_z_diffusion: f32,

    // bounds on z coordinate
    pub zmin: f32,
    pub zmax: f32,
}

// Model global parameters.
pub struct ModelParams {
    pub transcript_positions: Vec<(f32, f32, f32)>,
    proposed_transcript_positions: Vec<(f32, f32, f32)>,
    accept_proposed_transcript_positions: Vec<bool>,
    transcript_position_updates: Vec<(u32, u32, u32, u32)>,

    init_nuclear_cell_assignment: Vec<CellIndex>,

    pub cell_assignments: Vec<CellIndex>,
    pub cell_assignment_time: Vec<u32>,

    pub cell_population: Vec<usize>,

    // per-cell volumes
    pub cell_volume: Array1<f32>,

    // per-component volumes
    pub component_volume: Array1<f32>,

    // area of the convex hull containing all transcripts
    full_layer_volume: f32,

    // [ntranscripts]
    transcript_density: Array1<f32>,

    z0: f32,
    layer_depth: f32,

    // current assignment of transcripts to background
    isbackground: Array1<bool>,

    // [ngenes, ncells, nlayers] transcripts counts
    pub counts: Array3<u32>,

    // [ncells, ngenes, nlayers] foreground transcripts counts
    foreground_counts: Array3<u32>,

    // [ngenes, nlayers] background transcripts counts
    background_counts: Array2<u32>,

    // [ngenes, nlayers] total gene occourance counts
    pub total_gene_counts: Array2<u32>,

    // Not parameters, but needed for sampling global params
    logfactorial: LogFactorial,

    // TODO: This needs to be an matrix I guess!
    loggammaplus: Array2<LogGammaPlus>,

    pub z: Array1<u32>, // assignment of cells to components

    // [ngenes, ncomponents] number of transcripts of each gene assigned to each component
    component_counts: Array2<u32>,

    component_population: Array1<u32>, // number of cells assigned to each component

    // thread-local space used for sampling z
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,

    π: Vec<f32>, // mixing proportions over components

    μ_volume: Array1<f32>, // volume dist mean param by component
    σ_volume: Array1<f32>, // volume dist std param by component

    // Prior on NB dispersion parameters
    h: f32,

    // [ncomponents, ngenes] NB r parameters.
    pub r: Array2<f32>,

    // [ncomponents, ngenes] gamama parameters for sampling r
    uv: Array2<(u32, f32)>,

    // Precomputing lgamma(r)
    lgamma_r: Array2<f32>,

    // [ncomponents, ngenes] NB p parameters.
    θ: Array2<f32>,

    // // log(ods_to_prob(θ))
    // logp: Array2<f32>,

    // // log(1 - ods_to_prob(θ))
    // log1mp: Array2<f32>,

    // [ngenes, ncells] Poisson rates
    pub λ: Array2<f32>,

    // [ngenes, nlayers] background rate
    pub λ_bg: Array2<f32>,

    // time, which is incremented after every iteration
    t: u32,
}

impl ModelParams {
    // initialize model parameters, with random cell assignments
    // and other parameterz unninitialized.
    pub fn new(
        priors: &ModelPriors,
        full_layer_volume: f32,
        z0: f32,
        layer_depth: f32,
        transcripts: &Vec<Transcript>,
        transcript_density: &Array1<f32>,
        init_cell_assignments: &Vec<u32>,
        init_cell_population: &Vec<usize>,
        ncomponents: usize,
        nlayers: usize,
        ncells: usize,
        ngenes: usize,
    ) -> Self {
        let transcript_positions = transcripts
            .iter()
            .map(|t| (t.x, t.y, t.z))
            .collect::<Vec<_>>();

        let proposed_transcript_positions = transcript_positions.clone();
        let accept_proposed_transcript_positions = vec![false; transcripts.len()];
        let transcript_position_updates = vec![(0, 0, 0, 0); transcripts.len()];

        let r = Array2::<f32>::from_elem((ncomponents, ngenes), 100.0_f32);
        let cell_volume = Array1::<f32>::zeros(ncells);
        let h = 10.0;

        // compute initial counts
        let mut counts = Array3::<u32>::from_elem((ngenes, ncells, nlayers), 0);
        let mut total_gene_counts = Array2::<u32>::from_elem((ngenes, nlayers), 0);
        for (i, &j) in init_cell_assignments.iter().enumerate() {
            let gene = transcripts[i].gene as usize;
            let layer = ((transcripts[i].z - z0) / layer_depth) as usize;
            if j != BACKGROUND_CELL {
                counts[[gene, j as usize, layer]] += 1;
            }
            total_gene_counts[[gene, layer]] += 1;
        }

        // initial component assignments
        let kmeans_samples =
            DatasetBase::from(counts.sum_axis(Axis(0)).map(|&x| (x as f32).ln_1p()));

        let rng = rand::thread_rng();
        let model = KMeans::params_with_rng(ncomponents, rng)
            .tolerance(1e-1)
            .fit(&kmeans_samples)
            .expect("kmeans failed to converge");

        let z = model.predict(&kmeans_samples).map(|&x| x as u32);
        let uv = Array2::<(u32, f32)>::from_elem((ncomponents, ngenes), (0_u32, 0_f32));
        let lgamma_r = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);
        let loggammaplus = Array2::<LogGammaPlus>::from_elem((ncomponents, ngenes), LogGammaPlus::default());

        // // initial component assignments
        // let mut rng = rand::thread_rng();
        // let z = (0..ncells)
        //     .map(|_| rng.gen_range(0..ncomponents) as u32)
        //     .collect::<Vec<_>>()
        //     .into();

        let component_volume = Array1::<f32>::from_elem(ncomponents, 0.0);
        let isbackground = Array1::<bool>::from_elem(transcripts.len(), false);

        return ModelParams {
            transcript_positions,
            proposed_transcript_positions,
            accept_proposed_transcript_positions,
            transcript_position_updates,
            init_nuclear_cell_assignment: init_cell_assignments.clone(),
            cell_assignments: init_cell_assignments.clone(),
            cell_assignment_time: vec![0; init_cell_assignments.len()],
            cell_population: init_cell_population.clone(),
            cell_volume,
            component_volume,
            full_layer_volume,
            transcript_density: transcript_density.clone(),
            z0,
            layer_depth,
            isbackground,
            counts,
            foreground_counts: Array3::<u32>::from_elem((ncells, ngenes, nlayers), 0),
            background_counts: Array2::<u32>::from_elem((ngenes, nlayers), 0),
            total_gene_counts,
            logfactorial: LogFactorial::new(),
            loggammaplus,
            z,
            component_counts: Array2::<u32>::from_elem((ngenes, ncomponents), 0),
            component_population: Array1::<u32>::from_elem(ncomponents, 0),
            z_probs: ThreadLocal::new(),
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_volume: Array1::<f32>::from_elem(ncomponents, priors.μ_μ_volume),
            σ_volume: Array1::<f32>::from_elem(ncomponents, priors.σ_μ_volume),
            h,
            r,
            uv,
            lgamma_r,
            θ: Array2::<f32>::from_elem((ncomponents, ngenes), 0.1),
            λ: Array2::<f32>::from_elem((ngenes, ncells), 0.1),
            λ_bg: Array2::<f32>::from_elem((ngenes, nlayers), 0.0),
            t: 0,
        };
    }

    pub fn ncomponents(&self) -> usize {
        return self.π.len();
    }

    fn zlayer(&self, z: f32) -> usize {
        let layer = ((z - self.z0) / self.layer_depth).max(0.0) as usize;
        let layer = layer.min(self.nlayers() - 1);
        return layer;
    }

    fn recompute_counts(&mut self, transcripts: &Vec<Transcript>) {
        self.counts.fill(0);
        for (i, &j) in self.cell_assignments.iter().enumerate() {
            let gene = transcripts[i].gene as usize;
            if j != BACKGROUND_CELL {
                let layer = self.zlayer(self.transcript_positions[i].2);
                self.counts[[gene, j as usize, layer]] += 1;
            }
        }

        self.check_counts(transcripts);
    }

    fn check_counts(&self, transcripts: &Vec<Transcript>) {
        for (i, (transcript, &assignment)) in transcripts.iter().zip(&self.cell_assignments).enumerate() {
            let layer = self.zlayer(self.transcript_positions[i].2);
            if assignment != BACKGROUND_CELL {
                assert!(self.counts[[transcript.gene as usize, assignment as usize, layer]] > 0);
            }
        }
    }

    pub fn nforeground(&self) -> usize {
        return self.foreground_counts.iter().map(|x| *x as usize).sum();
    }

    pub fn nassigned(&self) -> usize {
        return self
            .cell_assignments
            .iter()
            .filter(|&c| *c != BACKGROUND_CELL)
            .count();
    }

    pub fn ncells(&self) -> usize {
        return self.cell_population.len();
    }

    pub fn ngenes(&self) -> usize {
        return self.total_gene_counts.shape()[0];
    }

    pub fn nlayers(&self) -> usize {
        return self.total_gene_counts.shape()[1];
    }

    pub fn log_likelihood(&self, priors: &ModelPriors) -> f32 {
        // iterate over cells
        let mut ll = Zip::from(self.λ.columns())
            .and(&self.cell_volume)
            // .and(self.counts.axis_iter(Axis(1)))
            .and(self.foreground_counts.axis_iter(Axis(0)))
            .fold(0_f32, |accum, λ, cell_volume, cs| {
                // λs: [ngenes]
                // cell_volume: f32
                // cs: [ngenes, nlayers]
                // λ_bg: [ngenes, nlayers]

                // iterate over genes
                let part = Zip::from(λ)
                        .and(cs.outer_iter())
                        .fold(0_f32, |accum, λ, cs| {
                            accum
                                + Zip::from(cs).fold(0_f32, |accum, &c| {
                                    if c > 0 {
                                        // accum + (c as f32) * (λ + λ_bg).ln()
                                        accum + (c as f32) * λ.ln()
                                    } else {
                                        accum
                                    }
                                })
                                - λ * cell_volume
                        });
                // if part < -57983890000.0 {
                //     dbg!(part, cell_volume, cs.sum());
                //     dbg!(λ);
                //     panic!();
                // }
                accum + part
            });

        // nuclear reassignment terms
        ll += Zip::from(&self.cell_assignments)
            .and(&self.init_nuclear_cell_assignment)
            .fold(0_f32, |accum, &cell, &nuc_cell| {
                if nuc_cell != BACKGROUND_CELL {
                    if cell == nuc_cell {
                        accum + priors.nuclear_reassignment_1mlog_prob
                    } else {
                        accum + priors.nuclear_reassignment_log_prob
                    }
                } else {
                    accum
                }
            });

        // cell volume terms
        ll += Zip::from(&self.cell_volume)
            .and(&self.z)
            .fold(0_f32, |accum, &v, &z| {
                accum + lognormal_logpdf(self.μ_volume[z as usize], self.σ_volume[z as usize], v)
            });

        // background terms
        ll += Zip::from(&self.background_counts)
            .and(&self.λ_bg)
            .fold(0_f32, |accum, &c, &λ_bg| {
                if c > 0 {
                    accum + (c as f32) * λ_bg.ln() - λ_bg * self.full_layer_volume
                } else {
                    accum - λ_bg * self.full_layer_volume
                }
            });

        return ll;
    }

    pub fn write_cell_hulls(
        &self,
        transcripts: &Vec<Transcript>,
        counts: &Array2<u32>,
        filename: &str,
    ) {
        // We are not maintaining any kind of per-cell array, so I guess I have
        // no choice but to compute such a thing here.
        let mut cell_transcripts: Vec<Vec<usize>> = vec![Vec::new(); self.ncells()];
        for (i, &cell) in self.cell_assignments.iter().enumerate() {
            if cell != BACKGROUND_CELL {
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

        for (i, js) in cell_transcripts.iter().enumerate() {
            vertices.clear();
            for j in js {
                let transcript = transcripts[*j];
                vertices.push((transcript.x, transcript.y));
            }

            let area = convex_hull_area(&mut vertices, &mut hull);
            let count = counts.column(i).sum();

            writeln!(
                encoder,
                concat!(
                    "    {{\n",
                    "      \"type\": \"Feature\",\n",
                    "      \"properties\": {{\n",
                    "        \"cell\": {},\n",
                    "        \"area\": {},\n",
                    "        \"count\": {}\n",
                    "      }},\n",
                    "      \"geometry\": {{\n",
                    "        \"type\": \"Polygon\",\n",
                    "        \"coordinates\": [",
                    "          ["
                ),
                i, area, count
            )
            .unwrap();
            for (i, (x, y)) in hull.iter().enumerate() {
                writeln!(encoder, "            [{}, {}]", x, y).unwrap();
                if i < hull.len() - 1 {
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
    pub fn new() -> Self {
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

pub struct UncertaintyTracker {
    cell_assignment_duration: HashMap<(usize, CellIndex), u32>,
}

impl UncertaintyTracker {
    pub fn new() -> UncertaintyTracker {
        let cell_assignment_duration = HashMap::new();

        return UncertaintyTracker {
            cell_assignment_duration,
        };
    }

    // record the duration of the current cell assignment. Called when the state
    // is about to change.
    fn update(&mut self, params: &ModelParams, i: usize) {
        let duration = params.t - params.cell_assignment_time[i];
        self.cell_assignment_duration
            .entry((i, params.cell_assignments[i]))
            .and_modify(|d| *d += duration)
            .or_insert(duration);
    }

    pub fn finish(&mut self, params: &ModelParams) {
        for ((i, &j), &t) in params
            .cell_assignments
            .iter()
            .enumerate()
            .zip(&params.cell_assignment_time)
        {
            let duration = params.t - t + 1;
            self.cell_assignment_duration
                .entry((i, j))
                .and_modify(|d| *d += duration)
                .or_insert(duration);
        }
    }

    fn max_posterior_cell_assignments(&self, params: &ModelParams) -> Vec<(u32, f32)> {
        // sort ascending on (transcript, cell)
        let sorted_durations: Vec<(usize, u32, u32)> = self
            .cell_assignment_duration
            .iter()
            .map(|((i, j), d)| (*i, *j, *d))
            .sorted_by(|(i_a, j_a, _), (i_b, j_b, _)| (*i_a, *j_a).cmp(&(*i_b, *j_b)))
            .collect();

        let mut summed_durations: Vec<(usize, u32, u32)> = Vec::new();
        let mut ij_prev = (usize::MAX, u32::MAX);
        for (i, j, d) in sorted_durations.iter().cloned() {
            if (i, j) == ij_prev {
                summed_durations.last_mut().unwrap().2 += d;
            } else if i == ij_prev.0 || (i == ij_prev.0 + 1) {
                summed_durations.push((i, j, d));
                ij_prev = (i, j);
            } else {
                panic!("Missing transcript in cell assignments.");
            }
        }

        // sort ascending on (transcript, cell) and descending on duration
        summed_durations.sort_by(|(i_a, j_a, d_a), (i_b, j_b, d_b)| {
            (*i_a, *j_a, *d_b).cmp(&(*i_b, *j_b, *d_a))
        });

        let mut maxpost_cell_assignments = Vec::new();
        let mut i_prev = usize::MAX;
        for (i, j, d) in summed_durations.iter().cloned() {
            if i == i_prev {
                continue;
            } else if i == i_prev + 1 {
                maxpost_cell_assignments.push((j, d as f32 / params.t as f32));
                i_prev = i;
            } else {
                panic!("Missing transcript in cell assignments.");
            }
        }

        return maxpost_cell_assignments;
    }

    pub fn max_posterior_transcript_counts_assignments(
        &self,
        params: &ModelParams,
        transcripts: &Vec<Transcript>,
        count_pr_cutoff: f32,
        foreground_pr_cutoff: f32,
    ) -> (Array2<u32>, Vec<(u32, f32)>) {
        let mut counts = Array2::<u32>::from_elem((params.ngenes(), params.ncells()), 0_u32);
        let maxpost_assignments = self.max_posterior_cell_assignments(&params);
        for (i, (j, pr)) in maxpost_assignments.iter().enumerate() {
            if *pr > count_pr_cutoff && *j != BACKGROUND_CELL {
                let gene = transcripts[i].gene;
                let layer = params.zlayer(params.transcript_positions[i].2);
                let density = params.transcript_density[i];

                let λ_fg = params.λ[[gene as usize, *j as usize]];
                let λ_bg = params.λ_bg[[gene as usize, layer]];
                let fg_pr = λ_fg / (λ_fg + density * λ_bg);

                if fg_pr > foreground_pr_cutoff {
                    counts[[gene as usize, *j as usize]] += 1;
                }
            }
        }

        return (counts, maxpost_assignments);
    }

    pub fn expected_counts(
        &self,
        params: &ModelParams,
        transcripts: &Vec<Transcript>,
    ) -> Array2<f32> {
        let mut ecounts = Array2::<f32>::zeros((params.ngenes(), params.ncells()));

        for (&(i, j), &d) in self.cell_assignment_duration.iter() {
            if j == BACKGROUND_CELL {
                continue;
            }

            let gene = transcripts[i].gene;
            let layer = params.zlayer(params.transcript_positions[i].2);
            let density = params.transcript_density[i];

            let w_d = d as f32 / params.t as f32;

            let λ_fg = params.λ[[gene as usize, j as usize]];
            let λ_bg = params.λ_bg[[gene as usize, layer]];
            let w_bg = λ_fg / (λ_fg + density * λ_bg);

            ecounts[[gene as usize, j as usize]] += w_d * w_bg;
        }

        return ecounts;
    }
}

pub trait Proposal {
    fn accept(&mut self);
    fn reject(&mut self);

    fn ignored(&self) -> bool;
    fn accepted(&self) -> bool;

    // Return updated cell size minus current cell size `old_cell`
    fn old_cell_volume_delta(&self) -> f32;

    // Return updated cell size minus current cell size `new_cell`
    fn new_cell_volume_delta(&self) -> f32;

    fn old_cell(&self) -> u32;
    fn new_cell(&self) -> u32;

    fn log_weight(&self) -> f32;

    fn transcripts<'b, 'c>(&'b self) -> &'c [usize]
    where
        'b: 'c;

    // Iterator over number of transcripts in the proposal of each gene
    // Returns a [ngenes, nlayers]
    fn gene_count<'b, 'c>(&'b self) -> &'c Array2<u32>
    where
        'b: 'c;

    fn density<'b, 'c>(&'b self) -> &'c Array1<f32>
    where
        'b: 'c;

    fn evaluate(&mut self, priors: &ModelPriors, params: &ModelParams) {
        if self.ignored() {
            self.reject();
            return;
        }

        let old_cell = self.old_cell();
        let new_cell = self.new_cell();
        let from_background = old_cell == BACKGROUND_CELL;
        let to_background = new_cell == BACKGROUND_CELL;

        // Log Metropolis-Hastings acceptance ratio
        let mut δ = 0.0;

        // Tally penalties from mis-assigning nuclear transcripts
        for &t in self.transcripts() {
            let cell = params.init_nuclear_cell_assignment[t];
            if cell != BACKGROUND_CELL {
                if cell == old_cell {
                    δ -= priors.nuclear_reassignment_1mlog_prob;
                } else {
                    δ -= priors.nuclear_reassignment_log_prob;
                }

                if cell == new_cell {
                    δ += priors.nuclear_reassignment_1mlog_prob;
                } else {
                    δ += priors.nuclear_reassignment_log_prob;
                }
            }
        }

        if from_background {
            Zip::from(self.gene_count().rows())
                .and(self.density())
                .and(params.λ_bg.rows())
                .for_each(|gene_counts, &d, λ_bg| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        δ -= count as f32 * (d * λ_bg).ln();
                    });
                });
        } else {
            let volume_diff = self.old_cell_volume_delta();

            let prev_volume = params.cell_volume[old_cell as usize];
            let new_volume = prev_volume + volume_diff;

            // normalization term difference
            δ += Zip::from(params.λ.column(old_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * volume_diff);

            Zip::from(self.gene_count().rows())
                .and(self.density())
                .and(params.λ_bg.rows())
                .and(params.λ.column(old_cell as usize))
                .for_each(|gene_counts, &d, λ_bg, λ| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        if count > 0 {
                            δ -= count as f32 * (d * λ_bg + λ).ln();
                        }
                    })
                });

            let z = params.z[old_cell as usize];
            δ -= lognormal_logpdf(
                params.μ_volume[z as usize],
                params.σ_volume[z as usize],
                prev_volume,
            );
            δ += lognormal_logpdf(
                params.μ_volume[z as usize],
                params.σ_volume[z as usize],
                new_volume,
            );
        }

        if to_background {
            Zip::from(self.gene_count().rows())
                .and(self.density())
                .and(params.λ_bg.rows())
                .for_each(|gene_counts, &d, λ_bg| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        δ += count as f32 * (d * λ_bg).ln();
                    });
                });
        } else {
            let volume_diff = self.new_cell_volume_delta();

            let prev_volume = params.cell_volume[new_cell as usize];
            let new_volume = prev_volume + volume_diff;

            // normalization term difference
            δ += Zip::from(params.λ.column(new_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * volume_diff);

            // add in new cell likelihood terms
            Zip::from(self.gene_count().rows())
                .and(self.density())
                .and(params.λ_bg.rows())
                .and(params.λ.column(new_cell as usize))
                .for_each(|gene_counts, &d, λ_bg, λ| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        if count > 0 {
                            δ += count as f32 * (d * λ_bg + λ).ln();
                        }
                    })
                });

            let z = params.z[new_cell as usize];
            δ -= lognormal_logpdf(
                params.μ_volume[z as usize],
                params.σ_volume[z as usize],
                prev_volume,
            );
            δ += lognormal_logpdf(
                params.μ_volume[z as usize],
                params.σ_volume[z as usize],
                new_volume,
            );
        }

        let mut rng = thread_rng();
        let logu = rng.gen::<f32>().ln();

        if logu < δ + self.log_weight() {
            self.accept();
        } else {
            self.reject();
        }
    }
}

pub trait Sampler<P>
where
    P: Proposal + Send,
    Self: Sync
{
    // fn generate_proposals<'b, 'c>(&'b mut self, params: &ModelParams) -> &'c mut [P] where 'b: 'c;

    fn repopulate_proposals(&mut self, priors: &ModelPriors, params: &ModelParams);
    fn proposals<'a, 'b>(&'a self) -> &'b [P]
    where
        'a: 'b;
    fn proposals_mut<'a, 'b>(&'a mut self) -> &'b mut [P]
    where
        'a: 'b;

    // Called by `apply_accepted_proposals` to handle any sampler specific
    // updates needed after applying accepted proposals. This is mainly
    // updating mismatch edges.
    fn update_sampler_state(&mut self, params: &ModelParams);

    // fn update_transcript_position(&mut self, i: usize, prev_pos: (f32, f32, f32), new_pos: (f32, f32, f32));
    // fn update_transcript_positions(&mut self, accept: &Vec<bool>, positions: &Vec<(f32, f32, f32)>, proposed_positions: &Vec<(f32, f32, f32)>);
    fn update_transcript_positions(&mut self, updated: &Vec<bool>, positions: &Vec<(f32, f32, f32)>);

    fn cell_at_position(&self, pos: (f32, f32, f32)) -> u32;

    fn sample_cell_regions(
        &mut self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        stats: &mut ProposalStats,
        transcripts: &Vec<Transcript>,
        uncertainty: &mut Option<&mut UncertaintyTracker>,
    ) {
        // don't count time unless we are tracking uncertainty
        if uncertainty.is_some() {
            params.t += 1;
        }
        self.repopulate_proposals(priors, params);
        self.proposals_mut()
            .par_iter_mut()
            .for_each(|p| p.evaluate(priors, params));
        self.apply_accepted_proposals(stats, transcripts, priors, params, uncertainty);
    }

    fn apply_accepted_proposals(
        &mut self,
        stats: &mut ProposalStats,
        transcripts: &Vec<Transcript>,
        priors: &ModelPriors,
        params: &mut ModelParams,
        uncertainty: &mut Option<&mut UncertaintyTracker>,
    ) {
        // Update cell assignments
        for proposal in self
            .proposals()
            .iter()
            .filter(|p| p.accepted() && !p.ignored())
        {
            let old_cell = proposal.old_cell();
            let new_cell = proposal.new_cell();

            let mut count = 0;
            for &i in proposal.transcripts() {
                if let Some(uncertainty) = uncertainty.as_mut() {
                    uncertainty.update(params, i);
                }
                params.cell_assignments[i] = new_cell;
                params.cell_assignment_time[i] = params.t;
                count += 1;
            }

            // Update count matrix and areas
            if old_cell != BACKGROUND_CELL {
                params.cell_population[old_cell as usize] -= count;

                let mut cell_volume = params.cell_volume[old_cell as usize];
                cell_volume += proposal.old_cell_volume_delta();
                cell_volume = cell_volume.max(priors.min_cell_volume);
                params.cell_volume[old_cell as usize] = cell_volume;

                for &i in proposal.transcripts() {
                    let gene = transcripts[i].gene;
                    let layer = ((params.transcript_positions[i].2 - params.z0) / params.layer_depth).max(0.0) as usize;
                    let layer = layer.min(params.nlayers() - 1);
                    params.counts[[gene as usize, old_cell as usize, layer]] -= 1;
                }
            }

            if new_cell != BACKGROUND_CELL {
                params.cell_population[new_cell as usize] += count;

                let mut cell_volume = params.cell_volume[new_cell as usize];
                cell_volume += proposal.new_cell_volume_delta();
                cell_volume = cell_volume.max(priors.min_cell_volume);
                params.cell_volume[new_cell as usize] = cell_volume;

                for &i in proposal.transcripts() {
                    let gene = transcripts[i].gene;
                    let layer = ((params.transcript_positions[i].2 - params.z0) / params.layer_depth).max(0.0) as usize;
                    let layer = layer.min(params.nlayers() - 1);
                    params.counts[[gene as usize, new_cell as usize, layer]] += 1;
                }
            }

            // Keep track of stats
            if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.background_to_cell_accept += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                stats.cell_to_background_accept += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.cell_to_cell_accept += 1;
            }
        }

        for proposal in self
            .proposals()
            .iter()
            .filter(|p| !p.accepted() && !p.ignored())
        {
            let old_cell = proposal.old_cell();
            let new_cell = proposal.new_cell();

            if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.background_to_cell_reject += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                stats.cell_to_background_reject += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.cell_to_cell_reject += 1;
            }
        }

        self.update_sampler_state(params);
    }

    fn sample_global_params(
        &mut self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
    ) {
        let mut rng = thread_rng();

        self.sample_volume_params(priors, params);

        // Sample background/foreground counts
        // let t0 = Instant::now();
        self.sample_background_counts(priors, params, transcripts);
        // println!("  Sample background counts: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_component_nb_params(priors, params);
        // println!("  Sample nb params: {:?}", t0.elapsed());

        // Sample λ
        // let t0 = Instant::now();
        self.sample_rates(priors, params);
        // println!("  Sample λ: {:?}", t0.elapsed());

        // TODO:
        // This is the most expensive part. We could sample this less frequently,
        // but we should try to optimize as much as possible.
        // Ideas:
        //   - Main bottlneck is computing log(p) and log(1-p). I don't see
        //     anything obvious to do about that.

        // Sample z
        // let t0 = Instant::now();
        self.sample_component_assignments(priors, params);
        // println!("  Sample z: {:?}", t0.elapsed());

        // sample π
        let mut α = vec![1_f32; params.ncomponents()];
        for z_i in params.z.iter() {
            α[*z_i as usize] += 1.0;
        }

        if α.len() == 1 {
            params.π.clear();
            params.π.push(1.0);
        } else {
            params.π.clear();
            params.π.extend(
                Dirichlet::new(&α)
                    .unwrap()
                    .sample(&mut rng)
                    .iter()
                    .map(|x| *x as f32),
            );
        }

        self.sample_background_rates(priors, params);
        if params.t > 0 {
            self.sample_transcript_positions(priors, params, transcripts);
        }
    }

    fn sample_background_counts(
        &mut self,
        _priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
    ) {
        let nlayers = params.nlayers();
        Zip::indexed(&mut params.isbackground)
            .and(&params.transcript_density)
            .and(transcripts)
            .par_for_each(|i, bg, &d, t| {
                let mut rng = thread_rng();
                let cell = params.cell_assignments[i];

                if cell == BACKGROUND_CELL {
                    *bg = true;
                } else {
                    let gene = t.gene as usize;
                    let layer = ((params.transcript_positions[i].2 - params.z0) / params.layer_depth).max(0.0) as usize;
                    let layer = layer.min(nlayers - 1);

                    let λ = params.λ[[gene, cell as usize]];
                    let p = λ / (λ + d * params.λ_bg[[gene, layer]]);
                    // if p < 0.5 {
                    //     dbg!(p, λ, d, params.λ_bg[[gene, layer]]);
                    // }

                    // if λ != 0.1 && p > 0.99 {
                    //     dbg!(λ, d, params.λ_bg[[gene, layer]]);
                    // }

                    *bg = rng.gen::<f32>() > p;
                }
            });

        params.background_counts.fill(0_u32);
        params.foreground_counts.fill(0_u32);
        Zip::from(&params.isbackground)
            .and(transcripts)
            .and(&params.cell_assignments)
            .and(&params.transcript_positions)
            .for_each(|&bg, t, &cell, pos| {
                let gene = t.gene as usize;
                // let layer = params.zlayer(pos.2);
                let layer = ((pos.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                let layer = layer.min(nlayers - 1);

                if bg {
                    params.background_counts[[gene, layer]] += 1;
                } else {
                    params.foreground_counts[[cell as usize, gene, layer]] += 1;
                }
            });
    }

    fn sample_component_nb_params(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // total component area
        // let mut component_cell_area = vec![0_f32; params.ncomponents()];
        params.component_volume.fill(0.0);
        params
            .cell_volume
            .iter()
            .zip(&params.z)
            .for_each(|(volume, z_i)| {
                params.component_volume[*z_i as usize] += *volume;
            });

        // compute per component transcript counts
        params.component_counts.fill(0);
        Zip::from(params.component_counts.rows_mut())
            .and(params.foreground_counts.axis_iter(Axis(1)))
            .par_for_each(|mut compc, cellc| {
                for (cs, component) in cellc.outer_iter().zip(&params.z) {
                    let c = cs.sum();
                    compc[*component as usize] += c;
                }
            });

        // Sample θ
        // let t0 = Instant::now();
        Zip::from(params.θ.columns_mut())
            .and(params.r.columns())
            .and(params.component_counts.rows())
            .par_for_each(|θs, rs, cs| {
                let mut rng = thread_rng();
                for (θ, &c, &r, &a) in izip!(θs, cs, rs, &params.component_volume) {
                    *θ = prob_to_odds(
                        Beta::new(priors.α_θ + c as f32, priors.β_θ + a * r)
                            .unwrap()
                            .sample(&mut rng),
                    );

                    // *θ = θ.max(1e-6).min(1e6);
                }
            });
        // println!("  Sample θ: {:?}", t0.elapsed());

        // dbg!(
        //     params.θ.iter().min_by(|a, b| a.partial_cmp(b).unwrap()),
        //     params.θ.iter().max_by(|a, b| a.partial_cmp(b).unwrap()),
        // );

        // dbg!(params.h);

        // Sample r
        // let t0 = Instant::now();

        // TODO: Need some argument to determine dispersion sampling
        // behavior.

        fn set_constant_dispersion(params: &mut ModelParams, dispersion: f32) {
            params.r.fill(dispersion);
            Zip::from(&params.r)
                .and(&mut params.lgamma_r)
                .and(&mut params.loggammaplus)
                .par_for_each(|r, lgamma_r, loggammaplus| {
                    *lgamma_r = lgammaf(*r);
                    loggammaplus.reset(*r);
                });
        }

        if let Some(dispersion) = priors.dispersion {
            set_constant_dispersion(params, dispersion);
        } else if params.t == 0 && priors.burnin_dispersion.is_some() {
            let dispersion = priors.burnin_dispersion.unwrap();
            set_constant_dispersion(params, dispersion);
        } else {
            // for each gene
            params.uv.fill((0_u32, 0_f32));
            Zip::from(params.r.columns_mut())
                .and(params.lgamma_r.columns_mut())
                .and(params.loggammaplus.columns_mut())
                .and(params.θ.columns())
                .and(params.foreground_counts.axis_iter(Axis(1)))
                .and(params.uv.columns_mut())
                .par_for_each(|
                    rs,
                    lgamma_rs,
                    loggammaplus,
                    θs,
                    cs,
                    mut uv|
                {
                    let mut rng = thread_rng();

                    // iterate over cells computing u and v
                    Zip::from(&params.z)
                        .and(cs.axis_iter(Axis(0)))
                        .and(&params.cell_volume)
                        .for_each(|&z, c, &vol| {
                            let z = z as usize;
                            let c = c.sum();
                            let r = rs[z];
                            let θ = θs[z];

                            let uv_z = uv[z];
                            uv[z] = (
                                uv_z.0 + rand_crt(&mut rng, c, r),
                                uv_z.1 + log1pf(-odds_to_prob(θ * vol))
                            );

                            // if uv[z].1.is_infinite() {
                            //     dbg!(uv[z], θ, vol, c, r);
                            // }
                        });

                    // iterate over components sampling r
                    Zip::from(rs)
                        .and(lgamma_rs)
                        .and(loggammaplus)
                        .and(uv)
                        .for_each(|
                            r,
                            lgamma_r,
                            loggammaplus,
                            uv |
                        {
                            let dist = Gamma::new(priors.e_r + uv.0 as f32, (params.h - uv.1).recip());
                            if dist.is_err() {
                                dbg!(uv.0, uv.1, params.h);
                            }
                            let dist = dist.unwrap();
                            *r = dist.sample(&mut rng);

                            // if *r < 0.001 {
                            //     dbg!(uv.0, uv.1, params.h, *r);
                            // }

                            // *r = Gamma::new(priors.e_r + uv.0 as f32, (params.h - uv.1).recip())
                            //     .unwrap()
                            //     .sample(&mut rng);

                            assert!(r.is_finite());

                            // *r = r.min(200.0).max(1e-5);

                            *lgamma_r = lgammaf(*r);
                            loggammaplus.reset(*r);
                        });

                    // // self.cell_areas.slice(0..self.ncells)

                    // let u = Zip::from(counts.axis_iter(Axis(0))).fold(0, |accum, cs| {
                    //     let c = cs.sum();
                    //     accum + rand_crt(&mut rng, c, *r)
                    // });

                    // let v =
                    //     Zip::from(&params.z)
                    //         .and(&params.cell_volume)
                    //         .fold(0.0, |accum, z, a| {
                    //             let w = θs[*z as usize];
                    //             accum + log1pf(-odds_to_prob(w * *a))
                    //         });

                    // // dbg!(u, v);

                    // *r = Gamma::new(priors.e_r + u as f32, (params.h - v).recip())
                    //     .unwrap()
                    //     .sample(&mut rng);

                    // assert!(r.is_finite());

                    // *r = r.min(200.0).max(1e-5);

                    // *lgamma_r = lgammaf(*r);
                    // loggammaplus.reset(*r);
                });
        }

        // params.h = 0.1;
        params.h = Gamma::new(
            priors.e_h * (1_f32 + params.r.len() as f32),
            (priors.f_h + params.r.sum()).recip(),
        )
        .unwrap()
        .sample(&mut thread_rng());
    }

    fn sample_rates(&mut self, _priors: &ModelPriors, params: &mut ModelParams) {
        // loop over genes
        Zip::from(params.λ.rows_mut())
            .and(params.foreground_counts.axis_iter(Axis(1)))
            .and(params.θ.columns())
            .and(params.r.columns())
            .par_for_each(|mut λs, cs, θs, rs| {
                let mut rng = thread_rng();
                // loop over cells
                for (λ, &z, cs, cell_volume) in
                    izip!(&mut λs, &params.z, cs.outer_iter(), &params.cell_volume)
                {
                    let z = z as usize;

                    let c = cs.sum();
                    let θ = θs[z];
                    let r = rs[z];
                    *λ = Gamma::new(
                        r + c as f32,
                        θ / (cell_volume * θ + 1.0), // ((cell_area * θ + 1.0) / θ) as f64,
                    )
                    .unwrap()
                    .sample(&mut rng);
                    // .max(1e-14);

                    assert!(λ.is_finite());
                }
            });
    }

    fn sample_background_rates(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // TODO: How does this work now? What exactly are we adding to β?
        //
        // I think the right thing is to take the full_layer_volume multiplied by the total density.
        let mut rng = thread_rng();

        Zip::from(params.λ_bg.rows_mut())
            .and(params.background_counts.rows())
            .for_each(|λs, cs| {
                Zip::from(λs).and(cs).for_each(|λ, c| {
                    let α = priors.α_bg + *c as f32;
                    let β = priors.β_bg + params.full_layer_volume;
                    // let β = priors.β_bg + total_density * params.full_layer_volume;
                    // let β = priors.β_bg + total_density;
                    *λ = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) as f32;
                });
            });

        // dbg!(&params.total_transcript_density);
        // dbg!(params.full_layer_volume);
        // dbg!(&params.λ_bg);

        // Zip::from(&mut params.λ_bg)
        //     .and(&params.background_counts)
        //     .for_each(|λ, c| {
        //         let α = priors.α_bg + *c as f32;
        //         let β = priors.β_bg + params.full_layer_volume;
        //         *λ = Gamma::new(α, β.recip())
        //             .unwrap()
        //             .sample(&mut rng) as f32;
        //     });
    }

    fn sample_component_assignments(&mut self, _priors: &ModelPriors, params: &mut ModelParams) {
        let ncomponents = params.ncomponents();

        // loop over cells
        Zip::from(params.foreground_counts.axis_iter(Axis(0)))
            .and(&mut params.z)
            .and(&params.cell_volume)
            .par_for_each(|cs, z_i, cell_volume| {
                let mut z_probs = params
                    .z_probs
                    .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                    .borrow_mut();

                // loop over components
                for (zp, π, θs, rs, lgamma_r, loggammaplus, &μ_volume, &σ_volume) in izip!(
                    z_probs.iter_mut(),
                    &params.π,
                    params.θ.rows(),
                    params.r.rows(),
                    params.lgamma_r.rows(),
                    params.loggammaplus.rows(),
                    &params.μ_volume,
                    &params.σ_volume
                ) {
                    // sum over genes
                    *zp = (*π as f64)
                        * (Zip::from(cs.axis_iter(Axis(0)))
                            .and(rs)
                            .and(θs)
                            .and(lgamma_r)
                            .and(loggammaplus)
                            .fold(0_f32, |accum, cs, &r, θ, &lgamma_r, lgammaplus| {
                                let c = cs.sum(); // sum counts across layers
                                accum
                                    + negbin_logpmf_fast(
                                        r,
                                        lgamma_r,
                                        lgammaplus.eval(c),
                                        odds_to_prob(*θ * cell_volume),
                                        c,
                                        params.logfactorial.eval(c),
                                    )
                            }) as f64)
                            .exp();

                    *zp *= lognormal_logpdf(μ_volume, σ_volume, *cell_volume).exp() as f64;
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
    }

    fn sample_volume_params(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // compute sample means
        params.component_population.fill(0_u32);
        params.μ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_volume)
            .for_each(|&z, &volume| {
                params.μ_volume[z as usize] += volume.ln();
                params.component_population[z as usize] += 1;
            });

        // sample μ parameters
        Zip::from(&mut params.μ_volume)
            .and(&params.σ_volume)
            .and(&params.component_population)
            .par_for_each(|μ, &σ, &pop| {
                let mut rng = thread_rng();

                let v = (1_f32 / priors.σ_μ_volume.powi(2) + pop as f32 / σ.powi(2)).recip();
                *μ = Normal::new(
                    v * (priors.μ_μ_volume / priors.σ_μ_volume.powi(2) + *μ / σ.powi(2)),
                    v.sqrt(),
                )
                .unwrap()
                .sample(&mut rng);
            });

        // compute sample variances
        params.σ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_volume)
            .for_each(|&z, &volume| {
                params.σ_volume[z as usize] += (params.μ_volume[z as usize] - volume.ln()).powi(2);
            });

        // sample σ parameters
        Zip::from(&mut params.σ_volume)
            .and(&params.component_population)
            .par_for_each(|σ, &pop| {
                let mut rng = thread_rng();
                *σ = Gamma::new(
                    priors.α_σ_volume + (pop as f32) / 2.0,
                    (priors.β_σ_volume + *σ / 2.0).recip(),
                )
                .unwrap()
                .sample(&mut rng)
                .recip()
                .sqrt();
            });
    }


    fn propose_eval_transcript_positions(&mut self, priors: &ModelPriors, params: &mut ModelParams, transcripts: &Vec<Transcript>)
    {
        // make proposals
        // let t0 = Instant::now();
        params.proposed_transcript_positions
            .par_iter_mut()
            .zip(&params.transcript_positions)
            .for_each(|(proposed_position, current_position)| {
                let mut rng = thread_rng();
                *proposed_position = (
                    current_position.0 + priors.σ_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal),
                    current_position.1 + priors.σ_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal),
                    (current_position.2 + priors.σ_z_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal)).min(priors.zmax).max(priors.zmin),
                );
            });
        // println!("  Generate transcript position proposals: {:?}", t0.elapsed());

        // accept/reject proposals
        // let t0 = Instant::now();
        params.accept_proposed_transcript_positions
            .par_iter_mut()
            .zip(&params.transcript_positions)
            .zip(&params.proposed_transcript_positions)
            .zip(transcripts)
            .enumerate()
            .for_each(|(i, (((accept, position), proposed_position), transcript))| {
                // Reject out of bounds proposals, to avoid detailed balance
                // issues.
                if proposed_position.2 == priors.zmin || proposed_position.2 == priors.zmax {
                    *accept = false;
                    return;
                }

                // Reject nops
                if proposed_position == position {
                    *accept = false;
                    return;
                }

                let sq_dist_new =
                    (proposed_position.0 - transcript.x).powi(2) +
                    (proposed_position.1 - transcript.y).powi(2);
                let sq_dist_prev =
                    (position.0 - transcript.x).powi(2) +
                    (position.1 - transcript.y).powi(2);
                let z_sq_dist_new = (proposed_position.2 - transcript.z).powi(2);
                let z_sq_dist_prev = (position.2 - transcript.z).powi(2);

                let mut δ = 0.0;

                δ -= ((1.0 - priors.p_diffusion) * normal_pdf(priors.σ_diffusion_near, sq_dist_prev) +
                    priors.p_diffusion * normal_pdf(priors.σ_diffusion_far, sq_dist_prev)).ln();
                δ += ((1.0 - priors.p_diffusion) * normal_pdf(priors.σ_diffusion_near, sq_dist_new) +
                    priors.p_diffusion * normal_pdf(priors.σ_diffusion_far, sq_dist_new)).ln();

                δ -= -0.5 * (z_sq_dist_prev / priors.σ_z_diffusion.powi(2));
                δ += -0.5 * (z_sq_dist_new / priors.σ_z_diffusion.powi(2));

                let density = params.transcript_density[i];
                let gene = transcript.gene as usize;

                let layer_prev = ((position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                let layer_prev = layer_prev.min(params.λ_bg.ncols() - 1);
                let cell_prev = self.cell_at_position(*position);
                let λ_prev = if cell_prev == BACKGROUND_CELL {
                    0.0
                } else {
                    params.λ[[gene, cell_prev as usize]]
                } + density * params.λ_bg[[gene, layer_prev]];

                let layer_new = ((proposed_position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                let layer_new = layer_new.min(params.λ_bg.ncols() - 1);
                let cell_new = self.cell_at_position(*proposed_position);
                let λ_new = if cell_new == BACKGROUND_CELL {
                    0.0
                } else {
                    params.λ[[gene, cell_new as usize]]
                } + density * params.λ_bg[[gene, layer_new]];

                let ln_λ_diff = λ_new.ln() - λ_prev.ln();
                δ += ln_λ_diff;

                let cell_nuc = params.init_nuclear_cell_assignment[i];
                if cell_nuc != BACKGROUND_CELL {
                    if cell_nuc == cell_prev {
                        δ -= priors.nuclear_reassignment_1mlog_prob;
                    } else {
                        δ -= priors.nuclear_reassignment_log_prob;
                    }

                    if cell_nuc == cell_new {
                        δ += priors.nuclear_reassignment_1mlog_prob;
                    } else {
                        δ += priors.nuclear_reassignment_log_prob;
                    }
                }

                let mut rng = thread_rng();
                let logu = rng.gen::<f32>().ln();
                *accept = logu < δ;
            });
        // println!("  Eval transcript position proposals: {:?}", t0.elapsed());
    }

    fn sample_transcript_positions(&mut self, priors: &ModelPriors, params: &mut ModelParams, transcripts: &Vec<Transcript>)
    {
        self.propose_eval_transcript_positions(priors, params, transcripts);

        // Update position and compute cell and layer changes for updates
        params.transcript_position_updates
            .par_iter_mut()
            .zip(&mut params.transcript_positions)
            .zip(&params.proposed_transcript_positions)
            .zip(&mut params.cell_assignments)
            .zip(&params.accept_proposed_transcript_positions)
            .for_each(|((((update, position), proposed_position), cell_prev), &accept)| {
                if accept {
                    let layer_prev = ((position.2 - params.z0) / params.layer_depth) as usize;
                    let layer_prev = layer_prev.min(params.λ_bg.ncols() - 1);

                    let cell_new = self.cell_at_position(*proposed_position);
                    let layer_new = ((proposed_position.2 - params.z0) / params.layer_depth) as usize;
                    let layer_new = layer_new.min(params.λ_bg.ncols() - 1);

                    // assert!(self.cell_at_position(*position) == *cell_prev);

                    *update = (*cell_prev, cell_new, layer_prev as u32, layer_new as u32);
                    *position = *proposed_position;
                    *cell_prev = cell_new;
                }
            });

        // Update counts and cell_population
        params.transcript_position_updates
            .iter()
            .zip(transcripts)
            .zip(&params.accept_proposed_transcript_positions)
            .for_each(|((update, transcript), &accept)| {
                let (cell_prev, cell_new, layer_prev, layer_new) = *update;
                if accept {
                    let gene = transcript.gene as usize;
                    if cell_prev != BACKGROUND_CELL {
                        assert!(params.counts[[gene, cell_prev as usize, layer_prev as usize]] > 0);
                        params.counts[[gene, cell_prev as usize, layer_prev as usize]] -= 1;
                        assert!(params.cell_population[cell_prev as usize] > 0);
                        params.cell_population[cell_prev as usize] -= 1;
                    }
                    if cell_new != BACKGROUND_CELL {
                        params.counts[[gene, cell_new as usize, layer_new as usize]] += 1;
                        params.cell_population[cell_new as usize] += 1;
                    }
                }
            });

        self.update_transcript_positions(&params.accept_proposed_transcript_positions, &params.transcript_positions);
    }
}
