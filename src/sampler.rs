mod connectivity;
pub mod cubebinsampler;
pub mod hull;
mod math;
mod sampleset;
mod polygons;
pub mod transcripts;
pub mod polyagamma;

use core::fmt::Debug;
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::{izip, Itertools};
use libm::{lgammaf, log1pf};
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use linfa_clustering::GaussianMixtureModel;
use math::{
    logistic, normal_pdf, normal_x2_pdf, normal_x2_logpdf,
    lognormal_logpdf, negbin_logpmf_fast, rand_crt, LogFactorial,
    LogGammaPlus,
};
use polyagamma::PolyaGamma;
use ndarray::{Array1, Array2, Array3, Axis, Zip};
use rand::{thread_rng, Rng};
use rand_distr::{Dirichlet, Distribution, Gamma, Normal, StandardNormal};
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

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum TranscriptState {
    Background,
    Foreground,
    Confusion,
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

    // normal precision parameter for β
    pub γ: f32,

    // gamma prior for background rates
    pub α_bg: f32,
    pub β_bg: f32,

    // gamma prior for confusion rates
    pub α_c: f32,
    pub β_c: f32,

    // scaling factor for circle perimeters
    pub perimeter_eta: f32,
    pub perimeter_bound: f32,

    pub nuclear_reassignment_log_prob: f32,
    pub nuclear_reassignment_1mlog_prob: f32,

    pub prior_seg_reassignment_log_prob: f32,
    pub prior_seg_reassignment_1mlog_prob: f32,

    // mixture between diffusion prior components
    pub use_diffusion_model: bool,
    pub p_diffusion: f32,
    pub σ_diffusion_proposal: f32,
    pub σ_diffusion_near: f32,
    pub σ_diffusion_far: f32,

    pub σ_z_diffusion_proposal: f32,
    pub σ_z_diffusion: f32,

    // bounds on z coordinate
    pub zmin: f32,
    pub zmax: f32,

    // whether to check if voxel updates break local connectivity
    pub enforce_connectivity: bool,
}

// Model global parameters.
pub struct ModelParams {
    pub transcript_positions: Vec<(f32, f32, f32)>,
    proposed_transcript_positions: Vec<(f32, f32, f32)>,
    accept_proposed_transcript_positions: Vec<bool>,
    transcript_position_updates: Vec<(u32, u32, u32, u32)>,

    init_nuclear_cell_assignment: Vec<CellIndex>,
    prior_seg_cell_assignment: Vec<CellIndex>,

    pub cell_assignments: Vec<CellIndex>,
    pub cell_assignment_time: Vec<u32>,

    pub cell_population: Vec<usize>,

    // per-cell volumes
    pub cell_volume: Array1<f32>,
    pub cell_log_volume: Array1<f32>,

    // per-component volumes
    pub component_volume: Array1<f32>,

    // area of the convex hull containing all transcripts
    full_layer_volume: f32,

    z0: f32,
    layer_depth: f32,

    // [ntranscripts] current assignment of transcripts to background
    pub transcript_state: Array1<TranscriptState>,
    pub prev_transcript_state: Array1<TranscriptState>,

    // [ngenes, ncells, nlayers] transcripts counts
    pub counts: Array3<u16>,

    // [ncells, ngenes, nlayers] foreground transcripts counts
    foreground_counts: Array3<u16>,

    // [ngenes] background transcripts counts
    confusion_counts: Array1<u32>,

    // [ngenes, nlayers] background transcripts counts
    background_counts: Array2<u32>,

    // [ngenes, nlayers] total gene occourance counts
    pub total_gene_counts: Array2<u32>,

    // Not parameters, but needed for sampling global params
    logfactorial: LogFactorial,

    // TODO: This needs to be an matrix I guess!
    loggammaplus: Array2<LogGammaPlus>,

    pub z: Array1<u32>, // assignment of cells to components

    component_population: Array1<u32>, // number of cells assigned to each component

    // thread-local space used for sampling z
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,

    π: Vec<f32>, // mixing proportions over components

    μ_volume: Array1<f32>, // volume dist mean param by component
    σ_volume: Array1<f32>, // volume dist std param by component

    // Prior on NB dispersion parameters
    h: f32,

    // [ncells, ngenes] Polya-gamma samples, used for sampling NB rates
    ω: Array2<f32>,

    // [ncomponents, ngenes] NB logit(p) parameters
    pub φ: Array2<f32>,

    // [ncomponents, ngenes] Parameters for sampling φ
    μ_φ: Array2<f32>,
    σ_φ: Array2<f32>,

    // [ncomponents, ngenes] NB r parameters.
    pub r: Array2<f32>,

    // [ncomponents, ngenes] gamama parameters for sampling r
    uv: Array2<(u32, f32)>,

    // Precomputing lgamma(r)
    lgamma_r: Array2<f32>,

    // // [ncomponents, ngenes] NB p parameters.
    // θ: Array2<f32>,

    // // log(ods_to_prob(θ))
    // logp: Array2<f32>,

    // // log(1 - ods_to_prob(θ))
    // log1mp: Array2<f32>,

    // [ngenes, ncells] Poisson rates
    pub λ: Array2<f32>,

    // [ngenes, nlayers] background rate: rate at which halucinate transcripts
    // across the entire layer
    pub λ_bg: Array2<f32>,

    // [ngenes] confusion: rate at which we halucinate transcripts within cells
    pub λ_c: Array1<f32>,

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
        init_cell_assignments: &Vec<u32>,
        init_cell_population: &Vec<usize>,
        prior_seg_cell_assignment: &Vec<u32>,
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
        let cell_log_volume = Array1::<f32>::zeros(ncells);
        let h = 10.0;

        // compute initial counts
        let mut counts = Array3::<u16>::from_elem((ngenes, ncells, nlayers), 0);
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
        let norm_constant = 1e4;
        let mut init_samples = counts.sum_axis(Axis(2)).map(|&x| (x as f32)).reversed_axes();
        init_samples.rows_mut()
            .into_iter()
            .for_each(|mut row| {
                let rowsum = row.sum();
                row.mapv_inplace(|x| (norm_constant * (x / rowsum)).ln_1p());
            });
        let init_samples = DatasetBase::from(init_samples);

        // log1p transformed counts
        // let init_samples =
        //     DatasetBase::from(counts.sum_axis(Axis(2)).map(|&x| (x as f32).ln_1p()).reversed_axes());

        let rng = rand::thread_rng();
        let model = KMeans::params_with_rng(ncomponents, rng)
            .tolerance(1e-1)
            .fit(&init_samples)
            .expect("kmeans failed to converge");

        let z = model.predict(&init_samples).map(|&x| x as u32);

        // let rng = rand::thread_rng();
        // let model = GaussianMixtureModel::params_with_rng(ncomponents, rng)
        //     .tolerance(1e-1)
        //     .fit(&init_samples)
        //     .expect("gmm failed to converge");

        // let z = model.predict(&init_samples).map(|&x| x as u32);

        // // initial component assignments
        // let mut rng = rand::thread_rng();
        // let z = (0..ncells)
        //     .map(|_| rng.gen_range(0..ncomponents) as u32)
        //     .collect::<Vec<_>>()
        //     .into();

        let uv = Array2::<(u32, f32)>::from_elem((ncomponents, ngenes), (0_u32, 0_f32));
        let lgamma_r = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);
        let loggammaplus = Array2::<LogGammaPlus>::from_elem((ncomponents, ngenes), LogGammaPlus::default());
        let ω = Array2::<f32>::from_elem((ncells, ngenes), 0.0);
        let φ = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);
        let μ_φ = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);
        let σ_φ = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);


        let component_volume = Array1::<f32>::from_elem(ncomponents, 0.0);
        let transcript_state = Array1::<TranscriptState>::from_elem(transcripts.len(), TranscriptState::Foreground);
        let prev_transcript_state = Array1::<TranscriptState>::from_elem(transcripts.len(), TranscriptState::Foreground);

        return ModelParams {
            transcript_positions,
            proposed_transcript_positions,
            accept_proposed_transcript_positions,
            transcript_position_updates,
            init_nuclear_cell_assignment: init_cell_assignments.clone(),
            prior_seg_cell_assignment: prior_seg_cell_assignment.clone(),
            cell_assignments: init_cell_assignments.clone(),
            cell_assignment_time: vec![0; init_cell_assignments.len()],
            cell_population: init_cell_population.clone(),
            cell_volume,
            cell_log_volume,
            component_volume,
            full_layer_volume,
            z0,
            layer_depth,
            transcript_state,
            prev_transcript_state,
            counts,
            foreground_counts: Array3::<u16>::from_elem((ncells, ngenes, nlayers), 0),
            confusion_counts: Array1::<u32>::from_elem(ngenes, 0),
            background_counts: Array2::<u32>::from_elem((ngenes, nlayers), 0),
            total_gene_counts,
            logfactorial: LogFactorial::new(),
            loggammaplus,
            z,
            component_population: Array1::<u32>::from_elem(ncomponents, 0),
            z_probs: ThreadLocal::new(),
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_volume: Array1::<f32>::from_elem(ncomponents, priors.μ_μ_volume),
            σ_volume: Array1::<f32>::from_elem(ncomponents, priors.σ_μ_volume),
            h,
            ω,
            φ,
            μ_φ,
            σ_φ,
            r,
            uv,
            lgamma_r,
            // θ: Array2::<f32>::from_elem((ncomponents, ngenes), 0.1),
            λ: Array2::<f32>::from_elem((ngenes, ncells), 0.1),
            λ_bg: Array2::<f32>::from_elem((ngenes, nlayers), 0.0),
            λ_c: Array1::<f32>::from_elem(ngenes, 1e-4),
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

        // prior seg reassignment terms
        ll += Zip::from(&self.cell_assignments)
            .and(&self.prior_seg_cell_assignment)
            .fold(0_f32, |accum, &cell, &nuc_cell| {
                    if cell == nuc_cell {
                        accum + priors.prior_seg_reassignment_1mlog_prob
                    } else {
                        accum + priors.prior_seg_reassignment_log_prob
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
    cell_to_cell_ignore: usize,
    background_to_cell_accept: usize,
    background_to_cell_reject: usize,
    background_to_cell_ignore: usize,
    cell_to_background_accept: usize,
    cell_to_background_reject: usize,
    cell_to_background_ignore: usize,
}

impl ProposalStats {
    pub fn new() -> Self {
        ProposalStats {
            cell_to_cell_accept: 0,
            cell_to_cell_reject: 0,
            cell_to_cell_ignore: 0,
            background_to_cell_accept: 0,
            background_to_cell_reject: 0,
            background_to_cell_ignore: 0,
            cell_to_background_accept: 0,
            cell_to_background_reject: 0,
            cell_to_background_ignore: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cell_to_cell_accept = 0;
        self.cell_to_cell_reject = 0;
        self.cell_to_cell_ignore = 0;
        self.background_to_cell_accept = 0;
        self.background_to_cell_reject = 0;
        self.background_to_cell_ignore = 0;
        self.cell_to_background_accept = 0;
        self.cell_to_background_reject = 0;
        self.cell_to_background_ignore = 0;
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
        let assignment = if params.transcript_state[i] != TranscriptState::Foreground {
            BACKGROUND_CELL
        } else {
            params.cell_assignments[i]
        };
        self.cell_assignment_duration
            .entry((i, assignment))
            .and_modify(|d| *d += duration)
            .or_insert(duration);
    }

    fn update_assignment_duration(&mut self, i: usize, cell: CellIndex, duration: u32) {
        self.cell_assignment_duration
            .entry((i, cell))
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
            } else if ij_prev.0 == usize::MAX || i == ij_prev.0 || (i > 0 && i - 1 == ij_prev.0) {
                summed_durations.push((i, j, d));
                ij_prev = (i, j);
            } else {
                panic!("Missing transcript in cell assignments.");
            }
        }

        // sort ascending on (transcript, cell) and descending on duration
        summed_durations.sort_by(|(i_a, j_a, d_a), (i_b, j_b, d_b)| {
            (*i_a, *d_b, *j_a).cmp(&(*i_b, *d_a, *j_b))
        });

        let mut maxpost_cell_assignments = Vec::new();
        let mut i_prev = usize::MAX;
        let mut j_prev = CellIndex::MAX;
        let mut d_prev = u32::MAX;
        for (i, j, d) in summed_durations.iter().cloned() {
            if i == i_prev {
                assert!(j != j_prev);
                assert!(d <= d_prev);
                continue;
            } else if i_prev == usize::MAX || (i > 0 && i - 1 == i_prev) {
                maxpost_cell_assignments.push((j, d as f32 / params.t as f32));
                i_prev = i;
                j_prev = j;
                d_prev = d;
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
        _foreground_pr_cutoff: f32,
    ) -> (Array2<u32>, Vec<(u32, f32)>) {
        let mut counts = Array2::<u32>::from_elem((params.ngenes(), params.ncells()), 0_u32);
        let maxpost_assignments = self.max_posterior_cell_assignments(&params);
        for (i, (j, pr)) in maxpost_assignments.iter().enumerate() {
            if *pr > count_pr_cutoff && *j != BACKGROUND_CELL {
                let gene = transcripts[i].gene;
                // let layer = params.zlayer(params.transcript_positions[i].2);

                // TODO: This doesn't really make sense, because we are tracking
                // the proportion of time a transcript is assigned to background
                // or confusion.

                // let λ_cell = params.λ[[gene as usize, *j as usize]];
                // let λ_bg = params.λ_bg[[gene as usize, layer]];
                // let λ_c = params.λ_c[gene as usize];
                // let fg_pr = λ_cell / (λ_cell + λ_bg + λ_c);

                // if fg_pr > foreground_pr_cutoff {
                    counts[[gene as usize, *j as usize]] += 1;
                // }
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
            // let layer = params.zlayer(params.transcript_positions[i].2);

            let w_d = d as f32 / (params.t - 1) as f32;

            // TODO: not accounting for λ_c here!!!

            // let λ_fg = params.λ[[gene as usize, j as usize]];
            // let λ_bg = params.λ_bg[[gene as usize, layer]];
            // let w_bg = λ_fg / (λ_fg + density * λ_bg);

            // ecounts[[gene as usize, j as usize]] += w_d * w_bg;

            ecounts[[gene as usize, j as usize]] += w_d;
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

    fn evaluate(&mut self, priors: &ModelPriors, params: &ModelParams, hillclimb: bool) {
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

        for &t in self.transcripts() {
            let cell = params.prior_seg_cell_assignment[t];
            if cell == old_cell {
                δ -= priors.prior_seg_reassignment_1mlog_prob;
            } else {
                δ -= priors.prior_seg_reassignment_log_prob;
            }

            if cell == new_cell {
                δ += priors.prior_seg_reassignment_1mlog_prob;
            } else {
                δ += priors.prior_seg_reassignment_log_prob;
            }
        }

        if from_background {
            Zip::from(self.gene_count().rows())
                .and(params.λ_bg.rows())
                .for_each(|gene_counts, λ_bg| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        δ -= count as f32 * λ_bg.ln();
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
                .and(params.λ_bg.rows())
                .and(&params.λ_c)
                .and(params.λ.column(old_cell as usize))
                .for_each(|gene_counts, λ_bg, &λ_c, λ| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        if count > 0 {
                            δ -= count as f32 * (λ_bg + λ_c + λ).ln();
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
                .and(params.λ_bg.rows())
                .for_each(|gene_counts, λ_bg| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        δ += count as f32 * λ_bg.ln();
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
                .and(params.λ_bg.rows())
                .and(&params.λ_c)
                .and(params.λ.column(new_cell as usize))
                .for_each(|gene_counts, λ_bg, &λ_c, λ| {
                    Zip::from(gene_counts).and(λ_bg).for_each(|&count, &λ_bg| {
                        if count > 0 {
                            δ += count as f32 * (λ_bg + λ_c + λ).ln();
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

        if (hillclimb && δ > 0.0) || (!hillclimb && logu < δ + self.log_weight()) {
            self.accept();
            // TODO: debugging
            // if from_background && !to_background {
            //     dbg!(
            //         self.log_weight(),
            //         δ,
            //         self.gene_count().sum(),
            //     );
            // }
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
    fn initialize(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // get to a reasonably high probability assignment
        for _ in 0..20 {
            self.sample_component_nb_params(priors, params, true);
        }
    }

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
        hillclimb: bool,
        uncertainty: &mut Option<&mut UncertaintyTracker>,
    ) {
        // don't count time unless we are tracking uncertainty
        if uncertainty.is_some() {
            params.t += 1;
        }
        self.repopulate_proposals(priors, params);
        self.proposals_mut()
            .par_iter_mut()
            .for_each(|p| p.evaluate(priors, params, hillclimb));
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
        // Keep track of stats
        for proposal in self.proposals().iter() {
            let old_cell = proposal.old_cell();
            let new_cell = proposal.new_cell();
            if proposal.accepted() {
                if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                    stats.background_to_cell_accept += 1;
                } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                    stats.cell_to_background_accept += 1;
                } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                    stats.cell_to_cell_accept += 1;
                }
            } else if proposal.ignored() {
                if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                    stats.background_to_cell_ignore += 1;
                } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                    stats.cell_to_background_ignore += 1;
                } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                    stats.cell_to_cell_ignore += 1;
                }
            } else {
                if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                    stats.background_to_cell_reject += 1;
                } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                    stats.cell_to_background_reject += 1;
                } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                    stats.cell_to_cell_reject += 1;
                }
            }
        }

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
                    if params.transcript_state[i] == TranscriptState::Foreground {
                        uncertainty.update(params, i);
                    }
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

        }

        self.update_sampler_state(params);
    }

    fn sample_global_params(
        &mut self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        uncertainty: &mut Option<&mut UncertaintyTracker>,
        burnin: bool,
    ) {
        let mut rng = thread_rng();

        // let t0 = Instant::now();
        self.sample_volume_params(priors, params);
        // println!("  Sample volume params: {:?}", t0.elapsed());

        // // Sample background/foreground counts
        // let t0 = Instant::now();
        self.sample_transcript_state(priors, params, transcripts, uncertainty);
        // println!("  Sample transcript states: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.compute_counts(priors, params, transcripts);
        // println!("  Compute counts: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_component_nb_params(priors, params, burnin);
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

        // let t0 = Instant::now();
        self.sample_background_rates(priors, params);
        // println!("  Sample background rates: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_confusion_rates(priors, params);
        // TODO: disabling confusion to see if it actually does anything
        // println!("  Sample confusion rates: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        if !burnin && priors.use_diffusion_model {
            self.sample_transcript_positions(priors, params, transcripts, uncertainty);
        }
        // println!("  Sample transcript positions: {:?}", t0.elapsed());
    }

    fn sample_transcript_state(
        &mut self,
        _priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        uncertainty: &mut Option<&mut UncertaintyTracker>)
    {
        params.prev_transcript_state.clone_from(&params.transcript_state);
        let nlayers = params.nlayers();
        Zip::from(&mut params.transcript_state)
            .and(&params.cell_assignments)
            .and(&params.transcript_positions)
            .and(transcripts)
            .into_par_iter()
            .with_min_len(100)
            .for_each(|(state, &cell, position, t)| {
                if cell == BACKGROUND_CELL {
                    *state = TranscriptState::Background;
                } else {
                    let gene = t.gene as usize;
                    let layer = ((position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                    let layer = layer.min(nlayers - 1);

                    let λ_cell = params.λ[[gene, cell as usize]];
                    let λ_bg = params.λ_bg[[gene, layer]];
                    let λ_c = params.λ_c[gene];
                    let λ = λ_cell + λ_bg + λ_c;

                    let u = thread_rng().gen::<f32>();
                    *state = if u < λ_cell / λ {
                        TranscriptState::Foreground
                    } else if u < (λ_cell + λ_bg) / λ {
                        TranscriptState::Background
                    } else {
                        TranscriptState::Confusion
                    };
                }
            });

            if let Some(uncertainty) = uncertainty.as_mut() {
                Zip::indexed(&mut params.cell_assignment_time)
                    .and(&params.prev_transcript_state)
                    .and(&params.transcript_state)
                    .and(&params.cell_assignments)
                    .for_each(|i, cell_assignment_time, &prev_state, &state, &assignment| {
                        let is_foreground = state == TranscriptState::Foreground;
                        let was_foreground = prev_state == TranscriptState::Foreground;

                        if is_foreground == was_foreground {
                            return;
                        }

                        let prev_assignment = if !was_foreground {
                            BACKGROUND_CELL
                        } else {
                            assignment
                        };

                        let duration = params.t - *cell_assignment_time;
                        uncertainty.update_assignment_duration(i, prev_assignment, duration);
                        *cell_assignment_time = params.t;
                    });
            }

    }

    fn compute_counts(&self, _priors: &ModelPriors, params: &mut ModelParams, transcripts: &Vec<Transcript>) {
        let nlayers = params.nlayers();
        params.confusion_counts.fill(0_u32);
        params.background_counts.fill(0_u32);
        params.foreground_counts.fill(0_u16);
        Zip::from(&params.transcript_state)
            .and(transcripts)
            .and(&params.cell_assignments)
            .and(&params.transcript_positions)
            .for_each(|&state, t, &cell, pos| {
                let gene = t.gene as usize;
                // let layer = params.zlayer(pos.2);
                let layer = ((pos.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                let layer = layer.min(nlayers - 1);

                match state {
                    TranscriptState::Background => {
                        params.background_counts[[gene, layer]] += 1;
                    },
                    TranscriptState::Confusion => {
                        params.confusion_counts[gene] += 1;
                    },
                    TranscriptState::Foreground => {
                        params.foreground_counts[[cell as usize, gene, layer]] += 1;
                    },
                }
            });

        // dbg!(params.background_counts.sum());
        // dbg!(params.confusion_counts.sum());
    }

    fn sample_component_nb_params(&mut self, priors: &ModelPriors, params: &mut ModelParams, burnin: bool) {
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

        // Sample ω

        // let rmin = params.r.iter().min_by(|a, b| a.partial_cmp(b).unwrap());
        // let rmax = params.r.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
        // dbg!(rmin, rmax);

        // let φmin = params.φ.iter().min_by(|a, b| a.partial_cmp(b).unwrap());
        // let φmax = params.φ.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
        // dbg!(φmin, φmax);

        // let vmin = params.cell_volume.iter().min_by(|a, b| a.partial_cmp(b).unwrap());
        // let vmax = params.cell_volume.iter().max_by(|a, b| a.partial_cmp(b).unwrap());
        // dbg!(vmin, vmax);

        // let t0 = Instant::now();
        Zip::from(params.ω.rows_mut()) // for every cell
            .and(params.foreground_counts.axis_iter(Axis(0)))
            .and(&params.cell_log_volume)
            .and(&params.z)
            .par_for_each(|ωs, cs, &logv, &z| {
                let mut rng = thread_rng();
                Zip::from(cs.axis_iter(Axis(0))) // for every gene
                    .and(ωs)
                    .and(params.φ.row(z as usize))
                    .and(params.r.row(z as usize))
                    .for_each(|c, ω, φ, &r| {
                        *ω = PolyaGamma::new(c.sum() as f32 + r, logv + φ).sample(&mut rng);
                    });
                });
        // println!("  Sample ω: {:?}", t0.elapsed());

        // Compute parameters to sample φ
        // let t0 = Instant::now();
        params.μ_φ.fill(0.0);
        params.σ_φ.fill(0.0);
        Zip::from(params.ω.rows()) // for every cell
            .and(params.foreground_counts.axis_iter(Axis(0))) // for each cell
            .and(&params.cell_log_volume)
            .and(&params.z)
            .for_each(|ωs, cs, &logv, &z| {
                Zip::from(params.μ_φ.row_mut(z as usize)) // for every gene
                    .and(params.σ_φ.row_mut(z as usize))
                    .and(params.r.row(z as usize))
                    .and(ωs)
                    .and(cs.axis_iter(Axis(0)))
                    .for_each(|μ, σ, &r, &ω, c| {
                        *σ += ω;
                        *μ += (c.sum() as f32 - r) / 2.0 - ω * logv;
                    });
            });

        Zip::from(&mut params.σ_φ)
            .for_each(|σ| *σ = (priors.γ + *σ).recip());

        Zip::from(&mut params.μ_φ)
            .and(&params.σ_φ)
            .for_each(|μ, σ| *μ *= σ);
        // println!("  Compute φ parameters: {:?}", t0.elapsed());

        // Sample φ
        // let t0 = Instant::now();
        Zip::from(&mut params.φ)
            .and(&params.μ_φ)
            .and(&params.σ_φ)
            .for_each(|φ, &μ, &σ| {
                let mut rng = thread_rng();
                *φ = Normal::new(μ, σ.sqrt()).unwrap().sample(&mut rng);
            });
        // println!("  Sample φ: {:?}", t0.elapsed());

        //     });
        // Zip::from(&mut params.ω)
        //     .and(&params.foreground_counts)

        // Sample θ
        // let t0 = Instant::now();
        // Zip::from(params.θ.columns_mut())
        //     .and(params.r.columns())
        //     .and(params.component_counts.rows())
        //     .par_for_each(|θs, rs, cs| {
        //         let mut rng = thread_rng();
        //         for (θ, &c, &r, &a) in izip!(θs, cs, rs, &params.component_volume) {
        //             *θ = prob_to_odds(
        //                 Beta::new(priors.α_θ + c as f32, priors.β_θ + a * r)
        //                     .unwrap()
        //                     .sample(&mut rng),
        //             );

        //             // *θ = θ.max(1e-6).min(1e6);
        //         }
        //     });
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
        } else if burnin && priors.burnin_dispersion.is_some() {
            let dispersion = priors.burnin_dispersion.unwrap();
            set_constant_dispersion(params, dispersion);
        } else {
            // for each gene
            params.uv.fill((0_u32, 0_f32));
            Zip::from(params.r.columns_mut())
                .and(params.lgamma_r.columns_mut())
                .and(params.loggammaplus.columns_mut())
                .and(params.φ.columns())
                .and(params.foreground_counts.axis_iter(Axis(1)))
                .and(params.uv.columns_mut())
                .par_for_each(|
                    rs,
                    lgamma_rs,
                    loggammaplus,
                    φs,
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
                            let φ = φs[z];
                            let ψ = φ + vol.ln();
                            let δv = -ψ - log1pf((-ψ).exp());

                            let uv_z = uv[z];

                            if (uv[z].1 + δv).is_infinite() {
                                dbg!(uv[z], ψ, φ, vol);
                            }

                            uv[z] = (
                                uv_z.0 + rand_crt(&mut rng, c as u32, r),
                                uv_z.1 + δv
                            );

                            assert!(uv[z].1.is_finite());
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

                            // TODO: Without this, things get kind of fucky.
                            // *r = r.min(200.0).max(2e-4);
                            *r = r.max(2e-4);
                            // *r = r.min(50.0);
                            // *r = r.max(1e-5);

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
        // dbg!(params.h);
    }

    fn sample_rates(&mut self, _priors: &ModelPriors, params: &mut ModelParams) {
        // loop over genes
        Zip::from(params.λ.rows_mut())
            .and(params.foreground_counts.axis_iter(Axis(1)))
            .and(params.φ.columns())
            .and(params.r.columns())
            .par_for_each(|mut λs, cs, φs, rs| {
                let mut rng = thread_rng();
                // loop over cells
                for (λ, &z, cs, cell_volume) in
                    izip!(&mut λs, &params.z, cs.outer_iter(), &params.cell_volume)
                {
                    let z = z as usize;

                    let c = cs.sum();
                    let φ = φs[z];
                    // dbg!(φ, φ.exp(), (-φ).exp());
                    // let ψ = φ + cell_volume.ln();
                    let r = rs[z];

                    let α = r + c as f32;
                    // let β = (-φ).exp() / cell_volume + 1.0;
                    let β0 = (-φ).exp();
                    let β = β0 + cell_volume;


                    *λ = Gamma::new(
                        α,
                        β.recip()
                    )
                    .unwrap()
                    .sample(&mut rng);
                    // .max(1e-14);

                    // if c > 10 {
                    // // TODO: How far off is than from just count divided by volume?
                    //     dbg!(
                    //         *λ,
                    //         r,
                    //         c as f32 / *cell_volume, *cell_volume);
                    // }

                    assert!(λ.is_finite());
                }
            });
    }

    fn sample_background_rates(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        let mut rng = thread_rng();

        Zip::from(params.λ_bg.rows_mut())
            .and(params.background_counts.rows())
            .for_each(|λs, cs| {
                Zip::from(λs).and(cs).for_each(|λ, c| {
                    let α = priors.α_bg + *c as f32;
                    let β = priors.β_bg + params.full_layer_volume;
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

    fn sample_confusion_rates(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        let total_cell_volume = params.cell_volume.sum();
        let mut rng = thread_rng();
        Zip::from(&mut params.λ_c)
            .and(&params.confusion_counts)
            .for_each(|λ, c| {
                let α = priors.α_bg + *c as f32;
                let β = priors.β_bg + total_cell_volume;
                *λ = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) as f32;
            });
    }

    fn sample_component_assignments(&mut self, _priors: &ModelPriors, params: &mut ModelParams) {
        let ncomponents = params.ncomponents();

        // loop over cells
        Zip::from(params.foreground_counts.axis_iter(Axis(0)))
            .and(&mut params.z)
            .and(&params.cell_log_volume)
            .par_for_each(|cs, z_i, cell_log_volume| {
                let mut z_probs = params
                    .z_probs
                    .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                    .borrow_mut();

                // loop over components
                for (zp, π, φs, rs, lgamma_r, loggammaplus, &μ_volume, &σ_volume) in izip!(
                    z_probs.iter_mut(),
                    &params.π,
                    params.φ.rows(),
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
                            .and(φs)
                            .and(lgamma_r)
                            .and(loggammaplus)
                            .fold(0_f32, |accum, cs, &r, φ, &lgamma_r, lgammaplus| {
                                let ψ = φ + cell_log_volume;
                                let c = cs.iter().map(|&x| x as u32).sum(); // sum counts across layers
                                accum
                                    + negbin_logpmf_fast(
                                        r,
                                        lgamma_r,
                                        lgammaplus.eval(c),
                                        logistic(ψ),
                                        c,
                                        params.logfactorial.eval(c),
                                    )
                            }) as f64)
                            .exp();

                    *zp *= normal_pdf(μ_volume, σ_volume, *cell_log_volume).exp() as f64;
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
        Zip::from(&mut params.cell_log_volume)
            .and(&params.cell_volume)
            .into_par_iter()
            .with_min_len(50)
            .for_each(|(log_volume, &volume)| {
                *log_volume = volume.ln();
            });

        // compute sample means
        params.component_population.fill(0_u32);
        params.μ_volume.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_log_volume)
            .for_each(|&z, &log_volume| {
                params.μ_volume[z as usize] += log_volume;
                params.component_population[z as usize] += 1;
            });

        // Sampling is parallelizable, but unually number of components is low,
        // so it's dominated by overhead.

        // sample μ parameters
        Zip::from(&mut params.μ_volume)
            .and(&params.σ_volume)
            .and(&params.component_population)
            .for_each(|μ, &σ, &pop| {
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
            .and(&params.cell_log_volume)
            .for_each(|&z, &log_volume| {
                params.σ_volume[z as usize] += (params.μ_volume[z as usize] - log_volume).powi(2);
            });

        // sample σ parameters
        Zip::from(&mut params.σ_volume)
            .and(&params.component_population)
            .for_each(|σ, &pop| {
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
            // .zip(&params.transcript_positions)
            .zip(transcripts)
            .for_each(|(proposed_position, t)| {
                let mut rng = thread_rng();
                *proposed_position = (
                    t.x + priors.σ_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal),
                    t.y + priors.σ_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal),
                    (t.z + priors.σ_z_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal)).min(priors.zmax).max(priors.zmin),
                );

                // Only z-axis repo
                // *proposed_position = (
                //     t.x,
                //     t.y,
                //     (t.z + priors.σ_z_diffusion_proposal * rng.sample::<f32, StandardNormal>(StandardNormal)).min(priors.zmax).max(priors.zmin),
                // );
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

                // TODO: account for the possibility that sigma is 0

                // prior on xy-diffusion distances
                δ -= ((1.0 - priors.p_diffusion) * normal_x2_pdf(priors.σ_diffusion_near, sq_dist_prev) +
                    priors.p_diffusion * normal_x2_pdf(priors.σ_diffusion_far, sq_dist_prev)).ln();
                δ += ((1.0 - priors.p_diffusion) * normal_x2_pdf(priors.σ_diffusion_near, sq_dist_new) +
                    priors.p_diffusion * normal_x2_pdf(priors.σ_diffusion_far, sq_dist_new)).ln();

                // weight by xy proposal distribution
                δ += normal_x2_logpdf(priors.σ_diffusion_proposal, sq_dist_prev);
                δ -= normal_x2_logpdf(priors.σ_diffusion_proposal, sq_dist_new);

                // prior on z diffusion distance
                δ -= -0.5 * (z_sq_dist_prev / priors.σ_z_diffusion.powi(2));
                δ += -0.5 * (z_sq_dist_new / priors.σ_z_diffusion.powi(2));

                // weight by z proposal distribution
                δ += normal_x2_logpdf(priors.σ_z_diffusion_proposal, z_sq_dist_prev);
                δ -= normal_x2_logpdf(priors.σ_z_diffusion_proposal, z_sq_dist_new);

                let gene = transcript.gene as usize;

                let layer_prev = ((position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                let layer_prev = layer_prev.min(params.λ_bg.ncols() - 1);
                let cell_prev = self.cell_at_position(*position);
                let λ_prev = if cell_prev == BACKGROUND_CELL {
                    0.0
                } else {
                    params.λ[[gene, cell_prev as usize]] + params.λ_c[gene]
                } + params.λ_bg[[gene, layer_prev]];

                let layer_new = ((proposed_position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                let layer_new = layer_new.min(params.λ_bg.ncols() - 1);
                let cell_new = self.cell_at_position(*proposed_position);
                let λ_new = if cell_new == BACKGROUND_CELL {
                    0.0
                } else {
                    params.λ[[gene, cell_new as usize]] + params.λ_c[gene]
                } + params.λ_bg[[gene, layer_new]];

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

                let cell_prior = params.prior_seg_cell_assignment[i];
                if cell_prior == cell_prev {
                    δ -= priors.prior_seg_reassignment_1mlog_prob;
                } else {
                    δ -= priors.prior_seg_reassignment_log_prob;
                }

                if cell_prior == cell_new {
                    δ += priors.prior_seg_reassignment_1mlog_prob;
                } else {
                    δ += priors.prior_seg_reassignment_log_prob;
                }

                let mut rng = thread_rng();
                let logu = rng.gen::<f32>().ln();
                *accept = logu < δ;
            });
        // println!("  Eval transcript position proposals: {:?}", t0.elapsed());

        // let naccepted = params.accept_proposed_transcript_positions.iter().map(|&x| x as u32).sum::<u32>();
        // let prop_accepted = naccepted as f32 / params.accept_proposed_transcript_positions.len() as f32;
        // println!("  Accepted {}% of transcript repo proposals", 100.0 * prop_accepted);
    }

    fn sample_transcript_positions(
        &mut self, priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        uncertainty: &mut Option<&mut UncertaintyTracker>)
    {
        self.propose_eval_transcript_positions(priors, params, transcripts);

        // Update position and compute cell and layer changes for updates
        params.transcript_position_updates
            .par_iter_mut()
            .zip(&mut params.transcript_positions)
            .zip(&params.proposed_transcript_positions)
            .zip(&params.cell_assignments)
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
                }
            });

        // Update counts and cell_population
        params.transcript_position_updates
            .iter()
            .zip(transcripts)
            .zip(&params.accept_proposed_transcript_positions)
            .zip(&mut params.cell_assignments)
            .zip(&mut params.cell_assignment_time)
            .enumerate()
            .for_each(|(i, ((((update, transcript), &accept), cell_assignment), cell_assignment_time))| {
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

                    if cell_prev != cell_new {
                        // annoying borrowing bullshit
                        // uncertainty.update_assignment(params, i, cell_new);

                        if let Some(uncertainty) = uncertainty.as_mut() {
                            if params.transcript_state[i] == TranscriptState::Foreground {
                                let duration = params.t - *cell_assignment_time;
                                uncertainty.update_assignment_duration(i, cell_prev, duration);
                            }
                        }

                        *cell_assignment = cell_new;
                        *cell_assignment_time = params.t;
                    }
                }
            });

        self.update_transcript_positions(&params.accept_proposed_transcript_positions, &params.transcript_positions);
    }
}
