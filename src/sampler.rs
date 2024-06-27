mod connectivity;
pub mod hull;
mod math;
pub mod polyagamma;
mod polygons;
mod sampleset;
pub mod transcripts;
pub mod voxelsampler;

use core::fmt::Debug;
use std::thread::Thread;
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::{izip, Itertools};
use libm::log1pf;
use linfa::traits::{Fit, Predict};
use linfa::DatasetBase;
use linfa_clustering::KMeans;
use math::{lognormal_logpdf, normal_x2_logpdf, normal_x2_pdf, rand_crt, randn, sq};
use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
use ndarray::{s, Array1, Array2, Array3, Axis, Zip};
use ndarray_linalg::cholesky::*;
use ndarray_linalg::solve::Inverse;
use polyagamma::PolyaGamma;
use rand::{thread_rng, Rng};
use rand_distr::{Beta, Binomial, Dirichlet, Distribution, Gamma, Normal, StandardNormal};
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::f32;
use std::fs::File;
use std::io::Write;
use std::iter::Iterator;
use std::sync::{Arc, Mutex};
use thread_local::ThreadLocal;
use transcripts::{CellIndex, Transcript, BACKGROUND_CELL};

use std::time::Instant;

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
    bound * eta * (2.0 * (f32::consts::PI * population).sqrt())
}

// Compute chunk and quadrant for a single a single (x,y) point.
fn chunkquad(x: f32, y: f32, xmin: f32, ymin: f32, chunk_size: f32, nxchunks: usize) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size / 2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size / 2.0)).floor() as u32;

    let chunk = (xchunkquad / 2) + (ychunkquad / 2) * (nxchunks as u32);
    let quad = (xchunkquad % 2) + (ychunkquad % 2) * 2;

    (chunk, quad)
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

    // dirichlet prior for θ
    pub α_θ: f32,

    pub ε_p: f32,
    pub c_p: f32,

    pub c_r: f32,
    pub r_r: f32,

    // gamma rate prior
    pub e_r: f32,

    pub e_h: f32,
    pub f_h: f32,

    // gamma prior on γ
    pub e_γ: f32,
    pub f_γ: f32,

    // TODO: probably should rename this to not collide with precios params for θ
    // normal precision parameter for β
    pub γ: f32,

    // gamma prior for background rates
    pub α_bg: f32,
    pub β_bg: f32,

    // gamma prior for confusion rates
    pub α_c: f32,
    pub β_c: f32,

    // gaussian prior on μ_φ
    pub σ_μ_φ: f32,

    pub α_σ_φ: f32,
    pub β_σ_φ: f32,

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
#[allow(non_snake_case)]
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

    // [ncells] per-cell volumes
    pub cell_volume: Array1<f32>,
    pub cell_log_volume: Array1<f32>,

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

    // [ngenes, nhidden]
    pub cell_latent_counts: Array2<u32>,

    // [ncells, nhidden]
    pub gene_latent_counts: Array2<u32>,

    // `gene_latent_counts` rows behind mutexes for parallel accumulation
    pub gene_latent_counts_rows: Vec<Arc<Mutex<Array1<u32>>>>,

    // [nhidden]
    pub latent_counts: Array1<u32>,

    // [nhidden] thread local storage for sampling latent counts
    pub multinomial_rates: ThreadLocal<RefCell<Array1<f32>>>,
    pub multinomial_sample: ThreadLocal<RefCell<Array1<u32>>>,

    pub z: Array1<u32>, // assignment of cells to components

    component_population: Array1<u32>, // number of cells assigned to each component

    // thread-local space used for sampling z
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,

    // [ncomponents, nhidden]: mixture mean parameter
    μ_φ: Array2<f32>,

    // [ncomponents, nhidden]: mixture stddev parameters
    σ_φ: Array2<f32>,

    π: Vec<f32>, // mixing proportions over components

    μ_volume: Array1<f32>, // volume dist mean param by component
    σ_volume: Array1<f32>, // volume dist std param by component

    // Prior on NB dispersion parameters
    h: f32,

    // [ncells, ngenes] Polya-gamma samples, used for sampling NB rates
    ω: Array2<f32>,

    // [ngenes]
    η: Array1<f32>,

    // [ncells, nhidden]: cell ψ parameter in the latent space
    pub φ: Array2<f32>,

    // [ncells, nhidden, hidden]: φ posterior covariance matrices (and temporary space to compute φφ^T products)
    Σφ: Array3<f32>,

    // [ncells, nhidden]: φ posterior, mean parameters
    μφ: Array2<f32>,

    // [ngenes, nhidden]: gene loadings in the latent space
    pub θ: Array2<f32>,

    // [ngenes, nhidden, nhidden]: θ posterior covariance matrices (and temporary space to compute θθ^T products)
    Σθ: Array3<f32>,

    // [ngenes, nhidden]: θ posterior mean parameters
    μθ: Array2<f32>,

    // [nhidden]: θ precision parameters
    γ: Array1<f32>,

    // [ncells, ngenes] NB logit(p) parameters
    pub ψ: Array2<f32>,

    // [nhidden] φ gamma scale parameters
    pub p: Array1<f32>,

    // [nhidden] φ gamma shape parameters
    pub r: Array1<f32>,

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
    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    pub fn new(
        priors: &ModelPriors,
        full_layer_volume: f32,
        z0: f32,
        layer_depth: f32,
        initial_perturbation_sd: f32,
        transcripts: &[Transcript],
        init_cell_assignments: &[u32],
        init_cell_population: &[usize],
        prior_seg_cell_assignment: &[u32],
        ncomponents: usize,
        nhidden: usize,
        nlayers: usize,
        ncells: usize,
        ngenes: usize,
    ) -> Self {
        let mut transcript_positions = transcripts
            .iter()
            .map(|t| (t.x, t.y, t.z))
            .collect::<Vec<_>>();

        if initial_perturbation_sd > 0.0 {
            let mut rng = rand::thread_rng();
            for pos in &mut transcript_positions {
                pos.0 +=
                    rng.sample::<f32, StandardNormal>(StandardNormal) * initial_perturbation_sd;
                pos.1 +=
                    rng.sample::<f32, StandardNormal>(StandardNormal) * initial_perturbation_sd;
                pos.2 +=
                    rng.sample::<f32, StandardNormal>(StandardNormal) * initial_perturbation_sd;
            }
        }

        let proposed_transcript_positions = transcript_positions.clone();
        let accept_proposed_transcript_positions = vec![false; transcripts.len()];
        let transcript_position_updates = vec![(0, 0, 0, 0); transcripts.len()];

        let r = Array1::<f32>::from_elem(nhidden, 1.0_f32);
        let p = Array1::<f32>::from_elem(nhidden, 1.0_f32);
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
        let mut init_samples = counts
            .sum_axis(Axis(2))
            .map(|&x| (x as f32))
            .reversed_axes();
        init_samples.rows_mut().into_iter().for_each(|mut row| {
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

        let ω = Array2::<f32>::from_elem((ncells, ngenes), 0.0);
        let η = Array1::<f32>::from_elem(ngenes, 0.0);

        let mut rng = rand::thread_rng();
        let φ = Array2::<f32>::from_shape_simple_fn((ncells, nhidden), || 1e-2 * randn(&mut rng).exp());
        let Σφ = Array3::<f32>::from_elem((ncells, nhidden, nhidden), 0.0);
        let μφ = Array2::<f32>::from_elem((ncells, nhidden), 0.0);
        let θ = Array2::<f32>::from_shape_simple_fn((ngenes, nhidden), || randn(&mut rng).exp());

        // {
        //     // TODO: debug attempt
        //     //   find a special CD3e
        //     //   make one component identity for that gene
        //     //   zero it out for everything else
        //     //   don't sample θ
        //
        //     let cd3e_pos = 290;
        //     θ.row_mut(cd3e_pos).fill(0.0);
        //     θ.column_mut(0).fill(0.0);
        //     θ[[cd3e_pos, 0]] = 1.0;
        // }

        let Σθ = Array3::<f32>::from_elem((ngenes, nhidden, nhidden), 0.0);
        let μθ = Array2::<f32>::from_elem((ngenes, nhidden), 0.0);

        let γ = Array1::<f32>::from_elem(nhidden, 1.0);

        let ψ = Array2::<f32>::from_elem((ncells, ngenes), 0.0);

        // let μ_ψ = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);
        // let σ_ψ = Array2::<f32>::from_elem((ncomponents, ngenes), 0.0);

        let transcript_state =
            Array1::<TranscriptState>::from_elem(transcripts.len(), TranscriptState::Foreground);
        let prev_transcript_state =
            Array1::<TranscriptState>::from_elem(transcripts.len(), TranscriptState::Foreground);

        let z_probs = ThreadLocal::new();
        let μ_φ = Array2::<f32>::from_elem((ncomponents, nhidden), 0.0);
        let σ_φ = Array2::<f32>::from_elem((ncomponents, nhidden), 1.0);

        let foreground_counts = Array3::<u16>::from_elem((ncells, ngenes, nlayers), 0);
        let confusion_counts = Array1::<u32>::from_elem(ngenes, 0);
        let background_counts = Array2::<u32>::from_elem((ngenes, nlayers), 0);
        let cell_latent_counts = Array2::<u32>::from_elem((ncells, nhidden), 0);
        let gene_latent_counts = Array2::<u32>::from_elem((ngenes, nhidden), 0);
        let latent_counts = Array1::<u32>::from_elem(nhidden, 0);

        let mut gene_latent_counts_rows = Vec::new();
        for _ in 0..ngenes {
            gene_latent_counts_rows.push(
                Arc::new(Mutex::new(
                    Array1::from_elem(nhidden, 0)
                ))
            );
        }

        ModelParams {
            transcript_positions,
            proposed_transcript_positions,
            accept_proposed_transcript_positions,
            transcript_position_updates,
            init_nuclear_cell_assignment: init_cell_assignments.to_vec(),
            prior_seg_cell_assignment: prior_seg_cell_assignment.to_vec(),
            cell_assignments: init_cell_assignments.to_vec(),
            cell_assignment_time: vec![0; init_cell_assignments.len()],
            cell_population: init_cell_population.to_vec(),
            cell_volume,
            cell_log_volume,
            full_layer_volume,
            z0,
            layer_depth,
            transcript_state,
            prev_transcript_state,
            counts,
            foreground_counts,
            confusion_counts,
            background_counts,
            total_gene_counts,
            cell_latent_counts,
            gene_latent_counts,
            gene_latent_counts_rows,
            latent_counts,
            multinomial_rates: ThreadLocal::new(),
            multinomial_sample: ThreadLocal::new(),
            z,
            component_population: Array1::<u32>::from_elem(ncomponents, 0),
            z_probs,
            μ_φ,
            σ_φ,
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_volume: Array1::<f32>::from_elem(ncomponents, priors.μ_μ_volume),
            σ_volume: Array1::<f32>::from_elem(ncomponents, priors.σ_μ_volume),
            h,
            ω,
            η,
            φ,
            Σφ,
            μφ,
            θ,
            Σθ,
            μθ,
            γ,
            ψ,
            p,
            r,
            // θ: Array2::<f32>::from_elem((ncomponents, ngenes), 0.1),
            λ: Array2::<f32>::from_elem((ncells, ngenes), 0.1),
            λ_bg: Array2::<f32>::from_elem((ngenes, nlayers), 0.0),
            λ_c: Array1::<f32>::from_elem(ngenes, 1e-4),
            t: 0,
        }
    }

    pub fn ncomponents(&self) -> usize {
        self.π.len()
    }

    fn zlayer(&self, z: f32) -> usize {
        let layer = ((z - self.z0) / self.layer_depth).max(0.0) as usize;
        layer.min(self.nlayers() - 1)
    }

    fn recompute_counts(&mut self, transcripts: &[Transcript]) {
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

    fn check_counts(&self, transcripts: &[Transcript]) {
        for (i, (transcript, &assignment)) in
            transcripts.iter().zip(&self.cell_assignments).enumerate()
        {
            let layer = self.zlayer(self.transcript_positions[i].2);
            if assignment != BACKGROUND_CELL {
                assert!(self.counts[[transcript.gene as usize, assignment as usize, layer]] > 0);
            }
        }
    }

    pub fn nforeground(&self) -> usize {
        self.foreground_counts.iter().map(|x| *x as usize).sum()
    }

    pub fn nassigned(&self) -> usize {
        self.cell_assignments
            .iter()
            .filter(|&c| *c != BACKGROUND_CELL)
            .count()
    }

    pub fn ncells(&self) -> usize {
        self.cell_population.len()
    }

    pub fn ngenes(&self) -> usize {
        self.total_gene_counts.shape()[0]
    }

    pub fn nlayers(&self) -> usize {
        self.total_gene_counts.shape()[1]
    }

    pub fn log_likelihood(&self, priors: &ModelPriors) -> f32 {
        // iterate over cells
        let mut ll = Zip::from(self.λ.rows())
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

        ll
    }

    pub fn write_cell_hulls(
        &self,
        transcripts: &[Transcript],
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

        UncertaintyTracker {
            cell_assignment_duration,
        }
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

        maxpost_cell_assignments
    }

    pub fn max_posterior_transcript_counts_assignments(
        &self,
        params: &ModelParams,
        transcripts: &[Transcript],
        count_pr_cutoff: f32,
        _foreground_pr_cutoff: f32,
    ) -> (Array2<u32>, Vec<(u32, f32)>) {
        let mut counts = Array2::<u32>::from_elem((params.ngenes(), params.ncells()), 0_u32);
        let maxpost_assignments = self.max_posterior_cell_assignments(params);
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

        (counts, maxpost_assignments)
    }

    pub fn expected_counts(&self, params: &ModelParams, transcripts: &[Transcript]) -> Array2<f32> {
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

        ecounts
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
            δ += Zip::from(params.λ.row(old_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * volume_diff);

            Zip::from(self.gene_count().rows())
                .and(params.λ_bg.rows())
                .and(&params.λ_c)
                .and(params.λ.row(old_cell as usize))
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
            δ += Zip::from(params.λ.row(new_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * volume_diff);

            // add in new cell likelihood terms
            Zip::from(self.gene_count().rows())
                .and(params.λ_bg.rows())
                .and(&params.λ_c)
                .and(params.λ.row(new_cell as usize))
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
    Self: Sync,
{
    // fn generate_proposals<'b, 'c>(&'b mut self, params: &ModelParams) -> &'c mut [P] where 'b: 'c;
    fn initialize(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        Zip::from(&mut params.cell_log_volume)
            .and(&params.cell_volume)
            .into_par_iter()
            .with_min_len(50)
            .for_each(|(log_volume, &volume)| {
                *log_volume = volume.ln();
            });

        // get to a reasonably high probability assignment
        // for _ in 0..40 {
        for _ in 0..10 {
            self.sample_factor_model(priors, params);
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
    fn update_transcript_positions(&mut self, updated: &[bool], positions: &[(f32, f32, f32)]);

    fn cell_at_position(&self, pos: (f32, f32, f32)) -> u32;

    fn sample_cell_regions(
        &mut self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        stats: &mut ProposalStats,
        transcripts: &[Transcript],
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
        transcripts: &[Transcript],
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
            } else if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.background_to_cell_reject += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                stats.cell_to_background_reject += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.cell_to_cell_reject += 1;
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
                    let layer = ((params.transcript_positions[i].2 - params.z0)
                        / params.layer_depth)
                        .max(0.0) as usize;
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
                    let layer = ((params.transcript_positions[i].2 - params.z0)
                        / params.layer_depth)
                        .max(0.0) as usize;
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
        // let t0 = Instant::now();
        self.sample_volume_params(priors, params);
        // println!("  Sample volume params: {:?}", t0.elapsed());

        // // Sample background/foreground counts
        // let t0 = Instant::now();
        self.sample_transcript_state(priors, params, transcripts, uncertainty);
        // println!("  Sample transcript states: {:?}", t0.elapsed());

        self.compute_counts(priors, params, transcripts);

        self.sample_factor_model(priors, params);

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
        uncertainty: &mut Option<&mut UncertaintyTracker>,
    ) {
        params
            .prev_transcript_state
            .clone_from(&params.transcript_state);
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

                    let λ_cell = params.λ[[cell as usize, gene]];
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
                .for_each(
                    |i, cell_assignment_time, &prev_state, &state, &assignment| {
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
                    },
                );
        }
    }

    fn compute_counts(
        &self,
        _priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
    ) {
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
                    }
                    TranscriptState::Confusion => {
                        params.confusion_counts[gene] += 1;
                    }
                    TranscriptState::Foreground => {
                        params.foreground_counts[[cell as usize, gene, layer]] += 1;
                    }
                }
            });

        // dbg!(params.background_counts.sum());
        // dbg!(params.confusion_counts.sum());

        // // TODO: trying to debug factorization by inserting some artificial counts
        // // What's the simplest version of this? Separate cells into to odd and even
        // let mut rng = thread_rng();
        // let nlayers = params.foreground_counts.shape()[2];
        // params.foreground_counts.fill(0);
        // for (i, mut x_i) in params.foreground_counts.outer_iter_mut().enumerate() {
        //     for (j, mut x_ij) in x_i.outer_iter_mut().enumerate() {
        //         if (i % 3) == (j % 3) {
        //             // choose a random layer
        //             let z = rng.gen_range(0..nlayers);
        //             x_ij[z] = Poisson::new(1.0).unwrap().sample(&mut rng) as u16;
        //         }
        //     }
        // }

        //
    }

    fn sample_latent_counts(&mut self, params: &mut ModelParams) {
        for gene_latent_counts_g in params.gene_latent_counts_rows.iter_mut() {
            gene_latent_counts_g.lock().unwrap().fill(0);
        }

        params.cell_latent_counts.fill(0);

        Zip::from(params.cell_latent_counts.outer_iter_mut()) // for every cell
            .and(params.φ.outer_iter())
            .and(params.foreground_counts.outer_iter())
            .par_for_each(|mut cell_latent_counts_c, φ_c, x_c| {
            // .for_each(|mut cell_latent_counts_c, φ_c, x_c| {
                let nhidden = cell_latent_counts_c.shape()[0];
                let mut rng = thread_rng();
                let mut multinomial_rates = params.multinomial_rates
                    .get_or(|| RefCell::new(Array1::zeros(nhidden)))
                    .borrow_mut();

                let mut multinomial_sample = params.multinomial_sample
                    .get_or(|| RefCell::new(Array1::zeros(nhidden)))
                    .borrow_mut();

                // TODO: The way we juice this further is to have thread local gene_latent_counts which we add
                // together at the end.

                Zip::indexed(x_c.outer_iter()) // for every gene
                    .and(params.θ.outer_iter())
                    .for_each(|g, x_cg, θ_g| {
                        let x_cg = x_cg.sum();

                        if x_cg == 0 {
                            multinomial_sample.fill(0);
                        } else {
                            // rates: normalized element-wise product
                            multinomial_rates.assign(&φ_c);
                            *multinomial_rates *= &θ_g;
                            let rate_norm = multinomial_rates.sum();
                            *multinomial_rates /= rate_norm;

                            // multinomial sampling
                            {
                                let mut ρ = 1.0;
                                let mut s = x_cg as u32;
                                for (p, x) in izip!(multinomial_rates.iter(), multinomial_sample.iter_mut()) {
                                    if ρ > 0.0 {
                                        *x = Binomial::new(s as u64, (*p/ρ) as f64).unwrap().sample(&mut rng) as u32;
                                    } else {
                                        *x = 0;
                                    }
                                    s -= *x;
                                    ρ = (ρ - *p).max(1.0);
                                }
                            }
                        }

                        // add to cell marginal
                        cell_latent_counts_c.scaled_add(1, &multinomial_sample);

                        // add to gene marginal
                        let mut gene_latent_counts_g = params.gene_latent_counts_rows[g].lock().unwrap();
                        gene_latent_counts_g.scaled_add(1, &multinomial_sample);
                    });
            });

        // copy from the mutex protected rows
        Zip::from(params.gene_latent_counts.outer_iter_mut())
            .and(&params.gene_latent_counts_rows)
            .for_each(|mut gene_latent_counts_g, gene_latent_counts_g_tmp| {
                gene_latent_counts_g.assign(
                    &gene_latent_counts_g_tmp.lock().unwrap()
                )
            });

        // marginal count along the hidden axis
        Zip::from(&mut params.latent_counts)
            .and(params.gene_latent_counts.columns())
            .for_each(|lc, glc| {
                *lc = glc.sum();
            });

        // These should all be the same
        dbg!(params.gene_latent_counts.sum());
        dbg!(params.cell_latent_counts.sum());
        dbg!(params.latent_counts.sum());
    }

    fn sample_factor_model(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        let t0 = Instant::now();
        self.sample_latent_counts(params);
        println!("  sample_latent_counts: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_θ(priors, params);
        // println!("  sample_θ: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_p(priors, params);
        // println!("  sample_p: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_r(priors, params);
        // println!("  sample_r: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.sample_φ(params);
        // println!("  sample_φ: {:?}", t0.elapsed());

        // let t0 = Instant::now();
        self.compute_rates(params);
        // println!("  compute_rates: {:?}", t0.elapsed());
    }

    fn sample_p(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // TODO: I wonder if `ncells` here should be total cellular volume?
        let mut rng = thread_rng();
        let ncells = params.foreground_counts.shape()[0];
        Zip::from(&mut params.p)
            .and(&params.r)
            .and(&params.latent_counts)
            .for_each(|p_k, r_k, x_k| {
                let a = priors.c_p * priors.ε_p + *x_k as f32;
                let b = priors.c_p * (1.0 - priors.ε_p) + (ncells as f32) * *r_k;
                *p_k = Beta::new(a, b).unwrap().sample(&mut rng);
            });
    }

    fn sample_r(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        let mut rng = thread_rng();
        let ncells = params.foreground_counts.shape()[0];
        Zip::from(&mut params.r)
            .and(&params.p)
            .and(&params.latent_counts)
            .for_each(|r_k, p_k, x_k| {
                if *x_k == 0 {
                    let shape = priors.c_r * priors.r_r;
                    let scale = (priors.c_r - (ncells as f32) * log1pf(-*p_k)).recip();
                    *r_k = Gamma::new(shape, scale).unwrap().sample(&mut rng);
                } else {
                    // TODO: oof, we've got to do full on metropolis hastings here
                    *r_k = 1e-1;
                }
            });
    }

    fn compute_rates(&mut self, params: &mut ModelParams) {
        general_mat_mul(1.0, &params.φ, &params.θ.t(), 0.0, &mut params.λ);
        dbg!(params.λ.mean());
        dbg!(params.λ_bg.mean());
    }

    fn sample_ω(&mut self, params: &mut ModelParams) {
        Zip::from(params.ω.rows_mut()) // for every cell
            .and(params.ψ.rows())
            .and(params.foreground_counts.axis_iter(Axis(0)))
            .par_for_each(|ωs, ψs, cs| {
                let mut rng = thread_rng();

                Zip::from(cs.axis_iter(Axis(0))) // for every gene
                    .and(ωs)
                    .and(ψs)
                    .and(&params.r)
                    .for_each(|c, ω, &ψ, &r| {
                        *ω = PolyaGamma::new(c.sum() as f32 + r, ψ).sample(&mut rng);
                    });
            });
    }

    fn sample_component_params(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // sample component mean parameters

        // component populations and sums
        params.component_population.fill(0);
        params.μ_φ.fill(0.0);
        Zip::from(&params.z) // for every cell
            .and(params.φ.outer_iter())
            .for_each(|&z_c, φ_c| {
                let z_c = z_c as usize;
                params.component_population[z_c] += 1;
                let mut μ_φ_z_c = params.μ_φ.row_mut(z_c);
                μ_φ_z_c.scaled_add(1.0, &φ_c);
            });
        dbg!(&params.component_population);
        // dbg!(&params.π);

        // sample from posterion gaussian
        let mut rng = thread_rng();
        Zip::from(params.μ_φ.outer_iter_mut())
            .and(params.σ_φ.outer_iter())
            .and(&params.component_population)
            .for_each(|μ_φ_h, σ_φ_h, &n_h| {
                Zip::from(μ_φ_h).and(σ_φ_h).for_each(|μ_φ_hk, σ_φ_hk| {
                    let σ = (n_h as f32 / sq(*σ_φ_hk) + 1.0 / sq(priors.σ_μ_φ)).recip();
                    let μ = σ * (*μ_φ_hk / sq(*σ_φ_hk));
                    *μ_φ_hk = Normal::new(μ, σ).unwrap().sample(&mut rng);
                });
            });

        // sample component std dev parameters

        // accomulate sums of squares
        params.σ_φ.fill(0.0);
        Zip::from(&params.z) // for every cell
            .and(params.φ.outer_iter())
            .for_each(|&z_c, φ_c| {
                let z_c = z_c as usize;
                let μ_φ_z_c = params.μ_φ.row(z_c);
                let σ_φ_z_c = params.σ_φ.row_mut(z_c);
                Zip::from(μ_φ_z_c)
                    .and(σ_φ_z_c)
                    .and(φ_c)
                    .for_each(|μ_φ_z_ch, σ_φ_z_ch, φ_ch| {
                        *σ_φ_z_ch += sq(*φ_ch - *μ_φ_z_ch);
                    });
            });

        // sample from posterior gamma
        Zip::from(params.σ_φ.outer_iter_mut())
            .and(&params.component_population)
            .for_each(|σ_φ_h, &n_h| {
                Zip::from(σ_φ_h).for_each(|σ_φ_hk| {
                    let α = priors.α_σ_φ + n_h as f32 / 2.0;
                    let β = priors.β_σ_φ + *σ_φ_hk / 2.0;
                    // dbg!(α, β);
                    *σ_φ_hk = Gamma::new(α, β.recip())
                        .unwrap()
                        .sample(&mut rng)
                        .recip()
                        .sqrt();
                });
            });

        dbg!(&params.σ_φ);
        // params.σ_φ.fill(1e-1);

        // sample component assignments
        let ncomponents = params.ncomponents();
        Zip::from(&mut params.z) // for every cell
            .and(params.φ.outer_iter())
            .par_for_each(|z_c, φ_c| {
                let mut z_probs = params
                    .z_probs
                    .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                    .borrow_mut();

                // compute probabilities for each component
                for (μ_φ_k, σ_φ_k, z_probs_ck, π_c) in izip!(
                    params.μ_φ.outer_iter(),
                    params.σ_φ.outer_iter(),
                    z_probs.iter_mut(),
                    &params.π
                ) {
                    *z_probs_ck = *π_c as f64;

                    Zip::from(φ_c)
                        .and(μ_φ_k)
                        .and(σ_φ_k)
                        .for_each(|φ_ch, μ_φ_kh, σ_φ_kh| {
                            *z_probs_ck *= ((-sq(φ_ch - *μ_φ_kh) / (2.0 * sq(*σ_φ_kh))) as f64)
                                .exp()
                                / *σ_φ_kh as f64;
                        });
                }

                let z_probs_sum = z_probs.iter().sum::<f64>();

                // cumulative probabilities in-place
                z_probs.iter_mut().fold(0.0, |mut acc, x| {
                    acc += *x / z_probs_sum;
                    *x = acc;
                    acc
                });

                let rng = &mut thread_rng();
                let u = rng.gen::<f64>();
                *z_c = z_probs.partition_point(|x| *x < u) as u32;
            });

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
            params
                .π
                .extend(Dirichlet::new(&α).unwrap().sample(&mut rng).iter());
        }

        // dbg!(&params.σ_φ);
    }

    fn sample_φ(&mut self, params: &mut ModelParams) {
        Zip::from(params.φ.outer_iter_mut()) // for each cell
            .and(params.cell_latent_counts.outer_iter())
            .and(&params.cell_volume)
            // .par_for_each(|φ_c, x_c, v_c| {
            .for_each(|φ_c, x_c, v_c| {
                let mut rng = thread_rng();
                Zip::from(φ_c) // for each latent dim
                    .and(x_c)
                    .and(&params.r)
                    .and(&params.p)
                    .for_each(|φ_ck, x_ck, r_k, p_k| {

                        // *φ_ck = Gamma::new(*r_k + *x_ck as f32, *p_k).unwrap().sample(&mut rng);
                        // *φ_ck /= *v_c;

                        // // TODO: is this equivalent to just dividing by volume?
                        let scale = *p_k / (1.0 + (*v_c - 1.0) * *p_k);
                        *φ_ck = Gamma::new(*r_k + *x_ck as f32, scale).unwrap().sample(&mut rng);

                        // let naive_rate = *x_ck as f32 / *v_c;
                        // if *x_ck > 0 && (naive_rate / *φ_ck).ln().abs() > 3.0 {
                        // // if *x_ck == 0 && rng.gen::<f32>() < 0.001 {
                        //     dbg!((
                        //         *φ_ck,
                        //         *x_ck,
                        //         *x_ck as f32 / *v_c,
                        //         *p_k,
                        //         *r_k,
                        //     ));
                        // //     // panic!();
                        // }

                    });
            });
    }

    fn sample_θ(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        Zip::from(&mut params.θ)
            .and(&params.gene_latent_counts)
            .par_for_each(|θ_gk, x_gk| {
                let mut rng = thread_rng();
                let a = *x_gk as f32 + priors.α_θ;
                *θ_gk = Gamma::new(a, 1.0).unwrap().sample(&mut rng);
            });

        // normalize to get Dirichlet samples
        for mut θ_k in params.θ.columns_mut() {
            let θ_k_norm = θ_k.sum();
            θ_k /= θ_k_norm;
        }
    }

    fn sample_η(&mut self, _priors: &ModelPriors, params: &mut ModelParams) {
        Zip::from(&mut params.η)
            .and(params.ω.columns())
            .and(params.foreground_counts.axis_iter(Axis(1)))
            .and(&params.r)
            .and(params.θ.rows())
            // .par_for_each(|η_g, ω_g, x_g, r_g, θ_g| {
            .for_each(|η_g, ω_g, x_g, r_g, θ_g| {
                // TODO: prior precision parameter
                // TODO: is this σ or σ^2 ???
                let σ2_g = (1.0 + ω_g.sum()).recip();

                // dbg!(x_g.shape());
                // dbg!(ω_g.shape());
                // dbg!(params.cell_log_volume.shape());
                // dbg!(params.φ.rows().shape());

                let mut μ_g = 0.0;
                Zip::from(x_g.outer_iter())
                    .and(ω_g)
                    .and(&params.cell_log_volume)
                    .and(params.φ.rows())
                    .for_each(|x_cg, ω_cg, logv_c, φ_c| {
                        μ_g += (x_cg.sum() as f32 - r_g) / 2.0 - ω_cg * (θ_g.dot(&φ_c) + logv_c);
                    });
                μ_g *= σ2_g;

                // dbg!(μ_g, σ2_g);

                let mut rng = rand::thread_rng();
                *η_g = μ_g + σ2_g.sqrt() * randn(&mut rng);
                // dbg!(*η_g);
            });
    }

    fn compute_ψ(&mut self, params: &mut ModelParams) {
        // ψ = φ θ^T + 1 η^T + log(v) 1^T
        Zip::from(params.ψ.columns_mut()).for_each(|mut ψ_g| ψ_g.assign(&params.cell_log_volume));
        Zip::from(params.ψ.rows_mut()).for_each(|mut ψ_c| ψ_c.scaled_add(1.0, &params.η));
        general_mat_mul(1.0, &params.φ, &params.θ.t(), 1.0, &mut params.ψ);

        let mut ω_min = std::f32::MAX;
        let mut ω_max = std::f32::MIN;
        for ω_cg in &params.ω {
            ω_min = ω_min.min(*ω_cg);
            ω_max = ω_max.max(*ω_cg);
        }
        // dbg!(ω_min, ω_max);
    }

    // fn sample_r(&mut self, priors: &ModelPriors, params: &mut ModelParams, burnin: bool) {
    //     fn set_constant_dispersion(params: &mut ModelParams, dispersion: f32) {
    //         params.r.fill(dispersion);
    //     }

    //     if let Some(dispersion) = priors.dispersion {
    //         set_constant_dispersion(params, dispersion);
    //     } else if burnin && priors.burnin_dispersion.is_some() {
    //         let dispersion = priors.burnin_dispersion.unwrap();
    //         set_constant_dispersion(params, dispersion);
    //     } else {
    //         Zip::from(&mut params.r) // for each gene
    //             .and(params.ψ.columns())
    //             .and(params.foreground_counts.axis_iter(Axis(1)))
    //             .par_for_each(|r_g, ψ_g, x_g| {
    //                 let mut rng = thread_rng();

    //                 let l_g = x_g
    //                     .outer_iter()
    //                     .map(|x_cg| rand_crt(&mut rng, x_cg.sum() as u32, *r_g))
    //                     .sum::<u32>();

    //                 // summing log(1 - p_cg)
    //                 // let v = ψ_g
    //                 //     .iter()
    //                 //     .map(|ψ_cg| -ψ_cg - log1pf((-ψ_cg).exp()))
    //                 //     .sum::<f32>();

    //                 let mut v = 0.0;
    //                 // TODO: can I avoid having to do f64 math here?
    //                 for ψ_cg in ψ_g {
    //                     // if !(v + -ψ_cg - log1pf((-ψ_cg).exp())).is_finite() {
    //                     //     dbg!(v, ψ_cg, log1pf((-ψ_cg).exp()));
    //                     // }
    //                     // v += -ψ_cg - log1pf((-ψ_cg).exp());
    //                     // v += -ψ_cg - log1pf((-ψ_cg).exp());
    //                     v += -ψ_cg - log1p((-ψ_cg as f64).exp()) as f32;
    //                     assert!(v.is_finite());
    //                 }

    //                 // TODO: error here when v going to negative infinity somehow

    //                 let dist = Gamma::new(priors.e_r + l_g as f32, (params.h - v).recip());
    //                 if dist.is_err() {
    //                     dbg!(l_g, v, params.h);
    //                 }
    //                 let dist = dist.unwrap();
    //                 *r_g = dist.sample(&mut rng);
    //                 assert!(r_g.is_finite());
    //                 *r_g = r_g.max(2e-4);
    //             });

    //         let mut r_min = std::f32::MAX;
    //         let mut r_max = std::f32::MIN;
    //         for r_cg in &params.r {
    //             r_min = r_min.min(*r_cg);
    //             r_max = r_max.max(*r_cg);
    //         }
    //         // dbg!(r_min, r_max);
    //     }
    // }

    fn sample_nb_params(&mut self, priors: &ModelPriors, params: &mut ModelParams, burnin: bool) {
        let t0 = Instant::now();
        self.sample_ω(params);
        println!("  sample_ω: {:?}", t0.elapsed());

        let t0 = Instant::now();
        self.sample_φ(params);
        println!("  sample_φ: {:?}", t0.elapsed());
        dbg!(params.φ.mean());

        let t0 = Instant::now();
        self.sample_θ(priors, params);
        println!("  sample_θ: {:?}", t0.elapsed());
        dbg!(params.θ.mean());

        let t0 = Instant::now();
        self.sample_η(priors, params);
        println!("  sample_η: {:?}", t0.elapsed());
        dbg!(params.η.mean());

        let t0 = Instant::now();
        self.compute_ψ(params);
        println!("  compute_ψ: {:?}", t0.elapsed());
        dbg!(params.ψ.mean());

        let t0 = Instant::now();
        self.sample_r(priors, params);
        println!("  sample_r: {:?}", t0.elapsed());

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
        Zip::from(params.λ.rows_mut()) // for every cell
            .and(params.ψ.outer_iter())
            .and(params.foreground_counts.outer_iter())
            .and(&params.cell_volume)
            // .par_for_each(|mut λ_c, ψ_c, x_c, cell_volume| {
            .for_each(|mut λ_c, ψ_c, x_c, cell_volume| {
                let mut rng = thread_rng();
                for (λ_cg, &ψ_cg, x_cg, r_g) in izip!(&mut λ_c, ψ_c, x_c.outer_iter(), &params.r)
                {
                    let α = r_g + x_cg.sum() as f32;
                    let β = (-ψ_cg).exp() + 1.0;

                    // let α = 0.1 + x_cg.sum() as f32;
                    // let β: f32 = 1.0;

                    // From:
                    // (v_c τ_cg | -) ~ Gamma(α_g + x_cg, β_cg / v_c + 1)
                    // and
                    // ψ_cg = log(v_c / β_cg)
                    //
                    // let dist = Gamma::new(α, β.recip());
                    // if dist.is_err() {
                    //     dbg!(α, β, x_cg.sum(), ψ_cg);
                    // }

                    *λ_cg = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) / cell_volume;

                    // if x_cg.sum() == 1 {
                    //     let raw_est = x_cg.sum() as f32 / cell_volume;
                    //     dbg!(*λ_cg, r_g, ψ_cg, raw_est);
                    // }

                    assert!(λ_cg.is_finite());
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
                let α = priors.α_c + *c as f32;
                let β = priors.β_c + total_cell_volume;
                *λ = Gamma::new(α, β.recip()).unwrap().sample(&mut rng) as f32;
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

        // dbg!(&params.component_population);

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

    fn propose_eval_transcript_positions(
        &mut self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
    ) {
        // make proposals
        // let t0 = Instant::now();
        params
            .proposed_transcript_positions
            .par_iter_mut()
            // .zip(&params.transcript_positions)
            .zip(transcripts)
            .for_each(|(proposed_position, t)| {
                let mut rng = thread_rng();
                *proposed_position = (
                    t.x + priors.σ_diffusion_proposal
                        * rng.sample::<f32, StandardNormal>(StandardNormal),
                    t.y + priors.σ_diffusion_proposal
                        * rng.sample::<f32, StandardNormal>(StandardNormal),
                    (t.z + priors.σ_z_diffusion_proposal
                        * rng.sample::<f32, StandardNormal>(StandardNormal))
                    .min(priors.zmax)
                    .max(priors.zmin),
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
        params
            .accept_proposed_transcript_positions
            .par_iter_mut()
            .zip(&params.transcript_positions)
            .zip(&params.proposed_transcript_positions)
            .zip(transcripts)
            .enumerate()
            .for_each(
                |(i, (((accept, position), proposed_position), transcript))| {
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

                    let sq_dist_new = (proposed_position.0 - transcript.x).powi(2)
                        + (proposed_position.1 - transcript.y).powi(2);
                    let sq_dist_prev =
                        (position.0 - transcript.x).powi(2) + (position.1 - transcript.y).powi(2);
                    let z_sq_dist_new = (proposed_position.2 - transcript.z).powi(2);
                    let z_sq_dist_prev = (position.2 - transcript.z).powi(2);

                    let mut δ = 0.0;

                    // TODO: account for the possibility that sigma is 0

                    // prior on xy-diffusion distances
                    δ -= ((1.0 - priors.p_diffusion)
                        * normal_x2_pdf(priors.σ_diffusion_near, sq_dist_prev)
                        + priors.p_diffusion * normal_x2_pdf(priors.σ_diffusion_far, sq_dist_prev))
                    .ln();
                    δ += ((1.0 - priors.p_diffusion)
                        * normal_x2_pdf(priors.σ_diffusion_near, sq_dist_new)
                        + priors.p_diffusion * normal_x2_pdf(priors.σ_diffusion_far, sq_dist_new))
                    .ln();

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

                    let layer_prev =
                        ((position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                    let layer_prev = layer_prev.min(params.λ_bg.ncols() - 1);
                    let cell_prev = self.cell_at_position(*position);
                    let λ_prev = if cell_prev == BACKGROUND_CELL {
                        0.0
                    } else {
                        params.λ[[cell_prev as usize, gene]] + params.λ_c[gene]
                    } + params.λ_bg[[gene, layer_prev]];

                    let layer_new =
                        ((proposed_position.2 - params.z0) / params.layer_depth).max(0.0) as usize;
                    let layer_new = layer_new.min(params.λ_bg.ncols() - 1);
                    let cell_new = self.cell_at_position(*proposed_position);
                    let λ_new = if cell_new == BACKGROUND_CELL {
                        0.0
                    } else {
                        params.λ[[cell_new as usize, gene]] + params.λ_c[gene]
                    } + params.λ_bg[[gene, layer_new]];

                    let ln_λ_diff = λ_new.ln() - λ_prev.ln();
                    δ += ln_λ_diff;

                    // let cell_nuc = params.init_nuclear_cell_assignment[i];
                    // if cell_nuc != BACKGROUND_CELL {
                    //     if cell_nuc == cell_prev {
                    //         δ -= priors.nuclear_reassignment_1mlog_prob;
                    //     } else {
                    //         δ -= priors.nuclear_reassignment_log_prob;
                    //     }

                    //     if cell_nuc == cell_new {
                    //         δ += priors.nuclear_reassignment_1mlog_prob;
                    //     } else {
                    //         δ += priors.nuclear_reassignment_log_prob;
                    //     }
                    // }

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
                },
            );
        // println!("  Eval transcript position proposals: {:?}", t0.elapsed());

        // let naccepted = params.accept_proposed_transcript_positions.iter().map(|&x| x as u32).sum::<u32>();
        // let prop_accepted = naccepted as f32 / params.accept_proposed_transcript_positions.len() as f32;
        // println!("  Accepted {}% of transcript repo proposals", 100.0 * prop_accepted);
    }

    fn sample_transcript_positions(
        &mut self,
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        uncertainty: &mut Option<&mut UncertaintyTracker>,
    ) {
        self.propose_eval_transcript_positions(priors, params, transcripts);

        // Update position and compute cell and layer changes for updates
        params
            .transcript_position_updates
            .par_iter_mut()
            .zip(&mut params.transcript_positions)
            .zip(&params.proposed_transcript_positions)
            .zip(&params.cell_assignments)
            .zip(&params.accept_proposed_transcript_positions)
            .for_each(
                |((((update, position), proposed_position), cell_prev), &accept)| {
                    if accept {
                        let layer_prev = ((position.2 - params.z0) / params.layer_depth) as usize;
                        let layer_prev = layer_prev.min(params.λ_bg.ncols() - 1);

                        let cell_new = self.cell_at_position(*proposed_position);
                        let layer_new =
                            ((proposed_position.2 - params.z0) / params.layer_depth) as usize;
                        let layer_new = layer_new.min(params.λ_bg.ncols() - 1);

                        // assert!(self.cell_at_position(*position) == *cell_prev);

                        *update = (*cell_prev, cell_new, layer_prev as u32, layer_new as u32);
                        *position = *proposed_position;
                    }
                },
            );

        // Update counts and cell_population
        params
            .transcript_position_updates
            .iter()
            .zip(transcripts)
            .zip(&params.accept_proposed_transcript_positions)
            .zip(&mut params.cell_assignments)
            .zip(&mut params.cell_assignment_time)
            .enumerate()
            .for_each(
                |(
                    i,
                    ((((update, transcript), &accept), cell_assignment), cell_assignment_time),
                )| {
                    let (cell_prev, cell_new, layer_prev, layer_new) = *update;
                    if accept {
                        let gene = transcript.gene as usize;
                        if cell_prev != BACKGROUND_CELL {
                            assert!(
                                params.counts[[gene, cell_prev as usize, layer_prev as usize]] > 0
                            );
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
                },
            );

        self.update_transcript_positions(
            &params.accept_proposed_transcript_positions,
            &params.transcript_positions,
        );
    }
}
