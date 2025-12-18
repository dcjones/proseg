pub mod connectivity;
pub mod csrmat;
mod math;
mod multinomial;
pub mod onlinestats;
pub mod paramsampler;
mod polyagamma;
mod polygons;
pub mod runvec;
mod sampleset;
mod shardedvec;
pub mod sparsevec;
pub mod transcriptrepo;
pub mod transcripts;
pub mod voxelcheckerboard;
pub mod voxelsampler;

use clustering::kmeans;
use csrmat::CSRMat;
use csrmat::Increment;

use itertools::izip;
use math::randn;
use multinomial::Multinomial;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, Array2, Array3, Axis, Zip, s};
use num::traits::Zero;
use onlinestats::CountMeanEstimator;
use rand::rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use shardedvec::ShardedVec;
use std::cell::RefCell;
use std::ops::{Add, AddAssign};
use thread_local::ThreadLocal;
use transcripts::BACKGROUND_CELL;
use voxelcheckerboard::VoxelCheckerboard;

// Shard size used for sharded vectors and matrices
const CELL_SHARDSIZE: usize = 256;
const GENE_SHARDSIZE: usize = 16;

const RAYON_CELL_MIN_LEN: usize = 32;

// Model prior parameters.
#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub dispersion: Option<f32>,
    pub burnin_dispersion: Option<f32>,

    pub use_cell_scales: bool,
    pub unmodeled_fixed_cells: bool,

    // pub min_cell_volume: f32,

    // params for normal prior
    pub μ_μ_volume: f32,
    pub σ_μ_volume: f32,

    // params for inverse-gamma prior
    pub α_σ_volume: f32,
    pub β_σ_volume: f32,

    pub use_factorization: bool,
    pub enforce_connectivity: bool,

    // dirichlet prior on θ
    pub αθ: f32,

    // gamma prior on rφ
    pub eφ: f32,
    pub fφ: f32,

    // log-normal prior on sφ
    pub μφ: f32,
    pub τφ: f32,

    // gamma prior for background rates
    pub α_bg: f32,
    pub β_bg: f32,

    pub σ_iiq: f32,

    // // scaling factor for circle perimeters
    // pub perimeter_eta: f32,
    // pub perimeter_bound: f32,

    // pub nuclear_reassignment_log_prob: f32,
    // pub nuclear_reassignment_1mlog_prob: f32,

    // pub prior_seg_reassignment_log_prob: f32,
    // pub prior_seg_reassignment_1mlog_prob: f32,

    // mixture between diffusion prior components
    pub use_diffusion_model: bool,
    pub p_diffusion: f32,

    // pub σ_z_diffusion_proposal: f32,
    pub σ_xy_diffusion_near: f32,
    pub σ_xy_diffusion_far: f32,
    pub σ_z_diffusion: f32,
    pub σ_xy_diffusion_proposal: f32,
    pub σ_z_diffusion_proposal: f32,

    // prior precision on effective log cell volume
    pub τv: f32,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CountMatRowKey {
    gene: u32,
    layer: u32,
    density: u8,
}

impl CountMatRowKey {
    pub fn new(gene: u32, layer: u32, density: u8) -> Self {
        CountMatRowKey {
            gene,
            layer,
            density,
        }
    }
}

impl Add for CountMatRowKey {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        CountMatRowKey {
            gene: self.gene + other.gene,
            layer: self.layer + other.layer,
            density: self.density + other.density,
        }
    }
}

impl AddAssign for CountMatRowKey {
    fn add_assign(&mut self, other: Self) {
        self.gene += other.gene;
        self.layer += other.layer;
        self.density += other.density;
    }
}

impl Zero for CountMatRowKey {
    fn zero() -> Self {
        CountMatRowKey {
            gene: 0,
            layer: 0,
            density: 0,
        }
    }

    fn is_zero(&self) -> bool {
        self.gene == 0 && self.layer == 0 && self.density == 0
    }
}

impl Increment for CountMatRowKey {
    fn inc(&self, bound: CountMatRowKey) -> CountMatRowKey {
        // treating this as three digits, incrementing density then layer then gene
        if self.density + 1 > bound.density {
            if self.layer + 1 > bound.layer {
                CountMatRowKey {
                    gene: self.gene + 1,
                    layer: 0,
                    density: 0,
                }
            } else {
                CountMatRowKey {
                    gene: self.gene,
                    layer: self.layer + 1,
                    density: 0,
                }
            }
        } else {
            CountMatRowKey {
                gene: self.gene,
                layer: self.layer,
                density: self.density + 1,
            }
        }
    }
}

// In general, subscripts indicate dimension:
//   t: component
//   k: latent dim
//   g: gene
//   c: cell
#[allow(non_snake_case)]
pub struct ModelParams {
    // [ncells] cell volume in voxel count
    pub cell_voxel_count: ShardedVec<u32>,

    // [nlayers, ncells] cell volume in exposed voxel surface count
    pub cell_layer_voxel_count: Vec<ShardedVec<u32>>,

    // [nlayers, ncells] cell volume in exposed voxel surface count
    pub cell_layer_surface_area: Vec<ShardedVec<u32>>,

    // [ncells] cell volume in cubic microns
    pub log_cell_volume: Array1<f32>,

    // [ncells] cell_volume * cell_scale
    pub effective_cell_volume: Array1<f32>,

    // [ncells] per-cell "effective" volume scaling factor
    pub cell_scale: Array1<f32>,

    // [ncells, (ngenes x nlayers)] transcripts counts, split into total
    // transcript count in each cell and gene and layer.
    counts: CSRMat<CountMatRowKey, u32>,

    // [ncells, ngenes] sparse matrix of just foreground (non-noise) counts
    pub foreground_counts: CSRMat<u32, u32>,

    // [ncells, ngenes] upper and lower credible intervals for cell-by-gene counts
    // foreground_counts_lower: CountQuantileEstimator,
    // foreground_counts_upper: CountQuantileEstimator,
    pub foreground_counts_mean: CountMeanEstimator,

    // [ncells, ncells] sparse matrix recording the number of times the sampler
    // moved transcripts between pairs of cells.
    pub transition_counts: CSRMat<u32, u32>,

    // [density_nbins, nlayers, ngenes] background transcripts counts
    unassigned_counts: Vec<Vec<ShardedVec<u32>>>,

    // [density_nbins, nlayers, ngenes]
    background_counts: Vec<Vec<ShardedVec<u32>>>,

    // [ncells, nhidden]
    pub cell_latent_counts: CSRMat<u32, u32>,

    // [ngenes, nhidden]
    pub gene_latent_counts: Array2<u32>,

    // Thread local [ngenes, nhidden] matrices for accumulation
    pub gene_latent_counts_tl: ThreadLocal<RefCell<Array2<u32>>>,

    // [nhidden]
    pub latent_counts: Array1<u32>,

    // [nhidden] thread local storage for sampling latent counts
    pub multinomials: ThreadLocal<RefCell<Multinomial<f32>>>,

    // [ncells, ncomponents] space for sampling component assignments
    pub z_probs: ThreadLocal<RefCell<Vec<f64>>>,

    // [ncells] assignment of cells to components
    pub z: Array1<u32>,

    // [ncomponents] component probabilities
    pub π: Array1<f32>,
    pub log_π: Array1<f32>,

    // [ncomponents] number of cells assigned to each component
    component_population: Array1<u32>,

    // [ncomponents] total volume of each component
    component_volume: Array1<f32>,

    // [ncomponents, nhidden]
    component_latent_counts: Array2<u32>,

    // [ncomponents]
    μ_volume: Array1<f32>, // volume dist mean param by component
    σ_volume: Array1<f32>, // volume dist std param by component

    // [ncells, nhidden]: cell ψ parameter in the latent space
    pub φ: Array2<f32>,

    // [nhidden]: precompute φ_k.dot(cell_volume)
    φ_v_dot: Array1<f32>,

    // [ncells]: precompute φ_c.dot(θksum)
    pub φ_θksum_dot: Array1<f32>,

    // [ncells, nhidden] aux CRT variables for sampling rφ
    pub lφ: Array2<u32>,

    // [ncells, nhidden] aux PolyaGamma variables for sampling sφ
    pub ωφ: Array2<f32>,

    // [ncomponents, nhidden] φ gamma shape parameters
    pub rφ: Array2<f32>,

    // [ncomponents, nhidden]
    // for precomputing lgamma(rφ)
    lgamma_rφ: Array2<f32>,

    // [ncomponents, nhidden] φ gamma scale parameters
    pub sφ: Array2<f32>,

    // [ncomponents, nhidden]
    // posterior params for sampling sφ
    μ_sφ: Array2<f32>,
    τ_sφ: Array2<f32>,

    // [ncomponent, nhidden] thread local temporary matrices for computing μ_sφ and τ_sφ in parallel
    sφ_work_tl: ThreadLocal<RefCell<Array2<f32>>>,

    // [ngenes, nhidden]: gene loadings in the latent space
    pub θ: Array2<f32>,

    // [nhidden]: Sums across the first axis of θ
    pub θksum: Array1<f32>,

    // [ngenes, nlayers, density_nbins] background rate: rate at which halucinate transcripts
    // across the entire layer
    pub λ_bg: Array3<f32>,
    pub logλ_bg: Array3<f32>,

    // Size of the upper block of θ that is the identity matrix
    nunfactored: usize,

    // volume of a single voxel (in μm)
    pub voxel_volume: f32,

    // volume (in μm) of a particular subset of the sample, partitioned by layer
    // and transcript density
    // [density_nbins]
    background_region_volume: Array1<f32>,

    // [ncells] True where morphology updates are prohibited.
    pub frozen_cells: Vec<bool>,

    // time, which is incremented after every iteration
    t: u32,
}

impl ModelParams {
    pub fn new(
        voxels: &VoxelCheckerboard,
        priors: &ModelPriors,
        nhidden: usize,
        nunfactored: usize,
        ncomponents: usize,
        density_nbins: usize,
    ) -> ModelParams {
        let ncells = voxels.ncells;
        let ngenes = voxels.ngenes;
        let nlayers = (voxels.kmax + 1) as usize;
        let (nhidden, nunfactored) = if priors.use_factorization {
            (nhidden + nunfactored, nunfactored)
        } else {
            (ngenes, ngenes)
        };

        let mut cell_voxel_count = ShardedVec::zeros(ncells, CELL_SHARDSIZE);
        let mut cell_layer_voxel_count = Vec::new();
        let mut cell_layer_surface_area = Vec::new();
        for _ in 0..nlayers {
            cell_layer_voxel_count.push(ShardedVec::zeros(ncells, CELL_SHARDSIZE));
            cell_layer_surface_area.push(ShardedVec::zeros(ncells, CELL_SHARDSIZE));
        }

        voxels.compute_cell_volume_surface_area(
            &mut cell_voxel_count,
            &mut cell_layer_voxel_count,
            &mut cell_layer_surface_area,
        );
        let voxel_volume = voxels.voxel_volume;

        let effective_cell_volume = cell_voxel_count
            .iter()
            .map(|count| count as f32 * voxels.voxel_volume)
            .collect::<Array1<f32>>();

        let log_cell_volume = effective_cell_volume.map(|v| v.ln());
        let cell_scale = Array1::<f32>::ones(ncells);

        let mut counts = CSRMat::zeros(
            ncells,
            CountMatRowKey::new(
                ngenes as u32 - 1,
                nlayers as u32 - 1,
                density_nbins as u8 - 1,
            ),
        );
        let mut unassigned_counts = (0..density_nbins)
            .map(|_density| {
                (0..nlayers)
                    .map(|_layer| ShardedVec::zeros(ngenes, GENE_SHARDSIZE))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        voxels.compute_counts(&mut counts, &mut unassigned_counts);

        let foreground_counts = CSRMat::zeros(ncells, ngenes as u32 - 1);

        // Initializing with everything assigned as foreground
        counts
            .par_rows()
            .zip(foreground_counts.par_rows())
            .with_min_len(RAYON_CELL_MIN_LEN)
            .for_each_init(rng, |_rng, (row, foreground_row)| {
                let mut foreground_row = foreground_row.write();
                for (gene_layer, count) in row.read().iter_nonzeros() {
                    foreground_row.add(gene_layer.gene, count);
                }
            });

        // let foreground_counts_lower =
        //     CountQuantileEstimator::new(ncells, ngenes, 0.05, CELL_SHARDSIZE);
        // let foreground_counts_upper =
        //     CountQuantileEstimator::new(ncells, ngenes, 0.95, CELL_SHARDSIZE);
        let foreground_counts_mean = CountMeanEstimator::new(ncells, ngenes, CELL_SHARDSIZE);
        let background_counts = (0..density_nbins)
            .map(|_density| {
                (0..nlayers)
                    .map(|_layer| ShardedVec::zeros(ngenes, GENE_SHARDSIZE))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let cell_latent_counts = CSRMat::zeros(ncells, nhidden as u32 - 1);
        let gene_latent_counts = Array2::<u32>::zeros((ngenes, nhidden));
        let gene_latent_counts_tl = ThreadLocal::new();
        let latent_counts = Array1::<u32>::zeros(nhidden);
        let multinomials = ThreadLocal::new();
        let z_probs = ThreadLocal::new();
        let z = initial_component_assignments(&counts, ncomponents);

        let π = Array1::<f32>::zeros(ncomponents);
        let log_π = Array1::<f32>::zeros(ncomponents);
        let mut component_population = Array1::<u32>::zeros(ncomponents);
        for z_c in z.iter() {
            component_population[*z_c as usize] += 1;
        }
        let component_volume = Array1::<f32>::zeros(ncomponents);
        let component_latent_counts = Array2::<u32>::zeros((ncomponents, nhidden));
        let μ_volume = Array1::<f32>::from_elem(ncomponents, priors.μ_μ_volume);
        let σ_volume = Array1::<f32>::from_elem(ncomponents, priors.σ_μ_volume);

        let mut rng = rng();
        let φ = Array2::<f32>::from_shape_simple_fn((ncells, nhidden), || randn(&mut rng).exp());
        let mut φ_v_dot = Array1::<f32>::zeros(nhidden); // TODO: may have initialize this
        Zip::from(&mut φ_v_dot)
            .and(φ.axis_iter(Axis(1)))
            .for_each(|φ_v_dot_k, φ_k| {
                *φ_v_dot_k = φ_k.dot(&effective_cell_volume);
            });

        let mut φ_θksum_dot = Array1::<f32>::zeros(ncells);

        let lφ = Array2::<u32>::zeros((ncells, nhidden));
        let ωφ = Array2::<f32>::zeros((ncells, nhidden));
        let rφ = Array2::<f32>::from_elem((ncomponents, nhidden), 1.0);
        let lgamma_rφ = Array2::<f32>::zeros((ncomponents, nhidden));
        let sφ = Array2::<f32>::from_elem((ncomponents, nhidden), 1.0);
        let μ_sφ = Array2::<f32>::zeros((ncomponents, nhidden));
        let τ_sφ = Array2::<f32>::zeros((ncomponents, nhidden));
        let sφ_work_tl = ThreadLocal::new();

        let mut θ = Array2::<f32>::zeros((ngenes, nhidden));
        θ.slice_mut(s![0..nunfactored, 0..nunfactored])
            .diag_mut()
            .fill(1.0);
        θ.slice_mut(s![nunfactored.., nunfactored..])
            .mapv_inplace(|_v| randn(&mut rng).exp());
        let mut θksum = Array1::<f32>::zeros(nhidden); // TODO: make have to initialize this
        Zip::from(&mut θksum)
            .and(θ.axis_iter(Axis(1)))
            .for_each(|θksum, θ_k| {
                *θksum = θ_k.sum();
            });

        Zip::from(&mut φ_θksum_dot)
            .and(φ.rows())
            .for_each(|dot, φ_c| {
                *dot = φ_c.dot(&θksum);
            });

        let λ_bg = Array3::<f32>::zeros((ngenes, nlayers, density_nbins));
        let logλ_bg = Array3::<f32>::zeros((ngenes, nlayers, density_nbins));

        // Initialize this here to the layer volume, and voxelcheckerboard will
        // update it when it computes density values.
        let mut background_region_volume = Array1::zeros(density_nbins);
        voxels.compute_background_region_volumes(&mut background_region_volume);

        let transition_counts = CSRMat::zeros(ncells, ncells as u32 - 1);

        let frozen_cells = voxels.frozen_cells.clone();

        let t = 0;

        ModelParams {
            cell_voxel_count,
            cell_layer_voxel_count,
            cell_layer_surface_area,
            log_cell_volume,
            effective_cell_volume,
            cell_scale,
            counts,
            foreground_counts,
            // foreground_counts_lower,
            // foreground_counts_upper,
            transition_counts,
            foreground_counts_mean,
            unassigned_counts,
            background_counts,
            cell_latent_counts,
            gene_latent_counts,
            gene_latent_counts_tl,
            latent_counts,
            multinomials,
            z_probs,
            z,
            π,
            log_π,
            component_population,
            component_volume,
            component_latent_counts,
            μ_volume,
            σ_volume,
            φ,
            φ_v_dot,
            φ_θksum_dot,
            lφ,
            ωφ,
            rφ,
            lgamma_rφ,
            sφ,
            μ_sφ,
            τ_sφ,
            sφ_work_tl,
            θ,
            θksum,
            λ_bg,
            logλ_bg,
            nunfactored,
            voxel_volume,
            background_region_volume,
            frozen_cells,
            t,
        }
    }

    pub fn update_phi_theta_dot(&mut self) {
        Zip::from(&mut self.φ_θksum_dot)
            .and(self.φ.rows())
            .for_each(|dot, φ_c| {
                *dot = φ_c.dot(&self.θksum);
            });
    }

    // Compute the Poisson rate for cell and gene pair.
    pub fn λ(&self, cell: usize, gene: usize) -> f32 {
        if cell as u32 == BACKGROUND_CELL {
            return 0.0;
        }

        if gene < self.nunfactored {
            return self.φ[[cell, gene]];
        }

        let φ_c = self.φ.row(cell);
        let θ_g = self.θ.row(gene);
        φ_c.dot(&θ_g)
    }

    pub fn log_likelihood(&self, _priors: &ModelPriors) -> f32 {
        let mut ll = self
            .foreground_counts
            .par_rows()
            .enumerate()
            .map(|(c, x_c)| {
                let v_c = self.effective_cell_volume[c];
                let x_c = x_c.read();
                let mut accum_c = 0.0;

                for (g, x_cg) in x_c.iter_nonzeros() {
                    if x_cg > 0 {
                        let λ_cg = self.λ(c, g as usize);
                        accum_c += (x_cg as f32) * λ_cg.ln();
                    }
                }
                accum_c - v_c * self.φ_θksum_dot[c]
            })
            .sum();

        ll += self
            .background_counts
            .par_iter()
            .zip(self.λ_bg.axis_iter(Axis(2)))
            .zip(self.background_region_volume.as_slice().unwrap())
            .map(|((x_d, λ_d), &v_d)| {
                let mut accum_l = 0.0;
                for (x_ld, λ_ld) in izip!(x_d, λ_d.axis_iter(Axis(1))) {
                    for (x_lg, &λ_lg) in x_ld.iter().zip(λ_ld) {
                        accum_l += (x_lg as f32) * λ_lg.ln() - λ_lg * v_d;
                    }
                }
                accum_l
            })
            .sum::<f32>();

        // TODO: Do we want to include other parameter probabilities?

        ll
    }

    pub fn nassigned(&self) -> usize {
        self.counts.sum() as usize
    }

    pub fn nforeground(&self) -> usize {
        self.foreground_counts.sum() as usize
    }

    pub fn ncomponents(&self) -> usize {
        self.π.shape()[0]
    }

    pub fn ncells(&self) -> usize {
        self.φ.shape()[0]
    }

    pub fn ngenes(&self) -> usize {
        self.θ.shape()[0]
    }

    pub fn nhidden(&self) -> usize {
        self.θ.shape()[1]
    }

    // pub fn nlayers(&self) -> usize {
    //     self.background_counts.len()
    // }

    pub fn check_consistency(&self, voxels: &VoxelCheckerboard) {
        let ncells = voxels.ncells;
        let ngenes = voxels.ngenes;
        let nlayers = (voxels.kmax + 1) as usize;
        let density_nbins = voxels.density_nbins;

        let mut cell_voxel_count = ShardedVec::zeros(ncells, CELL_SHARDSIZE);
        let mut cell_layer_voxel_count = Vec::new();
        let mut cell_layer_surface_area = Vec::new();
        for _ in 0..nlayers {
            cell_layer_voxel_count.push(ShardedVec::zeros(ncells, CELL_SHARDSIZE));
            cell_layer_surface_area.push(ShardedVec::zeros(ncells, CELL_SHARDSIZE));
        }
        voxels.compute_cell_volume_surface_area(
            &mut cell_voxel_count,
            &mut cell_layer_voxel_count,
            &mut cell_layer_surface_area,
        );

        assert!(self.cell_voxel_count == cell_voxel_count);
        assert!(self.cell_layer_voxel_count == cell_layer_voxel_count);
        assert!(self.cell_layer_surface_area == cell_layer_surface_area);

        let mut counts = CSRMat::zeros(
            ncells,
            CountMatRowKey::new(
                ngenes as u32 - 1,
                nlayers as u32 - 1,
                density_nbins as u8 - 1,
            ),
        );
        let mut unassigned_counts = (0..density_nbins)
            .map(|_density| {
                (0..nlayers)
                    .map(|_layer| ShardedVec::zeros(ngenes, GENE_SHARDSIZE))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        voxels.compute_counts(&mut counts, &mut unassigned_counts);
        assert!(self.counts == counts);
        assert!(self.unassigned_counts == unassigned_counts);
    }

    pub fn total_cell_surface_area(&self) -> Array1<u32> {
        let mut total_surface_area = Array1::<u32>::zeros(self.ncells());
        for sa_k in self.cell_layer_surface_area.iter() {
            for (tsa_c, sa_kc) in izip!(total_surface_area.iter_mut(), sa_k.iter()) {
                *tsa_c += sa_kc;
            }
        }

        total_surface_area
    }
}

fn initial_component_assignments(
    counts: &CSRMat<CountMatRowKey, u32>,
    ncomponents: usize,
) -> Array1<u32> {
    let (ncells, j_bound) = counts.shape();
    let ngenes = j_bound.gene as usize + 1;

    const EMBEDDING_DIM: usize = 25;
    let mut rng = rng();

    // sample random projection
    let mut proj = Array2::<f32>::from_shape_simple_fn((EMBEDDING_DIM, ngenes), || {
        (EMBEDDING_DIM as f32).recip().sqrt() * randn(&mut rng)
    });
    for mut proj_i in proj.rows_mut() {
        let norm = proj_i.map(|&proj_ij| proj_ij * proj_ij).sum().sqrt();
        proj_i.map_inplace(|proj_ij| *proj_ij /= norm);
    }

    // normalize counts and project to low dimensionality
    let mut embedding = Array2::<f32>::zeros((ncells, EMBEDDING_DIM));
    const NORM_CONSTANT: f32 = 1e2;
    let expr_row = ThreadLocal::new();
    // for each cell
    Zip::indexed(embedding.rows_mut()).par_for_each(|c, mut embedding_c| {
        let mut expr_row = expr_row
            .get_or(|| RefCell::new(Array1::<f32>::zeros(ngenes)))
            .borrow_mut();
        expr_row.fill(0.0);

        // marginalize counts
        let counts_c = counts.row(c);
        for (
            CountMatRowKey {
                gene,
                layer: _layer,
                density: _density,
            },
            count,
        ) in counts_c.read().iter_nonzeros()
        {
            expr_row[gene as usize] += count as f32;
        }

        // normalize
        let row_sum = expr_row.sum();
        if row_sum == 0.0 {
            let c = (NORM_CONSTANT / ngenes as f32).ln_1p();
            expr_row.fill(c);
        } else {
            expr_row.mapv_inplace(|x| (NORM_CONSTANT * x / row_sum).ln_1p());
        }

        // apply projection
        general_mat_vec_mul(1.0, &proj, &expr_row, 0.0, &mut embedding_c);
    });

    // kmeans
    let embedding: Vec<Vec<f32>> = embedding
        .rows()
        .into_iter()
        .map(|row| row.iter().cloned().collect())
        .collect();

    const KMEANS_ITERATIONS: usize = 500;
    let kmeans_results = kmeans(ncomponents, &embedding, KMEANS_ITERATIONS);
    let z: Array1<u32> = kmeans_results
        .membership
        .iter()
        .map(|z_c| *z_c as u32)
        .collect();

    z
}
