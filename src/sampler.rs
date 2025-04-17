mod math;
pub mod paramsampler;
mod polyagamma;
pub mod runvec;
mod sampleset;
mod shardedvec;
mod sparsemat;
pub mod transcripts;
pub mod voxelcheckerboard;
pub mod voxelsampler;

use clustering::kmeans;
use math::randn;
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{s, Array1, Array2, Axis, Zip};
use num::traits::Zero;
use rand::{rng, Rng};
use rayon::iter::IntoParallelRefIterator;
use shardedvec::ShardedVec;
use sparsemat::SparseMat;
use std::cell::RefCell;
use std::ops::{Add, AddAssign, SubAssign};
use thread_local::ThreadLocal;
use voxelcheckerboard::VoxelCheckerboard;

// Shard size used for sharded vectors and matrices
const CELL_SHARDSIZE: usize = 256;
const GENE_SHARDSIZE: usize = 16;

// Model prior parameters.
#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub dispersion: Option<f32>,
    pub burnin_dispersion: Option<f32>,

    pub use_cell_scales: bool,

    pub min_cell_volume: f32,

    // params for normal prior
    pub μ_μ_volume: f32,
    pub σ_μ_volume: f32,

    // params for inverse-gamma prior
    pub α_σ_volume: f32,
    pub β_σ_volume: f32,

    pub use_factorization: bool,

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

    // prior precision on effective log cell volume
    pub τv: f32,

    // whether to check if voxel updates break local connectivity
    pub enforce_connectivity: bool,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CountMatRowKey {
    gene: u32,
    layer: u32,
}

impl CountMatRowKey {
    pub fn new(gene: u32, layer: u32) -> Self {
        CountMatRowKey { gene, layer }
    }
}

impl Add for CountMatRowKey {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        CountMatRowKey {
            gene: self.gene + other.gene,
            layer: self.layer + other.layer,
        }
    }
}

impl Zero for CountMatRowKey {
    fn zero() -> Self {
        CountMatRowKey { gene: 0, layer: 0 }
    }

    fn is_zero(&self) -> bool {
        self.gene == 0 && self.layer == 0
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

    // [ncells] cell volume in exposed voxel surface count
    pub cell_surface_area: ShardedVec<u32>,

    // [ncells] cell volume in cubic microns
    pub log_cell_volume: Array1<f32>,

    // [ncells] cell_volume * cell_scale
    pub effective_cell_volume: Array1<f32>,

    // [ncells] per-cell "effective" volume scaling factor
    pub cell_scale: Array1<f32>,

    // [ncells, (ngenes x nlayers)] transcripts counts, split into total
    // transcript count in each cell and gene and layer.
    counts: SparseMat<u32, CountMatRowKey>,

    // [ncells, ngenes] sparse matrix of just foreground (non-noise) counts
    foreground_counts: SparseMat<u32, u32>,

    // [nlayers, ngenes] background transcripts counts
    unassigned_counts: Vec<ShardedVec<u32>>,

    // [nlayers, ngenes]
    background_counts: Vec<ShardedVec<u32>>,

    // [ncells, nhidden]
    pub cell_latent_counts: SparseMat<u32, u32>,

    // [ngenes, nhidden]
    pub gene_latent_counts: Array2<u32>,

    // Thread local [ngenes, nhidden] matrices for accumulation
    pub gene_latent_counts_tl: ThreadLocal<RefCell<Array2<u32>>>,

    // [nhidden]
    pub latent_counts: Array1<u32>,

    // [nhidden] thread local storage for sampling latent counts
    pub multinomial_rates: ThreadLocal<RefCell<Array1<f32>>>,
    pub multinomial_sample: ThreadLocal<RefCell<Array1<u32>>>,

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

    // [ngenes, nlayers] background rate: rate at which halucinate transcripts
    // across the entire layer
    pub λ_bg: Array2<f32>,
    pub logλ_bg: Array2<f32>,

    // Size of the upper block of θ that is the identity matrix
    nunfactored: usize,

    // volume of a single voxel (in μm)
    pub voxel_volume: f32,

    // volume of a single voxel layer (in μm)
    layer_volume: f32,

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
        layer_volume: f32,
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
        let mut cell_surface_area = ShardedVec::zeros(ncells, CELL_SHARDSIZE);
        voxels.compute_cell_volume_surface_area(&mut cell_voxel_count, &mut cell_surface_area);
        let voxel_volume = voxels.voxel_volume;

        let effective_cell_volume = cell_voxel_count
            .iter()
            .map(|count| count as f32 * voxels.voxel_volume)
            .collect::<Array1<f32>>();

        let log_cell_volume = effective_cell_volume.map(|v| v.ln());
        let cell_scale = Array1::<f32>::ones(ncells);

        let mut counts = SparseMat::zeros(
            ncells,
            CountMatRowKey::new(ngenes as u32, nlayers as u32),
            CELL_SHARDSIZE,
        );
        voxels.compute_counts(&mut counts);

        let foreground_counts = SparseMat::zeros(ncells, ngenes as u32, CELL_SHARDSIZE);
        let unassigned_counts = (0..nlayers)
            .map(|_layer| ShardedVec::zeros(ngenes, GENE_SHARDSIZE))
            .collect::<Vec<_>>();
        let background_counts = (0..nlayers)
            .map(|_layer| ShardedVec::zeros(ngenes, GENE_SHARDSIZE))
            .collect::<Vec<_>>();

        let cell_latent_counts = SparseMat::zeros(ncells, nhidden as u32, CELL_SHARDSIZE);
        let gene_latent_counts = Array2::<u32>::zeros((ngenes, nhidden));
        let gene_latent_counts_tl = ThreadLocal::new();
        let latent_counts = Array1::<u32>::zeros(nhidden);
        let multinomial_rates = ThreadLocal::new();
        let multinomial_sample = ThreadLocal::new();
        let z_probs = ThreadLocal::new();
        let z = initial_component_assignments(&counts, ncomponents);

        let π = Array1::<f32>::zeros(ncomponents);
        let log_π = Array1::<f32>::zeros(ncomponents);
        let component_population = Array1::<u32>::zeros(ncomponents);
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

        let lφ = Array2::<u32>::zeros((ncells, nhidden));
        let ωφ = Array2::<f32>::zeros((ncells, nhidden));
        let rφ = Array2::<f32>::from_elem((ncomponents, nhidden), 1.0);
        let lgamma_rφ = Array2::<f32>::zeros((ncomponents, nhidden));
        let sφ = Array2::<f32>::from_elem((ncomponents, nhidden), 1.0);
        let μ_sφ = Array2::<f32>::zeros((ncomponents, nhidden));
        let τ_sφ = Array2::<f32>::zeros((ncomponents, nhidden));
        let sφ_work_tl = ThreadLocal::new();

        let mut θ = Array2::<f32>::zeros((ngenes, nhidden));
        θ.slice_mut(s![0..nunfactored, 0..nunfactored]).fill(1.0);
        θ.slice_mut(s![nunfactored.., nunfactored..])
            .mapv_inplace(|_v| randn(&mut rng).exp());
        let mut θksum = Array1::<f32>::zeros(nhidden); // TODO: make have to initialize this
        Zip::from(&mut θksum)
            .and(θ.axis_iter(Axis(1)))
            .for_each(|θksum, θ_k| {
                *θksum = θ_k.sum();
            });

        let λ_bg = Array2::<f32>::zeros((ngenes, nlayers));
        let logλ_bg = Array2::<f32>::zeros((ngenes, nlayers));

        let t = 0;

        ModelParams {
            cell_voxel_count,
            cell_surface_area,
            log_cell_volume,
            effective_cell_volume,
            cell_scale,
            counts,
            foreground_counts,
            unassigned_counts,
            background_counts,
            cell_latent_counts,
            gene_latent_counts,
            gene_latent_counts_tl,
            latent_counts,
            multinomial_rates,
            multinomial_sample,
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
            layer_volume,
            t,
        }
    }

    // Compute the Poisson rate for cell and gene pair.
    pub fn λ(&self, cell: usize, gene: usize) -> f32 {
        let φ_cell = self.φ.row(cell);
        if gene < self.nunfactored {
            φ_cell[gene]
        } else {
            let θ_g = self.θ.row(gene);
            φ_cell.dot(&θ_g)
        }
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
}

fn initial_component_assignments(
    counts: &SparseMat<u32, CountMatRowKey>,
    ncomponents: usize,
) -> Array1<u32> {
    let (
        ncells,
        CountMatRowKey {
            gene: ngenes,
            layer: _nlayers,
        },
    ) = counts.shape();
    let ngenes = ngenes as usize;

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
            },
            count,
        ) in counts_c.read().iter_nonzeros()
        {
            expr_row[gene as usize] += count as f32;
        }

        // normalize
        let row_sum = expr_row.sum();
        expr_row.mapv_inplace(|x| (NORM_CONSTANT * x / row_sum).ln_1p());

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
