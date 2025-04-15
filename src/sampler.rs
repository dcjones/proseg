mod math;
pub mod paramsampler;
pub mod runvec;
mod sampleset;
mod shardedvec;
mod sparsemat;
pub mod transcripts;
pub mod voxelcheckerboard;
pub mod voxelsampler2;

use ndarray::{s, Array1, Array2, Axis, Zip};
use num::traits::Zero;
use shardedvec::ShardedVec;
use sparsemat::SparseMat;
use std::cell::RefCell;
use std::ops::{Add, AddAssign, SubAssign};
use thread_local::ThreadLocal;

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

    // gamma prior for confusion rates
    pub α_c: f32,
    pub β_c: f32,

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

    // bounds on z coordinate
    pub zmin: f32,
    pub zmax: f32,

    // whether to check if voxel updates break local connectivity
    pub enforce_connectivity: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountPair {
    pub count: u32,
    pub foreground_count: u32,
}

impl CountPair {
    fn new(count: u32, foreground_count: u32) -> Self {
        CountPair {
            count,
            foreground_count,
        }
    }

    fn count(count: u32) -> Self {
        CountPair {
            count,
            foreground_count: 0,
        }
    }
}

impl Add for CountPair {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        CountPair {
            count: self.count + other.count,
            foreground_count: self.foreground_count + other.foreground_count,
        }
    }
}

impl AddAssign for CountPair {
    fn add_assign(&mut self, other: Self) {
        self.count += other.count;
        self.foreground_count += other.foreground_count;
    }
}

impl SubAssign for CountPair {
    fn sub_assign(&mut self, other: Self) {
        self.count -= other.count;
        self.foreground_count -= other.foreground_count;
    }
}

impl Zero for CountPair {
    fn zero() -> Self {
        CountPair {
            count: 0,
            foreground_count: 0,
        }
    }

    fn is_zero(&self) -> bool {
        self.count == 0 && self.foreground_count == 0
    }
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
    // [ncell] total count for each cell
    pub cell_population: ShardedVec<usize>,

    // [ncells] cell volume in voxel count
    pub cell_volume: ShardedVec<u32>,

    // [ncells] cell volume in exposed voxel surface count
    pub cell_surface_area: ShardedVec<u32>,

    // [ncells] cell volume in cubic microns
    pub log_cell_volume: Array1<f32>,

    // [ncells] cell_volume * cell_scale
    pub effective_cell_volume: Array1<f32>,

    // [ncells] per-cell "effective" volume scaling factor
    pub cell_scale: Array1<f32>,

    // area of the convex hull containing all transcripts
    full_layer_volume: f32,

    z0: f32,
    layer_depth: f32,

    // [ntranscripts] current assignment of transcripts to background
    // pub transcript_state: Array1<TranscriptState>,
    // pub prev_transcript_state: Array1<TranscriptState>,

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

    μ_volume: Array1<f32>, // volume dist mean param by component
    σ_volume: Array1<f32>, // volume dist std param by component

    // [ncells, nhidden]: cell ψ parameter in the latent space
    pub φ: Array2<f32>,

    // [nhidden]: precompute φ_k.dot(cell_volume)
    φ_v_dot: Array1<f32>,

    // For components as well???
    // [ncells, nhidden] aux CRT variables for sampling rφ
    // pub lφ: Array2<f32>,

    // [ncells, nhidden] aux CRT variables for sampling rφ
    pub lφ: Array2<u32>,

    // [ncells, nhidden] aux PolyaGamma variables for sampling sφ
    pub ωφ: Array2<f32>,

    // [ncomponents, nhidden] φ gamma shape parameters
    pub rφ: Array2<f32>,

    // for precomputing lgamma(rφ)
    lgamma_rφ: Array2<f32>,

    // [ncomponents, nhidden] φ gamma scale parameters
    pub sφ: Array2<f32>,

    // Size of the upper block of θ that is the identity matrix
    nunfactored: usize,

    // [ngenes, nhidden]: gene loadings in the latent space
    pub θ: Array2<f32>,

    // [nhidden]: Sums across the first axis of θ
    pub θksum: Array1<f32>,

    // // [ncomponents, ngenes] NB p parameters.
    // θ: Array2<f32>,

    // // log(ods_to_prob(θ))
    // logp: Array2<f32>,

    // // log(1 - ods_to_prob(θ))
    // log1mp: Array2<f32>,

    // We should avoid ever computing this entire matrix, since it's a lot
    // of memory and a big matrix multiply.
    // // [ncells, ngenes] Poisson rates
    // pub λ: Array2<f32>,

    // [ngenes, nlayers] background rate: rate at which halucinate transcripts
    // across the entire layer
    pub λ_bg: Array2<f32>,
    pub logλ_bg: Array2<f32>,

    // [ngenes] confusion: rate at which we halucinate transcripts within cells
    // pub λ_c: Array1<f32>,
    pub voxel_volume: f32,

    // time, which is incremented after every iteration
    t: u32,
}

impl ModelParams {
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
