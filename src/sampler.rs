mod math;
pub mod runvec;
mod sampleset;
mod shardedvec;
mod sparsemat;
pub mod transcripts;
pub mod voxelcheckerboard;
pub mod voxelsampler2;

use ndarray::{s, Array1, Array2, Axis, Zip};
use shardedvec::ShardedVec;
use sparsemat::SparseMat;
use std::cell::RefCell;
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

#[allow(non_snake_case)]
pub struct ModelParams {
    // [ncell] total count for each cell
    pub cell_population: ShardedVec<usize>,

    // [ncells] per-cell volumes
    pub cell_volume: ShardedVec<u32>,
    pub cell_surface_area: ShardedVec<u32>,
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

    // [ncells, ngenes] foreground transcripts counts
    foreground_counts: SparseMat<u32>,

    // [nlayers, ngenes] background transcripts counts
    background_counts: Vec<ShardedVec<u32>>,

    // [ncells, nhidden]
    pub cell_latent_counts: SparseMat<u32>,

    // [ngenes, nhidden]
    pub gene_latent_counts: SparseMat<u32>,

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

    // [ncells, ngenes] Poisson rates
    pub λ: Array2<f32>,

    // [ngenes, nlayers] background rate: rate at which halucinate transcripts
    // across the entire layer
    pub λ_bg: Array2<f32>,
    pub logλ_bg: Array2<f32>,

    // [ngenes] confusion: rate at which we halucinate transcripts within cells
    pub λ_c: Array1<f32>,

    pub voxel_volume: f32,

    // time, which is incremented after every iteration
    t: u32,
}
