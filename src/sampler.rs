pub mod connectivity;
pub mod csrmat;
mod featureselection;
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
pub mod transcriptrunmap;
pub mod transcripts;
pub mod transitionmat;
pub mod voxelcheckerboard;
pub mod voxelsampler;

use clustering::kmeans;
use csrmat::CSRMat;
use csrmat::Increment;
use transcripts::CellIndex;
use transitionmat::TransitionMat;

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
    pub prior_weight: f32,

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
    pub record_state_transitions: bool,

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

// Bit-packed structure storing gene (20 bits), density (4 bits), and layer (8 bits) in a single u32
// Layout (MSB to LSB): gene[31:12] | density[11:8] | layer[7:0]
// This reduces memory from 8 bytes to 4 bytes per instance (50% reduction)
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct CountMatRowKey {
    // Packed as: [gene: 20 bits][density: 4 bits][layer: 8 bits]
    packed: u32,
}

impl CountMatRowKey {
    const LAYER_BITS: u32 = 8;
    const DENSITY_BITS: u32 = 4;
    const GENE_BITS: u32 = 20;

    const LAYER_MASK: u32 = (1 << Self::LAYER_BITS) - 1;
    const DENSITY_MASK: u32 = (1 << Self::DENSITY_BITS) - 1;
    const GENE_MASK: u32 = (1 << Self::GENE_BITS) - 1;

    const LAYER_SHIFT: u32 = 0;
    const DENSITY_SHIFT: u32 = Self::LAYER_BITS;
    const GENE_SHIFT: u32 = Self::LAYER_BITS + Self::DENSITY_BITS;

    pub fn new(gene: u32, layer: u32, density: u8) -> Self {
        debug_assert!(
            gene <= Self::GENE_MASK,
            "Gene index {} exceeds maximum of {} (20 bits)",
            gene,
            Self::GENE_MASK
        );
        debug_assert!(
            layer <= Self::LAYER_MASK,
            "Layer index {} exceeds maximum of {} (8 bits)",
            layer,
            Self::LAYER_MASK
        );
        debug_assert!(
            density <= Self::DENSITY_MASK as u8,
            "Density bin {} exceeds maximum of {} (4 bits)",
            density,
            Self::DENSITY_MASK
        );

        let packed = ((gene & Self::GENE_MASK) << Self::GENE_SHIFT)
            | ((density as u32 & Self::DENSITY_MASK) << Self::DENSITY_SHIFT)
            | ((layer & Self::LAYER_MASK) << Self::LAYER_SHIFT);
        CountMatRowKey { packed }
    }

    #[inline]
    pub fn gene(&self) -> u32 {
        (self.packed >> Self::GENE_SHIFT) & Self::GENE_MASK
    }

    #[inline]
    pub fn layer(&self) -> u32 {
        (self.packed >> Self::LAYER_SHIFT) & Self::LAYER_MASK
    }

    #[inline]
    pub fn density(&self) -> u8 {
        ((self.packed >> Self::DENSITY_SHIFT) & Self::DENSITY_MASK) as u8
    }
}

impl Add for CountMatRowKey {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        CountMatRowKey::new(
            self.gene() + other.gene(),
            self.layer() + other.layer(),
            self.density().saturating_add(other.density()),
        )
    }
}

impl AddAssign for CountMatRowKey {
    fn add_assign(&mut self, other: Self) {
        *self = CountMatRowKey::new(
            self.gene() + other.gene(),
            self.layer() + other.layer(),
            self.density().saturating_add(other.density()),
        );
    }
}

impl Zero for CountMatRowKey {
    fn zero() -> Self {
        CountMatRowKey { packed: 0 }
    }

    fn is_zero(&self) -> bool {
        self.packed == 0
    }
}

impl Increment for CountMatRowKey {
    fn inc(&self, bound: CountMatRowKey) -> CountMatRowKey {
        // treating this as three digits, incrementing density then layer then gene
        if self.density() + 1 > bound.density() {
            if self.layer() + 1 > bound.layer() {
                CountMatRowKey::new(self.gene() + 1, 0, 0)
            } else {
                CountMatRowKey::new(self.gene(), self.layer() + 1, 0)
            }
        } else {
            CountMatRowKey::new(self.gene(), self.layer(), self.density() + 1)
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct TransitionMatRowKey {
    pub gene: u32,
    pub dest_cell: CellIndex,
}

impl Add for TransitionMatRowKey {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        TransitionMatRowKey {
            gene: self.gene + other.gene,
            dest_cell: self.dest_cell + other.dest_cell,
        }
    }
}

impl AddAssign for TransitionMatRowKey {
    fn add_assign(&mut self, other: Self) {
        *self = TransitionMatRowKey {
            gene: self.gene + other.gene,
            dest_cell: self.dest_cell + other.dest_cell,
        };
    }
}

impl Zero for TransitionMatRowKey {
    fn zero() -> Self {
        TransitionMatRowKey {
            gene: 0,
            dest_cell: CellIndex::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.gene == 0 && self.dest_cell.is_zero()
    }
}

impl Increment for TransitionMatRowKey {
    fn inc(&self, bound: TransitionMatRowKey) -> TransitionMatRowKey {
        if self.dest_cell + 1 > bound.dest_cell {
            TransitionMatRowKey {
                gene: self.gene + 1,
                dest_cell: 0,
            }
        } else {
            TransitionMatRowKey {
                gene: self.gene,
                dest_cell: self.dest_cell + 1,
            }
        }
    }
}

use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TranscriptAssignment {
    pub cell: CellIndex,
    pub background: bool,
}

pub struct TranscriptState(AtomicU32);

impl TranscriptState {
    const BACKGROUND_FLAG_MASK: u32 = 1 << 31;
    const CELL_INDEX_MASK: u32 = !(1 << 31);

    pub fn new(assignment: TranscriptAssignment) -> Self {
        TranscriptState(AtomicU32::new(Self::pack(assignment)))
    }

    #[inline]
    fn pack(assignment: TranscriptAssignment) -> u32 {
        if assignment.background {
            assignment.cell | Self::BACKGROUND_FLAG_MASK
        } else {
            assignment.cell
        }
    }

    #[inline]
    fn unpack(val: u32) -> TranscriptAssignment {
        TranscriptAssignment {
            cell: if val == BACKGROUND_CELL {
                BACKGROUND_CELL
            } else {
                val & Self::CELL_INDEX_MASK
            },
            background: (val & Self::BACKGROUND_FLAG_MASK) != 0,
        }
    }

    pub fn load(&self) -> TranscriptAssignment {
        Self::unpack(self.0.load(Ordering::Relaxed))
    }

    pub fn store(&self, assignment: TranscriptAssignment) {
        self.0.store(Self::pack(assignment), Ordering::Relaxed);
    }
}

/// Per-entry statistics for tracking expected flow and its sample variance across MCMC samples.
/// The variance is computed using the algebraically equivalent form of Welford's online algorithm:
/// sum-of-squares accumulation avoids needing to process zero-valued sample observations explicitly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FlowStats {
    /// Flow events accumulated during the current sample; reset to 0 after each flush.
    pub sample_count: u32,
    /// Total accumulated count across all recorded samples: sum(x_i).
    pub count: u32,
    /// Sum of squared per-sample counts: sum(x_i^2); used for variance computation.
    pub count_sq: u64,
}

impl FlowStats {
    /// Sample variance of the per-sample flow count across `nsamples` samples.
    pub fn variance(&self, nsamples: usize) -> f32 {
        if nsamples <= 1 {
            return 0.0;
        }
        let n = nsamples as f64;
        let count = self.count as f64;
        let count_sq = self.count_sq as f64;
        // Welford-equivalent: M2 = count_sq - count^2/n, variance = M2 / (n - 1)
        ((count_sq - count * count / n) / (n - 1.0)) as f32
    }
}

impl std::ops::Add for FlowStats {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            sample_count: self.sample_count + rhs.sample_count,
            count: self.count + rhs.count,
            count_sq: self.count_sq + rhs.count_sq,
        }
    }
}

impl num::traits::Zero for FlowStats {
    fn zero() -> Self {
        Self::default()
    }
    fn is_zero(&self) -> bool {
        // An entry is considered zero (and will be skipped by iter_nonzeros)
        // only when no events have ever been recorded for it.
        self.count == 0 && self.sample_count == 0
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

    // [ntranscripts] Current state for each transcript.
    pub transcript_state: Vec<TranscriptState>,

    // [ntranscripts] State vector used for the reported point estimate.
    reported_transcript_state: Vec<TranscriptState>,

    // Counts the number of transitions between cells for each gene.
    // We index as counts as (state, (gene, state)).
    // An encoding quirk used here is that we let 0 be the background state and
    // +1 is added to cell indexes to make the indexing here dense.
    pub state_transitions: TransitionMat,

    // [ncells, ngenes]
    // For cell c and gene g, count the number of times a transcript that was reported
    // in cell c is in a cell/state other than c.
    pub expected_inflow: CSRMat<u32, FlowStats>,

    // [ncells, ngenes]
    // For cell c and gene g, count the number of times a transcript is in cell c that was
    // reported in another cell/state.
    pub expected_outflow: CSRMat<u32, FlowStats>,

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
        ntranscripts: usize,
        nhidden: usize,
        nunfactored: usize,
        ncomponents: usize,
        density_nbins: usize,
    ) -> ModelParams {
        let ncells = voxels.ncells;
        let ngenes = voxels.ngenes;
        let nlayers = (voxels.kmax + 1) as usize;
        if nlayers > 256 {
            panic!(
                "Number of voxel layers ({nlayers}) exceeds maximum of 256. Please reduce --voxel-layers."
            );
        }
        if ngenes > CountMatRowKey::GENE_MASK as usize + 1 {
            panic!(
                "Number of genes ({}) exceeds maximum of {} (20-bit limit). Consider filtering genes.",
                ngenes,
                CountMatRowKey::GENE_MASK + 1
            );
        }
        if density_nbins > CountMatRowKey::DENSITY_MASK as usize + 1 {
            panic!(
                "Number of density bins ({}) exceeds maximum of {} (4-bit limit). Please reduce --density-bins.",
                density_nbins,
                CountMatRowKey::DENSITY_MASK + 1
            );
        }
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
                (nlayers - 1) as u32,
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
                    foreground_row.add(gene_layer.gene(), count);
                }
            });

        let transcript_state =
            std::iter::repeat_with(|| TranscriptState::new(TranscriptAssignment::default()))
                .take(ntranscripts)
                .collect();

        let reported_transcript_state =
            std::iter::repeat_with(|| TranscriptState::new(TranscriptAssignment::default()))
                .take(ntranscripts)
                .collect();

        let state_transitions = TransitionMat::new(ncells + 1);

        let expected_inflow = CSRMat::zeros(ncells, ngenes as u32 - 1);
        let expected_outflow = CSRMat::zeros(ncells, ngenes as u32 - 1);

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
        let (z, θ_centroids) = initial_component_assignments(&counts, ncomponents);

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
        // Seed factored columns from k-means cluster centroids so the sampler
        // starts with meaningful gene programs rather than pure noise.
        let nfactors = nhidden - nunfactored;
        for k in 0..nfactors {
            let src = k % ncomponents;
            for g in nunfactored..ngenes {
                θ[[g, nunfactored + k]] = θ_centroids[[g, src]];
            }
            // Perturb duplicated columns (when nfactors > ncomponents) so they
            // can diverge during sampling.
            if k >= ncomponents {
                for g in nunfactored..ngenes {
                    θ[[g, nunfactored + k]] *= randn(&mut rng).exp();
                }
            }
        }
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
            transcript_state,
            reported_transcript_state,
            state_transitions,
            expected_inflow,
            expected_outflow,
            foreground_counts,
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

    pub fn set_point_estimate(&mut self) {
        for (reported, state) in self
            .reported_transcript_state
            .iter()
            .zip(self.transcript_state.iter())
        {
            reported.store(state.load());
        }
    }

    /// Finalizes per-sample flow statistics for variance tracking.
    /// Must be called once at the end of each recorded sample: squares the current
    /// `sample_count` into `count_sq` (Welford-equivalent M2 accumulation), then
    /// resets `sample_count` to zero for the next sample.
    pub fn flush_flow_stats(&self) {
        let flush_row = |row: csrmat::CSRRow<'_, u32, FlowStats>| {
            let mut guard = row.write();
            guard.guard.scale_all(|stats| {
                if stats.sample_count > 0 {
                    stats.count_sq += stats.sample_count as u64 * stats.sample_count as u64;
                    stats.sample_count = 0;
                }
            });
        };
        self.expected_inflow.par_rows().for_each(flush_row);
        self.expected_outflow.par_rows().for_each(flush_row);
    }

    pub fn update_phi_theta_dot(&mut self) {
        Zip::from(&mut self.φ_θksum_dot)
            .and(self.φ.rows())
            .for_each(|dot, φ_c| {
                *dot = φ_c.dot(&self.θksum);
            });
    }

    // Compute the Poisson rate for cell and gene pair.

    pub fn log_likelihood(&self, _priors: &ModelPriors) -> f32 {
        let mut ll = self
            .foreground_counts
            .par_rows()
            .enumerate()
            .map(|(c, x_c)| {
                let v_c = self.effective_cell_volume[c];
                let x_c = x_c.read();
                let mut accum_c = 0.0;
                let φ_c = self.φ.row(c);
                let φ_c_factored = φ_c.slice(s![self.nunfactored..]);

                for (g, x_cg) in x_c.iter_nonzeros() {
                    let g = g as usize;
                    let λ_cg = if g < self.nunfactored {
                        φ_c[g]
                    } else {
                        φ_c_factored.dot(&self.θ.slice(s![g, self.nunfactored..]))
                    };
                    accum_c += (x_cg as f32) * λ_cg.ln();
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
                (nlayers - 1) as u32,
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
) -> (Array1<u32>, Array2<f32>) {
    let (ncells, j_bound) = counts.shape();
    let ngenes = j_bound.gene() as usize + 1;

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
    const NORM_CONSTANT: f32 = 1e3;
    let expr_row = ThreadLocal::new();
    // for each cell
    Zip::indexed(embedding.rows_mut()).par_for_each(|c, mut embedding_c| {
        let mut expr_row = expr_row
            .get_or(|| RefCell::new(Array1::<f32>::zeros(ngenes)))
            .borrow_mut();
        expr_row.fill(0.0);

        // marginalize counts
        let counts_c = counts.row(c);
        for (key, count) in counts_c.read().iter_nonzeros() {
            expr_row[key.gene() as usize] += count as f32;
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
    let mut membership = kmeans_results.membership.clone();

    // Debug: write membership vector to file
    {
        use std::io::Write;
        let mut f = std::fs::File::create("membership_debug.txt")
            .expect("Unable to create membership_debug.txt");
        for (i, &z_i) in kmeans_results.membership.iter().enumerate() {
            writeln!(f, "{} {}", i, z_i).expect("Unable to write to membership_debug.txt");
        }
    }

    let min_pop = (ncells / ncomponents / 5).max(10);
    rebalance_components(&mut membership, &embedding, ncomponents, min_pop);

    // Debug: write rebalanced membership vector to file
    {
        use std::io::Write;
        let mut f = std::fs::File::create("rebalanced_membership_debug.txt")
            .expect("Unable to create membership_debug.txt");
        for (i, &z_i) in membership.iter().enumerate() {
            writeln!(f, "{} {}", i, z_i)
                .expect("Unable to write to rebalanced_membership_debug.txt");
        }
    }

    let z: Array1<u32> = membership.iter().map(|z_c| *z_c as u32).collect();

    // Compute per-cluster mean gene expression (marginalizing over layers and
    // density bins) as a starting point for θ column initialization.
    let mut centroids = Array2::<f32>::zeros((ngenes, ncomponents));
    let mut cluster_pop = vec![0usize; ncomponents];
    for (c, &z_c) in membership.iter().enumerate() {
        cluster_pop[z_c] += 1;
        for (key, count) in counts.row(c).read().iter_nonzeros() {
            centroids[[key.gene() as usize, z_c]] += count as f32;
        }
    }
    for t in 0..ncomponents {
        let pop = cluster_pop[t].max(1) as f32;
        for g in 0..ngenes {
            centroids[[g, t]] = (NORM_CONSTANT * centroids[[g, t]] / pop).ln_1p();
        }
        // Normalize each column to mean 1 so scale is comparable to the
        // random log-normal init that this replaces.
        let mean = centroids.column(t).sum() / ngenes as f32;
        if mean > 0.0 {
            for g in 0..ngenes {
                centroids[[g, t]] /= mean;
            }
        } else {
            centroids.column_mut(t).fill(1.0);
        }
    }

    (z, centroids)
}

fn rebalance_components(
    membership: &mut [usize],
    embedding: &[Vec<f32>],
    ncomponents: usize,
    min_pop: usize,
) {
    let dim = embedding[0].len();

    let mut pop = vec![0usize; ncomponents];
    for &z_c in membership.iter() {
        pop[z_c] += 1;
    }

    let mut centroids = vec![vec![0.0f64; dim]; ncomponents];
    for (i, &z_i) in membership.iter().enumerate() {
        for (d, &val) in embedding[i].iter().enumerate() {
            centroids[z_i][d] += val as f64;
        }
    }
    for (t, centroid_t) in centroids.iter_mut().enumerate() {
        if pop[t] > 0 {
            for d in centroid_t.iter_mut() {
                *d /= pop[t] as f64;
            }
        }
    }

    // For empty clusters, seed the centroid with a cell from the most
    // populous cluster so that the rebalancing step has a meaningful
    // reference point to attract cells toward.
    let empty_clusters: Vec<usize> = (0..ncomponents).filter(|&t| pop[t] == 0).collect();
    for t in empty_clusters {
        let largest = pop
            .iter()
            .enumerate()
            .max_by_key(|&(_, &p)| p)
            .map(|(i, _)| i)
            .unwrap();
        let cells_in_largest: Vec<usize> = membership
            .iter()
            .enumerate()
            .filter(|&(_, &z_i)| z_i == largest)
            .map(|(i, _)| i)
            .collect();
        if cells_in_largest.is_empty() {
            continue;
        }
        let lc = centroids[largest].clone();
        let seed = cells_in_largest
            .iter()
            .max_by(|&&i, &&j| {
                let di: f64 = embedding[i]
                    .iter()
                    .zip(lc.iter())
                    .map(|(a, b)| (*a as f64 - b).powi(2))
                    .sum();
                let dj: f64 = embedding[j]
                    .iter()
                    .zip(lc.iter())
                    .map(|(a, b)| (*a as f64 - b).powi(2))
                    .sum();
                di.partial_cmp(&dj).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(cells_in_largest[0]);
        for (d, &val) in embedding[seed].iter().enumerate() {
            centroids[t][d] = val as f64;
        }
    }

    // Iteratively move cells from over-populated components to
    // under-populated ones, choosing cells closest to the target
    // centroid.
    loop {
        let (min_comp, &min_pop_val) = pop.iter().enumerate().min_by_key(|&(_, &p)| p).unwrap();

        if min_pop_val >= min_pop {
            break;
        }

        let deficit = min_pop - min_pop_val;
        let centroid = &centroids[min_comp];

        let mut candidates: Vec<(usize, f64)> = Vec::new();
        for (i, &z_i) in membership.iter().enumerate() {
            if z_i != min_comp && pop[z_i] > min_pop {
                let dist: f64 = embedding[i]
                    .iter()
                    .zip(centroid.iter())
                    .map(|(a, b)| (*a as f64 - b).powi(2))
                    .sum();
                candidates.push((i, dist));
            }
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut moved = 0;
        for (cell_idx, _) in candidates {
            if moved >= deficit {
                break;
            }
            let old_comp = membership[cell_idx];
            if pop[old_comp] > min_pop {
                membership[cell_idx] = min_comp;
                pop[old_comp] -= 1;
                pop[min_comp] += 1;
                moved += 1;
            }
        }

        if moved == 0 {
            break;
        }
    }
}
