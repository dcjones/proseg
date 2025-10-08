#![allow(confusable_idents)]

use clap::Parser;

mod output;
mod sampler;
mod schemas;
mod spatialdata_input;
mod spatialdata_output;

use core::f32;
use indicatif::{ProgressBar, ProgressStyle};
use log::{info, trace, warn};
use rayon::current_num_threads;
use regex::Regex;
use sampler::paramsampler::ParamSampler;
use sampler::transcriptrepo::TranscriptRepo;
use sampler::transcripts::{read_transcripts_csv, read_visium_data};
use sampler::voxelcheckerboard::{PixelTransform, VoxelCheckerboard};
use sampler::voxelsampler::VoxelSampler;
use sampler::{ModelParams, ModelPriors};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::time::Instant;

use output::*;
use schemas::OutputFormat;
use spatialdata_input::{read_cell_polygons_zarr, read_transcripts_zarr};
use spatialdata_output::write_spatialdata_zarr;

const DEFAULT_BURNIN_VOXEL_SIZE: f32 = 2.0;
const DEFAULT_VOXEL_SIZE: f32 = 1.0;
const DEFAULT_DIFFUSION_SIGMA_NEAR: f32 = 1.0;
const DEFAULT_DIFFUSION_SIGMA_FAR: f32 = 4.0;

#[derive(Parser)]
#[command(version)]
#[command(name = "proseg")]
#[command(author = "Daniel C. Jones")]
#[command(
    about = "High-speed cell segmentation of transcript-resolution spatial transcriptomics data."
)]
struct Args {
    /// CSV with transcript information. How this is interpreted is determined
    /// either by using a preset (`--xenium`, `--cosmx`, `--cosmx-micron`, `--merfish`)
    /// or by manually setting column names using (`--x-column`, `--transcript-column`, etc).
    transcript_csv: String,

    /// Preset for 10X Xenium data
    #[arg(long, default_value_t = false)]
    xenium: bool,

    /// Preset for NanoString CosMx data that using pixel coordinates. Output will still be
    /// in microns.
    #[arg(long, default_value_t = false)]
    cosmx: bool,

    /// Preset for NanoString CosMx data that has been pre-scaled to microns.
    #[arg(long, default_value_t = false)]
    cosmx_micron: bool,

    /// Preset for Vizgen MERFISH/MERSCOPE.
    #[arg(long, default_value_t = false)]
    merscope: bool,

    /// (Deprecated) Preset for Vizgen MERFISH/MERSCOPE.
    #[arg(long, default_value_t = false)]
    merfish: bool,

    /// Input file is a spatialdata zarr directory or zip file
    #[arg(long, default_value_t = false)]
    zarr: bool,

    /// Regex pattern matching names of genes/features to be excluded
    #[arg(long, default_value = None)]
    excluded_genes: Option<String>,

    /// Initialize with cell assignments rather than nucleus assignments
    #[arg(long, default_value_t = false)]
    use_cell_initialization: bool,

    /// Preset for Visium HD
    #[arg(long, default_value_t = false)]
    visiumhd: bool,

    /// Specify if ids may be re-used in different fovs in the input data
    #[arg(long, default_value_t = false)]
    non_unique_cell_ids: bool,

    /// Name of column containing the feature/gene name
    #[arg(long, default_value = None)]
    gene_column: Option<String>,

    /// Name of column containing the transcript ID
    #[arg(long, default_value = None)]
    transcript_id_column: Option<String>,

    /// Cellpose cell masks matrix in .npy format.
    #[arg(long, default_value = None)]
    cellpose_masks: Option<String>,

    /// Spaceranger Visium HD segmentation parquet file.
    #[arg(long, default_value = None)]
    spaceranger_barcode_mappings: Option<String>,

    /// Cellpose cell probability matrix in .npy format.
    #[arg(long, default_value = None)]
    cellpose_cellprobs: Option<String>,

    // Affine x-coordinate transformation to transform mask pixels to slide microns.
    #[arg(long, value_delimiter = ' ', num_args = 3)]
    cellpose_x_transform: Option<Vec<f32>>,

    // Affine y-coordinate transformation to transform mask pixels to slide microns.
    #[arg(long, value_delimiter = ' ', num_args = 3)]
    cellpose_y_transform: Option<Vec<f32>>,

    // Scale in microns per pixel
    #[arg(long, default_value = None)]
    cellpose_scale: Option<f32>,

    #[arg(long, default_value_t = 0.85)]
    cellpose_cellprob_discount: f32,

    /// Expand initialized cells outward by this many voxels.
    #[arg(long, default_value_t = 0)]
    expand_initialized_cells: usize,

    /// Name of column containing the x coordinate
    #[arg(short, long, default_value = None)]
    x_column: Option<String>,

    /// Name of column containing the y coordinate
    #[arg(short, long, default_value = None)]
    y_column: Option<String>,

    /// Name of column containing the z coordinate
    #[arg(short, long, default_value = None)]
    z_column: Option<String>,

    /// Name of column containing the cellular compartment
    #[arg(long, default_value = None)]
    compartment_column: Option<String>,

    /// Value in the cellular compartment column indicated the nucleus
    #[arg(long, default_value = None)]
    compartment_nuclear: Option<String>,

    /// Name of column containing the field of view
    #[arg(long, default_value = None)]
    fov_column: Option<String>,

    /// Column indicating whether a transcript is assigned to a cell
    #[arg(long, default_value = None)]
    cell_assignment_column: Option<String>,

    /// Value in the cell assignment column indicating an unassigned transcript
    #[arg(long, default_value = None, allow_hyphen_values = true)]
    cell_assignment_unassigned: Option<String>,

    /// Name of column containing the cell ID
    #[arg(long, default_value = None)]
    cell_id_column: Option<String>,

    /// Value in the cell ID column indicating an unassigned transcript
    #[arg(long, default_value = None, allow_hyphen_values = true)]
    cell_id_unassigned: Option<String>,

    /// Name of column containing the quality value
    #[arg(long, default_value = None)]
    qv_column: Option<String>,

    /// Spatialdata cell boundary geometry to use as prior segmentation
    #[arg(long, default_value = None)]
    zarr_shape: Option<String>,

    /// Geometry column name for spatialdata cell boundaries
    #[arg(long, default_value = None)]
    zarr_shape_geometry_column: Option<String>,

    /// Cell ID column name for spatialdata cell boundaries
    #[arg(long, default_value = None)]
    zarr_shape_cell_id_column: Option<String>,

    /// Ignore the z coordinate if any, treating the data as 2D
    #[arg(long, default_value_t = false)]
    ignore_z_coord: bool,

    /// Filter out transcripts with quality values below this threshold
    #[arg(long, default_value = None)]
    min_qv: Option<f32>,

    /// Target number of cells per chunk in the parallelization scheme
    /// Smaller number enabled more parallelization, but too small a number
    /// risks inconsistent updates.
    #[arg(long, default_value_t = 100)]
    cells_per_chunk: usize,

    /// Number of components in the mixture model of cellular gene expression
    #[arg(long, default_value_t = 10)]
    ncomponents: usize,

    /// Dimenionality of the latent space
    #[arg(long, default_value_t = 100)]
    nhidden: usize,

    /// Number of layers of voxels in the z-axis used for segmentation
    #[arg(long, default_value_t = 4)]
    voxel_layers: usize,

    // /// Sampler schedule, indicating the number of iterations between doubling resolution.
    // #[arg(long, num_args=1.., value_delimiter=',', default_values_t=[150, 150, 300])]
    // schedule: Vec<usize>,

    // Number of initial burnin samples
    #[arg(long, default_value_t = 200)]
    burnin_samples: usize,

    // Number of (post-burnin) samples to run
    #[arg(long, default_value_t = 200)]
    samples: usize,

    // Number of optimization iterations to run after sampling
    #[arg(long, default_value_t = 50)]
    hillclimb: usize,

    /// Number of samples at the end of the schedule used to compute
    /// expectations and uncertainty
    #[arg(long, default_value_t = 100)]
    recorded_samples: usize,

    /// Number of CPU threads (by default, all cores are used)
    #[arg(short = 't', long, default_value=None)]
    nthreads: Option<usize>,

    // Exponential pior on cell jompactness. (smaller numbers induce more compact cells)
    #[arg(long, default_value_t = 0.04)]
    cell_compactness: f32,

    /// Number of sub-iterations sampling cell morphology per overall iteration
    #[arg(short, long, default_value_t = 4000)]
    morphology_steps_per_iter: usize,

    #[arg(long, default_value_t = 2e-1_f32)]
    nuclear_reassignment_prob: f32,

    #[arg(long, default_value_t = 5e-1_f32)]
    prior_seg_reassignment_prob: f32,

    /// Scale transcript coordinates by this factor to arrive at microns
    #[arg(long, default_value=None)]
    coordinate_scale: Option<f32>,

    /// Size x/y size of voxels during initial burn-in phase.
    #[arg(long, default_value=None, help=format!("Size x/y size in microns of voxels during initial burn-in phase. [default: {DEFAULT_BURNIN_VOXEL_SIZE}, on most platforms]"))]
    burnin_voxel_size: Option<f32>,

    /// Size x/y size of voxels.
    #[arg(long, default_value=None, help=format!("Size x/y size in microns of voxels. [default: {DEFAULT_VOXEL_SIZE}, on most platforms]"))]
    voxel_size: Option<f32>,

    /// Size of quads in voxel checkerboard
    #[arg(long, default_value_t = 150.0)]
    quad_size: f32,

    /// Exclude transcripts that are more than this distance from any nucleus
    #[arg(long, default_value_t = 60_f32)]
    max_transcript_nucleus_distance: f32,

    /// Disable transcript diffusion model
    #[arg(long, default_value_t = false)]
    no_diffusion: bool,

    /// Probability of transcript diffusion
    #[arg(long, default_value_t = 0.2)]
    diffusion_probability: f32,

    /// Stddev parameter for repositioning of un-diffused transcripts
    #[arg(long, default_value=None, help="Stddev parameter for repositioning of un-diffused transcripts. [default: {DEFAULT_DIFFUSION_SIGMA_NEAR}]]")]
    diffusion_sigma_near: Option<f32>,

    /// Stddev parameter for repositioning of diffused transcripts
    #[arg(long, default_value=None, help="Stddev parameter for repositioning of diffused transcripts. [default: {DEFAULT_DIFFUSION_SIGMA_FAR}]]")]
    diffusion_sigma_far: Option<f32>,

    /// Stddev parameter for repositioning transcripts on the z-axis.
    #[arg(long, default_value_t = 0.1)]
    diffusion_sigma_z: f32,

    /// Stddev parameter for sampler proposals during transcript repo
    #[arg(long, default_value_t = 4.0)]
    diffusion_proposal_sigma: f32,

    #[arg(long, default_value_t = 0.2)]
    diffusion_proposal_sigma_z: f32,

    /// Allow dispersion parameter to vary during burn-in
    #[arg(long, default_value_t = false)]
    variable_burnin_dispersion: bool,

    /// Fixed dispersion parameter value during burn-in
    #[arg(long, default_value_t = 1.0)]
    burnin_dispersion: f32,

    /// Fixed dispersion parameter throughout sampling
    #[arg(long, default_value = None)]
    dispersion: Option<f32>,

    /// Probability of proposing ab nihlo bubble formation
    #[arg(long, default_value_t = 0.05)]
    ab_nihlo_bubble_prob: f32,

    /// Run time consuming checks to make sure data structures are in a consistent state
    #[arg(long, default_value_t = false)]
    check_consistency: bool,

    /// Prepend a path name to every  output file name
    #[arg(long, default_value = None)]
    output_path: Option<String>,

    /// Write a SpatialData object to zarr format
    #[arg(long, default_value = "proseg-output.zarr")]
    output_spatialdata: Option<String>,

    /// Exclude transcript data from spatialdata output
    #[arg(long, default_value_t = false)]
    exclude_spatialdata_transcripts: bool,

    /// Overwrite existing spatialdata file (other output is always overwritten)
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    /// Output a point estimate of transcript counts per cell
    #[arg(long, default_value = None)]
    output_counts: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_counts_fmt: OutputFormat,

    /// Output a matrix of expected transcript counts per cell
    #[arg(long, default_value = None)]
    output_expected_counts: Option<String>,

    /// Output a matrix of estimated Poisson expression rates per cell
    #[arg(long, default_value = None)]
    output_rates: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_rates_fmt: OutputFormat,

    /// Output cell metadata
    #[arg(long, default_value = None)]
    output_cell_metadata: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_cell_metadata_fmt: OutputFormat,

    /// Output transcript metadata
    #[arg(long, default_value = None)]
    output_transcript_metadata: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_transcript_metadata_fmt: OutputFormat,

    /// Output gene metadata
    #[arg(long, default_value = None)]
    output_gene_metadata: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_gene_metadata_fmt: OutputFormat,

    /// Output cell metagene rates
    #[arg(long, default_value=None)]
    output_metagene_rates: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_metagene_rates_fmt: OutputFormat,

    /// Output metagene loadings
    #[arg(long, default_value=None)]
    output_metagene_loadings: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_metagene_loadings_fmt: OutputFormat,

    /// Output consensus non-overlapping 2D polygons, formed by taking the
    /// dominant cell at each x/y location.
    #[arg(long, default_value = None)]
    output_cell_polygons: Option<String>,

    /// Output cell polygons flattened (unioned) to 2D
    #[arg(long, default_value = None)]
    output_union_cell_polygons: Option<String>,

    /// Output separate cell polygons for each layer of voxels along the z-axis
    #[arg(long, default_value = None)]
    output_cell_polygon_layers: Option<String>,

    /// Output a table of voxel-level counts (mainly for debugging)
    #[arg(long, default_value = None)]
    output_voxel_counts: Option<String>,

    /// Output cell polygons repeatedly during sampling
    #[arg(long, default_value = None)]
    monitor_cell_polygons: Option<String>,

    /// How frequently to output cell polygons during monitoring
    #[arg(long, default_value_t = 10)]
    monitor_cell_polygons_freq: usize,

    /// Use connectivity checks to prevent cells from having any disconnected voxels
    #[arg(long, default_value_t = false)]
    enforce_connectivity: bool,

    #[arg(long, default_value_t = 300)]
    nunfactored: usize,

    /// Disable factorization model and use genes directly
    #[arg(long, default_value_t = false)]
    no_factorization: bool,

    /// Include negative controls which are excluded by default.
    #[arg(long, default_value_t = false)]
    include_neg_ctrls: bool,

    /// Disable cell scale factors
    #[arg(long, default_value_t = false)]
    use_scaled_cells: bool,

    // Hyperparameters
    #[arg(long, default_value_t = 1.0)]
    hyperparam_e_phi: f32,

    #[arg(long, default_value_t = 1.0)]
    hyperparam_f_phi: f32,

    #[arg(long, default_value_t = 1.0)]
    hyperparam_neg_mu_phi: f32,

    #[arg(long, default_value_t = 0.1)]
    hyperparam_tau_phi: f32,

    #[arg(long, default_value_t = 0.5)]
    density_bandwidth: f32,

    #[arg(long, default_value_t = 5)]
    density_bins: usize,
}

fn set_xenium_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("feature_name"));
    args.transcript_id_column
        .get_or_insert(String::from("transcript_id"));
    args.x_column.get_or_insert(String::from("x_location"));
    args.y_column.get_or_insert(String::from("y_location"));
    args.z_column.get_or_insert(String::from("z_location"));
    args.compartment_column
        .get_or_insert(String::from("overlaps_nucleus"));
    args.compartment_nuclear.get_or_insert(String::from("1"));
    args.cell_id_column.get_or_insert(String::from("cell_id"));
    args.cell_id_unassigned
        .get_or_insert(String::from("UNASSIGNED"));
    args.qv_column.get_or_insert(String::from("qv"));
    args.min_qv.get_or_insert(20.0);
    if !args.include_neg_ctrls {
        args.excluded_genes.get_or_insert(String::from(
            "^(Deprecated|NegControl|Unassigned|Intergenic)",
        ));
    }

    // This seems pretty consistent, but seems possible it could change
    if args.cellpose_scale.is_none()
        && args.cellpose_x_transform.is_none()
        && args.cellpose_y_transform.is_none()
    {
        args.cellpose_scale = Some(0.2125);
    }

    // newer xenium data does have a fov column
    args.fov_column.get_or_insert(String::from("fov_name"));
}

fn set_cosmx_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("target"));
    args.x_column.get_or_insert(String::from("x_global_px"));
    args.y_column.get_or_insert(String::from("y_global_px"));
    args.z_column.get_or_insert(String::from("z"));
    args.compartment_column
        .get_or_insert(String::from("CellComp"));
    args.compartment_nuclear
        .get_or_insert(String::from("Nuclear"));
    args.fov_column.get_or_insert(String::from("fov"));
    args.cell_id_column.get_or_insert(String::from("cell"));
    args.cell_id_unassigned.get_or_insert(String::from(""));
    args.cell_assignment_column
        .get_or_insert(String::from("cell_ID"));
    args.cell_assignment_unassigned
        .get_or_insert(String::from("0"));
    if !args.include_neg_ctrls {
        args.excluded_genes
            .get_or_insert(String::from("^(FalseCode|SystemControl|NegPrb|Negative)"));
    }
    args.non_unique_cell_ids = true;

    // CosMx reports values in pixels and pixel size appears to always be 0.12028 microns.
    args.coordinate_scale.get_or_insert(0.12028);
}

fn set_cosmx_micron_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("target"));
    args.x_column.get_or_insert(String::from("x"));
    args.y_column.get_or_insert(String::from("y"));
    args.z_column.get_or_insert(String::from("z"));
    args.compartment_column
        .get_or_insert(String::from("CellComp"));
    args.compartment_nuclear
        .get_or_insert(String::from("Nuclear"));
    args.fov_column.get_or_insert(String::from("fov"));
    args.cell_id_column.get_or_insert(String::from("cell_ID"));
    args.cell_id_unassigned.get_or_insert(String::from("0"));
    if !args.include_neg_ctrls {
        args.excluded_genes
            .get_or_insert(String::from("^(FalseCode|SystemControl|NegPrb|Negative)"));
    }
    args.non_unique_cell_ids = true;
}

fn set_merfish_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("gene"));
    args.x_column.get_or_insert(String::from("x"));
    args.y_column.get_or_insert(String::from("y"));
    args.z_column.get_or_insert(String::from("z"));
    args.cell_id_column.get_or_insert(String::from("cell"));
    args.cell_id_unassigned.get_or_insert(String::from("NA"));
    // args.cell_id_unassigned.get_or_insert(String::from("0"));
}

fn set_merscope_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("gene"));
    args.x_column.get_or_insert(String::from("global_x"));
    args.y_column.get_or_insert(String::from("global_y"));
    args.z_column.get_or_insert(String::from("global_z"));
    args.fov_column.get_or_insert(String::from("fov"));
    args.cell_id_column.get_or_insert(String::from("cell_id"));
    args.cell_id_unassigned.get_or_insert(String::from("-1"));
}

fn set_visiumhd_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("gene_id"));
    args.x_column.get_or_insert(String::from("x"));
    args.y_column.get_or_insert(String::from("y"));
    args.z_column.get_or_insert(String::from("z")); // ignored
    args.cell_id_column.get_or_insert(String::from("cell"));
    args.cell_id_unassigned.get_or_insert(String::from("0"));
    args.burnin_voxel_size = Some(2.0);
    args.voxel_size = Some(2.0);
    args.voxel_layers = 1;
    args.ignore_z_coord = true;
    args.diffusion_sigma_near.get_or_insert(2.0);
    args.diffusion_sigma_far.get_or_insert(8.0);
}

fn main() {
    env_logger::init();
    let start_time = Instant::now();

    let mut args = Args::parse();

    if let Some(nthreads) = args.nthreads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build_global()
            .unwrap();
    }
    let nthreads = current_num_threads();
    println!("Using {nthreads} threads");

    if let Some(ref output_spatialdata) = args.output_spatialdata {
        let path = if let Some(ref output_path) = args.output_path {
            Path::new(output_path).join(output_spatialdata)
        } else {
            Path::new(output_spatialdata).to_path_buf()
        };

        if path.exists() {
            if args.overwrite {
                std::fs::remove_dir_all(&path).unwrap_or_else(|e| {
                    panic!(
                        "Failed to remove existing directory {}: {}",
                        path.display(),
                        e
                    );
                });
            } else {
                panic!(
                    "Output directory {} already exists. Use --overwrite to replace it.",
                    path.display()
                );
            }
        }
    }

    if (args.xenium as u8)
        + (args.cosmx as u8)
        + (args.cosmx_micron as u8)
        + (args.merfish as u8)
        + (args.merscope as u8)
        + (args.visiumhd as u8)
        > 1
    {
        panic!(
            "At most one of --xenium, --cosmx, --cosmx-micron, --merfish, --merscope, --visiumhd can be set"
        );
    }

    if args.xenium {
        set_xenium_presets(&mut args);
    }

    if args.cosmx {
        set_cosmx_presets(&mut args);
    }

    if args.cosmx_micron {
        set_cosmx_micron_presets(&mut args);
    }

    if args.merfish {
        println!("WARNING: --merfish is deprecated, use --merscope instead");
        set_merfish_presets(&mut args);
    }

    if args.merscope {
        set_merscope_presets(&mut args);
    }

    if args.visiumhd {
        set_visiumhd_presets(&mut args);
    }

    if args.visiumhd
        && args.cellpose_masks.is_none()
        && args.spaceranger_barcode_mappings.is_none()
        && !args.zarr
    {
        panic!(
            "Visium HD input must be initialized with either --cellpose-masks, --spacerange-barcode-mappings, or --spatialdata"
        );
    }

    if args.cellpose_masks.is_some() && args.spaceranger_barcode_mappings.is_some() {
        panic!("Only one of --cellpose-masks or --spaceranger-barcode-mappings can be used.");
    }

    if args.recorded_samples > args.samples {
        panic!("recorded-samples must be <= samples");
    }

    if args.use_cell_initialization {
        args.compartment_column = None;
        args.compartment_nuclear = None;
    }

    assert!(args.ncomponents > 0);

    fn expect_arg<T>(arg: Option<T>, argname: &str) -> T {
        arg.unwrap_or_else(|| panic!("Missing required argument: --{argname}"))
    }

    let burnin_voxel_size = args.burnin_voxel_size.unwrap_or(DEFAULT_BURNIN_VOXEL_SIZE);
    let voxel_size = args.voxel_size.unwrap_or(DEFAULT_VOXEL_SIZE);
    if voxel_size > burnin_voxel_size {
        panic!("Voxel size must be less than or equal to burnin voxel size");
    }

    // We arrive at voxel size by doubling resolution k times, so the ratio has to be a power of two.
    let voxel_burnin_scale = burnin_voxel_size / voxel_size;
    if voxel_burnin_scale < 1.0 || voxel_burnin_scale.fract().abs() > 1e-5 {
        panic!("Ratio between --burnin-voxel-size and --voxel-size must an integer");
    }
    let burnin_voxel_scale = voxel_burnin_scale as usize;

    let excluded_genes = args.excluded_genes.map(|pat| Regex::new(&pat).unwrap());

    let t0 = Instant::now();

    let mut dataset = if args.zarr {
        read_transcripts_zarr(
            &args.transcript_csv,
            &excluded_genes,
            &expect_arg(args.x_column, "x"),
            &expect_arg(args.y_column, "y"),
            &args.z_column,
            &expect_arg(args.gene_column, "gene"),
            &args.cell_id_column,
            &args.cell_id_unassigned.unwrap_or("".to_string()),
            args.coordinate_scale.unwrap_or(1.0),
        )
    } else if args.visiumhd {
        read_visium_data(&args.transcript_csv, excluded_genes)
    } else {
        read_transcripts_csv(
            &args.transcript_csv,
            excluded_genes,
            &expect_arg(args.gene_column, "gene-column"),
            args.transcript_id_column,
            args.compartment_column,
            args.compartment_nuclear,
            args.fov_column,
            args.cell_assignment_column,
            args.cell_assignment_unassigned,
            &expect_arg(args.cell_id_column, "cell-id-column"),
            &expect_arg(args.cell_id_unassigned, "cell-id-unassigned"),
            args.qv_column,
            &expect_arg(args.x_column, "x-column"),
            &expect_arg(args.y_column, "y-column"),
            &expect_arg(args.z_column, "z-column"),
            args.min_qv.unwrap_or(0.0),
            args.ignore_z_coord,
            args.coordinate_scale.unwrap_or(1.0),
            args.non_unique_cell_ids,
        )
    };

    if dataset.ncells > 0 {
        dataset.filter_cellfree_transcripts(args.max_transcript_nucleus_distance);
    }
    info!("loaded transcripts: {:?}", t0.elapsed());

    if args.nunfactored >= dataset.ngenes() {
        args.no_factorization = true;
    }

    let (zmin, zmax) = dataset.normalize_z_coordinates();

    if zmin == zmax && args.voxel_layers != 1 {
        println!("Z-coordinate span is zero. Setting voxel layers to 1.");
        args.voxel_layers = 1;
    }

    // TODO: In various sharded data structures we assume cells index proximity
    // is correlated with spatial proximity. We may want to shuffle indexes so
    // this is true.

    if !args.no_factorization {
        dataset.select_unfactored_genes(args.nunfactored);
    }

    // We are going to try to initialize at full resolution.
    let t0 = Instant::now();
    let mut voxels = if args.zarr && args.zarr_shape.is_some() {
        if args.zarr_shape_geometry_column.is_none() || args.zarr_shape_cell_id_column.is_none() {
            panic!(
                "--zarr-shape-geometry-column and --zarr-shape-cell-id-column must be specified."
            );
        }
        let cell_polygons = read_cell_polygons_zarr(
            &args.transcript_csv,
            &args.zarr_shape.unwrap(),
            &args.zarr_shape_geometry_column.unwrap(),
            &args.zarr_shape_cell_id_column.unwrap(),
            args.coordinate_scale.unwrap_or(1.0),
        );

        let cell_polygons = cell_polygons.unwrap_or_else(|| panic!("Could not read cell polygons"));

        VoxelCheckerboard::from_cell_polygons(
            &mut dataset,
            &cell_polygons,
            burnin_voxel_size,
            args.quad_size,
            args.voxel_layers,
            1.0 - args.prior_seg_reassignment_prob,
            args.expand_initialized_cells,
            args.density_bandwidth,
            args.density_bins,
        )
    } else if let Some(cellpose_masks) = args.cellpose_masks {
        if args.cellpose_scale.is_some()
            && (args.cellpose_x_transform.is_some() || args.cellpose_y_transform.is_some())
        {
            panic!(
                "Cellpose mask transform must be supplied with either --cellpose-scale or both of --cellpose-x-transform, --cellpose-y-transform"
            );
        }

        let pixel_transform = if let Some(cellpose_scale) = args.cellpose_scale {
            PixelTransform::scale(cellpose_scale)
        } else {
            if args.cellpose_x_transform.is_none() || args.cellpose_x_transform.is_none() {
                panic!(
                    "Cellpose mask transform must be supplied with either --cellpose-scale or both of --cellpose-x-transform, --cellpose-y-transform"
                );
            }

            let cellpose_x_transform = args.cellpose_x_transform.unwrap();
            let cellpose_y_transform = args.cellpose_y_transform.unwrap();

            let tx = [
                cellpose_x_transform[0],
                cellpose_x_transform[1],
                cellpose_x_transform[2],
            ];
            let ty = [
                cellpose_y_transform[0],
                cellpose_y_transform[1],
                cellpose_y_transform[2],
            ];
            PixelTransform { tx, ty }
        };

        VoxelCheckerboard::from_cellpose_masks(
            &mut dataset,
            &cellpose_masks,
            &args.cellpose_cellprobs,
            args.cellpose_cellprob_discount,
            burnin_voxel_size,
            args.quad_size,
            args.voxel_layers,
            &pixel_transform,
            1.0 - args.prior_seg_reassignment_prob,
            args.expand_initialized_cells,
            args.density_bandwidth,
            args.density_bins,
        )
    } else if let Some(spaceranger_barcode_mappings) = args.spaceranger_barcode_mappings {
        VoxelCheckerboard::from_visium_barcode_mappings(
            &mut dataset,
            &spaceranger_barcode_mappings,
            burnin_voxel_size,
            args.quad_size,
            args.voxel_layers,
            1.0 - args.nuclear_reassignment_prob,
            args.expand_initialized_cells,
            args.density_bandwidth,
            args.density_bins,
        )
    } else {
        VoxelCheckerboard::from_prior_transcript_assignments(
            &dataset,
            burnin_voxel_size,
            args.quad_size,
            args.voxel_layers,
            1.0 - args.nuclear_reassignment_prob,
            1.0 - args.prior_seg_reassignment_prob,
            args.expand_initialized_cells,
            args.density_bandwidth,
            args.density_bins,
        )
    };
    info!("initialized voxels: {:?}", t0.elapsed());

    println!("Read dataset:");
    println!("{:>9} transcripts", dataset.transcripts.len());
    println!("{:>9} cells", voxels.ncells);
    println!("{:>9} genes", dataset.ngenes());
    println!("{:>9} fovs", dataset.fov_names.len());

    // Warn if any nucleus has extremely high population, which is likely
    // an error interpreting the file. (e.g. Misinterpreting the unassigned indicator as a cell)
    dataset.prior_nuclei_populations().iter().for_each(|&p| {
        if p > 50000 {
            warn!("Nucleus with large initial population: {p}");
        }
    });

    let μ_vol0: f32 = 10.0 * 10.0 * 0.5;
    let priors = ModelPriors {
        dispersion: args.dispersion,
        burnin_dispersion: if args.variable_burnin_dispersion {
            None
        } else {
            Some(args.burnin_dispersion)
        },

        use_cell_scales: args.use_scaled_cells,

        // min_cell_volume: 1e-6 * μ_vol0,
        μ_μ_volume: (μ_vol0).ln(),
        σ_μ_volume: 3.0_f32,
        α_σ_volume: 4.0,
        β_σ_volume: 1.0,

        use_factorization: !args.no_factorization,
        enforce_connectivity: args.enforce_connectivity,

        // TODO: mean/var ratio is always 1/fφ, but that doesn't seem like the whole
        // story. Seems like it needs to change as a function of the dimensionality
        // of the latent space.

        // I also don't know if this "severe prior" approach is going to work in
        // the long run because we may have far more cells. Needs more testing.
        // eφ: 1000.0,
        // fφ: 1.0,

        // μφ: -20.0,
        // τφ: 0.1,
        αθ: 1e-1,

        eφ: args.hyperparam_e_phi,
        fφ: args.hyperparam_f_phi,

        μφ: -args.hyperparam_neg_mu_phi,
        τφ: args.hyperparam_tau_phi,

        α_bg: 1.0,
        β_bg: 1.0,

        σ_iiq: args.cell_compactness,

        // perimeter_eta: 5.3,
        // perimeter_bound: args.perimeter_bound,

        // nuclear_reassignment_log_prob: args.nuclear_reassignment_prob.ln(),
        // nuclear_reassignment_1mlog_prob: (1.0 - args.nuclear_reassignment_prob).ln(),

        // prior_seg_reassignment_log_prob: args.prior_seg_reassignment_prob.ln(),
        // prior_seg_reassignment_1mlog_prob: (1.0 - args.prior_seg_reassignment_prob).ln(),
        use_diffusion_model: !args.no_diffusion,
        p_diffusion: args.diffusion_probability,
        σ_xy_diffusion_near: args
            .diffusion_sigma_near
            .unwrap_or(DEFAULT_DIFFUSION_SIGMA_NEAR),
        σ_xy_diffusion_far: args
            .diffusion_sigma_far
            .unwrap_or(DEFAULT_DIFFUSION_SIGMA_FAR),
        σ_z_diffusion: args.diffusion_sigma_z,
        σ_xy_diffusion_proposal: args.diffusion_proposal_sigma,
        σ_z_diffusion_proposal: args.diffusion_proposal_sigma_z,
        τv: 10.0,
    };

    // let full_volume = dataset.estimate_full_volume();
    // let layer_volume = full_volume / args.voxel_layers as f32;
    // dbg!(full_volume, layer_volume);

    let mut params = ModelParams::new(
        &voxels,
        &priors,
        args.nhidden,
        args.nunfactored,
        args.ncomponents,
        args.density_bins,
    );

    let param_sampler = ParamSampler::new();
    let mut voxel_sampler =
        VoxelSampler::new(0, args.voxel_layers as i32 - 1, args.ab_nihlo_bubble_prob);

    let mut transcript_repo = TranscriptRepo::new(&priors, voxels.voxelsize, voxels.voxelsize_z);

    const INIT_ITERATIONS: usize = 20;

    let total_iterations = INIT_ITERATIONS + args.samples + args.burnin_samples + args.hillclimb;
    let prog = ProgressBar::new(total_iterations as u64);
    prog.set_style(
        ProgressStyle::with_template("{eta_precise} {bar:60} | {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    for _ in 0..INIT_ITERATIONS {
        param_sampler.sample(&priors, &mut params, true, 1.0, false, false, false);
        prog.inc(1);
    }

    for _it in 0..args.burnin_samples {
        run_sampler(
            &param_sampler,
            &mut voxel_sampler,
            &transcript_repo,
            &mut voxels,
            &priors,
            &mut params,
            dataset.transcripts.len(),
            args.morphology_steps_per_iter,
            true,
            1.0,
            false,
            args.check_consistency,
            &prog,
        );
    }

    let mut voxels = if burnin_voxel_scale != 1 {
        voxels.increase_resolution(
            burnin_voxel_scale,
            &mut params,
            &dataset,
            args.density_bandwidth,
            args.density_bins,
        )
    } else {
        voxels
    };

    transcript_repo.set_voxel_size(&priors, voxels.voxelsize, voxels.voxelsize_z);

    let cooling_factor = (0.01_f32.ln() / args.burnin_samples as f32).exp();
    let mut temperature = 1.0;

    for it in 0..(args.samples + args.hillclimb) {
        if it > args.samples {
            temperature *= cooling_factor;
        }

        run_sampler(
            &param_sampler,
            &mut voxel_sampler,
            &transcript_repo,
            &mut voxels,
            &priors,
            &mut params,
            dataset.transcripts.len(),
            args.morphology_steps_per_iter,
            false,
            temperature,
            (args.samples - args.recorded_samples) <= it && it < args.samples,
            args.check_consistency,
            &prog,
        );
    }
    prog.finish();

    if let Some(output_voxel_counts) = args.output_voxel_counts {
        voxels.dump_counts(&dataset, &output_voxel_counts);
    }

    let t0 = Instant::now();
    write_sparse_mtx(
        &args.output_path,
        &args.output_expected_counts,
        &params.foreground_counts_mean.estimates,
    );
    trace!("write_sparse_mtx (expected counts): {:?}", t0.elapsed());

    let t0 = Instant::now();
    write_sparse_mtx(
        &args.output_path,
        &args.output_counts,
        &params.foreground_counts,
    );
    trace!("write_sparse_mtx (max post counts): {:?}", t0.elapsed());

    let cell_centroids = voxels.cell_centroids(&params);
    let transcript_metadata = voxels.transcript_metadata(&params, &dataset.transcripts);

    let original_cell_ids: Vec<_> = voxels
        .used_cells_map
        .iter()
        .map(|old_cell_id| dataset.original_cell_ids[*old_cell_id as usize].clone())
        .collect();

    let t0 = Instant::now();
    write_cell_metadata(
        &args.output_path,
        &args.output_cell_metadata,
        args.output_cell_metadata_fmt,
        &params,
        &cell_centroids,
        &original_cell_ids,
        // &dataset.fov_names,
    );
    trace!("write_cell_metadata: {:?}", t0.elapsed());

    let t0 = Instant::now();
    write_gene_metadata(
        &args.output_path,
        &args.output_gene_metadata,
        args.output_gene_metadata_fmt,
        &params,
        &dataset.gene_names,
        &dataset.transcripts,
        &params.foreground_counts_mean.estimates,
    );
    trace!("write_gene_metadata: {:?}", t0.elapsed());

    let t0 = Instant::now();
    write_metagene_rates(
        &args.output_path,
        &args.output_metagene_rates,
        args.output_metagene_rates_fmt,
        &params.φ,
    );
    trace!("write_metagene_rates: {:?}", t0.elapsed());

    let t0 = Instant::now();
    write_transcript_metadata(
        &args.output_path,
        &args.output_transcript_metadata,
        args.output_transcript_metadata_fmt,
        &voxels,
        &dataset.transcripts,
        &dataset.transcript_ids,
        &transcript_metadata,
        &dataset.gene_names,
    );
    info!("write_transcript_metadata: {:?}", t0.elapsed());

    let t0 = Instant::now();
    write_metagene_loadings(
        &args.output_path,
        &args.output_metagene_loadings,
        args.output_metagene_loadings_fmt,
        &dataset.gene_names,
        &params.θ,
    );
    trace!("write_metagene_loadings: {:?}", t0.elapsed());

    let t0 = Instant::now();
    let (cell_polygons, cell_flattened_polygons) = voxels.cell_polygons();
    info!("generating polygon layers: {:?}", t0.elapsed());

    if args.output_cell_polygon_layers.is_some() || args.output_union_cell_polygons.is_some() {
        let t0 = Instant::now();
        write_cell_multipolygons(
            &args.output_path,
            &args.output_union_cell_polygons,
            &cell_flattened_polygons,
        );
        info!("write union polygons: {:?}", t0.elapsed());

        let t0 = Instant::now();
        write_cell_layered_multipolygons(
            &args.output_path,
            &args.output_cell_polygon_layers,
            &cell_polygons,
        );
        info!("write polygon layers: {:?}", t0.elapsed());
    }

    let consensus_cell_polygons = voxels.consensus_cell_polygons();
    if args.output_cell_polygons.is_some() {
        let t0 = Instant::now();
        info!("generate consensus polygons: {:?}", t0.elapsed());

        let t0 = Instant::now();
        write_cell_multipolygons(
            &args.output_path,
            &args.output_cell_polygons,
            &consensus_cell_polygons,
        );
        info!("write consensus polygons: {:?}", t0.elapsed());
    }

    let mut run_metadata: HashMap<String, String> = HashMap::new();
    run_metadata.insert(
        String::from("version"),
        String::from(env!("CARGO_PKG_VERSION")),
    );
    run_metadata.insert(
        String::from("args"),
        env::args().collect::<Vec<String>>().join(" "),
    );
    run_metadata.insert(
        String::from("duration"),
        format!("{:?}", start_time.elapsed()),
    );

    if let Some(output_spatialdata) = args.output_spatialdata {
        let t0 = Instant::now();
        write_spatialdata_zarr(
            &args.output_path,
            &output_spatialdata,
            &params.foreground_counts,
            &params,
            &voxels,
            &cell_centroids,
            &original_cell_ids,
            &dataset.gene_names,
            &dataset.transcripts,
            &dataset.transcript_ids,
            &transcript_metadata,
            &consensus_cell_polygons,
            &run_metadata,
            args.exclude_spatialdata_transcripts,
        );
        info!("write SpatialData: {:?}", t0.elapsed());
    }
}

#[allow(clippy::too_many_arguments)]
fn run_sampler(
    param_sampler: &ParamSampler,
    voxel_sampler: &mut VoxelSampler,
    transcript_repo: &TranscriptRepo,
    voxels: &mut VoxelCheckerboard,
    priors: &ModelPriors,
    params: &mut ModelParams,
    ntranscripts: usize,
    morphology_steps_per_iter: usize,
    burnin: bool,
    temperature: f32,
    record_samples: bool,
    check_consistency: bool,
    prog: &ProgressBar,
) {
    let t0 = Instant::now();
    for _ in 0..morphology_steps_per_iter {
        voxel_sampler.sample(voxels, priors, params, temperature);
    }
    info!("morphology sampling: {:?}", t0.elapsed());

    if !burnin && priors.use_diffusion_model {
        let t0 = Instant::now();
        transcript_repo.sample(voxels, priors, params, temperature);
        info!("repo transcripts: {:?}", t0.elapsed());
    }

    param_sampler.sample(
        priors,
        params,
        burnin,
        temperature,
        record_samples,
        true,
        true,
    );

    prog.inc(1);

    let nassigned = params.nassigned();
    let nforeground = params.nforeground();
    prog.set_message(format!(
        "log-likelihood: {ll} | assigned: {nassigned} / {ntranscripts} ({perc_assigned:.2}%) | non-background: ({perc_foreground:.2}%)",
        ll = params.log_likelihood(priors),
        nassigned = nassigned,
        perc_assigned = 100.0 * (nassigned as f32) / (ntranscripts as f32),
        perc_foreground = 100.0 * (nforeground as f32) / (ntranscripts as f32),
    ));

    if check_consistency {
        voxels.check_mirrored_quad_edges();
        voxels.check_mismatch_edges();
        params.check_consistency(voxels);
    }
}
