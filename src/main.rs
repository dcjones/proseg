#![allow(confusable_idents)]

use clap::Parser;

mod hull;
mod output;
mod polygon_area;
mod sampler;
mod schemas;

use core::f32;
use hull::convex_hull_area;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::current_num_threads;
use regex::Regex;
use sampler::transcripts::{
    coordinate_span, estimate_full_area, filter_cellfree_transcripts, normalize_z_coordinates,
    read_transcripts_csv, CellIndex, Transcript, BACKGROUND_CELL,
};
use sampler::voxelsampler::{filter_sparse_cells, VoxelSampler};
use sampler::{ModelParams, ModelPriors, ProposalStats, Sampler, UncertaintyTracker};
use schemas::OutputFormat;
use std::cell::RefCell;
use std::collections::HashSet;

use output::*;

const DEFAULT_INITIAL_VOXEL_SIZE: f32 = 4.0;

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

    /// Regex pattern matching names of genes/features to be excluded
    #[arg(long, default_value = None)]
    excluded_genes: Option<String>,

    /// Initialize with cell assignments rather than nucleus assignments
    #[arg(long, default_value_t = false)]
    use_cell_initialization: bool,

    /// Preset for Visium HD
    #[arg(long, default_value_t = false)]
    visiumhd: bool,

    /// Name of column containing the feature/gene name
    #[arg(long, default_value = None)]
    gene_column: Option<String>,

    /// Name of column containing the transcript ID
    #[arg(long, default_value = None)]
    transcript_id_column: Option<String>,

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
    #[arg(long, default_value = None)]
    cell_assignment_unassigned: Option<String>,

    /// Name of column containing the cell ID
    #[arg(long, default_value = None)]
    cell_id_column: Option<String>,

    /// Value in the cell ID column indicating an unassigned transcript
    #[arg(long, default_value = None)]
    cell_id_unassigned: Option<String>,

    /// Name of column containing the quality value
    #[arg(long, default_value = None)]
    qv_column: Option<String>,

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

    /// Number of z-axis layers used to model background expression
    #[arg(long, default_value_t = 4)]
    nbglayers: usize,

    /// Detect the number of z-layers from the data when it's discrete
    #[arg(long, default_value_t = false)]
    detect_layers: bool,

    /// Number of layers of voxels in the z-axis used for segmentation
    #[arg(long, default_value_t = 1)]
    voxel_layers: usize,

    /// Sampler schedule, indicating the number of iterations between doubling resolution.
    #[arg(long, num_args=1.., value_delimiter=',', default_values_t=[150, 150, 300])]
    schedule: Vec<usize>,

    /// Whether to double the z-layers when doubling resolution
    #[arg(long, default_value_t = false)]
    no_z_layer_doubling: bool,

    /// Number of samples at the end of the schedule used to compute
    /// expectations and uncertainty
    #[arg(long, default_value_t = 100)]
    recorded_samples: usize,

    /// Number of CPU threads (by default, all cores are used)
    #[arg(short = 't', long, default_value=None)]
    nthreads: Option<usize>,

    /// Number of sub-iterations sampling cell morphology per overall iteration
    #[arg(short, long, default_value_t = 1000)]
    morphology_steps_per_iter: usize,

    #[arg(long, default_value_t = 0.1)]
    count_pr_cutoff: f32,

    #[arg(long, default_value_t = 0.9)]
    foreground_pr_cutoff: f32,

    #[arg(long, default_value_t = 1.3_f32)]
    perimeter_bound: f32,

    #[arg(long, default_value_t = 2e-1_f32)]
    nuclear_reassignment_prob: f32,

    #[arg(long, default_value_t = 5e-1_f32)]
    prior_seg_reassignment_prob: f32,

    /// Scale transcript coordinates by this factor to arrive at microns
    #[arg(long, default_value=None)]
    coordinate_scale: Option<f32>,

    /// Initial size x/y size of voxels.
    #[arg(long, default_value=None)]
    initial_voxel_size: Option<f32>,

    /// Exclude transcripts that are more than this distance from any nucleus
    #[arg(long, default_value_t = 60_f32)]
    max_transcript_nucleus_distance: f32,

    /// Disable transcript diffusion model
    #[arg(long, default_value_t = false)]
    no_diffusion: bool,

    /// Probability of transcript diffusion
    #[arg(long, default_value_t = 0.2)]
    diffusion_probability: f32,

    /// Stddev of the proposal distribution for transcript repositioning
    #[arg(long, default_value_t = 4.0)]
    diffusion_proposal_sigma: f32,

    /// Stddev parameter for repositioning of non-diffused transcripts
    #[arg(long, default_value_t = 0.5)]
    diffusion_sigma_near: f32,

    /// Stddev parameter for repositioning of diffused transcripts
    #[arg(long, default_value_t = 4.0)]
    diffusion_sigma_far: f32,

    /// Allow dispersion parameter to vary during burn-in
    #[arg(long, default_value_t = false)]
    variable_burnin_dispersion: bool,

    /// Fixed dispersion parameter value during burn-in
    #[arg(long, default_value_t = 1.0)]
    burnin_dispersion: f32,

    /// Fixed dispersion parameter throughout sampling
    #[arg(long, default_value = None)]
    dispersion: Option<f32>,

    /// Perturb initial transcript positions with this standard deviation
    #[arg(long, default_value = None)]
    initial_perturbation_sd: Option<f32>,

    /// Run time consuming checks to make sure data structures are in a consistent state
    #[arg(long, default_value_t = false)]
    check_consistency: bool,

    /// Prepend a path name to every  output file name
    #[arg(long, default_value = None)]
    output_path: Option<String>,

    /// Output a point estimate of transcript counts per cell
    #[arg(long, default_value = None)]
    output_maxpost_counts: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_maxpost_counts_fmt: OutputFormat,

    /// Output a matrix of expected transcript counts per cell
    #[arg(long, default_value = "expected-counts.csv.gz")]
    output_expected_counts: Option<String>,

    /// Output a matrix of estimated Poisson expression rates per cell
    #[arg(long, default_value = None)]
    output_rates: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_rates_fmt: OutputFormat,

    // /// Output per-component parameter values
    // #[arg(long, default_value = None)]
    // output_component_params: Option<String>,
    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_component_params_fmt: OutputFormat,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_expected_counts_fmt: OutputFormat,

    /// Output cell convex hulls
    #[arg(long, default_value = None)]
    output_cell_hulls: Option<String>,

    /// Output cell metadata
    #[arg(long, default_value = "cell-metadata.csv.gz")]
    output_cell_metadata: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_cell_metadata_fmt: OutputFormat,

    /// Output transcript metadata
    #[arg(long, default_value = "transcript-metadata.csv.gz")]
    output_transcript_metadata: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_transcript_metadata_fmt: OutputFormat,

    /// Output gene metadata
    #[arg(long, default_value=None)]
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

    /// Output a table of each voxel in each cell
    #[arg(long, default_value=None)]
    output_cell_voxels: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_cell_voxels_fmt: OutputFormat,

    /// Output consensus non-overlapping 2D polygons, formed by taking the
    /// dominant cell at each x/y location.
    #[arg(long, default_value = "cell-polygons.geojson.gz")]
    output_cell_polygons: Option<String>,

    /// Output cell polygons flattened (unioned) to 2D
    #[arg(long, default_value = "union-cell-polygons.geojson.gz")]
    output_union_cell_polygons: Option<String>,

    /// Output separate cell polygons for each layer of voxels along the z-axis
    #[arg(long, default_value = "cell-polygons-layers.geojson.gz")]
    output_cell_polygon_layers: Option<String>,

    /// Output cell polygons repeatedly during sampling
    #[arg(long, default_value = None)]
    monitor_cell_polygons: Option<String>,

    /// How frequently to output cell polygons during monitoring
    #[arg(long, default_value_t = 10)]
    monitor_cell_polygons_freq: usize,

    /// Use connectivity checks to prevent cells from having any disconnected voxels
    #[arg(long, default_value_t = true)]
    enforce_connectivity: bool,

    #[arg(long, default_value_t = 300)]
    nunfactored: usize,

    /// Disable factorization model and use genes directly
    #[arg(long, default_value_t = false)]
    no_factorization: bool,

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
    args.excluded_genes.get_or_insert(String::from(
        "^(Deprecated|NegControl|Unassigned|Intergenic)",
    ));

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
    args.excluded_genes
        .get_or_insert(String::from("^(FalseCode|NegPrb)"));

    // CosMx reports values in pixels and pixel size appears to always be 0.12 microns.
    args.coordinate_scale.get_or_insert(0.12);
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
    args.excluded_genes
        .get_or_insert(String::from("^(FalseCode|NegPrb)"));
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
    args.initial_voxel_size = Some(1.0);
    args.voxel_layers = 1;
    args.nbglayers = 1;
    args.ignore_z_coord = true;

    // TODO: This is the resolution on the one dataset I have. It probably
    // doesn't generalize.
    args.coordinate_scale.get_or_insert(1.0 / 3.08);
    args.initial_perturbation_sd.get_or_insert(1.0);
}

fn main() {
    // // TODO: Just testing PG sampling
    // {
    //     let mut rng = rand::thread_rng();
    //     // let pg = PolyaGamma::new(1e-6, -80.0);
    //     let mut rs = Vec::<f32>::new();
    //     for _ in 0..100000 {
    //         let pg = PolyaGamma::new(
    //             rng.gen_range(1e-5..1000.0),
    //             rng.gen_range(-50.0..50.0));
    //         rs.push(pg.sample(&mut rng));
    //     }
    //     // dbg!(rs.iter().sum());
    //     // dbg!(pg.mean());
    //     // dbg!(pg.var());
    //     panic!();
    // }

    let mut args = Args::parse();

    if let Some(nthreads) = args.nthreads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build_global()
            .unwrap();
    }
    let nthreads = current_num_threads();
    println!("Using {} threads", nthreads);

    if (args.xenium as u8)
        + (args.cosmx as u8)
        + (args.cosmx_micron as u8)
        + (args.merfish as u8)
        + (args.merscope as u8)
        + (args.visiumhd as u8)
        > 1
    {
        panic!("At most one of --xenium, --cosmx, --cosmx-micron, --merfish, --merscope, --visiumhd can be set");
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

    if args.recorded_samples > *args.schedule.last().unwrap() {
        panic!("recorded-samples must be <= the last entry in the schedule");
    }

    if args.use_cell_initialization {
        args.compartment_column = None;
        args.compartment_nuclear = None;
    }

    assert!(args.ncomponents > 0);

    fn expect_arg<T>(arg: Option<T>, argname: &str) -> T {
        arg.unwrap_or_else(|| panic!("Missing required argument: --{}", argname))
    }

    let initial_voxel_size = args
        .initial_voxel_size
        .unwrap_or(DEFAULT_INITIAL_VOXEL_SIZE);

    /* let (transcript_names,
    mut transcripts,
    mut nucleus_assignments,
    mut cell_assignments,
    mut nucleus_population) = */

    let excluded_genes = args.excluded_genes.map(|pat| Regex::new(&pat).unwrap());

    let mut dataset = read_transcripts_csv(
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
    );

    if !args.no_factorization {
        dataset.select_unfactored_genes(args.nunfactored);
    }

    // let cd3e_idx = dataset
    //     .transcript_names
    //     .iter()
    //     .position(|gene| gene == "CXCL6")
    //     .unwrap();
    // dbg!(cd3e_idx);
    // panic!();

    // Warn if any nucleus has extremely high population, which is likely
    // an error interpreting the file.
    dataset.nucleus_population.iter().for_each(|&p| {
        if p > 10000 {
            eprintln!("Warning: nucleus with population {}", p);
        }
    });

    let mut ncells = dataset.nucleus_population.len();
    filter_cellfree_transcripts(&mut dataset, ncells, args.max_transcript_nucleus_distance);
    normalize_z_coordinates(&mut dataset);

    // keep removing cells until we can initialize with every cell having at least one voxel
    loop {
        let prev_ncells = ncells;

        filter_sparse_cells(
            initial_voxel_size,
            args.voxel_layers,
            &dataset.transcripts,
            &mut dataset.nucleus_assignments,
            &mut dataset.cell_assignments,
            &mut dataset.nucleus_population,
            &mut dataset.original_cell_ids,
        );
        ncells = dataset.nucleus_population.len();
        if ncells == prev_ncells {
            break;
        }
    }

    let ngenes = dataset.transcript_names.len();
    let ncells = dataset.nucleus_population.len();
    let ntranscripts = dataset.transcripts.len();

    if args.nunfactored >= ngenes {
        args.no_factorization = true;
        args.nunfactored = ngenes;
    }

    let nucleus_areas =
        compute_cell_areas(ncells, &dataset.transcripts, &dataset.nucleus_assignments);
    let mean_nucleus_area = nucleus_areas.iter().sum::<f32>()
        / nucleus_areas.iter().filter(|a| **a > 0.0).count() as f32;

    if args.detect_layers {
        const MAX_ZLAYERS: usize = 30;
        let mut undetectable = false;
        let mut zlayers = HashSet::new();
        for t in &dataset.transcripts {
            if t.z.round() == t.z {
                zlayers.insert(t.z as i32);
            } else {
                undetectable = true;
                break;
            }
        }

        if !undetectable && zlayers.len() <= MAX_ZLAYERS {
            args.nbglayers = zlayers.len();
            println!("Detected {} z-layers", args.nbglayers);
        }
    }

    let zmin = dataset
        .transcripts
        .iter()
        .fold(f32::INFINITY, |zmin, t| zmin.min(t.z));
    let zmax = dataset
        .transcripts
        .iter()
        .fold(f32::NEG_INFINITY, |zmin, t| zmin.max(t.z));

    let mut layer_depth = 1.01 * (zmax - zmin) / (args.nbglayers as f32);
    if layer_depth == 0.0 {
        layer_depth = 1.0;
    }

    println!("Read {} transcripts", ntranscripts);
    println!("     {} cells", ncells);
    println!("     {} genes", ngenes);

    let (xmin, xmax, ymin, ymax, zmin, zmax) = coordinate_span(&dataset.transcripts);
    let (xspan, yspan, mut zspan) = (xmax - xmin, ymax - ymin, zmax - zmin);
    if zspan == 0.0 {
        zspan = 1.0;
    }

    let full_area = estimate_full_area(&dataset.transcripts, mean_nucleus_area);
    println!("Estimated full area: {}", full_area);
    let full_volume = full_area * zspan;

    let full_layer_volume = full_volume / (args.nbglayers as f32);
    println!("Full volume: {}", full_volume);

    // Find a reasonable grid size to use to chunk the data
    let area = (xmax - xmin) * (ymax - ymin);

    let cell_density = ncells as f32 / area;
    let chunk_size = (args.cells_per_chunk as f32 / cell_density).sqrt();

    let nchunks = |chunk_size: f32, xspan: f32, yspan: f32| {
        ((xspan / chunk_size).ceil() as usize) * ((yspan / chunk_size).ceil() as usize)
    };

    println!(
        "Using grid size {}. Chunks: {}",
        chunk_size,
        nchunks(chunk_size, xspan, yspan)
    );

    let min_cell_volume = 1e-6 * mean_nucleus_area * zspan;

    let priors = ModelPriors {
        dispersion: args.dispersion,
        burnin_dispersion: if args.variable_burnin_dispersion {
            None
        } else {
            Some(args.burnin_dispersion)
        },

        use_cell_scales: args.use_scaled_cells,

        min_cell_volume,

        μ_μ_volume: (2.0 * mean_nucleus_area * zspan).ln(),
        σ_μ_volume: 3.0_f32,
        α_σ_volume: 0.1,
        β_σ_volume: 0.1,

        use_factorization: !args.no_factorization,

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

        α_c: 1.0,
        β_c: 1.0,

        perimeter_eta: 5.3,
        perimeter_bound: args.perimeter_bound,

        nuclear_reassignment_log_prob: args.nuclear_reassignment_prob.ln(),
        nuclear_reassignment_1mlog_prob: (1.0 - args.nuclear_reassignment_prob).ln(),

        prior_seg_reassignment_log_prob: args.prior_seg_reassignment_prob.ln(),
        prior_seg_reassignment_1mlog_prob: (1.0 - args.prior_seg_reassignment_prob).ln(),

        use_diffusion_model: !args.no_diffusion,
        σ_diffusion_proposal: args.diffusion_proposal_sigma,
        p_diffusion: args.diffusion_probability,
        σ_diffusion_near: args.diffusion_sigma_near,
        σ_diffusion_far: args.diffusion_sigma_far,

        σ_z_diffusion_proposal: 0.2 * zspan,
        σ_z_diffusion: 0.2 * zspan,

        τv: 10.0,

        zmin,
        zmax,

        enforce_connectivity: args.enforce_connectivity,
    };

    let mut params = ModelParams::new(
        &priors,
        full_layer_volume,
        zmin,
        layer_depth,
        args.initial_perturbation_sd.unwrap_or(0.0),
        &dataset.transcripts,
        &dataset.nucleus_assignments,
        &dataset.nucleus_population,
        &dataset.cell_assignments,
        args.ncomponents,
        args.nhidden,
        args.nunfactored,
        args.nbglayers,
        ncells,
        ngenes,
    );

    let total_iterations = args.schedule.iter().sum::<usize>();
    let mut prog = ProgressBar::new(total_iterations as u64);
    prog.set_style(
        ProgressStyle::with_template("{eta_precise} {bar:60} | {msg}")
            .unwrap()
            .progress_chars("##-"),
    );

    let mut uncertainty = UncertaintyTracker::new();

    let mut sampler = RefCell::new(VoxelSampler::new(
        &priors,
        &mut params,
        &dataset.transcripts,
        ngenes,
        args.voxel_layers,
        args.nbglayers,
        zmin,
        layer_depth,
        initial_voxel_size,
        chunk_size,
    ));
    sampler
        .borrow_mut()
        .initialize(&priors, &mut params, &dataset.transcripts);

    let mut total_steps = 0;

    if args.schedule.len() > 1 {
        run_hexbin_sampler(
            &mut prog,
            sampler.get_mut(),
            &priors,
            &mut params,
            &dataset.transcripts,
            args.schedule[0],
            args.morphology_steps_per_iter,
            None,
            &mut total_steps,
            &args.monitor_cell_polygons,
            args.monitor_cell_polygons_freq,
            true,
            true,
            false,
            &args.output_path,
        );

        for &niter in args.schedule[1..args.schedule.len() - 1].iter() {
            if args.check_consistency {
                sampler.borrow_mut().check_consistency(&priors, &mut params);
            }

            sampler.replace_with(|sampler| {
                sampler.double_resolution(&params, !args.no_z_layer_doubling)
            });
            run_hexbin_sampler(
                &mut prog,
                sampler.get_mut(),
                &priors,
                &mut params,
                &dataset.transcripts,
                niter,
                args.morphology_steps_per_iter,
                None,
                &mut total_steps,
                &args.monitor_cell_polygons,
                args.monitor_cell_polygons_freq,
                true,
                true,
                false,
                &args.output_path,
            );
        }
        if args.check_consistency {
            sampler.borrow_mut().check_consistency(&priors, &mut params);
        }
        sampler
            .replace_with(|sampler| sampler.double_resolution(&params, !args.no_z_layer_doubling));
    }

    run_hexbin_sampler(
        &mut prog,
        sampler.get_mut(),
        &priors,
        &mut params,
        &dataset.transcripts,
        *args.schedule.last().unwrap() - args.recorded_samples,
        args.morphology_steps_per_iter,
        None,
        &mut total_steps,
        &args.monitor_cell_polygons,
        args.monitor_cell_polygons_freq,
        true,
        false,
        false,
        &args.output_path,
    );

    run_hexbin_sampler(
        &mut prog,
        sampler.get_mut(),
        &priors,
        &mut params,
        &dataset.transcripts,
        args.recorded_samples,
        args.morphology_steps_per_iter,
        Some(&mut uncertainty),
        &mut total_steps,
        &args.monitor_cell_polygons,
        args.monitor_cell_polygons_freq,
        true,
        false,
        false,
        &args.output_path,
    );

    if args.check_consistency {
        sampler.borrow_mut().check_consistency(&priors, &mut params);
    }
    prog.finish();

    uncertainty.finish(&params);
    let (counts, cell_assignments) = uncertainty.max_posterior_transcript_counts_assignments(
        &params,
        &dataset.transcripts,
        args.count_pr_cutoff,
        args.foreground_pr_cutoff,
    );

    let ecounts = uncertainty.expected_counts(&params, &dataset.transcripts);
    let cell_centroids = sampler.borrow().cell_centroids();

    write_expected_counts(
        &args.output_path,
        &args.output_expected_counts,
        args.output_expected_counts_fmt,
        &dataset.transcript_names,
        &ecounts,
    );
    write_counts(
        &args.output_path,
        &args.output_maxpost_counts,
        args.output_maxpost_counts_fmt,
        &dataset.transcript_names,
        &counts,
    );
    write_rates(
        &args.output_path,
        &args.output_rates,
        args.output_rates_fmt,
        &params,
        &dataset.transcript_names,
    );
    // write_component_params(
    //     &args.output_component_params,
    //     args.output_component_params_fmt,
    //     &params,
    //     &dataset.transcript_names,
    // );
    write_cell_metadata(
        &args.output_path,
        &args.output_cell_metadata,
        args.output_cell_metadata_fmt,
        &params,
        &cell_centroids,
        &cell_assignments,
        &dataset.original_cell_ids,
        &dataset.fovs,
        &dataset.fov_names,
    );
    write_transcript_metadata(
        &args.output_path,
        &args.output_transcript_metadata,
        args.output_transcript_metadata_fmt,
        &dataset.transcripts,
        &params.transcript_positions,
        &dataset.transcript_names,
        &cell_assignments,
        &params.transcript_state,
        &dataset.qvs,
        &dataset.fovs,
        &dataset.fov_names,
    );
    write_gene_metadata(
        &args.output_path,
        &args.output_gene_metadata,
        args.output_gene_metadata_fmt,
        &params,
        &dataset.transcript_names,
        &ecounts,
    );
    write_metagene_rates(
        &args.output_path,
        &args.output_metagene_rates,
        args.output_metagene_rates_fmt,
        &params.φ,
    );
    write_metagene_loadings(
        &args.output_path,
        &args.output_metagene_loadings,
        args.output_metagene_loadings_fmt,
        &dataset.transcript_names,
        &params.θ,
    );
    write_voxels(
        &args.output_path,
        &args.output_cell_voxels,
        args.output_cell_voxels_fmt,
        &sampler.borrow(),
    );

    if args.output_cell_polygon_layers.is_some() || args.output_union_cell_polygons.is_some() {
        let (cell_polygons, cell_flattened_polygons) = sampler.borrow().cell_polygons();
        write_cell_multipolygons(
            &args.output_path,
            &args.output_union_cell_polygons,
            cell_flattened_polygons,
        );
        write_cell_layered_multipolygons(
            &args.output_path,
            &args.output_cell_polygon_layers,
            cell_polygons,
        );
    }

    if args.output_cell_polygons.is_some() {
        let consensus_cell_polygons = sampler.borrow().consensus_cell_polygons();
        write_cell_multipolygons(
            &args.output_path,
            &args.output_cell_polygons,
            consensus_cell_polygons,
        );
    }

    if let Some(output_cell_hulls) = args.output_cell_hulls {
        params.write_cell_hulls(
            &dataset.transcripts,
            &counts,
            &args.output_path,
            &output_cell_hulls,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn run_hexbin_sampler(
    prog: &mut ProgressBar,
    sampler: &mut VoxelSampler,
    priors: &ModelPriors,
    params: &mut ModelParams,
    transcripts: &Vec<Transcript>,
    niter: usize,
    local_steps_per_iter: usize,
    mut uncertainty: Option<&mut UncertaintyTracker>,
    total_steps: &mut usize,
    monitor_cell_polygons: &Option<String>,
    monitor_cell_polygons_freq: usize,
    sample_cell_regions: bool,
    burnin: bool,
    hillclimb: bool,
    output_path: &Option<String>,
) {
    sampler.sample_global_params(priors, params, transcripts, &mut uncertainty, burnin);
    let mut proposal_stats = ProposalStats::new();

    for _ in 0..niter {
        // sampler.check_perimeter_bounds(priors);

        // let t0 = std::time::Instant::now();
        if sample_cell_regions {
            // let t0 = std::time::Instant::now();
            for _ in 0..local_steps_per_iter {
                sampler.sample_cell_regions(
                    priors,
                    params,
                    &mut proposal_stats,
                    hillclimb,
                    &mut uncertainty,
                );
            }
            // println!("Sample cell regions: {:?}", t0.elapsed());
        }
        // println!("Sample cell regions: {:?}", t0.elapsed());

        // let t0 = std::time::Instant::now();
        sampler.sample_global_params(priors, params, transcripts, &mut uncertainty, burnin);
        // println!("Sample global parameters: {:?}", t0.elapsed());

        let nassigned = params.nassigned();
        let nforeground = params.nforeground();
        prog.inc(1);
        prog.set_message(format!(
            "log-likelihood: {ll} | assigned: {nassigned} / {n} ({perc_assigned:.2}%) | non-background: ({perc_foreground:.2}%)",
            ll = params.log_likelihood(priors),
            nassigned = nassigned,
            n = transcripts.len(),
            perc_assigned = 100.0 * (nassigned as f32) / (transcripts.len() as f32),
            perc_foreground = 100.0 * (nforeground as f32) / (transcripts.len() as f32),
        ));

        // println!("Log likelihood: {}", params.log_likelihood());

        // let empty_cell_count = params.cell_population.iter().filter(|p| **p == 0).count();
        // println!("Empty cells: {}", empty_cell_count);

        // dbg!(&proposal_stats);
        // dbg!(sampler.mismatch_edge_stats());
        proposal_stats.reset();

        // if i % 100 == 0 {
        //     println!("Iteration {} ({} unassigned transcripts)", i, params.nunassigned());
        // }

        if *total_steps % monitor_cell_polygons_freq == 0 {
            if let Some(basename) = monitor_cell_polygons {
                let filename = format!("{}-{:04}.geojson.gz", basename, *total_steps);
                let (cell_polygons, _cell_flattened_polygons) = sampler.cell_polygons();
                write_cell_layered_multipolygons(output_path, &Some(filename), cell_polygons);
            }
        }

        *total_steps += 1;
    }
}

fn compute_cell_areas(
    ncells: usize,
    transcripts: &[Transcript],
    cell_assignments: &[CellIndex],
) -> Vec<f32> {
    let mut vertices: Vec<Vec<(f32, f32)>> = vec![Vec::new(); ncells];
    for (&c, &t) in cell_assignments.iter().zip(transcripts.iter()) {
        if c != BACKGROUND_CELL {
            vertices[c as usize].push((t.x, t.y));
        }
    }

    let mut hull = Vec::new();
    let areas = vertices
        .iter_mut()
        .map(|vs| convex_hull_area(vs, &mut hull))
        .collect();

    areas
}
