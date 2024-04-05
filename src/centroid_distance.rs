use clap::Parser;

mod sampler;
use sampler::transcripts::{
    estimate_cell_centroids, read_transcripts_csv, Transcript, CellIndex};

mod output;
use output::{write_table, OutputFormat};

use kiddo::SquaredEuclidean;
use kiddo::float::kdtree::KdTree;
use rayon::prelude::*;
use rayon::current_num_threads;
use arrow2;
use arrow2::array;
use arrow2::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "proseg-centroid-distance")]
#[command(author = "Daniel C. Jones")]
#[command(about = "Simple program to compute the distance of each transcript to the nearest cell centroid.")]
struct Args {
    transcript_csv: String,
    output: String,

    #[arg(long, default_value_t=false)]
    xenium: bool,

    #[arg(long, default_value_t=false)]
    cosmx: bool,

    #[arg(long, default_value_t=false)]
    cosmx_micron: bool,

    #[arg(long, default_value_t=false)]
    merfish: bool,

    #[arg(long, default_value = None)]
    gene_column: Option<String>,

    #[arg(long, default_value = None)]
    transcript_id_column: Option<String>,

    #[arg(long, default_value = None)]
    compartment_column: Option<String>,

    #[arg(long, default_value = None)]
    compartment_nuclear: Option<String>,

    #[arg(long, default_value = None)]
    fov_column: Option<String>,

    #[arg(long, default_value = None)]
    cell_id_column: Option<String>,

    #[arg(long, default_value = None)]
    cell_id_unassigned: Option<String>,

    #[arg(long, default_value = None)]
    qv_column: Option<String>,

    #[arg(short, long, default_value = None)]
    x_column: Option<String>,

    #[arg(short, long, default_value = None)]
    y_column: Option<String>,

    #[arg(short, long, default_value = None)]
    z_column: Option<String>,

    #[arg(long, value_enum, default_value_t = OutputFormat::Infer)]
    output_fmt: OutputFormat,

    #[arg(long, default_value_t)]
    ignore_z_coord: bool,

    #[arg(short = 't', long, default_value=None)]
    nthreads: Option<usize>,
}

fn set_xenium_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("feature_name"));
    args.transcript_id_column.get_or_insert(String::from("transcript_id"));
    args.x_column.get_or_insert(String::from("x_location"));
    args.y_column.get_or_insert(String::from("y_location"));
    args.z_column.get_or_insert(String::from("z_location"));
    args.compartment_column.get_or_insert(String::from("overlaps_nucleus"));
    args.compartment_nuclear.get_or_insert(String::from("1"));
    args.cell_id_column.get_or_insert(String::from("cell_id"));
    args.cell_id_unassigned.get_or_insert(String::from("UNASSIGNED"));
    args.qv_column.get_or_insert(String::from("qv"));
}


fn set_cosmx_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("target"));
    args.x_column.get_or_insert(String::from("x_global_px"));
    args.y_column.get_or_insert(String::from("y_global_px"));
    args.z_column.get_or_insert(String::from("z"));
    args.compartment_column.get_or_insert(String::from("CellComp"));
    args.compartment_nuclear.get_or_insert(String::from("Nuclear"));
    args.fov_column.get_or_insert(String::from("fov"));
    args.cell_id_column.get_or_insert(String::from("cell_ID"));
    args.cell_id_unassigned.get_or_insert(String::from("0"));
}


fn set_cosmx_micron_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("target"));
    args.x_column.get_or_insert(String::from("x"));
    args.y_column.get_or_insert(String::from("y"));
    args.z_column.get_or_insert(String::from("z"));
    args.compartment_column.get_or_insert(String::from("CellComp"));
    args.compartment_nuclear.get_or_insert(String::from("Nuclear"));
    args.fov_column.get_or_insert(String::from("fov"));
    args.cell_id_column.get_or_insert(String::from("cell_ID"));
    args.cell_id_unassigned.get_or_insert(String::from("0"));
}

fn set_merfish_presets(args: &mut Args) {
    args.gene_column.get_or_insert(String::from("gene"));
    args.x_column.get_or_insert(String::from("x"));
    args.y_column.get_or_insert(String::from("y"));
    args.z_column.get_or_insert(String::from("z"));
    args.cell_id_column.get_or_insert(String::from("cell"));
    args.cell_id_unassigned.get_or_insert(String::from("NA"));
}

fn main() {
    let mut args = Args::parse();

    if let Some(nthreads) = args.nthreads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nthreads)
            .build_global()
            .unwrap();
    }
    let nthreads = current_num_threads();
    println!("Using {} threads", nthreads);

    if (args.xenium as u8) + (args.cosmx as u8) + (args.cosmx_micron as u8) + (args.merfish as u8) > 1 {
        panic!("At most one of --xenium, --cosmx, --cosmx-micron, --merfish can be set");
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
        set_merfish_presets(&mut args);
    }

    fn expect_arg<T>(arg: Option<T>, argname: &str) -> T {
        return arg.expect(&format!("Missing required argument: --{}", argname));
    }

    println!("Reading transcripts...");
    // let (transcript_names,
    //      transcripts,
    //      nucleus_assignments,
    //      cell_assignments,
    //      nucleus_population) =
    //     read_transcripts_csv(
    //         &args.transcript_csv,
    //         &expect_arg(args.transcript_column, "transcript-column"),
    //         args.transcript_id_column,
    //         args.compartment_column,
    //         args.compartment_nuclear,
    //         args.fov_column,
    //         &expect_arg(args.cell_id_column, "cell-id-column"),
    //         &expect_arg(args.cell_id_unassigned, "cell-id-unassigned"),
    //         args.qv_column,
    //         &expect_arg(args.x_column, "x-column"),
    //         &expect_arg(args.y_column, "y-column"),
    //         &expect_arg(args.z_column, "z-column"),
    //         f32::NEG_INFINITY,
    //         args.ignore_z_coord,
    //     );
    let mut dataset = read_transcripts_csv(
        &args.transcript_csv,
        &expect_arg(args.gene_column, "transcript-column"),
        args.transcript_id_column,
        args.compartment_column,
        args.compartment_nuclear,
        args.fov_column,
        &expect_arg(args.cell_id_column, "cell-id-column"),
        &expect_arg(args.cell_id_unassigned, "cell-id-unassigned"),
        args.qv_column,
        &expect_arg(args.x_column, "x-column"),
        &expect_arg(args.y_column, "y-column"),
        &expect_arg(args.z_column, "z-column"),
        f32::NEG_INFINITY,
        args.ignore_z_coord,
        1.0
    );

    let ncells = dataset.nucleus_population.len();

    println!("Computing distances...");
    let distances = transcript_centroid_distance(
        &dataset.transcripts, &dataset.nucleus_assignments, ncells);

    println!("Writing output...");
    write_transcript_centroid_distances(
        &args.output,
        args.output_fmt,
        &dataset.transcripts,
        &dataset.transcript_names,
        &dataset.cell_assignments,
        &distances);
}

fn transcript_centroid_distance(
    transcripts: &Vec<Transcript>,
    nucleus_assignments: &Vec<CellIndex>,
    ncells: usize) -> Vec<f32>
{
    let centroids = estimate_cell_centroids(transcripts, nucleus_assignments, ncells);
    let mut kdtree: KdTree<f32, u32, 2, 32, u32> = KdTree::with_capacity(centroids.len());
    for (i, (x, y)) in centroids.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        kdtree.add(&[*x, *y], i as u32);
    }

    return transcripts.par_iter()
        .map(|t| {
            let d = kdtree.nearest_one::<SquaredEuclidean>(&[t.x, t.y]).distance;
            return d;
        })
        .collect();
}

pub fn write_transcript_centroid_distances(
    output: &String,
    output_fmt: OutputFormat,
    transcripts: &Vec<Transcript>,
    transcript_names: &Vec<String>,
    cell_assignments: &Vec<CellIndex>,
    distances: &Vec<f32>,
) {
    let schema = Schema::from(vec![
        Field::new("transcript_id", DataType::UInt64, false),
        Field::new("observed_x", DataType::Float32, false),
        Field::new("observed_y", DataType::Float32, false),
        Field::new("observed_z", DataType::Float32, false),
        Field::new("gene", DataType::Utf8, false),
        Field::new("assignment", DataType::UInt32, false),
        Field::new("centroid_distance", DataType::UInt32, false),
    ]);

    let columns: Vec<Arc<dyn arrow2::array::Array>> = vec![
        Arc::new(array::UInt64Array::from_values(
            transcripts.iter().map(|t| t.transcript_id),
        )),
        Arc::new(array::Float32Array::from_values(
            transcripts.iter().map(|t| t.x),
        )),
        Arc::new(array::Float32Array::from_values(
            transcripts.iter().map(|t| t.y),
        )),
        Arc::new(array::Float32Array::from_values(
            transcripts.iter().map(|t| t.z),
        )),
        Arc::new(array::Utf8Array::<i32>::from_iter_values(
            transcripts
                .iter()
                .map(|t| transcript_names[t.gene as usize].clone()),
        )),
        Arc::new(array::UInt32Array::from_values(
            cell_assignments.iter().cloned()
        )),
        Arc::new(array::Float32Array::from_values(
            distances.iter().cloned()
        )),
    ];

    let chunk = arrow2::chunk::Chunk::new(columns);

    write_table(
        &output,
        output_fmt,
        schema,
        chunk,
    );
}