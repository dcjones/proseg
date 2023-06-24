#![allow(confusable_idents)]

use clap::Parser;

mod sampler;

use sampler::{Sampler, Segmentation, ModelPriors};
use sampler::transcripts::{read_transcripts_csv, read_nuclei_csv, neighborhood_graph, coordinate_span};
use rayon::current_num_threads;
use csv;
use std::fs::File;
use flate2::Compression;
use flate2::write::GzEncoder;

// use signal_hook::{consts::SIGINT, iterator::Signals};
// use std::{error::Error, thread, time::Duration};


#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args{
    transcript_csv: String,
    cell_centers_csv: String,

    #[arg(short, long, default_value="feature_name")]
    transcript_column: String,

    #[arg(short, long, default_value="x_location")]
    x_column: String,

    #[arg(short, long, default_value="y_location")]
    y_column: String,

    #[arg(short, long, default_value=None)]
    z_column: Option<String>,

    #[arg(short, long, default_value="x_centroid")]
    cell_x_column: String,

    #[arg(short, long, default_value="y_centroid")]
    cell_y_column: String,

    #[arg(short, long, default_value_t=20)]
    ncomponents: usize,

    #[arg(short, long, default_value_t=100000)]
    niter: usize,

    #[arg(short, long, default_value=None)]
    nthreads: Option<usize>,

    #[arg(short, long, default_value_t=0.01_f32)]
    background_prob: f32,

    #[arg(short, long, default_value_t=100)]
    local_steps_per_iter: usize,

    #[arg(short, long, default_value="counts.csv.gz")]
    output_counts: String,
}


fn main() {
    // let mut signals = Signals::new(&[SIGINT])?;

    // thread::spawn(move || {
    //     for sig in signals.forever() {
    //         panic!();
    //     }
    // });

    let args = Args::parse();

    assert!(args.ncomponents > 0);

    let (transcript_names, transcripts) = read_transcripts_csv(
        &args.transcript_csv, &args.transcript_column, &args.x_column,
        &args.y_column, args.z_column.as_deref());
    let ngenes = transcript_names.len();
    let ntranscripts = transcripts.len();

    println!("Read {} transcripts", ntranscripts);

    let nuclei_centroids = read_nuclei_csv(
        &args.cell_centers_csv, &args.cell_x_column, &args.cell_y_column);
    let ncells = nuclei_centroids.len();

    let (xmin, xmax, ymin, ymax) = coordinate_span(&transcripts, &nuclei_centroids);
    let (xspan, yspan) = (xmax - xmin, ymax - ymin);

    if let Some(nthreads) = args.nthreads {
        rayon::ThreadPoolBuilder::new().num_threads(nthreads).build_global().unwrap();
    }
    let nthreads = current_num_threads();
    println!("Using {} threads", nthreads);

    println!("Read {} nuclei centroids", ncells);

    // Find a reasonable grid size to use to chunk the data
    const CHUNK_FACTOR: usize = 4;
    let area = (xmax - xmin) * (ymax - ymin);
    let mut chunk_size = (area / ((nthreads * CHUNK_FACTOR) as f32)).sqrt();

    let min_cells_per_chunk = (ncells as f64).min(100.0);

    let nchunks = |chunk_size: f32, xspan: f32, yspan: f32| {
        ((xspan / chunk_size).ceil() as usize) * ((yspan / chunk_size).ceil() as usize)
    };

    while (ncells as f64) / (nchunks(chunk_size, xspan, yspan) as f64) < min_cells_per_chunk {
        chunk_size *= std::f32::consts::SQRT_2;
    }

    // while (ncells as f64) / ((grid_size * grid_size) as f64) < min_cells_per_chunk {
    //     grid_size *= std::f32::consts::SQRT_2;
    // }
    println!("Using grid size {}. Chunks: {}", chunk_size, nchunks(chunk_size, xspan, yspan));

    let quadrant_size = chunk_size / 2.0;
    let (adjacency, avg_edge_length) = neighborhood_graph(&transcripts, quadrant_size);

    println!("Built neighborhood graph with {} edges", adjacency.edge_count()/2);

    // can't just divide area by number of cells, because a large portion may have to cells.

    let priors = ModelPriors {
        min_cell_size: avg_edge_length,
        background_logprob: args.background_prob.ln(),
        foreground_logprob: (1_f32 - args.background_prob).ln(),
        μ_μ_a: (avg_edge_length * avg_edge_length * (ntranscripts as f32) / (ncells as f32)).ln(),
        σ_μ_a: 3.0_f32,
        α_σ_a: 0.1,
        β_σ_a: 0.1,
        α_w: 1.0,
        β_w: 1.0,
        e_r: 1.0,
        f_r: 1.0,
    };

    let mut seg = Segmentation::new(&transcripts, &nuclei_centroids, &adjacency);
    let mut sampler = Sampler::new(priors, &mut seg, args.ncomponents, ngenes, chunk_size);

    for i in 0..args.niter {
        for _ in 0..args.local_steps_per_iter {
            sampler.sample_local_updates(&seg);
            seg.apply_local_updates(&mut sampler);
        }
        sampler.sample_global_params();
        sampler.compute_cell_logprobs(&mut seg);

        if i % 100 == 0 {
            println!("Iteration {} ({} unassigned transcripts)", i, seg.nunassigned());
            // dbg!(&seg.cell_logprobs);
        }
    }

    {
        let file = File::create(&args.output_counts).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(transcript_names.iter()).unwrap();
        for row in sampler.counts().t().rows() {
            writer.write_record(row.iter().map(|x| x.to_string())).unwrap();
        }
    }

    {
        let file = File::create("z.csv.gz").unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(["z"]).unwrap();
        for z in sampler.z.iter() {
            writer.write_record([z.to_string()]).unwrap();
        }
    }

}
