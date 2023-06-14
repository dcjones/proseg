
use clap::Parser;

mod sampler;

use sampler::{Sampler, Segmentation};
use sampler::transcripts::{read_transcripts_csv, read_nuclei_csv, neighborhood_graph};


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

    #[arg(short, long, default_value_t=1000)]
    niter: usize,
}


fn main() {
    let args = Args::parse();

    let (transcript_names, transcripts) = read_transcripts_csv(
        &args.transcript_csv, &args.transcript_column, &args.x_column,
        &args.y_column, args.z_column.as_deref());

    println!("Read {} transcripts", transcripts.len());

    let nuclei_centroids = read_nuclei_csv(
        &args.cell_centers_csv, &args.cell_x_column, &args.cell_y_column);

    println!("Read {} nuclei centroids", nuclei_centroids.len());

    let adjacency = neighborhood_graph(&transcripts);

    println!("Built neighborhood graph with {} edges", adjacency.nnz());

    let mut seg = Segmentation::new(&transcripts, &nuclei_centroids, &adjacency);
    let mut sampler = Sampler::new(&seg);

    for i in 0..args.niter {
        sampler.sample_local_updates(&seg);
        seg.apply_local_updates(&sampler);
        sampler.sample_global_params(&seg);
        if i % 100 == 0 {
            println!("Iteration {}", i);
        }
    }

    // TODO: Run the sampler
}
