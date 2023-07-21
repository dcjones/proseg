#![allow(confusable_idents)]

use clap::Parser;

mod sampler;

use itertools::Itertools;
use sampler::{Sampler, ModelPriors, ModelParams, ProposalStats, UncertaintyTracker};
use sampler::transcripts::{read_transcripts_csv, neighborhood_graph, coordinate_span, Transcript};
use sampler::hexbinsampler::CubeBinSampler;
use rayon::current_num_threads;
use indicatif::{ProgressBar, ProgressStyle};
use csv;
use std::fs::File;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::cell::RefCell;

// use signal_hook::{consts::SIGINT, iterator::Signals};
// use std::{error::Error, thread, time::Duration};


#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args{
    transcript_csv: String,
    // cell_centers_csv: String,

    #[arg(long, default_value="feature_name")]
    transcript_column: String,

    #[arg(long, default_value="x_location")]
    x_column: String,

    #[arg(long, default_value="y_location")]
    y_column: String,

    #[arg(short, long, default_value=None)]
    z_column: Option<String>,

    #[arg(short, long, default_value_t=20)]
    ncomponents: usize,

    #[arg(long, default_value_t=100000)]
    niter: usize,

    #[arg(short = 't', long, default_value=None)]
    nthreads: Option<usize>,

    #[arg(short, long, default_value_t=100)]
    local_steps_per_iter: usize,

    #[arg(long, default_value_t=0.5_f32)]
    count_pr_cutoff: f32,

    #[arg(short, long, default_value="counts.csv.gz")]
    output_counts: String,

    #[arg(long, default_value="cells.geojson.gz")]
    output_cells: Option<String>,

    #[arg(long, default_value=None)]
    output_cell_cubes: Option<String>,

    #[arg(long, default_value=None)]
    output_transcripts: Option<String>,
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

    let (transcript_names, mut transcripts, init_cell_assignments, init_cell_population) = read_transcripts_csv(
        &args.transcript_csv, &args.transcript_column, &args.x_column,
        &args.y_column, args.z_column.as_deref());
    let ngenes = transcript_names.len();
    let ntranscripts = transcripts.len();
    let ncells = init_cell_population.len();

    // Clamp transcript depth
    // This is we get some reasonable depth slices when we step up to
    // 3d sampling.
    let zs: Vec<f32> = transcripts.iter()
        .map(|t| t.z)
        .sorted_by(|a, b| a.partial_cmp(b).unwrap())
        .collect();

    let (q0, q1) = (0.01, 0.99);
    let zmin = zs[(q0 * (zs.len() as f32)) as usize];
    let zmax = zs[(q1 * (zs.len() as f32)) as usize];
    for t in &mut transcripts {
        t.z = t.z.max(zmin).min(zmax);
    }

    println!("Read {} transcripts", ntranscripts);

    let (xmin, xmax, ymin, ymax, zmin, zmax) = coordinate_span(&transcripts);
    let (xspan, yspan, zspan) = (xmax - xmin, ymax - ymin, zmax - zmin);

    let full_area = sampler::hull::compute_full_area(&transcripts) * zspan;
    println!("Full volume: {}", full_area);

    if let Some(nthreads) = args.nthreads {
        rayon::ThreadPoolBuilder::new().num_threads(nthreads).build_global().unwrap();
    }
    let nthreads = current_num_threads();
    println!("Using {} threads", nthreads);

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
    let (adjacency, transcript_areas, avg_edge_length) =
        neighborhood_graph(&transcripts, quadrant_size);

    println!("Built neighborhood graph with {} edges", adjacency.edge_count()/2);

    // can't just divide area by number of cells, because a large portion may have to cells.

    let min_cell_area = avg_edge_length;
    let min_cell_surface_area = 10.0_f32 * min_cell_area;

    let priors = ModelPriors {
        min_cell_area,
        min_cell_surface_area,

        μ_μ_area: (avg_edge_length * avg_edge_length * (ntranscripts as f32) / (ncells as f32)).ln(),
        σ_μ_area: 3.0_f32,
        α_σ_area: 0.1,
        β_σ_area: 0.1,

        μ_μ_comp: 2.0,
        σ_μ_comp: 1.0_f32,
        α_σ_comp: 0.1,
        β_σ_comp: 0.1,

        α_θ: 1.0,
        β_θ: 1.0,
        e_r: 1.0,
        f_r: 1.0,
    };

    let mut params = ModelParams::new(
        &priors,
        full_area,
        &transcripts,
        &init_cell_assignments,
        &init_cell_population,
        &transcript_areas,
        args.ncomponents,
        ncells,
        ngenes
    );

    // TODO: Need to somehow make this a command line argument.
    // let initial_avgbinpop = 64.0_f32;
    // let sampler_schedule = [
    //     // 200, // 64
    //     // 200, // 16
    //     // 200, // 2
    //     // 200, // 0.5
    //     ];

    let initial_avgbinpop = 16.0_f32;
    let sampler_schedule = [
        200, // 16
        200, // 2
        200, // 0.5
        ];

    let total_iterations = sampler_schedule.iter().sum::<usize>();
    let mut prog = ProgressBar::new(total_iterations as u64);
    prog.set_style(
        ProgressStyle::with_template("{eta_precise} {bar:60} | {msg}")
        .unwrap()
        .progress_chars("##-")
    );

    let mut uncertainty = UncertaintyTracker::new();

    let mut sampler = RefCell::new(CubeBinSampler::new(
        &priors,
        &mut params,
        &transcripts,
        ngenes,
        initial_avgbinpop,
        chunk_size
    ));

    if sampler_schedule.len() > 1 {
        run_hexbin_sampler(
            &mut prog,
            sampler.get_mut(),
            &priors,
            &mut params,
            &transcripts,
            sampler_schedule[0],
            args.local_steps_per_iter,
            None);

        for &niter in sampler_schedule[1..sampler_schedule.len()-1].iter() {
            sampler.replace_with(|sampler| sampler.double_resolution(&transcripts));
            run_hexbin_sampler(
                &mut prog,
                sampler.get_mut(),
                &priors,
                &mut params,
                &transcripts,
                niter,
                args.local_steps_per_iter,
                None);
        }
    }

    sampler.replace_with(|sampler| sampler.double_resolution(&transcripts));
    run_hexbin_sampler(
        &mut prog,
        sampler.get_mut(),
        &priors,
        &mut params,
        &transcripts,
        sampler_schedule[sampler_schedule.len()-1],
        args.local_steps_per_iter,
        Some(&mut uncertainty));

    prog.finish();

    uncertainty.finish(&params);
    let counts = uncertainty.max_posterior_transcript_counts(
        &params, &transcripts, args.count_pr_cutoff);

    if let Some(output_cell_cubes) = args.output_cell_cubes {
        sampler.borrow().write_cell_cubes(&output_cell_cubes);
    }

    {
        let file = File::create(&args.output_counts).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(transcript_names.iter()).unwrap();
        // for row in params.counts.t().rows() {
        for row in counts.t().rows() {
            writer.write_record(row.iter().map(|x| x.to_string())).unwrap();
        }
    }

    // TODO: Make a proper optional cell metadata csv with area/volume along with 
    {
        let file = File::create("z.csv.gz").unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(["z"]).unwrap();
        for z in params.z.iter() {
            writer.write_record([z.to_string()]).unwrap();
        }
    }

    if let Some(output_cells) = args.output_cells {
        params.write_cell_hulls(&transcripts, &counts, &output_cells);
    }

    if let Some(output_transcripts) = args.output_transcripts {
        let file = File::create(output_transcripts).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(["x", "y", "z", "gene", "assignment"]).unwrap();
        for (cell, transcript) in params.cell_assignments.iter().zip(&transcripts) {
            writer.write_record([
                transcript.x.to_string(),
                transcript.y.to_string(),
                transcript.z.to_string(),
                transcript_names[transcript.gene as usize].clone(),
                cell.to_string().to_string()]).unwrap();
        }
    }

}


fn run_hexbin_sampler(
        prog: &mut ProgressBar,
        sampler: &mut CubeBinSampler,
        priors: &ModelPriors,
        params: &mut ModelParams,
        transcripts: &Vec<Transcript>,
        niter: usize,
        local_steps_per_iter: usize,
        mut uncertainty: Option<&mut UncertaintyTracker>)
{
    sampler.sample_global_params(priors, params);
    let mut proposal_stats = ProposalStats::new();

    for i in 0..niter {
        for _ in 0..local_steps_per_iter {
            sampler.sample_cell_regions(priors, params, &mut proposal_stats, transcripts, &mut uncertainty);
        }
        sampler.sample_global_params(priors, params);

        let nassigned = params.nassigned();
        prog.inc(1);
        prog.set_message(format!(
            "log-likelihood: {ll} | assigned transcripts: {n_assigned} / {n} ({perc_assigned:.2}%)",
            ll=params.log_likelihood(),
            n_assigned=nassigned,
            n=transcripts.len(),
            perc_assigned=100.0 * (nassigned as f32) / (transcripts.len() as f32)
        ));

        // println!("Log likelihood: {}", params.log_likelihood());

        // let empty_cell_count = params.cell_population.iter().filter(|p| **p == 0).count();
        // println!("Empty cells: {}", empty_cell_count);

        // dbg!(&proposal_stats);
        proposal_stats.reset();

        // if i % 100 == 0 {
        //     println!("Iteration {} ({} unassigned transcripts)", i, params.nunassigned());
        // }
    }
}
