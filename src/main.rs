#![allow(confusable_idents)]

use clap::Parser;

mod sampler;

use itertools::Itertools;
use sampler::{Sampler, ModelPriors, ModelParams, ProposalStats, UncertaintyTracker};
use sampler::transcripts::{read_transcripts_csv, coordinate_span, Transcript};
use sampler::hexbinsampler::CubeBinSampler;
use sampler::hull::compute_cell_areas;
use rayon::current_num_threads;
use indicatif::{ProgressBar, ProgressStyle};
use csv;
use std::fs::File;
use flate2::Compression;
use flate2::write::GzEncoder;
use std::cell::RefCell;
use std::sync::Arc;
use arrow;

// use std::{error::Error, thread, time::Duration};


#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args{
    transcript_csv: String,
    // cell_centers_csv: String,

    #[arg(long, default_value="feature_name")]
    transcript_column: String,

    #[arg(short, long, default_value="x_location")]
    x_column: String,

    #[arg(short, long, default_value="y_location")]
    y_column: String,

    #[arg(short, long, default_value="z_location")]
    z_column: Option<String>,

    #[arg(short, long, default_value_t=10)]
    ncomponents: usize,

    // #[arg(long, default_value_t=100000)]
    // niter: usize,

    #[arg(long, default_value_t=8.0_f32)]
    inital_bin_population: f32,

    // 32, 4, 1
    // 16, 2, 0.25
    // #[arg(long, num_args=1.., value_delimiter=',', default_values_t=[0, 200, 200, 200])]
    #[arg(long, num_args=1.., value_delimiter=',', default_values_t=[100, 100, 100])]
    schedule: Vec<usize>,

    #[arg(short = 't', long, default_value=None)]
    nthreads: Option<usize>,

    #[arg(short, long, default_value_t=1000)]
    local_steps_per_iter: usize,

    #[arg(long, default_value_t=0.5_f32)]
    count_pr_cutoff: f32,

    #[arg(long, default_value_t=6.0_f32)]
    perimeter_bound: f32,

    #[arg(long, default_value_t=1e-2_f32)]
    nuclear_reassignment_prob: f32,

    #[arg(long, default_value="counts.csv.gz")]
    output_counts_csv: Option<String>,

    #[arg(long, default_value="counts.arrow")]
    output_counts_arrow: Option<String>,

    #[arg(long, default_value="cells.geojson.gz")]
    output_cells: Option<String>,

    #[arg(long, default_value="cell_metadata.csv.gz")]
    output_cell_metadata: Option<String>,

    #[arg(long, default_value=None)]
    output_normalized_expression: Option<String>,

    #[arg(long, default_value=None)]
    output_cell_cubes_csv: Option<String>,

    #[arg(long, default_value=None)]
    output_cell_cubes_arrow: Option<String>,

    #[arg(long, default_value=None)]
    output_transcripts: Option<String>,

    #[arg(long, default_value_t=false)]
    check_consistency: bool,
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

    let full_volume = sampler::hull::compute_full_area(&transcripts) * zspan;
    println!("Full volume: {}", full_volume);

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

    println!("Using grid size {}. Chunks: {}", chunk_size, nchunks(chunk_size, xspan, yspan));

    let nucleus_areas = compute_cell_areas(ncells, &transcripts, &init_cell_assignments);
    let mean_nucleus_area = nucleus_areas.iter().sum::<f32>() / (ncells as f32);

    let min_cell_volume = 1e-6 * mean_nucleus_area * zspan;

    let priors = ModelPriors {
        min_cell_volume,

        μ_μ_volume: (2.0 * mean_nucleus_area * zspan).ln(),
        σ_μ_volume: 3.0_f32,
        α_σ_volume: 0.1,
        β_σ_volume: 0.1,

        α_θ: 1.0,
        β_θ: 1.0,
        e_r: 1.0,
        f_r: 1.0,

        perimeter_eta: 5.3,
        perimeter_bound: args.perimeter_bound,

        nuclear_reassignment_log_prob: args.nuclear_reassignment_prob.ln(),
        nuclear_reassignment_1mlog_prob: (1.0 - args.nuclear_reassignment_prob).ln(),
    };

    let mut params = ModelParams::new(
        &priors,
        full_volume,
        &transcripts,
        &init_cell_assignments,
        &init_cell_population,
        args.ncomponents,
        ncells,
        ngenes
    );

    let total_iterations = args.schedule.iter().sum::<usize>();
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
        args.inital_bin_population,
        chunk_size
    ));

    if args.schedule.len() > 1 {
        run_hexbin_sampler(
            &mut prog,
            sampler.get_mut(),
            &priors,
            &mut params,
            &transcripts,
            args.schedule[0],
            args.local_steps_per_iter,
            None);

        for &niter in args.schedule[1..args.schedule.len()-1].iter() {
            if args.check_consistency {
                sampler.borrow_mut().check_consistency(&priors, &mut params);
            }

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
        if args.check_consistency {
            sampler.borrow_mut().check_consistency(&priors, &mut params);
        }
        sampler.replace_with(|sampler| sampler.double_resolution(&transcripts));
    }

    run_hexbin_sampler(
        &mut prog,
        sampler.get_mut(),
        &priors,
        &mut params,
        &transcripts,
        args.schedule[args.schedule.len()-1],
        args.local_steps_per_iter,
        Some(&mut uncertainty));

    if args.check_consistency {
        sampler.borrow_mut().check_consistency(&priors, &mut params);
    }
    prog.finish();

    uncertainty.finish(&params);
    let counts = uncertainty.max_posterior_transcript_counts(
        &params, &transcripts, args.count_pr_cutoff);

    if let Some(output_cell_cubes_csv) = args.output_cell_cubes_csv {
        sampler.borrow().write_cell_cubes_csv(&output_cell_cubes_csv);
    }

    if let Some(output_cell_cubes_arrow) = args.output_cell_cubes_arrow {
        sampler.borrow().write_cell_cubes_arrow(&output_cell_cubes_arrow);
    }

    if let Some(output_counts_csv) = args.output_counts_csv {
        let file = File::create(&output_counts_csv).unwrap();
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

    if let Some(output_counts_arrow) = args.output_counts_arrow {
        let schema = arrow::datatypes::Schema::new(
            transcript_names.iter().map(
                |name| arrow::datatypes::Field::new(name, arrow::datatypes::DataType::UInt32, false))
                .collect::<Vec<_>>());

        // unimplemented!();
        let file = File::create(&output_counts_arrow).unwrap();
        // let mut writer = arrow::ipc::writer::FileWriter::try_new(file, &schema).unwrap();
        let mut writer = arrow::ipc::writer::FileWriter::try_new_with_options(
            file, &schema,
            arrow::ipc::writer::IpcWriteOptions::default().try_with_compression(
                Some(arrow::ipc::CompressionType::ZSTD)).unwrap(),
        ).unwrap();

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        for row in counts.rows() {
            columns.push(Arc::new(arrow::array::UInt32Array::from_iter(row.iter().cloned())));
        }

        let recbatch = arrow::record_batch::RecordBatch::try_new(
            Arc::new(schema), columns,
        ).unwrap();

        writer.write(&recbatch).unwrap();
    }

    if let Some(output_normalized_expression) = args.output_normalized_expression {
        let file = File::create(output_normalized_expression).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(transcript_names.iter()).unwrap();
        for row in params.λ.t().rows() {
            writer.write_record(row.iter().map(|x| x.to_string())).unwrap();
        }
    }

    if let Some(output_cell_metadata) = args.output_cell_metadata {
        let file = File::create(output_cell_metadata).unwrap();
        let encoder = GzEncoder::new(file, Compression::default());
        let mut writer = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(encoder);

        writer.write_record(["cell", "cluster", "volume", "population"]).unwrap();
        for cell in 0..ncells {
            let cluster = params.z[cell];
            let volume = params.cell_volume[cell];
            let population = params.cell_population[cell];
            writer.write_record([
                cell.to_string(),
                cluster.to_string(),
                volume.to_string(),
                population.to_string(),
                ]
            ).unwrap();
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

    for _ in 0..niter {
        for _ in 0..local_steps_per_iter {
            sampler.sample_cell_regions(priors, params, &mut proposal_stats, transcripts, &mut uncertainty);
        }
        sampler.sample_global_params(priors, params);

        let nassigned = params.nassigned();
        prog.inc(1);
        prog.set_message(format!(
            "log-likelihood: {ll} | assigned transcripts: {n_assigned} / {n} ({perc_assigned:.2}%)",
            ll=params.log_likelihood(priors),
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
