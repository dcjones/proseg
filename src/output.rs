use arrow::array::RecordBatch;
use arrow::csv;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use flate2::Compression;
use flate2::write::GzEncoder;
use geo::MultiPolygon;
use ndarray::Array2;
use num::traits::Zero;
use ordered_float::OrderedFloat;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression::ZSTD, ZstdLevel};
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use crate::sampler::transcripts::BACKGROUND_CELL;

use super::sampler::ModelParams;
use super::sampler::runvec::RunVec;
use super::sampler::sparsemat::SparseMat;
use super::sampler::transcripts::Transcript;
use super::sampler::voxelcheckerboard::{TranscriptMetadata, VoxelCheckerboard};
// use crate::schemas::{transcript_metadata_schema, OutputFormat};
// use crate::schemas::OutputFormat;

use clap::ValueEnum;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OutputFormat {
    Infer,
    Csv,
    CsvGz,
    Parquet,
}

// We need this for reading, because csv can't read LargeUtf8 for some reason
// fn large_utf8_if_parquet(fmt: OutputFormat) -> DataType {
//     match fmt {
//         OutputFormat::Parquet => DataType::LargeUtf8,
//         _ => DataType::Utf8,
//     }
// }

pub fn write_table(
    output_path: &Option<String>,
    filename: &str,
    fmt: OutputFormat,
    batch: &RecordBatch,
) {
    let fmt = match fmt {
        OutputFormat::Infer => infer_format_from_filename(filename),
        _ => fmt,
    };

    let mut file = if let Some(output_path) = output_path {
        File::create(Path::new(output_path).join(filename)).unwrap()
    } else {
        File::create(filename).unwrap()
    };

    match fmt {
        OutputFormat::Csv => {
            if write_table_csv(&mut file, batch).is_err() {
                panic!("Error writing csv file: {}", filename);
            }
        }
        OutputFormat::CsvGz => {
            let mut encoder = GzEncoder::new(file, Compression::default());
            if write_table_csv(&mut encoder, batch).is_err() {
                panic!("Error writing csv.gz file: {}", filename);
            }
        }
        OutputFormat::Parquet => {
            if write_table_parquet(&mut file, batch).is_err() {
                panic!("Error writing parquet file: {}", filename);
            }
        }
        OutputFormat::Infer => {
            panic!("Cannot infer output format for filename: {}", filename);
        }
    }
}

fn write_table_csv<W>(output: &mut W, batch: &RecordBatch) -> Result<(), ArrowError>
where
    W: std::io::Write,
{
    let mut writer = csv::WriterBuilder::new().with_header(true).build(output);
    writer.write(batch)
}

fn write_table_parquet<W>(output: &mut W, batch: &RecordBatch) -> Result<(), ParquetError>
where
    W: std::io::Write + Send,
{
    // TODO: Any non-defaults?
    let props = WriterProperties::builder()
        .set_compression(ZSTD(ZstdLevel::try_new(3).unwrap()))
        .build();

    let mut writer = ArrowWriter::try_new(output, batch.schema(), Some(props)).unwrap();
    writer.write(batch)?;
    writer.close()?;

    Ok(())
}

pub fn infer_format_from_filename(filename: &str) -> OutputFormat {
    if filename.ends_with(".csv.gz") {
        OutputFormat::CsvGz
    } else if filename.ends_with(".csv") {
        OutputFormat::Csv
    } else if filename.ends_with(".parquet") {
        OutputFormat::Parquet
    } else {
        panic!("Unknown file format for filename: {}", filename);
    }
}
//
// TODO: We should really just write sparse matrices in 10Xs h5 format (or maybe
// just mtx.gz to avoid the hdf5 dependency). but maybe we try to keep these
// outputs for reverse compatibility.

// pub fn write_counts(
//     output_path: &Option<String>,
//     output_counts: &Option<String>,
//     output_counts_fmt: OutputFormat,
//     gene_names: &[String],
//     counts: &SparseMat<u32, u32>,
// ) {
//     if let Some(output_counts) = output_counts {
//         let schema = Schema::new(
//             gene_names
//                 .iter()
//                 .map(|name| Field::new(name, DataType::UInt32, false))
//                 .collect::<Vec<Field>>(),
//         );

//         let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
//         for row in counts.rows() {
//             let row_lock = row.read();
//             columns.push(Arc::new(
//                 row_lock.iter().collect::<arrow::array::UInt32Array>(),
//             ));
//         }

//         let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

//         write_table(output_path, output_counts, output_counts_fmt, &batch);
//     }
// }

pub fn write_sparse_mtx<T>(
    output_path: &Option<String>,
    filename: &Option<String>,
    mat: &SparseMat<T, u32>,
) where
    T: Copy + Clone + Zero + PartialEq + Display,
{
    if filename.is_none() {
        return;
    }
    let filename = filename.as_ref().unwrap();

    let file = if let Some(output_path) = output_path {
        File::create(Path::new(output_path).join(filename)).unwrap()
    } else {
        File::create(filename).unwrap()
    };
    let mut encoder = GzEncoder::new(file, Compression::default());

    // count the true number of nonzeros
    let mut nnzs = 0;
    for x_c in mat.rows() {
        for (_g, est) in x_c.read().iter_nonzeros() {
            if est != T::zero() {
                nnzs += 1;
            }
        }
    }

    write!(
        &mut encoder,
        "%%MatrixMarket matrix coordinate real general\n{} {} {}\n",
        mat.m, mat.n, nnzs
    )
    .unwrap();

    for (c, x_c) in mat.rows().enumerate() {
        for (g, est) in x_c.read().iter_nonzeros() {
            if est != T::zero() {
                writeln!(&mut encoder, "{} {} {:.2}", c + 1, g + 1, est).unwrap();
            }
        }
    }
}

// pub fn write_expected_counts(
//     output_path: &Option<String>,
//     output_expected_counts: &Option<String>,
//     output_expected_counts_fmt: OutputFormat,
//     gene_names: &[String],
//     ecounts: &CountMeanEstimator,
// ) {
//     if let Some(output_expected_counts) = output_expected_counts {
//         let schema = Schema::new(
//             gene_names
//                 .iter()
//                 .map(|name| Field::new(name, DataType::Float32, false))
//                 .collect::<Vec<Field>>(),
//         );

//         let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
//         for row in ecounts.estimates.rows() {
//             let row_lock = row.read();
//             columns.push(Arc::new(
//                 row_lock.iter().collect::<arrow::array::Float32Array>(),
//             ));
//         }

//         let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

//         write_table(
//             output_path,
//             output_expected_counts,
//             output_expected_counts_fmt,
//             &batch,
//         );
//     }
// }

pub fn write_metagene_rates(
    output_path: &Option<String>,
    output_metagene_rates: &Option<String>,
    output_metagene_rates_fmt: OutputFormat,
    φ: &Array2<f32>,
) {
    if let Some(output_metagene_rates) = output_metagene_rates {
        let k = φ.shape()[1];
        let schema = Schema::new(
            (0..k)
                .map(|name| Field::new(format!("phi{}", name), DataType::Float32, false))
                .collect::<Vec<Field>>(),
        );

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        for row in φ.columns() {
            columns.push(Arc::new(
                row.iter().cloned().collect::<arrow::array::Float32Array>(),
            ));
        }

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(
            output_path,
            output_metagene_rates,
            output_metagene_rates_fmt,
            &batch,
        );
    }
}

pub fn write_metagene_loadings(
    output_path: &Option<String>,
    output_metagene_rates: &Option<String>,
    output_metagene_rates_fmt: OutputFormat,
    transcript_names: &[String],
    θ: &Array2<f32>,
) {
    if let Some(output_metagene_rates) = output_metagene_rates {
        let k = θ.shape()[1];

        let mut schema = vec![Field::new("gene", DataType::Utf8, false)];

        for i in 0..k {
            schema.push(Field::new(format!("theta{}", i), DataType::Float32, false));
        }

        let schema = Schema::new(schema);

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();

        columns.push(Arc::new(
            transcript_names
                .iter()
                .map(Some)
                .collect::<arrow::array::StringArray>(),
        ));

        for row in θ.columns() {
            columns.push(Arc::new(
                row.iter().cloned().collect::<arrow::array::Float32Array>(),
            ));
        }

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(
            output_path,
            output_metagene_rates,
            output_metagene_rates_fmt,
            &batch,
        );
    }
}

// TODO: I should write a function to do the full matmul and output the results
// just for debugcging.
/*
pub fn write_rates(
    output_path: &Option<String>,
    output_rates: &Option<String>,
    output_rates_fmt: OutputFormat,
    params: &ModelParams,
    transcript_names: &[String],
) {
    if let Some(output_rates) = output_rates {
        let schema = Schema::new(
            transcript_names
                .iter()
                .map(|name| Field::new(name, DataType::Float32, false))
                .collect::<Vec<Field>>(),
        );

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        for row in params.λ.columns() {
            columns.push(Arc::new(
                row.iter().cloned().collect::<arrow::array::Float32Array>(),
            ));
        }

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(output_path, output_rates, output_rates_fmt, &batch);
    }
}
*/

// pub fn write_component_params(
//     output_component_params: &Option<String>,
//     output_component_params_fmt: OutputFormat,
//     params: &ModelParams,
//     transcript_names: &[String],
// ) {
//     if let Some(output_component_params) = output_component_params {
//         // What does this look like: rows for each gene, columns for α1, β1, α2, β2, etc.
//         let α = &params.r;
//         let φ = &params.φ;
//         let β = φ.map(|φ| (-φ).exp());
//         let ncomponents = params.ncomponents();
//
//         let mut fields = Vec::new();
//         fields.push(Field::new("gene", DataType::Utf8, false));
//         for i in 0..ncomponents {
//             fields.push(Field::new(&format!("α_{}", i), DataType::Float32, false));
//             fields.push(Field::new(&format!("β_{}", i), DataType::Float32, false));
//         }
//         let schema = Schema::new(fields);
//
//         let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
//         columns.push(Arc::new(arrow::array::StringArray::from(
//             transcript_names.iter().cloned().collect::<Vec<String>>(),
//         )));
//
//         Zip::from(α.rows()).and(β.rows()).for_each(|α, β| {
//             columns.push(Arc::new(
//                 α.iter().cloned().collect::<arrow::array::Float32Array>(),
//             ));
//             columns.push(Arc::new(
//                 β.iter().cloned().collect::<arrow::array::Float32Array>(),
//             ));
//         });
//
//         let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();
//
//         write_table(output_component_params, output_component_params_fmt, &batch);
//     }
// }

pub fn write_cell_metadata(
    output_path: &Option<String>,
    output_cell_metadata: &Option<String>,
    output_cell_metadata_fmt: OutputFormat,
    params: &ModelParams,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
    // fovs: &RunVec<usize, u32>,
    // fov_names: &[String],
) {
    let ncells = cell_centroids.shape()[0];
    // let nfovs = fov_names.len();

    // TODO: I can no longer do this because I don't know what transcripts are
    // assigned to what fovs. I think the solution if I really want to do this
    // is to work out fov bounds from the transcripts then see what bound each
    // cell centroid is in.
    // let cell_fovs = cell_fov_vote(ncells, nfovs, cell_assignments, fovs);
    //

    if let Some(output_cell_metadata) = output_cell_metadata {
        let schema = Schema::new(vec![
            Field::new("cell", DataType::UInt32, false),
            Field::new("original_cell_id", DataType::Utf8, false),
            Field::new("centroid_x", DataType::Float32, false),
            Field::new("centroid_y", DataType::Float32, false),
            Field::new("centroid_z", DataType::Float32, false),
            // Field::new("fov", DataType::Utf8, true),
            Field::new("cluster", DataType::UInt16, false),
            Field::new("volume", DataType::Float32, false),
            Field::new("surface_area", DataType::UInt32, false),
            Field::new("scale", DataType::Float32, false),
            // Field::new("population", DataType::UInt64, false),
        ]);

        let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(
                (0..ncells)
                    .map(|cell| Some(cell as u32))
                    .collect::<arrow::array::UInt32Array>(),
            ),
            Arc::new(
                original_cell_ids
                    .iter()
                    .map(|id| Some(id.to_string()))
                    .collect::<arrow::array::StringArray>(),
            ),
            Arc::new(
                cell_centroids
                    .rows()
                    .into_iter()
                    .map(|row| row[0])
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                cell_centroids
                    .rows()
                    .into_iter()
                    .map(|row| row[1])
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                cell_centroids
                    .rows()
                    .into_iter()
                    .map(|row| row[2])
                    .collect::<arrow::array::Float32Array>(),
            ),
            // Arc::new(
            //     cell_fovs
            //         .iter()
            //         .map(|fov| {
            //             if *fov == u32::MAX {
            //                 None
            //             } else {
            //                 Some(fov_names[*fov as usize].clone())
            //             }
            //         })
            //         .collect::<arrow::array::StringArray>(),
            // ),
            Arc::new(
                params
                    .z
                    .iter()
                    .map(|&z| z as u16)
                    .collect::<arrow::array::UInt16Array>(),
            ),
            Arc::new(
                params
                    .cell_voxel_count
                    .iter()
                    .map(|v| v as f32 * params.voxel_volume)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                params
                    .cell_surface_area
                    .iter()
                    .collect::<arrow::array::UInt32Array>(),
            ),
            Arc::new(
                params
                    .cell_scale
                    .iter()
                    .cloned()
                    .collect::<arrow::array::Float32Array>(),
            ),
            // Arc::new(
            //     params
            //         .cell_population
            //         .iter()
            //         .map(|p| p as u64)
            //         .collect::<arrow::array::UInt64Array>(),
            // ),
        ];

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(
            output_path,
            output_cell_metadata,
            output_cell_metadata_fmt,
            &batch,
        );
    }
}

pub fn write_transcript_metadata(
    output_path: &Option<String>,
    output_transcript_metadata: &Option<String>,
    output_transcript_metadata_fmt: OutputFormat,
    voxels: &VoxelCheckerboard,
    transcripts: &RunVec<u32, Transcript>,
    metadata: &RunVec<u32, TranscriptMetadata>,
    gene_names: &[String],
) {
    if output_transcript_metadata.is_none() {
        return;
    }
    let output_transcript_metadata = output_transcript_metadata.as_ref().unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("observed_x", DataType::Float32, false),
        Field::new("observed_y", DataType::Float32, false),
        Field::new("observed_z", DataType::Float32, false),
        Field::new("gene", DataType::LargeUtf8, false),
        Field::new("assignment", DataType::UInt32, true),
        Field::new("background", DataType::Boolean, false),
    ]));

    let fmt = match output_transcript_metadata_fmt {
        OutputFormat::Infer => infer_format_from_filename(output_transcript_metadata),
        _ => output_transcript_metadata_fmt,
    };

    let output = if let Some(output_path) = output_path {
        File::create(Path::new(output_path).join(output_transcript_metadata)).unwrap()
    } else {
        File::create(output_transcript_metadata).unwrap()
    };

    match fmt {
        OutputFormat::Csv => {
            let mut writer = csv::WriterBuilder::new().with_header(true).build(output);
            write_transcript_metadata_with_fn(
                schema.clone(),
                voxels,
                transcripts,
                metadata,
                gene_names,
                |batch| writer.write(batch).unwrap(),
            );
        }
        OutputFormat::CsvGz => {
            let encoder = GzEncoder::new(output, Compression::default());
            let mut writer = csv::WriterBuilder::new().with_header(true).build(encoder);
            write_transcript_metadata_with_fn(
                schema.clone(),
                voxels,
                transcripts,
                metadata,
                gene_names,
                |batch| writer.write(batch).unwrap(),
            );
        }
        OutputFormat::Parquet => {
            let props = WriterProperties::builder()
                .set_column_dictionary_enabled("gene".into(), true)
                .set_compression(ZSTD(ZstdLevel::try_new(3).unwrap()))
                .build();
            let mut writer = ArrowWriter::try_new(output, schema.clone(), Some(props)).unwrap();
            write_transcript_metadata_with_fn(
                schema.clone(),
                voxels,
                transcripts,
                metadata,
                gene_names,
                |batch| writer.write(batch).unwrap(),
            );
            writer.close().unwrap();
        }
        OutputFormat::Infer => {
            panic!("Cannot infer output format for filename: {output_transcript_metadata}");
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn write_transcript_metadata_with_fn<F: FnMut(&RecordBatch)>(
    schema: Arc<Schema>,
    voxels: &VoxelCheckerboard,
    transcripts: &RunVec<u32, Transcript>,
    metadata: &RunVec<u32, TranscriptMetadata>,
    gene_names: &[String],
    mut write_batch: F,
) {
    // This is one table I probably should try to write in batches.
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    let mut observed_x = Vec::new();
    let mut observed_y = Vec::new();
    let mut observed_z = Vec::new();
    let mut gene = Vec::new();
    let mut assignment = Vec::new();
    let mut background = Vec::new();

    const BATCH_SIZE: usize = 65536;
    let mut count = 0;
    for (transcript, metadata) in transcripts.iter().zip(metadata.iter()) {
        let [dx, dy, dz] = metadata.offset.coords();

        x.push(transcript.x + (dx as f32) * voxels.voxelsize);
        y.push(transcript.y + (dy as f32) * voxels.voxelsize);
        z.push(transcript.z + (dz as f32) * voxels.voxelsize_z);
        observed_x.push(transcript.x);
        observed_y.push(transcript.y);
        observed_z.push(transcript.z);
        gene.push(gene_names[transcript.gene as usize].clone());
        assignment.push(if metadata.cell == BACKGROUND_CELL {
            None
        } else {
            Some(metadata.cell)
        });
        background.push(metadata.cell == BACKGROUND_CELL || !metadata.foreground);

        count += 1;
        if count == BATCH_SIZE {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(x.drain(..).collect::<arrow::array::Float32Array>()),
                    Arc::new(y.drain(..).collect::<arrow::array::Float32Array>()),
                    Arc::new(z.drain(..).collect::<arrow::array::Float32Array>()),
                    Arc::new(observed_x.drain(..).collect::<arrow::array::Float32Array>()),
                    Arc::new(observed_y.drain(..).collect::<arrow::array::Float32Array>()),
                    Arc::new(observed_z.drain(..).collect::<arrow::array::Float32Array>()),
                    Arc::new(
                        gene.drain(..)
                            .map(Some)
                            .collect::<arrow::array::LargeStringArray>(),
                    ),
                    Arc::new(assignment.drain(..).collect::<arrow::array::UInt32Array>()),
                    Arc::new(
                        background
                            .drain(..)
                            .map(Some)
                            .collect::<arrow::array::BooleanArray>(),
                    ),
                ],
            )
            .unwrap();

            write_batch(&batch);
        }
    }

    if count > 0 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(x.drain(..).collect::<arrow::array::Float32Array>()),
                Arc::new(y.drain(..).collect::<arrow::array::Float32Array>()),
                Arc::new(z.drain(..).collect::<arrow::array::Float32Array>()),
                Arc::new(observed_x.drain(..).collect::<arrow::array::Float32Array>()),
                Arc::new(observed_y.drain(..).collect::<arrow::array::Float32Array>()),
                Arc::new(observed_z.drain(..).collect::<arrow::array::Float32Array>()),
                Arc::new(
                    gene.drain(..)
                        .map(Some)
                        .collect::<arrow::array::LargeStringArray>(),
                ),
                Arc::new(assignment.drain(..).collect::<arrow::array::UInt32Array>()),
                Arc::new(
                    background
                        .drain(..)
                        .map(Some)
                        .collect::<arrow::array::BooleanArray>(),
                ),
            ],
        )
        .unwrap();

        write_batch(&batch);
    }
}

pub fn write_gene_metadata(
    output_path: &Option<String>,
    output_gene_metadata: &Option<String>,
    output_gene_metadata_fmt: OutputFormat,
    params: &ModelParams,
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
    expected_counts: &SparseMat<f32, u32>,
) {
    let ngenes = gene_names.len();
    let mut total_gene_counts = vec![0; ngenes];
    for transcript_run in transcripts.iter_runs() {
        total_gene_counts[transcript_run.value.gene as usize] += transcript_run.len as usize;
    }

    let mut gene_expected_counts = vec![0_f32; ngenes];
    for x_c in expected_counts.rows() {
        for (g, x_cg) in x_c.read().iter_nonzeros() {
            gene_expected_counts[g as usize] += x_cg;
        }
    }

    if let Some(output_gene_metadata) = output_gene_metadata {
        let mut schema_fields = vec![
            Field::new("gene", DataType::Utf8, false),
            Field::new("total_count", DataType::UInt64, false),
            Field::new("expected_assigned_count", DataType::Float32, false),
        ];

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(
                gene_names
                    .iter()
                    .map(|s| Some(s.clone()))
                    .collect::<arrow::array::StringArray>(),
            ),
            Arc::new(
                total_gene_counts
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<arrow::array::UInt64Array>(),
            ),
            Arc::new(
                gene_expected_counts
                    .iter()
                    .cloned()
                    .collect::<arrow::array::Float32Array>(),
            ),
        ];

        // background rates
        for i in 0..params.nlayers() {
            schema_fields.push(Field::new(format!("λ_bg_{}", i), DataType::Float32, false));
            columns.push(Arc::new(
                params
                    .λ_bg
                    .column(i)
                    .iter()
                    .cloned()
                    .collect::<arrow::array::Float32Array>(),
            ));
        }

        let schema = Schema::new(schema_fields);
        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(
            output_path,
            output_gene_metadata,
            output_gene_metadata_fmt,
            &batch,
        );
    }
}

/*
pub fn write_voxels(
    output_path: &Option<String>,
    output_voxels: &Option<String>,
    output_voxels_fmt: OutputFormat,
    sampler: &VoxelSampler,
) {
    if let Some(output_voxels) = output_voxels {
        let nvoxels = sampler.voxels().count();

        let mut cells = Vec::with_capacity(nvoxels);
        let mut x0s = Vec::with_capacity(nvoxels);
        let mut y0s = Vec::with_capacity(nvoxels);
        let mut z0s = Vec::with_capacity(nvoxels);
        let mut x1s = Vec::with_capacity(nvoxels);
        let mut y1s = Vec::with_capacity(nvoxels);
        let mut z1s = Vec::with_capacity(nvoxels);

        for (cell, (x0, y0, z0, x1, y1, z1)) in sampler.voxels() {
            cells.push(cell);
            x0s.push(x0);
            y0s.push(y0);
            z0s.push(z0);
            x1s.push(x1);
            y1s.push(y1);
            z1s.push(z1);
        }

        let schema = Schema::new(vec![
            Field::new("cell", DataType::UInt32, false),
            Field::new("x0", DataType::Float32, false),
            Field::new("y0", DataType::Float32, false),
            Field::new("z0", DataType::Float32, false),
            Field::new("x1", DataType::Float32, false),
            Field::new("y1", DataType::Float32, false),
            Field::new("z1", DataType::Float32, false),
        ]);

        let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(cells.iter().cloned().collect::<arrow::array::UInt32Array>()),
            Arc::new(x0s.iter().cloned().collect::<arrow::array::Float32Array>()),
            Arc::new(y0s.iter().cloned().collect::<arrow::array::Float32Array>()),
            Arc::new(z0s.iter().cloned().collect::<arrow::array::Float32Array>()),
            Arc::new(x1s.iter().cloned().collect::<arrow::array::Float32Array>()),
            Arc::new(y1s.iter().cloned().collect::<arrow::array::Float32Array>()),
            Arc::new(z1s.iter().cloned().collect::<arrow::array::Float32Array>()),
        ];

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(output_path, output_voxels, output_voxels_fmt, &batch);
    }
}
*/

// TODO:
// If we want to import things into qupath, I think we need a way to scale
// the coordinates to pixel space. It also doesn't seem like it supports
// MultiPolygons, so we need to write each polygon in a cell to a separate Polygon entry.

pub fn write_cell_multipolygons(
    output_path: &Option<String>,
    output_cell_polygons: &Option<String>,
    polygons: &Vec<MultiPolygon<f32>>,
) {
    if let Some(output_cell_polygons) = output_cell_polygons {
        let file = if let Some(output_path) = output_path {
            File::create(Path::new(output_path).join(output_cell_polygons)).unwrap()
        } else {
            File::create(output_cell_polygons).unwrap()
        };
        let mut encoder = GzEncoder::new(file, Compression::default());

        // memoize conversion of voxel coordinates to string for efficiency
        let mut coord_strings: HashMap<OrderedFloat<f32>, String> = HashMap::new();

        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

        let ncells = polygons.len();
        for (cell, polys) in polygons.into_iter().enumerate() {
            writeln!(
                encoder,
                concat!(
                    "    {{\n",
                    "      \"type\": \"Feature\",\n",
                    "      \"properties\": {{\n",
                    "        \"cell\": {}\n",
                    "      }},\n",
                    "      \"geometry\": {{\n",
                    "        \"type\": \"MultiPolygon\",\n",
                    "        \"coordinates\": ["
                ),
                cell
            )
            .unwrap();

            let npolys = polys.iter().count();
            for (i, poly) in polys.into_iter().enumerate() {
                writeln!(encoder, concat!("          [\n", "            [")).unwrap();

                let ncoords = poly.exterior().coords().count();
                for (j, coord) in poly.exterior().coords().enumerate() {
                    let x_str = coord_strings
                        .entry(OrderedFloat(coord.x))
                        .or_insert_with(|| coord.x.to_string());
                    write!(encoder, "              [{}, ", x_str).unwrap();

                    let y_str = coord_strings
                        .entry(OrderedFloat(coord.y))
                        .or_insert_with(|| coord.y.to_string());
                    write!(encoder, "{}]", y_str).unwrap();

                    if j < ncoords - 1 {
                        writeln!(encoder, ",").unwrap();
                    } else {
                        writeln!(encoder).unwrap();
                    }
                }

                write!(encoder, concat!("            ]\n", "          ]")).unwrap();

                if i < npolys - 1 {
                    writeln!(encoder, ",").unwrap();
                } else {
                    writeln!(encoder).unwrap();
                }
            }

            write!(encoder, concat!("        ]\n", "      }}\n", "    }}")).unwrap();
            if cell < ncells - 1 {
                writeln!(encoder, ",").unwrap();
            } else {
                writeln!(encoder).unwrap();
            }
        }

        writeln!(encoder, "  ]\n}}").unwrap();
    }
}

pub fn write_cell_layered_multipolygons(
    output_path: &Option<String>,
    output_cell_polygons: &Option<String>,
    polygons: &Vec<Vec<(i32, MultiPolygon<f32>)>>,
) {
    if let Some(output_cell_polygons) = output_cell_polygons {
        let file = if let Some(output_path) = output_path {
            File::create(Path::new(output_path).join(output_cell_polygons)).unwrap()
        } else {
            File::create(output_cell_polygons).unwrap()
        };
        let mut encoder = GzEncoder::new(file, Compression::default());

        // memoize conversion of voxel coordinates to string for efficiency
        let mut coord_strings: HashMap<OrderedFloat<f32>, String> = HashMap::new();

        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

        let mut nmultipolys = 0;
        for cell_polys in polygons.iter() {
            nmultipolys += cell_polys.len();
        }

        let mut count = 0;
        for (cell, cell_polys) in polygons.iter().enumerate() {
            for (layer, polys) in cell_polys.iter() {
                writeln!(
                    encoder,
                    concat!(
                        "    {{\n",
                        "      \"type\": \"Feature\",\n",
                        "      \"properties\": {{\n",
                        "        \"cell\": {},\n",
                        "        \"layer\": {}\n",
                        "      }},\n",
                        "      \"geometry\": {{\n",
                        "        \"type\": \"MultiPolygon\",\n",
                        "        \"coordinates\": ["
                    ),
                    cell, layer
                )
                .unwrap();

                let npolys = polys.iter().count();
                for (i, poly) in polys.into_iter().enumerate() {
                    writeln!(encoder, concat!("          [\n", "            [")).unwrap();

                    let ncoords = poly.exterior().coords().count();
                    for (j, coord) in poly.exterior().coords().enumerate() {
                        let x_str = coord_strings
                            .entry(OrderedFloat(coord.x))
                            .or_insert_with(|| coord.x.to_string());
                        write!(encoder, "              [{}, ", x_str).unwrap();

                        let y_str = coord_strings
                            .entry(OrderedFloat(coord.y))
                            .or_insert_with(|| coord.y.to_string());
                        write!(encoder, "{}]", y_str).unwrap();

                        if j < ncoords - 1 {
                            writeln!(encoder, ",").unwrap();
                        } else {
                            writeln!(encoder).unwrap();
                        }
                    }

                    write!(encoder, concat!("            ]\n", "          ]")).unwrap();

                    if i < npolys - 1 {
                        writeln!(encoder, ",").unwrap();
                    } else {
                        writeln!(encoder).unwrap();
                    }
                }

                write!(encoder, concat!("        ]\n", "      }}\n", "    }}")).unwrap();
                if count < nmultipolys - 1 {
                    writeln!(encoder, ",").unwrap();
                } else {
                    writeln!(encoder).unwrap();
                }

                count += 1;
            }
        }

        writeln!(encoder, "  ]\n}}").unwrap();
    }
}
