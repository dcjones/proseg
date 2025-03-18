use arrow::array::RecordBatch;
use arrow::csv;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::error::ArrowError;
use flate2::write::GzEncoder;
use flate2::Compression;
use geo::MultiPolygon;
use ndarray::{Array1, Array2, Axis};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression::ZSTD, ZstdLevel};
use parquet::errors::ParquetError;
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use super::sampler::transcripts::Transcript;
use super::sampler::transcripts::BACKGROUND_CELL;
use super::sampler::voxelsampler::VoxelSampler;
use super::sampler::{ModelParams, TranscriptState};
use crate::schemas::{transcript_metadata_schema, OutputFormat};

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

pub fn write_counts(
    output_path: &Option<String>,
    output_counts: &Option<String>,
    output_counts_fmt: OutputFormat,
    transcript_names: &[String],
    counts: &Array2<u32>,
) {
    if let Some(output_counts) = output_counts {
        let schema = Schema::new(
            transcript_names
                .iter()
                .map(|name| Field::new(name, DataType::UInt32, false))
                .collect::<Vec<Field>>(),
        );

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        for row in counts.rows() {
            columns.push(Arc::new(
                row.iter().cloned().collect::<arrow::array::UInt32Array>(),
            ));
        }

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(output_path, output_counts, output_counts_fmt, &batch);
    }
}

pub fn write_expected_counts(
    output_path: &Option<String>,
    output_expected_counts: &Option<String>,
    output_expected_counts_fmt: OutputFormat,
    transcript_names: &[String],
    ecounts: &Array2<f32>,
) {
    if let Some(output_expected_counts) = output_expected_counts {
        let schema = Schema::new(
            transcript_names
                .iter()
                .map(|name| Field::new(name, DataType::Float32, false))
                .collect::<Vec<Field>>(),
        );

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = Vec::new();
        for row in ecounts.rows() {
            columns.push(Arc::new(
                row.iter().cloned().collect::<arrow::array::Float32Array>(),
            ));
        }

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(
            output_path,
            output_expected_counts,
            output_expected_counts_fmt,
            &batch,
        );
    }
}

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
                .map(|gene| Some(gene))
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

// Assign cells to fovs by finding the most common transcript fov of the
// assigned transcripts.
fn cell_fov_vote(
    ncells: usize,
    nfovs: usize,
    cell_assignments: &[(u32, f32)],
    fovs: &[u32],
) -> Vec<u32> {
    let mut fov_votes = Array2::<u32>::zeros((ncells, nfovs));
    for (fov, (cell, _)) in fovs.iter().zip(cell_assignments) {
        if *cell != BACKGROUND_CELL {
            fov_votes[[*cell as usize, *fov as usize]] += 1;
        }
    }

    fov_votes
        .outer_iter()
        .map(|votes| {
            let mut winner = u32::MAX;
            let mut winner_count: u32 = 0;
            for (fov, count) in votes.iter().enumerate() {
                if *count > winner_count {
                    winner_count = *count;
                    winner = fov as u32;
                }
            }
            winner
        })
        .collect::<Vec<u32>>()
}

pub fn write_cell_metadata(
    output_path: &Option<String>,
    output_cell_metadata: &Option<String>,
    output_cell_metadata_fmt: OutputFormat,
    params: &ModelParams,
    cell_centroids: &[(f32, f32, f32)],
    cell_assignments: &[(u32, f32)],
    original_cell_ids: &Vec<String>,
    fovs: &[u32],
    fov_names: &[String],
) {
    // TODO: write factorization
    let ncells = cell_centroids.len();
    let nfovs = fov_names.len();
    let cell_fovs = cell_fov_vote(ncells, nfovs, cell_assignments, fovs);

    if let Some(output_cell_metadata) = output_cell_metadata {
        let schema = Schema::new(vec![
            Field::new("cell", DataType::UInt32, false),
            Field::new("original_cell_id", DataType::Utf8, false),
            Field::new("centroid_x", DataType::Float32, false),
            Field::new("centroid_y", DataType::Float32, false),
            Field::new("centroid_z", DataType::Float32, false),
            Field::new("fov", DataType::Utf8, true),
            Field::new("cluster", DataType::UInt16, false),
            Field::new("volume", DataType::Float32, false),
            Field::new("scale", DataType::Float32, false),
            Field::new("population", DataType::UInt64, false),
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
                    .iter()
                    .map(|(x, _, _)| *x)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                cell_centroids
                    .iter()
                    .map(|(_, y, _)| *y)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                cell_centroids
                    .iter()
                    .map(|(_, _, z)| *z)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                cell_fovs
                    .iter()
                    .map(|fov| {
                        if *fov == u32::MAX {
                            None
                        } else {
                            Some(fov_names[*fov as usize].clone())
                        }
                    })
                    .collect::<arrow::array::StringArray>(),
            ),
            Arc::new(
                params
                    .z
                    .iter()
                    .map(|&z| z as u16)
                    .collect::<arrow::array::UInt16Array>(),
            ),
            Arc::new(
                params
                    .cell_volume
                    .iter()
                    .cloned()
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                params
                    .cell_scale
                    .iter()
                    .cloned()
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                params
                    .cell_population
                    .iter()
                    .map(|&p| p as u64)
                    .collect::<arrow::array::UInt64Array>(),
            ),
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

#[allow(clippy::too_many_arguments)]
pub fn write_transcript_metadata(
    output_path: &Option<String>,
    output_transcript_metadata: &Option<String>,
    output_transcript_metadata_fmt: OutputFormat,
    transcripts: &[Transcript],
    transcript_positions: &[(f32, f32, f32)],
    transcript_names: &[String],
    cell_assignments: &[(u32, f32)],
    transcript_state: &Array1<TranscriptState>,
    qvs: &[f32],
    fovs: &[u32],
    fov_names: &[String],
) {
    if let Some(output_transcript_metadata) = output_transcript_metadata {
        // arraw_csv has no problem outputting LargeStringArray, but can't read them.
        // As a work around we always output the same schema, but change the schema
        // when reading csv.
        let schema = transcript_metadata_schema(OutputFormat::Parquet);

        let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(
                transcripts
                    .iter()
                    .map(|t| t.transcript_id)
                    .collect::<arrow::array::UInt64Array>(),
            ),
            Arc::new(
                transcript_positions
                    .iter()
                    .map(|(x, _, _)| *x)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcript_positions
                    .iter()
                    .map(|(_, y, _)| *y)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcript_positions
                    .iter()
                    .map(|(_, _, z)| *z)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcripts
                    .iter()
                    .map(|t| t.x)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcripts
                    .iter()
                    .map(|t| t.y)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcripts
                    .iter()
                    .map(|t| t.z)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcripts
                    .iter()
                    .map(|t| Some(transcript_names[t.gene as usize].clone()))
                    .collect::<arrow::array::LargeStringArray>(),
            ),
            Arc::new(qvs.iter().cloned().collect::<arrow::array::Float32Array>()),
            Arc::new(
                fovs.iter()
                    .map(|fov| Some(fov_names[*fov as usize].clone()))
                    .collect::<arrow::array::LargeStringArray>(),
            ),
            Arc::new(
                cell_assignments
                    .iter()
                    .map(|(cell, _)| *cell)
                    .collect::<arrow::array::UInt32Array>(),
            ),
            Arc::new(
                cell_assignments
                    .iter()
                    .map(|(_, pr)| *pr)
                    .collect::<arrow::array::Float32Array>(),
            ),
            Arc::new(
                transcript_state
                    .iter()
                    .map(|&s| (s == TranscriptState::Background) as u8)
                    .collect::<arrow::array::UInt8Array>(),
            ),
            Arc::new(
                transcript_state
                    .iter()
                    .map(|&s| (s == TranscriptState::Confusion) as u8)
                    .collect::<arrow::array::UInt8Array>(),
            ),
        ];

        let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

        write_table(
            output_path,
            output_transcript_metadata,
            output_transcript_metadata_fmt,
            &batch,
        );
    }
}

pub fn write_gene_metadata(
    output_path: &Option<String>,
    output_gene_metadata: &Option<String>,
    output_gene_metadata_fmt: OutputFormat,
    params: &ModelParams,
    transcript_names: &[String],
    expected_counts: &Array2<f32>,
) {
    // TODO: write factorization
    if let Some(output_gene_metadata) = output_gene_metadata {
        let mut schema_fields = vec![
            Field::new("gene", DataType::Utf8, false),
            Field::new("total_count", DataType::UInt64, false),
            Field::new("expected_assigned_count", DataType::Float32, false),
            // Field::new("dispersion", DataType::Float32, false),
        ];

        let mut columns: Vec<Arc<dyn arrow::array::Array>> = vec![
            Arc::new(
                transcript_names
                    .iter()
                    .map(|s| Some(s.clone()))
                    .collect::<arrow::array::StringArray>(),
            ),
            Arc::new(
                params
                    .total_gene_counts
                    .sum_axis(Axis(1))
                    .iter()
                    .map(|x| *x as u64)
                    .collect::<arrow::array::UInt64Array>(),
            ),
            Arc::new(
                expected_counts
                    .sum_axis(Axis(1))
                    .iter()
                    .cloned()
                    .collect::<arrow::array::Float32Array>(),
            ),
            // Arc::new(array::Float32Array::from_values(
            //     params.r.iter().cloned(),
            // ))
        ];

        // // cell type dispersions
        // schema_fields.push(Field::new("dispersion", DataType::Float32, false));
        // columns.push(Arc::new(
        //     params
        //         .r
        //         .iter()
        //         .cloned()
        //         .collect::<arrow::array::Float32Array>(),
        // ));

        // // cell type rates
        // for i in 0..params.ncomponents() {
        //     schema_fields.push(Field::new(&format!("λ_{}", i), DataType::Float32, false));
        //

        //     let mut λ_component = Array1::<f32>::from_elem(params.ngenes(), 0_f32);
        //     let mut count = 0;
        //     Zip::from(&params.z)
        //         .and(params.λ.columns())
        //         .for_each(|&z, λ| {
        //             if i == z as usize {
        //                 Zip::from(&mut λ_component).and(λ).for_each(|a, b| *a += b);
        //                 count += 1;
        //             }
        //         });
        //     λ_component /= count as f32;
        //

        //     columns.push(Arc::new(
        //         λ_component
        //             .iter()
        //             .cloned()
        //             .collect::<arrow::array::Float32Array>(),
        //     ));
        // }

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

// TODO:
// If we want to import things into qupath, I think we need a way to scale
// the coordinates to pixel space. It also doesn't seem like it supports
// MultiPolygons, so we need to write each polygon in a cell to a separate Polygon entry.

pub fn write_cell_multipolygons(
    output_path: &Option<String>,
    output_cell_polygons: &Option<String>,
    polygons: Vec<MultiPolygon<f32>>,
) {
    if let Some(output_cell_polygons) = output_cell_polygons {
        let file = if let Some(output_path) = output_path {
            File::create(Path::new(output_path).join(output_cell_polygons)).unwrap()
        } else {
            File::create(output_cell_polygons).unwrap()
        };
        let mut encoder = GzEncoder::new(file, Compression::default());

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
                    write!(encoder, "              [{}, {}]", coord.x, coord.y).unwrap();
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
    polygons: Vec<Vec<(i32, MultiPolygon<f32>)>>,
) {
    if let Some(output_cell_polygons) = output_cell_polygons {
        let file = if let Some(output_path) = output_path {
            File::create(Path::new(output_path).join(output_cell_polygons)).unwrap()
        } else {
            File::create(output_cell_polygons).unwrap()
        };
        let mut encoder = GzEncoder::new(file, Compression::default());

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
                        write!(encoder, "              [{}, {}]", coord.x, coord.y).unwrap();
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
