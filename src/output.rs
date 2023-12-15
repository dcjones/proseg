use arrow2;
use arrow2::array;
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Field, Schema};
use flate2::write::GzEncoder;
use flate2::Compression;
use ndarray::{Array1, Array2, Axis, Zip};
use std::fs::File;
use std::io::Write;
use std::sync::Arc;

use super::sampler::cubebinsampler::CubeBinSampler;
use super::sampler::transcripts::Transcript;
use super::sampler::ModelParams;

pub enum OutputFormat {
    Csv,
    CsvGz,
    Parquet,
}

pub fn write_table(
    filename: &str,
    fmtstr: &Option<String>,
    schema: Schema,
    chunk: Chunk<Arc<dyn arrow2::array::Array>>,
) {
    let fmt = determine_format(filename, fmtstr);
    let mut file = File::create(&filename).unwrap();

    match fmt {
        OutputFormat::Csv => {
            if write_table_csv(&mut file, schema, chunk).is_err() {
                panic!("Error writing csv file: {}", filename);
            }
        }
        OutputFormat::CsvGz => {
            let mut encoder = GzEncoder::new(file, Compression::default());
            if write_table_csv(&mut encoder, schema, chunk).is_err() {
                panic!("Error writing csv.gz file: {}", filename);
            }
        }
        OutputFormat::Parquet => {
            if write_table_parquet(&mut file, schema, chunk).is_err() {
                panic!("Error writing parquet file: {}", filename);
            }
        }
    }
}

fn write_table_csv<W>(
    output: &mut W,
    schema: Schema,
    chunk: Chunk<Arc<dyn arrow2::array::Array>>,
) -> arrow2::error::Result<()>
where
    W: std::io::Write,
{
    let options = arrow2::io::csv::write::SerializeOptions::default();
    let names = schema
        .fields
        .iter()
        .map(|f| f.name.clone())
        .collect::<Vec<_>>();
    arrow2::io::csv::write::write_header(output, &names, &options)?;
    arrow2::io::csv::write::write_chunk(output, &chunk, &options)?;
    return Ok(());
}

fn write_table_parquet<W>(
    output: &mut W,
    schema: Schema,
    chunk: Chunk<Arc<dyn arrow2::array::Array>>,
) -> arrow2::error::Result<()>
where
    W: std::io::Write,
{
    let options = arrow2::io::parquet::write::WriteOptions {
        write_statistics: true,
        version: arrow2::io::parquet::write::Version::V2,
        compression: arrow2::io::parquet::write::CompressionOptions::Zstd(Some(
            arrow2::io::parquet::write::ZstdLevel::default(),
        )),
        data_pagesize_limit: None,
    };

    let encodings = schema
        .fields
        .iter()
        // .map(|f| arrow2::io::parquet::write::Encoding::Plain)
        .map(|f| {
            arrow2::io::parquet::write::transverse(&f.data_type, |_| {
                arrow2::io::parquet::write::Encoding::Plain
            })
        })
        .collect();

    let chunk_iter = vec![Ok(chunk)];
    let row_groups = arrow2::io::parquet::write::RowGroupIterator::try_new(
        chunk_iter.into_iter(),
        &schema,
        options,
        encodings,
    )?;

    let mut writer = arrow2::io::parquet::write::FileWriter::try_new(output, schema, options)?;

    for group in row_groups {
        writer.write(group?)?;
    }

    writer.end(None)?;

    return Ok(());
}

pub fn determine_format(filename: &str, fmtstr: &Option<String>) -> OutputFormat {
    if let Some(fmtstr) = fmtstr {
        if fmtstr == "csv.gz" {
            return OutputFormat::CsvGz;
        } else if fmtstr == "csv" {
            return OutputFormat::Csv;
        } else if fmtstr == "parquet" {
            return OutputFormat::Parquet;
        } else {
            panic!("Unknown file format: {}", fmtstr);
        }
    }

    if filename.ends_with(".csv.gz") {
        return OutputFormat::CsvGz;
    } else if filename.ends_with(".csv") {
        return OutputFormat::Csv;
    } else if filename.ends_with(".parquet") {
        return OutputFormat::Parquet;
    } else {
        panic!("Unknown file format for: {}", filename);
    }
}

pub fn write_counts(
    output_counts: &Option<String>,
    output_counts_fmt: &Option<String>,
    transcript_names: &Vec<String>,
    counts: &Array2<u32>,
) {
    if let Some(output_counts) = output_counts {
        let schema = arrow2::datatypes::Schema::from(
            transcript_names
                .iter()
                .map(|name| {
                    arrow2::datatypes::Field::new(name, arrow2::datatypes::DataType::UInt32, false)
                })
                .collect::<Vec<_>>(),
        );

        let mut columns: Vec<Arc<dyn arrow2::array::Array>> = Vec::new();
        for row in counts.rows() {
            columns.push(Arc::new(arrow2::array::UInt32Array::from_values(
                row.iter().cloned(),
            )));
        }
        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(&output_counts, output_counts_fmt, schema, chunk);
    }
}

pub fn write_expected_counts(
    output_expected_counts: &Option<String>,
    output_expected_counts_fmt: &Option<String>,
    transcript_names: &Vec<String>,
    ecounts: &Array2<f32>,
) {
    if let Some(output_expected_counts) = output_expected_counts {
        let schema = arrow2::datatypes::Schema::from(
            transcript_names
                .iter()
                .map(|name| {
                    arrow2::datatypes::Field::new(name, arrow2::datatypes::DataType::Float32, false)
                })
                .collect::<Vec<_>>(),
        );

        let mut columns: Vec<Arc<dyn arrow2::array::Array>> = Vec::new();
        for row in ecounts.rows() {
            columns.push(Arc::new(arrow2::array::Float32Array::from_values(
                row.iter().cloned(),
            )));
        }
        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(
            &output_expected_counts,
            &output_expected_counts_fmt,
            schema,
            chunk,
        );
    }
}

pub fn write_rates(
    output_rates: &Option<String>,
    output_rates_fmt: &Option<String>,
    params: &ModelParams,
    transcript_names: &Vec<String>,
) {
    if let Some(output_rates) = output_rates {
        let schema = arrow2::datatypes::Schema::from(
            transcript_names
                .iter()
                .map(|name| {
                    arrow2::datatypes::Field::new(name, arrow2::datatypes::DataType::Float32, false)
                })
                .collect::<Vec<_>>(),
        );

        let mut columns: Vec<Arc<dyn arrow2::array::Array>> = Vec::new();
        for row in params.λ.rows() {
            columns.push(Arc::new(arrow2::array::Float32Array::from_values(
                row.iter().cloned(),
            )));
        }
        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(&output_rates, &output_rates_fmt, schema, chunk);
    }
}

pub fn write_component_params(
    output_component_params: &Option<String>,
    output_component_params_fmt: &Option<String>,
    params: &ModelParams,
    transcript_names: &Vec<String>
) {
    if let Some(output_component_params) = output_component_params {
        // What does this look like: rows for each gene, columns for α1, β1, α2, β2, etc.
        let α = &params.r;
        let φ = &params.φ;
        let β = φ.map(|φ| (-φ).exp());

        let ncomponents = params.ncomponents();

        let mut fields = Vec::new();
        fields.push(Field::new("gene", DataType::Utf8, false));
        for i in 0..ncomponents {
            fields.push(Field::new(&format!("α_{}", i), DataType::Float32, false));
            fields.push(Field::new(&format!("β_{}", i), DataType::Float32, false));
        }
        let schema = Schema::from(fields);

        let mut columns: Vec<Arc<dyn arrow2::array::Array>> = Vec::new();
        columns.push(Arc::new(array::Utf8Array::<i32>::from_iter_values(
            transcript_names.iter().cloned(),
        )));
        Zip::from(α.rows()).and(β.rows()).for_each(|α, β| {
            columns.push(Arc::new(array::Float32Array::from_values(α.iter().cloned())));
            columns.push(Arc::new(array::Float32Array::from_values(β.iter().cloned())));
        });

        let chunk = arrow2::chunk::Chunk::new(columns);
        write_table(
            &output_component_params,
            &output_component_params_fmt,
            schema,
            chunk,
        );
    }
}

pub fn write_cell_metadata(
    output_cell_metadata: &Option<String>,
    output_cell_metadata_fmt: &Option<String>,
    params: &ModelParams,
    cell_centroids: &Vec<(f32, f32)>,
) {
    if let Some(output_cell_metadata) = output_cell_metadata {
        let schema = Schema::from(vec![
            Field::new("cell", DataType::UInt32, false),
            Field::new("centroid_x", DataType::Float32, false),
            Field::new("centroid_y", DataType::Float32, false),
            Field::new("cluster", DataType::UInt16, false),
            Field::new("volume", DataType::Float32, false),
            Field::new("population", DataType::UInt64, false),
        ]);

        let columns: Vec<Arc<dyn arrow2::array::Array>> = vec![
            Arc::new(array::UInt32Array::from_values(0..params.ncells() as u32)),
            Arc::new(array::Float32Array::from_values(
                cell_centroids.iter().map(|(x, _)| *x),
            )),
            Arc::new(array::Float32Array::from_values(
                cell_centroids.iter().map(|(_, y)| *y),
            )),
            Arc::new(array::UInt16Array::from_values(
                params.z.iter().map(|&z| z as u16),
            )),
            Arc::new(array::Float32Array::from_values(
                params.cell_volume.iter().cloned(),
            )),
            Arc::new(array::UInt64Array::from_values(
                params.cell_population.iter().map(|&p| p as u64),
            )),
        ];

        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(
            &output_cell_metadata,
            &output_cell_metadata_fmt,
            schema,
            chunk,
        );
    }
}

pub fn write_transcript_metadata(
    output_transcript_metadata: &Option<String>,
    output_transcript_metadata_fmt: &Option<String>,
    transcripts: &Vec<Transcript>,
    transcript_positions: &Vec<(f32, f32, f32)>,
    transcript_names: &Vec<String>,
    cell_assignments: &Vec<(u32, f32)>,
) {
    if let Some(output_transcript_metadata) = output_transcript_metadata {
        let schema = Schema::from(vec![
            Field::new("transcript_id", DataType::UInt64, false),
            Field::new("x", DataType::Float32, false),
            Field::new("y", DataType::Float32, false),
            Field::new("z", DataType::Float32, false),
            Field::new("observed_x", DataType::Float32, false),
            Field::new("observed_y", DataType::Float32, false),
            Field::new("observed_z", DataType::Float32, false),
            Field::new("gene", DataType::Utf8, false),
            Field::new("assignment", DataType::UInt32, false),
            Field::new("probability", DataType::Float32, false),
        ]);

        let columns: Vec<Arc<dyn arrow2::array::Array>> = vec![
            Arc::new(array::UInt64Array::from_values(
                transcripts.iter().map(|t| t.transcript_id),
            )),
            Arc::new(array::Float32Array::from_values(
                transcript_positions.iter().map(|(x, _, _)| *x),
            )),
            Arc::new(array::Float32Array::from_values(
                transcript_positions.iter().map(|(_, y, _)| *y),
            )),
            Arc::new(array::Float32Array::from_values(
                transcript_positions.iter().map(|(_, _, z)| *z),
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
                cell_assignments.iter().map(|(cell, _)| *cell),
            )),
            Arc::new(array::Float32Array::from_values(
                cell_assignments.iter().map(|(_, pr)| *pr),
            )),
        ];

        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(
            &output_transcript_metadata,
            &output_transcript_metadata_fmt,
            schema,
            chunk,
        );
    }
}

pub fn write_gene_metadata(
    output_gene_metadata: &Option<String>,
    output_gene_metadata_fmt: &Option<String>,
    params: &ModelParams,
    transcript_names: &Vec<String>,
    expected_counts: &Array2<f32>,
) {
    if let Some(output_gene_metadata) = output_gene_metadata {
        let mut schema_fields = vec![
            Field::new("gene", DataType::Utf8, false),
            Field::new("total_count", DataType::UInt64, false),
            Field::new("expected_assigned_count", DataType::Float32, false),
            // Field::new("dispersion", DataType::Float32, false),
        ];

        let mut columns: Vec<Arc<dyn arrow2::array::Array>> = vec![
            Arc::new(array::Utf8Array::<i32>::from_iter_values(
                transcript_names.iter().cloned(),
            )),
            Arc::new(array::UInt64Array::from_values(
                params
                    .total_gene_counts
                    .sum_axis(Axis(1))
                    .iter()
                    .map(|x| *x as u64),
            )),
            Arc::new(array::Float32Array::from_values(
                expected_counts.sum_axis(Axis(1)).iter().cloned(),
            )),
            // Arc::new(array::Float32Array::from_values(
            //     params.r.iter().cloned(),
            // ))
        ];

        // cell type dispersions
        for i in 0..params.ncomponents() {
            schema_fields.push(Field::new(&format!("dispersion_{}", i), DataType::Float32, false));
            columns.push(Arc::new(array::Float32Array::from_values(
                params.r.row(i).iter().cloned()
            )));
        }

        // cell type rates
        for i in 0..params.ncomponents() {
            schema_fields.push(Field::new(&format!("λ_{}", i), DataType::Float32, false));

            let mut λ_component = Array1::<f32>::from_elem(params.ngenes(), 0_f32);
            let mut count = 0;
            Zip::from(&params.z)
                .and(params.λ.columns())
                .for_each(|&z, λ| {
                    if i == z as usize {
                        Zip::from(&mut λ_component).and(λ).for_each(|a, b| *a += b);
                        count += 1;
                    }
                });
            λ_component /= count as f32;

            columns.push(Arc::new(array::Float32Array::from_values(
                λ_component.iter().cloned(),
            )));
        }

        // background rates
        for i in 0..params.nlayers() {
            schema_fields.push(Field::new(format!("λ_bg_{}", i), DataType::Float32, false));
            columns.push(Arc::new(array::Float32Array::from_values(
                params.λ_bg.column(i).iter().cloned(),
            )));
        }

        let schema = Schema::from(schema_fields);
        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(
            &output_gene_metadata,
            &output_gene_metadata_fmt,
            schema,
            chunk,
        );
    }
}

pub fn write_cubes(
    output_cubes: &Option<String>,
    output_cubes_fmt: &Option<String>,
    sampler: &CubeBinSampler,
) {
    if let Some(output_cubes) = output_cubes {
        let ncubes = sampler.cubes().count();

        let mut cells = Vec::with_capacity(ncubes);
        let mut x0s = Vec::with_capacity(ncubes);
        let mut y0s = Vec::with_capacity(ncubes);
        let mut z0s = Vec::with_capacity(ncubes);
        let mut x1s = Vec::with_capacity(ncubes);
        let mut y1s = Vec::with_capacity(ncubes);
        let mut z1s = Vec::with_capacity(ncubes);

        for (cell, (x0, y0, z0, x1, y1, z1)) in sampler.cubes() {
            cells.push(cell);
            x0s.push(x0);
            y0s.push(y0);
            z0s.push(z0);
            x1s.push(x1);
            y1s.push(y1);
            z1s.push(z1);
        }

        let schema = Schema::from(vec![
            Field::new("cell", DataType::UInt32, false),
            Field::new("x0", DataType::Float32, false),
            Field::new("y0", DataType::Float32, false),
            Field::new("z0", DataType::Float32, false),
            Field::new("x1", DataType::Float32, false),
            Field::new("y1", DataType::Float32, false),
            Field::new("z1", DataType::Float32, false),
        ]);

        let columns: Vec<Arc<dyn arrow2::array::Array>> = vec![
            Arc::new(array::UInt32Array::from_vec(cells)),
            Arc::new(array::Float32Array::from_vec(x0s)),
            Arc::new(array::Float32Array::from_vec(y0s)),
            Arc::new(array::Float32Array::from_vec(z0s)),
            Arc::new(array::Float32Array::from_vec(x1s)),
            Arc::new(array::Float32Array::from_vec(y1s)),
            Arc::new(array::Float32Array::from_vec(z1s)),
        ];

        let chunk = arrow2::chunk::Chunk::new(columns);

        write_table(&output_cubes, &output_cubes_fmt, schema, chunk);
    }
}

// TODO:
// If we want to import things into qupath, I think we need a way to scale
// the coordinates to pixel space. It also doesn't seem like it supports
// MultiPolygons, so we need to write each polygon in a cell to a separate Polygon entry.

pub fn write_cell_multipolygons(output_cell_polygons: &Option<String>, sampler: &CubeBinSampler) {
    if let Some(output_cell_polygons) = output_cell_polygons {
        // Need to collect cubes and organize
        let cell_polys = sampler.cell_polygons();

        let file = File::create(output_cell_polygons).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());

        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

        let ncells = cell_polys.len();
        for (cell, polys) in cell_polys.into_iter().enumerate() {
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
                        writeln!(encoder, "").unwrap();
                    }
                }

                write!(encoder, concat!("            ]\n", "          ]")).unwrap();

                if i < npolys - 1 {
                    writeln!(encoder, ",").unwrap();
                } else {
                    writeln!(encoder, "").unwrap();
                }
            }

            write!(encoder, concat!("        ]\n", "      }}\n", "    }}")).unwrap();
            if cell < ncells - 1 {
                writeln!(encoder, ",").unwrap();
            } else {
                writeln!(encoder, "").unwrap();
            }
        }

        writeln!(encoder, "  ]\n}}").unwrap();
    }
}

pub fn write_cell_layered_multipolygons(
    output_cell_polygons: &Option<String>,
    sampler: &CubeBinSampler,
) {
    if let Some(output_cell_polygons) = output_cell_polygons {
        // Need to collect cubes and organize
        let cell_layered_polys = sampler.cell_layered_polygons();

        let file = File::create(output_cell_polygons).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());

        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

        let mut nmultipolys = 0;
        for (_, cell_polys) in cell_layered_polys.iter() {
            nmultipolys += cell_polys.len();
        }

        let mut count = 0;
        for (layer, cell_polys) in cell_layered_polys {
            for (cell, polys) in cell_polys.into_iter().enumerate() {
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
                            writeln!(encoder, "").unwrap();
                        }
                    }

                    write!(encoder, concat!("            ]\n", "          ]")).unwrap();

                    if i < npolys - 1 {
                        writeln!(encoder, ",").unwrap();
                    } else {
                        writeln!(encoder, "").unwrap();
                    }
                }

                write!(encoder, concat!("        ]\n", "      }}\n", "    }}")).unwrap();
                if count < nmultipolys - 1 {
                    writeln!(encoder, ",").unwrap();
                } else {
                    writeln!(encoder, "").unwrap();
                }

                count += 1;
            }
        }

        writeln!(encoder, "  ]\n}}").unwrap();
    }
}
