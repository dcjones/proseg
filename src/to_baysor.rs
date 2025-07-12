mod polygon_area;
mod schemas;
use crate::polygon_area::polygon_area;
use crate::schemas::*;

use arrow::array::RecordBatch;
use arrow::csv;
use arrow::datatypes::{DataType, Field, Schema};
use clap::Parser;
use geo_traits::{CoordTrait, GeometryTrait, LineStringTrait, MultiPolygonTrait, PolygonTrait};
use json::JsonValue;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use wkb::reader::read_wkb;
use zarrs::filesystem::FilesystemStore;

pub const BACKGROUND_CELL: u32 = u32::MAX;

#[derive(Parser, Debug)]
#[command(name = "proseg-to-baysor")]
#[command(author = "Daniel C. Jones")]
#[command(about = "Convert proseg output to Baysor-compatible output.")]
struct Args {
    proseg_spatialdata_zarr: String,

    #[arg(long, default_value = "proseg-to-baysor-transcript-metadata.csv")]
    output_transcript_metadata: String,

    #[arg(long, default_value = "proseg-to-baysor-cell-polygons.geojson")]
    output_cell_polygons: String,
}

fn main() {
    let args = Args::parse();

    let zarr_store = Arc::new(FilesystemStore::new(&args.proseg_spatialdata_zarr).expect(
        &format!(
            "Unable to open proseg spatialdata at {}",
            &args.proseg_spatialdata_zarr
        ),
    ));

    let transcript_metadata = read_proseg_transcript_metadata_from_zarr(zarr_store.clone());
    let cell_polygons = read_proseg_cell_polygons_from_zarr(zarr_store.clone());

    let cells_with_transcripts: HashSet<u32> = transcript_metadata.cell.iter().cloned().collect();

    let mask: Vec<bool> = cell_polygons
        .iter()
        .map(|polygon| {
            let cell = polygon["cell"].as_u32().unwrap();
            cells_with_transcripts.contains(&cell) && !polygon["coordinates"].is_null()
        })
        .collect();

    write_baysor_transcript_metadata(args.output_transcript_metadata, transcript_metadata);
    write_baysor_cell_polygon_geojson(cell_polygons, args.output_cell_polygons, &mask);
}

fn find_column_index(schema: &Schema, column: &str) -> usize {
    let col = schema.index_of(column);
    match col {
        Ok(col) => col,
        _ => panic!("Column '{}' not found in CSV file", column),
    }
}

struct TranscriptMetadata {
    transcript_id: Vec<u64>,
    cell: Vec<u32>,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    qv: Vec<f32>,
}

fn read_proseg_transcript_metadata_from_zarr(
    zarr_store: Arc<FilesystemStore>,
) -> TranscriptMetadata {
    let parquet_path = zarr_store.key_to_fspath(
        &zarrs::storage::StoreKey::new(&format!(
            "points/{SD_TRANSCRIPTS_NAME}/points.parquet/part.0.parquet"
        ))
        .unwrap(),
    );
    let input_file = File::open(&parquet_path).expect(&format!(
        "Unable to open '{}'.",
        parquet_path.to_str().unwrap()
    ));

    let rdr = ParquetRecordBatchReaderBuilder::try_new(input_file)
        .unwrap()
        .build()
        .expect(&format!(
            "Unable to read parquet data from from {}",
            parquet_path.to_str().unwrap()
        ));

    let mut metadata = TranscriptMetadata {
        transcript_id: Vec::new(),
        cell: Vec::new(),
        x: Vec::new(),
        y: Vec::new(),
        z: Vec::new(),
        qv: Vec::new(),
    };

    let schema = transcript_metadata_schema();
    let transcript_id_col = find_column_index(&schema, "transcript_id");
    let assignment_col = find_column_index(&schema, "assignment");
    let x_col = find_column_index(&schema, "x");
    let y_col = find_column_index(&schema, "y");
    let z_col = find_column_index(&schema, "z");

    // TODO: We are no longer recording qv values. Is that necessary?
    // let qv_col = find_column_index(&schema, "qv");

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        for assignment in rec_batch
            .column(assignment_col)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .unwrap()
            .iter()
        {
            if let Some(assignment) = assignment {
                metadata.cell.push(assignment);
            } else {
                metadata.cell.push(BACKGROUND_CELL);
            }
        }

        for transcript_id in rec_batch
            .column(transcript_id_col)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap()
            .iter()
        {
            metadata.transcript_id.push(transcript_id.unwrap_or(0));
        }
        while metadata.transcript_id.len() < metadata.cell.len() {
            metadata
                .transcript_id
                .push(metadata.transcript_id.len() as u64);
        }

        for x in rec_batch
            .column(x_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.x.push(x.unwrap());
        }

        for y in rec_batch
            .column(y_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.y.push(y.unwrap());
        }

        for z in rec_batch
            .column(z_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.z.push(z.unwrap());
        }

        // for qv in rec_batch
        //     .column(qv_col)
        //     .as_any()
        //     .downcast_ref::<arrow::array::Float32Array>()
        //     .unwrap()
        //     .iter()
        // {
        //     metadata.qv.push(qv.unwrap());
        // }

        metadata.qv.push(0.0);
    }

    metadata
}

fn filter_option<T>(value: T, mask: bool) -> Option<T> {
    if mask { Some(value) } else { None }
}

fn write_baysor_transcript_metadata(filename: String, metadata: TranscriptMetadata) {
    let output = File::create(filename).expect("Unable to create output transcript metadata file.");

    let schema = Schema::new(vec![
        Field::new("transcript_id", DataType::UInt64, false),
        Field::new("cell", DataType::LargeUtf8, false),
        Field::new("is_noise", DataType::Boolean, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
    ]);

    let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(
            metadata
                .transcript_id
                .iter()
                .cloned()
                .collect::<arrow::array::UInt64Array>(),
        ),
        Arc::new(
            metadata
                .cell
                .iter()
                .map(|cell| {
                    if *cell == BACKGROUND_CELL {
                        Some(String::new())
                    } else {
                        Some(format!("cell-{}", cell))
                    }
                })
                .collect::<arrow::array::LargeStringArray>(),
        ),
        Arc::new(
            metadata
                .cell
                .iter()
                .map(|x| Some(*x == u32::MAX))
                .collect::<arrow::array::BooleanArray>(),
        ),
        Arc::new(
            metadata
                .x
                .iter()
                .cloned()
                .collect::<arrow::array::Float32Array>(),
        ),
        Arc::new(
            metadata
                .y
                .iter()
                .cloned()
                .collect::<arrow::array::Float32Array>(),
        ),
        Arc::new(
            metadata
                .z
                .iter()
                .cloned()
                .collect::<arrow::array::Float32Array>(),
        ),
    ];

    let batch = RecordBatch::try_new(Arc::new(schema), columns).unwrap();

    let mut writer = csv::WriterBuilder::new().with_header(true).build(output);
    writer.write(&batch).expect("Unable to write CSV file");
}

fn read_proseg_cell_polygons_from_zarr(zarr_store: Arc<FilesystemStore>) -> Vec<JsonValue> {
    let parquet_path = zarr_store.key_to_fspath(
        &zarrs::storage::StoreKey::new(&format!("shapes/{SD_SHAPES_NAME}/shapes.parquet")).unwrap(),
    );

    let input_file = File::open(&parquet_path).expect(&format!(
        "Unable to open '{}'.",
        parquet_path.to_str().unwrap()
    ));

    let rdr = ParquetRecordBatchReaderBuilder::try_new(input_file)
        .unwrap()
        .build()
        .expect(&format!(
            "Unable to read parquet data from from {}",
            parquet_path.to_str().unwrap()
        ));

    let schema = wkb_shapes_schema();
    let cell_col = find_column_index(&schema, "cell");
    let geometry_col = find_column_index(&schema, "geometry");

    let mut polygons = Vec::new();

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        let cells = rec_batch
            .column(cell_col)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .unwrap();

        let geometries = rec_batch
            .column(geometry_col)
            .as_any()
            .downcast_ref::<arrow::array::BinaryArray>()
            .unwrap();

        let mut largest_poly = Vec::new();
        let mut current_poly = Vec::new();
        for (cell, geometry) in cells.iter().zip(geometries.iter()) {
            let cell = cell.unwrap();
            let geometry = geometry.unwrap();
            let geometry = read_wkb(geometry).unwrap();
            let geometry = match geometry.as_type() {
                geo_traits::GeometryType::MultiPolygon(mp) => mp,
                _ => panic!("Unexpected geometry type"),
            };

            // Read polygons and choose the largest exterior (xenium can't deal with multi-polygons)
            let mut largest_area = 0.0;
            for poly in geometry.polygons() {
                current_poly.clear();
                let poly = poly.exterior().unwrap();
                for coord in poly.coords() {
                    let (x, y) = coord.x_y();
                    current_poly.push((x as f32, y as f32));
                }

                let area = polygon_area(&mut current_poly);
                if area > largest_area {
                    largest_area = area;
                    largest_poly.clear();
                    for coord in poly.coords() {
                        let (x, y) = coord.x_y();
                        largest_poly.push((x as f32, y as f32));
                    }
                }
            }

            polygons.push(json::object! {
                "type": "Polygon",
                "cell": cell,
                "coordinates": [largest_poly.iter().cloned().map(|(x, y)| json::array![x, y]).collect::<Vec<JsonValue>>()]
            });
        }
    }

    polygons
}

fn write_baysor_cell_polygon_geojson(
    geometries: Vec<JsonValue>,
    output_filename: String,
    mask: &[bool],
) {
    let geometries = JsonValue::from(
        geometries
            .iter()
            .zip(mask)
            .filter_map(|(v, &m)| filter_option(v, m))
            .cloned()
            .collect::<Vec<JsonValue>>(),
    );

    let mut data = JsonValue::new_object();
    data.insert("geometries", geometries).unwrap();
    data.insert("type", JsonValue::from("GeometryCollection"))
        .unwrap();

    let mut output =
        File::create(output_filename).expect("Unable to create output cell polygon geojson file.");
    output
        .write_all(data.dump().as_bytes())
        .expect("Unable to write output cell polygon geojson file.");
}
