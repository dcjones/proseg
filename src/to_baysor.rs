use clap::Parser;

// mod sampler;
// mod output;
// use output::{OutputFormat, determine_format};
// use arrow2::datatypes::Schema;
// use arrow2::io::csv as arrow_csv;
// use arrow2::io::parquet;
// use csv::StringRecord;

mod schemas;
use crate::schemas::{transcript_metadata_schema, OutputFormat};

mod polygon_area;
use crate::polygon_area::polygon_area;

use arrow::array::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use arrow::error::ArrowError;
use arrow::csv;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use flate2::read::GzDecoder;
use json::JsonValue;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;
use std::collections::HashSet;

pub const BACKGROUND_CELL: u32 = std::u32::MAX;

#[derive(Parser, Debug)]
#[command(name = "proseg-to-baysor")]
#[command(author = "Daniel C. Jones")]
#[command(about = "Convert proseg output to Baysor-compatible output.")]
struct Args {
    transcript_metadata: String,
    cell_polygons: String,

    #[arg(long, default_value = "proseg-to-baysor-transcript-metadata.csv")]
    output_transcript_metadata: String,

    #[arg(long, default_value = "proseg-to-baysor-cell-polygons.geojson")]
    output_cell_polygons: String,
}

fn main() {
    let args = Args::parse();
    let metadata = read_proseg_transcript_metadata(args.transcript_metadata);
    let (polygon_data, polygons) = read_cell_polygons_geojson(args.cell_polygons);

    let cells_with_transcripts: HashSet<u32> = metadata.cell.iter().cloned().collect();

    let mask: Vec<bool> = polygons.iter().map(|polygon| {
        let cell = polygon["cell"].as_u32().unwrap();
        cells_with_transcripts.contains(&cell) && !polygon["coordinates"].is_null()
    }).collect();

    write_baysor_transcript_metadata(args.output_transcript_metadata, metadata);
    write_cell_polygon_geojson(polygon_data, polygons, args.output_cell_polygons, &mask);
}

fn determine_format(filename: &str, fmtstr: &Option<String>) -> OutputFormat {
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
        OutputFormat::CsvGz
    } else if filename.ends_with(".csv") {
        OutputFormat::Csv
    } else if filename.ends_with(".parquet") {
        OutputFormat::Parquet
    } else {
        panic!("Unknown file format for: {}", filename);
    }
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

fn read_proseg_transcript_metadata(filename: String) -> TranscriptMetadata {
    let fmt = determine_format(&filename, &None);

    let schema = transcript_metadata_schema(fmt);
    let input_file = File::open(&filename).expect(&format!("Unable to open '{}'.", &filename));

    match fmt {
        OutputFormat::Csv => {
            let rdr = csv::ReaderBuilder::new(Arc::new(schema.clone()))
                .with_header(true)
                .build(input_file)
                .expect(&format!("Unable to construct CSV reader for '{}'", filename));
            read_proseg_transcript_metadata_from_reader(rdr, &schema)
        }
        OutputFormat::CsvGz => {
            let input_decoder = GzDecoder::new(input_file);
            let rdr = csv::ReaderBuilder::new(Arc::new(schema.clone()))
                .with_header(true)
                .build(input_decoder)
                .expect(&format!("Unable to construct CSV reader for '{}'", filename));
            read_proseg_transcript_metadata_from_reader(rdr, &schema)
        }
        OutputFormat::Parquet => {
            let rdr = ParquetRecordBatchReaderBuilder::try_new(input_file)
                .unwrap()
                .build()
                .expect(&format!("Unable to read parquet data from frobm {}", filename));

            read_proseg_transcript_metadata_from_reader(rdr, &schema)
        },
        OutputFormat::Infer => panic!("Indeterminable output format")
    }
}

fn read_proseg_transcript_metadata_from_reader<T>(
    rdr: T, schema: &Schema) -> TranscriptMetadata
where
    T: Iterator<Item = Result<RecordBatch, ArrowError>>
{
    let mut metadata = TranscriptMetadata {
        transcript_id: Vec::new(),
        cell: Vec::new(),
        x: Vec::new(),
        y: Vec::new(),
        z: Vec::new(),
        qv: Vec::new(),
    };

    let transcript_id_col = find_column_index(schema, "transcript_id");
    let assignment_col = find_column_index(schema, "assignment");
    let x_col = find_column_index(schema, "x");
    let y_col = find_column_index(schema, "y");
    let z_col = find_column_index(schema, "z");
    let qv_col = find_column_index(schema, "qv");

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        for transcript_id in rec_batch.column(transcript_id_col)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap()
            .iter()
        {
            metadata.transcript_id.push(transcript_id.unwrap());
        }

        for assignment in rec_batch.column(assignment_col)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .unwrap()
            .iter()
        {
            metadata.cell.push(assignment.unwrap());
        }

        for x in rec_batch.column(x_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.x.push(x.unwrap());
        }

        for y in rec_batch.column(y_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.y.push(y.unwrap());
        }

        for z in rec_batch.column(z_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.z.push(z.unwrap());
        }

        for qv in rec_batch.column(qv_col)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap()
            .iter()
        {
            metadata.qv.push(qv.unwrap());
        }
    }

    metadata
}

fn filter_option<T>(value: T, mask: bool) -> Option<T> {
    if mask {
        Some(value)
    } else {
        None
    }
}

fn write_baysor_transcript_metadata(filename: String, metadata: TranscriptMetadata) {
    let output =
        File::create(filename).expect("Unable to create output transcript metadata file.");

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
            metadata.transcript_id.iter().cloned().collect::<arrow::array::UInt64Array>()
        ),
        Arc::new(
            metadata.cell.iter().map(|cell|
                if *cell == u32::MAX {
                   Some(String::new())
                } else {
                    Some(format!("cell-{}", cell))
                }
            )
            .collect::<arrow::array::LargeStringArray>()
        ),
        Arc::new(
            metadata.cell
                .iter().map(|x| Some(*x == u32::MAX))
                .collect::<arrow::array::BooleanArray>()
        ),
        Arc::new(
            metadata.x.iter().cloned().collect::<arrow::array::Float32Array>()
        ),
        Arc::new(
            metadata.y.iter().cloned().collect::<arrow::array::Float32Array>()
        ),
        Arc::new(
            metadata.z.iter().cloned().collect::<arrow::array::Float32Array>()
        ),
    ];

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        columns
    ).unwrap();

    let mut writer = csv::WriterBuilder::new()
        .with_header(true)
        .build(output);
    writer.write(&batch).expect("Unable to write CSV file");
}


fn read_cell_polygons_geojson(input_filename: String) -> (JsonValue, Vec<JsonValue>) {
    let input =
        File::open(input_filename).expect("Unable to open input cell polygon geojson file.");
    let mut input = GzDecoder::new(input);

    let mut content = String::new();
    input
        .read_to_string(&mut content)
        .expect("Unable to read input cell polygon geojson file.");
    let mut data = json::parse(&content).expect("Unable to parse input cell polygon geojson file.");

    let features = data.remove("features");

    let geometries: Vec<JsonValue> = features.members().map(|feature| {
        let polygons = &feature["geometry"]["coordinates"];

        let mut largest_poygon = 0;
        let mut largest_poygon_area = 0_f32;
        for (i, polygon) in polygons.members().enumerate() {
            assert!(polygon.members().len() == 1);
            let polygon = polygon.members().next().unwrap();

            let mut coords = polygon
                .members()
                .map(|xy| (xy[0].as_f32().unwrap(), xy[1].as_f32().unwrap()))
                .collect::<Vec<(f32, f32)>>();

            let area = polygon_area(&mut coords);

            if area > largest_poygon_area {
                largest_poygon_area = area;
                largest_poygon = i;
            }
        }

        // let mut coordinates = JsonValue::new_array();
        let coordinates = polygons[largest_poygon].clone();

        let mut geometry = JsonValue::new_object();
        geometry.insert("type", "Polygon").unwrap();
        geometry.insert("coordinates", coordinates).unwrap();
        geometry
            .insert("cell", feature["properties"]["cell"].clone())
            .unwrap();

        geometry
    }).collect();

    return (data, geometries);
}

// We need to rename
//   "features" -> "geometries"
//   "FeatureCollection" -> "GeometryCollection"
//
// We also need to reduce the MultiPolygons to a single polygon. We just take the largest one.
fn write_cell_polygon_geojson(mut data: JsonValue, geometries: Vec<JsonValue>, output_filename: String, mask: &[bool]) {
    let geometries = JsonValue::from(geometries
        .iter()
        .zip(mask)
        .filter_map(|(v, &m)| filter_option(v, m))
        .cloned()
        .collect::<Vec<JsonValue>>());

    data.insert("geometries", geometries).unwrap();
    data.insert("type", JsonValue::from("GeometryCollection"))
        .unwrap();

    let mut output =
        File::create(output_filename).expect("Unable to create output cell polygon geojson file.");
    output
        .write_all(data.dump().as_bytes())
        .expect("Unable to write output cell polygon geojson file.");
}
