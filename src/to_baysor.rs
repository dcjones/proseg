use clap::Parser;

// mod sampler;
// mod output;
// use output::{OutputFormat, determine_format};
use arrow2::datatypes::Schema;
use arrow2::io::csv as arrow_csv;
use arrow2::io::parquet;
use csv::StringRecord;
use flate2::read::GzDecoder;
use json::JsonValue;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{Read, Write};
use std::sync::Arc;

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

    let ncells = polygons.len();
    let pops = cell_populations(&metadata, ncells, 20.0);

    let mask: Vec<bool> = pops.iter().zip(&polygons).map(|(&pop, polygon)| {
        pop > 0 && !polygon["coordinates"].is_null()
    }).collect();

    write_baysor_transcript_metadata(args.output_transcript_metadata, metadata);
    write_cell_polygon_geojson(polygon_data, polygons, args.output_cell_polygons, &mask);
}

enum OutputFormat {
    Csv,
    CsvGz,
    Parquet,
}

fn cell_populations(metadata: &TranscriptMetadata, ncells: usize, min_qv: f32) -> Vec<u32> {
    let mut cell_population: Vec<u32> = vec![0; ncells];
    for (&cell, &qv) in metadata.cell.iter().zip(&metadata.qv) {
        if qv >= min_qv && cell != BACKGROUND_CELL {
            cell_population[cell as usize] += 1;
        }
    }
    return cell_population;
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

fn find_csv_column(headers: &StringRecord, column: &str) -> usize {
    let col = headers.iter().position(|x| x == column);
    match col {
        Some(col) => col,
        None => panic!("Column '{}' not found in CSV file", column),
    }
}

fn find_parquet_column(schema: &Schema, column: &str) -> usize {
    let col = schema.fields.iter().position(|field| field.name == column);
    match col {
        Some(col) => col,
        None => panic!("Column '{}' not found in CSV file", column),
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

    match fmt {
        OutputFormat::Csv => {
            let rdr =
                arrow_csv::read::Reader::from_path(filename).expect("Unable to open csv file.");
            read_proseg_transcript_metadata_csv(rdr)
        }
        OutputFormat::CsvGz => {
            let rdr = arrow_csv::read::Reader::from_reader(GzDecoder::new(
                File::open(filename).expect("Unable to open csv.gz file."),
            ));
            read_proseg_transcript_metadata_csv(rdr)
        }
        OutputFormat::Parquet => {
            let mut metadata = TranscriptMetadata {
                transcript_id: Vec::new(),
                cell: Vec::new(),
                x: Vec::new(),
                y: Vec::new(),
                z: Vec::new(),
                qv: Vec::new(),
            };

            let mut file = File::open(filename).expect("Unable to open parquet file.");

            let file_metadata =
                parquet::read::read_metadata(&mut file).expect("Unable to read parquet metadata.");
            let schema = parquet::read::infer_schema(&file_metadata)
                .expect("Unable to infer parquet schema.");
            let schema = schema.filter(|_idx, field| {
                field.name == "transcript_id"
                    || field.name == "assignment"
                    || field.name == "x"
                    || field.name == "y"
                    || field.name == "z"
                    || field.name == "qv"
            });

            let transcript_id_col = find_parquet_column(&schema, "transcript_id");
            let assignment_col = find_parquet_column(&schema, "assignment");
            let x_col = find_parquet_column(&schema, "x");
            let y_col = find_parquet_column(&schema, "y");
            let z_col = find_parquet_column(&schema, "z");
            let qv_col = find_parquet_column(&schema, "qv");

            let chunks = parquet::read::FileReader::new(
                file,
                file_metadata.row_groups,
                schema,
                Some(1024 * 8 * 8),
                None,
                None,
            );

            for chunk in chunks {
                let chunk = chunk.expect("Unable to read parquet chunk.");
                let columns = chunk.columns();

                for transcript_id in columns[transcript_id_col]
                    .as_any()
                    .downcast_ref::<arrow2::array::UInt64Array>()
                    .unwrap()
                    .iter()
                {
                    metadata.transcript_id.push(*transcript_id.unwrap());
                }

                for assignment in columns[assignment_col]
                    .as_any()
                    .downcast_ref::<arrow2::array::UInt32Array>()
                    .unwrap()
                    .iter()
                {
                    metadata.cell.push(*assignment.unwrap());
                }

                for x in columns[x_col]
                    .as_any()
                    .downcast_ref::<arrow2::array::Float32Array>()
                    .unwrap()
                    .iter()
                {
                    metadata.x.push(*x.unwrap());
                }

                for y in columns[y_col]
                    .as_any()
                    .downcast_ref::<arrow2::array::Float32Array>()
                    .unwrap()
                    .iter()
                {
                    metadata.y.push(*y.unwrap());
                }

                for z in columns[z_col]
                    .as_any()
                    .downcast_ref::<arrow2::array::Float32Array>()
                    .unwrap()
                    .iter()
                {
                    metadata.z.push(*z.unwrap());
                }

                for qv in columns[qv_col]
                    .as_any()
                    .downcast_ref::<arrow2::array::Float32Array>()
                    .unwrap()
                    .iter()
                {
                    metadata.qv.push(*qv.unwrap());
                }
            }

            metadata
        }
    }
}

fn read_proseg_transcript_metadata_csv<T>(mut rdr: arrow_csv::read::Reader<T>) -> TranscriptMetadata
where
    T: Read,
{
    let headers = rdr.headers().expect("Unable to read CSV headers.");

    let mut metadata = TranscriptMetadata {
        transcript_id: Vec::new(),
        cell: Vec::new(),
        x: Vec::new(),
        y: Vec::new(),
        z: Vec::new(),
        qv: Vec::new(),
    };

    let transcript_id_col = find_csv_column(headers, "transcript_id");
    let assignment_col = find_csv_column(headers, "assignment");
    let x_col = find_csv_column(headers, "x");
    let y_col = find_csv_column(headers, "y");
    let z_col = find_csv_column(headers, "z");
    let qv_col = find_csv_column(headers, "qv");

    for result in rdr.records() {
        let row = result.expect("Unable to read CSV record.");

        metadata
            .transcript_id
            .push(row[transcript_id_col].parse::<u64>().unwrap());
        metadata
            .cell
            .push(row[assignment_col].parse::<u32>().unwrap());
        metadata.x.push(row[x_col].parse::<f32>().unwrap());
        metadata.y.push(row[y_col].parse::<f32>().unwrap());
        metadata.z.push(row[z_col].parse::<f32>().unwrap());
        metadata.qv.push(row[qv_col].parse::<f32>().unwrap());
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
    let mut output =
        File::create(filename).expect("Unable to create output transcript metadata file.");

    let names = ["transcript_id", "cell", "is_noise", "x", "y", "z"];
    let columns: Vec<Arc<dyn arrow2::array::Array>> = vec![
        Arc::new(arrow2::array::UInt64Array::from_values(
            metadata.transcript_id.iter().cloned(),
        )),
        Arc::new(arrow2::array::Utf8Array::<i64>::from_iter_values(
            metadata.cell.iter().map(|cell|
                if *cell == u32::MAX {
                   String::new()
                } else {
                    format!("cell-{}", cell)
                }
            ),
        )),
        Arc::new(arrow2::array::BooleanArray::from_iter(
            metadata.cell.iter().map(|x| Some(*x == u32::MAX)),
        )),
        Arc::new(arrow2::array::Float32Array::from_values(
            metadata.x.iter().cloned(),
        )),
        Arc::new(arrow2::array::Float32Array::from_values(
            metadata.y.iter().cloned(),
        )),
        Arc::new(arrow2::array::Float32Array::from_values(
            metadata.z.iter().cloned(),
        )),
    ];

    let chunk = arrow2::chunk::Chunk::new(columns);

    let options = arrow2::io::csv::write::SerializeOptions::default();
    arrow2::io::csv::write::write_header(&mut output, &names, &options)
        .expect("Unable to write CSV header.");
    arrow2::io::csv::write::write_chunk(&mut output, &chunk, &options)
        .expect("Unable to write CSV chunk.")
}

fn center(vertices: &[(f32, f32)]) -> (f32, f32) {
    let mut x = 0.0;
    let mut y = 0.0;
    for v in vertices {
        x += v.0;
        y += v.1;
    }
    x /= vertices.len() as f32;
    y /= vertices.len() as f32;
    (x, y)
}

fn clockwise_cmp(c: (f32, f32), a: (f32, f32), b: (f32, f32)) -> Ordering {
    // From: https://stackoverflow.com/a/6989383
    if a.0 - c.0 >= 0.0 && b.0 - c.0 < 0.0 {
        return Ordering::Less;
    } else if a.0 - c.0 < 0.0 && b.0 - c.0 >= 0.0 {
        return Ordering::Greater;
    } else if a.0 - c.0 == 0.0 && b.0 - c.0 == 0.0 {
        if a.1 - c.1 >= 0.0 || b.1 - c.1 >= 0.0 {
            return a.1.partial_cmp(&b.1).unwrap();
        } else {
            return b.1.partial_cmp(&a.1).unwrap();
        }
    }

    // compute the cross product of vectors (c -> a) x (c -> b)
    let det = (a.0 - c.0) * (b.1 - c.1) - (b.0 - c.0) * (a.1 - c.1);

    if det < 0.0 {
        Ordering::Less
    } else if det > 0.0 {
        Ordering::Greater
    } else {
        // points a and b are on the same line from the c
        // check which point is closer to the c
        let d1 = (a.0 - c.0) * (a.0 - c.0) + (a.1 - c.1) * (a.1 - c.1);
        let d2 = (b.0 - c.0) * (b.0 - c.0) + (b.1 - c.1) * (b.1 - c.1);
        if d1 >= d2 {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

fn polygon_area(vertices: &mut [(f32, f32)]) -> f32 {
    let c = center(vertices);
    vertices.sort_unstable_by(|a, b| clockwise_cmp(c, *a, *b));

    let mut area = 0.0;

    for (i, u) in vertices.iter().enumerate() {
        let j = (i + 1) % vertices.len();
        let v = vertices[j];

        // triangle formula.
        // area += u.0 * v.1 - v.0 * u.1;

        // trapezoid formula (this is more numerically stable with large coordinates)
        area += (v.0 + u.0) * (v.1 - u.1);
    }
    area = area.abs() / 2.0;

    area
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
