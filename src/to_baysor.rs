
use clap::Parser;

// mod sampler;
// mod output;
// use output::{OutputFormat, determine_format};
use csv::StringRecord;
use arrow2::io::csv as arrow_csv;
use arrow2::io::parquet;
use arrow2::datatypes::Schema;
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::{Read, Write};
use json;
use std::sync::Arc;


#[derive(Parser, Debug)]
#[command(name="proseg-to-baysor")]
#[command(author="Daniel C. Jones")]
#[command(about="Convert proseg output to Baysor-compatible output.")]
struct Args {
    transcript_metadata: String,
    cell_polygons: String,

    #[arg(long, default_value="proseg-to-baysor-transcript-metadata.csv")]
    output_transcript_metadata: String,

    #[arg(long, default_value="proseg-to-baysor-cell-polygons.geojson")]
    output_cell_polygons: String,
}

fn main() {
    let args = Args::parse();

    let metadata = read_proseg_transcript_metadata(args.transcript_metadata);
    write_baysor_transcript_metadata(args.output_transcript_metadata, metadata);

    rerwrite_cell_polygon_geojson(args.cell_polygons, args.output_cell_polygons);
}

enum OutputFormat {
    Csv,
    CsvGz,
    Parquet,
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
        return OutputFormat::CsvGz;
    } else if filename.ends_with(".csv") {
        return OutputFormat::Csv;
    } else if filename.ends_with(".parquet") {
        return OutputFormat::Parquet;
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
    cell: Vec<u32>,
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    // TODO: What else?
}


fn read_proseg_transcript_metadata(filename: String) -> TranscriptMetadata {
    let fmt = determine_format(&filename, &None);

    match fmt {
        OutputFormat::Csv => {
            let rdr = arrow_csv::read::Reader::from_path(filename)
                .expect("Unable to open csv file.");
            return read_proseg_transcript_metadata_csv(rdr);
        },
        OutputFormat::CsvGz => {
            let rdr = arrow_csv::read::Reader::from_reader(GzDecoder::new(File::open(filename).expect("Unable to open csv.gz file.")));
            return read_proseg_transcript_metadata_csv(rdr);
        },
        OutputFormat::Parquet => {
            let mut metadata = TranscriptMetadata {
                cell: Vec::new(),
                x: Vec::new(),
                y: Vec::new(),
                z: Vec::new(),
            };

            let mut file = File::open(filename).expect("Unable to open parquet file.");

            let file_metadata = parquet::read::read_metadata(&mut file).expect("Unable to read parquet metadata.");
            let schema = parquet::read::infer_schema(&file_metadata).expect("Unable to infer parquet schema.");
            let schema = schema.filter(|_idx, field| {
                field.name == "assignment" || field.name == "x" || field.name == "y" || field.name == "z"
            });

            let assignment_col = find_parquet_column(&schema, "assignment");
            let x_col = find_parquet_column(&schema, "x");
            let y_col = find_parquet_column(&schema, "y");
            let z_col = find_parquet_column(&schema, "z");

            let chunks = parquet::read::FileReader::new(
                file, file_metadata.row_groups, schema, Some(1024 * 8 * 8), None, None);

            for chunk in chunks {
                let chunk = chunk.expect("Unable to read parquet chunk.");
                let columns = chunk.columns();

                for assignment in columns[assignment_col].as_any().downcast_ref::<arrow2::array::UInt32Array>().unwrap().iter() {
                    metadata.cell.push(*assignment.unwrap());
                }

                for x in columns[x_col].as_any().downcast_ref::<arrow2::array::Float32Array>().unwrap().iter() {
                    metadata.x.push(*x.unwrap());
                }

                for y in columns[y_col].as_any().downcast_ref::<arrow2::array::Float32Array>().unwrap().iter() {
                    metadata.y.push(*y.unwrap());
                }

                for z in columns[z_col].as_any().downcast_ref::<arrow2::array::Float32Array>().unwrap().iter() {
                    metadata.z.push(*z.unwrap());
                }
            }

            return metadata;
        },
    };
}


fn read_proseg_transcript_metadata_csv<T>(mut rdr: arrow_csv::read::Reader<T>) -> TranscriptMetadata where T: Read {
    let headers = rdr.headers().expect("Unable to read CSV headers.");

    let mut metadata = TranscriptMetadata{
        cell: Vec::new(),
        x: Vec::new(),
        y: Vec::new(),
        z: Vec::new(),
    };

    let assignment_col = find_csv_column(headers, "assignment");
    let x_col = find_csv_column(headers, "x");
    let y_col = find_csv_column(headers, "y");
    let z_col = find_csv_column(headers, "z");

    for result in rdr.records() {
        let row = result.expect("Unable to read CSV record.");

        metadata.cell.push(row[assignment_col].parse::<u32>().unwrap());
        metadata.x.push(row[x_col].parse::<f32>().unwrap());
        metadata.y.push(row[y_col].parse::<f32>().unwrap());
        metadata.z.push(row[z_col].parse::<f32>().unwrap());
    }

    return metadata;
}


fn write_baysor_transcript_metadata(filename: String, metadata: TranscriptMetadata) {
    let mut output = File::create(filename).expect("Unable to create output transcript metadata file.");

    let names = ["cell", "x", "y", "z"];
    let mut columns: Vec<Arc<dyn arrow2::array::Array>> = Vec::new();
    columns.push(Arc::new(arrow2::array::UInt32Array::from_values(metadata.cell.iter().cloned())));
    columns.push(Arc::new(arrow2::array::Float32Array::from_values(metadata.x.iter().cloned())));
    columns.push(Arc::new(arrow2::array::Float32Array::from_values(metadata.y.iter().cloned())));
    columns.push(Arc::new(arrow2::array::Float32Array::from_values(metadata.z.iter().cloned())));
    let chunk = arrow2::chunk::Chunk::new(columns);

    let options = arrow2::io::csv::write::SerializeOptions::default();
    arrow2::io::csv::write::write_header(&mut output, &names, &options).expect("Unable to write CSV header.");
    arrow2::io::csv::write::write_chunk(&mut output, &chunk, &options).expect("Unable to write CSV chunk.")
}

// Just need to rename a couple keys to make the geojson file compatible with Baysor output.
fn rerwrite_cell_polygon_geojson(input_filename: String, output_filename: String) {
    let input = File::open(input_filename).expect("Unable to open input cell polygon geojson file.");
    let mut input = GzDecoder::new(input);

    let mut content = String::new();
    input.read_to_string(&mut content).expect("Unable to read input cell polygon geojson file.");
    let mut data = json::parse(&content).expect("Unable to parse input cell polygon geojson file.");

    let features = data.remove("features");
    data.insert("geometries", features).unwrap();

    data.remove("type");
    data.insert("type", "GeometryCollection").unwrap();

    let mut output = File::create(output_filename).expect("Unable to create output cell polygon geojson file.");
    output.write_all(data.dump().as_bytes()).expect("Unable to write output cell polygon geojson file.");
}

