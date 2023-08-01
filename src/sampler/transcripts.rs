use csv;
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;

pub type CellIndex = u32;
pub const BACKGROUND_CELL: CellIndex = std::u32::MAX;

#[derive(Copy, Clone, PartialEq)]
pub struct Transcript {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: u32,
}

pub fn read_transcripts_csv(
    path: &str,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
    z_column: Option<&str>,
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>) {
    let mut rdr = csv::Reader::from_reader(GzDecoder::new(File::open(path).unwrap()));
    // let mut rdr = csv::Reader::from_path(path).unwrap();

    match z_column {
        Some(z_column) => {
            return read_transcripts_csv_xyz(
                &mut rdr,
                transcript_column,
                x_column,
                y_column,
                z_column,
            );
        }
        None => {
            return read_transcripts_csv_xy(&mut rdr, transcript_column, x_column, y_column);
        }
    }
}

fn find_column(headers: &csv::StringRecord, column: &str) -> usize {
    let col = headers.iter().position(|x| x == column);
    match col {
        Some(col) => col,
        None => panic!("Column '{}' not found in CSV file", column),
    }
}

fn postprocess_cell_assignments(cell_assignments: &Vec<CellIndex>) -> Vec<usize> {
    let mut ncells = usize::MAX;
    for &cell_id in cell_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            if ncells == usize::MAX || cell_id as usize >= ncells {
                ncells = (cell_id + 1) as usize;
            }
        }
    }

    if ncells == usize::MAX {
        ncells = 0;
    }

    let mut cell_population = vec![0; ncells];
    for &cell_id in cell_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            cell_population[cell_id as usize] += 1;
        }
    }

    return cell_population;
}

fn read_transcripts_csv_xy<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>)
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let cell_id_col = find_column(headers, "cell_id");
    let overlaps_nucleus_col = find_column(headers, "overlaps_nucleus");

    let mut transcripts = Vec::new();
    let mut transcript_name_map = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut cell_assignments = Vec::new();

    for result in rdr.records() {
        let row = result.unwrap();

        let transcript_name = &row[transcript_col];

        let gene = if let Some(gene) = transcript_name_map.get(transcript_name) {
            *gene
        } else {
            transcript_names.push(transcript_name.to_string());
            transcript_name_map.insert(transcript_name.to_string(), transcript_names.len() - 1);
            transcript_names.len() - 1
        };

        let x = row[x_col].parse::<f32>().unwrap();
        let y = row[y_col].parse::<f32>().unwrap();

        transcripts.push(Transcript {
            x,
            y,
            z: 0.0,
            gene: gene as u32,
        });

        let cell_id = row[cell_id_col].parse::<i32>().unwrap();
        let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();
        if cell_id >= 0 && overlaps_nucleus > 0 {
            cell_assignments.push(cell_id as u32);
        } else {
            cell_assignments.push(BACKGROUND_CELL);
        }
    }

    let cell_population = postprocess_cell_assignments(&cell_assignments);

    return (
        transcript_names,
        transcripts,
        cell_assignments,
        cell_population,
    );
}

fn read_transcripts_csv_xyz<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
    z_column: &str,
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>)
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let z_col = find_column(headers, z_column);

    // TODO:
    // Just assuming we have xeinum output at this point.
    // We'll have to specialize for various platforms in the future.
    let cell_id_col = find_column(headers, "cell_id");
    let overlaps_nucleus_col = find_column(headers, "overlaps_nucleus");
    // let qv_col = find_column(headers, "qv");

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut cell_assignments = Vec::new();

    for result in rdr.records() {
        let row = result.unwrap();

        // let qv = row[qv_col].parse::<f32>().unwrap();

        let transcript_name = &row[transcript_col];
        // let transcript_name = "FAKE";

        let gene = if let Some(gene) = transcript_name_map.get(transcript_name) {
            *gene
        } else {
            transcript_names.push(transcript_name.to_string());
            transcript_name_map.insert(transcript_name.to_string(), transcript_names.len() - 1);
            transcript_names.len() - 1
        };

        let x = row[x_col].parse::<f32>().unwrap();
        let y = row[y_col].parse::<f32>().unwrap();
        let z = row[z_col].parse::<f32>().unwrap();

        transcripts.push(Transcript {
            x,
            y,
            z,
            gene: gene as u32,
        });

        let cell_id = row[cell_id_col].parse::<i32>().unwrap();
        let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();
        if cell_id >= 0 && overlaps_nucleus > 0 {
            // if cell_id >= 0 {
            cell_assignments.push(cell_id as u32);
        } else {
            cell_assignments.push(BACKGROUND_CELL);
        }
    }

    let cell_population = postprocess_cell_assignments(&mut cell_assignments);

    return (
        transcript_names,
        transcripts,
        cell_assignments,
        cell_population,
    );
}

pub fn coordinate_span(transcripts: &Vec<Transcript>) -> (f32, f32, f32, f32, f32, f32) {
    let mut min_x = std::f32::MAX;
    let mut max_x = std::f32::MIN;
    let mut min_y = std::f32::MAX;
    let mut max_y = std::f32::MIN;
    let mut min_z = std::f32::MAX;
    let mut max_z = std::f32::MIN;

    for t in transcripts {
        min_x = min_x.min(t.x);
        max_x = max_x.max(t.x);
        min_y = min_y.min(t.y);
        max_y = max_y.max(t.y);
        min_z = min_z.min(t.z);
        max_z = max_z.max(t.z);
    }

    return (min_x, max_x, min_y, max_y, min_z, max_z);
}
