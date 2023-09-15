use csv;
use flate2::read::GzDecoder;
use std::collections::HashMap;
use std::fs::File;
use ndarray::Array2;
use kiddo::float::kdtree::KdTree;
use kiddo::distance::squared_euclidean;

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
    z_column: &str,
    min_qv: f32
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>) {
    let mut rdr = csv::Reader::from_reader(GzDecoder::new(File::open(path).unwrap()));
    // let mut rdr = csv::Reader::from_path(path).unwrap();

    return read_transcripts_csv_xyz(
        &mut rdr,
        transcript_column,
        x_column,
        y_column,
        z_column,
        min_qv,
    );
}

fn find_column(headers: &csv::StringRecord, column: &str) -> usize {
    let col = headers.iter().position(|x| x == column);
    match col {
        Some(col) => col,
        None => panic!("Column '{}' not found in CSV file", column),
    }
}

fn postprocess_cell_assignments(cell_assignments: &mut Vec<CellIndex>) -> Vec<usize> {
    // reassign cell ids to exclude anything that no initial transcripts assigned
    let mut used_cell_ids: HashMap<CellIndex, CellIndex> = HashMap::new();
    for &cell_id in cell_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            let next_id = used_cell_ids.len() as CellIndex;
            used_cell_ids.entry(cell_id).or_insert(next_id);
        }
    }

    for cell_id in cell_assignments.iter_mut() {
        if *cell_id != BACKGROUND_CELL {
            *cell_id = *used_cell_ids.get(cell_id).unwrap();
        }
    }

    let ncells = used_cell_ids.len();

    let mut cell_population = vec![0; ncells];
    for &cell_id in cell_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            cell_population[cell_id as usize] += 1;
        }
    }

    return cell_population;
}

fn read_transcripts_csv_xyz<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
    z_column: &str,
    min_qv: f32,
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
    let qv_col = find_column(headers, "qv");

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut cell_assignments = Vec::new();

    let mut cell_id_map: HashMap<String, CellIndex> = HashMap::new();

    for result in rdr.records() {
        let row = result.unwrap();

        let qv = row[qv_col].parse::<f32>().unwrap();
        if qv < min_qv {
            continue;
        }

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
        let z = row[z_col].parse::<f32>().unwrap();

        transcripts.push(Transcript {
            x,
            y,
            z,
            gene: gene as u32,
        });

        let cell_id_str = &row[cell_id_col];
        let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();

        // Earlier version of Xenium used numeric cell ids and -1 for unassigned.
        // Newer versions use alphanumeric hash codes and "UNASSIGNED" for unasssigned.
        if cell_id_str == "-1" || cell_id_str == "UNASSIGNED" {
            cell_assignments.push(BACKGROUND_CELL);
        } else {
            let next_cell_id = cell_id_map.len() as CellIndex;
            let cell_id = *cell_id_map
                .entry(cell_id_str.to_string())
                .or_insert_with(|| next_cell_id);

            if overlaps_nucleus > 0 {
                cell_assignments.push(cell_id);
            } else {
                cell_assignments.push(BACKGROUND_CELL);
            }
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


// Estimate what region of the slide to model by counting the number of occupied bins.
pub fn estimate_full_area(transcripts: &Vec<Transcript>, mean_nucleus_area: f32) -> f32 {
    let (xmin, xmax, ymin, ymax, _, _) = coordinate_span(&transcripts);

    const SCALE: f32 = 2.0;
    let binsize = SCALE * mean_nucleus_area.sqrt();

    let xbins = ((xmax - xmin) / binsize).ceil() as usize;
    let ybins = ((ymax - ymin) / binsize).ceil() as usize;

    dbg!(xbins, ybins, binsize, mean_nucleus_area);

    let mut occupied = Array2::from_elem((xbins, ybins), false);

    for transcript in transcripts {
        let xbin = ((transcript.x - xmin) / binsize).floor() as usize;
        let ybin = ((transcript.y - ymin) / binsize).floor() as usize;

        occupied[[xbin, ybin]] = true;
    }

    dbg!(
        occupied.iter().filter(|&&x| x).count(),
        occupied.len(),
    );

    return occupied.iter().filter(|&&x| x).count() as f32 * binsize * binsize;
}


// Estimate cell centroids by averaging the coordinates of all transcripts assigned to each cell.
pub fn estimate_cell_centroids(transcripts: &Vec<Transcript>, cell_assignments: &Vec<CellIndex>, ncells: usize) -> Vec<(f32, f32)> {
    let mut cell_transcripts: Vec<Vec<usize>> = vec![Vec::new(); ncells];
    for (i, &cell) in cell_assignments.iter().enumerate() {
        if cell != BACKGROUND_CELL {
            cell_transcripts[cell as usize].push(i);
        }
        }

    let mut centroids = Vec::with_capacity(ncells);
    for ts in cell_transcripts.iter() {
        if ts.len() == 0 {
            centroids.push((f32::NAN, f32::NAN));
            continue;
        }

        let mut x = 0.0;
        let mut y = 0.0;
        for &t in ts {
            x += transcripts[t].x;
            y += transcripts[t].y;
        }
        x /= ts.len() as f32;
        y /= ts.len() as f32;
        centroids.push((x, y));
    }

    return centroids;
}


pub fn filter_cellfree_transcripts(transcripts: &Vec<Transcript>, init_cell_assignments: &Vec<CellIndex>, ncells: usize, max_distance: f32) ->
    (Vec<Transcript>, Vec<CellIndex>)
{
    let max_distance_squared = max_distance * max_distance;

    let centroids = estimate_cell_centroids(transcripts, init_cell_assignments, ncells);
    let mut kdtree: KdTree<f32, u32, 2, 32, u32> = KdTree::with_capacity(centroids.len());
    for (i, (x, y)) in centroids.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        kdtree.add(&[*x, *y], i as u32);
    }

    let mut mask = vec![false; transcripts.len()];
    for (i, t) in transcripts.iter().enumerate() {
        let (d, _) = kdtree.nearest_one(&[t.x, t.y], &squared_euclidean);

        if d <= max_distance_squared {
            mask[i] = true;
        }
    }

    let filtered_transcripts = transcripts.iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(t, _)| t)
        .cloned()
        .collect::<Vec<_>>();
    let filtered_cell_assignments = init_cell_assignments.iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(t, _)| t)
        .cloned()
        .collect::<Vec<_>>();


    return (filtered_transcripts, filtered_cell_assignments);
}