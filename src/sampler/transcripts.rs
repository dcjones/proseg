use csv;
use flate2::read::GzDecoder;
use kiddo::distance::squared_euclidean;
use kiddo::float::kdtree::KdTree;
use ndarray::Array2;
use std::collections::HashMap;
use std::fs::File;

pub type CellIndex = u32;
pub const BACKGROUND_CELL: CellIndex = std::u32::MAX;

// Should probably rearrange this...
use super::super::output::{determine_format, OutputFormat};

#[derive(Copy, Clone, PartialEq)]
pub struct Transcript {
    pub transcript_id: u64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: u32,
}

pub fn read_transcripts_csv(
    path: &str,
    transcript_column: &str,
    id_column: Option<String>,
    compartment_column: &str,
    compartment_nuclear: &str,
    fov_column: Option<String>,
    cell_id_column: &str,
    cell_id_unassigned: &str,
    qv_column: Option<String>,
    x_column: &str,
    y_column: &str,
    z_column: &str,
    min_qv: f32,
    ignore_z_column: bool,
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>) {

    let fmt = determine_format(path, &None);

    match fmt {
        OutputFormat::Csv => {
            let mut rdr = csv::Reader::from_path(path).unwrap();
            return read_transcripts_csv_xyz(
                &mut rdr,
                transcript_column,
                id_column,
                compartment_column,
                compartment_nuclear,
                fov_column,
                cell_id_column,
                cell_id_unassigned,
                qv_column,
                x_column,
                y_column,
                z_column,
                min_qv,
                ignore_z_column,
            );
        },
        OutputFormat::CsvGz => {
            let mut rdr = csv::Reader::from_reader(GzDecoder::new(File::open(path).unwrap()));
            return read_transcripts_csv_xyz(
                &mut rdr,
                transcript_column,
                id_column,
                compartment_column,
                compartment_nuclear,
                fov_column,
                cell_id_column,
                cell_id_unassigned,
                qv_column,
                x_column,
                y_column,
                z_column,
                min_qv,
                ignore_z_column,
            );
        },
        OutputFormat::Parquet => unimplemented!("Parquet input not supported yet"),
    };
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
    id_column: Option<String>,
    compartment_column: &str,
    compartment_nuclear: &str,
    fov_column: Option<String>,
    cell_id_column: &str,
    cell_id_unassigned: &str,
    qv_column: Option<String>,
    x_column: &str,
    y_column: &str,
    z_column: &str,
    min_qv: f32,
    ignore_z_column: bool,
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
    let id_col = id_column.map(|id_column| find_column(headers, &id_column));

    let cell_id_col = find_column(headers, cell_id_column);
    let compartment_col = find_column(headers, compartment_column);
    let qv_col = qv_column.map(|qv_column| find_column(headers, &qv_column));
    let fov_col = fov_column.map(|fov_column| find_column(headers, &fov_column));

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut cell_assignments = Vec::new();
    let mut fovs = Vec::new();

    let mut fov_map: HashMap<String, u32> = HashMap::new();
    let mut cell_id_map: HashMap<(u32, String), CellIndex> = HashMap::new();


    for result in rdr.records() {
        let row = result.unwrap();

        let qv = if let Some(qv_col) = qv_col {
            row[qv_col].parse::<f32>().unwrap()
        } else {
            f32::INFINITY
        };

        if qv < min_qv {
            continue;
        }

        let fov = if let Some(fov_col) = fov_col {
            match fov_map.get(&row[fov_col]) {
                Some(fov) => *fov,
                None => {
                    let next_fov = fov_map.len();
                    fov_map.insert(row[fov_col].to_string(), next_fov as u32);
                    next_fov as u32
                }
            }
        } else {
            0
        };

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
        let transcript_id = if let Some(id_col) = id_col {
            row[id_col].parse::<u64>().expect(&format!("Transcript ID must be an integer: {}", &row[id_col]))
        } else {
            transcripts.len() as u64
        };

        transcripts.push(Transcript {
            transcript_id,
            x,
            y,
            z: if ignore_z_column { 0.0 } else { z },
            gene: gene as u32,
        });

        fovs.push(fov);

        let cell_id_str = &row[cell_id_col];
        let compartment = &row[compartment_col];
        // let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();

        // Earlier version of Xenium used numeric cell ids and -1 for unassigned.
        // Newer versions use alphanumeric hash codes and "UNASSIGNED" for unasssigned.
        if cell_id_str == cell_id_unassigned {
            cell_assignments.push(BACKGROUND_CELL);
        } else {
            let next_cell_id = cell_id_map.len() as CellIndex;
            let cell_id = *cell_id_map
                .entry((fov, cell_id_str.to_string()))
                .or_insert_with(|| next_cell_id);

            if compartment == compartment_nuclear {
                cell_assignments.push(cell_id);
            } else {
                cell_assignments.push(BACKGROUND_CELL);
            }
        }
    }

    // per-fov zscore normalization of z coordinate.
    // TODO: make this an option
    // normalize_z_coord(&mut transcripts, fovs);

    // Sort on x for better memory locality (This doesn't actually seem to make any difference)
    // let mut ord = (0..transcripts.len())
    //     .collect::<Vec<_>>();
    // ord.sort_unstable_by(|&i, &j| transcripts[i].x.partial_cmp(&transcripts[j].x).unwrap());
    // let transcripts = ord.iter().map(|&i| transcripts[i]).collect::<Vec<_>>();
    // let mut cell_assignments = ord.iter().map(|&i| cell_assignments[i]).collect::<Vec<_>>();

    let cell_population = postprocess_cell_assignments(&mut cell_assignments);

    return (
        transcript_names,
        transcripts,
        cell_assignments,
        cell_population,
    );
}


// pub fn normalize_z_coord(transcripts: &mut Vec<Transcript>, fovs: Vec<u32>) {
//     let nfovs = (*fovs.iter().max().unwrap() + 1) as usize;
//     let mut z_mean = vec![0.0; nfovs];
//     let mut fov_pop = vec![0; nfovs];
//     for (transcript, fov) in transcripts.iter().zip(fovs.iter()) {
//         z_mean[*fov as usize] += transcript.z;
//         fov_pop[*fov as usize] += 1;
//     }

//     for (fov, &pop) in fov_pop.iter().enumerate() {
//         if pop > 0 {
//             z_mean[fov] /= pop as f32;
//         }
//     }

//     let mut z_std = vec![0.0; nfovs];
//     for (transcript, fov) in transcripts.iter().zip(fovs.iter()) {
//         z_std[*fov as usize] += (transcript.z - z_mean[*fov as usize]).powi(2);
//     }

//     for (fov, &pop) in fov_pop.iter().enumerate() {
//         if pop > 0 {
//             z_std[fov] = (z_std[fov] / pop as f32).sqrt();
//         }
//     }

//     for (transcript, fov) in transcripts.iter_mut().zip(fovs.iter()) {
//         transcript.z = (transcript.z - z_mean[*fov as usize]) / z_std[*fov as usize];
//     }
// }


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

    dbg!(occupied.iter().filter(|&&x| x).count(), occupied.len(),);

    return occupied.iter().filter(|&&x| x).count() as f32 * binsize * binsize;
}

// Estimate cell centroids by averaging the coordinates of all transcripts assigned to each cell.
pub fn estimate_cell_centroids(
    transcripts: &Vec<Transcript>,
    cell_assignments: &Vec<CellIndex>,
    ncells: usize,
) -> Vec<(f32, f32)> {
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

pub fn filter_cellfree_transcripts(
    transcripts: &Vec<Transcript>,
    init_cell_assignments: &Vec<CellIndex>,
    ncells: usize,
    max_distance: f32,
) -> (Vec<Transcript>, Vec<CellIndex>) {
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

    let filtered_transcripts = transcripts
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(t, _)| t)
        .cloned()
        .collect::<Vec<_>>();
    let filtered_cell_assignments = init_cell_assignments
        .iter()
        .zip(mask.iter())
        .filter(|(_, &m)| m)
        .map(|(t, _)| t)
        .cloned()
        .collect::<Vec<_>>();

    return (filtered_transcripts, filtered_cell_assignments);
}
