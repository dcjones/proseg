use arrow;
use csv;
use flate2::read::MultiGzDecoder;
use itertools::izip;
use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use ndarray::{Array1, Array2};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use rand::Rng;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::str;

pub type CellIndex = u32;
pub const BACKGROUND_CELL: CellIndex = std::u32::MAX;

// Should probably rearrange this...
use super::super::output::infer_format_from_filename;
use crate::schemas::OutputFormat;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transcript {
    pub transcript_id: u64,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: u32,
    pub fov: u32,
}

pub struct TranscriptDataset {
    pub transcript_names: Vec<String>,
    pub transcripts: Vec<Transcript>,
    pub nucleus_assignments: Vec<CellIndex>,
    pub cell_assignments: Vec<CellIndex>,
    pub nucleus_population: Vec<usize>,
    pub fovs: Vec<u32>,
    pub qvs: Vec<f32>,
    pub fov_names: Vec<String>,
    pub original_cell_ids: Vec<String>,
}

impl TranscriptDataset {
    pub fn select_unfactored_genes(&mut self, _nunfactored: usize) {
        // Current heuristic is just to select the highest expression genes.
        let mut gene_counts = Array1::<u32>::zeros(self.transcript_names.len());
        for transcript in self.transcripts.iter() {
            gene_counts[transcript.gene as usize] += 1;
        }

        let mut ord = (0..self.transcript_names.len()).collect::<Vec<_>>();
        ord.sort_unstable_by(|&i, &j| gene_counts[i].cmp(&gene_counts[j]).reverse());

        let mut rev_ord = vec![0; ord.len()];
        for (i, j) in ord.iter().enumerate() {
            rev_ord[*j] = i;
        }

        self.transcript_names = ord
            .iter()
            .map(|&i| self.transcript_names[i].clone())
            .collect();
        for transcript in self.transcripts.iter_mut() {
            transcript.gene = rev_ord[transcript.gene as usize] as u32;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn read_transcripts_csv(
    path: &str,
    excluded_genes: Option<Regex>,
    transcript_column: &str,
    id_column: Option<String>,
    compartment_column: Option<String>,
    compartment_nuclear: Option<String>,
    fov_column: Option<String>,
    cell_assignment_column: Option<String>,
    cell_assignment_unassigned: Option<String>,
    cell_id_column: &str,
    cell_id_unassigned: &str,
    qv_column: Option<String>,
    x_column: &str,
    y_column: &str,
    z_column: &str,
    min_qv: f32,
    ignore_z_column: bool,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let fmt = infer_format_from_filename(path);

    match fmt {
        OutputFormat::Csv => {
            let mut rdr = csv::Reader::from_path(path).unwrap();
            read_transcripts_csv_xyz(
                &mut rdr,
                excluded_genes,
                transcript_column,
                id_column,
                compartment_column,
                compartment_nuclear,
                fov_column,
                cell_assignment_column,
                cell_assignment_unassigned,
                cell_id_column,
                cell_id_unassigned,
                qv_column,
                x_column,
                y_column,
                z_column,
                min_qv,
                ignore_z_column,
                coordinate_scale,
            )
        }
        OutputFormat::CsvGz => {
            let mut rdr = csv::Reader::from_reader(MultiGzDecoder::new(File::open(path).unwrap()));
            read_transcripts_csv_xyz(
                &mut rdr,
                excluded_genes,
                transcript_column,
                id_column,
                compartment_column,
                compartment_nuclear,
                fov_column,
                cell_assignment_column,
                cell_assignment_unassigned,
                cell_id_column,
                cell_id_unassigned,
                qv_column,
                x_column,
                y_column,
                z_column,
                min_qv,
                ignore_z_column,
                coordinate_scale,
            )
        }
        OutputFormat::Parquet => read_xenium_transcripts_parquet(
            path,
            excluded_genes,
            transcript_column,
            &id_column.unwrap(),
            &compartment_column.unwrap(),
            compartment_nuclear.unwrap().parse::<u8>().unwrap(),
            &fov_column.unwrap(),
            cell_id_column,
            cell_id_unassigned,
            &qv_column.unwrap(),
            x_column,
            y_column,
            z_column,
            min_qv,
            ignore_z_column,
            coordinate_scale,
        ),
        OutputFormat::Infer => panic!("Could not infer format of file '{}'", path),
    }
}

fn find_column(headers: &csv::StringRecord, column: &str) -> usize {
    let col = headers.iter().position(|x| x == column);
    match col {
        Some(col) => col,
        None => panic!("Column '{}' not found in CSV file", column),
    }
}

fn find_optional_column(headers: &csv::StringRecord, column: &Option<String>) -> Option<usize> {
    if let Some(column) = column {
        headers.iter().position(|x| x == column)
    } else {
        None
    }
}

fn postprocess_cell_assignments(
    nucleus_assignments: &mut [CellIndex],
    cell_assignments: &mut [CellIndex],
) -> Vec<usize> {
    // reassign cell ids to exclude anything that no initial transcripts assigned
    let mut used_cell_ids: HashMap<CellIndex, CellIndex> = HashMap::new();
    for &cell_id in nucleus_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            let next_id = used_cell_ids.len() as CellIndex;
            used_cell_ids.entry(cell_id).or_insert(next_id);
        }
    }
    for &cell_id in cell_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            let next_id = used_cell_ids.len() as CellIndex;
            used_cell_ids.entry(cell_id).or_insert(next_id);
        }
    }

    for cell_id in nucleus_assignments.iter_mut() {
        if *cell_id != BACKGROUND_CELL {
            *cell_id = *used_cell_ids.get(cell_id).unwrap();
        }
    }

    for cell_id in cell_assignments.iter_mut() {
        if *cell_id != BACKGROUND_CELL {
            *cell_id = *used_cell_ids.get(cell_id).unwrap();
        }
    }

    let ncells = used_cell_ids.len();

    let mut nucleus_population = vec![0; ncells];
    for &cell_id in nucleus_assignments.iter() {
        if cell_id != BACKGROUND_CELL {
            nucleus_population[cell_id as usize] += 1;
        }
    }

    nucleus_population
}

#[allow(clippy::too_many_arguments)]
fn read_transcripts_csv_xyz<T>(
    rdr: &mut csv::Reader<T>,
    excluded_genes: Option<Regex>,
    transcript_column: &str,
    id_column: Option<String>,
    compartment_column: Option<String>,
    compartment_nuclear: Option<String>,
    fov_column: Option<String>,
    cell_assignment_column: Option<String>,
    cell_assignment_unassigned: Option<String>,
    cell_id_column: &str,
    cell_id_unassigned: &str,
    qv_column: Option<String>,
    x_column: &str,
    y_column: &str,
    z_column: &str,
    min_qv: f32,
    no_z_column: bool,
    coordinate_scale: f32,
) -> TranscriptDataset
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let ncolumns = headers.len();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let z_col = if no_z_column {
        ncolumns
    } else {
        find_column(headers, z_column)
    };
    let id_col = id_column.map(|id_column| find_column(headers, &id_column));

    let cell_id_col = find_column(headers, cell_id_column);
    let compartment_col =
        compartment_column.map(|compartment_column| find_column(headers, &compartment_column));
    let has_compartment = compartment_col.is_some();

    let compartment_nuclear = if has_compartment {
        compartment_nuclear.unwrap()
    } else {
        String::new()
    };

    let qv_col = find_optional_column(headers, &qv_column);
    let fov_col = find_optional_column(headers, &fov_column);
    let cell_assignment_col = find_optional_column(headers, &cell_assignment_column);
    let cell_assignment_unassigned = cell_assignment_unassigned.unwrap_or(String::from(""));

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut nucleus_assignments = Vec::new();
    let mut cell_assignments = Vec::new();
    let mut qvs = Vec::new();
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

        if let Some(excluded_genes) = &excluded_genes {
            if excluded_genes.is_match(transcript_name) {
                continue;
            }
        }

        let gene = if let Some(gene) = transcript_name_map.get(transcript_name) {
            *gene
        } else {
            transcript_names.push(transcript_name.to_string());
            transcript_name_map.insert(transcript_name.to_string(), transcript_names.len() - 1);
            transcript_names.len() - 1
        };

        let x = coordinate_scale * row[x_col].parse::<f32>().unwrap();
        let y = coordinate_scale * row[y_col].parse::<f32>().unwrap();
        let z = if z_col == ncolumns {
            0.0
        } else {
            row[z_col].parse::<f32>().unwrap()
        };
        let transcript_id = if let Some(id_col) = id_col {
            row[id_col]
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("Transcript ID must be an integer: {}", &row[id_col]))
        } else {
            transcripts.len() as u64
        };

        transcripts.push(Transcript {
            transcript_id,
            x,
            y,
            z,
            gene: gene as u32,
            fov,
        });

        qvs.push(qv);
        fovs.push(fov);

        if let Some(cell_assignment_col) = cell_assignment_col {
            if row[cell_assignment_col] == cell_assignment_unassigned {
                nucleus_assignments.push(BACKGROUND_CELL);
                cell_assignments.push(BACKGROUND_CELL);
                continue;
            }
        };

        let cell_id_str = &row[cell_id_col];
        // let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();

        // Earlier version of Xenium used numeric cell ids and -1 for unassigned.
        // Newer versions use alphanumeric hash codes and "UNASSIGNED" for unasssigned.
        if cell_id_str == cell_id_unassigned {
            nucleus_assignments.push(BACKGROUND_CELL);
            cell_assignments.push(BACKGROUND_CELL);
        } else {
            let next_cell_id = cell_id_map.len() as CellIndex;
            let cell_id = *cell_id_map
                .entry((fov, cell_id_str.to_string()))
                .or_insert_with(|| next_cell_id);

            let is_nuclear = if let Some(compartment_col) = compartment_col {
                row[compartment_col] == compartment_nuclear
            } else {
                // If we have no compartment information, use anything assigned to the cell.
                true
            };

            if is_nuclear {
                nucleus_assignments.push(cell_id);
            } else {
                nucleus_assignments.push(BACKGROUND_CELL);
            }
            cell_assignments.push(cell_id);
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

    let mut fov_names = vec![String::new(); fov_map.len().max(1)];
    if fov_map.is_empty() {
        fov_names[0] = String::from("0");
    } else {
        for (fov_name, fov) in fov_map {
            fov_names[fov as usize] = fov_name;
        }
    }

    let nucleus_population =
        postprocess_cell_assignments(&mut nucleus_assignments, &mut cell_assignments);

    let mut original_cell_ids = vec![String::new(); cell_id_map.len()];
    for ((_fov, cell_id), i) in cell_id_map {
        original_cell_ids[i as usize] = cell_id;
    }

    TranscriptDataset {
        transcript_names,
        transcripts,
        nucleus_assignments,
        cell_assignments,
        nucleus_population,
        qvs,
        fovs,
        fov_names,
        original_cell_ids,
    }
}

#[allow(clippy::too_many_arguments)]
fn read_xenium_transcripts_parquet(
    filename: &str,
    excluded_genes: Option<Regex>,
    transcript_col_name: &str,
    id_col_name: &str,
    compartment_col_name: &str,
    compartment_nuclear: u8,
    fov_col_name: &str,
    cell_id_col_name: &str,
    cell_id_unassigned: &str,
    qv_col_name: &str,
    x_col_name: &str,
    y_col_name: &str,
    z_col_name: &str,
    min_qv: f32,
    ignore_z_column: bool,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let input_file = File::open(&filename).expect(&format!("Unable to open '{}'.", &filename));
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).unwrap();
    let schema = builder.schema().as_ref().clone();
    let rdr = builder.build().expect(&format!(
        "Unable to read parquet data from frobm {}",
        filename
    ));

    // Xenium parquet files can use i32 or i64 indexes in their string arrays,
    // so we have to dynamically dispatch here.
    let transcript_field = schema.field_with_name(transcript_col_name).unwrap();
    let string_type = transcript_field.data_type();

    match string_type {
        arrow::datatypes::DataType::Utf8 => {
            read_xenium_transcripts_parquet_str_type::<arrow::array::StringArray>(
                rdr,
                schema,
                excluded_genes,
                transcript_col_name,
                id_col_name,
                compartment_col_name,
                compartment_nuclear,
                fov_col_name,
                cell_id_col_name,
                cell_id_unassigned,
                qv_col_name,
                x_col_name,
                y_col_name,
                z_col_name,
                min_qv,
                ignore_z_column,
                coordinate_scale,
            )
        }
        arrow::datatypes::DataType::LargeUtf8 => {
            read_xenium_transcripts_parquet_str_type::<arrow::array::LargeStringArray>(
                rdr,
                schema,
                excluded_genes,
                transcript_col_name,
                id_col_name,
                compartment_col_name,
                compartment_nuclear,
                fov_col_name,
                cell_id_col_name,
                cell_id_unassigned,
                qv_col_name,
                x_col_name,
                y_col_name,
                z_col_name,
                min_qv,
                ignore_z_column,
                coordinate_scale,
            )
        }
        _ => panic!("Unexpected string array type in Xenium parquet file"),
    }
}

#[allow(clippy::too_many_arguments)]
fn read_xenium_transcripts_parquet_str_type<T>(
    rdr: ParquetRecordBatchReader,
    schema: arrow::datatypes::Schema,
    excluded_genes: Option<Regex>,
    transcript_col_name: &str,
    id_col_name: &str,
    compartment_col_name: &str,
    compartment_nuclear: u8,
    fov_col_name: &str,
    cell_id_col_name: &str,
    cell_id_unassigned: &str,
    qv_col_name: &str,
    x_col_name: &str,
    y_col_name: &str,
    z_col_name: &str,
    min_qv: f32,
    ignore_z_column: bool,
    coordinate_scale: f32,
) -> TranscriptDataset
where
    T: 'static,
    for<'a> &'a T: IntoIterator<Item = Option<&'a str>>,
{
    let transcript_col_idx = schema.index_of(transcript_col_name).unwrap();
    let id_col_idx = schema.index_of(id_col_name).unwrap();
    let compartment_col_idx = schema.index_of(compartment_col_name).unwrap();
    let cell_id_col_idx = schema.index_of(cell_id_col_name).unwrap();
    let fov_col_idx = schema.index_of(fov_col_name).unwrap();
    let x_col_idx = schema.index_of(x_col_name).unwrap();
    let y_col_idx = schema.index_of(y_col_name).unwrap();
    let z_col_idx = schema.index_of(z_col_name).unwrap();
    let qv_col_idx = schema.index_of(qv_col_name).unwrap();

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut nucleus_assignments = Vec::new();
    let mut cell_assignments = Vec::new();
    let mut qvs = Vec::new();
    let mut fovs = Vec::new();

    let mut fov_map: HashMap<String, u32> = HashMap::new();
    let mut cell_id_map: HashMap<(u32, String), CellIndex> = HashMap::new();

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        let transcript_col = rec_batch
            .column(transcript_col_idx)
            .as_any()
            .downcast_ref::<T>()
            .unwrap();

        let id_col = rec_batch
            .column(id_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();

        let compartment_col = rec_batch
            .column(compartment_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::UInt8Array>()
            .unwrap();

        let cell_id_col = rec_batch
            .column(cell_id_col_idx)
            .as_any()
            .downcast_ref::<T>()
            .unwrap();

        let fov_col = rec_batch
            .column(fov_col_idx)
            .as_any()
            .downcast_ref::<T>()
            .unwrap();

        let x_col = rec_batch
            .column(x_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();

        let y_col = rec_batch
            .column(y_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();

        let z_col = rec_batch
            .column(z_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();

        let qv_col = rec_batch
            .column(qv_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .unwrap();

        for (transcript, id, compartment, cell_id, fov, x, y, z, qv) in izip!(
            transcript_col,
            id_col,
            compartment_col,
            cell_id_col,
            fov_col,
            x_col,
            y_col,
            z_col,
            qv_col
        ) {
            let transcript = transcript.unwrap();
            let transcript_id = id.unwrap();
            let compartment = compartment.unwrap();
            let cell_id = cell_id.unwrap();
            let fov = fov.unwrap();
            let x = x.unwrap();
            let y = y.unwrap();
            let z = z.unwrap();
            let qv = qv.unwrap();

            if qv < min_qv {
                continue;
            }

            if let Some(excluded_genes) = &excluded_genes {
                if excluded_genes.is_match(transcript) {
                    continue;
                }
            }

            let fov = match fov_map.get(fov) {
                Some(fov) => *fov,
                None => {
                    let next_fov = fov_map.len();
                    fov_map.insert(fov.to_string(), next_fov as u32);
                    next_fov as u32
                }
            };

            let gene = if let Some(gene) = transcript_name_map.get(transcript) {
                *gene
            } else {
                transcript_names.push(transcript.to_string());
                transcript_name_map.insert(transcript.to_string(), transcript_names.len() - 1);
                transcript_names.len() - 1
            };

            let x = coordinate_scale * x;
            let y = coordinate_scale * y;

            transcripts.push(Transcript {
                transcript_id,
                x,
                y,
                z: if ignore_z_column { 0.0 } else { z },
                gene: gene as u32,
                fov,
            });

            qvs.push(qv);
            fovs.push(fov);

            if cell_id == cell_id_unassigned {
                nucleus_assignments.push(BACKGROUND_CELL);
                cell_assignments.push(BACKGROUND_CELL);
            } else {
                let next_cell_id = cell_id_map.len() as CellIndex;
                let cell_id = *cell_id_map
                    .entry((fov, cell_id.to_string()))
                    .or_insert_with(|| next_cell_id);

                let is_nuclear = compartment == compartment_nuclear;

                if is_nuclear {
                    nucleus_assignments.push(cell_id);
                } else {
                    nucleus_assignments.push(BACKGROUND_CELL);
                }
                cell_assignments.push(cell_id);
            }
        }
    }

    let mut fov_names = vec![String::new(); fov_map.len().max(1)];
    if fov_map.is_empty() {
        fov_names[0] = String::from("0");
    } else {
        for (fov_name, fov) in fov_map {
            fov_names[fov as usize] = fov_name;
        }
    }

    let nucleus_population =
        postprocess_cell_assignments(&mut nucleus_assignments, &mut cell_assignments);

    let mut original_cell_ids = vec![String::new(); cell_id_map.len()];
    for ((_fov, cell_id), i) in cell_id_map {
        original_cell_ids[i as usize] = cell_id;
    }

    TranscriptDataset {
        transcript_names,
        transcripts,
        nucleus_assignments,
        cell_assignments,
        nucleus_population,
        qvs,
        fovs,
        fov_names,
        original_cell_ids,
    }
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

    (min_x, max_x, min_y, max_y, min_z, max_z)
}

// Estimate what region of the slide to model by counting the number of occupied bins.
pub fn estimate_full_area(transcripts: &Vec<Transcript>, mean_nucleus_area: f32) -> f32 {
    let (xmin, xmax, ymin, ymax, _, _) = coordinate_span(transcripts);

    const SCALE: f32 = 2.0;
    let binsize = SCALE * mean_nucleus_area.sqrt();

    let xbins = ((xmax - xmin) / binsize).ceil() as usize;
    let ybins = ((ymax - ymin) / binsize).ceil() as usize;

    let mut occupied = Array2::from_elem((xbins, ybins), false);

    for transcript in transcripts {
        let xbin = ((transcript.x - xmin) / binsize).floor() as usize;
        let ybin = ((transcript.y - ymin) / binsize).floor() as usize;

        occupied[[xbin, ybin]] = true;
    }

    occupied.iter().filter(|&&x| x).count() as f32 * binsize * binsize
}

// pub fn estimate_cell_fovs(
//     transcripts: &Vec<Transcript>,
//     cell_assignments: &Vec<CellIndex>,
//     ncells: usize) -> Vec<u32>
// {
//     // TODO: Plan is for each cell to vote, and assign it to the fov that the
//     // majority of its transcripts belong to.

// }

// Estimate cell centroids by averaging the coordinates of all transcripts assigned to each cell.
pub fn estimate_cell_centroids(
    transcripts: &[Transcript],
    cell_assignments: &[CellIndex],
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
        if ts.is_empty() {
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

    centroids
}

pub fn filter_cellfree_transcripts(
    // transcripts: &[Transcript],
    // nucleus_assignments: &[CellIndex],
    // cell_assignments: &[CellIndex],
    dataset: &mut TranscriptDataset,
    ncells: usize,
    max_distance: f32,
) {
    let max_distance_squared = max_distance * max_distance;

    let centroids =
        estimate_cell_centroids(&dataset.transcripts, &dataset.nucleus_assignments, ncells);

    let mut kdtree: KdTree<f32, u32, 2, 32, u32> = KdTree::with_capacity(centroids.len());
    for (i, (x, y)) in centroids.iter().enumerate() {
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        kdtree.add(&[*x, *y], i as u32);
    }

    let mut mask = vec![false; dataset.transcripts.len()];
    for (i, t) in dataset.transcripts.iter().enumerate() {
        let d = kdtree.nearest_one::<SquaredEuclidean>(&[t.x, t.y]).distance;

        if d <= max_distance_squared {
            mask[i] = true;
        }
    }

    dataset.transcripts.clone_from(
        &dataset
            .transcripts
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(t, _)| t)
            .cloned()
            .collect::<Vec<_>>(),
    );

    dataset.nucleus_assignments.clone_from(
        &dataset
            .nucleus_assignments
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(t, _)| t)
            .cloned()
            .collect::<Vec<_>>(),
    );

    dataset.cell_assignments.clone_from(
        &dataset
            .cell_assignments
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(t, _)| t)
            .cloned()
            .collect::<Vec<_>>(),
    );

    dataset.fovs.clone_from(
        &dataset
            .fovs
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(t, _)| t)
            .cloned()
            .collect::<Vec<_>>(),
    );

    dataset.qvs.clone_from(
        &dataset
            .qvs
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(t, _)| t)
            .cloned()
            .collect::<Vec<_>>(),
    );
}

fn regress_out_tilt(xs: &[f32], ys: &[f32], zs: &mut [f32]) {
    assert!(xs.len() == ys.len());
    assert!(xs.len() == zs.len());

    // downsample transcript positions
    let max_points: usize = 1_000_000;
    let npoints = xs.len().min(max_points);

    let mut xvec = Vec::with_capacity(npoints);
    let mut yvec = Vec::with_capacity(npoints);
    let mut zvec = Vec::with_capacity(npoints);

    let mut rng = rand::rng();
    let shuffle_key = (0..npoints)
        .map(|_| rng.random::<u32>())
        .collect::<Vec<_>>();
    let mut random_perm = (0..npoints).collect::<Vec<_>>();
    random_perm.sort_by_key(|&i| shuffle_key[i]);
    for &i in random_perm[0..npoints].iter() {
        xvec.push(xs[i] as f64);
        yvec.push(ys[i] as f64);
        zvec.push(zs[i] as f64);
    }

    // run linear regression
    let data = RegressionDataBuilder::new()
        .build_from([("x", xvec), ("y", yvec), ("z", zvec)])
        .unwrap();
    let model = FormulaRegressionBuilder::new()
        .data(&data)
        .formula("z ~ x + y")
        .fit()
        .unwrap();

    let model_params = model.parameters();
    let b = model_params[0] as f32;
    let wx = model_params[1] as f32;
    let wy = model_params[2] as f32;

    // replace z coordinates with regression residuals
    for (x, y, z) in izip!(xs, ys, zs) {
        *z = *z - b - wx * x - wy * y;
    }
}

pub fn normalize_z_coordinates(dataset: &mut TranscriptDataset) {
    let xs = dataset.transcripts.iter().map(|t| t.x).collect::<Vec<_>>();
    let ys = dataset.transcripts.iter().map(|t| t.y).collect::<Vec<_>>();
    let mut zs = dataset.transcripts.iter().map(|t| t.z).collect::<Vec<_>>();

    regress_out_tilt(&xs, &ys, &mut zs);

    let mut zs_sorted = zs.clone();
    zs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let (q0, q1) = (0.01, 0.99);
    let zmin = zs_sorted[(q0 * (zs.len() as f32)) as usize];
    let zmax = zs_sorted[(q1 * (zs.len() as f32)) as usize];
    // dbg!(zmin, zmax);
    zs.iter_mut().for_each(|z| *z = z.max(zmin).min(zmax));

    for (t, z) in dataset.transcripts.iter_mut().zip(zs.iter()) {
        t.z = *z;
    }
}
