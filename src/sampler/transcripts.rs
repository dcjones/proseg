use arrow;
use csv;
use flate2::read::{GzDecoder, MultiGzDecoder};
use itertools::izip;
use json;
use kiddo::SquaredEuclidean;
use kiddo::float::kdtree::KdTree;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use ndarray::{Array1, Array2, Zip};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use rand::Rng;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::str;

pub type CellIndex = u32;
pub const BACKGROUND_CELL: CellIndex = u32::MAX;

// Should probably rearrange this...
use super::super::output::infer_format_from_filename;
use super::runvec::RunVec;
use crate::schemas::OutputFormat;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Transcript {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: u32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PriorTranscriptSeg {
    pub nucleus: CellIndex,
    pub cell: CellIndex,
}

pub struct TranscriptDataset {
    pub transcripts: RunVec<u32, Transcript>, // [ntranscripts]
    pub priorseg: RunVec<u32, PriorTranscriptSeg>, // [ntranscripts]
    pub fovs: RunVec<usize, u32>,             // [ntranscripts] (Why to we need to save this?)
    pub barcode_positions: Option<HashMap<String, (f32, f32)>>,

    pub gene_names: Vec<String>,        // [ngenes]
    pub fov_names: Vec<String>,         // [nfovs]
    pub original_cell_ids: Vec<String>, // [ncells]
    pub ncells: usize,
}

impl TranscriptDataset {
    pub fn select_unfactored_genes(&mut self, _nunfactored: usize) {
        // Current heuristic is just to select the highest expression genes.
        let mut gene_counts = Array1::<u32>::zeros(self.gene_names.len());
        for transcript_run in self.transcripts.iter_runs() {
            gene_counts[transcript_run.value.gene as usize] += transcript_run.len;
        }

        let mut ord = (0..self.gene_names.len()).collect::<Vec<_>>();
        ord.sort_unstable_by(|&i, &j| gene_counts[i].cmp(&gene_counts[j]).reverse());

        let mut rev_ord = vec![0; ord.len()];
        for (i, j) in ord.iter().enumerate() {
            rev_ord[*j] = i;
        }

        self.gene_names = ord.iter().map(|&i| self.gene_names[i].clone()).collect();
        for transcript_run in self.transcripts.iter_runs_mut() {
            transcript_run.value.gene = rev_ord[transcript_run.value.gene as usize] as u32;
        }
    }

    // Estimate cell centroids by averaging the coordinates of all transcripts assigned to each cell.
    pub fn estimate_cell_centroids(&self) -> Array2<f32> {
        let mut centroids = Array2::zeros((self.ncells, 2));
        let mut centroids_count = Array1::<u32>::zeros(self.ncells);

        for (prior, transcript) in self.priorseg.iter().zip(self.transcripts.iter()) {
            if prior.cell == BACKGROUND_CELL {
                continue;
            }

            let i = prior.cell as usize;
            centroids[[i, 0]] += transcript.x;
            centroids[[i, 1]] += transcript.y;
            centroids_count[i] += 1;
        }

        Zip::from(centroids.rows_mut())
            .and(&centroids_count)
            .for_each(|mut centroid, count| {
                centroid[0] /= *count as f32;
                centroid[1] /= *count as f32;
            });

        centroids
    }

    pub fn filter_cellfree_transcripts(&mut self, max_distance: f32) {
        let max_distance_squared = max_distance * max_distance;

        let centroids = self.estimate_cell_centroids();

        let mut kdtree: KdTree<f32, u32, 2, 32, u32> = KdTree::with_capacity(centroids.len());
        for (i, xy) in centroids.rows().into_iter().enumerate() {
            if !xy[0].is_finite() || !xy[1].is_finite() {
                continue;
            }
            kdtree.add(&[xy[0], xy[1]], i as u32);
        }

        let mut mask = vec![false; self.transcripts.len()];
        for (i, t) in self.transcripts.iter().enumerate() {
            let d = kdtree.nearest_one::<SquaredEuclidean>(&[t.x, t.y]).distance;

            if d <= max_distance_squared {
                mask[i] = true;
            }
        }

        self.transcripts.retain_masked(&mask);
        self.priorseg.retain_masked(&mask);
        self.fovs.retain_masked(&mask);
    }

    pub fn normalize_z_coordinates(&mut self) -> (f32, f32) {
        let xs = self
            .transcripts
            .iter_runs()
            .map(|run| run.value.x)
            .collect::<Vec<_>>();
        let ys = self
            .transcripts
            .iter_runs()
            .map(|run| run.value.y)
            .collect::<Vec<_>>();
        let mut zs = self
            .transcripts
            .iter_runs()
            .map(|run| run.value.z)
            .collect::<Vec<_>>();

        regress_out_tilt(&xs, &ys, &mut zs);

        // clip extreme quantiles to avoid the z-axis binning being defined by a few outliers
        let mut zs_sorted = zs.clone();
        zs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let (q0, q1) = (0.01, 0.99);
        let zmin = zs_sorted[(q0 * (zs.len() as f32)) as usize];
        let zmax = zs_sorted[(q1 * (zs.len() as f32)) as usize];
        let zspan = if zmin == zmax { 1.0 } else { zmax - zmin };

        zs.iter_mut()
            .for_each(|z| *z = z.max(zmin).min(zmax) / zspan);

        for (run, z) in self.transcripts.iter_runs_mut().zip(zs.iter()) {
            run.value.z = *z;
        }

        (zmin, zmax)
    }

    pub fn z_mean(&self) -> f32 {
        let z_sum = self
            .transcripts
            .iter_runs()
            .map(|run| run.len as f64 * run.value.z as f64)
            .sum::<f64>();
        let z_mean = z_sum / self.transcripts.len as f64;
        z_mean as f32
    }

    pub fn coordinate_span(&self) -> (f32, f32, f32, f32, f32, f32) {
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;

        for run in self.transcripts.iter_runs() {
            let t = &run.value;
            min_x = min_x.min(t.x);
            max_x = max_x.max(t.x);
            min_y = min_y.min(t.y);
            max_y = max_y.max(t.y);
            min_z = min_z.min(t.z);
            max_z = max_z.max(t.z);
        }

        (min_x, max_x, min_y, max_y, min_z, max_z)
    }

    pub fn estimate_full_volume(&self) -> f32 {
        let (xmin, xmax, ymin, ymax, zmin, zmax) = self.coordinate_span();

        const BINSIZE: f32 = 10.0;

        let xbins = ((xmax - xmin) / BINSIZE).ceil() as usize;
        let ybins = ((ymax - ymin) / BINSIZE).ceil() as usize;

        let mut occupied = Array2::from_elem((xbins, ybins), false);

        for run in self.transcripts.iter_runs() {
            let xbin = ((run.value.x - xmin) / BINSIZE).floor() as usize;
            let ybin = ((run.value.y - ymin) / BINSIZE).floor() as usize;

            occupied[[xbin, ybin]] = true;
        }

        let zspan = if zmin == zmax { 1.0 } else { zmax - zmin };

        occupied.iter().filter(|&&x| x).count() as f32 * BINSIZE * BINSIZE * zspan
    }

    pub fn prior_nuclei_populations(&self) -> Array1<u32> {
        let mut counts = Array1::zeros(self.ncells);
        for run in self.priorseg.iter_runs() {
            if run.value.nucleus != BACKGROUND_CELL {
                counts[run.value.nucleus as usize] += run.len;
            }
        }
        counts
    }

    pub fn ngenes(&self) -> usize {
        self.gene_names.len()
    }
}

fn read_visium_features_tsv(filename: &str) -> Vec<String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_reader(GzDecoder::new(File::open(filename).unwrap()));

    let mut genes = Vec::new();
    for result in rdr.records() {
        let row = result.unwrap();
        // TODO: row[0] is ensembl id, row[2] is gene type
        // We may wan to keep this info
        genes.push(row[1].to_string());
    }

    genes
}

fn read_visium_barcodes_tsv(filename: &str) -> Vec<String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b'\t')
        .from_reader(GzDecoder::new(File::open(filename).unwrap()));

    let mut barcodes = Vec::new();
    for result in rdr.records() {
        let row = result.unwrap();
        barcodes.push(row[0].to_string());
    }

    barcodes
}

fn read_visium_tissue_positions_parquet(
    filename: &str,
    microns_per_pixel: f32,
) -> HashMap<String, (f32, f32)> {
    let input_file =
        File::open(filename).unwrap_or_else(|_| panic!("Unable to open '{}'.", &filename));
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).unwrap();
    let schema = builder.schema().as_ref().clone();
    let rdr = builder
        .build()
        .unwrap_or_else(|_| panic!("Unable to read parquet data from frobm {}", filename));

    let barcode_col_idx = schema.index_of("barcode").unwrap();
    let row_px_col_idx = schema.index_of("pxl_row_in_fullres").unwrap();
    let col_px_col_idx = schema.index_of("pxl_col_in_fullres").unwrap();

    let mut barcode_positions = HashMap::new();
    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        let barcodes = rec_batch
            .column(barcode_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        let row_pxs = rec_batch
            .column(row_px_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        let col_pxs = rec_batch
            .column(col_px_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();

        // TODO: Should row and col be flipped here?
        for (barcode, row_px, col_px) in izip!(barcodes, row_pxs, col_pxs) {
            barcode_positions.insert(
                barcode.unwrap().to_string(),
                (
                    microns_per_pixel * row_px.unwrap() as f32,
                    microns_per_pixel * col_px.unwrap() as f32,
                ),
            );
        }
    }

    barcode_positions
}

fn read_visium_scalefactors(filename: &str) -> f32 {
    let file = File::open(filename).unwrap();
    let json_str = std::io::read_to_string(file).unwrap();
    let parsed = json::parse(&json_str).unwrap();

    parsed["microns_per_pixel"].as_f32().unwrap()
}

pub fn read_visium_data(path: &str, excluded_genes: Option<Regex>) -> TranscriptDataset {
    const SQUARE_DIR: &str = "square_002um";
    const MATRIX_DIR: &str = "raw_feature_bc_matrix";

    let path = Path::new(path);

    let gene_names = read_visium_features_tsv(
        path.join(SQUARE_DIR)
            .join(MATRIX_DIR)
            .join("features.tsv.gz")
            .to_str()
            .unwrap(),
    );

    let gene_index: HashMap<usize, usize> = if let Some(excluded_genes) = excluded_genes {
        gene_names
            .iter()
            .enumerate()
            .filter_map(|(i, gene)| {
                if excluded_genes.is_match(gene) {
                    None
                } else {
                    Some(i)
                }
            })
            .enumerate()
            .map(|(j, i)| (i, j))
            .collect()
    } else {
        (1..gene_names.len()).enumerate().collect()
    };

    let barcodes = read_visium_barcodes_tsv(
        path.join(SQUARE_DIR)
            .join(MATRIX_DIR)
            .join("barcodes.tsv.gz")
            .to_str()
            .unwrap(),
    );

    let microns_per_pixel = read_visium_scalefactors(
        path.join(SQUARE_DIR)
            .join("spatial")
            .join("scalefactors_json.json")
            .to_str()
            .unwrap(),
    );

    let barcode_positions = read_visium_tissue_positions_parquet(
        path.join(SQUARE_DIR)
            .join("spatial")
            .join("tissue_positions.parquet")
            .to_str()
            .unwrap(),
        microns_per_pixel,
    );

    // read count matrix
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .comment(Some(b'%'))
        .delimiter(b' ')
        .from_reader(GzDecoder::new(
            File::open(path.join(SQUARE_DIR).join(MATRIX_DIR).join("matrix.mtx.gz")).unwrap(),
        ));

    let headers = rdr.headers().unwrap();
    let ngenes = headers[0].parse::<usize>().unwrap();
    assert_eq!(ngenes, gene_names.len());

    let nsquares = headers[1].parse::<usize>().unwrap();
    assert_eq!(nsquares, barcodes.len());

    let nnz = headers[2].parse::<usize>().unwrap();
    let mut transcripts = RunVec::with_run_capacity(nnz);
    for result in rdr.records() {
        let row = result.unwrap();

        // "-1" beacuse mtx is 1-based indexing
        let gene = gene_index.get(&(row[0].parse::<usize>().unwrap() - 1));
        if gene.is_none() {
            continue; // excluded gene
        }
        let gene = *gene.unwrap() as u32;

        let square = row[1].parse::<usize>().unwrap() - 1;
        let count = row[2].parse::<usize>().unwrap() as u32;

        let (x, y) = barcode_positions[&barcodes[square]];

        transcripts.push_run(Transcript { x, y, z: 0.0, gene }, count);
    }

    let gene_names: Vec<String> = gene_names
        .iter()
        .enumerate()
        .filter_map(|(i, gene)| {
            if gene_index.contains_key(&i) {
                Some(gene.clone())
            } else {
                None
            }
        })
        .collect();

    TranscriptDataset {
        transcripts,
        priorseg: RunVec::new(),
        fovs: RunVec::new(),
        barcode_positions: Some(barcode_positions),
        gene_names,
        fov_names: Vec::new(),
        original_cell_ids: Vec::new(),
        ncells: 0,
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

// Make sure numerical cell ids are sequential with no empty cells.
fn compact_priorseg(priorseg: &mut RunVec<u32, PriorTranscriptSeg>) -> usize {
    let mut used_cell_ids: HashMap<CellIndex, CellIndex> = HashMap::new();

    for assignment_run in priorseg.iter_runs() {
        if assignment_run.value.nucleus != BACKGROUND_CELL {
            let next_cell_id = used_cell_ids.len() as CellIndex;
            used_cell_ids
                .entry(assignment_run.value.nucleus)
                .or_insert(next_cell_id);
        }
        if assignment_run.value.cell != BACKGROUND_CELL {
            let next_cell_id = used_cell_ids.len() as CellIndex;
            used_cell_ids
                .entry(assignment_run.value.cell)
                .or_insert(next_cell_id);
        }
    }
    used_cell_ids.insert(BACKGROUND_CELL, BACKGROUND_CELL);

    for assignment_run in priorseg.iter_runs_mut() {
        assignment_run.value.nucleus = *used_cell_ids.get(&assignment_run.value.nucleus).unwrap();
        assignment_run.value.cell = *used_cell_ids.get(&assignment_run.value.cell).unwrap();
    }

    used_cell_ids.len() - 1
}

#[allow(clippy::too_many_arguments)]
fn read_transcripts_csv_xyz<T>(
    rdr: &mut csv::Reader<T>,
    excluded_genes: Option<Regex>,
    gene_column: &str,
    _id_column: Option<String>,
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
    let gene_col = find_column(headers, gene_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let z_col = if no_z_column {
        ncolumns
    } else {
        find_column(headers, z_column)
    };
    // let id_col = id_column.map(|id_column| find_column(headers, &id_column));

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

    let mut transcripts = RunVec::new();
    let mut gene_name_map: HashMap<String, usize> = HashMap::new();
    let mut gene_names = Vec::new();
    let mut priorseg = RunVec::new();
    let mut fovs = RunVec::new();

    let mut fov_map: HashMap<String, u32> = HashMap::new();

    // Need to map (fov, cell_id), since on some platform (e.g. CosMx cell_ids are only unique within fovs.)
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

        let gene_name = &row[gene_col];

        if let Some(excluded_genes) = &excluded_genes {
            if excluded_genes.is_match(gene_name) {
                continue;
            }
        }

        let gene = if let Some(gene) = gene_name_map.get(gene_name) {
            *gene
        } else {
            gene_names.push(gene_name.to_string());
            gene_name_map.insert(gene_name.to_string(), gene_names.len() - 1);
            gene_names.len() - 1
        };

        let x = coordinate_scale * row[x_col].parse::<f32>().unwrap();
        let y = coordinate_scale * row[y_col].parse::<f32>().unwrap();
        let z = if z_col == ncolumns {
            0.0
        } else {
            row[z_col].parse::<f32>().unwrap()
        };

        transcripts.push(Transcript {
            x,
            y,
            z,
            gene: gene as u32,
        });

        fovs.push(fov);

        if let Some(cell_assignment_col) = cell_assignment_col {
            if row[cell_assignment_col] == cell_assignment_unassigned {
                priorseg.push(PriorTranscriptSeg {
                    nucleus: BACKGROUND_CELL,
                    cell: BACKGROUND_CELL,
                });
                continue;
            }
        };

        let cell_id_str = &row[cell_id_col];
        if cell_id_str == cell_id_unassigned {
            priorseg.push(PriorTranscriptSeg {
                nucleus: BACKGROUND_CELL,
                cell: BACKGROUND_CELL,
            });
        } else {
            let next_cell_id = cell_id_map.len() as CellIndex;
            let cell_id = *cell_id_map
                .entry((fov, cell_id_str.to_string()))
                .or_insert(next_cell_id);

            let is_nuclear = if let Some(compartment_col) = compartment_col {
                row[compartment_col] == compartment_nuclear
            } else {
                // If we have no compartment information, use anything assigned to the cell.
                true
            };

            priorseg.push(PriorTranscriptSeg {
                nucleus: if is_nuclear { cell_id } else { BACKGROUND_CELL },
                cell: cell_id,
            });
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

    let ncells = compact_priorseg(&mut priorseg);
    let mut original_cell_ids = vec![String::new(); cell_id_map.len()];
    for ((_fov, cell_id), i) in cell_id_map {
        original_cell_ids[i as usize] = cell_id;
    }

    let nucleus_population = postprocess_cell_assignments(
        &mut nucleus_assignments,
        &mut cell_assignments,
        &mut original_cell_ids,
    );

    TranscriptDataset {
        transcripts,
        priorseg,
        fovs,
        barcode_positions: None,
        gene_names,
        fov_names,
        original_cell_ids,
        ncells,
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
    let input_file =
        File::open(filename).unwrap_or_else(|_| panic!("Unable to open '{}'.", &filename));
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).unwrap();
    let schema = builder.schema().as_ref().clone();
    let rdr = builder
        .build()
        .unwrap_or_else(|_| panic!("Unable to read parquet data from frobm {}", filename));

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
    gene_col_name: &str,
    _id_col_name: &str,
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
    let gene_col_idx = schema.index_of(gene_col_name).unwrap();
    // let id_col_idx = schema.index_of(id_col_name).unwrap();
    let compartment_col_idx = schema.index_of(compartment_col_name).unwrap();
    let cell_id_col_idx = schema.index_of(cell_id_col_name).unwrap();
    let fov_col_idx = schema.index_of(fov_col_name).unwrap();
    let x_col_idx = schema.index_of(x_col_name).unwrap();
    let y_col_idx = schema.index_of(y_col_name).unwrap();
    let z_col_idx = schema.index_of(z_col_name).unwrap();
    let qv_col_idx = schema.index_of(qv_col_name).unwrap();

    let mut transcripts = RunVec::new();
    let mut gene_name_map: HashMap<String, usize> = HashMap::new();
    let mut gene_names = Vec::new();
    let mut priorseg = RunVec::new();
    let mut fovs = RunVec::new();

    let mut fov_map: HashMap<String, u32> = HashMap::new();
    let mut cell_id_map: HashMap<(u32, String), CellIndex> = HashMap::new();

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        let transcript_col = rec_batch
            .column(gene_col_idx)
            .as_any()
            .downcast_ref::<T>()
            .unwrap();

        // let id_col = rec_batch
        //     .column(id_col_idx)
        //     .as_any()
        //     .downcast_ref::<arrow::array::UInt64Array>()
        //     .unwrap();

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

        for (transcript, compartment, cell_id, fov, x, y, z, qv) in izip!(
            transcript_col,
            compartment_col,
            cell_id_col,
            fov_col,
            x_col,
            y_col,
            z_col,
            qv_col
        ) {
            let transcript = transcript.unwrap();
            // let transcript_id = id.unwrap();
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

            let gene = if let Some(gene) = gene_name_map.get(transcript) {
                *gene
            } else {
                gene_names.push(transcript.to_string());
                gene_name_map.insert(transcript.to_string(), gene_names.len() - 1);
                gene_names.len() - 1
            };

            let x = coordinate_scale * x;
            let y = coordinate_scale * y;

            transcripts.push(Transcript {
                x,
                y,
                z: if ignore_z_column { 0.0 } else { z },
                gene: gene as u32,
            });

            fovs.push(fov);

            if cell_id == cell_id_unassigned {
                priorseg.push(PriorTranscriptSeg {
                    nucleus: BACKGROUND_CELL,
                    cell: BACKGROUND_CELL,
                });
            } else {
                let next_cell_id = cell_id_map.len() as CellIndex;
                let cell_id = *cell_id_map
                    .entry((fov, cell_id.to_string()))
                    .or_insert_with(|| next_cell_id);

                let is_nuclear = compartment == compartment_nuclear;
                priorseg.push(PriorTranscriptSeg {
                    nucleus: if is_nuclear { cell_id } else { BACKGROUND_CELL },
                    cell: cell_id,
                });
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

    let ncells = compact_priorseg(&mut priorseg);
    let mut original_cell_ids = vec![String::new(); cell_id_map.len()];
    for ((_fov, cell_id), i) in cell_id_map {
        original_cell_ids[i as usize] = cell_id;
    }

    let nucleus_population = postprocess_cell_assignments(
        &mut nucleus_assignments,
        &mut cell_assignments,
        &mut original_cell_ids,
    );

    TranscriptDataset {
        transcripts,
        priorseg,
        fovs,
        barcode_positions: None,
        gene_names,
        fov_names,
        original_cell_ids,
        ncells,
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

// Estimate what region of the slide to model by counting the number of occupied bins.

// pub fn estimate_cell_fovs(
//     transcripts: &Vec<Transcript>,
//     cell_assignments: &Vec<CellIndex>,
//     ncells: usize) -> Vec<u32>
// {
//     // TODO: Plan is for each cell to vote, and assign it to the fov that the
//     // majority of its transcripts belong to.

// }

fn regress_out_tilt(xs: &[f32], ys: &[f32], zs: &mut [f32]) {
    assert!(xs.len() == ys.len());
    assert!(xs.len() == zs.len());

    // If all z values are the same, return early (nothing to regress out)
    if zs.is_empty() || zs.iter().all(|&z| z == zs[0]) {
        return;
    }

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
