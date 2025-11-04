use crate::sampler::runvec::RunVec;
use crate::sampler::transcripts::{
    BACKGROUND_CELL, CellIndex, PriorTranscriptSeg, Transcript, TranscriptDataset,
    filter_unexpressed_genes,
};
use itertools::izip;
use num::traits::AsPrimitive;
use regex::Regex;
use std::collections::HashMap;
use std::hash::Hash;
use std::path::Path;
use std::str::FromStr;
use std::string::ToString;
use std::sync::Arc;
use zarrs::array::{Array, DataType, ElementOwned};
use zarrs::array_subset::ArraySubset;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::ReadableStorageTraits;

pub fn read_anndata_zarr_transcripts(
    filename: &str,
    excluded_genes: &Option<Regex>,
    feature_column: &Option<String>,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_key: &str,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let path = Path::new(filename).to_path_buf();
    if !path.exists() {
        panic!("File/directory not found: {filename}");
    }

    if path.is_dir() {
        let store = Arc::new(FilesystemStore::new(path).unwrap());
        read_anndata_zarr_transcripts_from_store(
            store,
            excluded_genes,
            feature_column,
            cell_id_column,
            cell_id_unassigned,
            coordinate_key,
            coordinate_scale,
        )
    } else {
        unimplemented!("Zipped zarr input is not supoorted. Uncompress the file first.");
    }
}

fn read_anndata_zarr_transcripts_from_store(
    store: Arc<FilesystemStore>,
    excluded_genes: &Option<Regex>,
    feature_column: &Option<String>,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_key: &str,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let _root_group = Group::open(store.clone(), "/").unwrap();

    let (data, indices, indptr) = read_counts_matrix(store.clone());

    let gene_names = read_gene_names(store.clone(), feature_column);
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
        (0..gene_names.len()).enumerate().collect()
    };

    let (xs, ys) = read_coordinates(store.clone(), coordinate_key);

    let (cell_ids, original_cell_ids) =
        read_cell_assignments(store.clone(), cell_id_column, cell_id_unassigned, xs.len());

    let mut nruns = 0;
    for (&count, &j) in data.iter().zip(&indices) {
        if gene_index.contains_key(&(j as usize)) {
            nruns += count as usize;
        }
    }

    let mut transcripts: RunVec<u32, Transcript> = RunVec::with_run_capacity(nruns);
    let mut priorseg: RunVec<u32, PriorTranscriptSeg> = RunVec::with_run_capacity(nruns);

    for (&indfrom, &indto, &x, &y, &cell_id) in izip!(
        &indptr[0..indptr.len() - 1],
        &indptr[1..],
        &xs,
        &ys,
        &cell_ids
    ) {
        let range = (indfrom as usize)..(indto as usize);
        for (&count, &j) in data[range.clone()]
            .iter()
            .zip(indices[range.clone()].iter())
        {
            if let Some(&gene) = gene_index.get(&(j as usize)) {
                transcripts.push_run(
                    Transcript {
                        x: x * coordinate_scale,
                        y: y * coordinate_scale,
                        z: 0.0,
                        gene: gene as u32,
                    },
                    count,
                );

                priorseg.push_run(
                    PriorTranscriptSeg {
                        nucleus: cell_id,
                        cell: cell_id,
                    },
                    count,
                );
            }
        }
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

    let gene_names = filter_unexpressed_genes(&mut transcripts, gene_names);

    let ncells = original_cell_ids.len();
    let mut original_cell_ids_vec = vec![String::new(); ncells];
    for (original_cell_id, i) in original_cell_ids {
        original_cell_ids_vec[i as usize] = original_cell_id;
    }

    TranscriptDataset {
        transcripts,
        transcript_ids: None,
        priorseg,
        fovs: RunVec::new(),
        barcode_positions: None,
        gene_names,
        fov_names: Vec::new(),
        original_cell_ids: original_cell_ids_vec,
        ncells,
    }
}

fn read_counts_matrix(store: Arc<FilesystemStore>) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let counts_group = Group::open(store.clone(), "/X").unwrap_or_else(|_err| {
        panic!("Group /X not found in zarr store");
    });

    let attrs = counts_group.attributes();

    if attrs
        .get("encoding-type")
        .unwrap_or(&serde_json::Value::Null)
        != "csr_matrix"
    {
        panic!("Expected AnnData input to encode a csr_matrix");
    }

    let data = read_array1d_fram_path::<u32, FilesystemStore>(store.clone(), "/X/data");
    let indices = read_array1d_fram_path::<u32, FilesystemStore>(store.clone(), "/X/indices");
    let indptr = read_array1d_fram_path::<u32, FilesystemStore>(store.clone(), "/X/indptr");

    (data, indices, indptr)
}

fn read_gene_names(store: Arc<FilesystemStore>, gene_column_name: &Option<String>) -> Vec<String> {
    let gene_column_name: &str = if let Some(gene_column_name) = gene_column_name {
        gene_column_name.as_str()
    } else {
        "_index"
    };
    let path = format!("/var/{gene_column_name}");

    let arr = Array::open(store.clone(), &path)
        .unwrap_or_else(|_err| panic!("No var_names gene column in AnnData object."));
    arr.retrieve_array_subset_elements::<String>(&arr.subset_all())
        .unwrap_or_else(|_err| panic!("Unable to read gene names"))
}

fn read_coordinates(store: Arc<FilesystemStore>, coordinate_key: &str) -> (Vec<f32>, Vec<f32>) {
    let coordinate_path = format!("/obsm/{coordinate_key}");
    let arr = zarrs::array::Array::open(store.clone(), &coordinate_path)
        .unwrap_or_else(|_err| panic!("Array {} not found in zarr store", &coordinate_path));

    // Should be a [ncells, 2] matrix
    let shape = arr.shape();
    assert!(shape.len() == 2);
    assert!(shape[1] == 2);

    let xs = read_float_matrix_column::<f32, FilesystemStore>(&arr, 0);
    let ys = read_float_matrix_column::<f32, FilesystemStore>(&arr, 1);

    (xs, ys)
}

fn read_cell_assignments(
    store: Arc<FilesystemStore>,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    n: usize,
) -> (Vec<u32>, HashMap<String, u32>) {
    if let Some(cell_id_column) = cell_id_column {
        let path = format!("/obs/{cell_id_column}");

        let arr = zarrs::array::Array::open(store.clone(), &path)
            .unwrap_or_else(|_err| panic!("Array {} not found in zarr store", "/obsm/spatial"));

        match arr.data_type() {
            DataType::String => read_ids_from_str(&arr, cell_id_unassigned),
            DataType::Int32 => read_ids_from_int::<i32, FilesystemStore>(&arr, cell_id_unassigned),
            DataType::Int64 => read_ids_from_int::<i64, FilesystemStore>(&arr, cell_id_unassigned),
            DataType::UInt32 => read_ids_from_int::<u32, FilesystemStore>(&arr, cell_id_unassigned),
            DataType::UInt64 => read_ids_from_int::<u64, FilesystemStore>(&arr, cell_id_unassigned),
            _ => panic!("Unsupported data type for cell IDs"),
        }
    } else {
        (vec![BACKGROUND_CELL; n], HashMap::new())
    }
}

fn read_ids_from_str<S>(
    arr: &Array<S>,
    cell_id_unassigned: &str,
) -> (Vec<CellIndex>, HashMap<String, CellIndex>)
where
    S: 'static + ReadableStorageTraits,
{
    let chunk_grid = arr.chunk_grid();
    assert!(chunk_grid.dimensionality() == 1);
    let nchunks = chunk_grid.grid_shape()[0];

    let mut cell_ids = Vec::new();
    let mut original_ids = HashMap::new();

    for i in 0..nchunks {
        let chunk_els = arr
            .retrieve_chunk_elements::<String>(&[i as u64])
            .unwrap_or_else(|_err| panic!(""));
        for el in chunk_els.iter() {
            if el == cell_id_unassigned {
                cell_ids.push(BACKGROUND_CELL)
            } else {
                let next_id = original_ids.len() as CellIndex;
                cell_ids.push(*original_ids.entry(el.clone()).or_insert(next_id));
            }
        }
    }

    (cell_ids, original_ids)
}

fn read_ids_from_int<F, S>(
    arr: &Array<S>,
    cell_id_unassigned: &str,
) -> (Vec<CellIndex>, HashMap<String, CellIndex>)
where
    F: Hash + Eq + ToString + FromStr + ElementOwned + Clone + Copy,
    S: 'static + ReadableStorageTraits,
{
    let chunk_grid = arr.chunk_grid();
    assert!(chunk_grid.dimensionality() == 1);
    let nchunks = chunk_grid.grid_shape()[0];

    let cell_id_unassigned = cell_id_unassigned
        .parse()
        .unwrap_or_else(|_err| panic!("Unassigned cell value is not an integer"));

    let mut original_ids = HashMap::new();
    let mut cell_ids = Vec::new();

    for i in 0..nchunks {
        let chunk_els = arr
            .retrieve_chunk_elements::<F>(&[i as u64])
            .unwrap_or_else(|_err| panic!(""));

        for el in chunk_els.iter() {
            if *el == cell_id_unassigned {
                cell_ids.push(BACKGROUND_CELL)
            } else {
                let next_id = original_ids.len() as CellIndex;
                cell_ids.push(*original_ids.entry(el.clone()).or_insert(next_id));
            }
        }
    }

    let original_ids = original_ids
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();

    (cell_ids, original_ids)
}

fn read_float_matrix_column<T, S>(arr: &Array<S>, j: usize) -> Vec<T>
where
    T: 'static + Copy,
    S: 'static + ReadableStorageTraits,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    match arr.data_type() {
        DataType::Float32 => read_typed_float_matrix_column::<f32, T, S>(arr, j as u64),
        DataType::Float64 => read_typed_float_matrix_column::<f64, T, S>(arr, j as u64),
        _ => panic!("Unsupported data type"),
    }
}

fn read_typed_float_matrix_column<F, T, S>(arr: &Array<S>, j: u64) -> Vec<T>
where
    F: AsPrimitive<T> + ElementOwned,
    T: 'static + Copy,
    S: 'static + ReadableStorageTraits,
{
    let m = arr.shape()[0];
    arr.retrieve_array_subset_elements::<F>(&ArraySubset::new_with_ranges(&[0..m, j..j + 1]))
        .unwrap_or_else(|_err| panic!(""))
        .iter()
        .map(|x| x.as_())
        .collect()
}

fn read_array1d_fram_path<T, S>(store: Arc<FilesystemStore>, path: &str) -> Vec<T>
where
    T: 'static + Copy,
    S: 'static + ReadableStorageTraits,
    i16: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    let arr = zarrs::array::Array::open(store.clone(), path)
        .unwrap_or_else(|_err| panic!("Array {} not found in zarr store", path));
    read_array1d::<T, FilesystemStore>(&arr)
}

fn read_array1d<T, S>(arr: &zarrs::array::Array<S>) -> Vec<T>
where
    T: 'static + Copy,
    S: 'static + ReadableStorageTraits,
    i16: AsPrimitive<T>,
    i32: AsPrimitive<T>,
    i64: AsPrimitive<T>,
    u16: AsPrimitive<T>,
    u32: AsPrimitive<T>,
    u64: AsPrimitive<T>,
    f32: AsPrimitive<T>,
    f64: AsPrimitive<T>,
{
    match arr.data_type() {
        DataType::Int16 => read_typed_array1d::<i16, T, S>(arr),
        DataType::Int32 => read_typed_array1d::<i32, T, S>(arr),
        DataType::Int64 => read_typed_array1d::<i64, T, S>(arr),
        DataType::UInt16 => read_typed_array1d::<u16, T, S>(arr),
        DataType::UInt32 => read_typed_array1d::<u32, T, S>(arr),
        DataType::UInt64 => read_typed_array1d::<u64, T, S>(arr),
        DataType::Float32 => read_typed_array1d::<f32, T, S>(arr),
        DataType::Float64 => read_typed_array1d::<f64, T, S>(arr),
        _ => panic!("Unsupported array type: {:?}", arr.data_type()),
    }
}

fn read_typed_array1d<F, T, S>(arr: &zarrs::array::Array<S>) -> Vec<T>
where
    F: AsPrimitive<T> + zarrs::array::ElementOwned,
    T: 'static + Copy,
    S: 'static + ReadableStorageTraits,
{
    arr.retrieve_array_subset_elements::<F>(&arr.subset_all())
        .unwrap_or_else(|_err| panic!("Unable to retrieve array elements"))
        .iter()
        .map(|x| x.as_())
        .collect()
}
