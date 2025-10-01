use arrow::array::ArrowPrimitiveType;
use arrow::downcast_dictionary_array;
use itertools::izip;
use num::traits::AsPrimitive;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::StoreKey;

use crate::sampler::runvec::RunVec;
use crate::sampler::transcripts::{
    BACKGROUND_CELL, CellIndex, PriorTranscriptSeg, Transcript, TranscriptDataset, compact_priorseg,
};

#[allow(clippy::too_many_arguments)]
pub fn read_zarr(
    filename: &str,
    excluded_genes: &Option<Regex>,
    x_column: &str,
    y_column: &str,
    z_column: &Option<String>,
    feature_column: &str,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let path = Path::new(filename).to_path_buf();
    if !path.exists() {
        panic!("File/directory not found: {filename}");
    }

    if path.is_dir() {
        let store = Arc::new(FilesystemStore::new(path).unwrap());
        read_zarr_from_store(
            store,
            excluded_genes,
            x_column,
            y_column,
            z_column,
            feature_column,
            cell_id_column,
            cell_id_unassigned,
            coordinate_scale,
        )
    } else {
        unimplemented!("Zipped zarr input is not supoorted. Uncompress the file first.");
    }
}

#[allow(clippy::too_many_arguments)]
fn read_zarr_from_store(
    store: Arc<FilesystemStore>,
    excluded_genes: &Option<Regex>,
    x_column: &str,
    y_column: &str,
    z_column: &Option<String>,
    feature_column: &str,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let _root_group = Group::open(store.clone(), "/").unwrap();

    // dbg!(root_group.child_group_paths());

    let transcripts_group = Group::open(store.clone(), "/points/transcripts");

    if let Ok(_transcripts_group) = transcripts_group {
        // TODO: Maybe extract some data from this?
        // let transcript_attrib = transcripts_group.attributes();
        // dbg!(transcript_attrib);

        let parquet_dir_path =
            store.key_to_fspath(&StoreKey::new("points/transcripts/points.parquet").unwrap());
        let parquet_filenames: Vec<_> = parquet_dir_path
            .read_dir()
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();

        read_transcript_parquet_files(
            &parquet_filenames,
            excluded_genes,
            x_column,
            y_column,
            z_column,
            feature_column,
            cell_id_column,
            cell_id_unassigned,
            coordinate_scale,
        )
    } else {
        unimplemented!("AnnData zarr files are not yet supported.")
    }
}

#[allow(clippy::too_many_arguments)]
fn read_transcript_parquet_files(
    filenames: &[PathBuf],
    excluded_genes: &Option<Regex>,
    x_column: &str,
    y_column: &str,
    z_column: &Option<String>,
    feature_column: &str,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let mut gene_name_map = HashMap::new();
    let mut cell_id_map = HashMap::new();
    let mut gene_exclusion_mask = Vec::new();
    let mut transcripts = RunVec::new();
    let mut priorseg = RunVec::new();

    for filename in filenames {
        read_transcript_parquet(
            filename,
            excluded_genes,
            x_column,
            y_column,
            z_column,
            feature_column,
            cell_id_column,
            cell_id_unassigned,
            coordinate_scale,
            &mut cell_id_map,
            &mut gene_name_map,
            &mut gene_exclusion_mask,
            &mut transcripts,
            &mut priorseg,
        );
    }

    let ncells = compact_priorseg(&mut priorseg);
    let mut original_cell_ids = vec![String::new(); cell_id_map.len()];
    for (cell_id, i) in cell_id_map {
        original_cell_ids[i as usize] = cell_id;
    }

    TranscriptDataset {
        transcripts,
        transcript_ids: None,
        priorseg,
        fovs: RunVec::new(),
        barcode_positions: None,
        gene_names: Vec::new(),
        fov_names: Vec::new(),
        original_cell_ids,
        ncells,
    }
}

#[allow(clippy::too_many_arguments)]
fn read_transcript_parquet(
    filename: &Path,
    excluded_genes: &Option<Regex>,
    x_column: &str,
    y_column: &str,
    z_column: &Option<String>,
    feature_column: &str,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_scale: f32,
    cell_id_map: &mut HashMap<String, u32>,
    gene_name_map: &mut HashMap<String, usize>,
    gene_exclusion_mask: &mut Vec<bool>,
    transcripts: &mut RunVec<u32, Transcript>,
    priorseg: &mut RunVec<u32, PriorTranscriptSeg>,
) {
    let input_file =
        File::open(filename).unwrap_or_else(|_| panic!("Unable to open '{filename:?}'."));
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).unwrap();
    let schema = builder.schema().as_ref().clone();
    let rdr = builder
        .build()
        .unwrap_or_else(|_| panic!("Unable to read parquet data from frobm {filename:?}"));

    let x_col_idx = schema.index_of(x_column).unwrap();
    let y_col_idx = schema.index_of(y_column).unwrap();
    let z_col_idx = z_column
        .as_ref()
        .map(|z_column| schema.index_of(z_column).unwrap());
    let cell_id_col_idx = cell_id_column
        .as_ref()
        .map(|cell_id_column| schema.index_of(cell_id_column).unwrap());

    let feature_col_idx = schema.index_of(feature_column).unwrap();

    let mut x_batch = Vec::new();
    let mut y_batch = Vec::new();
    let mut z_batch = Vec::new();
    let mut gene_batch = Vec::new();
    let mut cell_id_batch = Vec::new();

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");

        read_parquet_float_array(rec_batch.column(x_col_idx), &mut x_batch);
        read_parquet_float_array(rec_batch.column(y_col_idx), &mut y_batch);
        if let Some(z_col_idx) = z_col_idx {
            read_parquet_float_array(rec_batch.column(z_col_idx), &mut z_batch);
        } else {
            z_batch.resize(x_batch.len(), 0.0);
            z_batch.fill(0.0);
        }

        if let Some(cell_id_col_idx) = cell_id_col_idx {
            read_cell_ids(
                rec_batch.column(cell_id_col_idx),
                cell_id_unassigned,
                cell_id_map,
                &mut cell_id_batch,
            );
        } else {
            cell_id_batch.resize(x_batch.len(), BACKGROUND_CELL);
            cell_id_batch.fill(BACKGROUND_CELL);
        }

        read_gene_names(
            rec_batch.column(feature_col_idx),
            gene_name_map,
            gene_exclusion_mask,
            &mut gene_batch,
            excluded_genes,
        );

        for (&x, &y, &z, &gene, &cell_id) in
            izip!(&x_batch, &y_batch, &z_batch, &gene_batch, &cell_id_batch)
        {
            if gene_exclusion_mask[gene] {
                continue;
            }

            transcripts.push(Transcript {
                x: x * coordinate_scale,
                y: y * coordinate_scale,
                z,
                gene: gene as u32,
            });

            // TODO: support cell compartment
            priorseg.push(PriorTranscriptSeg {
                nucleus: cell_id,
                cell: cell_id,
            })
        }
    }
}

// Read cell ids column, supporting a variety of types it may be.
fn read_cell_ids(
    arr: &Arc<dyn arrow::array::Array>,
    cell_id_unassigned: &str,
    cell_id_map: &mut HashMap<String, CellIndex>,
    cell_ids: &mut Vec<CellIndex>,
) {
    downcast_dictionary_array!(
        arr => match arr.values().data_type() {
            arrow::datatypes::DataType::Utf8 => {
                for v in arr.downcast_dict::<arrow::array::StringArray>().unwrap() {
                    if let Some(v) = v {
                        let cell_id_map_len = cell_id_map.len();
                        let cell_id = *cell_id_map
                            .entry(v.to_string())
                            .or_insert(cell_id_map_len as CellIndex);
                        cell_ids.push(cell_id);
                    } else {
                        cell_ids.push(BACKGROUND_CELL);
                    }
                }
            },
            _ => panic!("Unsupported cell id type")
        },
        arrow::datatypes::DataType::Utf8 => {
            read_cell_ids_str::<i32>(arr, cell_id_unassigned, cell_id_map, cell_ids);
        },
        arrow::datatypes::DataType::LargeUtf8 => {
            read_cell_ids_str::<i64>(arr, cell_id_unassigned, cell_id_map, cell_ids);
        },
        arrow::datatypes::DataType::Int32 => {
            read_cell_ids_int::<arrow::datatypes::Int32Type>(arr, cell_id_unassigned, cell_id_map, cell_ids);
        },
        arrow::datatypes::DataType::UInt32 => {
            read_cell_ids_int::<arrow::datatypes::UInt32Type>(arr, cell_id_unassigned, cell_id_map, cell_ids);
        },
        arrow::datatypes::DataType::Int64 => {
            read_cell_ids_int::<arrow::datatypes::Int64Type>(arr, cell_id_unassigned, cell_id_map, cell_ids);
        },
        arrow::datatypes::DataType::UInt64 => {
            read_cell_ids_int::<arrow::datatypes::UInt64Type>(arr, cell_id_unassigned, cell_id_map, cell_ids);
        },
        _ => panic!("Unsupported cell id type")
    );
}

fn read_cell_ids_str<T: arrow::array::OffsetSizeTrait>(
    arr: &Arc<dyn arrow::array::Array>,
    cell_id_unassigned: &str,
    cell_id_map: &mut HashMap<String, CellIndex>,
    cell_ids: &mut Vec<CellIndex>,
) {
    cell_ids.clear();
    cell_ids.reserve(arr.len());
    arr.as_any()
        .downcast_ref::<arrow::array::GenericByteArray<arrow::datatypes::GenericStringType<T>>>()
        .unwrap()
        .iter()
        .for_each(|v| {
            if let Some(v) = v {
                if v == cell_id_unassigned {
                    cell_ids.push(BACKGROUND_CELL);
                } else {
                    let cell_id_map_len = cell_id_map.len();
                    let cell_id = *cell_id_map
                        .entry(v.to_string())
                        .or_insert(cell_id_map_len as CellIndex);
                    cell_ids.push(cell_id);
                }
            } else {
                cell_ids.push(BACKGROUND_CELL);
            }
        });
}

fn read_cell_ids_int<T: ArrowPrimitiveType>(
    arr: &Arc<dyn arrow::array::Array>,
    cell_id_unassigned: &str,
    cell_id_map: &mut HashMap<String, CellIndex>,
    cell_ids: &mut Vec<CellIndex>,
) where
    T::Native: ToString,
{
    cell_ids.clear();
    cell_ids.reserve(arr.len());
    arr.as_any()
        .downcast_ref::<arrow::array::PrimitiveArray<T>>()
        .unwrap()
        .iter()
        .for_each(|v| {
            if let Some(v) = v {
                let vstr = v.to_string();
                if vstr == cell_id_unassigned {
                    cell_ids.push(BACKGROUND_CELL);
                } else {
                    let cell_id_map_len = cell_id_map.len();
                    let cell_id = *cell_id_map
                        .entry(v.to_string())
                        .or_insert(cell_id_map_len as CellIndex);
                    cell_ids.push(cell_id);
                }
            } else {
                cell_ids.push(BACKGROUND_CELL);
            }
        });
}

// Read gene names from a column that may be various string types. Unique values are stored in `gene_name_map`
// and an array of indices is returned.
fn read_gene_names(
    arr: &Arc<dyn arrow::array::Array>,
    gene_name_map: &mut HashMap<String, usize>,
    gene_exclusion_map: &mut Vec<bool>,
    gene_names: &mut Vec<usize>,
    excluded_genes: &Option<Regex>,
) {
    gene_names.clear();
    gene_names.reserve(arr.len());

    let mut get_gene_index = |gene: String| {
        let gene_name_map_len = gene_name_map.len();
        let idx = gene_name_map.entry(gene.clone()).or_insert_with(|| {
            gene_exclusion_map.push(if let Some(excluded_genes) = excluded_genes {
                excluded_genes.is_match(&gene)
            } else {
                false
            });
            gene_name_map_len
        });
        *idx
    };

    downcast_dictionary_array!(
        arr => match arr.values().data_type() {
            arrow::datatypes::DataType::Utf8 => {
                for v in arr.downcast_dict::<arrow::array::StringArray>().unwrap() {
                    gene_names.push(get_gene_index(v.unwrap().to_string()))
                }
            },
            _ => panic!(
                "Unsupported data type for gene name values: {:?}",
                arr.values().data_type()
            )
        },
        arrow::datatypes::DataType::Utf8 => {
            arr
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap()
                .iter()
                .for_each(|v| {
                    gene_names.push(get_gene_index(v.unwrap().to_string()))
                });
        },
        arrow::datatypes::DataType::LargeUtf8 => {
            arr
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap()
                .iter()
                .for_each(|v| {
                    gene_names.push(get_gene_index(v.unwrap().to_string()))
                });
        },
        _ => {
            panic!(
                "Unsupported data type for gene name values: {:?}",
                arr.data_type()
            );
        }
    );
}

fn read_parquet_float_array_typed<T: ArrowPrimitiveType>(
    arr: &Arc<dyn arrow::array::Array>,
    output: &mut Vec<f32>,
) where
    T::Native: AsPrimitive<f32>,
{
    let arr = arr
        .as_any()
        .downcast_ref::<arrow::array::PrimitiveArray<T>>()
        .unwrap();
    for v in arr.iter() {
        output.push(v.unwrap().as_());
    }
}

fn read_parquet_float_array(arr: &Arc<dyn arrow::array::Array>, output: &mut Vec<f32>) {
    output.clear();
    output.reserve(arr.len());
    match arr.data_type() {
        arrow::datatypes::DataType::Float32 => {
            read_parquet_float_array_typed::<arrow::datatypes::Float32Type>(arr, output);
        }
        arrow::datatypes::DataType::Float64 => {
            read_parquet_float_array_typed::<arrow::datatypes::Float64Type>(arr, output);
        }
        arrow::datatypes::DataType::Int32 => {
            read_parquet_float_array_typed::<arrow::datatypes::Int32Type>(arr, output);
        }
        arrow::datatypes::DataType::UInt32 => {
            read_parquet_float_array_typed::<arrow::datatypes::UInt32Type>(arr, output);
        }
        arrow::datatypes::DataType::Int64 => {
            read_parquet_float_array_typed::<arrow::datatypes::Int64Type>(arr, output);
        }
        arrow::datatypes::DataType::UInt64 => {
            read_parquet_float_array_typed::<arrow::datatypes::UInt64Type>(arr, output);
        }
        _ => {
            panic!(
                "Unsupported data type for float values: {:?}",
                arr.data_type()
            );
        }
    }
}
