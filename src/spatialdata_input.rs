use arrow::array::{Array, ArrowPrimitiveType};
use arrow::downcast_dictionary_array;
use geo::geometry::{Coord, LineString, Polygon};
use geo_traits::{CoordTrait, GeometryTrait, LineStringTrait, MultiPolygonTrait, PolygonTrait};
use itertools::izip;
use log::info;
use num::traits::AsPrimitive;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use wkb::reader::{GeometryType, read_wkb};
use zarrs::filesystem::FilesystemStore;
use zarrs::group::Group;
use zarrs::storage::StoreKey;

use crate::anndata_input::read_anndata_zarr_transcripts;
use crate::sampler::runvec::RunVec;
use crate::sampler::transcripts::{
    BACKGROUND_CELL, CellIndex, PriorTranscriptSeg, Transcript, TranscriptDataset, compact_priorseg,
};

// Representation of a set of polygons
pub struct CellPolygons {
    // [npolygons]
    pub cells: Vec<CellIndex>,

    // [npolygons]
    pub polygons: Vec<Polygon<f32>>,

    // [ncells]
    pub original_cell_ids: Vec<String>,
}

impl CellPolygons {
    fn new() -> Self {
        Self {
            cells: Vec::new(),
            polygons: Vec::new(),
            original_cell_ids: Vec::new(),
        }
    }

    pub fn bounding_box(&self) -> (f32, f32, f32, f32) {
        let mut x0 = f32::MAX;
        let mut x1 = f32::MIN;
        let mut y0 = f32::MAX;
        let mut y1 = f32::MIN;

        for polygon in &self.polygons {
            for coord in polygon.exterior().coords() {
                let (x, y) = coord.x_y();
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }

        (x0, x1, y0, y1)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn read_spatialdata_zarr_transcripts(
    filename: &str,
    table: &Option<String>,
    excluded_genes: &Option<Regex>,
    x_column: &str,
    y_column: &str,
    z_column: &Option<String>,
    feature_column: &Option<String>,
    cell_id_column: &Option<String>,
    cell_id_unassigned: &str,
    coordinate_scale: f32,
) -> TranscriptDataset {
    let path = Path::new(filename).to_path_buf();
    if !path.exists() {
        panic!("File/directory not found: {filename}");
    }

    if let Some(table) = table {
        let table_path = path.join("tables").join(table);
        return read_anndata_zarr_transcripts(
            table_path.to_str().unwrap(),
            excluded_genes,
            feature_column,
            cell_id_column,
            cell_id_unassigned,
            coordinate_scale,
        );
    }

    let feature_column = feature_column
        .as_ref()
        .map(|v| v.as_str())
        .unwrap_or("gene");
    if path.is_dir() {
        let store = Arc::new(FilesystemStore::new(path).unwrap());
        read_transcripts_zarr_store(
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
fn read_transcripts_zarr_store(
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

pub fn read_spatialdata_zarr_cell_polygons(
    filename: &str,
    cell_shapes: &str,
    cell_shapes_geometry: &str,
    // TODO: we could make this optional and just count to get cell ids
    cell_shapes_id: &str,
    coordinate_scale: f32,
) -> Option<CellPolygons> {
    let path = Path::new(filename).to_path_buf();
    if !path.exists() {
        panic!("File/directory not found: {filename}");
    }

    if path.is_dir() {
        let store = Arc::new(FilesystemStore::new(path).unwrap());
        read_cell_polygons_zarr_store(
            store,
            cell_shapes,
            cell_shapes_geometry,
            cell_shapes_id,
            coordinate_scale,
        )
    } else {
        unimplemented!("Zipped zarr input is not supoorted. Uncompress the file first.");
    }
}

fn read_cell_polygons_zarr_store(
    store: Arc<FilesystemStore>,
    cell_shapes: &str,
    cell_shapes_geometry: &str,
    cell_shapes_id: &str,
    coordinate_scale: f32,
) -> Option<CellPolygons> {
    let shape_group = Group::open(store.clone(), "/shapes");
    if let Ok(_shape_group) = shape_group {
        let parquet_path = store
            .key_to_fspath(&StoreKey::new(format!("shapes/{cell_shapes}/shapes.parquet")).unwrap());

        let t0 = Instant::now();
        let cell_polygons = read_cell_polygons(
            &parquet_path,
            cell_shapes_geometry,
            cell_shapes_id,
            coordinate_scale,
        );
        info!("Read cell polygons from zarr: {:?}", t0.elapsed());

        Some(cell_polygons)
    } else {
        panic!("Failed to open shape group");
    }
}

fn read_cell_polygons(
    filename: &PathBuf,
    geometry_column: &str,
    cell_id_column: &str,
    coordinate_scale: f32,
) -> CellPolygons {
    let input_file =
        File::open(filename).unwrap_or_else(|_| panic!("Unable to open '{filename:?}'."));
    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file).unwrap();
    let schema = builder.schema().as_ref().clone();
    let rdr = builder
        .build()
        .unwrap_or_else(|_| panic!("Unable to read parquet data from frobm {filename:?}"));

    let geometry_col_idx = schema.index_of(geometry_column).unwrap();
    let cell_id_col_idx = schema.index_of(cell_id_column).unwrap();
    let cell_id_unassigned = "";
    let mut cell_id_map = HashMap::new();
    let mut cell_ids = Vec::new();

    let mut cell_polygons = CellPolygons::new();

    for rec_batch in rdr {
        let rec_batch = rec_batch.expect("Unable to read record batch.");
        cell_ids.clear();
        read_cell_ids(
            rec_batch.column(cell_id_col_idx),
            cell_id_unassigned,
            &mut cell_id_map,
            &mut cell_ids,
        );

        let geometry_byte_arrays = rec_batch
            .column(geometry_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::BinaryArray>()
            .unwrap();

        assert!(geometry_byte_arrays.len() == cell_ids.len());

        for (geometry_byte_array, &cell_id) in geometry_byte_arrays.iter().zip(cell_ids.iter()) {
            let geometry_byte_array = geometry_byte_array.unwrap();
            let geometry = read_wkb(geometry_byte_array).unwrap();

            match geometry.geometry_type() {
                GeometryType::Polygon => {
                    if let geo_traits::GeometryType::Polygon(polygon) = geometry.as_type() {
                        let ext = polygon.exterior().unwrap();
                        let ext_coords = ext
                            .coords()
                            .map(|coord| Coord {
                                x: coordinate_scale * coord.x() as f32,
                                y: coordinate_scale * coord.y() as f32,
                            })
                            .collect();
                        let polygon = Polygon::new(LineString(ext_coords), vec![]);
                        cell_polygons.polygons.push(polygon);
                        cell_polygons.cells.push(cell_id);
                    } else {
                        panic!("Incorrectly encoded Polygon");
                    }
                }
                GeometryType::MultiPolygon => {
                    if let geo_traits::GeometryType::MultiPolygon(multi_polygon) =
                        geometry.as_type()
                    {
                        for polygon in multi_polygon.polygons() {
                            let ext = polygon.exterior().unwrap();
                            let ext_coords = ext
                                .coords()
                                .map(|coord| Coord {
                                    x: coordinate_scale * coord.x() as f32,
                                    y: coordinate_scale * coord.y() as f32,
                                })
                                .collect();
                            let polygon = Polygon::new(LineString(ext_coords), vec![]);
                            cell_polygons.polygons.push(polygon);
                            cell_polygons.cells.push(cell_id);
                        }
                    } else {
                        panic!("Incorrectly encoded MultiPolygon");
                    }
                }
                _ => {
                    panic!("Unsupported geometry type: {:?}", geometry.geometry_type());
                }
            }
        }
    }

    cell_polygons
        .original_cell_ids
        .resize(cell_polygons.cells.len(), String::new());
    for (original_cell_id, cell_id) in cell_id_map.iter() {
        cell_polygons.original_cell_ids[*cell_id as usize] = original_cell_id.clone();
    }

    cell_polygons
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
    let mut gene_names = Vec::new();
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
            &mut gene_names,
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
        gene_names,
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
    gene_names: &mut Vec<String>,
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
            gene_names,
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
    gene_names: &mut Vec<String>,
    gene_exclusion_map: &mut Vec<bool>,
    gene_names_batch: &mut Vec<usize>,
    excluded_genes: &Option<Regex>,
) {
    gene_names_batch.clear();
    gene_names_batch.reserve(arr.len());

    let mut get_gene_index = |gene: String| {
        let gene_name_map_len = gene_name_map.len();
        let idx = gene_name_map.entry(gene.clone()).or_insert_with(|| {
            gene_exclusion_map.push(if let Some(excluded_genes) = excluded_genes {
                excluded_genes.is_match(&gene)
            } else {
                false
            });
            gene_names.push(gene);
            gene_name_map_len
        });
        *idx
    };

    downcast_dictionary_array!(
        arr => match arr.values().data_type() {
            arrow::datatypes::DataType::Utf8 => {
                for v in arr.downcast_dict::<arrow::array::StringArray>().unwrap() {
                    gene_names_batch.push(get_gene_index(v.unwrap().to_string()))
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
                    gene_names_batch.push(get_gene_index(v.unwrap().to_string()))
                });
        },
        arrow::datatypes::DataType::LargeUtf8 => {
            arr
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap()
                .iter()
                .for_each(|v| {
                    gene_names_batch.push(get_gene_index(v.unwrap().to_string()))
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
