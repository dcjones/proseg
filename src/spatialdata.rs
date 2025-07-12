// Support for writing (and partially reeading) proseg output to (from)
// SpatialData objects serialized in zarr format.

use arrow::array::RecordBatch;
use geo::geometry::Coord;
use geo::{MapCoords, MultiPolygon};
use ndarray::{Array1, Array2};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression::ZSTD, ZstdLevel};
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;
use serde_json::json;
use std::collections::HashMap;
use std::fs::{File, create_dir};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use wkb::writer::{WriteOptions, write_multi_polygon};
use zarrs::array::ChunkShape;
use zarrs::metadata::v2::{DataTypeMetadataV2, FillValueMetadataV2, MetadataV2};
use zarrs::storage::ReadableWritableStorageTraits;

use super::output::write_transcript_metadata;
use super::sampler::ModelParams;
use super::sampler::runvec::RunVec;
use super::sampler::sparsemat::SparseMat;
use super::sampler::transcripts::Transcript;
use super::sampler::voxelcheckerboard::TranscriptMetadata;
use crate::sampler::voxelcheckerboard::VoxelCheckerboard;
use crate::schemas::*;

pub const SD_TABLE_NAME: &str = "table";

#[allow(clippy::too_many_arguments)]
pub fn write_spatialdata_zarr(
    output_path: &Option<String>,
    filename: &str,
    counts: &SparseMat<u32, u32>,
    params: &ModelParams,
    voxels: &VoxelCheckerboard,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
    transcript_ids: &Option<Vec<u64>>,
    transcript_metadata: &RunVec<u32, TranscriptMetadata>,
    polygons: &Vec<MultiPolygon<f32>>,
    run_metadata: &HashMap<String, String>,
) {
    let path = if let Some(outputpath) = output_path {
        Path::new(outputpath).join(filename)
    } else {
        Path::new(filename).to_path_buf()
    };

    if let Err(e) = write_spatialdata_parts(
        &path,
        counts,
        params,
        voxels,
        cell_centroids,
        original_cell_ids,
        gene_names,
        transcripts,
        transcript_ids,
        transcript_metadata,
        polygons,
        run_metadata,
    ) {
        panic!(
            "Failed to write spatial data zarr file to {}: {}",
            path.display(),
            e
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn write_spatialdata_parts(
    path: &PathBuf,
    counts: &SparseMat<u32, u32>,
    params: &ModelParams,
    voxels: &VoxelCheckerboard,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
    transcript_ids: &Option<Vec<u64>>,
    transcript_metadata: &RunVec<u32, TranscriptMetadata>,
    polygons: &Vec<MultiPolygon<f32>>,
    run_metadata: &HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);

    new_zarr_group(store.clone(), "/", None)?.store_metadata()?;
    write_anndata_zarr(
        store.clone(),
        counts,
        params,
        cell_centroids,
        original_cell_ids,
        gene_names,
        transcripts,
        run_metadata,
    )?;

    write_shapes_zarr(path, store.clone(), polygons)?;
    write_transcripts_zarr(
        path,
        store.clone(),
        transcripts,
        transcript_ids,
        transcript_metadata,
        voxels,
        gene_names,
    )?;

    Ok(())
}

fn write_shapes_zarr<T: ReadableWritableStorageTraits>(
    path: &PathBuf,
    store: Arc<T>,
    polygons: &Vec<MultiPolygon<f32>>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ncells = polygons.len();

    new_zarr_group(store.clone(), "/shapes", None)?.store_metadata()?;

    new_zarr_group(
        store.clone(),
        &format!("/shapes/{SD_SHAPES_NAME}"),
        Some(
            json!({
                "spatialdata_attrs": {
                    "version": "0.2"
                },
                "encoding-type": "ngff:shapes",
                "axes": ["x", "y"],
                "coordinateTransformations": [
                    {
                        "input": {
                            "axes": [
                                {
                                    "name": "x",
                                    "type": "space",
                                    "unit": "unit"
                                },
                                {
                                    "name": "y",
                                    "type": "space",
                                    "unit": "unit"
                                }
                            ],
                            "name": "xy"
                        },
                        "output": {
                            "axes": [
                                {
                                    "name": "x",
                                    "type": "space",
                                    "unit": "unit"
                                },
                                {
                                    "name": "y",
                                    "type": "space",
                                    "unit": "unit"
                                }
                            ],
                            "name": "global"
                        },
                        "type": "identity"
                    }
                ],
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    let schema = wkb_shapes_schema();
    let mut buf = Vec::new();
    let wkb_write_opts = WriteOptions::default();
    let polygon_data = polygons
        .iter()
        .map(|poly| {
            buf.clear();
            write_multi_polygon(
                &mut buf,
                &poly.map_coords(|xy| Coord {
                    x: xy.x as f64,
                    y: xy.y as f64,
                }),
                &wkb_write_opts,
            )
            .ok();
            Some(buf.clone())
        })
        .collect::<arrow::array::BinaryArray>();

    let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new((0..ncells as u32).collect::<arrow::array::UInt32Array>()),
        Arc::new(polygon_data),
    ];

    let batch = RecordBatch::try_new(Arc::new(schema), columns)?;

    // this is the minimal metadata needed for geopandas to successfully read the data
    let geo_metadata_str = json!({
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
            }
        },
        "version": "1.0.0"
    })
    .to_string();

    let props = WriterProperties::builder()
        .set_compression(ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_key_value_metadata(Some(vec![KeyValue::new(
            String::from("geo"),
            Some(geo_metadata_str),
        )]))
        .build();

    let path = path
        .join("shapes")
        .join(SD_SHAPES_NAME)
        .join("shapes.parquet");
    let output = File::create(path)?;

    let mut writer = ArrowWriter::try_new(output, batch.schema(), Some(props)).unwrap();
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

fn write_transcripts_zarr<T: ReadableWritableStorageTraits>(
    path: &PathBuf,
    store: Arc<T>,
    transcripts: &RunVec<u32, Transcript>,
    transcript_ids: &Option<Vec<u64>>,
    transcript_metadata: &RunVec<u32, TranscriptMetadata>,
    voxels: &VoxelCheckerboard,
    gene_names: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    new_zarr_group(store.clone(), "/points", None)?.store_metadata()?;
    new_zarr_group(
        store.clone(),
        &format!("/points/{SD_TRANSCRIPTS_NAME}"),
        Some(
            json!({
                "axes": ["x", "y", "z"],
                "encoding-type": "ngff:points",
                "spatialdata_attrs": {
                    "version": "0.1"
                },
                "coordinateTransformations": [
                    {
                        "input": {
                            "axes": [
                                {
                                    "name": "x",
                                    "type": "space",
                                    "unit": "unit"
                                },
                                {
                                    "name": "y",
                                    "type": "space",
                                    "unit": "unit"
                                },
                                {
                                    "name": "z",
                                    "type": "space",
                                    "unit": "unit"
                                }
                            ],
                            "name": "xyz"
                        },
                        "output": {
                            "axes": [
                                {
                                    "name": "x",
                                    "type": "space",
                                    "unit": "unit"
                                },
                                {
                                    "name": "y",
                                    "type": "space",
                                    "unit": "unit"
                                },
                                {
                                    "name": "z",
                                    "type": "space",
                                    "unit": "unit"
                                }
                            ],
                            "name": "global"
                        },
                        "type": "identity"
                    }
                ],
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    let parquet_path = path
        .join("points")
        .join(SD_TRANSCRIPTS_NAME)
        .join("points.parquet");

    create_dir(&parquet_path).unwrap();

    write_transcript_metadata(
        &Some(parquet_path.into_os_string().into_string().unwrap()),
        &Some(String::from("part.0.parquet")),
        OutputFormat::Parquet,
        voxels,
        transcripts,
        transcript_ids,
        transcript_metadata,
        gene_names,
    );

    Ok(())
}

fn new_zarr_group<T: ReadableWritableStorageTraits>(
    store: Arc<T>,
    path: &str,
    attributes: Option<serde_json::Map<String, serde_json::Value>>,
) -> Result<zarrs::group::Group<T>, zarrs::group::GroupCreateError> {
    let mut metadata = zarrs::metadata::v2::GroupMetadataV2::default();
    if let Some(attributes) = attributes {
        metadata.attributes = attributes;
    }

    zarrs::group::Group::new_with_metadata(store, path, metadata.into())
}

// Choose reasonable chunking. Based on _guess_chunks in zarr-python.
fn guess_chunks_1d(size: usize, typesize: usize) -> usize {
    const INCREMENT_BYTES: usize = 256 * 1024;
    const MIN_BYTES: usize = 128 * 1024;
    const MAX_BYTES: usize = 64 * 1024 * 1024;

    let mut chunks = size.max(1);
    let dset_size = chunks * typesize;
    let mut target_size = ((INCREMENT_BYTES as f64)
        * (2.0_f64).powf((dset_size as f64 / (1024.0 * 1024.0)).log10()))
        as usize;

    target_size = target_size.clamp(MIN_BYTES, MAX_BYTES);

    loop {
        let chunk_bytes = chunks * typesize;

        if (chunk_bytes < target_size
            || (chunk_bytes as f64 - target_size as f64).abs() / (target_size as f64) < 0.5)
            && chunk_bytes < MAX_BYTES
        {
            break;
        }

        if chunks == 1 {
            break;
        }

        chunks = (chunks as f64 / 2.0).ceil() as usize;
    }

    chunks
}

#[allow(clippy::too_many_arguments)]
fn new_zarr_array<T: ReadableWritableStorageTraits>(
    store: Arc<T>,
    path: &str,
    shape: Vec<u64>,
    chunks: ChunkShape,
    dtype: DataTypeMetadataV2,
    fill_value: FillValueMetadataV2,
    compressor: Option<MetadataV2>,
    filters: Option<Vec<MetadataV2>>,
) -> Result<zarrs::array::Array<T>, zarrs::array::ArrayCreateError> {
    let metadata = zarrs::metadata::v2::ArrayMetadataV2::new(
        shape, chunks, dtype, fill_value, compressor, filters,
    );
    zarrs::array::Array::new_with_metadata(store, path, metadata.into())
}

fn write_anndata_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    counts: &SparseMat<u32, u32>,
    params: &ModelParams,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
    run_metadata: &HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    new_zarr_group(store.clone(), "/tables", None)?.store_metadata()?;
    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}"),
        Some(
            json!({
                "encoding-type": "anndata",
                "encoding-version": "0.1.0",
                "instance_key": null,
                "region": null,
                "region_key": null,
                "spatialdata-encoding-type": "ngff:regions_table",
                "version": "0.1"
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    write_anndata_x_zarr(store.clone(), counts)?;
    write_anndata_obs_zarr(store.clone(), params, cell_centroids, original_cell_ids)?;
    write_anndata_var_zarr(store.clone(), params, gene_names, transcripts)?;
    write_anndata_obsm_zarr(store.clone(), cell_centroids)?;

    // Empty fields
    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/layers"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obsp"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/proseg_run"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    for (key, value) in run_metadata {
        let mut arr = new_zarr_array(
            store.clone(),
            &format!("/tables/{SD_TABLE_NAME}/uns/proseg_run/{key}"),
            vec![],
            Vec::<u64>::new().try_into()?,
            DataTypeMetadataV2::Simple(String::from("|O")),
            FillValueMetadataV2::Null,
            None,
            Some(vec![serde_json::from_value(json!({
                        "id": "vlen-utf8"
                    } ))?]),
        )?;
        let attr = arr.attributes_mut();
        attr.insert("encoding-type".to_string(), "string".into());
        attr.insert("encoding-version".to_string(), "0.2.0".into());

        arr.store_chunk_elements(&[], &[value.clone()])?;
        arr.store_metadata()?;
    }

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/varp"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/varm"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    Ok(())
}

fn default_blosc_compressor() -> Result<MetadataV2, serde_json::Error> {
    serde_json::from_value(json!({
        "id": "blosc",
        "blocksize": 0,
        "clevel": 5,
        "cname": "lz4",
        "shuffle": 1
    }))
}

fn write_anndata_obs_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    params: &ModelParams,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let ncells = cell_centroids.shape()[0];

    let cols = vec![
        "cell".to_string(),
        "original_cell_id".to_string(),
        "centroid_x".to_string(),
        "centroid_y".to_string(),
        "centroid_z".to_string(),
        "component".to_string(),
        "volume".to_string(),
        "surface_area".to_string(),
        "scale".to_string(),
    ];

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs"),
        Some(
            json!({
                "encoding-type": "dataframe",
                "encoding-version": "0.2.0",
                "_index": "_index",
                "column-order": cols
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    // _index
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/_index"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 16) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("|O")),
        FillValueMetadataV2::Null,
        Some(default_blosc_compressor()?),
        Some(vec![serde_json::from_value(json!({
                    "id": "vlen-utf8"
                } ))?]),
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "string-array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    arr.store_array_subset_ndarray(
        &[0],
        (0..ncells)
            .map(|i| format!("{i}"))
            .collect::<Array1<String>>(),
    )?;
    arr.store_metadata()?;

    // cell
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/cell"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<u4")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_ndarray(&[0], (0..ncells as u32).collect::<Array1<u32>>())?;
    arr.store_metadata()?;

    // original_cell_id
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/original_cell_id"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 16) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("|O")),
        FillValueMetadataV2::Null,
        Some(default_blosc_compressor()?),
        Some(vec![serde_json::from_value(json!({
                    "id": "vlen-utf8"
                } ))?]),
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "string-array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    arr.store_array_subset_ndarray(
        &[0],
        original_cell_ids
            .iter()
            .cloned()
            .collect::<Array1<String>>(),
    )?;
    arr.store_metadata()?;

    // centroid_x
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/centroid_x"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(&[0], cell_centroids.column(0).to_owned())?;
    arr.store_metadata()?;

    // centroid_y
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/centroid_y"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(&[0], cell_centroids.column(1).to_owned())?;
    arr.store_metadata()?;

    // centroid_z
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/centroid_z"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(&[0], cell_centroids.column(2).to_owned())?;
    arr.store_metadata()?;

    // cluster
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/component"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<u4")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(&[0], params.z.clone())?;
    arr.store_metadata()?;

    // volume
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/volume"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(
        &[0],
        params
            .cell_voxel_count
            .iter()
            .map(|v| v as f32 * params.voxel_volume)
            .collect::<Array1<f32>>(),
    )?;
    arr.store_metadata()?;

    // surface_area
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/surface_area"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(
        &[0],
        params
            .cell_surface_area
            .iter()
            .map(|v| v as f32)
            .collect::<Array1<f32>>(),
    )?;
    arr.store_metadata()?;

    // scale
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/scale"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    arr.store_array_subset_ndarray(&[0], params.cell_scale.clone())?;
    arr.store_metadata()?;

    Ok(())
}

fn write_anndata_var_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    params: &ModelParams,
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ngenes = gene_names.len();

    let mut cols = vec!["gene".to_string(), "total_count".to_string()];
    cols.extend(
        (0..params.λ_bg.shape()[1])
            .map(|k| format!("lambda_bg_{k}"))
            .collect::<Vec<_>>(),
    );

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/var"),
        Some(
            json!({
                "encoding-type": "dataframe",
                "encoding-version": "0.2.0",
                "_index": "_index",
                "column-order": cols
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    // index_
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/var/_index"),
        vec![gene_names.len() as u64],
        vec![guess_chunks_1d(gene_names.len(), 16) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("|O")),
        FillValueMetadataV2::Null,
        Some(default_blosc_compressor()?),
        Some(vec![serde_json::from_value(json!({
                    "id": "vlen-utf8"
                } ))?]),
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "string-array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    arr.store_array_subset_ndarray(&[0], gene_names.iter().cloned().collect::<Array1<String>>())?;
    arr.store_metadata()?;

    // gene (which is just a copy of _index)
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/var/gene"),
        vec![gene_names.len() as u64],
        vec![guess_chunks_1d(gene_names.len(), 16) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("|O")),
        FillValueMetadataV2::Null,
        Some(default_blosc_compressor()?),
        Some(vec![serde_json::from_value(json!({
                    "id": "vlen-utf8"
                } ))?]),
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "string-array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    arr.store_array_subset_ndarray(&[0], gene_names.iter().cloned().collect::<Array1<String>>())?;
    arr.store_metadata()?;

    // total_count
    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/var/total_count"),
        vec![ngenes as u64],
        vec![guess_chunks_1d(ngenes, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<u4")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    let mut total_counts = Array1::<u32>::zeros(ngenes);
    for run in transcripts.iter_runs() {
        total_counts[run.value.gene as usize] += run.len;
    }

    arr.store_array_subset_ndarray(&[0], total_counts)?;
    arr.store_metadata()?;

    // λ_bg_k
    for (k, λ_bg_k) in params.λ_bg.columns().into_iter().enumerate() {
        let arr = new_zarr_array(
            store.clone(),
            &format!("/tables/{SD_TABLE_NAME}/var/lambda_bg_{k}"),
            vec![ngenes as u64],
            vec![guess_chunks_1d(ngenes, 4) as u64].try_into()?,
            DataTypeMetadataV2::Simple(String::from("<f4")),
            FillValueMetadataV2::NaN,
            Some(default_blosc_compressor()?),
            None,
        )?;

        arr.store_array_subset_ndarray(&[0], λ_bg_k.to_owned())?;
        arr.store_metadata()?;
    }

    Ok(())
}

fn write_anndata_obsm_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    cell_centroids: &Array2<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obsm"),
        Some(
            json!({
                "encoding-type": "dict",
                "encoding-version": "0.1.0",
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    let ncells = cell_centroids.shape()[0];

    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obsm/spatial"),
        vec![ncells as u64, 2],
        vec![guess_chunks_1d(ncells, 4) as u64, 1].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_ndarray(&[0, 0], cell_centroids.clone())?;
    arr.store_metadata()?;

    Ok(())
}

fn write_anndata_x_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    counts: &SparseMat<u32, u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/X"),
        Some(
            json!({
                "encoding-type": "csr_matrix",
                "encoding-version": "0.1.0",
                "shape": [
                    counts.m,
                    counts.n,
                ]
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    let mut nnz: u64 = 0;
    for x_c in counts.rows() {
        for (_g, est) in x_c.read().iter_nonzeros() {
            if est != 0 {
                nnz += 1;
            }
        }
    }

    // Just doing the simple thing and building the full arrays
    let mut data = Array1::<u32>::zeros(nnz as usize);
    let mut indices = Array1::<i32>::zeros(nnz as usize);
    let mut indptr = Array1::<i32>::zeros(counts.m + 1);
    let mut offset = 0;
    for (i, row) in counts.rows().enumerate() {
        let row_lock = row.read();
        for (j, count) in row_lock.iter_nonzeros() {
            if count > 0 {
                data[offset] = count;
                indices[offset] = j as i32;
                offset += 1;
            }
        }
        indptr[i] = offset as i32;
    }
    indptr[counts.m] = nnz as i32;

    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/X/data"),
        vec![nnz],
        vec![guess_chunks_1d(nnz as usize, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<u4")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_ndarray(&[0], data).unwrap();
    arr.store_metadata()?;

    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/X/indices"),
        vec![nnz],
        vec![guess_chunks_1d(nnz as usize, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<i4")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_ndarray(&[0], indices).unwrap();
    arr.store_metadata()?;

    let arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/X/indptr"),
        vec![indptr.len() as u64],
        vec![guess_chunks_1d(nnz as usize, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<i4")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_ndarray(&[0], indptr).unwrap();
    arr.store_metadata()?;

    Ok(())
}
