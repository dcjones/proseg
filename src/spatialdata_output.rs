// Support for writing (and partially reeading) proseg output to (from)
// SpatialData objects serialized in zarr format.

use arrow::array::RecordBatch;
use geo::geometry::Coord;
use geo::{MapCoords, MultiPolygon};
use ndarray::{Array1, Array2, s};
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
use super::sampler::csrmat::CSRMat;
use super::sampler::runvec::RunVec;
use super::sampler::transcripts::Transcript;
use super::sampler::voxelcheckerboard::TranscriptMetadata;
use super::sampler::{FlowStats, ModelParams};
use crate::sampler::voxelcheckerboard::VoxelCheckerboard;
use crate::schemas::*;

pub const SD_TABLE_NAME: &str = "table";

#[allow(clippy::too_many_arguments)]
pub fn write_spatialdata_zarr(
    output_path: &Option<String>,
    filename: &str,
    counts: &CSRMat<u32, u32>,
    params: &ModelParams,
    voxels: &VoxelCheckerboard,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
    transcript_ids: &Option<Vec<u64>>,
    transcript_metadata: &RunVec<u32, TranscriptMetadata>,
    polygons: &[MultiPolygon<f32>],
    run_metadata: &HashMap<String, String>,
    exclude_transcripts: bool,
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
        exclude_transcripts,
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
    counts: &CSRMat<u32, u32>,
    params: &ModelParams,
    voxels: &VoxelCheckerboard,
    cell_centroids: &Array2<f32>,
    original_cell_ids: &[String],
    gene_names: &[String],
    transcripts: &RunVec<u32, Transcript>,
    transcript_ids: &Option<Vec<u64>>,
    transcript_metadata: &RunVec<u32, TranscriptMetadata>,
    polygons: &[MultiPolygon<f32>],
    run_metadata: &HashMap<String, String>,
    exclude_transcripts: bool,
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

    if exclude_transcripts {
        new_zarr_group(store.clone(), "/points", None)?.store_metadata()?;
    } else {
        write_transcripts_zarr(
            path,
            store.clone(),
            transcripts,
            transcript_ids,
            transcript_metadata,
            voxels,
            gene_names,
        )?;
    }

    Ok(())
}

fn write_shapes_zarr<T: ReadableWritableStorageTraits>(
    path: &Path,
    store: Arc<T>,
    polygons: &[MultiPolygon<f32>],
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
                "crs": null,
                "edges": "planar",
                "geometry_types": ["MultiPolygon"],
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
    path: &Path,
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

#[allow(clippy::too_many_arguments)]
fn write_anndata_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    counts: &CSRMat<u32, u32>,
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
    write_anndata_obsm_zarr(store.clone(), cell_centroids, &params.φ)?;

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

    write_anndata_transition_counts_zarr(store.clone(), &params.transition_counts)?;

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
        &format!("/tables/{SD_TABLE_NAME}/uns/spatialdata_attrs"),
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

    write_single_string(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/spatialdata_attrs/region"),
        "cell_boundaries",
    )?;

    write_single_string(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/spatialdata_attrs/region_key"),
        "region",
    )?;

    write_single_string(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/spatialdata_attrs/instance_key"),
        "cell",
    )?;

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

        arr.store_array_subset_elements(&arr.subset_all(), &[value.clone()])?;
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

    write_anndata_varm_zarr(store.clone(), &params.θ)?;

    Ok(())
}

fn write_single_string<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    path: &str,
    value: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut arr = new_zarr_array(
        store.clone(),
        path,
        vec![],
        ChunkShape::from(vec![]),
        DataTypeMetadataV2::Simple("|O".to_string()),
        FillValueMetadataV2::String("".to_string()),
        None,
        Some(vec![serde_json::from_value(json!({
                    "id": "vlen-utf8"
                } ))?]),
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "string".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());
    arr.store_array_subset_elements(&arr.subset_all(), &[value.to_string()])?;
    arr.store_metadata()?;

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
        "region".to_string(),
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

    arr.store_array_subset_elements(
        &arr.subset_all(),
        &(0..ncells).map(|i| format!("{i}")).collect::<Vec<String>>(),
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

    arr.store_array_subset_elements(&arr.subset_all(), &(0..ncells as u32).collect::<Vec<u32>>())?;
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

    arr.store_array_subset_elements(&arr.subset_all(), original_cell_ids)?;
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
    arr.store_array_subset_elements(&arr.subset_all(), &cell_centroids.column(0).to_vec())?;
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
    arr.store_array_subset_elements(&arr.subset_all(), &cell_centroids.column(1).to_vec())?;
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
    arr.store_array_subset_elements(&arr.subset_all(), &cell_centroids.column(2).to_vec())?;
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
    arr.store_array_subset_elements(&arr.subset_all(), &params.z.to_vec())?;
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
    arr.store_array_subset_elements(
        &arr.subset_all(),
        &params
            .cell_voxel_count
            .iter()
            .map(|v| v as f32 * params.voxel_volume)
            .collect::<Vec<f32>>(),
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
    arr.store_array_subset_elements(
        &arr.subset_all(),
        &params
            .total_cell_surface_area()
            .iter()
            .map(|v| *v as f32)
            .collect::<Vec<f32>>(),
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
    arr.store_array_subset_elements(&arr.subset_all(), &params.cell_scale.to_vec())?;
    arr.store_metadata()?;

    // region
    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/region"),
        Some(
            json!({
                "encoding-type": "categorical",
                "encoding-version": "0.2.0",
                "ordered": "false"
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    // region categories
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/region/categories"),
        vec![1],
        vec![1].try_into()?,
        DataTypeMetadataV2::Simple(String::from("|O")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        Some(vec![serde_json::from_value(json!({
                    "id": "vlen-utf8"
                } ))?]),
    )?;

    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "string-array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    let regions = vec!["cell_boundaries".to_string(); 1];
    arr.store_array_subset_elements(&arr.subset_all(), &regions)?;
    arr.store_metadata()?;

    // region codes
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obs/region/codes"),
        vec![ncells as u64],
        vec![guess_chunks_1d(ncells, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("|i1")),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    let regions: Vec<i8> = vec![0; ncells];
    arr.store_array_subset_elements(&arr.subset_all(), &regions)?;
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

    arr.store_array_subset_elements(&arr.subset_all(), gene_names)?;
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

    arr.store_array_subset_elements(&arr.subset_all(), gene_names)?;
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

    arr.store_array_subset_elements(&arr.subset_all(), &total_counts.to_vec())?;
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

        arr.store_array_subset_elements(&arr.subset_all(), &λ_bg_k.to_vec())?;
        arr.store_metadata()?;
    }

    Ok(())
}

fn write_anndata_obsm_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    cell_centroids: &Array2<f32>,
    φ: &Array2<f32>,
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

    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obsm/spatial"),
        vec![ncells as u64, 2],
        vec![guess_chunks_1d(ncells, 4) as u64, 1].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;

    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    // Convert Array2 to Vec in row-major (C) order for zarrs
    let cell_centroids_vec: Vec<f32> = if cell_centroids.is_standard_layout() {
        cell_centroids.slice(s![.., 0..2]).iter().copied().collect()
    } else {
        cell_centroids
            .as_standard_layout()
            .slice(s![.., 0..2])
            .iter()
            .copied()
            .collect()
    };
    arr.store_array_subset_elements(&arr.subset_all(), &cell_centroids_vec)?;
    arr.store_metadata()?;

    // metagene_rates
    let nhidden = φ.shape()[1];

    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obsm/metagene_rates"),
        vec![ncells as u64, nhidden as u64],
        vec![guess_chunks_1d(ncells, 4) as u64, nhidden as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;

    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    let φ_vec: Vec<f32> = if φ.is_standard_layout() {
        φ.iter().copied().collect()
    } else {
        φ.as_standard_layout().iter().copied().collect()
    };
    arr.store_array_subset_elements(&arr.subset_all(), &φ_vec)?;
    arr.store_metadata()?;

    Ok(())
}

fn write_anndata_varm_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    θ: &Array2<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
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

    let ngenes = θ.shape()[0];
    let nhidden = θ.shape()[1];

    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/varm/metagene_loadings"),
        vec![ngenes as u64, nhidden as u64],
        vec![guess_chunks_1d(ngenes, 4) as u64, nhidden as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;

    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());

    let θ_vec: Vec<f32> = if θ.is_standard_layout() {
        θ.iter().copied().collect()
    } else {
        θ.as_standard_layout().iter().copied().collect()
    };
    arr.store_array_subset_elements(&arr.subset_all(), &θ_vec)?;
    arr.store_metadata()?;

    Ok(())
}

fn write_anndata_csr_matrix<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    path: &str,
    counts: &CSRMat<u32, u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut nnz: u64 = 0;
    for x_c in counts.rows() {
        for (_g, est) in x_c.read().iter_nonzeros() {
            if est != 0 {
                nnz += 1;
            }
        }
    }

    // Just doing the simple thing and building the full arrays
    let mut data = Vec::with_capacity(nnz as usize);
    let mut indices = Vec::with_capacity(nnz as usize);
    let mut indptr = Vec::with_capacity(counts.m + 1);
    let mut offset = 0;
    for row in counts.rows() {
        indptr.push(offset as i32);
        let row_lock = row.read();
        for (j, count) in row_lock.iter_nonzeros() {
            if count > 0 {
                data.push(count);
                indices.push(j as i32);
                offset += 1;
            }
        }
    }
    indptr.push(nnz as i32);

    write_anndata_csr_matrix_raw(
        store,
        path,
        counts.m,
        counts.n as usize,
        &data,
        &indices,
        &indptr,
        "<u4",
        "<i4",
    )
}

#[allow(clippy::too_many_arguments)]
fn write_anndata_csr_matrix_raw<
    T: ReadableWritableStorageTraits + 'static,
    V: zarrs::array::Element,
    Idx: zarrs::array::Element,
>(
    store: Arc<T>,
    path: &str,
    m: usize,
    n: usize,
    data: &[V],
    indices: &[Idx],
    indptr: &[Idx],
    dtype: &str,
    index_dtype: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    new_zarr_group(
        store.clone(),
        path,
        Some(
            json!({
                "encoding-type": "csr_matrix",
                "encoding-version": "0.1.0",
                "shape": [
                    m,
                    n,
                ]
            })
            .as_object()
            .unwrap()
            .clone(),
        ),
    )?
    .store_metadata()?;

    let nnz = data.len() as u64;

    let arr = new_zarr_array(
        store.clone(),
        &format!("{path}/data"),
        vec![nnz],
        vec![guess_chunks_1d(nnz as usize, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from(dtype)),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_elements(&arr.subset_all(), data)
        .unwrap();
    arr.store_metadata()?;

    let arr = new_zarr_array(
        store.clone(),
        &format!("{path}/indices"),
        vec![nnz],
        vec![guess_chunks_1d(nnz as usize, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from(index_dtype)),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_elements(&arr.subset_all(), indices)
        .unwrap();
    arr.store_metadata()?;

    let arr = new_zarr_array(
        store.clone(),
        &format!("{path}/indptr"),
        vec![indptr.len() as u64],
        vec![guess_chunks_1d(indptr.len(), 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from(index_dtype)),
        FillValueMetadataV2::Number(serde_json::Number::from(0)),
        Some(default_blosc_compressor()?),
        None,
    )?;

    arr.store_array_subset_elements(&arr.subset_all(), indptr)
        .unwrap();
    arr.store_metadata()?;

    Ok(())
}

pub fn write_state_transitions_zarr(
    output_path: &Option<String>,
    filename: &str,
    params: &ModelParams,
    gene_names: &[String],
    output_gene_transitions: bool,
) {
    let path = if let Some(outputpath) = output_path {
        Path::new(outputpath).join(filename)
    } else {
        Path::new(filename).to_path_buf()
    };

    if let Err(e) =
        write_state_transitions_parts(&path, params, gene_names, output_gene_transitions)
    {
        panic!(
            "Failed to write state transitions to {}: {}",
            path.display(),
            e
        )
    }
}

fn write_state_transitions_parts(
    path: &Path,
    params: &ModelParams,
    gene_names: &[String],
    output_gene_transitions: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);

    let ncells = params.ncells();

    // 1. Write aggregated transition matrix
    let mut agg_data = Vec::new();
    let mut agg_indices = Vec::new();
    let mut agg_indptr = Vec::with_capacity(ncells + 1);
    let mut agg_offset = 0;

    for i in 0..ncells {
        agg_indptr.push(agg_offset as i32);
        let row_entries = params.state_transitions.iter_row_sorted(i);

        let mut cell_sums = HashMap::new();
        let mut total_sum = 0.0;

        for &(key, count) in &row_entries {
            total_sum += count as f32;
            if (key.dest_cell as usize) < ncells {
                *cell_sums.entry(key.dest_cell).or_insert(0) += count;
            }
        }

        if total_sum > 0.0 {
            let mut sorted_cells: Vec<_> = cell_sums.into_iter().collect();
            sorted_cells.sort_by_key(|k| k.0);

            for (dest_cell, count) in sorted_cells {
                agg_data.push(count as f32 / total_sum);
                agg_indices.push(dest_cell as i32);
                agg_offset += 1;
            }
        }
    }
    agg_indptr.push(agg_offset as i32);

    write_anndata_csr_matrix_raw(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/obsp/state_transitions"),
        ncells,
        ncells,
        &agg_data,
        &agg_indices,
        &agg_indptr,
        "<f4",
        "<i4",
    )?;

    // 2. Write gene-wise transition matrices
    if output_gene_transitions {
        let ngenes = gene_names.len();
        let mut gene_entries: Vec<Vec<(i64, f32)>> = vec![Vec::new(); ngenes];

        for i in 0..ncells {
            let row_entries = params.state_transitions.iter_row_sorted(i);

            let mut current_gene = None;
            let mut current_sum = 0.0;
            let mut current_entries = Vec::new();

            for &(key, count) in &row_entries {
                if Some(key.gene) != current_gene {
                    if let Some(g) = current_gene {
                        if current_sum > 0.0 {
                            for (dest_cell, c) in current_entries {
                                gene_entries[g as usize].push((
                                    (i * ncells + dest_cell as usize) as i64,
                                    c as f32 / current_sum,
                                ));
                            }
                        }
                    }
                    current_gene = Some(key.gene);
                    current_sum = 0.0;
                    current_entries = Vec::new();
                }

                current_sum += count as f32;
                if (key.dest_cell as usize) < ncells {
                    current_entries.push((key.dest_cell, count));
                }
            }
            // handle last gene in row
            if let Some(g) = current_gene {
                if current_sum > 0.0 {
                    for (dest_cell, c) in current_entries {
                        gene_entries[g as usize].push((
                            (i * ncells + dest_cell as usize) as i64,
                            c as f32 / current_sum,
                        ));
                    }
                }
            }
        }

        let mut data = Vec::new();
        let mut indices: Vec<i64> = Vec::new();
        let mut indptr: Vec<i64> = Vec::with_capacity(ngenes + 1);
        let mut offset: i64 = 0;

        for g in 0..ngenes {
            indptr.push(offset);
            for (idx, val) in &gene_entries[g] {
                data.push(*val);
                indices.push(*idx);
                offset += 1;
            }
        }
        indptr.push(offset);

        if !data.is_empty() {
            write_anndata_csr_matrix_raw(
                store.clone(),
                &format!("/tables/{SD_TABLE_NAME}/varm/state_transitions"),
                ngenes,
                ncells * ncells,
                &data,
                &indices,
                &indptr,
                "<f4",
                "<i8",
            )?;
        }
    }

    Ok(())
}

fn write_anndata_x_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    counts: &CSRMat<u32, u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_anndata_csr_matrix(store, &format!("/tables/{SD_TABLE_NAME}/X"), counts)
}

fn write_anndata_transition_counts_zarr<T: ReadableWritableStorageTraits + 'static>(
    store: Arc<T>,
    transition_counts: &CSRMat<u32, u32>,
) -> Result<(), Box<dyn std::error::Error>> {
    write_anndata_csr_matrix(
        store,
        &format!("/tables/{SD_TABLE_NAME}/obsp/transition_counts"),
        transition_counts,
    )
}

pub fn write_expected_inflow_zarr(
    output_path: &Option<String>,
    filename: &str,
    params: &ModelParams,
    nsamples: usize,
) {
    let path = if let Some(outputpath) = output_path {
        Path::new(outputpath).join(filename)
    } else {
        Path::new(filename).to_path_buf()
    };

    if let Err(e) =
        write_expected_flow_parts(&path, &params.expected_inflow, nsamples, "expected_inflow")
    {
        panic!(
            "Failed to write expected inflow to {}: {}",
            path.display(),
            e
        )
    }
}

pub fn write_expected_outflow_zarr(
    output_path: &Option<String>,
    filename: &str,
    params: &ModelParams,
    nsamples: usize,
) {
    let path = if let Some(outputpath) = output_path {
        Path::new(outputpath).join(filename)
    } else {
        Path::new(filename).to_path_buf()
    };

    if let Err(e) = write_expected_flow_parts(
        &path,
        &params.expected_outflow,
        nsamples,
        "expected_outflow",
    ) {
        panic!(
            "Failed to write expected outflow to {}: {}",
            path.display(),
            e
        )
    }
}

fn write_expected_flow_parts(
    path: &Path,
    flow_matrix: &CSRMat<u32, FlowStats>,
    nsamples: usize,
    layer_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);

    let ncells = flow_matrix.m as usize;
    let ngenes = flow_matrix.n as usize;

    let mut mean_data: Vec<f32> = Vec::new();
    let mut var_data: Vec<f32> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(ncells + 1);
    let mut offset = 0i32;

    for i in 0..ncells {
        indptr.push(offset);

        let flow_read = flow_matrix.row(i).read();
        for (gene, stats) in flow_read.iter_nonzeros() {
            mean_data.push(stats.count as f32 / nsamples as f32);
            var_data.push(stats.variance(nsamples));
            indices.push(gene as i32);
            offset += 1;
        }
    }
    indptr.push(offset);

    // Mean layer (expected flow per sample)
    write_anndata_csr_matrix_raw(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/layers/{layer_name}"),
        ncells,
        ngenes,
        &mean_data,
        &indices,
        &indptr,
        "<f4",
        "<i4",
    )?;

    // Variance layer (sample variance of per-sample flow count)
    write_anndata_csr_matrix_raw(
        store,
        &format!("/tables/{SD_TABLE_NAME}/layers/{layer_name}_var"),
        ncells,
        ngenes,
        &var_data,
        &indices,
        &indptr,
        "<f4",
        "<i4",
    )?;

    Ok(())
}

pub fn write_dispersion_params_zarr(
    output_path: &Option<String>,
    filename: &str,
    params: &ModelParams,
) {
    let path = if let Some(outputpath) = output_path {
        Path::new(outputpath).join(filename)
    } else {
        Path::new(filename).to_path_buf()
    };

    if let Err(e) = write_dispersion_params_parts(&path, params) {
        panic!(
            "Failed to write dispersion params to {}: {}",
            path.display(),
            e
        )
    }
}

fn write_dispersion_params_parts(
    path: &Path,
    params: &ModelParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);

    let ncomponents = params.rφ.shape()[0];
    let nhidden = params.rφ.shape()[1];

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/dispersion_params"),
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

    // rφ: [ncomponents, nhidden] — Gamma shape (NB dispersion) parameter per component
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/dispersion_params/rphi"),
        vec![ncomponents as u64, nhidden as u64],
        vec![guess_chunks_1d(ncomponents, 4) as u64, nhidden as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());
    let rφ_vec: Vec<f32> = if params.rφ.is_standard_layout() {
        params.rφ.iter().copied().collect()
    } else {
        params.rφ.as_standard_layout().iter().copied().collect()
    };
    arr.store_array_subset_elements(&arr.subset_all(), &rφ_vec)?;
    arr.store_metadata()?;

    // sφ: [ncomponents, nhidden] — Gamma scale parameter per component
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/dispersion_params/sphi"),
        vec![ncomponents as u64, nhidden as u64],
        vec![guess_chunks_1d(ncomponents, 4) as u64, nhidden as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());
    let sφ_vec: Vec<f32> = if params.sφ.is_standard_layout() {
        params.sφ.iter().copied().collect()
    } else {
        params.sφ.as_standard_layout().iter().copied().collect()
    };
    arr.store_array_subset_elements(&arr.subset_all(), &sφ_vec)?;
    arr.store_metadata()?;

    // π: [ncomponents] — mixture weights
    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/dispersion_params/pi"),
        vec![ncomponents as u64],
        vec![guess_chunks_1d(ncomponents, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<f4")),
        FillValueMetadataV2::NaN,
        Some(default_blosc_compressor()?),
        None,
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());
    arr.store_array_subset_elements(&arr.subset_all(), &params.π.to_vec())?;
    arr.store_metadata()?;

    Ok(())
}

pub fn write_transcript_posteriors_zarr(
    output_path: &Option<String>,
    filename: &str,
    params: &ModelParams,
    transcripts: &RunVec<u32, Transcript>,
    nsamples: usize,
) {
    let path = if let Some(outputpath) = output_path {
        Path::new(outputpath).join(filename)
    } else {
        Path::new(filename).to_path_buf()
    };

    if let Err(e) = write_transcript_posteriors_parts(&path, params, transcripts, nsamples) {
        panic!(
            "Failed to write transcript posteriors to {}: {}",
            path.display(),
            e
        )
    }
}

fn write_transcript_posteriors_parts(
    path: &Path,
    params: &ModelParams,
    transcripts: &RunVec<u32, Transcript>,
    nsamples: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::sampler::transcripts::BACKGROUND_CELL;

    let transcript_assignment_counts = match &params.transcript_assignment_counts {
        Some(counts) => counts,
        None => return Ok(()),
    };

    let store = Arc::new(zarrs::filesystem::FilesystemStore::new(path)?);

    let ntranscripts = transcript_assignment_counts.len();
    let ncells = params.ncells();

    new_zarr_group(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/transcript_posteriors"),
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

    let mut data: Vec<f32> = Vec::new();
    let mut indices: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = Vec::with_capacity(ntranscripts + 1);
    let mut offset = 0i32;

    for counts in transcript_assignment_counts.iter() {
        indptr.push(offset);
        let map = counts.lock();
        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by_key(|(cell, _)| *cell);
        for (cell, count) in entries {
            if *cell == BACKGROUND_CELL {
                continue;
            }
            let prob = *count as f32 / nsamples as f32;
            data.push(prob);
            indices.push(*cell as i32);
            offset += 1;
        }
    }
    indptr.push(offset);

    write_anndata_csr_matrix_raw(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/transcript_posteriors/posteriors"),
        ntranscripts,
        ncells,
        &data,
        &indices,
        &indptr,
        "<f4",
        "<i4",
    )?;

    let mut genes: Vec<i32> = Vec::with_capacity(ntranscripts);
    for transcript in transcripts.iter() {
        genes.push(transcript.gene as i32);
    }

    let mut arr = new_zarr_array(
        store.clone(),
        &format!("/tables/{SD_TABLE_NAME}/uns/transcript_posteriors/genes"),
        vec![ntranscripts as u64],
        vec![guess_chunks_1d(ntranscripts, 4) as u64].try_into()?,
        DataTypeMetadataV2::Simple(String::from("<i4")),
        FillValueMetadataV2::Number(serde_json::Number::from(-1)),
        Some(default_blosc_compressor()?),
        None,
    )?;
    let attr = arr.attributes_mut();
    attr.insert("encoding-type".to_string(), "array".into());
    attr.insert("encoding-version".to_string(), "0.2.0".into());
    arr.store_array_subset_elements(&arr.subset_all(), &genes)?;
    arr.store_metadata()?;

    Ok(())
}
