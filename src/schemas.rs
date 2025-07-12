// Hold a constants that are used in writing and reading spatialdata files.

use arrow::datatypes::{DataType, Field, Schema};
use clap::ValueEnum;

pub const SD_SHAPES_NAME: &str = "cell_boundaries";
pub const SD_TRANSCRIPTS_NAME: &str = "transcripts";

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OutputFormat {
    Infer,
    Csv,
    CsvGz,
    Parquet,
}

pub fn transcript_metadata_schema() -> Schema {
    Schema::new(vec![
        Field::new("transcript_id", DataType::UInt64, true),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("observed_x", DataType::Float32, false),
        Field::new("observed_y", DataType::Float32, false),
        Field::new("observed_z", DataType::Float32, false),
        Field::new("gene", DataType::LargeUtf8, false),
        Field::new("assignment", DataType::UInt32, true),
        Field::new("background", DataType::Boolean, false),
    ])
}

pub fn wkb_shapes_schema() -> Schema {
    Schema::new(vec![
        Field::new("cell", DataType::UInt32, false),
        Field::new("geometry", DataType::Binary, false),
    ])
}
