
// Maintain file schemas that are used in both by output.rs ond by to_baysor.rs

use arrow::datatypes::{Schema, Field, DataType};
use clap::ValueEnum;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum OutputFormat {
    Infer,
    Csv,
    CsvGz,
    Parquet,
}

pub fn large_utf8_if_parquet(fmt: OutputFormat) -> DataType {
    match fmt {
        OutputFormat::Parquet => DataType::LargeUtf8,
        _ => DataType::Utf8
    }
}

pub fn transcript_metadata_schema(fmt: OutputFormat) -> Schema {
    Schema::new(vec![
        Field::new("transcript_id", DataType::UInt64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("observed_x", DataType::Float32, false),
        Field::new("observed_y", DataType::Float32, false),
        Field::new("observed_z", DataType::Float32, false),
        Field::new("gene", large_utf8_if_parquet(fmt), false),
        Field::new("qv", DataType::Float32, false),
        Field::new("fov", large_utf8_if_parquet(fmt), false),
        Field::new("assignment", DataType::UInt32, false),
        Field::new("probability", DataType::Float32, false),
        Field::new("background", DataType::UInt8, false),
        Field::new("confusion", DataType::UInt8, false),
    ])
}