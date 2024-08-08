
// Maintain file schemas that are used in both by output.rs ond by to_baysor.rs

use arrow::datatypes::{Schema, Field, DataType};

pub fn transcript_metadata_schema() -> Schema {
    Schema::new(vec![
        Field::new("transcript_id", DataType::UInt64, false),
        Field::new("x", DataType::Float32, false),
        Field::new("y", DataType::Float32, false),
        Field::new("z", DataType::Float32, false),
        Field::new("observed_x", DataType::Float32, false),
        Field::new("observed_y", DataType::Float32, false),
        Field::new("observed_z", DataType::Float32, false),
        Field::new("gene", DataType::LargeUtf8, false),
        Field::new("qv", DataType::Float32, false),
        Field::new("fov", DataType::LargeUtf8, false),
        Field::new("assignment", DataType::UInt32, false),
        Field::new("probability", DataType::Float32, false),
        Field::new("background", DataType::UInt8, false),
        Field::new("confusion", DataType::UInt8, false),
    ])
}