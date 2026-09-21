use clap::Parser;
use json::JsonValue;
use std::fs::File;
use std::io::Write;

#[derive(Parser, Debug)]
#[command(name = "baysor-to-spaceranger")]
#[command(author = "Daniel C. Jones")]
#[command(about = "Convert proseg-to-baysor output to Space Ranger-compatible GeoJSON.")]
struct Args {
    /// Cell polygons GeoJSON in Baysor format, as written by `proseg-to-baysor`.
    baysor_cell_polygons: String,

    /// Output GeoJSON `FeatureCollection` for `spaceranger count --custom-segmentation-file`.
    #[arg(long, default_value = "baysor-to-spaceranger-cell-polygons.geojson")]
    output_cell_polygons: String,

    /// Microns per pixel used to convert coordinates to pixels.
    /// One of `--microns-per-pixel` or `--scalefactors-json` must be used.
    #[arg(long)]
    microns_per_pixel: Option<f32>,

    /// Space Ranger `scalefactors_json.json` to read `microns_per_pixel` from.
    #[arg(long)]
    scalefactors_json: Option<String>,
}

fn main() {
    let args = Args::parse();

    if args.microns_per_pixel.is_some() && args.scalefactors_json.is_some() {
        panic!("Only one of --microns-per-pixel or --scalefactors-json can be used.");
    }

    if args.microns_per_pixel.is_none() && args.scalefactors_json.is_none() {
        panic!("One of --microns-per-pixel or --scalefactors-json must be used.");
    }

    let pixels_per_micron = if let Some(microns_per_pixel) = args.microns_per_pixel {
        1.0 / microns_per_pixel
    } else {
        1.0 / read_visium_scalefactors(args.scalefactors_json.as_ref().unwrap())
    };

    let geometries = read_baysor_cell_polygon_geojson(&args.baysor_cell_polygons);

    write_spaceranger_cell_polygon_geojson(
        geometries,
        args.output_cell_polygons,
        pixels_per_micron,
    );
}

fn read_visium_scalefactors(filename: &str) -> f32 {
    let file = File::open(filename).unwrap_or_else(|_err| panic!("Unable to open '{filename}'."));
    let json_str = std::io::read_to_string(file).unwrap();
    let parsed = json::parse(&json_str).unwrap();

    parsed["microns_per_pixel"].as_f32().unwrap()
}

fn read_baysor_cell_polygon_geojson(filename: &str) -> Vec<JsonValue> {
    let file = File::open(filename).unwrap_or_else(|_err| panic!("Unable to open '{filename}'."));
    let json_str = std::io::read_to_string(file).unwrap();
    let parsed = json::parse(&json_str).unwrap();

    parsed["geometries"].members().cloned().collect()
}

fn scale_coords(coords: &JsonValue, pixels_per_micron: f32) -> JsonValue {
    let mut scaled_coords = Vec::new();

    for point in coords[0].members() {
        let x = point[0].as_f32().unwrap();
        let y = point[1].as_f32().unwrap();

        scaled_coords.push(json::array![x * pixels_per_micron, y * pixels_per_micron]);
    }

    JsonValue::from(vec![JsonValue::from(scaled_coords)])
}

// https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/inputs/segmentation-inputs
// https://github.com/dcjones/proseg/issues/120#issuecomment-3745057194
fn write_spaceranger_cell_polygon_geojson(
    geometries: Vec<JsonValue>,
    output_filename: String,
    pixels_per_micron: f32,
) {
    let features = JsonValue::from(
        geometries
            .iter()
            .map(|geom| {
                let mut feature = JsonValue::new_object();
                feature.insert("type", JsonValue::from("Feature")).unwrap();
                feature
                    .insert("id", JsonValue::from(geom["cell"].to_string()))
                    .unwrap();

                let mut properties = JsonValue::new_object();
                properties.insert("cell", geom["cell"].clone()).unwrap();
                feature.insert("properties", properties).unwrap();

                let mut geometry = JsonValue::new_object();
                geometry.insert("type", geom["type"].clone()).unwrap();
                geometry
                    .insert(
                        "coordinates",
                        scale_coords(&geom["coordinates"], pixels_per_micron),
                    )
                    .unwrap();

                feature.insert("geometry", geometry).unwrap();
                feature
            })
            .collect::<Vec<JsonValue>>(),
    );

    let mut data = JsonValue::new_object();
    data.insert("type", JsonValue::from("FeatureCollection"))
        .unwrap();
    data.insert("features", features).unwrap();

    let mut output = File::create(output_filename).expect("Unable to create output GeoJSON file.");
    output
        .write_all(data.dump().as_bytes())
        .expect("Unable to write output GeoJSON file.");
}
