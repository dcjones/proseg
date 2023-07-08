

use super::transcripts::{Transcript, coordinate_span};

use hexx::{Hex, HexLayout, HexOrientation, Vec2};
use std::collections::HashMap;


#[derive(Clone)]
struct HexBin {
    hex: Hex,
    transcripts: Vec<usize>,
}

impl HexBin {
    fn new(hex: Hex) -> Self {
        Self {
            hex,
            transcripts: Vec::new(),
        }
    }
}

struct ChunkedHexBin {
    chunk: u32,
    quad: u32,
    hexbin: HexBin,
}

fn chunk_hexbins(
    layout: &HexLayout,
    hexbins: &Vec<HexBin>,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize) -> Vec<ChunkedHexBin>
{
    return hexbins
        .iter()
        .map(|hexbin| {
            let hex_xy = layout.hex_to_world_pos(hexbin.hex);
            let (chunk, quad) =
                chunkquad(hex_xy.x, hex_xy.y, xmin, ymin, chunk_size, nxchunks);

            ChunkedHexBin {
                hexbin: hexbin.clone(),
                chunk,
                quad,
            }
        })
        .collect();
}

fn bin_transcripts(transcripts: &Vec<Transcript>, full_area: f32, avgpop: f32) -> Vec<ChunkedHexBin> {
    let target_area = full_area / avgpop;
    let hex_size = (target_area * 2.0 / (3.0 * (3.0 as f32).sqrt())).sqrt();

    let layout = HexLayout {
        orientation: HexOrientation::Flat,
        origin: Vec2::ZERO,
        hex_size: Vec2::new(hex_size, hex_size),
    };

    // Bin transcripts into HexBins
    let mut hex_index = HashMap::new();

    for (i, transcript) in transcripts.iter().enumerate() {
        let hex = layout.world_pos_to_hex(Vec2::new(transcript.x, transcript.y));

        hex_index.entry(hex)
            .or_insert_with(|| HexBin::new(hex))
            .transcripts.push(i);
    }

    // TODO: Now need to try to calibrate the chunk size as the avoid multiple
    // updates to the same cell.

    return chunk_hexbins(
        &layout,
        &hex_index.values().cloned().collect::<Vec<_>>(),
        0.0,
        0.0,
        100.0,
        10,
    )
}

struct HexBinSampler {
    chunkquads: Vec<Vec<ChunkQuad<Hex>>>,
    hexbins: Vec<ChunkedHexBin>,
}


// TODO: Ok, then after this, I think we have build a whole new sampler for HexBins
