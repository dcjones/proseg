use super::sampleset::SampleSet;

#[derive(Clone)]
pub struct ChunkQuad<T> {
    pub chunk: u32,
    pub quad: u32,
    pub mismatch_edges: SampleSet<(T, T)>,
}

// Compute chunk and quadrant for a single a single (x,y) point.
pub fn chunkquad(
    x: f32,
    y: f32,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,
) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size / 2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size / 2.0)).floor() as u32;

    let chunk = (xchunkquad / 2) + (ychunkquad / 2) * (nxchunks as u32);
    let quad = (xchunkquad % 2) + (ychunkquad % 2) * 2;

    return (chunk, quad);
}
