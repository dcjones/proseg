use csv;
use flate2::read::GzDecoder;
use spade::{Point2, DelaunayTriangulation, Triangulation, HasPosition, LastUsedVertexHintGenerator};
use std::collections::HashMap;
use std::fs::File;
use sprs::{TriMat, CsMat};

pub struct Transcript {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: u32
}


pub struct NucleiCentroid {
    pub x: f32,
    pub y: f32,
}

pub fn read_transcripts_csv(
    path: &str,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
    z_column: Option<&str>,
) -> (Vec<String>, Vec<Transcript>) {
    let mut rdr = csv::Reader::from_reader(GzDecoder::new(File::open(path).unwrap()));
    // let mut rdr = csv::Reader::from_path(path).unwrap();

    match z_column {
        Some(z_column) => {
            return read_transcripts_csv_xyz(
                &mut rdr,
                transcript_column,
                x_column,
                y_column,
                z_column,
            );
        }
        None => {
            return read_transcripts_csv_xy(&mut rdr, transcript_column, x_column, y_column);
        }
    }
}

fn find_column(headers: &csv::StringRecord, column: &str) -> usize {
    let col = headers.iter().position(|x| x == column);
    match col {
        Some(col) => col,
        None => panic!("Column '{}' not found in CSV file", column),
    }
}

fn read_transcripts_csv_xy<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
) -> (Vec<String>, Vec<Transcript>)
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);

    let mut transcripts = Vec::new();
    let mut transcript_name_map = HashMap::new();
    let mut transcript_names = Vec::new();

    for result in rdr.records() {
        let row = result.unwrap();

        let transcript_name = &row[transcript_col];

        let gene = if let Some(gene) = transcript_name_map.get(transcript_name) {
            *gene
        } else {
            transcript_names.push(transcript_name.to_string());
            transcript_name_map.insert(transcript_name.to_string(), transcript_names.len() - 1);
            transcript_names.len() - 1
        };

        let x = row[x_col].parse::<f32>().unwrap();
        let y = row[y_col].parse::<f32>().unwrap();

        transcripts.push(Transcript {
            x,
            y,
            z: 0.0,
            gene: gene as u32,
        });
    }

    return (transcript_names, transcripts);
}

fn read_transcripts_csv_xyz<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
    z_column: &str,
) -> (Vec<String>, Vec<Transcript>)
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let z_col = find_column(headers, z_column);

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();

    for result in rdr.records() {
        let row = result.unwrap();

        let transcript_name = &row[transcript_col];

        let gene = if let Some(gene) = transcript_name_map.get(transcript_name) {
            *gene
        } else {
            transcript_names.push(transcript_name.to_string());
            transcript_name_map.insert(transcript_name.to_string(), transcript_names.len() - 1);
            transcript_names.len() - 1
        };

        let x = row[x_col].parse::<f32>().unwrap();
        let y = row[y_col].parse::<f32>().unwrap();
        let z = row[z_col].parse::<f32>().unwrap();

        transcripts.push(Transcript {
            x,
            y,
            z,
            gene: gene as u32,
        });
    }

    return (transcript_names, transcripts);
}

pub fn read_nuclei_csv(
    path: &str,
    x_column: &str,
    y_column: &str,
) -> Vec<NucleiCentroid>
{
    let mut rdr = csv::Reader::from_reader(GzDecoder::new(File::open(path).unwrap()));
    let headers = rdr.headers().unwrap();

    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let mut centroids = Vec::new();

    for result in rdr.records() {
        let row = result.unwrap();

        let x = row[x_col].parse::<f32>().unwrap();
        let y = row[y_col].parse::<f32>().unwrap();

        centroids.push(NucleiCentroid {
            x,
            y,
        });
    }

    return centroids;
}


// Vertex type for doing the triangulation in 2D
struct TranscriptPosIdx {
    x: f32,
    y: f32,
    idx: u32
}

impl HasPosition for TranscriptPosIdx {
    type Scalar = f32;

    fn position(&self) -> Point2<Self::Scalar> {
        Point2::new(self.x, self.y)
    }
}


pub fn neighborhood_graph(transcripts: &Vec<Transcript>) -> CsMat<f32> {
    let vertices =
        transcripts.iter().enumerate().map(
            |(i, t)| TranscriptPosIdx { x: t.x, y: t.y, idx: i as u32 }).collect();

    let triangulation: DelaunayTriangulation<TranscriptPosIdx, (), (), (), LastUsedVertexHintGenerator> =
        DelaunayTriangulation::bulk_load(vertices).unwrap();

    let n = transcripts.len();
    let mut adjacency = TriMat::new((n, n));

    for edge in triangulation.undirected_edges() {
        let [from, to] = edge.vertices();
        adjacency.add_triplet(from.data().idx as usize,  to.data().idx as usize, 1.0f32);
    }

    let adjacency = adjacency.to_csr();
    // return &adjacency + &adjacency.transpose_view();
    return adjacency;
}
