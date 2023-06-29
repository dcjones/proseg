use csv;
use flate2::read::GzDecoder;
use petgraph::csr::Csr;
use petgraph::Directed;
use spade::{
    DelaunayTriangulation, HasPosition, LastUsedVertexHintGenerator, Point2, Triangulation,
};
use std::collections::HashMap;
use std::fs::File;

pub type NeighborhoodGraph = Csr<(), (), Directed, usize>;

#[derive(Copy, Clone, PartialEq)]
pub struct Transcript {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub gene: u32,
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
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>) {
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


fn postprocess_cell_assignments(cell_assignments: &mut Vec<u32>) -> Vec<usize> {
    let mut ncells = usize::MAX;
    for &cell_id in cell_assignments.iter() {
        if cell_id < u32::MAX {
            if ncells == usize::MAX || cell_id as usize >= ncells {
                ncells = (cell_id + 1) as usize;
            }
        }
    }

    if ncells == usize::MAX {
        ncells = 0;
    }

    // use `ncells` as a placeholder for unassigned/background transcripts
    for cell_id in cell_assignments.iter_mut() {
        if *cell_id == u32::MAX {
            *cell_id = ncells as u32;
        }
    }

    let mut cell_population = vec![0; ncells + 1];
    for &cell_id in cell_assignments.iter() {
        cell_population[cell_id as usize] += 1;
    }

    return cell_population;
}


fn read_transcripts_csv_xy<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>)
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let cell_id_col = find_column(headers, "cell_id");
    let overlaps_nucleus_col = find_column(headers, "overlaps_nucleus");

    let mut transcripts = Vec::new();
    let mut transcript_name_map = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut cell_assignments = Vec::new();

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

        let cell_id = row[cell_id_col].parse::<i32>().unwrap();
        let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();
        if cell_id >= 0 && overlaps_nucleus > 0 {
            cell_assignments.push(cell_id as u32);
        } else {
            cell_assignments.push(u32::MAX);
        }
    }

    let cell_population = postprocess_cell_assignments(&mut cell_assignments);

    return (transcript_names, transcripts, cell_assignments, cell_population);
}

fn read_transcripts_csv_xyz<T>(
    rdr: &mut csv::Reader<T>,
    transcript_column: &str,
    x_column: &str,
    y_column: &str,
    z_column: &str,
) -> (Vec<String>, Vec<Transcript>, Vec<u32>, Vec<usize>)
where
    T: std::io::Read,
{
    // Find the column we need
    let headers = rdr.headers().unwrap();
    let transcript_col = find_column(headers, transcript_column);
    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let z_col = find_column(headers, z_column);

    // TODO:
    // Just assuming we have xeinum output at this point.
    // We'll have to specialize for various platforms in the future.
    let cell_id_col = find_column(headers, "cell_id");
    let overlaps_nucleus_col = find_column(headers, "overlaps_nucleus");

    let mut transcripts = Vec::new();
    let mut transcript_name_map: HashMap<String, usize> = HashMap::new();
    let mut transcript_names = Vec::new();
    let mut cell_assignments = Vec::new();

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

        let cell_id = row[cell_id_col].parse::<i32>().unwrap();
        let overlaps_nucleus = row[overlaps_nucleus_col].parse::<i32>().unwrap();
        if cell_id >= 0 && overlaps_nucleus > 0 {
            cell_assignments.push(cell_id as u32);
        } else {
            cell_assignments.push(u32::MAX);
        }
    }

    let cell_population = postprocess_cell_assignments(&mut cell_assignments);

    return (transcript_names, transcripts, cell_assignments, cell_population);
}

pub fn read_nuclei_csv(path: &str, x_column: &str, y_column: &str) -> Vec<NucleiCentroid> {
    let mut rdr = csv::Reader::from_reader(GzDecoder::new(File::open(path).unwrap()));
    let headers = rdr.headers().unwrap();

    let x_col = find_column(headers, x_column);
    let y_col = find_column(headers, y_column);
    let mut centroids = Vec::new();

    for result in rdr.records() {
        let row = result.unwrap();

        let x = row[x_col].parse::<f32>().unwrap();
        let y = row[y_col].parse::<f32>().unwrap();

        centroids.push(NucleiCentroid { x, y });
    }

    return centroids;
}

// Vertex type for doing the triangulation in 2D
struct TranscriptPosIdx {
    x: f32,
    y: f32,
    idx: u32,
}

impl HasPosition for TranscriptPosIdx {
    type Scalar = f32;

    fn position(&self) -> Point2<Self::Scalar> {
        Point2::new(self.x, self.y)
    }
}

pub fn neighborhood_graph(
    transcripts: &Vec<Transcript>,
    max_edge_length: f32,
) -> (NeighborhoodGraph, f32) {
    let max_edge_length_squared = max_edge_length * max_edge_length;

    let vertices = transcripts
        .iter()
        .enumerate()
        .map(|(i, t)| TranscriptPosIdx {
            x: t.x,
            y: t.y,
            idx: i as u32,
        })
        .collect();

    let triangulation: DelaunayTriangulation<
        TranscriptPosIdx,
        (),
        (),
        (),
        LastUsedVertexHintGenerator,
    > = DelaunayTriangulation::bulk_load(vertices).unwrap();

    let mut nrejected: usize = 0;
    let mut nedges: usize = 0;
    let mut edges = Vec::new();
    let mut avg_edge_length = 0.0;
    for edge in triangulation.directed_edges() {
        if edge.length_2() >= max_edge_length_squared {
            nrejected += 1;
            continue;
        }
        avg_edge_length += edge.length_2();

        let [from, to] = edge.vertices();
        edges.push((from.data().idx as usize, to.data().idx as usize));
        nedges += 1;
    }
    avg_edge_length = (avg_edge_length / nedges as f32).sqrt();
    assert!(avg_edge_length > 0.0);

    edges.sort();

    println!(
        "Rejected {} edges ({:0.3}%)",
        nrejected,
        nrejected as f64 / nedges as f64 * 100.0
    );

    return (NeighborhoodGraph::from_sorted_edges(&edges).unwrap(), avg_edge_length);
}

pub fn coordinate_span(transcripts: &Vec<Transcript>) -> (f32, f32, f32, f32) {
    let mut min_x = std::f32::MAX;
    let mut max_x = std::f32::MIN;
    let mut min_y = std::f32::MAX;
    let mut max_y = std::f32::MIN;

    for t in transcripts {
        min_x = min_x.min(t.x);
        max_x = max_x.max(t.x);
        min_y = min_y.min(t.y);
        max_y = max_y.max(t.y);
    }

    return (min_x, max_x, min_y, max_y);
}
