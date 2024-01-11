
use std::collections::HashSet;
use super::cubebinsampler::{Cube, CubeLayout};


use geo::geometry::{LineString, MultiPolygon, Polygon};
use geo::algorithm::simplify::Simplify;
use petgraph::adj;


// TODO: Plan for polygon generation:
//   - For each cell (in parallel)
//   -   for each voxel in the cells
//   -     construct a vec of all mismatching edges (end coordinates)
//   -     traverse these edges to get outlines
//   -     these outlines are the polygons
//
// Edge simplification:
//   To reduce the jagged edges: 

pub struct PolygonBuilder {
    edges: Vec<(i32, (i32, i32), (i32, i32))>,
    visited: Vec<bool>,
}


fn reverse_edge(edge: &(i32, (i32, i32), (i32, i32))) -> (i32, (i32, i32), (i32, i32)) {
    return (edge.0, edge.2, edge.1);
}


// return the lexigraphically next point
fn next_point(v: (i32, i32)) -> (i32, i32) {
    return (v.0, v.1+1);
}


fn mark_visited(edges_k: &[(i32, (i32, i32), (i32, i32))], visited_k: &mut [bool], edge: &(i32, (i32, i32), (i32, i32))) {
    let pos = edges_k.binary_search(edge).unwrap();
    assert!(!visited_k[pos]);
    visited_k[pos] = true;

    let pos = edges_k.binary_search(&reverse_edge(edge)).unwrap();
    assert!(!visited_k[pos]);
    visited_k[pos] = true;
}


impl PolygonBuilder {
    pub fn new() -> Self {
        return PolygonBuilder {
            edges: Vec::new(),
            visited: Vec::new(),
        };
    }

    pub fn cell_voxels_to_polygons(&mut self, layout: &CubeLayout, voxels: &HashSet<Cube>) -> Vec<(i32, MultiPolygon<f32>)> {
        // if we store edges in in μm, we run the risk of failing line up points
        // due to numerical imprecision. Instead we use integer coordinates,
        // where (i, j, k) represents the corner of voxel (i, j, k)

        // construct a data structure containing every cell edge (encoded in voxel coordinates)
        self.edges.clear();
        let mut kmin = i32::MAX;
        let mut kmax = i32::MIN;
        for voxel in voxels {
            kmin = kmin.min(voxel.k);
            kmax = kmax.max(voxel.k);

            for neighbor in voxel.von_neumann_neighborhood_xy() {
                if !voxels.contains(&neighbor) {
                    let edge = voxel.edge_xy(&neighbor);
                    self.edges.push((voxel.k, edge.0, edge.1));
                    self.edges.push((voxel.k, edge.1, edge.0));
                }
            }
        }
        self.edges.sort_unstable();

        // traverse the cell edges to construct polygons
        self.visited.fill(false);
        self.visited.resize(self.edges.len(), false);

        let mut multipolygons = Vec::new();

        for k in kmin..kmax+1 {
            let first = self.edges.partition_point(|edge| edge.0 < k);
            let last = self.edges.partition_point(|edge| edge.0 < k+1);

            let edges_k = &self.edges[first..last];
            let visited_k = &mut self.visited[first..last];

            let nedges = edges_k.len();
            let mut nvisited = 0;

            let mut polygons_k = Vec::new();

            while nvisited < nedges {
                let mut polygon = Vec::new();

                let pos = visited_k.iter().position(|v| !v);

                if let Some(pos) = pos {
                    let edge = edges_k[pos];
                    mark_visited(edges_k, visited_k, &edge);
                    nvisited += 2;

                    polygon.push(edge.1);
                    polygon.push(edge.2);

                    while polygon.first() != polygon.last() && nvisited < nedges {
                        let u = polygon.last().unwrap();

                        let δi = u.0 - polygon[polygon.len()-2].0;
                        let δj = u.1 - polygon[polygon.len()-2].1;
                        assert!(δi.abs() + δj.abs() == 1);

                        let first = edges_k.partition_point(|edge| edge.1 < *u);
                        let last = edges_k.partition_point(|edge| edge.1 < next_point(*u));
                        let adjacent_edges = &edges_k[first..last];
                        let adjacent_edges_visited = &mut visited_k[first..last];

                        // we have either an unambiguous path or we are at the corner of two voxels
                        assert!(adjacent_edges.len() == 1 || adjacent_edges.len() == 3);

                        if adjacent_edges.len() == 1 {
                            let edge = adjacent_edges[0];
                            assert!(!adjacent_edges_visited[0]);
                            mark_visited(edges_k, visited_k, &edge);
                            nvisited += 2;
                            polygon.push(edge.2);
                        } else {
                            let v;

                            // Might be a nicer way, but I'm just going to handle
                            // each case exhaustively here.
                            if voxels.contains(&Cube::new(k, u.0, u.1)) {
                                if δi == -1 {
                                    v = (u.0, u.1-1);
                                } else if δi == 1 {
                                    v = (u.0, u.1+1);
                                } else if δj == -1 {
                                    v = (u.0-1, u.1);
                                } else if δj == 1 {
                                    v = (u.0+1, u.1);
                                } else {
                                    unreachable!();
                                }
                            } else {
                                if δi == -1 {
                                    v = (u.0, u.1+1);
                                } else if δi == 1 {
                                    v = (u.0, u.1-1);
                                } else if δj == -1 {
                                    v = (u.0+1, u.1);
                                } else if δj == 1 {
                                    v = (u.0-1, u.1);
                                } else {
                                    unreachable!();
                                }
                            }
                            let edge = (k, *u, v);
                            assert!(adjacent_edges.contains(&edge));
                            assert!(!adjacent_edges_visited[adjacent_edges.iter().position(|e| *e == edge).unwrap()]);

                            mark_visited(edges_k, visited_k, &edge);
                            nvisited += 2;
                            polygon.push(v);
                        }
                    }

                    assert!(polygon.first() == polygon.last());

                    // TODO: run anti-aliaising on the polygon
                    // TODO: run line simplification on the polygon

                    // convert coordinates to μm
                    let polygon: Vec<(f32, f32)> = polygon
                        .iter()
                        .map(|v| {
                            let (x, y, _z) = layout.cube_corner_to_world_pos(Cube::new(v.0, v.1, 0));
                            return (x, y);
                        })
                        .collect();

                    polygons_k.push(Polygon::<f32>::new(
                        LineString::from(polygon),
                        Vec::new()));
                } else {
                    // no unvisited edges left
                    break;
                }
            }

            multipolygons.push((k, MultiPolygon::new(polygons_k)));
        }

        return multipolygons;
    }
}
