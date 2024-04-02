use super::voxelsampler::Voxel;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Undirected;
use std::collections::{HashMap, HashSet};

// Using adjacency list representation for the subgraphs because they will typically be
// very small, so I expect this to be fast and easier to resize/reset without allocating.
type NeighborhoodGraph = Graph<(), (), Undirected, usize>;

#[derive(Copy, Clone, PartialEq)]
struct DfsInfo {
    parent: NodeIndex<usize>,
    depth: u32,
    low: u32,
}

impl Default for DfsInfo {
    fn default() -> Self {
        Self {
            parent: NodeIndex::end(),
            depth: 0,
            low: 0,
        }
    }
}

pub struct ConnectivityChecker {
    subgraph: NeighborhoodGraph,
    voxel_to_subgraph: HashMap<Voxel, NodeIndex<usize>>,
    visited: HashSet<NodeIndex<usize>>,
}

impl ConnectivityChecker {
    pub fn new() -> Self {
        let subgraph: NeighborhoodGraph = Graph::default();

        Self {
            subgraph,
            voxel_to_subgraph: HashMap::new(),
            visited: HashSet::new(),
        }
    }

    pub fn voxel_isarticulation<F>(&mut self, root: Voxel, voxel_cell: F, cell: u32) -> bool
    where
        F: Fn(Voxel) -> u32,
    {
        self.construct_voxel_subgraph(root, voxel_cell, cell);

        // dbg!(self.subgraph.node_count(), self.subgraph.edge_count(), cell);

        self.visited.clear();
        is_articulation_dfs(
            &self.subgraph,
            &mut self.visited,
            self.voxel_to_subgraph[&root],
        )
    }

    fn construct_voxel_subgraph<F>(&mut self, root: Voxel, voxel_cell: F, cell: u32)
    where
        F: Fn(Voxel) -> u32,
    {
        self.subgraph.clear();
        self.voxel_to_subgraph.clear();

        let i_idx = self.subgraph.add_node(());
        self.voxel_to_subgraph.insert(root, i_idx);

        for neighbor in root.moore_neighborhood() {
            // for neighbor in root.von_neumann_neighborhood() {
            let neighbor_cell = voxel_cell(neighbor);
            if cell == neighbor_cell {
                self.voxel_to_subgraph
                    .entry(neighbor)
                    .or_insert_with(|| self.subgraph.add_node(()));
            }
        }

        // add edges from i to neighbors, and neighbors' neighbors
        for neighbor in root.moore_neighborhood() {
            // for neighbor in root.von_neumann_neighborhood() {
            if !self.voxel_to_subgraph.contains_key(&neighbor) {
                continue;
            }

            self.subgraph.add_edge(
                self.voxel_to_subgraph[&root],
                self.voxel_to_subgraph[&neighbor],
                (),
            );

            for k in neighbor.moore_neighborhood() {
                // for k in neighbor.von_neumann_neighborhood() {
                let k_cell = voxel_cell(k);
                if self.voxel_to_subgraph.contains_key(&k) && k_cell == cell {
                    self.subgraph.add_edge(
                        self.voxel_to_subgraph[&neighbor],
                        self.voxel_to_subgraph[&k],
                        (),
                    );
                }
            }
        }
    }
}

// Recursive traversal function to find articulation points
fn is_articulation_dfs(
    subgraph: &NeighborhoodGraph,
    visited: &mut HashSet<NodeIndex<usize>>,
    i: NodeIndex<usize>,
) -> bool {
    visited.insert(i);
    let mut child_count = 0;

    // wikipedia version

    for j in subgraph.neighbors(i) {
        if visited.contains(&j) {
            continue;
        }

        child_count += 1;
        is_articulation_dfs(subgraph, visited, j);
    }

    child_count > 1
}
