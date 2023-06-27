
use std::collections::HashMap;
use petgraph::{Directed, Undirected};
use petgraph::csr::Csr;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::IntoNeighbors;

type NeighborhoodGraphCsr = Csr<(), (), Directed, usize>;

// Using adjacency list representation for the subgraphs because they will typically be
// very small, so I expect this to be fast and easier to resize/reset without allocating.
type NeighborhoodGraph = Graph<(), (), Undirected, usize>;


struct DfsInfo {
    parent: NodeIndex<usize>,
    depth: u32,
    low: u32,
    is_articulation: bool,
}

impl Default for DfsInfo {
    fn default() -> Self {
        return Self {
            parent: NodeIndex::end(),
            depth: 0,
            low: 0,
            is_articulation: false,
        };
    }
}

pub struct ConnectivityCheck {
    subgraph: NeighborhoodGraph,
    graph_to_subgraph: HashMap<usize, NodeIndex<usize>>,
    dfsinfo: HashMap<NodeIndex<usize>, DfsInfo>,
}

impl ConnectivityCheck {
    pub fn new() -> Self {
        let subgraph: NeighborhoodGraph = Graph::default();

        return Self {
            subgraph: subgraph,
            graph_to_subgraph: HashMap::new(),
            dfsinfo: HashMap::new(),
        };
    }

    // Test if `i` is an articulation point on the subgraph of `graph`
    // consisting of `i` and any neighbors of `i` which are of cell `cell`.
    pub fn isarticulation(
        &mut self, graph: &NeighborhoodGraphCsr,
        cell_assignments: &Vec<u32>, i: usize, cell: u32) -> bool
    {
        self.costruct_subgraph(graph, cell_assignments, i, cell);
        self.dfsinfo.clear();
        return is_articulation_dfs(
            &self.subgraph,
            &mut self.dfsinfo,
            self.graph_to_subgraph[&i],
            NodeIndex::end(), 0);
    }

    // Construct a subgraph of `graph` containing i and any neighbors of `i`
    // that are of cell `cell`.
    fn costruct_subgraph(
        &mut self, graph: &NeighborhoodGraphCsr,
        cell_assignments: &Vec<u32>, i: usize, cell: u32)
    {
        self.subgraph.clear();
        self.graph_to_subgraph.clear();

        // insert nodes and build a map from graph indices to subgraph indices
        let i_idx = self.subgraph.add_node(());
        self.graph_to_subgraph.insert(i, i_idx);

        for j in graph.neighbors(i) {
            if cell_assignments[j] == cell {
                self.graph_to_subgraph.entry(j).or_insert_with(|| {
                    let j_idx = self.subgraph.add_node(());
                    j_idx
                });
            }
        }

        // add edges from i to neighbors, and neighbors' neighbors
        for j in graph.neighbors(i) {
            if self.graph_to_subgraph.contains_key(&j) {
                self.subgraph.add_edge(
                    self.graph_to_subgraph[&i],
                    self.graph_to_subgraph[&j], ());
            }

            for k in graph.neighbors(j) {
                if self.graph_to_subgraph.contains_key(&k) {
                    self.subgraph.add_edge(
                        self.graph_to_subgraph[&i],
                        self.graph_to_subgraph[&k], ());
                }
            }
        }
    }
}

// Recursive traversal function to find articulation points
fn is_articulation_dfs(
    subgraph: &NeighborhoodGraph,
    dfsinfo: &mut HashMap<NodeIndex<usize>, DfsInfo>,
    i: NodeIndex<usize>,
    parent: NodeIndex<usize>,
    depth: u32) -> bool
{
    let mut i_info = dfsinfo.insert(i, DfsInfo::default()).unwrap();
    i_info.parent = parent;
    i_info.depth = depth;
    i_info.low = depth;

    let mut child_count = 0;

    for j in subgraph.neighbors(i) {
        if let Some(j_info) = dfsinfo.get(&j) {
            // aready visited j
            i_info.low = i_info.low.min(j_info.depth);
        } else {
            // j is unvisited
            is_articulation_dfs(subgraph, dfsinfo, j, i, depth + 1);
            let j_info = &dfsinfo[&j];
            child_count += 1;

            if j_info.low >= i_info.depth {
                i_info.is_articulation = true;
            }
            i_info.low = i_info.low.min(j_info.low);
        }

        if i_info.parent == NodeIndex::end() && child_count > 1 {
            i_info.is_articulation = true;
        }
    }

    return i_info.is_articulation;
}
