use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use rand::Rng;
use rayon::prelude::*;
use std::collections::HashMap;

use super::transcripts::TranscriptDataset;

const NFEATURES: usize = 10_000;
const BIN_SIZE: f32 = 20.0;
const NGENES_CANDIDATES: usize = 5000;
// Subsample rows for the correlation step only; deviance ranking uses all sampled bins.
const NCORR_ROWS: usize = 2000;
// Minimum mean transcripts per region; genes below this are excluded before clustering
// to avoid selecting essentially-unexpressed genes whose correlation structure is pure noise.
const MIN_MEAN_TRANSCRIPTS_PER_REGION: f32 = 5e-2;

// Select `nregions` random 2D square regions, and generate a count matrix over
// these regions. This gives us some basis for doing feature selection without relying
fn random_region_counts(
    dataset: &TranscriptDataset,
    nregions: usize,
    bin_size: f32,
) -> Array2<f32> {
    let mut rng = rand::rng();
    let mut region_map = HashMap::new();

    for _ in 0..nregions {
        let centroid_transcript =
            &dataset.transcripts.runs[rng.random_range(0..dataset.transcripts.runs.len())].value;

        let cx = centroid_transcript.x;
        let cy = centroid_transcript.y;

        let bin_x = (cx / bin_size).floor() as usize;
        let bin_y = (cy / bin_size).floor() as usize;
        let next_region = region_map.len();
        region_map.entry((bin_x, bin_y)).or_insert(next_region);
    }

    let mut counts = Array2::zeros((region_map.len(), dataset.ngenes()));

    for transcript_run in dataset.transcripts.iter_runs() {
        let bin_x = (transcript_run.value.x / bin_size).floor() as usize;
        let bin_y = (transcript_run.value.y / bin_size).floor() as usize;
        let region = region_map.get(&(bin_x, bin_y));
        if let Some(region) = region {
            counts[[*region, transcript_run.value.gene as usize]] += transcript_run.len as f32;
        }
    }

    counts
}

fn deviance_ranking(counts: &Array2<f32>) -> Vec<usize> {
    let nregions = counts.nrows();
    let ngenes = counts.ncols();

    // Total counts per region (row sums)
    let region_totals: Vec<f64> = (0..nregions)
        .into_par_iter()
        .map(|i| counts.row(i).iter().map(|&x| x as f64).sum())
        .collect();

    // Total counts per gene (column sums)
    let gene_totals: Vec<f64> = (0..ngenes)
        .into_par_iter()
        .map(|j| counts.column(j).iter().map(|&x| x as f64).sum())
        .collect();

    // Grand total
    let grand_total: f64 = gene_totals.iter().sum();

    // Transpose so each gene's counts are contiguous in memory (row-major access
    // in the inner loop avoids cache-miss penalties from column-strided access).
    let counts_t = counts.t().to_owned();

    // Compute binomial deviance for each gene.
    //
    // Under the binomial model proposed by Townes et al., the saturated
    // deviance for gene j is:
    //
    //   D_j = 2 * sum_i [ y_ij * log(y_ij / mu_ij)
    //                    + (n_i - y_ij) * log((n_i - y_ij) / (n_i - mu_ij)) ]
    //
    // where y_ij is the count for region i and gene j,
    //       n_i  is the total count for region i,
    //       mu_ij = n_i * p_j  is the expected count under H0,
    //   and p_j = (sum_i y_ij) / grand_total is the MLE for gene j's rate.
    //
    // Terms where the observed count is 0 contribute 0 to the sum (limit of
    // x*log(x) as x->0 is 0).
    let mut deviances: Vec<(usize, f64)> = (0..ngenes)
        .into_par_iter()
        .map(|j| {
            let p_j = if grand_total > 0.0 {
                gene_totals[j] / grand_total
            } else {
                0.0
            };

            let deviance: f64 = (0..nregions)
                .map(|i| {
                    let y = counts_t[[j, i]] as f64;
                    let n = region_totals[i];
                    if n == 0.0 {
                        return 0.0;
                    }
                    let mu = n * p_j;

                    // Positive term: y * log(y / mu)
                    let pos = if y > 0.0 && mu > 0.0 {
                        y * (y / mu).ln()
                    } else {
                        0.0
                    };

                    // Negative term: (n - y) * log((n - y) / (n - mu))
                    let neg = if (n - y) > 0.0 && (n - mu) > 0.0 {
                        (n - y) * ((n - y) / (n - mu)).ln()
                    } else {
                        0.0
                    };

                    pos + neg
                })
                .sum::<f64>()
                * 2.0;

            (j, deviance)
        })
        .collect();

    // Sort genes from highest to lowest deviance
    deviances.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    deviances.into_iter().map(|(j, _)| j).collect()
}

fn select_k_clusters<T: PartialOrd + Copy>(
    hclust: &kodama::Dendrogram<T>,
    nclusters: usize,
) -> Vec<usize> {
    let n = hclust.observations();
    let steps = hclust.steps();

    // Binary search over the dissimilarity threshold to find one that yields
    // exactly `nclusters` clusters when we cut all links whose dissimilarity
    // exceeds it.
    //
    // The dendrogram merges observations from 0..n and intermediate cluster
    // labels from n..2n-1. After cutting, we do a union-find pass to assign
    // each of the original n observations to a root cluster.

    // Collect all unique dissimilarity values from the dendrogram steps.
    let mut thresholds: Vec<T> = steps.iter().map(|s| s.dissimilarity).collect();
    thresholds.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    thresholds.dedup_by(|a, b| a == b);

    // Count clusters produced by a given threshold: merge only those steps
    // whose dissimilarity is <= threshold.
    let count_clusters = |threshold: T| -> usize {
        // Union-Find over 2n-1 nodes (n leaves + n-1 internal nodes).
        let total = 2 * n - 1;
        let mut parent: Vec<usize> = (0..total).collect();

        fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }

        for (k, step) in steps.iter().enumerate() {
            if step.dissimilarity <= threshold {
                let internal = n + k;
                let ra = find(&mut parent, step.cluster1);
                let rb = find(&mut parent, step.cluster2);
                let ri = find(&mut parent, internal);
                parent[ra] = ri;
                parent[rb] = ri;
            }
        }

        // Count distinct roots among the n original observations.
        let mut roots = std::collections::HashSet::new();
        for i in 0..n {
            roots.insert(find(&mut parent, i));
        }
        roots.len()
    };

    // Binary search for the smallest threshold index where count_clusters <= nclusters.
    // At threshold index 0 (smallest dissimilarity), few merges occur → many clusters.
    // At the last threshold, all observations merge → 1 cluster. The cluster count is
    // monotonically non-increasing in threshold, so binary search is valid.
    let mut lo = 0usize;
    let mut hi = thresholds.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if count_clusters(thresholds[mid]) <= nclusters {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let chosen_threshold = if lo < thresholds.len() {
        thresholds[lo]
    } else {
        thresholds[thresholds.len() - 1]
    };

    // Assign cluster labels using the chosen threshold.
    let total = 2 * n - 1;
    let mut parent: Vec<usize> = (0..total).collect();

    fn find(parent: &mut Vec<usize>, mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    for (k, step) in steps.iter().enumerate() {
        if step.dissimilarity <= chosen_threshold {
            let internal = n + k;
            let ra = find(&mut parent, step.cluster1);
            let rb = find(&mut parent, step.cluster2);
            let ri = find(&mut parent, internal);
            parent[ra] = ri;
            parent[rb] = ri;
        }
    }

    // Map each original observation to a contiguous cluster index 0..nclusters.
    let mut root_to_cluster: HashMap<usize, usize> = HashMap::new();
    let mut labels = vec![0usize; n];
    for i in 0..n {
        let root = find(&mut parent, i);
        let next_id = root_to_cluster.len();
        let cluster_id = *root_to_cluster.entry(root).or_insert(next_id);
        labels[i] = cluster_id;
    }

    labels
}

fn select_features(dataset: &TranscriptDataset, nfeatures: usize) -> Option<Vec<usize>> {
    let counts = random_region_counts(dataset, NFEATURES, BIN_SIZE);
    let gene_ranking = deviance_ranking(&counts);

    // To avoid an overly large pairwise distance matrix, subset to the top
    // NGENES_CANDIDATES genes by deviance. col_indices[j] gives the original
    // gene index for local column j, and columns are ordered highest-deviance
    // first throughout. We always build via col_indices so the back-mapping is
    // uniform whether or not we actually truncate.
    let ncandidates = gene_ranking.len().min(NGENES_CANDIDATES);
    let col_indices: Vec<usize> = gene_ranking.into_iter().take(ncandidates).collect();
    let nrows = counts.nrows();
    let counts =
        Array2::from_shape_fn((nrows, ncandidates), |(i, j)| counts[[i, col_indices[j]]]);

    // Filter out zero-variance columns: genes absent from all random regions
    // produce a constant (all-zero) column, which causes pearson_correlation to
    // return NaN, which kodama rejects.
    let n = nrows as f32;
    let valid_cols: Vec<usize> = (0..ncandidates)
        .filter(|&j| {
            let col = counts.column(j);
            let mean = col.sum() / n;
            col.iter().any(|&v| (v - mean).abs() > 1e-9)
        })
        .collect();

    let gene_totals: Vec<f32> = valid_cols.iter().map(|&j| counts.column(j).sum()).collect();

    let col_indices: Vec<usize> = valid_cols.iter().map(|&j| col_indices[j]).collect();
    let counts =
        Array2::from_shape_fn((nrows, valid_cols.len()), |(i, j)| counts[[i, valid_cols[j]]]);

    // Filter out low-expression genes before clustering to avoid selecting genes
    // whose correlation structure is dominated by noise.
    let nrows_f = nrows as f32;
    let expr_indices: Vec<usize> = (0..col_indices.len())
        .filter(|&j| gene_totals[j] / nrows_f >= MIN_MEAN_TRANSCRIPTS_PER_REGION)
        .collect();
    let col_indices: Vec<usize> = expr_indices.iter().map(|&j| col_indices[j]).collect();
    let gene_totals: Vec<f32> = expr_indices.iter().map(|&j| gene_totals[j]).collect();
    let mut counts =
        Array2::from_shape_fn((nrows, expr_indices.len()), |(i, j)| counts[[i, expr_indices[j]]]);

    if col_indices.len() < 2 {
        return None;
    }

    // log1p transform counts
    counts.map_inplace(|v| *v = v.ln_1p());
    let ngenes = col_indices.len();

    // Subsample rows for the correlation step. Deviance ranking benefits from all
    // NFEATURES regions, but the gene-gene correlation estimate converges quickly;
    // NCORR_ROWS samples gives a good approximation at a fraction of the cost.
    let corr_counts = if nrows > NCORR_ROWS {
        let mut rng = rand::rng();
        let row_indices: Vec<usize> = rand::seq::index::sample(&mut rng, nrows, NCORR_ROWS).into_vec();
        Array2::from_shape_fn((NCORR_ROWS, ngenes), |(i, j)| counts[[row_indices[i], j]])
    } else {
        counts
    };

    let corr = corr_counts.t().pearson_correlation().unwrap();

    let mut condensed_dissim = Vec::with_capacity(ngenes * (ngenes - 1) / 2);
    for i in 0..ngenes {
        for j in i + 1..ngenes {
            let c = corr[[i, j]];
            condensed_dissim.push(if c.is_finite() {
                (1.0 - c).clamp(0.0, 2.0)
            } else {
                1.0
            });
        }
    }

    let hclust = kodama::linkage(&mut condensed_dissim, ngenes, kodama::Method::Average);

    let gene_cluster_assignments = select_k_clusters(&hclust, nfeatures);

    // For each cluster, select the representative gene with the highest total
    // transcript count so that highly-expressed genes are kept out of the
    // factored portion.
    let nclusters = gene_cluster_assignments
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    let mut best_local: Vec<Option<usize>> = vec![None; nclusters];
    for (local_idx, &cluster_id) in gene_cluster_assignments.iter().enumerate() {
        let current = best_local[cluster_id];
        if current.is_none() || gene_totals[local_idx] > gene_totals[current.unwrap()] {
            best_local[cluster_id] = Some(local_idx);
        }
    }

    // Map local indices back to original gene indices and return.
    Some(
        best_local
            .into_iter()
            .filter_map(|opt| opt.map(|local_idx| col_indices[local_idx]))
            .collect(),
    )
}

impl TranscriptDataset {
    pub fn select_unfactored_genes(&mut self, nunfactored: usize) -> bool {
        let Some(selected_features) = select_features(self, nunfactored) else {
            return false;
        };

        // Build an ordering that puts selected_features genes first (in the
        // order they appear in selected_features), followed by the remaining
        // genes in their original order.
        let selected_set: std::collections::HashSet<usize> =
            selected_features.iter().copied().collect();
        let mut ord: Vec<usize> = selected_features.clone();
        for gene_idx in 0..self.gene_names.len() {
            if !selected_set.contains(&gene_idx) {
                ord.push(gene_idx);
            }
        }

        let mut rev_ord = vec![0; ord.len()];
        for (i, j) in ord.iter().enumerate() {
            rev_ord[*j] = i;
        }

        self.gene_names = ord.iter().map(|&i| self.gene_names[i].clone()).collect();
        for transcript_run in self.transcripts.iter_runs_mut() {
            transcript_run.value.gene = rev_ord[transcript_run.value.gene as usize] as u32;
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::runvec::RunVec;
    use crate::sampler::transcripts::{PriorTranscriptSeg, Transcript};

    fn same_bin_dataset() -> TranscriptDataset {
        let mut transcripts = RunVec::new();
        transcripts.push_run(
            Transcript {
                x: 1.0,
                y: 1.0,
                z: 0.0,
                qv: 1.0,
                gene: 0,
            },
            2,
        );
        transcripts.push_run(
            Transcript {
                x: 2.0,
                y: 2.0,
                z: 0.0,
                qv: 1.0,
                gene: 1,
            },
            3,
        );
        TranscriptDataset {
            transcripts,
            transcript_ids: None,
            priorseg: RunVec::<u32, PriorTranscriptSeg>::new(),
            fovs: RunVec::new(),
            barcode_positions: None,
            gene_names: vec!["A".to_string(), "B".to_string()],
            fov_names: Vec::new(),
            original_cell_ids: Vec::new(),
            ncells: 0,
        }
    }

    #[test]
    fn random_region_counts_collapses_duplicate_bins() {
        let dataset = same_bin_dataset();

        let counts = random_region_counts(&dataset, 100, BIN_SIZE);

        assert_eq!(counts.shape(), &[1, 2]);
        assert_eq!(counts[[0, 0]], 2.0);
        assert_eq!(counts[[0, 1]], 3.0);
    }

    #[test]
    fn feature_selection_fails_cleanly_with_fewer_than_two_filtered_genes() {
        let mut dataset = same_bin_dataset();
        let original_gene_names = dataset.gene_names.clone();

        assert!(!dataset.select_unfactored_genes(1));
        assert_eq!(dataset.gene_names, original_gene_names);
    }
}
