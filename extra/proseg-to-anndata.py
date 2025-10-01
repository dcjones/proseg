#!/usr/bin/env python3

import argparse, os
import numpy as np
import pandas as pd
import anndata
from scipy.io import mmread
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Path to output h5ad file")
    ap.add_argument(
        "--count-matrix", required=True, help="Path to count matrix input file"
    )
    ap.add_argument(
        "--cell-metadata", required=True, help="Path to cell metadata input file"
    )
    ap.add_argument(
        "--gene-metadata", required=False, help="Path to gene meta input file"
    )
    args = vars(ap.parse_args())

    count_path = args["count_matrix"]
    if count_path.endswith(".parquet"):
        count_matrix = pd.read_parquet(count_path)
    elif count_path.endswith(".csv") or count_path.endswith(".csv.gz"):
        count_matrix = pd.read_csv(count_path, index_col=0)
    elif count_path.endswith(".mtx") or count_path.endswith(".mtx.gz"):
        count_matrix = mmread(count_path).tocsr().astype(np.int32)
    else:
        raise ValueError(f"Unsupported count matrix extension: {count_path}")

    cell_metadata_path = args["cell_metadata"]
    if cell_metadata_path.endswith(".parquet"):
        cell_metadata = pd.read_parquet(cell_metadata_path)
    elif cell_metadata_path.endswith(".csv") or cell_metadata_path.endswith(".csv.gz"):
        cell_metadata = pd.read_csv(cell_metadata_path)
    else:
        raise ValueError(f"Unsupported cell metadata extension: {cell_metadata_path}")

    gene_metadata_path = args["gene_metadata"]
    if gene_metadata_path is None:
        gene_metadata = None
    elif gene_metadata_path.endswith(".parquet"):
        gene_metadata = pd.read_parquet(gene_metadata_path)
    elif gene_metadata_path.endswith(".csv") or gene_metadata_path.endswith(".csv.gz"):
        gene_metadata = pd.read_csv(gene_metadata_path)
    else:
        raise ValueError(f"Unsupported gene metadata extension: {gene_metadata_path}")

    obsm = {
        "spatial": np.stack(
            [cell_metadata.centroid_x.to_numpy(), cell_metadata.centroid_y.to_numpy()],
            axis=1,
        )
    }

    if gene_metadata is not None:
        # MM rows=genes, cols=cells â†’ AnnData wants cells x genes
        gene_metadata = gene_metadata.set_index("gene")
        var = pd.DataFrame(index=gene_metadata.index)
    elif isinstance(count_matrix, pd.DataFrame):
        # CSV / Parquet assumed as cells x genes
        var = pd.DataFrame(index=count_matrix.columns)
    else:
        raise Exception(
            "Gene metadata must be provided when input matrix is in .mtx format."
        )

    if isinstance(count_matrix, pd.DataFrame):
        count_matrix = csr_matrix(count_matrix.to_numpy())

    obs = cell_metadata.set_index(pd.Index(cell_metadata.cell, str, True, "cell_d"))
    adata = anndata.AnnData(X=count_matrix, obs=obs, var=var, obsm=obsm)

    adata.write_h5ad(args["output"])
