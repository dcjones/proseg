#!/usr/bin/env python3

import pandas as pd
import anndata
import numpy as np
import argparse
import os
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("output", help="Path to output h5ad file")
    ap.add_argument("--count-matrix", required=True, help="Path to count matrix input file")
    ap.add_argument("--cell-metadata", required=True, help="Path to cell metadata input file")
    args = vars(ap.parse_args())

    count_matrix_ext = os.path.splitext(args["count_matrix"])[1]
    if count_matrix_ext == ".parquet":
        count_matrix = pd.read_parquet(args["count_matrix"])
    else:
        count_matrix = pd.read_csv(args["count_matrix"])

    cell_metadata_ext = os.path.splitext(args["cell_metadata"])[1]
    if cell_metadata_ext == ".parquet":
        cell_metadata = pd.read_parquet(args["cell_metadata"])
    else:
        cell_metadata = pd.read_csv(args["cell_metadata"])

    obsm = {"spatial":
        np.stack([cell_metadata.centroid_x.to_numpy(), cell_metadata.centroid_y.to_numpy()], axis=1)}
    var = pd.DataFrame(index=count_matrix.columns)
    obs = cell_metadata.set_index(pd.Index(cell_metadata.cell, str, True, "cell_d"))
    X = csr_matrix(count_matrix.to_numpy())

    # adata = anndata.AnnData(X=X, obs=cell_metadata, var=var, obsm=obsm)
    adata = anndata.AnnData(X=X, obs=obs, var=var, obsm=obsm)

    adata.write_h5ad(args["output"])
