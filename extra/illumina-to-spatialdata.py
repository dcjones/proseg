#!/usr/bin/env python

# Convert Illumina spatial output to an AnnData object that can be read and
# re-segmented by proseg.

from anndata import AnnData
from pathlib import Path
from scipy.io import mmread
from shapely import Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
import argparse
import geopandas as gpd
import numpy as np
import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Illumina spatial output to a zarr-serialized AnnData object."
    )
    parser.add_argument("input_path", help="Path to the Illumina spatial output.")
    parser.add_argument("sample_name", help="Sample name within the project.")
    parser.add_argument("output_file", help="Path to the output AnnData object file.")
    args = parser.parse_args()

    root_path = Path(args.input_path)

    if not root_path.exists():
        print(f"Input path does not exist: {root_path}")
        exit(1)

    # Read count matrix
    matrix_root_path = (
        root_path
        / "intermediate_results"
        / "02_matrix"
        / args.sample_name
        / f"{args.sample_name}_raw"
    )

    if not matrix_root_path.exists():
        print(f"Matrix root path does not exist: {matrix_root_path}")
        exit(1)

    X = mmread(matrix_root_path / "matrix.mtx.gz").transpose().tocsr()
    genes = pd.read_csv(
        matrix_root_path / "features.tsv.gz",
        header=None,
        sep="\t",
        index_col=1,
        names=["gene_id", "gene_name", "type"],
    )

    coords = pd.read_csv(
        matrix_root_path / "coords.tsv.gz", header=None, sep=":", names=["y", "x"]
    )
    coords = coords[["x", "y"]]

    # nanometers to micrometers (nuclei polygons are already in microns)
    coords["x"] /= 1000
    coords["y"] /= 1000

    adata = AnnData(
        X=X,
        obs=coords,
        var=genes,
        obsm={
            "spatial": coords.values,
        },
    )

    # Actually, I really would prefer to just include the polygons
    # cell_seg_path = (
    #     root_path
    #     / "intermediate_results"
    #     / "07_cell_segmentation"
    #     / args.sample
    #     / "cell_segmentation"
    #     / "nuclei_segmentation"
    #     / f"{args.sample}_nuclei_Segmentation_mask.tif"
    # )
    # mask_img = TiffFile(cell_seg_path)

    # Don't think I actually need this anywhere
    # # Read the registration transformation matriix from the image file metadata
    # primary_img_path = (
    #     root_path / "results" / args.sample_name / f"{args.sample_name}.ome.tiff"
    # )
    # img = TiffFile(primary_img_path)
    # img_metadata = ET.parse(StringIO(img.ome_metadata))
    # img_metadata_root = img_metadata.getroot()
    # annot_el = img_metadata_root.find(
    #     "{http://www.openmicroscopy.org/Schemas/OME/2016-06}StructuredAnnotations"
    # )

    # transform = None
    # for el in annot_el.findall(
    #     "{http://www.openmicroscopy.org/Schemas/OME/2016-06}MapAnnotation"
    # ):
    #     if el.get("ID") == "Annotation:ExportMetrics":
    #         jsondata = (
    #             el.find("{http://www.openmicroscopy.org/Schemas/OME/2016-06}Value")[0]
    #             .text.strip('"')
    #             .replace("\\", "")
    #         )
    #         data = json.loads(jsondata)
    #         transform = np.array(data["affine_transformation"]).reshape(2, 3)

    # if transform is None:
    #     raise ValueError("No affine transformation found")

    nuclei_poly_path = (
        root_path
        / "intermediate_results"
        / "07_cell_segmentation"
        / args.sample_name
        / "cell_segmentation"
        / "nuclei_segmentation"
        / f"{args.sample_name}_nuclei_contour_coords.csv"
    )

    nuclei_poly_coords = pd.read_csv(nuclei_poly_path)
    nuclei_polys = nuclei_poly_coords.groupby("cell_id").apply(
        lambda poly_coords: Polygon(
            np.stack([poly_coords["vertex_x"], poly_coords["vertex_y"]]).transpose()
        )
    )
    nuclei_polys = gpd.GeoDataFrame(geometry=gpd.GeoSeries(nuclei_polys))
    nuclei_polys = ShapesModel.parse(nuclei_polys)

    sdata = SpatialData(
        shapes={"nucleus_boundaries": nuclei_polys}, tables={"table": adata}
    )

    sdata.write(args.output_file)
