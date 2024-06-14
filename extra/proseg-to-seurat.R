

ProsegToSeurat <- function(proseg_output_path, expected_counts_basename="expected-counts", cell_metadata_basename="cell-metadata") {
  expected_counts_path <- file.path(proseg_output_path, paste0(expected_counts_basename, ".csv.gz"))
  if (!file.exists(expected_counts_path)) {
    expected_counts_path <- file.path(proseg_output_path, paste0(expected_counts_basename, ".parquet"))
    if (!file.exists(expected_counts_path)) {
      stop("Can't find expected-counts file.")
    }
    expected_counts <- arrow::read_parquet(expected_counts_path)

  } else {
    expected_counts <- read.csv(expected_counts_path, header=TRUE, sep=",")
  }

  cell_metadata_path <- file.path(proseg_output_path, paste0(cell_metadata_basename, ".csv.gz"))
  if (!file.exists(cell_metadata_path)) {
    cell_metadata_path <- file.path(proseg_output_path, paste0(cell_metadata_basename, ".parquet"))
    if (!file.exists(cell_metadata_path)) {
      stop("Can't find cell-metadata file.")
    }
    cell_metadata <- arrow::read_parquet(cell_metadata_path)

  } else {
    cell_metadata <- read.csv(cell_metadata_path, header=TRUE, sep=",")
  }

  # exclude cells with no assigned transcripts that end up with undefinied centroids
  mask <- is.finite(cell_metadata$centroid_x) & is.finite(cell_metadata$centroid_y)
  cell_metadata <- cell_metadata[mask,]
  expected_counts <- expected_counts[mask,]

  sobj <- Seurat::CreateSeuratObject(
    counts=Matrix::Matrix(t(as.matrix(expected_counts)), sparse=TRUE),
    meta.data=cell_metadata,
    assay="RNA"
  )

  # From: https://github.com/satijalab/seurat/issues/2790#issuecomment-721400652
  coords_df <- cell_metadata[c("centroid_x", "centroid_y")]
  names(coords_df) <- c("x", "y")
  rownames(coords_df) <- colnames(sobj)

  sobj@images$image =  new(
    Class = 'SlideSeq',
    assay = "Spatial",
    key = "image_",
    coordinates =  coords_df)

  return(sobj)
}

