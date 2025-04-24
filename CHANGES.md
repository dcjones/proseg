
# 2.0.4
  - Fix potential empty cells when large number of voxel layers are used.
  - Remove `--double-z-layers` which was always true, and add `--no-z-layer-doubling` to turn it off.
  - Use a default min qv of 20 in Xenium.

# 2.0.3
  - Fix crash on data with fewer than 300 genes.
  - Output corresponding original cell ids to the cell metadata table.

# 2.0.2
  - Updating dependencies to avoid some compilation errors.
  - Optimizations (should be 20-30% faster on high-plex data)
  - Automatic untilting of samples that are tilted on the z-axis.

# 2.0.1
  - Implement a partial factorization model to avoid model failures on high-plex
    data like Xenium Prime.
  - Fix `--initial-voxel-size` not actually doing anything.
  - Include an experiment `--visiumhd` preset, that has had to be tested.

# 1.1.9
  - Fix potential error using rust >=1.81 (Issue #41)

# 1.1.8
  - Fix error reading some newer Xenium parquet files.

# 1.1.7
  - Fix error compiling proseg using rust 1.81.

# 1.1.6
  - Fix errors when importing segmentation using xeniumranger 3. (Issue #38)

# 1.1.5
  - Fix failure to read certain .csv.gz input files. (Issue #29)

# 1.1.4
  - Fix error reading csv transcript metadata in `proseg-to-baysor`

# 1.1.3
  - Fix some cases in which weird or invalid polygons are reported.

# 1.1.2
  - Use newer Xenium parquet format.

# 1.1.1
  - Support for running on Xenium parquet files.
  - Switch from unmaintained arrow2 crate to maintained arrow/parquet crates.

# 1.1.0

  - `--output-cell-polygons` will now generate non-overlapping "consensus"
    polygons formed by taking the dominant cell assignment at each x/y location.
    The old behavior, taking the union along the z-axis, can be generated now with
    `--output-union-cell-polygons`.
  - Default behavior is now to initialize voxels in 1D and double z-resolution twice,
    which seems to lead to less spurious overlap between cells.
  - `proseg-to-baysor` now filter out empty cells that were causing issues with
    Xenium Ranger.

# 1.0.6

  - Add an option `--use-cell-initialization` to ignore compartment information
    and initialize with cell assignment.
  - Change the behavior of the `--cosmx` preset to support newer CosMx
    transcript tables.

# 1.0.5

  - Fix overflow error generating transcript metadata table with large number
    of transcripts. (Fixes #12)

# 1.0.4

  - Include fov in transcript metadata and cell metadata (Issue #6)
  - Read fov column from more recent Xenium data
  - Remove polygon interiors that form when flattening polygons into 2D.

# 1.0.3

  - Include a `--merscope` preset to work with MERSCOPE `detected_transcripts.csv.gz`
    files. Mark the `--merfish` preset as deprecated. (Issue #5)

# 1.0.2

  - Fix overflow in `proseg-to-baysor` with large number of cells.

# 1.0.1

  - Fix for incompatibility with different CosMx file versions (Issue #3)
  - Fix incompatibility with xenium ranger 2.0 (Issue #2)

# 1.0.0

Initial release
