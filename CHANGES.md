
# 1.0.6

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


