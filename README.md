
# Proseg

Proseg (**pro**babilistic **seg**mentation) is a cell segmentation method for in
situ spatial transcriptomics. Xenium, CosMx, and MERSCOPE platforms are
currently supported.


## Installing

Proseg can be built and installed with cargo.

```shell
cargo install --path /path/to/proseg
```

## Running on Xenium

A minimum invocation to run on Xenium data is simply:

```shell
proseg --xenium /path/to/transcripts.csv.gz
```
There are numerous arguments tat can be passed to the program that tweak the
behiavior of the algorithm. Because the method is still under development, some
of these arguments will do nothing at all, others might have dramatic effects.
Ideally one should not have to tinker with these, but a few might be useful to know:

  * `--nthreads N`: Number of CPU threads to use. By default this is set to the number of (virtual) cores available on the system.
  * `--no-diffusion`: By default Proseg models cells as leaky, under the assumption that some amount of RNA leaks from cells and diffuses elsewhere. This seems to be the case in much of the Xenium data we've seen, but could be a harmfully incorrect assumption in some data. This argument disables that part of the model.
  * `--min-qv 0`: Filter transcripts by quality value. There are relatively few low-quality transcripts included in Xenium output, so this doesn't seem to have much effect.
  * `--ncomponents 5`: Cell gene expression is a modeled as a mixture of negative binomial distributions. This parameter controls the number of mixture components. More components will tend to nudge the cells into more distinct types, but setting it too high risks manifesting cell types that are not real.

Proseg has a number of arguments to output various count matrices and metadata.
Most output files can be either `.csv.gz` files or `.parquet` files.

Most importantly:

  * `--expected-counts expected-counts.csv.gz`: Cell-by-gene count matrix. Proseg is a sampling method, so these are posterior expectations that will generally not be integers but fractional counts.
  * `--output-cell-metadata cell-metadata.csv.gz`: Cell centroids, volume, and other information.
  * `--output-cell-polygons cell-polygons.geojson.gz`: 2D polygons for each cell in GeoJSON format. These are flattened from 3D, so will overlap.
  * `--output-cell-polygon-layers cell-polygons-layers.geojson.gz`: 2D polygons for each cell in GeoJSON format. These are flattened from 3D, so will overlap.

Addition output options that may be useful:
  * `--transcript-metadata transcript-metadata.csv.gz`: Transcript position, repositioned location, cell assignemnet, etc.
  * `--output-counts counts.csv.gz`: Cell-by-gene count matrix point estimate. This is integer values, unlike expected-counts. Expected counts is somewhat more reliable in our experience.
  * `--output-rates rates.csv.gz`: Cell-by-gene Poisson rate parameters. These are essentially expected relative expression values, but may be too overly-smoothed for use in downstream analysis.


## Using Xenium Explorer with `proseg-to-baysor`

It is possible to use proseg segmentation with Xenium Explorer, but requires a
little work.

The [xeniumranger](https://www.10xgenomics.com/support/software/xenium-ranger) tool has a
command to import segmentation from [Baysor](https://github.com/kharchenkolab/Baysor). To use this,
we must first convert Proseg output to Baysor-compatible formatting.

For this we need transcript metadata and cell polygons from Proseg, then run the provided `proseg-to-baysor`
command like

```shell
proseg-to-baysor \
    transcript-metadata.csv.gz \
    cell-polygons.geojson.gz \
    --output-transcript-metadata baysor-transcript-metadata.csv \
    --output-cell-polygons baysor-cell-polygons.geojson
```

`xeniumranger` can then be run to import these into a format useable with Xenium Explorer:

```shell
xeniumranger \
    --id project-id \
    --xeinum-bundle /path/to/original/xenium/output \
    --viz-polygons baysor-cell-polygons.geojson \
    --transcript-assignment baysor-transcript-metadata.csv \
    --units microns
```

This will output a new Xenium bundle under the `project-id` directory

