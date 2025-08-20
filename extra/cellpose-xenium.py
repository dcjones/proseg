#!/usr/bin/env python

# This was intended for use with Cellpose 4.0, but could probably be adapted to other versions.

from argparse import ArgumentParser
from cellpose import models, io
from cellpose.io import imread
import numpy as np
import gzip


def divceil(n: int, d: int):
    q, r = divmod(n, d)
    return q + bool(r)

if __name__ == "__main__":
    ap = ArgumentParser("Basic command line tool for running Cellpose and outputing masks into a format that Proseg can read.")
    ap.add_argument("input_filename", help="Should be a singe or multi-channel tiff file")
    ap.add_argument("masks_output", nargs="?", default="cellpose-masks.npy.gz")
    ap.add_argument("cellprobs_output", nargs="?", default="cellpose-cellprobs.npy.gz")
    ap.add_argument("--tile-size", help="Split the image into square of this size to save memory (but will introduce some edge effects)", type=int, default=None)
    args = ap.parse_args()

    io.logger_setup()

    # The cellpose cpsam model is only on three channels. It's not obvious which of the four
    # channels to choose for xenium segmentation stains.
    model = models.CellposeModel(gpu=True)
    img = imread(args.input_filename)

    if img.ndim == 2:
        img = np.expand_dims(img, 0)

    assert img.ndim == 3, f"Expected img to have 3 dimensions, but got {img.ndim} dimensions"
    c, w, h = img.shape

    if args.tile_size is None:
        args.tile_size = max(w, h)

    masks = np.zeros((w, h), dtype=np.uint32)
    cellprob = np.zeros((w, h), dtype=np.float32)

    x_tiles = divceil(w, args.tile_size)
    y_tiles = divceil(h, args.tile_size)
    nextcell = 0

    for i in range(x_tiles):
        for j in range(y_tiles):
            print(f"Tile {i+1}/{x_tiles}, {j+1}/{y_tiles}")
            x_from = i * args.tile_size
            x_to = min((i+1) * args.tile_size, w)
            y_from = j * args.tile_size
            y_to = min((j+1) * args.tile_size, h)
            tile = img[:,x_from:x_to, y_from:y_to]
            masks_ij, flows_ij, styles_ij = model.eval(tile)

            masks_ij = masks_ij.astype(np.uint32)
            ncells_ij = masks_ij.max()

            masks_ij[masks_ij != 0] += nextcell

            nextcell += ncells_ij

            masks[x_from:x_to, y_from:y_to] = masks_ij
            cellprob[x_from:x_to, y_from:y_to] = flows_ij[2]

    print(f"Segmented {masks.max()} cells")

    with gzip.open(args.masks_output, "wb") as out:
        np.save(out, masks)

    with gzip.open(args.cellprobs_output, "wb") as out:
        np.save(out, cellprob)
