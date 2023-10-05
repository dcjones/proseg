
// Here I want to estimate and re-estimate a diffusion function estimating the
// rate at which transcripts diffuse outward from leaky cells. It's easy enough
// to compute gaussian blur out from some populated cells, but we then need to
// subtract out intra-cell diffusion.
//
// This means computing the global diffusion map, then separately on each cell
// computing a diffusion map, and subtracting the two.

use ndarray::{Array1, Array2, Zip, s};
use rayon::prelude::IntoParallelRefMutIterator;
use thread_local::ThreadLocal;
use std::cell::RefCell;
use rayon::prelude::*;
use itertools::izip;

use super::cubebinsampler::{CubeLayout, CubeCellMap};
use super::transcripts::{Transcript, BACKGROUND_CELL};
use super::conv::Conv2D;


fn normal_pdf(σ: f32, x: f32) -> f32 {
    const SQRT_TWO_PI: f32 = 2.5066282746310002;
    return (-0.5 * (x / σ).powi(2)).exp() / (σ * SQRT_TWO_PI);
}

fn gaus_kernel(binsize: f32, σ: f32, k: usize) -> Array2<f32> {
    let mut kernel = Array2::zeros((1 + 2 * k, 1 + 2 * k));

    for i in 0..(1 + 2 * k) {
        for j in 0..(1 + 2 * k) {
            kernel[[i, j]] = normal_pdf(σ, (i as f32 - k as f32) * binsize)
                * normal_pdf(σ, (j as f32 - k as f32) * binsize);
        }
    }

    kernel /= kernel.sum();

    return kernel;
}


#[derive(Clone, Copy)]
struct CellSpan {
    i0: i32,
    j0: i32,
    i1: i32,
    j1: i32,
}


pub struct TranscriptDiffusionModel {
    eps: f32,
    layout: CubeLayout,
    kernel: Array2<f32>,
    diffusion: Array2<f32>,
    occupancy: Array2<bool>,
    conv: Conv2D,
    intracell_conv: ThreadLocal<RefCell<Conv2D>>,
    intracell_diffusions: Vec<Array2<f32>>,
    intracell_occupancies: Vec<Array2<bool>>,
    cell_span: Vec<CellSpan>,
}


impl TranscriptDiffusionModel {
    pub fn new(σ: f32, eps: f32, k: usize, layout: CubeLayout, xmax: f32, ymax: f32) -> Self {
        let xspan = xmax - layout.origin.0;
        let yspan = ymax - layout.origin.1;
        let nxbins = (xspan / layout.cube_size.0).ceil() as usize;
        let nybins = (yspan / layout.cube_size.1).ceil() as usize;
        let diffusion = Array2::zeros((nxbins, nybins));
        let kernel = gaus_kernel(layout.cube_size.0, σ, k);
        let intracell_conv = ThreadLocal::new();
        let conv = Conv2D::new((nxbins, nybins), kernel.clone());
        let cell_span = Vec::new();
        let occupancy = Array2::from_elem((nxbins, nybins), false);

        return TranscriptDiffusionModel {
            eps,
            layout,
            kernel,
            diffusion,
            occupancy,
            conv,
            intracell_conv,
            intracell_diffusions: Vec::new(),
            intracell_occupancies: Vec::new(),
            cell_span,
        };
    }

    pub fn estimate_diffusion(&mut self, λ: &Array2<f32>, cubecells: &CubeCellMap, transcripts: &Vec<Transcript>, density: &mut Array1<f32>) {
        let ngenes = λ.shape()[0];
        let ncells = λ.shape()[1];

        println!("0");

        // compute the span for each cell
        if self.cell_span.len() != ncells {
            self.cell_span.resize(ncells, CellSpan {
                i0: 0,
                j0: 0,
                i1: 0,
                j1: 0,
            });
        }
        self.cell_span.fill(CellSpan { i0: i32::MAX, j0: i32::MAX, i1: i32::MAX, j1: i32::MAX });

        cubecells
            .iter()
            .for_each(|(&cube, &cell)| {
                if cell != BACKGROUND_CELL {
                    let span = &mut self.cell_span[cell as usize];
                    if cube.i < span.i0 || span.i0 == i32::MAX {
                        span.i0 = cube.i;
                    }
                    if cube.i > span.i1 || span.i1 == i32::MAX {
                        span.i1 = cube.i;
                    }
                    if cube.j < span.j0 || span.j0 == i32::MAX {
                        span.j0 = cube.j;
                    }
                    if cube.j > span.j1 || span.j1 == i32::MAX {
                        span.j1 = cube.j;
                    }
                }
            });

        println!("1");
        let imax = self.occupancy.shape()[0] - 1;
        let jmax = self.occupancy.shape()[1] - 1;

        self.cell_span
            .iter_mut()
            .for_each(|span| {
                if span.i0 == i32::MAX || span.i1 == i32::MAX || span.j0 == i32::MAX || span.j1 == i32::MAX {
                    span.i0 = 0;
                    span.i1 = 0;
                    span.j0 = 0;
                    span.j1 = 0;
                }

                span.i0 = span.i0.max(0).min(imax as i32);
                span.i1 = span.i0.max(0).min(imax as i32);
                span.j0 = span.j0.max(0).min(jmax as i32);
                span.j1 = span.j0.max(0).min(jmax as i32);
            });

        // resize things as needed
        let max_span = self.cell_span
            .iter()
            .fold((0, 0), |max_span, cell_span| {
                (max_span.0.max(cell_span.i1 - cell_span.i0 + 1), max_span.1.max(cell_span.j1 - cell_span.j0 + 1))
            });

        println!("3");

        if self.intracell_diffusions.len() != ncells {
            self.intracell_diffusions.resize(ncells, Array2::zeros((max_span.0 as usize, max_span.1 as usize)));
        }


        for (intracell_diffusion, cell_span) in self.intracell_diffusions.iter_mut().zip(self.cell_span.iter()) {
            if intracell_diffusion.shape()[0] < (cell_span.i1 - cell_span.i0 + 1) as usize ||
                intracell_diffusion.shape()[1] < (cell_span.j1 - cell_span.j0 + 1) as usize
            {
                *intracell_diffusion = Array2::zeros(((cell_span.i1 - cell_span.i0 + 1) as usize, (cell_span.j1 - cell_span.j0 + 1) as usize));
            }
        }

        println!("4");

        if self.intracell_occupancies.len() != ncells {
            self.intracell_occupancies.resize(ncells, Array2::from_elem((max_span.0 as usize, max_span.1 as usize), false));
        }

        for (intracell_occupancy, cell_span) in self.intracell_occupancies.iter_mut().zip(self.cell_span.iter()) {
            if intracell_occupancy.shape()[0] < (cell_span.i1 - cell_span.i0 + 1) as usize ||
                intracell_occupancy.shape()[1] < (cell_span.j1 - cell_span.j0 + 1) as usize
            {
                *intracell_occupancy = Array2::from_elem(((cell_span.i1 - cell_span.i0 + 1) as usize, (cell_span.j1 - cell_span.j0 + 1) as usize), false);
            }
        }

        println!("5");

        dbg!(self.occupancy.shape());
        dbg!(cubecells.iter().map(|(cube, _)| cube.i).max());
        dbg!(cubecells.iter().map(|(cube, _)| cube.j).max());

        // compute occupancy maps
        self.occupancy.fill(false);
        cubecells
            .iter()
            .for_each(|(&cube, &cell)| {
                if cell != BACKGROUND_CELL {
                    self.occupancy[[cube.i.max(0).min(imax as i32) as usize, cube.j.max(0).min(jmax as i32) as usize]] = true;
                }
            });

        println!("A");

        self.intracell_occupancies
            .par_iter_mut()
            // .iter_mut() // TODO:
            .zip(&self.cell_span)
            .for_each(|(intracell_occupancy, cell_span)| {
                intracell_occupancy.fill(false);
                let (i0, i1, j0, j1) =
                    (cell_span.i0 as usize, cell_span.i1 as usize, cell_span.j0 as usize, cell_span.j1 as usize);

                let i_span = i1 - i0 + 1;
                let j_span = j1 - j0 + 1;

                dbg!(intracell_occupancy.shape());
                dbg!((i0, i1, j0, j1));

                intracell_occupancy.slice_mut(s![0..i_span, 0..j_span]).assign(
                    &self.occupancy.slice(s![i0..i1 + 1, j0..j1 + 1])
                );
            });

        println!("B");

        for gene in 0..ngenes {
            dbg!(gene);
            self.diffusion.fill(self.eps);

            let λgene = λ.row(gene);

            // println!("C");

            // assign rates to cell polygons
            cubecells
                .iter()
                .for_each(|(&cube, &cell)| {
                    if cell != BACKGROUND_CELL {
                        self.diffusion[[cube.i.max(0).min(imax as i32) as usize, cube.j.max(0).min(jmax as i32) as usize]] = λgene[cell as usize];
                    }
                });

            // compute intra-cell diffusion for each cell
            self.intracell_diffusions
                .par_iter_mut() // TODO:
                // .iter_mut()
                .zip(&self.cell_span)
                .for_each(|(intracell_diffusion, cell_span)| {
                    intracell_diffusion.fill(0.0);
                    let (i0, i1, j0, j1) =
                        (cell_span.i0 as usize, cell_span.i1 as usize, cell_span.j0 as usize, cell_span.j1 as usize);

                    let i_span = i1 - i0 + 1;
                    let j_span = j1 - j0 + 1;

                    intracell_diffusion.slice_mut(s![0..i_span, 0..j_span]).assign(
                        &self.diffusion.slice(s![i0..i1 + 1, j0..j1 + 1])
                    );

                    let mut conv = self.intracell_conv
                        .get_or(|| RefCell::new(Conv2D::new((i_span, j_span), self.kernel.clone())))
                        .borrow_mut();

                    if conv.shape().0 < intracell_diffusion.shape()[0] || conv.shape().1 < intracell_diffusion.shape()[1] {
                        *conv = Conv2D::new(intracell_diffusion.dim(), self.kernel.clone());
                    }

                    conv.compute(intracell_diffusion);
                });

            // compute global diffusion map
            self.conv.compute(&mut self.diffusion);

            for (intracell_diffusion, intracell_occupancy, cell_span) in
                izip!(&self.intracell_diffusions, &self.intracell_occupancies, &self.cell_span)
            {
                let (i0, i1, j0, j1) =
                    (cell_span.i0 as usize, cell_span.i1 as usize, cell_span.j0 as usize, cell_span.j1 as usize);

                let i_span = i1 - i0 + 1;
                let j_span = j1 - j0 + 1;

                Zip::from(self.diffusion.slice_mut(s![i0..i1+1, j0..j1+1]))
                    .and(intracell_occupancy.slice(s![..i_span, ..j_span]))
                    .and(intracell_diffusion.slice(s![..i_span, ..j_span]))
                    .for_each(|d, &occ, &d_intra| {
                        if occ {
                            *d -= d_intra;
                        }
                    })
            }

            // assign densities to transcripts
            for (t, d) in transcripts.iter().zip(density.iter_mut()) {
                if t.gene == gene as u32 {
                    let cube = self.layout.world_pos_to_cube((t.x, t.y, 0.0));
                    *d = self.diffusion[[cube.i as usize, cube.j as usize]];
                }
            }
        }
    }
}

// TODO: The remaining big problem then is then to estimate `sigma`. I really
// have no clue how to do that because recomputing diffusion is expensive!

