use super::transcripts::{coordinate_span, Transcript};
use super::conv::Conv2D;
use ndarray::{Array1, Array2};

// Simple bivariate density estimation by approximate gaussian bluring.
// Returns a vector giving for each transcript the local density of its gene.
pub fn estimate_transcript_density(
    transcripts: &Vec<Transcript>,
    ngenes: usize,
    layer_depth: f32,
    σ: f32,
    binsize: f32,
    k: usize,
    eps: f32,
) -> (Array1<f32>, Array1<f32>) {
    let kernel = gaus_kernel(binsize, σ, k);

    let (x0, x1, y0, y1, _z0, _z1) = coordinate_span(transcripts);

    let nxbins = ((x1 - x0) / binsize).ceil() as usize;
    let nybins = ((y1 - y0) / binsize).ceil() as usize;

    // local density for each transcript
    let mut transcript_density = Array1::zeros(transcripts.len());

    // binned density, for computing gaussian blur
    let mut density = Array2::zeros((nxbins, nybins));

    let mut total_density = Array1::zeros(ngenes);

    // We do this sequentially one gene at a time to avoid allocating a huge array.
    for gene in 0..ngenes {
        density.fill(eps);

        for transcript in transcripts {
            if transcript.gene == gene as u32 {
                let xbin = ((transcript.x - x0) / binsize).floor() as usize;
                let ybin = ((transcript.y - y0) / binsize).floor() as usize;
                density[[xbin, ybin]] += 1.0;
            }
        }

        // Gaussian blur
        let mut conv2d = Conv2D::new((nxbins, nybins), kernel.clone());
        conv2d.compute(&mut density);

        for (transcript, d) in transcripts.iter().zip(&mut transcript_density) {
            if transcript.gene == gene as u32 {
                let xbin = ((transcript.x - x0) / binsize).floor() as usize;
                let ybin = ((transcript.y - y0) / binsize).floor() as usize;
                *d = density[[xbin, ybin]];
            }
        }

        total_density[gene] = density.sum() * binsize * binsize * layer_depth;
    }

    return (transcript_density, total_density);
}

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
