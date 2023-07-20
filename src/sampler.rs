mod connectivity;
mod math;
pub mod hull;
mod sampleset;
pub mod transcripts;
// pub mod transcriptsampler;
pub mod hexbinsampler;

use core::fmt::Debug;
use math::{
    negbin_logpmf_fast, normal_logpdf, lognormal_logpdf,
    odds_to_prob, prob_to_odds, rand_pois,
    LogFactorial,
    LogGammaPlus,
};
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::{izip, Itertools};
use libm::{lgammaf, log1pf};
use ndarray::{Array1, Array2, Zip, s};
use rand::{thread_rng, Rng};
use rand_distr::{Beta, Binomial, Dirichlet, Distribution, Gamma, Normal};
use rayon::prelude::*;
use std::cell::RefCell;
use std::fs::File;
use std::io::Write;
use std::iter::Iterator;
use std::collections::HashMap;
use thread_local::ThreadLocal;
use transcripts::{Transcript, CellIndex, BACKGROUND_CELL};

// use std::time::Instant;


// Compute chunk and quadrant for a single a single (x,y) point.
fn chunkquad(x: f32, y: f32, xmin: f32, ymin: f32, chunk_size: f32, nxchunks: usize) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size / 2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size / 2.0)).floor() as u32;

    let chunk = (xchunkquad / 2) + (ychunkquad / 2) * (nxchunks as u32);
    let quad = (xchunkquad % 2) + (ychunkquad % 2) * 2;

    return (chunk, quad);
}

// Model prior parameters.
#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub min_cell_area: f32,
    pub min_cell_surface_area: f32,

    // params for normal prior
    pub μ_μ_area: f32,
    pub σ_μ_area: f32,

    // params for inverse-gamma prior
    pub α_σ_area: f32,
    pub β_σ_area: f32,

    // params for normal prior on compactness
    pub μ_μ_comp: f32,
    pub σ_μ_comp: f32,

    // params for inverse-gamma prior on compactness
    pub α_σ_comp: f32,
    pub β_σ_comp: f32,

    pub α_θ: f32,
    pub β_θ: f32,

    // gamma rate prior
    pub e_r: f32,
    pub f_r: f32,
}


// Model global parameters.
pub struct ModelParams {
    pub cell_assignments: Vec<CellIndex>,
    pub cell_assignment_time: Vec<u32>,

    pub cell_population: Vec<usize>,

    // per-cell areas
    cell_areas: Array1<f32>,

    // per-cell surface areas
    cell_surface_areas: Array1<f32>,

    // per-cell compactness, computed when sampling
    cell_compactness: Array1<f32>,

    // area of the convex hull containing all transcripts
    full_area: f32,

    // [ngenes, ncells] transcripts counts
    pub counts: Array2<u32>,

    // [ncells, ngenes] foreground transcripts counts
    foreground_counts: Array2<u32>,

    // [ngenes] background transcripts counts
    background_counts: Array1<u32>,

    // [ngenes] total gene occourance counts
    total_gene_counts: Array1<u32>,

    // Not parameters, but needed for sampling global params
    logfactorial: LogFactorial,
    loggammaplus: Vec<LogGammaPlus>,

    pub z: Array1<u32>, // assignment of cells to components

    // [ngenes, ncomponents] number of transcripts of each gene assigned to each component
    component_counts: Array2<u32>,

    component_population: Array1<u32>, // number of cells assigned to each component

    // thread-local space used for sampling z
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,

    π: Vec<f32>, // mixing proportions over components

    μ_area: Array1<f32>, // area dist mean param by component
    σ_area: Array1<f32>, // area dist std param by component

    μ_comp: Array1<f32>, // compactness dist mean param by component
    σ_comp: Array1<f32>, // compactness dist std param by component

    // [ngenes] NB r parameters.
    r: Array1<f32>,

    // Precomputing lgamma(r)
    lgamma_r: Array1<f32>,

    // [ncomponents, ngenes] NB p parameters.
    θ: Array2<f32>,

    // // log(ods_to_prob(θ))
    // logp: Array2<f32>,

    // // log(1 - ods_to_prob(θ))
    // log1mp: Array2<f32>,

    // [ngenes, ncells] Poisson rates
    λ: Array2<f32>,

    γ_bg: Array1<f32>,
    γ_fg: Array1<f32>,

    // background rate
    λ_bg: Array1<f32>,

    // time, which is incremented after every iteration
    t: u32,
}

impl ModelParams {
    // initialize model parameters, with random cell assignments
    // and other parameterz unninitialized.
    pub fn new(
        priors: &ModelPriors,
        full_area: f32,
        transcripts: &Vec<Transcript>,
        init_cell_assignments: &Vec<u32>,
        init_cell_population: &Vec<usize>,
        transcript_areas: &Vec<f32>,
        ncomponents: usize,
        ncells: usize,
        ngenes: usize) -> Self {

        let r = Array1::<f32>::from_elem(ngenes, 100.0_f32);
        let lgamma_r = Array1::<f32>::from_iter(r.iter().map(|&x| lgammaf(x)));

        // compute initial cell areas
        let mut cell_areas = Array1::<f32>::zeros(ncells);
        for (i, &j) in init_cell_assignments.iter().enumerate() {
            if j != BACKGROUND_CELL {
                cell_areas[j as usize] += transcript_areas[i];
            }
        }
        for area in cell_areas.iter_mut() {
            *area = area.max(priors.min_cell_area);
        }

        let cell_surface_areas = Array1::<f32>::from_elem(ncells, 0.0);
        let cell_compactness = Array1::<f32>::from_elem(ncells, 0.0);

        // compute initial counts
        let mut counts = Array2::<u32>::from_elem((ngenes, ncells), 0);
        let mut total_gene_counts = Array1::<u32>::from_elem(ngenes, 0);
        for (i, &j) in init_cell_assignments.iter().enumerate() {
            let gene = transcripts[i].gene as usize;
            if j != BACKGROUND_CELL {
                counts[[gene, j as usize]] += 1;
            }
            total_gene_counts[gene] += 1;
        }

        // initial component assignments
        let mut rng = rand::thread_rng();
        let z = (0..ncells)
            .map(|_| rng.gen_range(0..ncomponents) as u32)
            .collect::<Vec<_>>()
            .into();

        return ModelParams {
            cell_assignments: init_cell_assignments.clone(),
            cell_assignment_time: vec![0; init_cell_assignments.len()],
            cell_population: init_cell_population.clone(),
            cell_areas,
            cell_surface_areas,
            cell_compactness,
            full_area,
            counts,
            foreground_counts: Array2::<u32>::from_elem((ncells, ngenes), 0),
            background_counts: Array1::<u32>::from_elem(ngenes, 0),
            total_gene_counts,
            logfactorial: LogFactorial::new(),
            loggammaplus: Vec::from_iter((0..ngenes).map(|_| LogGammaPlus::default())),
            z,
            component_counts: Array2::<u32>::from_elem((ngenes, ncomponents), 0),
            component_population: Array1::<u32>::from_elem(ncomponents, 0),
            z_probs: ThreadLocal::new(),
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_area: Array1::<f32>::from_elem(ncomponents, priors.μ_μ_area),
            σ_area: Array1::<f32>::from_elem(ncomponents, priors.σ_μ_area),
            μ_comp: Array1::<f32>::from_elem(ncomponents, priors.μ_μ_comp),
            σ_comp: Array1::<f32>::from_elem(ncomponents, priors.σ_μ_comp),
            r,
            lgamma_r,
            θ: Array2::<f32>::from_elem((ncomponents, ngenes), 0.1),
            λ: Array2::<f32>::from_elem((ngenes, ncells), 0.1),
            γ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            γ_fg: Array1::<f32>::from_elem(ngenes, 0.0),
            λ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            t: 0,
        };
    }

    fn ncomponents(&self) -> usize {
        return self.π.len();
    }

    fn recompute_counts(&mut self, transcripts: &Vec<Transcript>) {
        self.counts.fill(0);
        for (i, &j) in self.cell_assignments.iter().enumerate() {
            let gene = transcripts[i].gene as usize;
            if j != BACKGROUND_CELL {
                self.counts[[gene, j as usize]] += 1;
            }
        }
    }

    pub fn nassigned(&self) -> usize {
        return self
            .cell_assignments
            .iter()
            .filter(|&c| *c != BACKGROUND_CELL)
            .count();
    }

    pub fn nunassigned(&self) -> usize {
        return self
            .cell_assignments
            .iter()
            .filter(|&c| *c == BACKGROUND_CELL)
            .count();
    }

    fn ncells(&self) -> usize {
        return self.cell_population.len();
    }

    fn ngenes(&self) -> usize {
        return self.total_gene_counts.len();
    }

    pub fn log_likelihood(&self) -> f32 {
        // TODO:
        //   - area terms
        //   - compactness terms

        let mut ll = Zip::from(self.λ.columns())
            .and(&self.cell_areas)
            .and(self.counts.columns())
            .fold(0_f32, |accum, λs, cell_area, cs| {
                accum + Zip::from(λs)
                    .and(&self.λ_bg)
                    .and(cs)
                    .fold(0_f32, |accum, λ, λ_bg, &c| {
                        accum + (c as f32) * (λ + λ_bg).ln() - λ * cell_area
                    })
            });

        // background terms
        ll += Zip::from(&self.total_gene_counts)
            .and(self.counts.rows())
            .and(&self.λ_bg)
            .fold(0_f32, |accum, c_total, cs, &λ| {
                let c_bg = c_total - cs.sum();
                accum + (c_bg as f32) * λ.ln() - λ * self.full_area
            });

        return ll;
    }

    pub fn write_cell_hulls(&self, transcripts: &Vec<Transcript>, counts: &Array2<u32>, filename: &str) {
        // We are not maintaining any kind of per-cell array, so I guess I have
        // no choice but to compute such a thing here.
        // TODO: We area already keeping track of this in Sampler!!
        let mut cell_transcripts: Vec<Vec<usize>> = vec![Vec::new(); self.ncells()];
        for (i, &cell) in self.cell_assignments.iter().enumerate() {
            if cell != BACKGROUND_CELL {
                cell_transcripts[cell as usize].push(i);
            }
        }

        let file = File::create(filename).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        writeln!(
            encoder,
            "{{\n  \"type\": \"FeatureCollection\",\n  \"features\": ["
        )
        .unwrap();

        let vertices: Vec<(f32, f32)> = Vec::new();
        let hull: Vec<(f32, f32)> = Vec::new();

        let vertices_refcell = RefCell::new(vertices);
        let mut vertices_ref = vertices_refcell.borrow_mut();

        let hull_refcell = RefCell::new(hull);
        let mut hull_ref = hull_refcell.borrow_mut();

        for (i, js) in cell_transcripts.iter().enumerate() {
            vertices_ref.clear();
            for j in js {
                let transcript = transcripts[*j];
                vertices_ref.push((transcript.x, transcript.y));
            }

            let area = convex_hull_area(&mut vertices_ref, &mut hull_ref);
            let count = counts.column(i).sum();

            writeln!(
                encoder,
                concat!(
                    "    {{\n",
                    "      \"type\": \"Feature\",\n",
                    "      \"properties\": {{\n",
                    "        \"cell\": {},\n",
                    "        \"area\": {},\n",
                    "        \"count\": {}\n",
                    "      }},\n",
                    "      \"geometry\": {{\n",
                    "        \"type\": \"Polygon\",\n",
                    "        \"coordinates\": [",
                    "          ["
                ),
                i, area, count
            )
            .unwrap();
            for (i, (x, y)) in hull_ref.iter().enumerate() {
                writeln!(encoder, "            [{}, {}]", x, y).unwrap();
                if i < hull_ref.len() - 1 {
                    write!(encoder, ",").unwrap();
                }
            }
            write!(
                encoder,
                concat!(
                    "          ]\n", // polygon
                    "        ]\n",   // coordinates
                    "      }}\n",    // geometry
                    "    }}\n",      // feature
                )
            )
            .unwrap();

            if i < cell_transcripts.len() - 1 {
                write!(encoder, ",").unwrap();
            }
        }

        writeln!(encoder, "\n  ]\n}}").unwrap();
    }


}


#[derive(Clone, Debug)]
pub struct ProposalStats {
    cell_to_cell_accept: usize,
    cell_to_cell_reject: usize,
    background_to_cell_accept: usize,
    background_to_cell_reject: usize,
    background_to_cell_ignore: usize,
    cell_to_background_accept: usize,
    cell_to_background_reject: usize,
}


impl ProposalStats {
    pub fn new() -> Self {
        ProposalStats {
            cell_to_cell_accept: 0,
            cell_to_cell_reject: 0,
            background_to_cell_accept: 0,
            background_to_cell_reject: 0,
            background_to_cell_ignore: 0,
            cell_to_background_accept: 0,
            cell_to_background_reject: 0,
        }
    }

    pub fn reset(&mut self) {
        self.cell_to_cell_accept = 0;
        self.cell_to_cell_reject = 0;
        self.background_to_cell_accept = 0;
        self.background_to_cell_reject = 0;
        self.background_to_cell_ignore = 0;
        self.cell_to_background_accept = 0;
        self.cell_to_background_reject = 0;
    }
}


pub struct UncertaintyTracker {
    cell_assignment_duration: HashMap<(usize, CellIndex), u32>,
}


impl UncertaintyTracker {
    pub fn new() -> UncertaintyTracker {
        let cell_assignment_duration = HashMap::new();

        return UncertaintyTracker {
            cell_assignment_duration,
        }
    }

    // record the duration of the current cell assignment. Called when the state
    // is about to change.
    fn update(&mut self, params: &ModelParams, i: usize) {
        let duration = params.t - params.cell_assignment_time[i];
        self.cell_assignment_duration
            .entry((i, params.cell_assignments[i]))
            .and_modify(|d| *d += duration)
            .or_insert(duration);
    }

    pub fn finish(&mut self, params: &ModelParams) {
        for ((i, &j), &t) in params.cell_assignments.iter().enumerate().zip(&params.cell_assignment_time) {
            let duration = params.t - t + 1;
            self.cell_assignment_duration
                .entry((i, j))
                .and_modify(|d| *d += duration)
                .or_insert(duration);
        }
    }

    pub fn max_posterior_transcript_counts(&self, params: &ModelParams, transcripts: &Vec<Transcript>, pr_cutoff: f32) -> Array2<u32> {
        let mut counts = Array2::<u32>::from_elem((params.ngenes(), params.ncells()), 0);

        // We need this to be descending on duration
        let sorted_cell_assignment_durations: Vec<_> = self.cell_assignment_duration
            .iter()
            .sorted_by(|((i_a, _), d_a), ((i_b, _), d_b)|
                (*i_b, **d_b).cmp(&(*i_a, **d_a)) )
            .collect();

        let mut i_prev = usize::MAX;
        for ((i, j), &d) in sorted_cell_assignment_durations.iter() {
            if *i != i_prev && *j != BACKGROUND_CELL && (d as f32 / params.t as f32) > pr_cutoff {
                counts[[transcripts[*i].gene as usize, *j as usize]] += 1;
            }
            i_prev = *i;
        }

        return counts;
    }
}



pub trait Proposal {
    fn accept(&mut self);
    fn reject(&mut self);

    fn ignored(&self) -> bool;
    fn accepted(&self) -> bool;

    // Return updated cell size minus current cell size `old_cell`
    fn old_cell_area_delta(&self) -> f32;

    // Return updated cell size minus current cell size `new_cell`
    fn new_cell_area_delta(&self) -> f32;

    // Return updated cell surface area minus current cell size `old_cell`
    fn old_cell_surface_area_delta(&self) -> f32;

    // Return updated cell surface area minus current cell size `new_cell`
    fn new_cell_surface_area_delta(&self) -> f32;

    fn old_cell(&self) -> u32;
    fn new_cell(&self) -> u32;

    fn log_weight(&self) -> f32;

    fn transcripts<'b, 'c>(&'b self) -> &'c[usize] where 'b: 'c;

    // Iterator over number of transcripts in the proposal of each gene
    fn gene_count<'b, 'c>(&'b self) -> &'c[u32] where 'b: 'c;

    fn evaluate(&mut self, params: &ModelParams) {
        if self.ignored() {
            self.reject();
            return;
        }

        let old_cell = self.old_cell();
        let new_cell = self.new_cell();
        let from_background = old_cell == BACKGROUND_CELL;
        let to_background = new_cell == BACKGROUND_CELL;

        // Log Metropolis-Hastings acceptance ratio
        let mut δ = 0.0;

        if from_background {
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ -= count as f32 * params.λ_bg[i].ln()
            }
        } else {
            let area_diff = self.old_cell_area_delta();
            let surface_area_diff = self.old_cell_surface_area_delta();

            let prev_area = params.cell_areas[old_cell as usize];
            let new_area = prev_area + area_diff;

            let prev_surface_area = params.cell_surface_areas[old_cell as usize];
            let new_surface_area = prev_surface_area + surface_area_diff;

            let prev_compactness = prev_surface_area.powf(1.5) / prev_area;
            let new_compactness = new_surface_area.powf(1.5) / new_area;

            // normalization term difference
            δ += Zip::from(params.λ.column(old_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * area_diff);

            // subtract out old cell likelihood terms
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ -= count as f32 * (params.λ_bg[i as usize] + params.λ[[i, old_cell as usize]]).ln();
            }

            let z = params.z[old_cell as usize];
            δ -= lognormal_logpdf(
                params.μ_area[z as usize],
                params.σ_area[z as usize],
                prev_area);
            δ += lognormal_logpdf(
                params.μ_area[z as usize],
                params.σ_area[z as usize],
                new_area);

            δ -= lognormal_logpdf(
                params.μ_comp[z as usize],
                params.σ_comp[z as usize],
                prev_compactness);
            δ += lognormal_logpdf(
                params.μ_comp[z as usize],
                params.σ_comp[z as usize],
                new_compactness);
        }

        if to_background {
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ += count as f32 * params.λ_bg[i].ln();
            }
        } else {
            let area_diff = self.new_cell_area_delta();
            let surface_area_diff = self.new_cell_surface_area_delta();

            let prev_area = params.cell_areas[new_cell as usize];
            let new_area = prev_area + area_diff;

            let prev_surface_area = params.cell_surface_areas[new_cell as usize];
            let new_surface_area = prev_surface_area + surface_area_diff;

            let prev_compactness = prev_surface_area.powf(1.5) / prev_area;
            let new_compactness = new_surface_area.powf(1.5) / new_area;

            // normalization term difference
            δ += Zip::from(params.λ.column(new_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * area_diff);

            // add in new cell likelihood terms
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ += count as f32 * (params.λ_bg[i] + params.λ[[i, new_cell as usize]]).ln();
            }

            let z = params.z[new_cell as usize];
            δ -= lognormal_logpdf(
                params.μ_area[z as usize],
                params.σ_area[z as usize],
                prev_area);
            δ += lognormal_logpdf(
                params.μ_area[z as usize],
                params.σ_area[z as usize],
                new_area);

            δ -= lognormal_logpdf(
                params.μ_comp[z as usize],
                params.σ_comp[z as usize],
                prev_compactness);
            δ += lognormal_logpdf(
                params.μ_comp[z as usize],
                params.σ_comp[z as usize],
                new_compactness);
        }

        let mut rng = thread_rng();
        let logu = rng.gen::<f32>().ln();

        if logu < δ + self.log_weight() {
            self.accept();
        } else {
            self.reject();
        }
    }
}


pub trait Sampler<P> where P: Proposal + Send {
    // fn generate_proposals<'b, 'c>(&'b mut self, params: &ModelParams) -> &'c mut [P] where 'b: 'c;

    fn repopulate_proposals(&mut self, params: &ModelParams);
    fn proposals<'a, 'b>(&'a self) -> &'b [P] where 'a: 'b;
    fn proposals_mut<'a, 'b>(&'a mut self) -> &'b mut [P] where 'a: 'b;

    // Called by `apply_accepted_proposals` to handle any sampler specific
    // updates needed after applying accepted proposals. This is mainly
    // updating mismatch edges.
    fn update_sampler_state(&mut self, params: &ModelParams);

    fn sample_cell_regions(
        &mut self, priors: &ModelPriors, params: &mut ModelParams,
        stats: &mut ProposalStats, transcripts: &Vec<Transcript>,
        uncertainty: &mut Option<&mut UncertaintyTracker>)
    {
        // don't count time unless we are tracking uncertainty
        if uncertainty.is_some() {
            params.t += 1;
        }
        self.repopulate_proposals(params);
        self.proposals_mut()
            .par_iter_mut()
            .for_each(|p| p.evaluate(params));
        self.apply_accepted_proposals(stats, transcripts, priors, params, uncertainty);
    }

    fn apply_accepted_proposals(
        &mut self,
        stats: &mut ProposalStats,
        transcripts: &Vec<Transcript>,
        priors: &ModelPriors,
        params: &mut ModelParams,
        uncertainty: &mut Option<&mut UncertaintyTracker>)
    {
        // Update cell assignments
        for proposal in self.proposals().iter().filter(|p| p.accepted() && !p.ignored()) {
            let old_cell = proposal.old_cell();
            let new_cell = proposal.new_cell();

            let mut count = 0;
            for &i in proposal.transcripts() {
                if let Some(uncertainty) = uncertainty.as_mut() {
                    uncertainty.update(params, i);
                }
                params.cell_assignments[i] = new_cell;
                params.cell_assignment_time[i] = params.t;
                count += 1;
            }

            // Update count matrix and areas
            if old_cell != BACKGROUND_CELL {
                params.cell_population[old_cell as usize] -= count;

                let mut cell_area = params.cell_areas[old_cell as usize];
                cell_area += proposal.old_cell_area_delta();
                cell_area = cell_area.max(priors.min_cell_area);
                params.cell_areas[old_cell as usize] = cell_area;

                let mut cell_surface_area = params.cell_surface_areas[old_cell as usize];
                cell_surface_area += proposal.old_cell_surface_area_delta();
                cell_surface_area = cell_surface_area.max(priors.min_cell_surface_area);
                params.cell_surface_areas[old_cell as usize] = cell_surface_area;

                for &i in proposal.transcripts() {
                    let gene = transcripts[i].gene;
                    params.counts[[gene as usize, old_cell as usize]] -= 1;
                }
            }

            if new_cell != BACKGROUND_CELL {
                params.cell_population[new_cell as usize] += count;

                let mut cell_area = params.cell_areas[new_cell as usize];
                cell_area += proposal.new_cell_area_delta();
                cell_area = cell_area.max(priors.min_cell_area);
                params.cell_areas[new_cell as usize] = cell_area;

                let mut cell_surface_area = params.cell_surface_areas[new_cell as usize];
                cell_surface_area += proposal.new_cell_surface_area_delta();
                cell_surface_area = cell_surface_area.max(priors.min_cell_surface_area);
                params.cell_surface_areas[new_cell as usize] = cell_surface_area;

                for &i in proposal.transcripts() {
                    let gene = transcripts[i].gene;
                    params.counts[[gene as usize, new_cell as usize]] += 1;
                }
            }

            // Keep track of stats
            if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.background_to_cell_accept += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                stats.cell_to_background_accept += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.cell_to_cell_accept += 1;
            }
        }

        for proposal in self.proposals().iter().filter(|p| !p.accepted() && !p.ignored()) {
            let old_cell = proposal.old_cell();
            let new_cell = proposal.new_cell();

            if old_cell == BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.background_to_cell_reject += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell == BACKGROUND_CELL {
                stats.cell_to_background_reject += 1;
            } else if old_cell != BACKGROUND_CELL && new_cell != BACKGROUND_CELL {
                stats.cell_to_cell_reject += 1;
            }
        }

        self.update_sampler_state(params);

    }


    fn sample_global_params(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        let mut rng = thread_rng();
        let ncomponents = params.ncomponents();

        self.sample_area_params(priors, params);
        self.sample_compactness_params(priors, params);

        // dbg!(params.cell_areas.slice(s![0..10]));
        // dbg!(params.cell_surface_areas.slice(s![0..10]));
        // dbg!(params.cell_compactness.slice(s![0..10]));

        // Sample background/foreground counts
        // let t0 = Instant::now();
        params.background_counts.assign(&params.total_gene_counts);
        Zip::from(params.counts.rows())
            .and(params.foreground_counts.columns_mut())
            .and(&mut params.background_counts)
            .and(params.λ.rows())
            .and(&params.λ_bg)
            .par_for_each(|cs, fcs, bc, λs, λ_bg| {
                let mut rng = thread_rng();
                for (c, fc, λ) in izip!(cs, fcs, λs) {
                    let p = λ / (λ + λ_bg);
                    // TODO: We should be sampling the assignment of every transcript to background
                    // taking into account the depth distribution.
                    //
                    // And then using those assignments when sampling depth distribution parameters.
                    // I'm afraid that would be super expensive though.

                    *fc = Binomial::new(*c as u64, p as f64).unwrap().sample(&mut rng) as u32;
                    *bc -= *fc;
                }
            });
        // println!("  Sample background counts: {:?}", t0.elapsed());

        // total component area
        let mut component_cell_area = vec![0_f32; params.ncomponents()];
        params.cell_areas
            .iter()
            .zip(&params.z)
            .for_each(|(area, z_i)| {
                component_cell_area[*z_i as usize] += *area;
            });

        // compute per component transcript counts
        params.component_counts.fill(0);
        Zip::from(params.component_counts.rows_mut())
            .and(params.foreground_counts.columns())
            .par_for_each(|mut compc, cellc| {
                for (c, component) in cellc.iter().zip(&params.z) {
                    compc[*component as usize] += *c;
                }
            });

        // Sample θ
        // let t0 = Instant::now();
        Zip::from(params.θ.columns_mut())
            .and(params.component_counts.rows())
            .and(&params.r)
            .par_for_each(|θs, cs, &r| {
                let mut rng = thread_rng();
                for (θ, &c, &a) in izip!(θs, cs, &component_cell_area) {
                    *θ = prob_to_odds(
                        Beta::new(
                            priors.α_θ + c as f32,
                            priors.β_θ + a * r,
                        )
                        .unwrap()
                        .sample(&mut rng),
                    );

                    *θ = θ.max(1e-6);
                }
            });
        // println!("  Sample θ: {:?}", t0.elapsed());

        // Sample r
        // let t0 = Instant::now();
        Zip::from(&mut params.r)
            .and(&mut params.lgamma_r)
            .and(&mut params.loggammaplus)
            .and(params.θ.columns())
            // .and(self.counts.rows())
            .par_for_each(|r, lgamma_r, loggammaplus, θs| {
                let mut rng = thread_rng();

                // self.cell_areas.slice(0..self.ncells)

                let u = Zip::from(&params.z)
                    .and(&params.cell_areas)
                    .fold(0, |accum, z, a| {
                        let θ = θs[*z as usize];
                        let λ = -*r * log1pf(-odds_to_prob(θ * *a));

                        // I gueess because there is less overhead, our simple Knuth
                        // sampler is considerably faster here.
                        // accum + Poisson::new(λ).unwrap().sample(&mut rng) as i32
                        accum + rand_pois(&mut rng, λ)

                    }) as f32;
                let v = Zip::from(&params.z)
                    .and(&params.cell_areas)
                    .fold(0.0, |accum, z, a| {
                        let w = θs[*z as usize];
                        accum + log1pf(-odds_to_prob(w * *a))
                    });

                *r = Gamma::new(priors.e_r + u, (priors.f_r - v).recip())
                    .unwrap()
                    .sample(&mut rng);

                assert!(r.is_finite());

                *lgamma_r = lgammaf(*r);
                loggammaplus.reset(*r);

                // TODO: any better solution here?
                *r = r.min(100.0).max(1e-4);
            });
        // println!("  Sample r: {:?}", t0.elapsed());

        // Sample λ
        // let t0 = Instant::now();
        Zip::from(params.λ.rows_mut())
            .and(params.foreground_counts.columns())
            .and(params.θ.columns())
            .and(&params.r)
            .par_for_each(|mut λs, cs, θs, r| {
                let mut rng = thread_rng();

                // TODO: Afraid this is where we'll get killed on performance. Look for
                // a Gamma distribution sampler that runs as f32 precision. Maybe in rand_distr

                for (λ, z, c, cell_area) in izip!(
                    &mut λs,
                    &params.z,
                    cs,
                    &params.cell_areas
                ) {
                    let θ = θs[*z as usize];
                    *λ = Gamma::new(
                        *r + *c as f32,
                        θ / (cell_area * θ + 1.0)
                        // ((cell_area * θ + 1.0) / θ) as f64,
                    )
                    .unwrap()
                    .sample(&mut rng)
                    .max(1e-9);

                    assert!(λ.is_finite());

                    // dbg!(*λ, *r, θ, *c, cell_area, *c as f32 / cell_area);
                }
            });
        // println!("  Sample λ: {:?}", t0.elapsed());


        // TODO:
        // This is the most expensive part. We could sample this less frequently,
        // but we should try to optimize as much as possible.
        // Ideas:
        //   - Main bottlneck is computing log(p) and log(1-p).


        // Sample z
        // let t0 = Instant::now();
        Zip::from(
            params.foreground_counts
                .rows(),
        )
        .and(&mut params.z)
        .and(&params.cell_areas)
        .and(&params.cell_compactness)
        .par_for_each(|cs, z_i, cell_area, cell_compactness| {
            let mut z_probs = params
                .z_probs
                .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                .borrow_mut();

            // loop over components
            for (zp, π, θs, &μ_area, &σ_area, &μ_comp, &σ_comp) in
                izip!(z_probs.iter_mut(), &params.π, params.θ.rows(), &params.μ_area, &params.σ_area, &params.μ_comp, &params.σ_comp)
            {
                // sum over genes
                *zp = (*π as f64)
                    * (Zip::from(cs)
                        .and(&params.r)
                        .and(&params.lgamma_r)
                        .and(&params.loggammaplus)
                        .and(&θs)
                        .fold(0_f32, |accum, &c, &r, &lgamma_r, lgammaplus, θ| {
                            accum + negbin_logpmf_fast(
                                r, lgamma_r, lgammaplus.eval(c),
                                odds_to_prob(*θ * cell_area), c, params.logfactorial.eval(c))
                        }) as f64)
                        .exp();

                *zp *= lognormal_logpdf(
                    μ_area, σ_area, *cell_area).exp() as f64;

                *zp *= lognormal_logpdf(
                    μ_comp, σ_comp, *cell_compactness).exp() as f64;
            }

            // z_probs.iter_mut().enumerate().for_each(|(j, zp)| {
            //     *zp = (self.params.π[j] as f64) *
            //         negbin_logpmf(r, lgamma_r, p, k)
            //         // (self.params.cell_logprob_fast(j as usize, *cell_area, &cs, &clfs) as f64).exp();
            // });

            let z_prob_sum = z_probs.iter().sum::<f64>();

            assert!(z_prob_sum.is_finite());

            // cumulative probabilities in-place
            z_probs.iter_mut().fold(0.0, |mut acc, x| {
                acc += *x / z_prob_sum;
                *x = acc;
                acc
            });

            let rng = &mut thread_rng();
            let u = rng.gen::<f64>();
            *z_i = z_probs.partition_point(|x| *x < u) as u32;
        });
        // println!("  Sample z: {:?}", t0.elapsed());

        // sample π
        let mut α = vec![1_f32; params.ncomponents()];
        for z_i in params.z.iter() {
            α[*z_i as usize] += 1.0;
        }

        params.π.clear();
        params.π.extend(
            Dirichlet::new(&α)
                .unwrap()
                .sample(&mut rng)
                .iter()
                .map(|x| *x as f32),
        );

        // Sample background rates
        // if let Some(background_proportion) = background_proportion {
        //     Zip::from(&mut params.λ_bg)
        //         .and(&params.total_gene_counts)
        //         .for_each(|λ, c| {
        //             *λ = background_proportion * (*c as f32) / params.full_area;
        //         });
        // } else {
            Zip::from(&mut params.λ_bg)
                .and(&params.background_counts)
                .for_each(|λ, c| {
                    *λ = (*c as f32) / params.full_area;
                });
        // }

        // dbg!(&self.background_counts, &self.params.λ_bg);

        // Comptue γ_bg
        Zip::from(&mut params.γ_bg)
            .and(&params.λ_bg)
            .for_each(|γ, λ| {
                *γ = λ * params.full_area;
            });

        // Compute γ_fg
        Zip::from(&mut params.γ_fg)
            .and(params.λ.rows())
            .par_for_each(|γ, λs| {
                *γ = 0.0;
                for (λ, cell_area) in izip!(λs, &params.cell_areas) {
                    *γ += *λ * *cell_area;
                }
            });

    }

    fn sample_area_params(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // compute sample means
        params.component_population.fill(0_u32);
        params.μ_area.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_areas)
            .for_each(|&z, &area| {
                params.μ_area[z as usize] += area.ln();
                params.component_population[z as usize] += 1;
            });

        // sample μ parameters
        Zip::from(&mut params.μ_area)
            .and(&params.σ_area)
            .and(&params.component_population)
            .par_for_each(|μ, &σ, &pop| {
                let mut rng = thread_rng();

                let v = (1_f32 / priors.σ_μ_area.powi(2) + pop as f32 / σ.powi(2)).recip();
                *μ = Normal::new(
                    v * (priors.μ_μ_area / priors.σ_μ_area.powi(2) + *μ / σ.powi(2)),
                    v.sqrt()
                ).unwrap().sample(&mut rng);
            });

        // compute sample variances
        params.σ_area.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_areas)
            .for_each(|&z, &area| {
                params.σ_area[z as usize] += (params.μ_area[z as usize] - area.ln()).powi(2);
            });

        // sample σ parameters
        Zip::from(&mut params.σ_area)
            .and(&params.component_population)
            .par_for_each(|σ, &pop| {
                let mut rng = thread_rng();
                *σ = Gamma::new(
                    priors.α_σ_area + (pop as f32) / 2.0,
                    (priors.β_σ_area + *σ / 2.0).recip()).unwrap().sample(&mut rng).recip().sqrt();
            });
    }

    fn sample_compactness_params(&mut self, priors: &ModelPriors, params: &mut ModelParams) {
        // compute cell compactness
        Zip::from(&mut params.cell_compactness)
            .and(&params.cell_areas)
            .and(&params.cell_surface_areas)
            .for_each(|c, &a, &sa| {
                *c = sa.powf(1.5) / a;
            });

        // compute sample means
        // NOTE: assuming component_population has been computed at this point
        params.μ_comp.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_compactness)
            .for_each(|&z, &c| {
                params.μ_comp[z as usize] += c.ln();
            });

        // sample μ parameters
        Zip::from(&mut params.μ_comp)
            .and(&params.σ_comp)
            .and(&params.component_population)
            .par_for_each(|μ, &σ, &pop| {
                let mut rng = thread_rng();

                let v = (1_f32 / priors.σ_μ_comp.powi(2) + pop as f32 / σ.powi(2)).recip();
                *μ = Normal::new(
                    v * (priors.μ_μ_comp / priors.σ_μ_comp.powi(2) + *μ / σ.powi(2)),
                    v.sqrt()
                ).unwrap().sample(&mut rng);
            });

        // compute sample variances
        params.σ_comp.fill(0_f32);
        Zip::from(&params.z)
            .and(&params.cell_compactness)
            .for_each(|&z, &c| {
                params.σ_comp[z as usize] += (params.μ_comp[z as usize] - c.ln()).powi(2);
            });

        // sample σ parameters
        Zip::from(&mut params.σ_comp)
            .and(&params.component_population)
            .par_for_each(|σ, &pop| {
                let mut rng = thread_rng();
                *σ = Gamma::new(
                    priors.α_σ_comp + (pop as f32) / 2.0,
                    (priors.β_σ_comp + *σ / 2.0).recip()).unwrap().sample(&mut rng).recip().sqrt();
            });
    }

}
