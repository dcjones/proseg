mod connectivity;
mod math;
pub mod hull;
mod sampleset;
pub mod transcripts;
// pub mod transcriptsampler;
pub mod hexbinsampler;

use core::fmt::Debug;
use std::arch::x86_64::_popcnt32;
use math::{
    negbin_logpmf_fast,
    odds_to_prob, prob_to_odds, rand_pois,
    LogFactorial,
    LogGammaPlus,
};
use flate2::write::GzEncoder;
use flate2::Compression;
use hull::convex_hull_area;
use itertools::izip;
use libm::{lgammaf, log1pf};
use ndarray::{Array1, Array2, Zip};
use petgraph::visit::IntoNeighbors;
use rand::{thread_rng, Rng};
use rand_distr::{Beta, Binomial, Dirichlet, Distribution, Gamma, Normal};
use rayon::prelude::*;
use std::cell::{RefCell, RefMut};
use std::fs::File;
use std::io::Write;
use std::iter::Iterator;
use thread_local::ThreadLocal;
use transcripts::{coordinate_span, NeighborhoodGraph, Transcript, CellIndex, BACKGROUND_CELL};
use sampleset::SampleSet;

use std::time::Instant;


#[derive(Clone)]
struct ChunkQuad<T> {
    pub chunk: u32,
    pub quad: u32,
    pub mismatch_edges: SampleSet<(T, T)>,
}

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

    // params for normal prior
    pub μ_μ_a: f32,
    pub σ_μ_a: f32,

    // params for inverse-gamma prior
    pub α_σ_a: f32,
    pub β_σ_a: f32,

    pub α_θ: f32,
    pub β_θ: f32,

    // gamma rate prior
    pub e_r: f32,
    pub f_r: f32,
}


// Model global parameters.
pub struct ModelParams {
    pub cell_assignments: Vec<CellIndex>,
    cell_population: Vec<usize>,

    // per-cell areas
    cell_areas: Array1<f32>,

    // per-transcript areas
    transcript_areas: Array1<f32>,

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

    // thread-local space used for sampling z
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,

    π: Vec<f32>, // mixing proportions over components

    μ_a: Vec<f32>, // area dist mean param by component
    σ_a: Vec<f32>, // area dist std param by component

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
            cell_population: init_cell_population.clone(),
            cell_areas,
            transcript_areas: Array1::<f32>::from(transcript_areas.clone()),
            full_area,
            counts,
            foreground_counts: Array2::<u32>::from_elem((ncells, ngenes), 0),
            background_counts: Array1::<u32>::from_elem(ngenes, 0),
            total_gene_counts,
            logfactorial: LogFactorial::new(),
            loggammaplus: Vec::from_iter((0..ngenes).map(|_| LogGammaPlus::default())),
            z,
            component_counts: Array2::<u32>::from_elem((ngenes, ncomponents), 0),
            z_probs: ThreadLocal::new(),
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_a: vec![priors.μ_μ_a; ncomponents],
            σ_a: vec![priors.σ_μ_a; ncomponents],
            r,
            lgamma_r,
            θ: Array2::<f32>::from_elem((ncomponents, ngenes), 0.1),
            λ: Array2::<f32>::from_elem((ngenes, ncells), 0.1),
            γ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
            γ_fg: Array1::<f32>::from_elem(ngenes, 0.0),
            λ_bg: Array1::<f32>::from_elem(ngenes, 0.0),
        };
    }

    fn ncomponents(&self) -> usize {
        return self.π.len();
    }

    fn recompute_counts(&mut self, ncells: usize, transcripts: &Vec<Transcript>) {
        self.counts.fill(0);
        for (i, &j) in self.cell_assignments.iter().enumerate() {
            let gene = transcripts[i].gene as usize;
            if j != BACKGROUND_CELL {
                self.counts[[gene, j as usize]] += 1;
            }
        }
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

    pub fn log_likelihood(&self) -> f32 {
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

    pub fn write_cell_hulls(&self, transcripts: &Vec<Transcript>, filename: &str) {
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

            writeln!(
                encoder,
                concat!(
                    "    {{\n",
                    "      \"type\": \"Feature\",\n",
                    "      \"properties\": {{\n",
                    "        \"cell\": {},\n",
                    "        \"area\": {}\n",
                    "      }},\n",
                    "      \"geometry\": {{\n",
                    "        \"type\": \"Polygon\",\n",
                    "        \"coordinates\": [",
                    "          ["
                ),
                i, area
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



trait Proposal {
    fn accept(&mut self);
    fn reject(&mut self);

    fn ignored(&self) -> bool;
    fn accepted(&self) -> bool;

    // Return updated cell size minus current cell size `old_cell`
    fn old_cell_area_delta(&self) -> f32;

    // Return updated cell size minus current cell size `new_cell`
    fn new_cell_area_delta(&self) -> f32;

    fn old_cell(&self) -> u32;
    fn new_cell(&self) -> u32;

    // fn set_old_cell_area(&mut self, area: f32);
    // fn set_new_cell_area(&mut self, area: f32);

    // fn old_cell_area(&self) -> f32;
    // fn new_cell_area(&self) -> f32;

    fn log_weight(&self) -> f32;

    fn transcripts<'b, 'c>(&'b self) -> &'c[usize] where 'b: 'c;

    // Iterator over number of transcripts in the proposal of each gene
    fn gene_count<'b, 'c>(&'b self) -> &'c[u32] where 'b: 'c;

    fn evaluate(&mut self, priors: &ModelPriors, params: &ModelParams) {
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

        // TODO: Do all these terms change even for genes that aren't involved?

        if from_background {
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ -= count as f32 * params.λ_bg[i].ln()
            }
        } else {
            let area_diff = self.old_cell_area_delta();

            // normalization term difference
            δ += Zip::from(params.λ.column(old_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * area_diff);

            // subtract out old cell likelihood terms
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ -= count as f32 * (params.λ_bg[i as usize] + params.λ[[i, old_cell as usize]]).ln();
            }
        }

        if to_background {
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ += count as f32 * params.λ_bg[i].ln();
            }
        } else {
            let area_diff = self.new_cell_area_delta();

            // TODO: Ok, the totality of the hugely negative score comes from this bit here.
            // It seems we end up rejecting any more to enlarge cells because this
            // is always negative a million.

            // normalization term difference
            δ += Zip::from(params.λ.column(new_cell as usize))
                .fold(0.0, |acc, &λ| acc - λ * area_diff);


            // add in new cell likelihood terms
            for (i, &count) in self.gene_count().iter().enumerate() {
                δ += count as f32 * (params.λ_bg[i] + params.λ[[i, new_cell as usize]]).ln();
            }
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
        stats: &mut ProposalStats, transcripts: &Vec<Transcript>)
    {
        self.repopulate_proposals(params);
        self.proposals_mut()
            .par_iter_mut()
            .for_each(|p| p.evaluate(priors, params));

        self.apply_accepted_proposals(stats, transcripts, priors, params);
    }

    fn apply_accepted_proposals(
        &mut self,
        stats: &mut ProposalStats,
        transcripts: &Vec<Transcript>,
        priors: &ModelPriors,
        params: &mut ModelParams)
    {
        // Update cell assignments
        for proposal in self.proposals().iter().filter(|p| p.accepted() && !p.ignored()) {
            let old_cell = proposal.old_cell();
            let new_cell = proposal.new_cell();

            let mut count = 0;
            for &i in proposal.transcripts() {
                params.cell_assignments[i] = new_cell;
                count += 1;
            }

            // Update count matrix and areas
            if old_cell != BACKGROUND_CELL {
                params.cell_population[old_cell as usize] -= count;

                let mut cell_area = params.cell_areas[old_cell as usize];
                cell_area += proposal.old_cell_area_delta();
                cell_area = cell_area.max(priors.min_cell_area);
                params.cell_areas[old_cell as usize] = cell_area;

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

        // sample μ_a
        let mut μ_μ_a =
            vec![priors.μ_μ_a / (priors.σ_μ_a.powi(2)); params.ncomponents()];
        // let mut component_population = vec![0; self.params.ncomponents()];
        let component_population = Array1::<u32>::from_elem(params.ncomponents(), 0_u32);

        μ_μ_a.iter_mut().enumerate().for_each(|(i, μ)| {
            *μ /= (1_f32 / priors.σ_μ_a.powi(2))
                + (component_population[i] as f32 / params.σ_a[i].powi(2));
        });

        let σ2_μ_a: Vec<f32> = izip!(&params.σ_a, &component_population)
            .map(|(σ, n)| (1_f32 / priors.σ_μ_a.powi(2) + (*n as f32) / σ.powi(2)).recip())
            .collect();

        params.μ_a.clear();
        params
            .μ_a
            .extend(izip!(&μ_μ_a, &σ2_μ_a).map(|(μ, σ2)| {
                Normal::new(*μ, σ2.sqrt())
                    .unwrap()
                    .sample(&mut rng)
            }));


        // sample σ_a
        let mut δ2s = vec![0_f32; params.ncomponents()];
        for (z_i, cell_area) in params.z.iter().zip(&params.cell_areas) {
            δ2s[*z_i as usize] += (cell_area.ln() - μ_μ_a[*z_i as usize]).powi(2);
        }

        // sample σ_a
        params.σ_a.clear();
        params
            .σ_a
            .extend(izip!(&component_population, &δ2s).map(|(n, δ2)| {
                Gamma::new(
                    priors.α_σ_a + 0.5 * *n as f32,
                    (priors.β_σ_a + 0.5 * *δ2).recip(),
                ).unwrap().sample(&mut rng).recip().sqrt()
            }));

        // Sample background/foreground counts
        // (This seems a little pointless. As devised now, if a transcript is under a cell it has a
        // very high probability of coming from that cell)

        let t0 = Instant::now();
        params.background_counts.assign(&params.total_gene_counts);
        Zip::from(params.counts.rows())
            .and(params.foreground_counts.columns_mut())
            .and(&mut params.background_counts)
            .and(params.λ.rows())
            .and(&params.λ_bg)
            .par_for_each(|cs, fcs, bc, λs, λ_bg| {
                let mut rng = thread_rng();

                let p_bg = λ_bg * (-λ_bg * params.full_area).exp();

                for (c, fc, λ, cell_area) in izip!(cs, fcs, λs, &params.cell_areas) {

                    // let p_fg = λ * (-λ * cell_area).exp();
                    // let p = p_fg / (p_fg + p_bg);
                    //
                    let p = λ / (λ + λ_bg);

                    // if p == 0.0 {
                    //     *fc = 0;
                    // } else if p == 1.0 {
                    //     *fc = *c;
                    // } else {
                    //     *fc = Binomial::new(*c as u64, p as f64).unwrap().sample(&mut rng) as u32;
                    // }

                    // TODO: Maybe add a special fast case for when `p` is exceptionally low?

                    *fc = Binomial::new(*c as u64, p as f64).unwrap().sample(&mut rng) as u32;

                    // TODO: This seems fucked overall. I get
                    // *fc = *c;

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
        let t0 = Instant::now();
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

                    // if *w > 1000.0 {
                    //     dbg!(w, c, a, r);
                    //     panic!("w is too large");
                    // }
                }
            });
        // println!("  Sample θ: {:?}", t0.elapsed());

        // Sample r
        let t0 = Instant::now();
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
        let t0 = Instant::now();
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
        let t0 = Instant::now();
        Zip::from(
            params.foreground_counts
                .rows(),
        )
        .and(&mut params.z)
        .and(&params.cell_areas)
        .par_for_each(|cs, z_i, cell_area| {
            let mut z_probs = params
                .z_probs
                .get_or(|| RefCell::new(vec![0_f64; ncomponents]))
                .borrow_mut();

            // loop over components
            for (zp, π, θs) in izip!(z_probs.iter_mut(), &params.π, params.θ.rows())
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
}


// Pre-allocated thread-local storage used when computing cell areas.
struct AreaCalcStorage {
    vertices: Vec<(f32, f32)>,
    hull: Vec<(f32, f32)>,
}

impl AreaCalcStorage {
    fn new() -> Self {
        return AreaCalcStorage {
            vertices: Vec::new(),
            hull: Vec::new(),
        };
    }
}