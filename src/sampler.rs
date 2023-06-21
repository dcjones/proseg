pub mod transcripts;
mod hull;
mod distributions;

use kiddo::float::distance::squared_euclidean;
use kiddo::float::kdtree::KdTree;
use petgraph::visit::IntoNeighbors;
use rand::{seq::SliceRandom, Rng, thread_rng};
use rand::distributions::Distribution;
use rayon::prelude::*;
use std::collections::HashSet;
use transcripts::{coordinate_span, NeighborhoodGraph, NucleiCentroid, Transcript};
use hull::convex_hull_area;
use thread_local::ThreadLocal;
use std::cell::{RefCell, RefMut};
use distributions::lognormal_logpdf;
use statrs::distribution::{Dirichlet, InverseGamma, Normal};
use itertools::izip;

#[derive(Clone, Copy)]
pub struct ModelPriors {
    pub min_cell_size: f32,
    pub background_logprob: f32,
    pub foreground_logprob: f32,

    // params for normal prior
    pub μ_μ_a: f32,
    pub σ_μ_a: f32,

    // params for inverse-gamma prior
    pub α_σ_a: f32,
    pub β_σ_a: f32,
}


pub struct ModelParams {
    π: Vec<f32>, // mixing proportions over components

    μ_a: Vec<f32>, // area dist mean param by component
    σ_a: Vec<f32>, // area dist std param by component

    // TODO: feels like we need ndarrays to hold things like p and α
}


impl ModelParams {
    // initialize model parameters, with random cell assignments
    // and other parameterz unninitialized.
    fn new(priors: &ModelPriors, ncomponents: usize) -> Self {
        return ModelParams {
            π: vec![1_f32 / (ncomponents as f32); ncomponents],
            μ_a: vec![priors.μ_μ_a; ncomponents],
            σ_a: vec![priors.σ_μ_a; ncomponents],
        };
    }

    fn ncomponents(&self) -> usize {
        return self.π.len();
    }

    fn cell_logprob(&self, z: usize, cell_area: f32) -> f32 {
        return lognormal_logpdf(
            self.μ_a[z],
            self.σ_a[z],
            cell_area);
    }
}


struct ChunkedTranscript {
    transcript: Transcript,
    chunk: u32,
    quad: u32,
}

// Compute chunk and quadrant for a single a single (x,y) point.
fn chunkquad(x: f32, y: f32, xmin: f32, ymin: f32, chunk_size: f32, nxchunks: usize) -> (u32, u32) {
    let xchunkquad = ((x - xmin) / (chunk_size / 2.0)).floor() as u32;
    let ychunkquad = ((y - ymin) / (chunk_size / 2.0)).floor() as u32;

    let chunk = (xchunkquad / 4) + (ychunkquad / 4) * (nxchunks as u32);
    let quad = (xchunkquad % 2) + (ychunkquad % 2) * 2;

    return (chunk, quad);
}

// Figure out every transcript's chunk and quadrant.
fn chunk_transcripts(
    transcripts: &Vec<Transcript>,
    xmin: f32,
    ymin: f32,
    chunk_size: f32,
    nxchunks: usize,
) -> Vec<ChunkedTranscript> {
    return transcripts
        .iter()
        .map(|transcript| {
            let (chunk, quad) =
                chunkquad(transcript.x, transcript.y, xmin, ymin, chunk_size, nxchunks);

            ChunkedTranscript {
                transcript: transcript.clone(),
                chunk: chunk,
                quad: quad,
            }
        })
        .collect();
}

// Holds the segmentation state.
// This is kept outside of Sample mainly so we can borrow this in multiple threads
// while modifying the sampler when generating proposals.
pub struct Segmentation<'a> {
    transcripts: &'a Vec<Transcript>,
    nuclei_centroids: &'a Vec<NucleiCentroid>,
    adjacency: &'a NeighborhoodGraph,
    cell_assignments: Vec<u32>,
    cell_population: Vec<usize>,
    cell_logprobs: Vec<f32>,
}

impl<'a> Segmentation<'a> {
    pub fn new(
        transcripts: &'a Vec<Transcript>,
        nuclei_centroids: &'a Vec<NucleiCentroid>,
        adjacency: &'a NeighborhoodGraph,
    ) -> Segmentation<'a> {
        let (cell_assignments, cell_population) =
            init_cell_assignments(transcripts, nuclei_centroids, 15);

        let cell_logprobs = vec![0.0; nuclei_centroids.len()];

        return Segmentation {
            transcripts,
            nuclei_centroids,
            adjacency,
            cell_assignments,
            cell_population,
            cell_logprobs,
        };
    }

    fn ncells(&self) -> usize {
        return self.nuclei_centroids.len();
    }

    pub fn apply_local_updates(&mut self, sampler: &mut Sampler) {
        let accept_count = sampler.proposals.iter().filter(|p| p.accept).count();
        let unignored_count = sampler.proposals.iter().filter(|p| !p.ignore).count();
        println!("Applying {} of {} proposals", accept_count, unignored_count);

        // TODO: check if we are doing multiple updates on the same cell and warn
        // about it.

        // Update cell assignments
        for proposal in sampler.proposals.iter().filter(|p| p.accept) {
            let prev_state = self.cell_assignments[proposal.i];

            self.cell_population[prev_state as usize] -= 1;
            self.cell_population[proposal.state as usize] += 1;
            self.cell_assignments[proposal.i] = proposal.state;

            if prev_state as usize != self.ncells() {
                sampler.cell_areas[prev_state as usize] = proposal.from_cell_area;
                self.cell_logprobs[prev_state as usize] = proposal.from_cell_logprob;
            }

            if proposal.state as usize != self.ncells() {
                sampler.cell_areas[proposal.state as usize] = proposal.to_cell_area;
                self.cell_logprobs[proposal.state as usize] = proposal.to_cell_logprob;
            }
        }

        // Update mismatch edges
        for quad in 0..4 {
            sampler.chunkquads[quad]
                .par_iter_mut()
                .for_each(|chunkquad| {
                    for proposal in sampler.proposals.iter().filter(|p| p.accept) {
                        let i = proposal.i;
                        for j in self.adjacency.neighbors(i) {
                            if self.cell_assignments[i] != self.cell_assignments[j] {
                                if chunkquad.contains(&sampler.transcripts[j]) {
                                    chunkquad.mismatch_edges.insert((j, i));
                                }
                                if chunkquad.contains(&sampler.transcripts[i]) {
                                    chunkquad.mismatch_edges.insert((i, j));
                                }
                            } else {
                                if chunkquad.contains(&sampler.transcripts[j]) {
                                    chunkquad.mismatch_edges.remove(&(j, i));
                                }
                                if chunkquad.contains(&sampler.transcripts[i]) {
                                    chunkquad.mismatch_edges.remove(&(i, j));
                                }
                            }
                        }
                    }
                });
        }
    }
}

#[derive(Clone)]
struct ChunkQuad {
    chunk: u32,
    quad: u32,
    mismatch_edges: HashSet<(usize, usize)>,
    shuffled_mismatch_edges: Vec<(usize, usize)>,
}

impl ChunkQuad {
    fn contains(&self, transcript: &ChunkedTranscript) -> bool {
        return self.chunk == transcript.chunk && self.quad == transcript.quad;
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


pub struct Sampler {
    priors: ModelPriors,
    chunkquads: Vec<Vec<ChunkQuad>>,
    transcripts: Vec<ChunkedTranscript>,
    params: ModelParams,
    proposals: Vec<Proposal>,
    cell_transcripts: Vec<HashSet<usize>>,
    cell_areas: Vec<f32>,
    cell_area_calc_storage: ThreadLocal<RefCell<AreaCalcStorage>>,
    z: Vec<u32>, // assignment of cells to components
    z_probs: ThreadLocal<RefCell<Vec<f64>>>,
    quad: usize,
    sample_num: usize,
}

impl Sampler {
    pub fn new(priors: ModelPriors, seg: &mut Segmentation, ncomponents: usize, chunk_size: f32) -> Sampler {
        let (xmin, xmax, ymin, ymax) = coordinate_span(seg.transcripts, seg.nuclei_centroids);

        let nxchunks = ((xmax - xmin) / chunk_size).ceil() as usize;
        let nychunks = ((ymax - ymin) / chunk_size).ceil() as usize;
        let nchunks = nxchunks * nychunks;
        let chunked_transcripts =
            chunk_transcripts(seg.transcripts, xmin, ymin, chunk_size, nxchunks);

        let ncells = seg.nuclei_centroids.len();

        let mut chunkquads = Vec::with_capacity(4);
        for quad in 0..4 {
            let mut chunks = Vec::with_capacity(nchunks);
            for chunk in 0..nchunks {
                chunks.push(ChunkQuad {
                    chunk: chunk as u32,
                    quad: quad as u32,
                    mismatch_edges: HashSet::new(),
                    shuffled_mismatch_edges: Vec::new(),
                });
            }
            chunkquads.push(chunks);
        }

        // need to be able to look up a quad chunk given its indexes
        let mut nmismatchedges = 0;
        for i in 0..seg.adjacency.node_count() {
            for j in seg.adjacency.neighbors(i) {
                if seg.cell_assignments[i] != seg.cell_assignments[j] {
                    let ti = &chunked_transcripts[i];
                    chunkquads[ti.quad as usize][ti.chunk as usize]
                        .mismatch_edges
                        .insert((i, j));
                    nmismatchedges += 1;
                }
            }
        }
        println!(
            "Made initial cell assignments with {} mismatch edges",
            nmismatchedges
        );

        let params = ModelParams::new(&priors, ncomponents);

        let mut cell_transcripts = vec![HashSet::new(); ncells];
        for (i, cell) in seg.cell_assignments.iter().enumerate() {
            if (*cell as usize) < ncells {
                cell_transcripts[*cell as usize].insert(i);
            }
        }

        let cell_areas = vec![0.0; ncells];
        let proposals = vec![Proposal::new(); nchunks];

        let mut rng = rand::thread_rng();
        let z = (0..ncells).map(|_| rng.gen_range(0..ncomponents) as u32).collect();

        let mut sampler = Sampler {
            priors,
            chunkquads,
            transcripts: chunked_transcripts,
            params: params,
            cell_transcripts: cell_transcripts,
            cell_areas: cell_areas,
            cell_area_calc_storage: ThreadLocal::new(),
            z_probs: ThreadLocal::new(),
            z: z,
            proposals: proposals,
            quad: 0,
            sample_num: 0,
        };

        sampler.compute_cell_areas();
        sampler.sample_global_params();
        sampler.compute_cell_logprobs(seg);

        return sampler;
    }

    pub fn sample_local_updates(&mut self, seg: &Segmentation) {
        self.repoulate_proposals(seg);
        // dbg!(&self.proposals);

        self.proposals.par_iter_mut().for_each(|proposal| {
            let areacalc = self.cell_area_calc_storage.get_or(
                || RefCell::new(AreaCalcStorage::new()) ).borrow_mut();

            proposal.evaluate(seg, &self.priors, &self.params, &self.z, &self.cell_transcripts, areacalc);
        });

        self.sample_num += 1;
    }

    fn repoulate_proposals(&mut self, seg: &Segmentation) {
        self.proposals
            .par_iter_mut()
            .zip(&mut self.chunkquads[self.quad])
            .for_each(|(proposal, chunkquad)| {
                let mut rng = rand::thread_rng();

                // So we have a lack of mismatch_edges in a lot of places...?
                if chunkquad.mismatch_edges.is_empty() {
                    proposal.ignore = true;
                    // println!("A");
                    return;
                }

                // TODO: If (i, j) are both cells, with some low probability we
                // should propose making i background.

                chunkquad.shuffled_mismatch_edges.clear();
                chunkquad
                    .shuffled_mismatch_edges
                    .extend(chunkquad.mismatch_edges.iter().cloned());
                let (i, j) = chunkquad.shuffled_mismatch_edges.choose(&mut rng).unwrap();

                let k = *i; // target cell
                let c = seg.cell_assignments[*j]; // proposal state

                // Don't propose removing the last transcript from a cell. (This is
                // breaks the markov chain balance, since there's no path back to the previous state.)
                if seg.cell_population[seg.cell_assignments[k] as usize] == 1 {
                    proposal.ignore = true;
                    // println!("B");
                    return;
                }

                // compute the probability of selecting the proposal (k, c)
                let num_mismatching_edges = chunkquad.mismatch_edges.len();

                let num_new_state_neighbors = seg
                    .adjacency
                    .neighbors(k)
                    .filter(|j| seg.cell_assignments[*j] == c)
                    .count();

                let num_prev_state_neighbors = seg
                    .adjacency
                    .neighbors(k)
                    .filter(|j| seg.cell_assignments[*j] == seg.cell_assignments[*i])
                    .count();

                let proposal_prob = num_new_state_neighbors as f64 / num_mismatching_edges as f64;

                let new_num_mismatching_edges = num_mismatching_edges
                    + 2*num_prev_state_neighbors // edges that are newly mismatching
                    - 2*num_new_state_neighbors; // edges that are newly matching

                let reverse_proposal_prob = num_prev_state_neighbors as f64 / new_num_mismatching_edges as f64;

                proposal.i = *i;
                proposal.state = seg.cell_assignments[*j];
                proposal.log_weight = (reverse_proposal_prob.ln() - proposal_prob.ln()) as f32;
                proposal.accept = false;
                proposal.ignore = false;
            });

        self.quad = (self.quad + 1) % 4;
    }

    pub fn sample_global_params(&mut self) {
        let mut rng = thread_rng();
        let ncomponents = self.params.ncomponents();

        // TODO: move z out of Params?

        self.z.par_iter_mut().enumerate().for_each(|(i, z_i)| {
            let mut z_probs = self.z_probs.get_or(|| RefCell::new(vec![0_f64; ncomponents])).borrow_mut();
            let cell_area = self.cell_areas[i];
            z_probs.iter_mut().enumerate().for_each(|(j, zp)| {
                *zp = (self.params.π[j] as f64) *
                    (self.params.cell_logprob(j as usize, cell_area) as f64).exp();
            });
            let z_prob_sum = z_probs.iter().sum::<f64>();

            // cumsum in-place
            z_probs.iter_mut().fold(0.0, |mut acc, x| {
                acc += *x / z_prob_sum;
                *x = acc;
                acc
            });

            let rng = &mut thread_rng();
            let u = rng.gen::<f64>();
            *z_i = z_probs.partition_point(|x| *x < u) as u32;
        });

        // sample z
        // let mut z_probs = vec![0_f64; ncomponents];
        // for (i, cell_area) in self.cell_areas.iter().enumerate() {
        //     for z in 0..ncomponents {
        //         z_probs[z] = (self.params.π[z] as f64) * (self.params.cell_logprob(i, *cell_area) as f64).exp();
        //     }
        //     let probsum = z_probs.iter().sum::<f64>();
        //     z_probs.iter_mut().for_each(|p| *p /= probsum);
        //     self.params.z[i] = Categorical::new(&z_probs).unwrap().sample(&mut rng) as u32;
        // }

        // sample π
        let mut α = vec![1_f64; self.params.ncomponents()];
        for z_i in self.z.iter() {
            α[*z_i as usize] += 1.0;
        }
        self.params.π.clear();
        self.params.π.extend(Dirichlet::new(α).unwrap().sample(&mut rng).iter().map(|x| *x as f32));

        // sample μ_a
        let mut μ_μ_a = vec![self.priors.μ_μ_a / (self.priors.σ_μ_a.powi(2)) ; self.params.ncomponents()];
        let mut component_population = vec![0; self.params.ncomponents()];

        for (z_i, cell_area) in self.z.iter().zip(&self.cell_areas) {
            μ_μ_a[*z_i as usize] += cell_area.ln() / self.params.σ_a[*z_i as usize].powi(2);
            component_population[*z_i as usize] += 1;
        }
        μ_μ_a.iter_mut().enumerate().for_each(|(i, μ)| {
            *μ /= (1_f32/self.priors.σ_μ_a.powi(2)) +
                (component_population[i] as f32 / self.params.σ_a[i].powi(2));
        });

        let σ2_μ_a: Vec<f32> = izip!(&self.params.σ_a, &component_population).map(|(σ, n)| {
            (1_f32 / self.priors.σ_μ_a.powi(2) + (*n as f32) / σ.powi(2)).recip()
        }).collect();

        self.params.μ_a.clear();
        self.params.μ_a.extend(
            izip!(&μ_μ_a, &σ2_μ_a).map(|(μ, σ2)| Normal::new(*μ as f64, σ2.sqrt() as f64).unwrap().sample(&mut rng) as f32)
        );

        // sample σ_a
        let mut δ2s = vec![0_f32; self.params.ncomponents()];
        for (z_i, cell_area) in self.z.iter().zip(&self.cell_areas) {
            δ2s[*z_i as usize] += (cell_area.ln() - μ_μ_a[*z_i as usize]).powi(2);
        }

        // //sample σ_a
        self.params.σ_a.clear();
        self.params.σ_a.extend(izip!(&component_population, &δ2s).map(|(n, δ2)| {
            InverseGamma::new(
                self.priors.α_σ_a as f64 + 0.5 * *n as f64,
                self.priors.β_σ_a as f64 + 0.5 * *δ2 as f64
            ).unwrap().sample(&mut rng).sqrt() as f32
        }));
    }

    fn compute_cell_areas(&mut self) {
        self.cell_areas
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, area)| {
                let mut areacalc = self.cell_area_calc_storage.get_or(
                    || RefCell::new(AreaCalcStorage::new()) ).borrow_mut();
                areacalc.vertices.clear();

                for j in self.cell_transcripts[i].iter() {
                    let transcript = self.transcripts[*j].transcript;
                    areacalc.vertices.push((transcript.x, transcript.y));
                }

                let (mut vertices, mut hull) =
                    RefMut::map_split(areacalc, |areacalc| (&mut areacalc.vertices, &mut areacalc.hull));
                *area = convex_hull_area(&mut vertices, &mut hull).max(self.priors.min_cell_size);
            });

        let mut minarea: f32 = 1e12;
        let mut maxarea: f32 = 0.0;
        for area in &self.cell_areas {
            if *area < minarea {
                minarea = *area;
            }
            if *area > maxarea {
                maxarea = *area;
            }
        }
        println!("Area span: {}, {}", minarea, maxarea);
    }

    fn compute_cell_logprobs(&self, seg: &mut Segmentation) {
        // Assume at this point that cell areas and transcript counts are up to date

        seg.cell_logprobs.par_iter_mut().enumerate().for_each(|(i, logprob)| {
            let cell_area = self.cell_areas[i];
            *logprob = self.params.cell_logprob(self.z[i] as usize, cell_area);
        });
    }


}

#[derive(Clone, Debug)]
struct Proposal {
    i: usize,
    state: u32,

    // metroplis-hastings proposal weight weight
    log_weight: f32,

    ignore: bool,
    accept: bool,

    // updated cell areas and logprobs if the proposal is accepted
    to_cell_area: f32,
    to_cell_logprob: f32,

    from_cell_area: f32,
    from_cell_logprob: f32,
}

impl Proposal {
    fn new() -> Self {
        Proposal {
            i: 0,
            state: 0,
            log_weight: 0.0,
            ignore: true,
            accept: false,
            to_cell_area: 0.0,
            to_cell_logprob: 0.0,
            from_cell_area: 0.0,
            from_cell_logprob: 0.0,
        }
    }

    fn evaluate(
            &mut self,
            seg: &Segmentation,
            priors: &ModelPriors,
            params: &ModelParams,
            z: &Vec<u32>,
            cell_transcripts: &Vec<HashSet<usize>>,
            areacalc: RefMut<AreaCalcStorage>) {

        if self.ignore || seg.cell_assignments[self.i] == self.state {
            self.accept = false;
            return;
        }

        let (mut areacalc_vertices, mut areacalc_hull) =
            RefMut::map_split(areacalc, |areacalc| (&mut areacalc.vertices, &mut areacalc.hull));

        let prev_state = seg.cell_assignments[self.i];
        let from_background = prev_state == seg.ncells() as u32;
        let to_background = self.state == seg.ncells() as u32;
        let this_transcript = &seg.transcripts[self.i];

        // Current total log prob
        let mut current_logprob = 0.0_f32;
        if from_background {
            current_logprob += priors.background_logprob;
        } else {
            current_logprob += priors.foreground_logprob;
            current_logprob += seg.cell_logprobs[prev_state as usize];
        }

        if !to_background {
            current_logprob += seg.cell_logprobs[self.state as usize];
        }

        // Proposal total log prob
        let mut proposal_logprob = 0.0;
        if to_background {
            proposal_logprob += priors.background_logprob;
        } else {
            proposal_logprob += priors.foreground_logprob;

            // recompute area for `self.state` after reassigning this transcript
            areacalc_vertices.clear();

            for j in cell_transcripts[self.state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                areacalc_vertices.push((transcript.x, transcript.y));
            }
            areacalc_vertices.push((this_transcript.x, this_transcript.y));

            self.to_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull).max(priors.min_cell_size);
            self.to_cell_logprob = params.cell_logprob(
                z[self.state as usize] as usize, self.to_cell_area);

            proposal_logprob += self.to_cell_logprob;
        }

        if !from_background {
            // recompute area for `self.state` after reassigning this transcript
            areacalc_vertices.clear();
            for j in cell_transcripts[prev_state as usize].iter() {
                let transcript = &seg.transcripts[*j];
                if transcript != this_transcript {
                    areacalc_vertices.push((transcript.x, transcript.y));
                }
            }

            self.from_cell_area = convex_hull_area(&mut areacalc_vertices, &mut areacalc_hull).max(priors.min_cell_size);
            self.from_cell_logprob = params.cell_logprob(
                z[prev_state as usize] as usize, self.from_cell_area);

            proposal_logprob += self.from_cell_logprob;
        }

        let mut rng = thread_rng();
        let logu = rng.gen::<f32>().ln();

        // println!("Eval: {} {} {} {} {}", prev_state, self.state, proposal_logprob, current_logprob, self.log_weight);

        self.accept = logu <= (proposal_logprob - current_logprob) + self.log_weight;

    }
}

fn init_cell_assignments(
    transcripts: &Vec<Transcript>,
    nuclei_centroids: &Vec<NucleiCentroid>,
    k: usize,
) -> (Vec<u32>, Vec<usize>) {
    let mut kdtree: KdTree<f32, usize, 2, 32, u32> = KdTree::with_capacity(transcripts.len());

    for (i, transcript) in transcripts.iter().enumerate() {
        kdtree.add(&[transcript.x, transcript.y], i);
    }

    let ncells = nuclei_centroids.len();
    let ntranscripts = transcripts.len();
    let mut cell_assignments = vec![ncells as u32; ntranscripts];
    let mut cell_population = vec![0; ncells + 1];
    cell_population[ncells] = ntranscripts;

    for (i, centroid) in nuclei_centroids.iter().enumerate() {
        for neighbor in kdtree.nearest_n(&[centroid.x, centroid.y], k, &squared_euclidean) {
            cell_assignments[neighbor.item] = i as u32;
            cell_population[ncells] -= 1;
            cell_population[i] += 1;
        }
    }

    return (cell_assignments, cell_population);
}



// TODO:
// Ok, what am I doing here? When do I have to recompute cell area?
// 

// fn compute_cell_area(transcripts: &Vec<Transcript>, cell_assignments: &Vec<u32>, ncells: usize) -> Vec<f32> {
//     let mut cell_area = vec![0.0; ncells];

//     return cell_area;
// }